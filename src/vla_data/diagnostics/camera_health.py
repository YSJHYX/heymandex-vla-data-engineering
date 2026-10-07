"""Deterministic, read-only camera signal health diagnostics.

These metrics explain RAW signal behavior.  They are not D2 rejection reasons
and must never be used as training eligibility authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

HEALTHY = "HEALTHY"
DEGRADED = "DEGRADED"
FROZEN = "FROZEN"
STALE = "STALE"
MISSING = "MISSING"
CLOCK_MISMATCH = "CLOCK_MISMATCH"
UNKNOWN = "UNKNOWN"

_FROZEN_MIN_MASTER_SPAN_NS = 1_000_000_000
_FRESHNESS_WARNING_NS = 1_000_000_000
_CLOCK_MISMATCH_NS = 60_000_000_000


@dataclass(frozen=True)
class CameraHealth:
    """Serializable health summary for one RAW camera signal."""

    name: str
    total_rows: int
    valid_count: int
    valid_ratio: float
    timestamp_positive_count: int
    timestamp_positive_ratio: float
    timestamp_unique_count: int
    timestamp_unique_ratio: float
    frame_index_valid_count: int
    frame_index_valid_ratio: float
    timestamp_min: int | None
    timestamp_max: int | None
    master_minus_camera_ns: dict[str, int | float | None]
    max_consecutive_identical_timestamp: int
    first_stale_row: int | None
    last_timestamp_change_row: int | None
    estimated_frozen_duration_ns: int | None
    status: str
    findings: tuple[str, ...]
    evidence: tuple[str, ...]
    suggested_checks: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def camera_health(
    name: str,
    master_timestamp_ns: np.ndarray,
    camera_timestamp_ns: np.ndarray,
    camera_valid: np.ndarray,
    frame_index: np.ndarray,
) -> CameraHealth:
    """Classify one camera conservatively from RAW scalar signal arrays."""

    master = np.asarray(master_timestamp_ns, dtype=np.int64)
    timestamps = np.asarray(camera_timestamp_ns, dtype=np.int64)
    valid = np.asarray(camera_valid, dtype=bool)
    frames = np.asarray(frame_index, dtype=np.int64)
    _validate_shapes(master, timestamps, valid, frames)

    total = int(master.size)
    positive = timestamps > 0
    positive_count = int(np.count_nonzero(positive))
    valid_count = int(np.count_nonzero(valid))
    frame_count = int(np.count_nonzero(frames >= 0))
    positive_values = timestamps[positive]
    unique_count = int(np.unique(positive_values).size) if positive_count else 0

    delta_mask = positive & (master > 0)
    deltas = master[delta_mask] - timestamps[delta_mask]
    delta_summary = _delta_summary(deltas)
    repeat_count, repeat_start, repeat_end = _longest_positive_repeat(timestamps)
    last_change = _last_change_row(timestamps)
    estimated_frozen_duration = _estimated_frozen_duration(
        master,
        timestamps,
        repeat_count=repeat_count,
        repeat_start=repeat_start,
        repeat_end=repeat_end,
    )

    positive_ratio = _ratio(positive_count, total)
    valid_ratio = _ratio(valid_count, total)
    frame_ratio = _ratio(frame_count, total)
    unique_ratio = _ratio(unique_count, positive_count)
    master_span = _positive_span(master)

    status = _classify(
        total=total,
        positive_count=positive_count,
        positive_ratio=positive_ratio,
        valid_ratio=valid_ratio,
        frame_count=frame_count,
        frame_ratio=frame_ratio,
        unique_count=unique_count,
        unique_ratio=unique_ratio,
        repeat_count=repeat_count,
        master_span=master_span,
        deltas=deltas,
        delta_summary=delta_summary,
        estimated_frozen_duration=estimated_frozen_duration,
    )
    findings, evidence, checks = _explanation(
        name=name,
        status=status,
        total=total,
        valid_count=valid_count,
        positive_count=positive_count,
        unique_count=unique_count,
        frame_count=frame_count,
        repeat_count=repeat_count,
        master_span=master_span,
        delta_summary=delta_summary,
        estimated_frozen_duration=estimated_frozen_duration,
    )

    return CameraHealth(
        name=name,
        total_rows=total,
        valid_count=valid_count,
        valid_ratio=valid_ratio,
        timestamp_positive_count=positive_count,
        timestamp_positive_ratio=positive_ratio,
        timestamp_unique_count=unique_count,
        timestamp_unique_ratio=unique_ratio,
        frame_index_valid_count=frame_count,
        frame_index_valid_ratio=frame_ratio,
        timestamp_min=int(positive_values.min()) if positive_count else None,
        timestamp_max=int(positive_values.max()) if positive_count else None,
        master_minus_camera_ns=delta_summary,
        max_consecutive_identical_timestamp=repeat_count,
        first_stale_row=repeat_start if repeat_count >= 2 else None,
        last_timestamp_change_row=last_change,
        estimated_frozen_duration_ns=estimated_frozen_duration,
        status=status,
        findings=tuple(findings),
        evidence=tuple(evidence),
        suggested_checks=tuple(checks),
    )


def _validate_shapes(*arrays: np.ndarray) -> None:
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("camera diagnostic arrays must all be one-dimensional")
    sizes = {int(array.size) for array in arrays}
    if len(sizes) != 1:
        raise ValueError("camera diagnostic arrays must have identical row counts")


def _ratio(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _positive_span(values: np.ndarray) -> int:
    positive = values[values > 0]
    return int(positive[-1] - positive[0]) if positive.size >= 2 else 0


def _delta_summary(values: np.ndarray) -> dict[str, int | float | None]:
    if not values.size:
        return {"min": None, "median": None, "max": None}
    return {
        "min": int(np.min(values)),
        "median": float(np.median(values)),
        "max": int(np.max(values)),
    }


def _longest_positive_repeat(values: np.ndarray) -> tuple[int, int | None, int | None]:
    best_count = 0
    best_start = None
    best_end = None
    current_start = 0
    for index in range(int(values.size)):
        if index == 0 or values[index] != values[index - 1] or values[index] <= 0:
            current_start = index
        if values[index] <= 0:
            continue
        count = index - current_start + 1
        if count > best_count:
            best_count = count
            best_start = current_start
            best_end = index
    return best_count, best_start, best_end


def _last_change_row(values: np.ndarray) -> int | None:
    if not values.size:
        return None
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    return int(changes[-1]) if changes.size else 0


def _estimated_frozen_duration(
    master: np.ndarray,
    timestamps: np.ndarray,
    *,
    repeat_count: int,
    repeat_start: int | None,
    repeat_end: int | None,
) -> int | None:
    """Estimate frozen age; this value is derived and is not a RAW field."""

    if repeat_count < 2 or repeat_start is None or repeat_end is None:
        return None
    camera_value = int(timestamps[repeat_start])
    master_value = int(master[repeat_end])
    if camera_value <= 0 or master_value <= 0:
        return None
    return max(0, master_value - camera_value)


def _classify(
    *,
    total: int,
    positive_count: int,
    positive_ratio: float,
    valid_ratio: float,
    frame_count: int,
    frame_ratio: float,
    unique_count: int,
    unique_ratio: float,
    repeat_count: int,
    master_span: int,
    deltas: np.ndarray,
    delta_summary: dict[str, int | float | None],
    estimated_frozen_duration: int | None,
) -> str:
    if total == 0 or positive_count == 0 or frame_count == 0:
        return MISSING

    frozen_unique_limit = max(2, int(np.ceil(positive_count * 0.01)))
    if (
        positive_ratio >= 0.9
        and master_span >= _FROZEN_MIN_MASTER_SPAN_NS
        and unique_count <= frozen_unique_limit
        and repeat_count >= max(3, int(np.ceil(total * 0.8)))
        and (estimated_frozen_duration or 0) >= _FROZEN_MIN_MASTER_SPAN_NS
    ):
        return FROZEN

    median = delta_summary["median"]
    updating = positive_ratio >= 0.9 and unique_ratio >= 0.8
    if updating and median is not None and median <= -_FRESHNESS_WARNING_NS:
        return CLOCK_MISMATCH

    if updating and median is not None and median > _FRESHNESS_WARNING_NS:
        # A huge positive, nearly fixed offset is ambiguous: it could be a
        # clock-domain mismatch rather than genuine freshness lag.
        spread = int(np.max(deltas) - np.min(deltas)) if deltas.size else 0
        if median >= _CLOCK_MISMATCH_NS and spread < _FRESHNESS_WARNING_NS:
            return UNKNOWN
        return STALE

    if (
        positive_ratio >= 0.99
        and valid_ratio >= 0.95
        and frame_ratio >= 0.99
        and unique_ratio >= 0.9
        and median is not None
        and -100_000_000 <= median <= 250_000_000
    ):
        return HEALTHY

    if positive_ratio >= 0.5 and frame_ratio >= 0.5:
        return DEGRADED
    return UNKNOWN


def _explanation(
    *,
    name: str,
    status: str,
    total: int,
    valid_count: int,
    positive_count: int,
    unique_count: int,
    frame_count: int,
    repeat_count: int,
    master_span: int,
    delta_summary: dict[str, int | float | None],
    estimated_frozen_duration: int | None,
) -> tuple[list[str], list[str], list[str]]:
    label = name.upper()
    evidence = [
        f"{label} timestamp unique = {unique_count} / {positive_count}",
        f"{label} valid = {valid_count} / {total}",
        f"{label} frame index valid = {frame_count} / {total}",
        f"master timeline span = {master_span} ns",
    ]
    if delta_summary["median"] is not None:
        evidence.append(
            f"median master-camera delta = {delta_summary['median']:.0f} ns"
        )
    if estimated_frozen_duration is not None:
        evidence.append(
            f"estimated frozen duration (diagnostic) = {estimated_frozen_duration} ns"
        )

    if status == FROZEN:
        return (
            [
                f"{label} timestamp stopped advancing while the master timeline advanced.",
                f"Longest identical timestamp run covered {repeat_count} rows.",
            ],
            evidence,
            [
                "Check the camera capture thread alive state.",
                "Check the latest-frame update path.",
                "Check the camera pipeline and USB/device state.",
            ],
        )
    if status == MISSING:
        return (
            [f"{label} has no usable timestamp or frame-index evidence."],
            evidence,
            ["Check camera initialization, recorder wiring, and media references."],
        )
    if status == STALE:
        return (
            [f"{label} timestamps advance but observations remain persistently old."],
            evidence,
            ["Check producer latency, buffering, and recorder freshness state."],
        )
    if status == CLOCK_MISMATCH:
        return (
            [f"{label} timestamps advance in an incompatible or future clock domain."],
            evidence,
            ["Check timestamp source and host/device clock conversion."],
        )
    if status == HEALTHY:
        return (
            [f"{label} timestamp, validity, frame index, and freshness are healthy."],
            evidence,
            [],
        )
    if status == DEGRADED:
        return (
            [f"{label} has partial signal loss or reduced validity."],
            evidence,
            ["Inspect invalid rows and capture continuity."],
        )
    return (
        [f"{label} evidence is insufficient for a conservative classification."],
        evidence,
        ["Inspect RAW timestamps and producer logs before assigning a root cause."],
    )
