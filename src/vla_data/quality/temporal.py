"""Configurable D3 temporal quality evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from vla_data.quality.metrics import summary

TEMPORAL_METRIC_NAMES: tuple[str, ...] = (
    "state_component_skew_ns",
    "action_component_skew_ns",
    "post_state_component_skew_ns",
    "arm_state_age_at_action_ns",
    "hand_state_age_at_action_ns",
    "head_frame_age_at_action_ns",
    "wrist_frame_age_at_action_ns",
    "head_wrist_frame_skew_ns",
)


@dataclass(frozen=True)
class TemporalQualityConfig:
    """All ``None`` thresholds are measure/report-only."""

    max_state_skew_ns: int | None = None
    max_action_skew_ns: int | None = None
    max_post_state_skew_ns: int | None = None
    max_state_age_ns: int | None = None
    max_camera_age_ns: int | None = None
    max_camera_skew_ns: int | None = None
    exceedance_policy: Literal["warning", "exclude"] = "warning"

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if name.startswith("max_") and value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0 or None")
        if self.exceedance_policy not in {"warning", "exclude"}:
            raise ValueError("exceedance_policy must be 'warning' or 'exclude'")


@dataclass(frozen=True)
class TemporalEvaluation:
    metrics: dict[str, dict[str, int | float | None]]
    keep_mask: np.ndarray
    flag_counts: dict[str, int]
    warnings: tuple[str, ...]


def evaluate_temporal(
    transition_metrics: dict[str, object],
    transition_count: int,
    config: TemporalQualityConfig,
) -> TemporalEvaluation:
    """Measure temporal arrays and optionally warn or exclude by config."""

    values: dict[str, np.ndarray] = {}
    for name in TEMPORAL_METRIC_NAMES:
        if name not in transition_metrics:
            raise KeyError(f"D2 cleaning report is missing temporal metric {name!r}")
        array = np.asarray(transition_metrics[name], dtype=np.int64)
        if array.shape != (transition_count,):
            raise ValueError(
                f"temporal metric {name!r} must have shape "
                f"({transition_count},), got {array.shape}"
            )
        if np.any(array < 0):
            raise ValueError(f"temporal metric {name!r} contains negative values")
        values[name] = array

    threshold_map = {
        "state_component_skew_ns": config.max_state_skew_ns,
        "action_component_skew_ns": config.max_action_skew_ns,
        "post_state_component_skew_ns": config.max_post_state_skew_ns,
        "arm_state_age_at_action_ns": config.max_state_age_ns,
        "hand_state_age_at_action_ns": config.max_state_age_ns,
        "head_frame_age_at_action_ns": config.max_camera_age_ns,
        "wrist_frame_age_at_action_ns": config.max_camera_age_ns,
        "head_wrist_frame_skew_ns": config.max_camera_skew_ns,
    }

    keep = np.ones(transition_count, dtype=bool)
    flag_counts: dict[str, int] = {}
    warnings: list[str] = []
    for name, threshold in threshold_map.items():
        if threshold is None:
            flag_counts[name] = 0
            continue
        flagged = values[name] > threshold
        count = int(np.count_nonzero(flagged))
        flag_counts[name] = count
        if count:
            warnings.append(f"{name}: {count} transitions exceed {threshold} ns")
            if config.exceedance_policy == "exclude":
                keep &= ~flagged

    keep.setflags(write=False)
    return TemporalEvaluation(
        metrics={name: summary(array) for name, array in values.items()},
        keep_mask=keep,
        flag_counts=flag_counts,
        warnings=tuple(warnings),
    )
