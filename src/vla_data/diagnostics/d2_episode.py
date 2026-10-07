"""On-demand, read-only D2 evidence and RAW signal diagnostics.

``d2_evidence`` comes exclusively from the persisted cleaning report or the
existing :func:`synchronize_episode` implementation.  ``raw_diagnostics`` is a
separate explanatory layer and never changes or replaces a D2 rejection code.
"""

from __future__ import annotations

import copy
import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from vla_data.cleaning.causal_sync import (
    CausalSyncResult,
    RejectedCandidate,
    synchronize_episode,
)
from vla_data.diagnostics.camera_health import CameraHealth, camera_health
from vla_data.io.raw_episode import RawEpisode

SCHEMA_NAME = "vla_d2_episode_diagnostics"
SCHEMA_VERSION = 1
MAX_TIMELINE_POINTS = 600
MAX_CACHE_ENTRIES = 24


class _EpisodeDiagnosticsCache:
    def __init__(self, max_entries: int = MAX_CACHE_ENTRIES) -> None:
        self._max_entries = max_entries
        self._values: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple[Any, ...]) -> dict[str, Any] | None:
        with self._lock:
            value = self._values.get(key)
            if value is None:
                return None
            self._values.move_to_end(key)
            return copy.deepcopy(value)

    def put(self, key: tuple[Any, ...], value: dict[str, Any]) -> None:
        with self._lock:
            self._values[key] = copy.deepcopy(value)
            self._values.move_to_end(key)
            while len(self._values) > self._max_entries:
                self._values.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._values.clear()


_CACHE = _EpisodeDiagnosticsCache()


def clear_diagnostics_cache() -> None:
    """Clear the process-local cache (primarily for deterministic tests)."""

    _CACHE.clear()


def diagnose_episode(
    *,
    run_name: str,
    raw_path: str | Path,
    cleaning_report_path: str | Path | None = None,
    d2_status: str | None = None,
    reason: str | None = None,
    max_timeline_points: int = MAX_TIMELINE_POINTS,
) -> dict[str, Any]:
    """Return one Episode's cached D2 evidence and RAW diagnostics."""

    raw_file = Path(raw_path).resolve(strict=True)
    report_file = (
        Path(cleaning_report_path).resolve()
        if cleaning_report_path is not None
        else None
    )
    media_root = raw_file.with_name(f"{raw_file.stem}_media")
    key = _cache_key(
        run_name,
        raw_file,
        report_file,
        media_root,
        d2_status,
        max_timeline_points,
    )
    result = _CACHE.get(key)
    if result is None:
        result = _diagnose_uncached(
            run_name=run_name,
            raw_file=raw_file,
            report_file=report_file,
            d2_status=d2_status,
            max_timeline_points=max_timeline_points,
        )
        _CACHE.put(key, result)
    return _select_reason(result, reason)


def compare_episode_diagnostics(
    current: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    """Create a deterministic, presentation-ready comparison."""

    current_camera = current["raw_diagnostics"]["camera_health"]
    baseline_camera = baseline["raw_diagnostics"]["camera_health"]
    current_action = current["raw_diagnostics"]["action_health"]
    baseline_action = baseline["raw_diagnostics"]["action_health"]
    rows = [
        _comparison_row(
            "HEAD valid",
            baseline_camera["head"]["valid_ratio"],
            current_camera["head"]["valid_ratio"],
            lower_is_regression=True,
            kind="ratio",
        ),
        _comparison_row(
            "HEAD unique timestamps",
            baseline_camera["head"]["timestamp_unique_count"],
            current_camera["head"]["timestamp_unique_count"],
            lower_is_regression=True,
            kind="count",
        ),
        _comparison_row(
            "HEAD median age",
            baseline_camera["head"]["master_minus_camera_ns"]["median"],
            current_camera["head"]["master_minus_camera_ns"]["median"],
            lower_is_regression=False,
            kind="duration_ns",
        ),
        _comparison_row(
            "WRIST valid",
            baseline_camera["wrist"]["valid_ratio"],
            current_camera["wrist"]["valid_ratio"],
            lower_is_regression=True,
            kind="ratio",
        ),
        _comparison_row(
            "WRIST median age",
            baseline_camera["wrist"]["master_minus_camera_ns"]["median"],
            current_camera["wrist"]["master_minus_camera_ns"]["median"],
            lower_is_regression=False,
            kind="duration_ns",
        ),
        _comparison_row(
            "ACTION READY",
            baseline_action["action_ready_ratio"],
            current_action["action_ready_ratio"],
            lower_is_regression=True,
            kind="ratio",
        ),
        _comparison_row(
            "D2 accepted",
            baseline["d2_evidence"]["accepted_count"],
            current["d2_evidence"]["accepted_count"],
            lower_is_regression=True,
            kind="count",
        ),
    ]
    return {
        "schema_name": "vla_d2_episode_diagnostics_comparison",
        "schema_version": 1,
        "baseline_episode": baseline["episode_id"],
        "current_episode": current["episode_id"],
        "rows": rows,
    }


def _diagnose_uncached(
    *,
    run_name: str,
    raw_file: Path,
    report_file: Path | None,
    d2_status: str | None,
    max_timeline_points: int,
) -> dict[str, Any]:
    raw = RawEpisode.load(raw_file)
    # This is the only source of transition-level rejection evidence.  It is
    # deliberately called on demand and never writes or publishes anything.
    sync = synchronize_episode(raw)
    report = _load_report(report_file)
    d2_evidence = _d2_evidence(sync, report, raw, d2_status)

    master = raw.require("timestamp_ns")
    head = camera_health(
        "head",
        master,
        raw.require("head_camera_host_timestamp_ns"),
        raw.require("head_camera_valid"),
        raw.require("head_rgb_frame_index"),
    )
    wrist = camera_health(
        "wrist",
        master,
        raw.require("wrist_camera_host_timestamp_ns"),
        raw.require("wrist_camera_valid"),
        raw.require("wrist_rgb_frame_index"),
    )
    action = _action_health(raw, d2_evidence)
    feedback = _feedback_health(raw, d2_evidence, sync)
    causality = _causality_metrics(raw, d2_evidence)
    root_cause = _root_cause(head, wrist, d2_evidence, action)
    why_rejected = _why_rejected(d2_evidence, action, root_cause)
    timeline = _timeline(raw, max_timeline_points)

    result = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "scope": "DIAGNOSTIC_ONLY; read-only; not a D2/D3 eligibility authority",
        "run": run_name,
        "episode_id": raw.episode_id,
        "d2_status": d2_evidence["status"],
        "source": {
            "raw_path": str(raw_file),
            "raw_size": raw_file.stat().st_size,
            "raw_mtime_ns": raw_file.stat().st_mtime_ns,
            "cleaning_report_path": str(report_file)
            if report_file is not None and report_file.is_file()
            else None,
        },
        "d2_evidence": d2_evidence,
        "raw_diagnostics": {
            "camera_health": {"head": head.as_dict(), "wrist": wrist.as_dict()},
            "action_health": action,
            "feedback_health": feedback,
            "causality_metrics": causality,
            "inferred_root_cause": root_cause,
            "why_episode_rejected": why_rejected,
            "timeline": timeline,
        },
    }
    result["diagnostic_report_text"] = _report_text(result)
    return result


def _cache_key(
    run_name: str,
    raw_file: Path,
    report_file: Path | None,
    media_root: Path,
    d2_status: str | None,
    max_timeline_points: int,
) -> tuple[Any, ...]:
    raw_stat = raw_file.stat()
    return (
        run_name,
        raw_file.stem,
        str(raw_file),
        raw_stat.st_mtime_ns,
        raw_stat.st_size,
        _path_signature(report_file),
        _path_signature(media_root / "head" / "rgb"),
        _path_signature(media_root / "right_wrist" / "rgb"),
        d2_status,
        max_timeline_points,
    )


def _path_signature(path: Path | None) -> tuple[int, int] | None:
    if path is None or not path.exists():
        return None
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def _load_report(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    required = {
        "candidate_count",
        "accepted_count",
        "rejected_count",
        "rejection_counts",
        "rejection_group_counts",
    }
    return value if isinstance(value, dict) and required <= value.keys() else None


def _d2_evidence(
    sync: CausalSyncResult,
    report: dict[str, Any] | None,
    raw: RawEpisode,
    d2_status: str | None,
) -> dict[str, Any]:
    if report is not None:
        candidate_count = int(report["candidate_count"])
        accepted_count = int(report["accepted_count"])
        rejected_count = int(report["rejected_count"])
        rejection_counts = _int_dict(report["rejection_counts"])
        group_counts = _groups(report["rejection_group_counts"])
        aggregate_source = "cleaning_report.json"
    else:
        candidate_count = int(sync.candidate_count)
        accepted_count = len(sync.transitions)
        rejected_count = len(sync.rejected)
        rejection_counts = _int_dict(sync.rejection_counts)
        group_counts = _groups(sync.rejection_group_counts)
        aggregate_source = "synchronize_episode(read-only)"

    most_frequent_reason = _primary(rejection_counts)
    primary_reason, primary_basis = _primary_failure_reason(
        rejection_counts,
        group_counts,
        candidate_count=candidate_count,
        accepted_count=accepted_count,
    )
    primary_group = primary_reason.split("_", 1)[0] if primary_reason else None
    representatives = _sample_rejections(sync.rejected, raw)
    by_reason = {
        reason: _sample_rejections(sync.rejected, raw, reason=reason)
        for reason in rejection_counts
    }
    status = d2_status or ("FAILED" if accepted_count == 0 else "SUCCESS")
    return {
        "status": status,
        "aggregate_source": aggregate_source,
        "representative_source": "synchronize_episode(read-only)",
        "candidate_count": candidate_count,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "acceptance_rate": accepted_count / candidate_count if candidate_count else 0.0,
        "primary_failure_group": primary_group,
        "primary_failure_reason": primary_reason,
        "primary_failure_basis": primary_basis,
        "most_frequent_rejection_reason": most_frequent_reason,
        "rejection_counts": rejection_counts,
        "rejection_group_counts": group_counts,
        "rejection_groups_may_overlap": True,
        "exclusive_first_failure_funnel": None,
        "exclusive_funnel_note": (
            "Not shown: raw D2 rejection groups may overlap, and diagnostics do "
            "not infer an exclusive partition."
        ),
        "representative_rejections": representatives,
        "representative_rejections_by_reason": by_reason,
        "selected_reason": None,
        "selected_representative_rejections": representatives,
    }


def _int_dict(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(key): int(count) for key, count in sorted(value.items())}


def _groups(value: Any) -> dict[str, int]:
    raw = _int_dict(value)
    return {
        group: int(raw.get(group, 0))
        for group in ("ACTION", "FEEDBACK", "CAUSALITY", "CAMERA")
    }


def _primary(counts: dict[str, int]) -> str | None:
    positive = [(name, count) for name, count in counts.items() if count > 0]
    if not positive:
        return None
    return min(positive, key=lambda item: (-item[1], item[0]))[0]


def _primary_failure_reason(
    reason_counts: dict[str, int],
    group_counts: dict[str, int],
    *,
    candidate_count: int,
    accepted_count: int,
) -> tuple[str | None, str]:
    """Prefer a unique reason that blocked every action-valid candidate."""

    action_ready = max(0, candidate_count - int(group_counts.get("ACTION", 0)))
    if accepted_count == 0 and action_ready > 0:
        blockers = [
            reason
            for reason, count in reason_counts.items()
            if not reason.startswith("ACTION_") and count == action_ready
        ]
        if len(blockers) == 1:
            return blockers[0], "unique reason covering every action-valid candidate"
    return _primary(reason_counts), "most frequent D2 rejection reason"


def _sample_rejections(
    rejected: tuple[RejectedCandidate, ...],
    raw: RawEpisode,
    reason: str | None = None,
) -> list[dict[str, Any]]:
    candidates = [row for row in rejected if reason is None or reason in row.reasons]
    if not candidates:
        return []
    positions = {
        0: "first",
        len(candidates) // 2: "middle",
        len(candidates) - 1: "last",
    }
    samples = []
    for index in sorted(positions):
        rejected_row = candidates[index]
        samples.append(
            {
                "sample_position": positions[index],
                "source_tick_index": rejected_row.source_tick_index,
                "reasons": list(rejected_row.reasons),
                "raw_context": _raw_context(raw, rejected_row.source_tick_index),
            }
        )
    return samples


def _raw_context(raw: RawEpisode, row: int) -> dict[str, Any]:
    master = int(raw.require("timestamp_ns")[row])

    def camera(role: str) -> dict[str, Any]:
        timestamp = int(raw.require(f"{role}_camera_host_timestamp_ns")[row])
        return {
            "timestamp_ns": timestamp,
            "valid": bool(raw.require(f"{role}_camera_valid")[row]),
            "frame_index": int(raw.require(f"{role}_rgb_frame_index")[row]),
            "age_ns": master - timestamp if master > 0 and timestamp > 0 else None,
        }

    return {
        "master_timestamp_ns": master,
        "arm_command_valid": bool(raw.require("arm_command_valid")[row]),
        "hand_command_valid": bool(raw.require("hand_command_valid")[row]),
        "robot_qcmd_17d_valid": bool(raw.require("robot_qcmd_17d_valid")[row]),
        "head": camera("head"),
        "wrist": camera("wrist"),
    }


def _action_health(raw: RawEpisode, evidence: dict[str, Any]) -> dict[str, Any]:
    candidates = evidence["candidate_count"]
    size = min(candidates, int(raw.require("timestamp_ns").size))

    def values(key: str) -> np.ndarray:
        return np.asarray(raw.require(key)[:size])

    arm = values("arm_qcmd_sent_rad")
    hand = values("hand_qcmd_effective_canonical_rad")
    robot = values("robot_qcmd_17d_rad")
    finite = (
        np.all(np.isfinite(arm), axis=1)
        & np.all(np.isfinite(hand), axis=1)
        & np.all(np.isfinite(robot), axis=1)
    )
    expected = np.concatenate((arm, hand), axis=1)
    exact = finite & np.all(robot == expected, axis=1)
    action_ready_count = max(
        0,
        int(candidates) - int(evidence["rejection_group_counts"].get("ACTION", 0)),
    )
    return {
        "total_candidates": int(candidates),
        "arm_command_valid": _count_ratio(values("arm_command_valid"), size),
        "hand_command_valid": _count_ratio(values("hand_command_valid"), size),
        "robot_qcmd_17d_valid": _count_ratio(values("robot_qcmd_17d_valid"), size),
        "finite_action": _count_ratio(finite, size),
        "positive_action_timestamps": _count_ratio(
            (values("arm_qcmd_source_timestamp_ns") > 0)
            & (values("hand_qcmd_source_timestamp_ns") > 0),
            size,
        ),
        "component_match_17d": _count_ratio(exact, size),
        "action_ready_count": action_ready_count,
        "action_ready_ratio": action_ready_count / candidates if candidates else 0.0,
        "action_ready_definition_source": (
            "candidate_count minus D2 ACTION-group rejected candidates; D2 evidence "
            "comes from cleaning_report/synchronize_episode"
        ),
    }


def _count_ratio(mask: np.ndarray, total: int) -> dict[str, int | float]:
    count = int(np.count_nonzero(np.asarray(mask, dtype=bool)))
    return {"count": count, "total": total, "ratio": count / total if total else 0.0}


def _feedback_health(
    raw: RawEpisode, evidence: dict[str, Any], sync: CausalSyncResult
) -> dict[str, Any]:
    total = int(raw.require("timestamp_ns").size)
    arm = raw.require("arm_qpos_rad")
    hand = raw.require("hand_feedback_sdk_rad")
    modes = raw.require("hand_feedback_modes")
    reasons = evidence["rejection_counts"]
    action_ready = max(
        0,
        evidence["candidate_count"]
        - evidence["rejection_group_counts"].get("ACTION", 0),
    )
    pre_failed = sum(
        any(
            "_PRE_" in reason and reason.startswith(("FEEDBACK_", "CAUSALITY_"))
            for reason in row.reasons
        )
        for row in sync.rejected
    )
    post_failed = sum(
        any(
            "_POST_" in reason and reason.startswith(("FEEDBACK_", "CAUSALITY_"))
            for reason in row.reasons
        )
        for row in sync.rejected
    )
    return {
        "total_rows": total,
        "arm_feedback_valid": _count_ratio(raw.require("arm_feedback_valid"), total),
        "hand_feedback_valid": _count_ratio(raw.require("hand_feedback_valid"), total),
        "arm_finite": _count_ratio(np.all(np.isfinite(arm), axis=1), total),
        "hand_finite": _count_ratio(np.all(np.isfinite(hand), axis=1), total),
        "hand_mode7_all_joints": _count_ratio(np.all(modes == 7, axis=1), total),
        "positive_feedback_timestamps": _count_ratio(
            (raw.require("arm_qpos_source_timestamp_ns") > 0)
            & (raw.require("hand_feedback_source_timestamp_ns") > 0),
            total,
        ),
        "pre_feedback_availability": {
            "count": max(0, action_ready - pre_failed),
            "total": action_ready,
            "ratio": (action_ready - pre_failed) / action_ready
            if action_ready
            else 0.0,
        },
        "post_feedback_availability": {
            "count": max(0, action_ready - post_failed),
            "total": action_ready,
            "ratio": (action_ready - post_failed) / action_ready
            if action_ready
            else 0.0,
        },
        "d2_pre_failure_counts": {
            key: count
            for key, count in reasons.items()
            if "_PRE_" in key and key.startswith(("FEEDBACK_", "CAUSALITY_"))
        },
        "d2_post_failure_counts": {
            key: count
            for key, count in reasons.items()
            if "_POST_" in key and key.startswith(("FEEDBACK_", "CAUSALITY_"))
        },
    }


def _causality_metrics(raw: RawEpisode, evidence: dict[str, Any]) -> dict[str, Any]:
    master = np.asarray(raw.require("timestamp_ns"), dtype=np.int64)
    differences = np.diff(master)
    reasons = evidence["rejection_counts"]
    causality_reasons = {
        key: count for key, count in reasons.items() if key.startswith("CAUSALITY_")
    }
    pre_unavailable = sum(
        count for key, count in causality_reasons.items() if "_PRE_UNAVAILABLE" in key
    )
    post_unavailable = sum(
        count for key, count in causality_reasons.items() if "_POST_UNAVAILABLE" in key
    )
    strict_conflicts = sum(
        count
        for key, count in causality_reasons.items()
        if key.endswith(("PRE_NOT_STRICT", "POST_NOT_STRICT"))
    )
    group_count = int(evidence["rejection_group_counts"].get("CAUSALITY", 0))
    return {
        "status": "HEALTHY" if group_count == 0 else "DEGRADED",
        "d2_causality_group_rejections": group_count,
        "d2_causality_reasons": causality_reasons,
        "valid_strict_order_accepted_count": evidence["accepted_count"],
        "pre_unavailable_count": pre_unavailable,
        "post_unavailable_count": post_unavailable,
        "strict_order_conflict_count": strict_conflicts,
        "master_non_monotonic_count": int(np.count_nonzero(differences < 0)),
        "master_equal_timestamp_count": int(np.count_nonzero(differences == 0)),
        "note": (
            "Accepted strict-order count and rejection reasons are D2 evidence; "
            "master monotonicity counts are RAW diagnostics."
        ),
    }


def _root_cause(
    head: CameraHealth,
    wrist: CameraHealth,
    evidence: dict[str, Any],
    action: dict[str, Any],
) -> dict[str, Any]:
    reasons = evidence["rejection_counts"]
    candidates = (
        ("HEAD", head, "CAMERA_HEAD_CAMERA_UNAVAILABLE", wrist)
        if head.status in {"FROZEN", "MISSING", "STALE", "CLOCK_MISMATCH"}
        else ("WRIST", wrist, "CAMERA_WRIST_CAMERA_UNAVAILABLE", head)
        if wrist.status in {"FROZEN", "MISSING", "STALE", "CLOCK_MISMATCH"}
        else None
    )
    if candidates is None or reasons.get(candidates[2], 0) <= 0:
        return {
            "code": "UNKNOWN",
            "label": "UNKNOWN",
            "severity": "UNKNOWN",
            "inference_source": "RAW-derived diagnosis",
            "findings": [
                "No conservative RAW root cause explains the primary D2 evidence."
            ],
            "evidence": [],
            "suggested_subsystem": None,
            "suggested_checks": [
                "Inspect D2 rejection evidence and RAW producer logs."
            ],
            "safety_note": "D2 cleaning rules should NOT be loosened.",
        }

    role, health, reason, other = candidates
    rejected = int(reasons[reason])
    action_ready = int(action["action_ready_count"])
    all_remaining = (
        evidence["accepted_count"] == 0
        and rejected == action_ready
        and action_ready > 0
    )
    label = f"{role} CAMERA {health.status}"
    finding = (
        f"{rejected} action-valid candidate transitions had no usable "
        f"{role.lower()}-camera observation"
    )
    if all_remaining:
        finding += "; all action-valid transitions were lost at this camera evidence"
    finding += "."
    checks = [
        "Check the RealSense pipeline.",
        "Check the D455 capture thread alive state.",
        "Check the latest-frame update path.",
        "Check USB/device state and disconnect logs.",
        "Check recorder camera freshness state.",
    ]
    return {
        "code": label.replace(" ", "_"),
        "label": label,
        "severity": "CRITICAL" if evidence["accepted_count"] == 0 else "WARNING",
        "inference_source": "RAW-derived diagnosis",
        "findings": [finding, *health.findings],
        "evidence": [
            *health.evidence,
            f"D2 {reason} = {rejected}",
            f"action-ready candidates = {action_ready}",
            f"other camera status = {other.status}",
        ],
        "suggested_subsystem": (
            "D455 / head-camera producer path"
            if role == "HEAD"
            else "wrist-camera producer path"
        ),
        "suggested_checks": checks,
        "all_action_ready_lost_to_camera": all_remaining,
        "safety_note": "D2 cleaning rules should NOT be loosened.",
    }


def _why_rejected(
    evidence: dict[str, Any],
    action: dict[str, Any],
    root_cause: dict[str, Any],
) -> list[str]:
    if evidence["rejected_count"] == 0:
        return ["No D2 candidate transition was rejected."]
    steps = [
        f"{evidence['candidate_count']} candidate transitions were found.",
        f"{evidence['rejection_group_counts']['ACTION']} candidates failed ACTION validity.",
        f"{action['action_ready_count']} candidates passed the D2 ACTION gate.",
    ]
    if root_cause.get("all_action_ready_lost_to_camera"):
        reason = evidence["primary_failure_reason"]
        role = "head" if "HEAD" in str(reason) else "wrist"
        steps.extend(
            [
                f"All remaining candidates required a usable {role} observation.",
                f"D2 rejected them with {reason}.",
                f"RAW diagnostics classified the {role} timestamp signal as FROZEN.",
                "Therefore no transition could be published.",
            ]
        )
    else:
        steps.append(
            f"The primary D2 evidence was {evidence['primary_failure_reason'] or 'UNKNOWN'}."
        )
    return steps


def _timeline(raw: RawEpisode, max_points: int) -> dict[str, Any]:
    if max_points <= 1:
        raise ValueError("max_timeline_points must be greater than one")
    master = np.asarray(raw.require("timestamp_ns"), dtype=np.int64)
    indices = _downsample_indices(int(master.size), max_points)
    head_ts = raw.require("head_camera_host_timestamp_ns")
    wrist_ts = raw.require("wrist_camera_host_timestamp_ns")
    head_valid = raw.require("head_camera_valid")
    wrist_valid = raw.require("wrist_camera_valid")
    points = []
    for row in indices:
        master_value = int(master[row])
        head_value = int(head_ts[row])
        wrist_value = int(wrist_ts[row])
        points.append(
            {
                "row": row,
                "master_timestamp_ns": master_value,
                "head_age_ns": master_value - head_value
                if master_value > 0 and head_value > 0
                else None,
                "wrist_age_ns": master_value - wrist_value
                if master_value > 0 and wrist_value > 0
                else None,
                "head_valid": bool(head_valid[row]),
                "wrist_valid": bool(wrist_valid[row]),
            }
        )
    return {
        "source_rows": int(master.size),
        "point_count": len(points),
        "downsampled": len(points) < int(master.size),
        "max_points": max_points,
        "points": points,
    }


def _downsample_indices(total: int, max_points: int) -> list[int]:
    if total <= max_points:
        return list(range(total))
    return sorted(
        {round(float(value)) for value in np.linspace(0, total - 1, num=max_points)}
    )


def _select_reason(result: dict[str, Any], reason: str | None) -> dict[str, Any]:
    selected = copy.deepcopy(result)
    evidence = selected["d2_evidence"]
    if reason:
        by_reason = evidence["representative_rejections_by_reason"]
        if reason not in by_reason:
            raise ValueError(f"D2 rejection reason not present: {reason}")
        evidence["selected_reason"] = reason
        evidence["selected_representative_rejections"] = by_reason[reason]
    return selected


def _comparison_row(
    label: str,
    baseline: float | None,
    current: float | None,
    *,
    lower_is_regression: bool,
    kind: str,
) -> dict[str, Any]:
    regression = False
    if baseline is not None and current is not None:
        if lower_is_regression:
            regression = current < baseline * 0.8 if baseline else False
        else:
            regression = current > max(baseline * 2, baseline + 100_000_000)
    return {
        "label": label,
        "kind": kind,
        "baseline": baseline,
        "current": current,
        "regression": bool(regression),
    }


def _report_text(result: dict[str, Any]) -> str:
    d2 = result["d2_evidence"]
    raw = result["raw_diagnostics"]
    head = raw["camera_health"]["head"]
    wrist = raw["camera_health"]["wrist"]
    cause = raw["inferred_root_cause"]
    groups = d2["rejection_group_counts"]
    head_age = head["master_minus_camera_ns"]["median"]
    wrist_age = wrist["master_minus_camera_ns"]["median"]
    lines = [
        "[D2 Diagnostic Report]",
        "",
        f"Episode: {result['episode_id']}",
        f"Status: {d2['status']}",
        f"Candidates: {d2['candidate_count']}",
        f"Accepted: {d2['accepted_count']}",
        f"Rejected: {d2['rejected_count']}",
        f"Primary D2 failure reason: {d2['primary_failure_reason'] or 'NONE'}",
        f"Primary RAW-derived root cause: {cause['label']}",
        "",
        "Rejection groups (may overlap):",
        *(f"{group}={groups[group]}" for group in groups),
        "",
        "Head camera:",
        f"status={head['status']}",
        f"valid={head['valid_count']}/{head['total_rows']} ({head['valid_ratio']:.2%})",
        f"timestamp_unique={head['timestamp_unique_count']}/{head['timestamp_positive_count']}",
        f"frame_index_valid={head['frame_index_valid_count']}/{head['total_rows']}",
        f"median_age_ns={head_age}",
        "",
        "Wrist camera:",
        f"status={wrist['status']}",
        f"valid={wrist['valid_count']}/{wrist['total_rows']} ({wrist['valid_ratio']:.2%})",
        f"timestamp_unique={wrist['timestamp_unique_count']}/{wrist['timestamp_positive_count']}",
        f"median_age_ns={wrist_age}",
        "",
        "Conclusion:",
        *(cause["findings"] or ["No conservative root cause inferred."]),
        f"Suggested subsystem: {cause['suggested_subsystem'] or 'UNKNOWN'}",
        "Suggested checks:",
        *(f"- {check}" for check in cause["suggested_checks"]),
        cause["safety_note"],
    ]
    return "\n".join(lines)
