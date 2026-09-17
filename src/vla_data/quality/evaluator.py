"""Read-only D3 evaluator for one Curated v1 episode."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path

import numpy as np

from vla_data.io.curated_episode import CuratedEpisode
from vla_data.quality.report import (
    QUALITY_SCHEMA_NAME,
    QUALITY_SCHEMA_VERSION,
    aggregate_quality_reports,
    write_quality_artifacts,
)
from vla_data.quality.temporal import (
    TemporalQualityConfig,
    evaluate_temporal,
)
from vla_data.quality.trajectory import ActivityConfig, evaluate_trajectory
from vla_data.quality.visual import VisualQualityConfig, evaluate_visual
from vla_data.validation.curated_v1 import validate_curated_episode


@dataclass(frozen=True)
class QualityConfig:
    temporal: TemporalQualityConfig = field(default_factory=TemporalQualityConfig)
    visual: VisualQualityConfig = field(default_factory=VisualQualityConfig)
    activity: ActivityConfig = field(default_factory=ActivityConfig)


@dataclass(frozen=True)
class QualityEvaluationResult:
    report_path: Path
    mask_path: Path
    status: str
    transition_count: int
    clean_transition_count: int


def evaluate_curated_episode(
    episode_dir: str | Path,
    output_dir: str | Path,
    *,
    config: QualityConfig | None = None,
) -> QualityEvaluationResult:
    """Evaluate Curated data and emit only separate report/mask artifacts."""

    selected_config = config or QualityConfig()
    episode = CuratedEpisode.load(episode_dir)
    d2_validation = validate_curated_episode(episode)
    cleaning_report = _load_cleaning_report(episode.episode_dir)

    temporal = evaluate_temporal(
        dict(cleaning_report["transition_temporal_metrics"]),
        episode.transition_count,
        selected_config.temporal,
    )
    trajectory, activity = evaluate_trajectory(episode, selected_config.activity)
    visual = evaluate_visual(episode, selected_config.visual)
    visual_report = dict(visual.report)
    visual_report["head"]["frame_age_at_action_ns"] = temporal.metrics[
        "head_frame_age_at_action_ns"
    ]
    visual_report["right_wrist"]["frame_age_at_action_ns"] = temporal.metrics[
        "wrist_frame_age_at_action_ns"
    ]
    visual_report["head_wrist_frame_skew_ns"] = temporal.metrics[
        "head_wrist_frame_skew_ns"
    ]

    keep = temporal.keep_mask & visual.keep_mask
    warnings = (
        list(d2_validation.warnings) + list(temporal.warnings) + list(visual.warnings)
    )
    for component in ("arm_state", "hand_state", "arm_action", "hand_action"):
        skipped = int(
            trajectory[f"{component}_velocity_timing"]["nonpositive_dt_pair_count"]
        )
        if skipped:
            warnings.append(
                f"{component.replace('_', ' ')} velocity skipped {skipped} pairs "
                "with non-positive dt"
            )
    exclusion_reasons: list[str] = []
    if np.any(visual.keep_mask) and not np.all(visual.keep_mask):
        warnings.append(
            "QUALITY_INVALID: required RGB rows excluded; remaining clean rows retained"
        )

    # QUALITY_INVALID rows stay masked out. One bad image must not veto clean
    # rows; no metric, threshold, static policy or mask definition is changed.
    hard_invalid = not d2_validation.passed or not bool(np.any(visual.keep_mask))
    if hard_invalid:
        status = "REJECT"
        exclusion_reasons.extend(d2_validation.errors)
        if not np.all(visual.keep_mask):
            exclusion_reasons.append("required RGB hard-integrity failure")
    elif bool(activity["static_episode"]):
        status = "EXCLUDE_FROM_EXPERT_TRAINING"
        keep[:] = False
        exclusion_reasons.append(
            "static episode is unsuitable as expert BC demonstration"
        )
    elif not np.any(keep):
        status = "EXCLUDE_FROM_EXPERT_TRAINING"
        exclusion_reasons.append(
            "configured quality thresholds excluded all transitions"
        )
    elif warnings or not np.all(keep):
        status = "ACCEPT_WITH_WARNING"
        if not np.all(keep):
            exclusion_reasons.append(
                "QUALITY_INVALID: RGB integrity or configured quality masks excluded transitions"
            )
    else:
        status = "ACCEPT"

    keep.setflags(write=False)
    action_ts = np.asarray(episode.trajectory["action_timestamp_ns"], dtype=np.int64)
    timeline_span_s = (
        float((np.max(action_ts) - np.min(action_ts)) / 1e9)
        if action_ts.size > 1
        else 0.0
    )
    duration_s = sum(
        float((action_ts[end - 1] - action_ts[start]) / 1e9)
        for start, end in pairwise(episode.segment_offsets)
        if end - start > 1
    )
    report: dict[str, object] = {
        "schema_name": QUALITY_SCHEMA_NAME,
        "schema_version": QUALITY_SCHEMA_VERSION,
        "episode_id": str(episode.metadata["episode_id"]),
        "status": status,
        "transition_count": episode.transition_count,
        "causal_valid_transition_count": int(
            cleaning_report.get("accepted_count", episode.transition_count)
        ),
        "clean_transition_count": int(np.count_nonzero(keep)),
        "duration_s": duration_s,
        "timeline_span_s": timeline_span_s,
        "temporal": temporal.metrics,
        "temporal_flag_counts": temporal.flag_counts,
        "trajectory": trajectory,
        "visual": visual_report,
        "activity": activity,
        "warnings": warnings,
        "exclusion_reasons": exclusion_reasons,
        "d2_validation": {
            "quality": d2_validation.quality,
            "errors": list(d2_validation.errors),
            "warnings": list(d2_validation.warnings),
        },
        "config": {
            "temporal": asdict(selected_config.temporal),
            "visual": asdict(selected_config.visual),
            "activity": asdict(selected_config.activity),
        },
        "source_provenance": {
            "hardware_execution": episode.metadata.get("hardware_execution"),
            "source_dataset_status": episode.metadata.get("source_dataset_status"),
            "source_sg100_feedback_map_status": episode.metadata.get(
                "source_sg100_feedback_map_status"
            ),
        },
    }
    report_path, mask_path = write_quality_artifacts(output_dir, report, keep)
    return QualityEvaluationResult(
        report_path=report_path,
        mask_path=mask_path,
        status=status,
        transition_count=episode.transition_count,
        clean_transition_count=int(np.count_nonzero(keep)),
    )


# Concise public package API; keep the explicit historical name compatible.
evaluate_episode = evaluate_curated_episode


def _load_cleaning_report(episode_dir: Path) -> dict[str, object]:
    path = episode_dir / "cleaning_report.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError("cleaning_report.json root must be an object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    episode = subparsers.add_parser("episode")
    episode.add_argument("episode_dir", type=Path)
    episode.add_argument("output_dir", type=Path)
    dataset = subparsers.add_parser("dataset")
    dataset.add_argument("output_path", type=Path)
    dataset.add_argument("report_paths", nargs="+", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "episode":
        result = evaluate_curated_episode(args.episode_dir, args.output_dir)
        print(result.report_path)
    else:
        print(aggregate_quality_reports(args.report_paths, args.output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
