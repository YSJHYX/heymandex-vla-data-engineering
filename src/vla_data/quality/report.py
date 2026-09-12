"""D3 quality artifact writers and dataset-level aggregation."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from vla_data.quality.metrics import summary

QUALITY_SCHEMA_NAME = "vla_quality_report"
QUALITY_SCHEMA_VERSION = 1
QUALITY_STATUSES = (
    "ACCEPT",
    "ACCEPT_WITH_WARNING",
    "EXCLUDE_FROM_EXPERT_TRAINING",
    "REJECT",
)


def write_quality_artifacts(
    output_dir: str | Path,
    report: dict[str, object],
    keep_mask: np.ndarray,
) -> tuple[Path, Path]:
    """Write report/mask outside the Curated episode using atomic replaces."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "quality_report.json"
    mask_path = root / "quality_mask.npy"
    if report_path.exists() or mask_path.exists():
        raise FileExistsError(f"refusing to overwrite D3 quality artifacts in {root}")

    report_fd, report_temp_name = tempfile.mkstemp(dir=root, suffix=".json.tmp")
    mask_fd, mask_temp_name = tempfile.mkstemp(dir=root, suffix=".npy.tmp")
    os.close(report_fd)
    os.close(mask_fd)
    report_temp = Path(report_temp_name)
    mask_temp = Path(mask_temp_name)
    try:
        with report_temp.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        with mask_temp.open("wb") as stream:
            np.save(stream, np.asarray(keep_mask, dtype=bool), allow_pickle=False)
        os.replace(report_temp, report_path)
        os.replace(mask_temp, mask_path)
    finally:
        report_temp.unlink(missing_ok=True)
        mask_temp.unlink(missing_ok=True)
    return report_path, mask_path


def aggregate_quality_reports(
    report_paths: Sequence[str | Path],
    output_path: str | Path,
) -> Path:
    """Aggregate episode reports without opening or changing Curated data."""

    reports = [_load_report(Path(path)) for path in report_paths]
    status_counts = Counter(str(report["status"]) for report in reports)
    temporal_metrics: dict[str, list[float]] = {}
    trajectory_metrics: dict[str, list[float]] = {}
    visual_metrics: dict[str, dict[str, list[float]]] = {
        "head": {},
        "right_wrist": {},
    }

    for report in reports:
        for name, metric in dict(report.get("temporal", {})).items():
            median = dict(metric).get("median")
            if median is not None:
                temporal_metrics.setdefault(name, []).append(float(median))
        for name, metric in dict(report.get("trajectory", {})).items():
            if isinstance(metric, dict) and isinstance(metric.get("summary"), dict):
                median = metric["summary"].get("median")
                if median is not None:
                    trajectory_metrics.setdefault(name, []).append(float(median))
        visual = dict(report.get("visual", {}))
        for role, role_metrics in visual_metrics.items():
            role_report = dict(visual.get(role, {}))
            for name in ("mean_luminance", "blur_score", "dark_pixel_ratio"):
                metric = role_report.get(name)
                if isinstance(metric, dict) and metric.get("median") is not None:
                    role_metrics.setdefault(name, []).append(float(metric["median"]))

    output = {
        "schema_name": "vla_dataset_quality_summary",
        "schema_version": 1,
        "episode_count": len(reports),
        "transition_count": sum(int(r["transition_count"]) for r in reports),
        "clean_transition_count": sum(
            int(r["clean_transition_count"]) for r in reports
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "temporal_distributions_of_episode_medians": {
            name: summary(values) for name, values in temporal_metrics.items()
        },
        "trajectory_distributions_of_episode_medians": {
            name: summary(values) for name, values in trajectory_metrics.items()
        },
        "visual_distributions_of_episode_medians": {
            role: {name: summary(values) for name, values in metrics.items()}
            for role, metrics in visual_metrics.items()
        },
        "active_joint_distributions": {
            name: summary([int(dict(r["activity"])[name]) for r in reports])
            for name in (
                "active_joint_count",
                "state_active_joint_count",
                "action_active_joint_count",
                "arm_active_joint_count",
                "hand_active_joint_count",
                "arm_action_active_joint_count",
                "hand_action_active_joint_count",
            )
        },
        "episode_ids": [str(report["episode_id"]) for report in reports],
    }
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return destination.resolve()


def _load_report(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    if report.get("schema_name") != QUALITY_SCHEMA_NAME:
        raise ValueError(f"not a D3 quality report: {path}")
    return report
