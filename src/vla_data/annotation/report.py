"""Aggregation helpers over persisted dataset annotation summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vla_data.annotation.batch import SUMMARY_SCHEMA_NAME, SUMMARY_VERSION


def load_dataset_summary(path: str | Path) -> dict[str, Any]:
    """Load and validate one dataset annotation summary document."""

    with Path(path).open(encoding="utf-8") as stream:
        summary = json.load(stream)
    if not isinstance(summary, dict):
        raise TypeError(f"annotation summary must be an object: {path}")
    if summary.get("schema_name") != SUMMARY_SCHEMA_NAME:
        raise ValueError(f"unexpected schema_name in {path}")
    if summary.get("schema_version") != SUMMARY_VERSION:
        raise ValueError(f"unsupported schema_version in {path}")
    return summary


def aggregate_dataset_summaries(
    paths: list[str | Path],
) -> dict[str, Any]:
    """Merge multiple dataset annotation summaries deterministically."""

    merged_counts = [
        "episodes_discovered",
        "episodes_eligible",
        "episodes_ineligible",
        "episodes_processed",
        "episodes_skipped",
        "episodes_failed",
        "episodes_would_process",
        "episodes_would_skip",
        "invalid_response_count",
        "retry_count",
    ]
    totals = {key: 0 for key in merged_counts}
    review_state_counts: dict[str, int] = {}
    confidences: list[float] = []
    episodes: list[dict[str, Any]] = []
    for path in paths:
        summary = load_dataset_summary(path)
        for key in merged_counts:
            totals[key] += int(summary.get(key, 0))
        for status, count in summary.get("review_state_counts", {}).items():
            review_state_counts[status] = review_state_counts.get(status, 0) + int(
                count
            )
        for result in summary.get("results", []):
            if isinstance(result.get("confidence"), (int, float)):
                confidences.append(float(result["confidence"]))
            episodes.append(
                {
                    "episode_id": result.get("episode_id"),
                    "status": result.get("status"),
                    "instruction": result.get("instruction"),
                }
            )
    return {
        "schema_name": SUMMARY_SCHEMA_NAME,
        "schema_version": SUMMARY_VERSION,
        "aggregated_from": [str(path) for path in paths],
        **totals,
        "review_state_counts": review_state_counts,
        "mean_confidence": (
            round(sum(confidences) / len(confidences), 4) if confidences else None
        ),
        "episodes": sorted(episodes, key=lambda item: str(item["episode_id"])),
    }
