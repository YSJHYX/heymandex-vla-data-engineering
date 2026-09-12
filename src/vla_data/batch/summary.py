"""Atomic writers and compact summary aggregation."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from vla_data.batch.discovery import DiscoveryIssue
from vla_data.batch.status import EpisodeResult

SUMMARY_VERSION = 1
QUALITY_OUTCOMES = (
    "ACCEPT",
    "ACCEPT_WITH_WARNING",
    "EXCLUDE_FROM_EXPERT_TRAINING",
    "REJECT",
    "FAILED",
)


def build_summary(
    stage: str,
    results: Iterable[EpisodeResult],
    issues: Iterable[DiscoveryIssue],
) -> dict[str, Any]:
    ordered = tuple(sorted(results, key=lambda item: item.episode_id))
    issue_values = tuple(issues)
    processed = sum(
        item.status not in {"SKIPPED", "FAILED", "WOULD_PROCESS"} for item in ordered
    )
    skipped = sum(item.status == "SKIPPED" for item in ordered)
    failed = sum(item.status == "FAILED" for item in ordered)
    base: dict[str, Any] = {
        "schema_name": f"vla_dataset_{stage}_summary",
        "schema_version": SUMMARY_VERSION,
        "episodes_discovered": len(ordered),
        "episodes_processed": processed,
        "episodes_skipped": skipped,
        "episodes_failed": failed,
        "episodes_would_process": sum(
            item.status == "WOULD_PROCESS" for item in ordered
        ),
        "episodes_would_skip": skipped,
        "discovery_issues": [
            {
                "path": str(issue.path),
                "error_type": issue.error_type,
                "message": issue.message,
            }
            for issue in issue_values
            if issue.episode_id is None
        ],
        "results": [item.as_dict() for item in ordered],
    }

    if stage == "build":
        base["episodes_succeeded"] = sum(
            item.status in {"SUCCESS", "SKIPPED"} for item in ordered
        )
    elif stage == "validation":
        passed = sum(
            item.status in {"SUCCESS", "WARNING", "SKIPPED"} for item in ordered
        )
        base.update(
            {
                "episodes_total": len(ordered),
                "episodes_passed": passed,
                "episodes_failed": failed,
                "transitions_total": sum(item.transitions_total for item in ordered),
                "transitions_valid": sum(
                    item.transitions_valid
                    for item in ordered
                    if item.status != "FAILED"
                ),
                "failed_episode_ids": [
                    item.episode_id for item in ordered if item.status == "FAILED"
                ],
            }
        )
    elif stage == "quality":
        counts = Counter(
            "FAILED" if item.status == "FAILED" else item.outcome for item in ordered
        )
        base["eligibility_counts"] = {
            status: int(counts.get(status, 0)) for status in QUALITY_OUTCOMES
        }
    return base


def write_summary(path: str | Path, summary: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination.resolve()


def load_summary(path: str | Path, schema_name: str) -> dict[str, Any] | None:
    source = Path(path)
    if not source.is_file():
        return None
    try:
        with source.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(value, dict):
        return None
    if value.get("schema_name") != schema_name:
        return None
    if value.get("schema_version") != SUMMARY_VERSION:
        return None
    return value
