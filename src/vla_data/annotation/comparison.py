"""Read-only semantic comparison with explicit non-semantic failure states."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from vla_data.annotation.schema import load_annotation

SUCCESS = "SUCCESS"
PROVIDER_FAILED = "PROVIDER_FAILED"
VALIDATION_FAILED = "VALIDATION_FAILED"
MISSING = "MISSING"


def _legacy_result_status(result: dict[str, Any]) -> str:
    explicit = result.get("annotation_status")
    if explicit in {SUCCESS, PROVIDER_FAILED, VALIDATION_FAILED, MISSING}:
        return str(explicit)
    if result.get("status") in {"SUCCESS", "SKIPPED", "WOULD_SKIP"}:
        return SUCCESS
    message = str(result.get("message", ""))
    if message.startswith(("MCP_", "NETWORK_ERROR")):
        return PROVIDER_FAILED
    if result.get("status") == "FAILED":
        return VALIDATION_FAILED
    return MISSING


def _summary_results(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "dataset_annotation_summary.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    values = document.get("results", []) if isinstance(document, dict) else []
    return {
        str(item["episode_id"]): item
        for item in values
        if isinstance(item, dict) and isinstance(item.get("episode_id"), str)
    }


def _failure_record(root: Path, episode_id: str) -> dict[str, Any] | None:
    path = root / episode_id / "annotation_result.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def _side(
    root: Path,
    episode_id: str,
    *,
    explicit_result: dict[str, Any] | None,
    summary_result: dict[str, Any] | None,
) -> dict[str, Any]:
    annotation_path = root / episode_id / "annotation.json"
    failure = _failure_record(root, episode_id)
    result = explicit_result
    if result is None and failure is not None:
        nested = failure.get("result")
        result = nested if isinstance(nested, dict) else failure
    if result is None and not annotation_path.is_file():
        result = summary_result
    status = _legacy_result_status(result or {})
    if result is None and annotation_path.is_file():
        status = SUCCESS
    if status == SUCCESS and not annotation_path.is_file():
        status = MISSING
    side: dict[str, Any] = {
        "status": status,
        "failure_reason": (result or {}).get("failure_reason"),
        "semantic_evaluated": status == SUCCESS,
        "task": None,
        "segment_count": None,
        "positive_frames": None,
        "ambiguous_frames": None,
        "nontraining_frames": None,
    }
    if status != SUCCESS:
        if side["failure_reason"] is None and result is not None:
            message = str(result.get("message", ""))
            side["failure_reason"] = message.split(":", 1)[0] or None
        return side
    annotation = load_annotation(annotation_path)
    side.update(
        {
            "task": annotation["episode_task"]["instruction"],
            "segment_count": len(annotation["semantic_segments"]),
            "positive_frames": _covered_frames(annotation["semantic_segments"]),
            "ambiguous_frames": _reason_frames(
                annotation["non_training_intervals"],
                "AMBIGUOUS_VISUAL_EVIDENCE",
            ),
            "nontraining_frames": _covered_frames(annotation["non_training_intervals"]),
        }
    )
    return side


def compare_annotation_roots(
    baseline_root: str | Path,
    new_root: str | Path,
    *,
    results: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Compare success metrics while keeping failures semantically unevaluated."""

    baseline_path = Path(baseline_root).resolve()
    new_path = Path(new_root).resolve()
    current_results = results or {}
    baseline_summary = _summary_results(baseline_path)
    new_summary = _summary_results(new_path)
    episode_ids = {
        path.parent.name for path in new_path.glob("episode_*/annotation.json")
    }
    episode_ids.update(
        path.parent.name for path in new_path.glob("episode_*/annotation_result.json")
    )
    episode_ids.update(current_results)
    rows = []
    for episode_id in sorted(episode_ids):
        baseline = _side(
            baseline_path,
            episode_id,
            explicit_result=None,
            summary_result=baseline_summary.get(episode_id),
        )
        current = _side(
            new_path,
            episode_id,
            explicit_result=current_results.get(episode_id),
            summary_result=new_summary.get(episode_id),
        )
        rows.append(
            {
                "episode": episode_id,
                "baseline_status": baseline["status"],
                "new_status": current["status"],
                "new_failure_reason": current["failure_reason"],
                "new_semantic_evaluated": current["semantic_evaluated"],
                "baseline_task": baseline["task"],
                "new_task": current["task"],
                "baseline_segment_count": baseline["segment_count"],
                "new_segment_count": current["segment_count"],
                "baseline_positive_frames": baseline["positive_frames"],
                "new_positive_frames": current["positive_frames"],
                "baseline_ambiguous_frames": baseline["ambiguous_frames"],
                "new_ambiguous_frames": current["ambiguous_frames"],
                "baseline_nontraining_frames": baseline["nontraining_frames"],
                "new_nontraining_frames": current["nontraining_frames"],
            }
        )
    return rows


def write_semantic_comparison(
    baseline_root: str | Path,
    new_root: str | Path,
    *,
    results: dict[str, dict[str, Any]] | None = None,
) -> tuple[Path, Path]:
    """Write deterministic JSON/CSV without interpreting failure as zero recall."""

    root = Path(new_root).resolve()
    rows = compare_annotation_roots(baseline_root, root, results=results)
    document = {
        "schema_name": "vla_semantic_annotation_comparison",
        "schema_version": 2,
        "baseline_root": str(Path(baseline_root).resolve()),
        "new_root": str(root),
        "interpretation": (
            "Descriptive comparison only; more segments or positive frames are "
            "not automatically better. Provider/validation failures are not "
            "semantic zeroes. Visual evidence and D5 review remain authoritative."
        ),
        "episodes": rows,
    }
    json_path = root / "semantic_comparison.json"
    csv_path = root / "semantic_comparison.csv"
    json_path.write_text(
        json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    fields = list(rows[0]) if rows else _comparison_fields()
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def _covered_frames(blocks: list[dict[str, Any]]) -> int:
    return sum(
        block["end_curated_index"] - block["start_curated_index"] for block in blocks
    )


def _reason_frames(blocks: list[dict[str, Any]], reason: str) -> int:
    return _covered_frames([block for block in blocks if block["reason"] == reason])


def _comparison_fields() -> list[str]:
    return [
        "episode",
        "baseline_status",
        "new_status",
        "new_failure_reason",
        "new_semantic_evaluated",
        "baseline_task",
        "new_task",
        "baseline_segment_count",
        "new_segment_count",
        "baseline_positive_frames",
        "new_positive_frames",
        "baseline_ambiguous_frames",
        "new_ambiguous_frames",
        "baseline_nontraining_frames",
        "new_nontraining_frames",
    ]
