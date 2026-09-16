"""D4.3 baseline comparison stays descriptive and source-read-only."""

from __future__ import annotations

import copy
import hashlib
import json

from vla_data.annotation.comparison import write_semantic_comparison
from vla_data.annotation.schema import write_annotation


def _annotation(episode: str, *, positive: bool) -> dict:
    return {
        "schema_name": "vla_episode_annotation",
        "schema_version": 2,
        "annotation_schema_version": 2,
        "episode_id": episode,
        "quality_outcome": "ACCEPT_WITH_WARNING",
        "episode_task": {"instruction": "Grasp the connector", "confidence": 0.5},
        "semantic_segments": (
            [
                {
                    "segment_id": "grasp",
                    "start_curated_index": 4,
                    "end_curated_index": 10,
                    "instruction": "Grasp the connector",
                    "confidence": 0.5,
                }
            ]
            if positive
            else []
        ),
        "non_training_intervals": (
            [
                {
                    "start_curated_index": 0,
                    "end_curated_index": 4,
                    "reason": "PRE_TASK_IDLE",
                }
            ]
            if positive
            else [
                {
                    "start_curated_index": 0,
                    "end_curated_index": 10,
                    "reason": "AMBIGUOUS_VISUAL_EVIDENCE",
                }
            ]
        ),
        "clean_domains": [
            {
                "d2_segment_id": 0,
                "clean_run_id": 0,
                "start_curated_index": 0,
                "end_curated_index": 10,
            }
        ],
        "model_provenance": {
            "provider": "mock",
            "model": "mock",
            "pass_a_prompt_version": "a",
            "pass_b_prompt_version": "b",
        },
    }


def test_json_csv_comparison_and_baseline_immutability(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    for root in (baseline, current):
        (root / "episode_000004").mkdir(parents=True)
    baseline_file = baseline / "episode_000004/annotation.json"
    write_annotation(baseline_file, _annotation("episode_000004", positive=False))
    write_annotation(
        current / "episode_000004/annotation.json",
        copy.deepcopy(_annotation("episode_000004", positive=True)),
    )
    before = hashlib.sha256(baseline_file.read_bytes()).hexdigest()
    json_path, csv_path = write_semantic_comparison(baseline, current)
    after = hashlib.sha256(baseline_file.read_bytes()).hexdigest()
    assert before == after
    document = json.loads(json_path.read_text())
    row = document["episodes"][0]
    assert row["baseline_segment_count"] == 0
    assert row["new_segment_count"] == 1
    assert row["baseline_positive_frames"] == 0
    assert row["new_positive_frames"] == 6
    assert row["baseline_ambiguous_frames"] == 10
    assert row["new_ambiguous_frames"] == 0
    assert "not automatically better" in document["interpretation"]
    assert csv_path.is_file()


def test_comparison_reports_missing_baseline_without_semantic_zero(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    baseline.mkdir()
    (current / "episode_000004").mkdir(parents=True)
    write_annotation(
        current / "episode_000004/annotation.json",
        _annotation("episode_000004", positive=True),
    )
    json_path, _ = write_semantic_comparison(baseline, current)
    row = json.loads(json_path.read_text())["episodes"][0]
    assert row["baseline_status"] == "MISSING"
    assert row["baseline_positive_frames"] is None
    assert row["new_status"] == "SUCCESS"


def test_provider_failure_has_null_semantic_metrics(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    (baseline / "episode_000004").mkdir(parents=True)
    current.mkdir()
    write_annotation(
        baseline / "episode_000004/annotation.json",
        _annotation("episode_000004", positive=False),
    )
    json_path, _ = write_semantic_comparison(
        baseline,
        current,
        results={
            "episode_000004": {
                "episode_id": "episode_000004",
                "status": "FAILED",
                "annotation_status": "PROVIDER_FAILED",
                "failure_reason": "MCP_TIMEOUT",
                "semantic_evaluated": False,
            }
        },
    )
    row = json.loads(json_path.read_text())["episodes"][0]
    assert row["new_status"] == "PROVIDER_FAILED"
    assert row["new_failure_reason"] == "MCP_TIMEOUT"
    assert row["new_semantic_evaluated"] is False
    assert row["new_segment_count"] is None
    assert row["new_positive_frames"] is None
