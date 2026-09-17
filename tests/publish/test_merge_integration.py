"""Official LeRobot writer regression for overlapping standalone chunk paths."""

from __future__ import annotations

import json
import os

import pytest

from tests.export.test_mainline import _tree
from vla_data.export import export_lerobot
from vla_data.publish import validate_dataset_root
from vla_data.publish.merge import plan_logical_merge, rebuild_logical_dataset
from vla_data.publish.publisher import _official_reload


def test_official_logical_rebuild_preserves_chunk_zero_episodes(
    curated_v1_episode, tmp_path
) -> None:
    interpreter = os.environ.get("VLA_LEROBOT_PYTHON")
    if not interpreter:
        pytest.skip("set VLA_LEROBOT_PYTHON for official v2.1 merge integration")
    old_curated, old_quality, _ = _tree(
        curated_v1_episode, tmp_path / "old_source", count=3
    )
    new_curated, new_quality, _ = _tree(
        curated_v1_episode, tmp_path / "new_source", count=1
    )
    # These are test-only RAW identity files, distinct from production fixtures.
    for label, curated in (("old", old_curated), ("new", new_curated)):
        for episode in sorted(curated.iterdir()):
            raw = episode / "test_raw_identity.npz"
            raw.write_bytes(f"{label}:{episode.name}".encode())
            metadata_path = episode / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["source_episode_path"] = str(raw)
            metadata_path.write_text(json.dumps(metadata))
    baseline_export = tmp_path / "baseline_export"
    incoming_export = tmp_path / "incoming_export"
    for output, curated, quality in (
        (baseline_export, old_curated, old_quality),
        (incoming_export, new_curated, new_quality),
    ):
        assert (
            export_lerobot(
                None,
                output,
                curated_root=curated,
                quality_root=quality,
                lerobot_python=interpreter,
            )["status"]
            == "BUILT"
        )
    baseline = baseline_export / "train"
    incoming = incoming_export / "train"
    assert (baseline / "data/chunk-000/episode_000000.parquet").is_file()
    assert (incoming / "data/chunk-000/episode_000000.parquet").is_file()
    old_rows = [
        json.loads(line)
        for line in (baseline / "meta/source_provenance.jsonl").read_text().splitlines()
    ]
    new_rows = [
        json.loads(line)
        for line in (incoming / "meta/source_provenance.jsonl").read_text().splitlines()
    ]
    assert len(old_rows) == 4
    assert len(new_rows) == 2
    plan = plan_logical_merge(
        incoming,
        baseline,
        baseline_revision="a" * 40,
        source_export_fingerprint="b" * 64,
    )
    assert (plan["baseline_episodes"], plan["to_append"], plan["merged_episodes"]) == (
        4,
        2,
        6,
    )
    assert plan["already_present"] == 0
    assert len({row["source_run_key"] for row in plan["provenance"]}) == 6
    rebuilt = tmp_path / "rebuilt"
    evidence = rebuild_logical_dataset(
        incoming, baseline, plan, rebuilt, lerobot_python=interpreter
    )
    assert evidence["official_reload"] == "PASS"
    assert evidence["episode_lengths"] == [1, 1, 3, 3, 1, 1]
    assert evidence["frames"] == 10
    assert validate_dataset_root(rebuilt)["episodes"] == 6
    reload = _official_reload(
        local_root=str(rebuilt),
        remote_root=str(rebuilt),
        lerobot_python=interpreter,
        expected_tasks=[old_rows[0]["task_instruction"]],
    )
    assert reload["reload_pass"], reload["errors"]
    assert reload["episode_lengths"] == [1, 1, 3, 3, 1, 1]
