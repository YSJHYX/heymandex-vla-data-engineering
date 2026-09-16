from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.smoke_openpi_lerobot import validate_smoke_scope
from vla_data.cli import main
from vla_data.export import export_lerobot, validate_lerobot_export
from vla_data.export.plan import make_mainline_plan
from vla_data.manifest import SplitConfig
from vla_data.publish import validate_dataset_root
from vla_data.publish.publisher import _official_reload
from vla_data.quality.report import QUALITY_SCHEMA_NAME, QUALITY_SCHEMA_VERSION


def test_local_smoke_scope_is_exactly_003_to_006():
    rows = [
        {
            "source_episode_id": f"episode_{episode:06d}",
            "transition_count": count,
            "split": "train",
        }
        for episode, count in zip(range(3, 7), (343, 116, 286, 310), strict=True)
    ]
    summary = {"selected_transitions": 1055, "source_mode": "DIRECT_D2_D3"}
    assert validate_smoke_scope(summary, rows) == {
        f"episode_{episode:06d}" for episode in range(3, 7)
    }
    for broken in (
        [dict(rows[0], source_episode_id="episode_000002"), *rows[1:]],
        [*rows[:-1]],
        [dict(rows[0], split="val"), *rows[1:]],
        [dict(rows[0], transition_count=342), *rows[1:]],
    ):
        with pytest.raises(ValueError, match="unexpected source selection"):
            validate_smoke_scope(summary, broken)


def _tree(curated_v1_episode: Path, root: Path, count: int = 3):
    curated = root / "curated"
    quality = root / "quality"
    prompt = "  pick up 巧克力 pudding and place it in the basket  "
    for number in range(count):
        episode_id = f"episode_{number:06d}"
        destination = curated / episode_id
        shutil.copytree(curated_v1_episode, destination)
        metadata_path = destination / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata.update(
            episode_id=episode_id,
            language_instruction=prompt,
            source_schema_name="rm65_sg100_raw",
            source_dataset_status="REAL_DATA",
            expert_training_status="REVIEW_REQUIRED",
            segment_offsets=[0, 3],
        )
        metadata_path.write_text(json.dumps(metadata))
        trajectory_path = destination / "trajectory.npz"
        with np.load(trajectory_path, allow_pickle=False) as archive:
            trajectory = {key: np.asarray(archive[key]).copy() for key in archive.files}
        trajectory["source_tick_index"] = np.asarray([0, 1, 2], dtype=np.int64)
        trajectory["next_state_tick_index"] = np.asarray([1, 2, 3], dtype=np.int64)
        np.savez(trajectory_path, **trajectory)
        for role, colors in (
            ("head", [(220, 20, 20), (180, 60, 20)]),
            ("right_wrist", [(20, 30, 220), (20, 80, 180)]),
        ):
            for index, color in enumerate(colors):
                Image.new("RGB", (128, 96), color).save(
                    destination / "media" / role / "rgb" / f"{index:06d}.jpg"
                )

        quality_dir = quality / episode_id
        quality_dir.mkdir(parents=True)
        mask = np.asarray([True, False, True] if number == 0 else [True] * 3)
        np.save(quality_dir / "quality_mask.npy", mask)
        (quality_dir / "quality_report.json").write_text(
            json.dumps(
                {
                    "schema_name": QUALITY_SCHEMA_NAME,
                    "schema_version": QUALITY_SCHEMA_VERSION,
                    "episode_id": episode_id,
                    "status": "ACCEPT_WITH_WARNING" if not mask.all() else "ACCEPT",
                    "transition_count": 3,
                    "clean_transition_count": int(mask.sum()),
                }
            )
        )
    return curated, quality, prompt


def test_direct_d2_d3_runs_preserve_exact_prompt_and_boundaries(
    curated_v1_episode, tmp_path
):
    curated, quality, prompt = _tree(curated_v1_episode, tmp_path, count=1)
    plan = make_mainline_plan(curated, quality)
    runs = [run for run in plan["runs"] if run["source_episode_id"] == "episode_000000"]

    assert [(run["source_start_index"], run["source_end_index"]) for run in runs] == [
        (0, 1),
        (2, 3),
    ]
    assert all(run["task_instruction"] == prompt for run in runs)
    assert all(
        run["task_instruction_sha256"]
        == hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for run in runs
    )
    assert all(run["source_dataset_status"] == "REAL_DATA" for run in runs)
    assert all(run["expert_training_status"] == "REVIEW_REQUIRED" for run in runs)
    assert all(
        run["training_use_status"] == "INTEGRATION_ELIGIBLE_REVIEW_REQUIRED"
        for run in runs
    )
    assert plan["features"]["observation.state"]["shape"] == [17]
    assert plan["features"]["action"]["shape"] == [17]
    assert set(plan["camera_names"]) == {
        "observation.images.head",
        "observation.images.wrist",
    }
    assert all("final_instruction" not in run for run in runs)


def test_direct_export_does_not_read_optional_annotations(curated_v1_episode, tmp_path):
    curated, quality, _ = _tree(curated_v1_episode, tmp_path)
    before = make_mainline_plan(curated, quality)
    annotation = tmp_path / "annotation" / "episode_000000"
    annotation.mkdir(parents=True)
    (annotation / "annotation.json").write_text("not even valid JSON")
    verification = tmp_path / "verification" / "episode_000000"
    verification.mkdir(parents=True)
    (verification / "verification.json").write_text("{}")
    after = make_mainline_plan(curated, quality)

    assert after == before


def test_synthetic_and_missing_task_sources_are_excluded(curated_v1_episode, tmp_path):
    curated, quality, _ = _tree(curated_v1_episode, tmp_path)
    synthetic_path = curated / "episode_000001" / "metadata.json"
    synthetic = json.loads(synthetic_path.read_text())
    synthetic["source_dataset_status"] = "SYNTHETIC_TEST_ONLY"
    synthetic_path.write_text(json.dumps(synthetic))
    missing_path = curated / "episode_000002" / "metadata.json"
    missing = json.loads(missing_path.read_text())
    missing["language_instruction"] = ""
    missing_path.write_text(json.dumps(missing))

    plan = make_mainline_plan(curated, quality)
    assert {run["source_episode_id"] for run in plan["runs"]} == {"episode_000000"}
    sources = {source["episode_id"]: source for source in plan["sources"]}
    assert "SYNTHETIC_TEST_ONLY" in sources["episode_000001"]["reasons"]
    assert "TASK_INSTRUCTION_MISSING" in sources["episode_000002"]["reasons"]


def test_source_level_split_never_leaks_runs(curated_v1_episode, tmp_path):
    curated, quality, _ = _tree(curated_v1_episode, tmp_path, count=4)
    plan = make_mainline_plan(
        curated,
        quality,
        split_config=SplitConfig(validation_fraction=0.5, seed=19),
    )
    source_splits = {
        source["episode_id"]: source["split"] for source in plan["sources"]
    }
    assert {source["split"] for source in plan["sources"]} == {"train", "val"}
    assert all(
        run["split"] == source_splits[run["source_episode_id"]] for run in plan["runs"]
    )


def test_direct_cli_api_dry_run_needs_no_semantic_manifest(
    curated_v1_episode, tmp_path
):
    curated, quality, _ = _tree(curated_v1_episode, tmp_path)
    output = tmp_path / "export"
    result = export_lerobot(
        None,
        output,
        curated_root=curated,
        quality_root=quality,
        dry_run=True,
    )
    assert result["status"] == "DRY_RUN"
    assert result["source_mode"] == "DIRECT_D2_D3"
    assert result["export_runs"] == 4
    assert result["selected_transitions"] == 8
    assert not output.exists()


def test_direct_cli_dry_run_uses_only_curated_and_quality(
    curated_v1_episode, tmp_path, capsys
):
    curated, quality, _ = _tree(curated_v1_episode, tmp_path)
    output = tmp_path / "export"
    code = main(
        [
            "export-lerobot",
            "--curated-root",
            str(curated),
            "--quality-root",
            str(quality),
            "--output-root",
            str(output),
            "--dry-run",
        ]
    )

    assert code == 0
    assert '"source_mode": "DIRECT_D2_D3"' in capsys.readouterr().out
    assert not output.exists()


def test_direct_d2_d3_official_lerobot_roundtrip(curated_v1_episode, tmp_path):
    python = os.environ.get("VLA_LEROBOT_PYTHON")
    if not python:
        import pytest

        pytest.skip("set VLA_LEROBOT_PYTHON to the audited existing v2.1 environment")
    curated, quality, prompt = _tree(curated_v1_episode, tmp_path, count=1)
    output = tmp_path / "export"
    result = export_lerobot(
        None,
        output,
        curated_root=curated,
        quality_root=quality,
        lerobot_python=python,
    )
    check = validate_lerobot_export(output, lerobot_python=python)

    assert result["status"] == "BUILT"
    assert check.passed, check.errors
    assert check.evidence["total_frames"] == 2
    assert {run["task"] for run in check.evidence["runs"]} == {prompt}
    hub_provenance = output / "train" / "meta" / "source_provenance.jsonl"
    rows = [json.loads(line) for line in hub_provenance.read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["task_instruction"] == prompt for row in rows)
    assert all(row["expert_training_status"] == "REVIEW_REQUIRED" for row in rows)
    assert validate_dataset_root(output / "train")["episodes"] == 2

    remote_copy = tmp_path / "fresh_remote_copy"
    shutil.copytree(output / "train", remote_copy)
    reload_evidence = _official_reload(
        local_root=str(output / "train"),
        remote_root=str(remote_copy),
        lerobot_python=python,
        expected_tasks=[prompt],
    )
    assert reload_evidence["reload_pass"], reload_evidence["errors"]
    assert reload_evidence["episode_lengths"] == [1, 1]
    assert reload_evidence["source_ranges"] == [
        ["episode_000000", 0, 1],
        ["episode_000000", 2, 3],
    ]
    assert reload_evidence["camera_shapes"] == {
        "observation.images.head": [96, 128, 3],
        "observation.images.wrist": [96, 128, 3],
    }

    remote_provenance = remote_copy / "meta" / "source_provenance.jsonl"
    altered = [json.loads(line) for line in remote_provenance.read_text().splitlines()]
    altered[0]["task_instruction_sha256"] = "0" * 64
    remote_provenance.write_text("".join(json.dumps(row) + "\n" for row in altered))
    rejected = _official_reload(
        local_root=str(output / "train"),
        remote_root=str(remote_copy),
        lerobot_python=python,
        expected_tasks=[prompt],
    )
    assert not rejected["reload_pass"]
    assert "source provenance differs" in rejected["errors"][0]
