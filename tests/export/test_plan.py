import json
from pathlib import Path

import numpy as np
import pytest

from vla_data.export import export_lerobot
from vla_data.export.plan import contiguous_runs, make_plan, storage_vector
from vla_data.verification import VerificationPolicy, review_episode, verify_annotations


def test_segments_and_gaps_are_absolute_boundaries(export_tree):
    plan = make_plan(export_tree.manifest)
    assert [r["transition_count"] for r in plan["runs"]] == [3, 2, 1, 1, 1]
    assert sum(r["transition_count"] for r in plan["runs"]) == 8
    sources = {s["episode_id"]: s["split"] for s in plan["sources"]}
    for run in plan["runs"]:
        assert run["split"] == sources[run["source_episode_id"]]
        assert run["final_instruction"] != "test"
    assert {r["split"] for r in plan["runs"]} == {"train", "val"}
    assert plan["features"]["observation.state"]["shape"] == [17]
    assert plan["features"]["action"]["dtype"] == "float32"
    assert list(plan["runs"][0]["images"]) == [
        "observation.images.head",
        "observation.images.wrist",
    ]
    assert "right_wrist" in plan["runs"][0]["images"]["observation.images.wrist"][0]
    assert make_plan(export_tree.manifest)["fingerprint"] == plan["fingerprint"]


def test_run_holes_do_not_concatenate():
    assert contiguous_runs((0, 8), np.array([1, 1, 1, 0, 1, 1, 1, 1], dtype=bool)) == [
        (0, 0, 0, 3),
        (0, 1, 4, 8),
    ]
    assert contiguous_runs((0, 3, 8), np.ones(8, dtype=bool)) == [
        (0, 0, 0, 3),
        (1, 0, 3, 8),
    ]


@pytest.mark.parametrize("width", [16, 32])
def test_wrong_vector_width_rejected(width):
    with pytest.raises(ValueError):
        storage_vector(np.ones((2, width)))


@pytest.mark.parametrize("bad", [np.nan, np.inf, 1e100])
def test_nonfinite_or_float32_overflow_rejected(bad):
    with pytest.raises(ValueError), np.errstate(over="ignore"):
        storage_vector(np.full((1, 17), bad))


def test_float32_cast_only():
    values = np.arange(51).reshape(3, 17) / 13
    converted, error = storage_vector(values)
    assert converted.shape == values.shape and converted.dtype == np.float32
    assert np.array_equal(converted, values.astype(np.float32))
    assert error == np.max(np.abs(values - converted.astype(np.float64)))


@pytest.mark.parametrize("status", ["HUMAN_VERIFIED", "HUMAN_CORRECTED", "REJECTED"])
def test_verification_authority_propagates(export_tree, status):
    correction = (
        "  Synthetic HUMAN text. 保留原文  " if status == "HUMAN_CORRECTED" else None
    )
    review_episode(
        export_tree.verification,
        "episode_000000",
        status=status,
        instruction=correction,
    )
    export_tree.refresh()
    plan = make_plan(export_tree.manifest)
    runs = [r for r in plan["runs"] if r["source_episode_id"] == "episode_000000"]
    if status == "REJECTED":
        assert not runs
    else:
        assert runs[0]["final_instruction"] == (
            correction or "Move the SYNTHETIC block 0."
        )


def test_pending_and_empty_dataset_export(export_tree):
    verify_annotations(
        export_tree.annotation,
        export_tree.quality,
        export_tree.verification,
        policy=VerificationPolicy(1.0),
    )
    export_tree.refresh()
    assert make_plan(export_tree.manifest)["runs"] == []


def test_expert_status_does_not_exclude_optional_export(export_tree):
    path = export_tree.curated / "episode_000000/metadata.json"
    m = json.loads(path.read_text())
    m["expert_training_status"] = "EXCLUDE_FROM_EXPERT_TRAINING"
    path.write_text(json.dumps(m))
    export_tree.refresh()
    assert any(
        r["source_episode_id"] == "episode_000000"
        for r in make_plan(export_tree.manifest)["runs"]
    )


def test_dry_run_writes_nothing(export_tree):
    result = export_lerobot(export_tree.manifest, export_tree.output, dry_run=True)
    assert result["export_runs"] == 5 and result["selected_transitions"] == 8
    assert not export_tree.output.exists()


@pytest.mark.parametrize("fps", [None, 0, True, 29.5])
def test_no_fps_fallback(export_tree, fps):
    path = export_tree.curated / "episode_000000/metadata.json"
    m = json.loads(path.read_text())
    m["dataset_hz"] = fps
    path.write_text(json.dumps(m))
    export_tree.refresh()
    with pytest.raises(ValueError):
        make_plan(export_tree.manifest)


def test_manifest_instruction_and_boundary_change_invalidate(export_tree):
    first = make_plan(export_tree.manifest)["fingerprint"]
    review_episode(
        export_tree.verification,
        "episode_000000",
        status="HUMAN_CORRECTED",
        instruction="New synthetic text",
    )
    export_tree.refresh()
    second = make_plan(export_tree.manifest)["fingerprint"]
    assert first != second
    path = export_tree.curated / "episode_000000/metadata.json"
    m = json.loads(path.read_text())
    m["segment_offsets"] = [0, 1, 3]
    path.write_text(json.dumps(m))
    export_tree.refresh()
    assert make_plan(export_tree.manifest)["fingerprint"] != second


def test_missing_optional_dependency_actionable(export_tree):
    with pytest.raises(ValueError, match="--lerobot-python"):
        export_lerobot(export_tree.manifest, export_tree.output)
    assert not export_tree.output.exists()


def test_runtime_backend_change_forces_rebuild(export_tree, monkeypatch, tmp_path):
    """A changed video runtime (backend/version) must REBUILD, not SKIP."""

    from vla_data.export import runner, validator

    plan = make_plan(export_tree.manifest)
    output = tmp_path / "export"
    output.mkdir()

    def runtime_with(backend: str, version: str) -> dict:
        return {
            "codebase_version": "v2.1",
            "video_backend": backend,
            "video_dependency_versions": {"av": version},
        }

    (output / "export_summary.json").write_text(
        json.dumps(
            {
                "schema_name": "vla_lerobot_export",
                "schema_version": 1,
                "fingerprint": plan["fingerprint"],
                "manifest_root": str(Path(export_tree.manifest).resolve()),
                "dataset_name": "vla-local",
                "runtime": runtime_with("torchcodec", "0.1.0"),
            }
        )
    )
    (output / "export_provenance.jsonl").write_text(
        "".join(
            json.dumps({k: v for k, v in run.items() if k != "images"}) + "\n"
            for run in plan["runs"]
        )
    )
    for split in ("train", "val"):
        rows = [run for run in plan["runs"] if run["split"] == split]
        if not rows:
            continue
        meta = output / split / "meta"
        meta.mkdir(parents=True)
        (meta / "source_provenance.jsonl").write_text(
            "".join(
                json.dumps({k: v for k, v in run.items() if k != "images"}) + "\n"
                for run in rows
            )
        )
    calls = []

    def fake_worker(mode, *args, **kwargs):
        calls.append(mode)
        if mode == "probe":
            return runtime_with("torchcodec", "0.2.0")  # dependency version moved
        if mode == "validate":
            return {"official_reload": "PASS", "runs": [], "total_frames": 0}
        if mode == "write":
            raise ValueError("rebuild-attempted")
        raise AssertionError(f"unexpected worker mode {mode}")

    monkeypatch.setattr(runner, "worker_call", fake_worker)
    monkeypatch.setattr(validator, "worker_call", fake_worker)

    with pytest.raises(ValueError, match="rebuild-attempted"):
        export_lerobot(export_tree.manifest, output)
    assert "write" in calls  # runtime mismatch chose the rebuild path

    calls.clear()
    monkeypatch.setattr(
        runner,
        "worker_call",
        lambda mode, *a, **k: (
            runtime_with("torchcodec", "0.1.0")
            if mode == "probe"
            else fake_worker(mode, *a, **k)
        ),
    )
    result = export_lerobot(export_tree.manifest, output)
    assert result["status"] == "SKIPPED"
    assert "write" not in calls  # matching runtime performs no re-encode
