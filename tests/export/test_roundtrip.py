"""Opt-in integration against an existing pinned LeRobot interpreter."""

import hashlib
import json
import os

import pytest

from vla_data.export import export_lerobot, validate_lerobot_export
from vla_data.export.plan import make_plan
from vla_data.export.runtime import worker_call


def test_official_writer_roundtrip_horizons_resume_and_atomic_failure(
    export_tree, monkeypatch
):
    python = os.environ.get("VLA_LEROBOT_PYTHON")
    if not python:
        pytest.skip("set VLA_LEROBOT_PYTHON to the audited existing v2.1 environment")
    sources = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in export_tree.curated.rglob("*.jpg")
    }
    result = export_lerobot(
        export_tree.manifest, export_tree.output, lerobot_python=python
    )
    assert result["status"] == "BUILT"
    check = validate_lerobot_export(export_tree.output, lerobot_python=python)
    assert check.passed, check.errors
    assert check.evidence["total_frames"] == 8
    assert len(check.evidence["runs"]) == 5
    assert (
        export_lerobot(export_tree.manifest, export_tree.output, lerobot_python=python)[
            "status"
        ]
        == "SKIPPED"
    )
    horizon = worker_call(
        "openpi",
        {
            "plan": make_plan(export_tree.manifest),
            "output_root": str(export_tree.output),
        },
        python,
    )
    assert all(r["boundary_safe"] for r in horizon["runs"])
    assert min(r["length"] for r in horizon["runs"]) == 1
    before = {
        p.relative_to(export_tree.output): p.read_bytes()
        for p in export_tree.output.rglob("*")
        if p.is_file()
    }
    from vla_data.export import runner

    original = runner.worker_call

    def failing(mode, *args, **kwargs):
        if mode == "write":
            raise ValueError("injected writer failure")
        return original(mode, *args, **kwargs)

    monkeypatch.setattr(runner, "worker_call", failing)
    with pytest.raises(ValueError, match="injected"):
        export_lerobot(
            export_tree.manifest, export_tree.output, lerobot_python=python, force=True
        )
    assert {
        p.relative_to(export_tree.output): p.read_bytes()
        for p in export_tree.output.rglob("*")
        if p.is_file()
    } == before
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources} == sources
    monkeypatch.setattr(runner, "worker_call", original)
    # Swapping two actual encoded camera files must fail independent reload.
    run = make_plan(export_tree.manifest)["runs"][0]
    video_root = export_tree.output / run["split"] / "videos/chunk-000"
    filename = f"episode_{run['lerobot_episode_index']:06d}.mp4"
    head = video_root / "observation.images.head" / filename
    wrist = video_root / "observation.images.wrist" / filename
    head_bytes, wrist_bytes = head.read_bytes(), wrist.read_bytes()
    head.write_bytes(wrist_bytes)
    wrist.write_bytes(head_bytes)
    assert not validate_lerobot_export(export_tree.output, lerobot_python=python).passed
    head.write_bytes(head_bytes)
    wrist.write_bytes(wrist_bytes)
    # Manifest instruction and D2 boundary changes invalidate, then rebuild.
    from vla_data.verification import review_episode

    review_episode(
        export_tree.verification,
        "episode_000000",
        status="HUMAN_CORRECTED",
        instruction="Exact revised synthetic task.",
    )
    metadata_path = export_tree.curated / "episode_000000/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["segment_offsets"] = [0, 1, 3]
    metadata_path.write_text(json.dumps(metadata))
    export_tree.refresh()
    assert not validate_lerobot_export(export_tree.output, lerobot_python=python).passed
    rebuilt = export_lerobot(
        export_tree.manifest, export_tree.output, lerobot_python=python
    )
    assert rebuilt["status"] == "BUILT" and rebuilt["export_runs"] == 6
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources} == sources
    # Persist explicit integration evidence alongside pytest temp artifacts.
    (export_tree.output.parent / "openpi_smoke.json").write_text(
        json.dumps(horizon, indent=2)
    )
