"""Eight-field JSONL alongside the unchanged standard LeRobot export."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from tests.export.test_mainline import _tree
from vla_data.export import export_lerobot, validate_lerobot_export
from vla_data.export.json_frames import export_frame_json


def test_frame_json_output_is_separate_and_dry_run_writes_nothing(
    curated_v1_episode, tmp_path
):
    curated, quality, _ = _tree(curated_v1_episode, tmp_path, count=1)
    native, frames = tmp_path / "lerobot", tmp_path / "frame-json"
    result = export_lerobot(
        None,
        native,
        curated_root=curated,
        quality_root=quality,
        frame_json_root=frames,
        dry_run=True,
    )
    assert result["status"] == "DRY_RUN"
    assert not native.exists() and not frames.exists()
    for invalid in (native, native / "frames", curated / "frames"):
        with pytest.raises(ValueError, match="separate"):
            export_lerobot(
                None,
                native,
                curated_root=curated,
                quality_root=quality,
                frame_json_root=invalid,
                dry_run=True,
            )
    frames.mkdir()
    with pytest.raises(ValueError, match="new directory"):
        export_lerobot(
            None,
            native,
            curated_root=curated,
            quality_root=quality,
            frame_json_root=frames,
            dry_run=True,
        )


def test_json_failure_keeps_input_and_no_partial_output(tmp_path, monkeypatch):
    source, target = tmp_path / "lerobot", tmp_path / "frame-json"
    (source / "train/meta").mkdir(parents=True)
    (source / "train/meta/info.json").write_text("{}")

    def fail(mode, request, python):
        assert mode == "export_frame_json"
        destination = Path(request["output_root"])
        destination.mkdir()
        (destination / "partial").write_text("partial")
        raise ValueError("injected JSON failure")

    monkeypatch.setattr("vla_data.export.runtime.worker_call", fail)
    with pytest.raises(ValueError, match="injected"):
        export_frame_json(source, target, lerobot_python="unused")
    assert not target.exists()
    assert (source / "train/meta/info.json").read_text() == "{}"


def test_one_command_lerobot_and_eight_field_json_roundtrip(
    curated_v1_episode, tmp_path
):
    python = os.environ.get("VLA_LEROBOT_PYTHON")
    if not python:
        pytest.skip("set VLA_LEROBOT_PYTHON for official v2.1 integration")
    curated, quality, _ = _tree(curated_v1_episode, tmp_path, count=4)
    native, frames = tmp_path / "lerobot", tmp_path / "frame-json"
    before = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in curated.rglob("*")
        if p.is_file()
    }
    result = export_lerobot(
        None,
        native,
        curated_root=curated,
        quality_root=quality,
        validation_fraction=0.5,
        split_seed=19,
        frame_json_root=frames,
        lerobot_python=python,
    )
    assert result["status"] == "BUILT"
    assert validate_lerobot_export(native, lerobot_python=python).passed
    assert set(result["frame_json"]["splits"]) == {"train", "val"}
    assert not list(native.rglob("extras"))
    fields = {
        "t",
        "image_top",
        "image_hand",
        "depth_top",
        "depth_hand",
        "joint_pos",
        "action",
        "success",
    }
    for split in ("train", "val"):
        lengths = [
            json.loads(line)["length"]
            for line in (native / split / "meta/episodes.jsonl")
            .read_text()
            .splitlines()
        ]
        for episode, count in enumerate(lengths):
            directory = frames / split / f"episode_{episode:06d}"
            rows = [
                json.loads(line)
                for line in (directory / "frames.jsonl").read_text().splitlines()
            ]
            assert len(rows) == count and rows[0]["t"] == 0
            assert [row["success"] for row in rows] == [0] * (count - 1) + [1]
            assert all(type(row["success"]) is int for row in rows)
            parquet = next(
                (native / split / "data").rglob(f"episode_{episode:06d}.parquet")
            )
            expected = json.loads(
                subprocess.check_output(
                    [
                        python,
                        "-c",
                        "import json, sys, pyarrow.parquet as pq; print(json.dumps(pq.read_table(sys.argv[1]).to_pylist()))",
                        str(parquet),
                    ],
                    text=True,
                )
            )
            for row, original in zip(rows, expected, strict=True):
                assert set(row) == fields
                assert len(row["joint_pos"]) == len(row["action"]) == 17
                assert row["joint_pos"] == original["observation.state"]
                assert row["action"] == original["action"]
                assert row["t"] == original["timestamp"]
                assert row["depth_top"] is None and row["depth_hand"] is None
                for camera in ("image_top", "image_hand"):
                    with Image.open(directory / row[camera]) as image:
                        assert image.size == (128, 96)
    assert all(
        hashlib.sha256(p.read_bytes()).hexdigest() == sha for p, sha in before.items()
    )
    # Reusing an already validated LeRobot output can generate a fresh JSON directory.
    second = tmp_path / "frame-json-again"
    repeated = export_lerobot(
        None,
        native,
        curated_root=curated,
        quality_root=quality,
        validation_fraction=0.5,
        split_seed=19,
        frame_json_root=second,
        lerobot_python=python,
    )
    assert repeated["status"] == "SKIPPED"
    assert {
        p.relative_to(frames): p.read_bytes() for p in frames.rglob("frames.jsonl")
    } == {p.relative_to(second): p.read_bytes() for p in second.rglob("frames.jsonl")}
