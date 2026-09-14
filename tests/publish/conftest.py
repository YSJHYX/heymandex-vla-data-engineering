"""Minimal canonical LeRobot-style dataset root for publication tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_publication_staging(tmp_path, monkeypatch):
    """Mock publication tests must not depend on writable production /data."""
    monkeypatch.setattr(
        "vla_data.publish.publisher._staging_parent", lambda: str(tmp_path)
    )


INFO = {
    "codebase_version": "v2.1",
    "fps": 30,
    "features": {
        "observation.state": {"dtype": "float32", "shape": [17], "names": ["state"]},
        "action": {"dtype": "float32", "shape": [17], "names": ["action"]},
        "observation.images.head": {"dtype": "video", "shape": [480, 640, 3]},
        "observation.images.wrist": {"dtype": "video", "shape": [480, 640, 3]},
    },
}


@pytest.fixture
def hf_dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(INFO))
    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "Place the cable on the table."}) + "\n"
    )
    (root / "meta" / "episodes.jsonl").write_text("{}\n")
    data = root / "data" / "chunk-000"
    data.mkdir(parents=True)
    (data / "episode_000000.parquet").write_bytes(b"fake-parquet-bytes-0")
    for camera in ("head", "wrist"):
        video = root / "videos" / "chunk-000" / f"observation.images.{camera}"
        video.mkdir(parents=True)
        (video / "episode_000000.mp4").write_bytes(
            b"fake-mp4-" + camera.encode() + b"-bytes"
        )
    return root


def manifest_entries(root: Path) -> list[dict]:
    paths = sorted([*root.rglob("*")], key=lambda p: p.relative_to(root).as_posix())
    return [
        {
            "relative_path": p.relative_to(root).as_posix(),
            "size_bytes": len(p.read_bytes()),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in paths
        if p.is_file()
    ]
