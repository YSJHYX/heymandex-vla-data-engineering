"""Minimal canonical LeRobot-style dataset root for publication tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

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
    raw = tmp_path / "captured_raw.npz"
    raw.write_bytes(b"immutable-captured-episode")
    curated = tmp_path / "curated" / "episode_000000"
    curated.mkdir(parents=True)
    (curated / "metadata.json").write_text(
        json.dumps(
            {
                "source_episode_path": str(raw),
                "language_instruction": "Place the cable on the table.",
            }
        )
    )
    (curated / "cleaning_report.json").write_text(
        json.dumps({"source_provenance": {"dataset_status": "RAW_CAPTURE_QUARANTINED"}})
    )
    (tmp_path / "export_summary.json").write_text(
        json.dumps({"source_mode": "DIRECT_D2_D3", "fingerprint": "a" * 64})
    )
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(INFO))
    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "Place the cable on the table."}) + "\n"
    )
    (root / "meta" / "episodes.jsonl").write_text("{}\n")
    task = "Place the cable on the table."
    (root / "meta" / "source_provenance.jsonl").write_text(
        json.dumps(
            {
                "source_episode_id": "episode_000000",
                "curated_path": str(curated),
                "source_start_index": 0,
                "source_end_index": 3,
                "transition_count": 3,
                "task_instruction": task,
                "task_instruction_sha256": hashlib.sha256(task.encode()).hexdigest(),
                "source_dataset_status": "REAL_DATA",
                "expert_training_status": "REVIEW_REQUIRED",
            }
        )
        + "\n"
    )
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
