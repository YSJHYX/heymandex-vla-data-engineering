"""Official LeRobot writer regression for overlapping standalone chunk paths."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile

import numpy as np
import pytest
from PIL import Image

from tests.export.test_mainline import _tree
from vla_data.export import export_lerobot
from vla_data.export.camera_transform import camera_export_transform_contract
from vla_data.publish import validate_dataset_root
from vla_data.publish.merge import plan_logical_merge, rebuild_logical_dataset
from vla_data.publish.publisher import _official_reload

# The shared curated fixture uses flat frames whose 180-degree rotation is
# pixel-identical, so migration regressions replace them with asymmetric
# gradients before exporting legacy baselines.
_DECODE_ANCHORS_SCRIPT = """
import json, sys
import av
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

request = json.load(sys.stdin)

def anchors(spec):
    dataset = LeRobotDataset("local/decode", root=spec["root"])
    path = dataset.root / dataset.meta.get_video_file_path(
        spec["episode_index"], spec["key"]
    )
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    picks = sorted({0, len(frames) // 2, len(frames) - 1})
    return {"length": len(frames), "pixels": [frames[i].tolist() for i in picks]}

print(json.dumps([anchors(spec) for spec in request["specs"]]))
"""


def _asymmetric_frame(offset: int) -> np.ndarray:
    gradient = np.add.outer(np.arange(96), np.arange(128))
    image = np.zeros((96, 128, 3), dtype=np.uint8)
    image[..., 0] = gradient
    image[..., 1] = 255 - gradient
    image[..., 2] = (gradient + offset) % 256
    return image


def _mae(left, right) -> float:
    return float(
        np.mean(np.abs(np.asarray(left, np.float32) - np.asarray(right, np.float32)))
    )


def _decode_anchors(interpreter: str, specs) -> list[dict]:
    request = json.dumps(
        {
            "specs": [
                {"root": str(root), "episode_index": index, "key": key}
                for root, index, key in specs
            ]
        }
    )
    with tempfile.TemporaryDirectory(prefix="vla_decode_cache_") as cache:
        environment = dict(os.environ)
        environment["HF_DATASETS_CACHE"] = cache
        completed = subprocess.run(
            [interpreter, "-c", _DECODE_ANCHORS_SCRIPT],
            input=request,
            capture_output=True,
            text=True,
            check=True,
            timeout=600,
            env=environment,
        )
    lines = [line for line in completed.stdout.strip().splitlines() if line.strip()]
    return json.loads(lines[-1])


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
    assert plan["requires_camera_transform_migration"] is False
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


def test_legacy_multi_episode_baseline_migrates_once(
    curated_v1_episode, tmp_path
) -> None:
    interpreter = os.environ.get("VLA_LEROBOT_PYTHON")
    if not interpreter:
        pytest.skip("set VLA_LEROBOT_PYTHON for official v2.1 merge integration")
    old_curated, old_quality, _ = _tree(
        curated_v1_episode, tmp_path / "old_source", count=2
    )
    new_curated, new_quality, _ = _tree(
        curated_v1_episode, tmp_path / "new_source", count=1
    )
    newer_curated, newer_quality, _ = _tree(
        curated_v1_episode, tmp_path / "newer_source", count=1
    )
    for label, curated in (
        ("old", old_curated),
        ("new", new_curated),
        ("newer", newer_curated),
    ):
        for episode in sorted(curated.iterdir()):
            raw = episode / "test_raw_identity.npz"
            raw.write_bytes(f"{label}:{episode.name}".encode())
            metadata_path = episode / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["source_episode_path"] = str(raw)
            metadata_path.write_text(json.dumps(metadata))
    # Asymmetric media makes an applied (or skipped) 180-degree wrist rotation
    # measurable in the encoded MP4s instead of being pixel-invisible.
    for offset, camera in ((17, "head"), (83, "right_wrist")):
        for episode in sorted(old_curated.iterdir()):
            directory = episode / "media" / camera / "rgb"
            for index, path in enumerate(sorted(directory.glob("*.jpg"))):
                Image.fromarray(_asymmetric_frame(offset + index * 29)).save(path)
    baseline_export = tmp_path / "baseline_export"
    incoming_export = tmp_path / "incoming_export"
    newer_export = tmp_path / "newer_export"
    for output, curated, quality in (
        (baseline_export, old_curated, old_quality),
        (incoming_export, new_curated, new_quality),
        (newer_export, newer_curated, newer_quality),
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
    newer_incoming = newer_export / "train"
    newer_incoming_rows = [
        json.loads(line)
        for line in (newer_incoming / "meta/source_provenance.jsonl")
        .read_text()
        .splitlines()
    ]
    provenance_path = baseline / "meta/source_provenance.jsonl"
    legacy_rows = [
        json.loads(line) for line in provenance_path.read_text().splitlines()
    ]
    for row in legacy_rows:
        row.pop("camera_transforms")
    provenance_path.write_text("".join(json.dumps(row) + "\n" for row in legacy_rows))

    plan = plan_logical_merge(
        incoming,
        baseline,
        baseline_revision="a" * 40,
        source_export_fingerprint="b" * 64,
        legacy_camera_baseline_revisions=("a" * 40,),
    )
    assert plan["requires_camera_transform_migration"] is True
    assert plan["baseline_wrist_rotation_indices"] == list(
        range(plan["baseline_episodes"])
    )
    rebuilt = tmp_path / "migrated"
    evidence = rebuild_logical_dataset(
        incoming, baseline, plan, rebuilt, lerobot_python=interpreter
    )
    assert evidence["official_reload"] == "PASS"
    assert evidence["migrated_baseline_episodes"] == plan["baseline_episodes"]
    migrated_rows = [
        json.loads(line)
        for line in (rebuilt / "meta/source_provenance.jsonl").read_text().splitlines()
    ]
    assert all(
        row["camera_transforms"] == camera_export_transform_contract()
        for row in migrated_rows
    )

    # The migrated output becomes the next baseline and is physically rebuilt
    # again with a fresh incoming batch.
    second_plan = plan_logical_merge(
        newer_incoming,
        rebuilt,
        baseline_revision="c" * 40,
        source_export_fingerprint="d" * 64,
    )
    assert second_plan["requires_camera_transform_migration"] is False
    assert second_plan["baseline_wrist_rotation_indices"] == []
    assert second_plan["to_append"] == len(newer_incoming_rows)
    rebuilt_again = tmp_path / "second_rebuild"
    second_evidence = rebuild_logical_dataset(
        newer_incoming, rebuilt, second_plan, rebuilt_again, lerobot_python=interpreter
    )
    assert second_evidence["official_reload"] == "PASS"
    assert second_evidence["migrated_baseline_episodes"] == 0

    def rotated(frames):
        return [np.rot90(frame, 2) for frame in frames]

    (
        baseline_head,
        baseline_wrist,
        migrated_head,
        migrated_wrist,
        second_head,
        second_wrist,
    ) = (
        decoded["pixels"]
        for decoded in _decode_anchors(
            interpreter,
            [
                (baseline, 2, "observation.images.head"),
                (baseline, 2, "observation.images.wrist"),
                (rebuilt, 2, "observation.images.head"),
                (rebuilt, 2, "observation.images.wrist"),
                (rebuilt_again, 2, "observation.images.head"),
                (rebuilt_again, 2, "observation.images.wrist"),
            ],
        )
    )
    for old, migrated, second in zip(
        zip(baseline_wrist, rotated(baseline_wrist)),
        migrated_wrist,
        second_wrist,
        strict=True,
    ):
        # Exactly one 180-degree wrist migration, not zero and not two.
        assert _mae(old[1], migrated) <= 30
        assert _mae(old[0], migrated) > 30
        # The second rebuild keeps the migrated orientation untouched.
        assert _mae(migrated, second) <= 30
        assert _mae(old[1], second) <= 30
        assert _mae(old[0], second) > 30
    for old, migrated, second in zip(
        zip(baseline_head, rotated(baseline_head)),
        migrated_head,
        second_head,
        strict=True,
    ):
        # The head camera never rotates through either rebuild.
        assert _mae(old[0], migrated) <= 30
        assert _mae(old[0], second) <= 30
        assert _mae(old[1], second) > 30
