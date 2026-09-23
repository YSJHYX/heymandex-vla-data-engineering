"""Logical LeRobot v2.1 episode rebuild in the audited LeRobot interpreter.

This process deliberately imports no project package: the project and LeRobot
use different Python environments. It never reads or writes the Hub.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import av
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

EXPORT_MODULE_DIR = Path(__file__).resolve().parents[1] / "export"
sys.path.insert(0, str(EXPORT_MODULE_DIR))
from camera_transform import (
    CAMERA_EXPORT_TRANSFORMS,
    apply_camera_export_transform,
)

CAMERAS = ("observation.images.head", "observation.images.wrist")
VECTORS = ("observation.state", "action")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(root: str) -> LeRobotDataset:
    return LeRobotDataset("local/append-source", root=root)


def video_frames(dataset: LeRobotDataset, episode_index: int, key: str):
    path = dataset.root / dataset.meta.get_video_file_path(episode_index, key)
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")


def rows(
    dataset: LeRobotDataset,
    episode_index: int,
    *,
    rotate_legacy_wrist: bool,
):
    start = int(dataset.episode_data_index["from"][episode_index])
    stop = int(dataset.episode_data_index["to"][episode_index])
    table = dataset.hf_dataset.select(range(start, stop))
    head = video_frames(dataset, episode_index, CAMERAS[0])
    wrist = video_frames(dataset, episode_index, CAMERAS[1])
    for index, row in enumerate(table):
        images = (next(head, None), next(wrist, None))
        require(all(image is not None for image in images), "source video is short")
        require(int(row["episode_index"]) == episode_index, "source episode index")
        require(int(row["frame_index"]) == index, "source frame index")
        vectors = {key: np.asarray(row[key], dtype=np.float32) for key in VECTORS}
        for key, vector in vectors.items():
            require(
                vector.shape == (17,) and np.isfinite(vector).all(), f"invalid {key}"
            )
        wrist_image = images[1]
        if rotate_legacy_wrist:
            wrist_image = apply_camera_export_transform(
                wrist_image, camera_role="wrist"
            )
        yield {
            **vectors,
            CAMERAS[0]: images[0],
            CAMERAS[1]: wrist_image,
            "task": dataset.meta.tasks[int(row["task_index"])],
        }
    require(next(head, None) is None, "source head video is long")
    require(next(wrist, None) is None, "source wrist video is long")


def validate_source(dataset: LeRobotDataset) -> None:
    require(dataset.meta.info["codebase_version"] == "v2.1", "source format")
    for key in VECTORS:
        feature = dataset.features[key]
        require(feature["dtype"] == "float32" and list(feature["shape"]) == [17], key)
        require(dataset.hf_dataset.features[key].feature.dtype == "float32", key)
    for key in CAMERAS:
        feature = dataset.features[key]
        require(feature["dtype"] == "video" and list(feature["shape"])[-1] == 3, key)
    require(dataset.num_episodes > 0, "source has no episodes")


def build(request: dict) -> dict:
    require(CODEBASE_VERSION == "v2.1", "audited LeRobot v2.1 required")
    require(
        request.get("camera_transforms") == CAMERA_EXPORT_TRANSFORMS,
        "camera export transform contract mismatch",
    )
    baseline = load(request["baseline_root"]) if request.get("baseline_root") else None
    incoming = load(request["incoming_root"])
    validate_source(incoming)
    if baseline is not None:
        validate_source(baseline)
        require(baseline.fps == incoming.fps, "FPS mismatch")
        require(baseline.features == incoming.features, "feature mismatch")
    selected = request["append_indices"]
    require(
        isinstance(selected, list)
        and len(selected) == len(set(selected))
        and all(
            type(index) is int and 0 <= index < incoming.num_episodes
            for index in selected
        ),
        "invalid incoming episode selection",
    )
    migration_indices = request.get("baseline_wrist_rotation_indices")
    baseline_count = baseline.num_episodes if baseline is not None else 0
    require(
        isinstance(migration_indices, list)
        and len(migration_indices) == len(set(migration_indices))
        and all(
            type(index) is int and 0 <= index < baseline_count
            for index in migration_indices
        ),
        "invalid baseline camera migration selection",
    )
    migration_set = set(migration_indices)
    expected = (
        [
            (baseline, index, index in migration_set)
            for index in range(baseline.num_episodes)
        ]
        if baseline is not None
        else []
    )
    expected.extend((incoming, index, False) for index in selected)
    require(bool(expected), "merged dataset would be empty")
    root = Path(request["output_root"])
    require(not root.exists(), "rebuild output must not exist")
    writer = LeRobotDataset.create(
        repo_id="local/cumulative-merge",
        root=root,
        robot_type=incoming.meta.robot_type,
        fps=incoming.fps,
        features={
            key: {**feature, "shape": tuple(feature["shape"])}
            for key, feature in incoming.features.items()
            if key
            not in {"timestamp", "frame_index", "episode_index", "index", "task_index"}
        },
        use_videos=True,
        image_writer_threads=0,
        image_writer_processes=0,
    )
    lengths = []
    for source, episode_index, rotate_legacy_wrist in expected:
        length = int(source.meta.episodes[episode_index]["length"])
        count = 0
        for frame in rows(
            source,
            episode_index,
            rotate_legacy_wrist=rotate_legacy_wrist,
        ):
            writer.add_frame(frame)
            count += 1
        require(count == length, "source episode length mismatch")
        writer.save_episode()
        lengths.append(length)
    writer.stop_image_writer()
    merged = load(str(root))
    require(merged.num_episodes == len(expected), "merged episode count")
    require(merged.num_frames == sum(lengths), "merged frame count")
    require(
        [merged.meta.episodes[i]["length"] for i in range(merged.num_episodes)]
        == lengths,
        "merged episode boundaries",
    )
    for merged_index, (source, source_index, rotate_legacy_wrist) in enumerate(
        expected
    ):
        a = int(source.episode_data_index["from"][source_index])
        b = int(source.episode_data_index["to"][source_index])
        c = int(merged.episode_data_index["from"][merged_index])
        d = int(merged.episode_data_index["to"][merged_index])
        require(b - a == d - c, "merged frame range")
        for key in VECTORS:
            old = torch.stack(source.hf_dataset.select(range(a, b))[key]).numpy()
            new = torch.stack(merged.hf_dataset.select(range(c, d))[key]).numpy()
            require(np.array_equal(old, new), f"merged {key} differs")
        old_tasks = [
            source.meta.tasks[int(i)]
            for i in source.hf_dataset.select(range(a, b))["task_index"]
        ]
        new_tasks = [
            merged.meta.tasks[int(i)]
            for i in merged.hf_dataset.select(range(c, d))["task_index"]
        ]
        require(old_tasks == new_tasks, "merged task differs")
        for key in CAMERAS:
            old_video = video_frames(source, source_index, key)
            new_video = video_frames(merged, merged_index, key)
            count = 0
            for old_frame, new_frame in zip(old_video, new_video, strict=True):
                require(old_frame.shape == new_frame.shape, "merged camera shape")
                if count in {0, lengths[merged_index] // 2, lengths[merged_index] - 1}:
                    expected_frame = old_frame
                    if rotate_legacy_wrist and key == CAMERAS[1]:
                        expected_frame = apply_camera_export_transform(
                            old_frame, camera_role="wrist"
                        )
                    error = np.mean(
                        np.abs(
                            expected_frame.astype(np.float32)
                            - new_frame.astype(np.float32)
                        )
                    )
                    require(error <= 30, "merged camera frame correspondence")
                count += 1
            require(count == lengths[merged_index], "merged video length")
    return {
        "official_reload": "PASS",
        "episodes": merged.num_episodes,
        "frames": merged.num_frames,
        "episode_lengths": lengths,
        "tasks": list(merged.meta.tasks.values()),
        "state_shape": [merged.num_frames, 17],
        "action_shape": [merged.num_frames, 17],
        "camera_shapes": {key: list(merged.features[key]["shape"]) for key in CAMERAS},
        "camera_transforms": CAMERA_EXPORT_TRANSFORMS,
        "baseline_wrist_rotation_indices": migration_indices,
        "migrated_baseline_episodes": len(migration_indices),
    }


def main() -> int:
    try:
        result = {"ok": True, **build(json.load(sys.stdin))}
    except Exception as exc:  # noqa: BLE001 - standalone worker error boundary
        result = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
