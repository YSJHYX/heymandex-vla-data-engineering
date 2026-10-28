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
from writer_options import IMAGE_WRITER_THREADS

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


def _file_digest(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_relpath(episode_index: int, chunks_size: int) -> Path:
    return Path(
        f"data/chunk-{episode_index // chunks_size:03d}/"
        f"episode_{episode_index:06d}.parquet"
    )


def _video_relpath(episode_index: int, camera_key: str, chunks_size: int) -> Path:
    return Path(
        f"videos/chunk-{episode_index // chunks_size:03d}/{camera_key}/"
        f"episode_{episode_index:06d}.mp4"
    )


def _write_episodes(entries: list, root: Path, incoming) -> list[int]:
    """Run the audited frame-by-frame LeRobot writer over the given entries."""

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
        image_writer_threads=IMAGE_WRITER_THREADS,
        image_writer_processes=0,
    )
    lengths: list[int] = []
    for source, episode_index, rotate_legacy_wrist in entries:
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
    return lengths


def _merged_task_table(baseline, partial) -> tuple[dict[int, int], list[str]]:
    """Map partial task indices onto the baseline-first merged task table."""

    baseline_tasks = [baseline.meta.tasks[i] for i in range(len(baseline.meta.tasks))]
    lookup = {task: index for index, task in enumerate(baseline_tasks)}
    mapping: dict[int, int] = {}
    appended: list[str] = []
    for index in range(len(partial.meta.tasks)):
        task = partial.meta.tasks[index]
        if task not in lookup:
            lookup[task] = len(baseline_tasks) + len(appended)
            appended.append(task)
        mapping[index] = lookup[task]
    return mapping, appended


def _reindex_parquet(
    source_file: Path,
    target_file: Path,
    *,
    episode_index: int,
    index_delta: int,
    task_map: dict[int, int],
) -> None:
    """Copy an appended episode Parquet, rebasing its global bookkeeping columns."""

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(source_file)
    field = table.schema.get_field_index

    episode_column = table.column("episode_index").to_numpy()
    require(
        bool(np.all(episode_column == episode_column[0])),
        "partial parquet spans multiple episodes",
    )
    table = table.set_column(
        field("episode_index"),
        "episode_index",
        pa.array(
            np.full(len(table), episode_index, dtype=np.int64),
            type=table.schema.field(field("episode_index")).type,
        ),
    )
    index_values = table.column("index").to_numpy() + index_delta
    table = table.set_column(
        field("index"),
        "index",
        pa.array(index_values, type=table.schema.field(field("index")).type),
    )
    task_values = np.array(
        [task_map[int(value)] for value in table.column("task_index").to_numpy()],
        dtype=np.int64,
    )
    table = table.set_column(
        field("task_index"),
        "task_index",
        pa.array(task_values, type=table.schema.field(field("task_index")).type),
    )
    pq.write_table(table, target_file)


def _merge_meta_files(
    baseline_root: Path,
    partial_root: Path,
    root: Path,
    *,
    baseline_count: int,
    baseline_frames: int,
    migration_count: int,
    append_frames: int,
    total_episodes: int,
    appended_tasks: list[str],
    chunks_size: int,
) -> None:
    import json as _json

    def read_lines(path: Path) -> list[dict]:
        return [
            _json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]

    # tasks.jsonl: baseline rows first, then previously unseen tasks appended.
    task_rows = read_lines(baseline_root / "meta" / "tasks.jsonl")
    next_task = len(task_rows)
    for task in appended_tasks:
        task_rows.append({"task_index": next_task, "task": task})
        next_task += 1
    (root / "meta" / "tasks.jsonl").write_text(
        "".join(_json.dumps(row, sort_keys=True) + "\n" for row in task_rows)
    )

    # episodes.jsonl / episodes_stats.jsonl: baseline rows unchanged, appended
    # rows carry only a rebased episode_index.
    for name in ("episodes.jsonl", "episodes_stats.jsonl"):
        rows_out = read_lines(baseline_root / "meta" / name)
        for row in read_lines(partial_root / "meta" / name)[migration_count:]:
            rows_out.append({**row, "episode_index": len(rows_out)})
        (root / "meta" / name).write_text(
            "".join(_json.dumps(row, sort_keys=True) + "\n" for row in rows_out)
        )

    info = _json.loads((baseline_root / "meta" / "info.json").read_text())
    expected_data_path = (
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    expected_video_path = (
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    )
    require(
        info.get("data_path") == expected_data_path
        and info.get("video_path") == expected_video_path,
        "baseline LeRobot layout template differs from the assembly contract",
    )
    total_tasks = len(task_rows)
    info.update(
        {
            "total_episodes": total_episodes,
            "total_frames": baseline_frames + append_frames,
            "total_tasks": total_tasks,
            "total_videos": 2 * total_episodes,
            "total_chunks": (total_episodes + chunks_size - 1) // chunks_size,
            "splits": {"train": f"0:{total_episodes}"},
        }
    )
    (root / "meta" / "info.json").write_text(
        _json.dumps(info, indent=4, sort_keys=True)
    )


def _assemble_incremental(
    baseline,
    root: Path,
    partial_root: Path,
    migration_indices: list[int],
) -> None:
    """Byte-carry unchanged baseline episodes; graft rebuilt ones in place.

    Baseline episodes outside ``migration_indices`` are copied file-by-file so
    their LFS content hashes stay identical to the pinned remote revision
    (making the upload and the listing audit O(new data), not O(dataset)).
    Only migrated baseline episodes (new wrist-rotated MP4s) and appended
    episodes (new MP4s + reindexed Parquet rows) are produced by the writer.
    """

    import shutil

    baseline_root = Path(baseline.root)
    baseline_count = baseline.num_episodes
    baseline_frames = int(baseline.num_frames)
    n_migration = len(migration_indices)

    partial = load(str(partial_root))
    partial_count = partial.num_episodes
    append_count = partial_count - n_migration
    require(partial_count > 0, "incremental rebuild produced no episodes")

    chunks_size = int(partial.meta.info.get("chunks_size", 1000) or 1000)
    migration_frames = sum(
        int(baseline.meta.episodes[index]["length"]) for index in migration_indices
    )
    append_frames = int(partial.num_frames) - migration_frames
    task_map, appended_tasks = _merged_task_table(baseline, partial)

    root.mkdir(parents=True)
    shutil.copytree(baseline_root / "videos", root / "videos")
    shutil.copytree(baseline_root / "data", root / "data")
    shutil.copytree(baseline_root / "meta", root / "meta")

    for k in range(partial_count):
        target = (
            migration_indices[k]
            if k < n_migration
            else baseline_count + (k - n_migration)
        )
        for camera_key in CAMERAS:
            source_file = partial_root / _video_relpath(k, camera_key, chunks_size)
            target_file = root / _video_relpath(target, camera_key, chunks_size)
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target_file)

    index_delta = baseline_frames - migration_frames
    for k in range(n_migration, partial_count):
        target = baseline_count + (k - n_migration)
        source_file = partial_root / _data_relpath(k, chunks_size)
        target_file = root / _data_relpath(target, chunks_size)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        _reindex_parquet(
            source_file,
            target_file,
            episode_index=target,
            index_delta=index_delta,
            task_map=task_map,
        )

    _merge_meta_files(
        baseline_root,
        partial_root,
        root,
        baseline_count=baseline_count,
        baseline_frames=baseline_frames,
        migration_count=n_migration,
        append_frames=append_frames,
        total_episodes=baseline_count + append_count,
        appended_tasks=appended_tasks,
        chunks_size=chunks_size,
    )


def _write_incremental(
    baseline,
    incoming,
    expected: list,
    root: Path,
    migration_indices: list[int],
) -> tuple[list[int], set[int]]:
    """Rebuild only migrated/appended episodes and byte-carry the baseline."""

    import shutil

    migration_indices = sorted(migration_indices)
    require(
        incoming.meta.robot_type == baseline.meta.robot_type,
        "baseline/incoming robot_type differ",
    )
    copied_indices = {
        merged_index
        for merged_index, entry in enumerate(expected)
        if entry[0] is baseline and not entry[2]
    }
    rebuild_entries = [
        entry for entry in expected if entry[2] or entry[0] is not baseline
    ]
    partial_root = root.parent / f"{root.name}__partial"
    require(not partial_root.exists(), "partial rebuild output must not exist")
    try:
        _write_episodes(rebuild_entries, partial_root, incoming)
        _assemble_incremental(baseline, root, partial_root, migration_indices)
    finally:
        shutil.rmtree(partial_root, ignore_errors=True)

    baseline_count = baseline.num_episodes
    lengths = [
        int(baseline.meta.episodes[index]["length"]) for index in range(baseline_count)
    ]
    lengths.extend(
        int(incoming.meta.episodes[entry[1]]["length"])
        for entry in expected[baseline_count:]
    )
    return lengths, copied_indices


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
    if baseline is None:
        lengths = _write_episodes(expected, root, incoming)
        copied_indices: set[int] = set()
    else:
        lengths, copied_indices = _write_incremental(
            baseline, incoming, expected, root, migration_indices
        )
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
        if merged_index in copied_indices:
            # Byte-carried episodes: the merged MP4s are copies of the source
            # files, so a streamed digest comparison is stronger and far
            # cheaper than decoding both videos frame by frame.
            for key in CAMERAS:
                source_file = source.root / source.meta.get_video_file_path(
                    source_index, key
                )
                merged_file = merged.root / merged.meta.get_video_file_path(
                    merged_index, key
                )
                require(
                    _file_digest(source_file) == _file_digest(merged_file),
                    "copied camera file digest differs",
                )
            continue
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
        "byte_carried_episodes": len(copied_indices),
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
