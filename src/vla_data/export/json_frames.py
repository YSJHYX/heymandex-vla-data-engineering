"""Separate eight-field JSONL export; never changes the LeRobot dataset."""

from __future__ import annotations

import json
import tempfile
from contextlib import ExitStack
from pathlib import Path

from PIL import Image


def export_frame_json(dataset_root, output_root, *, lerobot_python):
    """Export a LeRobot root (or its train/val roots) into a separate directory."""
    from vla_data.export.runtime import worker_call

    source = Path(dataset_root).resolve(strict=True)
    destination = Path(output_root).absolute()
    if destination.is_symlink():
        raise ValueError("JSON output must not be a symlink")
    destination = destination.resolve()
    if (
        source == destination
        or source in destination.parents
        or destination in source.parents
    ):
        raise ValueError("JSON output must be separate from the LeRobot input tree")
    if destination.exists():
        raise ValueError("JSON output already exists; choose a new output directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sources = (
        {"": source}
        if (source / "meta/info.json").is_file()
        else {
            split: source / split
            for split in ("train", "val")
            if (source / split / "meta/info.json").is_file()
        }
    )
    if not sources:
        raise ValueError("no LeRobot datasets found for JSON export")
    with tempfile.TemporaryDirectory(
        prefix=".frame-json-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary) / "dataset"
        if "" not in sources:
            staging.mkdir()
        results = {}
        for split, root in sources.items():
            results[split or "dataset"] = worker_call(
                "export_frame_json",
                {"dataset_root": str(root), "output_root": str(staging / split)},
                lerobot_python,
            )
        staging.rename(destination)
    return {"output_root": str(destination), "splits": results}


def write_json_frames(request):
    """Standalone worker implementation: stream decoded RGB and JSONL rows."""
    import av
    import numpy as np
    import pyarrow.parquet as pq

    source, output = Path(request["dataset_root"]), Path(request["output_root"])
    info = json.loads((source / "meta/info.json").read_text())
    if info.get("codebase_version") != "v2.1":
        raise ValueError("JSON export requires a LeRobot v2.1 dataset")
    for key in ("observation.state", "action"):
        feature = info["features"][key]
        if feature["dtype"] != "float32" or feature["shape"] != [17]:
            raise ValueError("JSON export requires physical float32[17] state/action")
    episodes = [
        json.loads(line)
        for line in (source / "meta/episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    output.mkdir()
    total = 0
    for episode in episodes:
        index, count = episode["episode_index"], episode["length"]
        chunk = index // info["chunks_size"]
        paths = {"episode_chunk": chunk, "episode_index": index}
        native = pq.read_table(source / info["data_path"].format(**paths)).to_pylist()
        if len(native) != count:
            raise ValueError("JSON source Parquet/episode frame count mismatch")
        directory = output / f"episode_{index:06d}"
        directory.mkdir()
        with ExitStack() as stack:
            videos = {}
            for alias, key in (
                ("image_top", "observation.images.head"),
                ("image_hand", "observation.images.wrist"),
            ):
                container = stack.enter_context(
                    av.open(
                        str(source / info["video_path"].format(**paths, video_key=key))
                    )
                )
                videos[alias] = iter(container.decode(video=0))
                (directory / alias).mkdir()
            stream = stack.enter_context(
                (directory / "frames.jsonl").open("w", encoding="utf-8")
            )
            for frame, native_row in enumerate(native):
                if (
                    native_row["episode_index"] != index
                    or native_row["frame_index"] != frame
                ):
                    raise ValueError("JSON source frame order mismatch")
                row = {"t": float(native_row["timestamp"])}
                for name, video in videos.items():
                    image = next(video, None)
                    if image is None:
                        raise ValueError("JSON source video is shorter than episode")
                    relative = f"{name}/{frame:06d}.png"
                    Image.fromarray(image.to_ndarray(format="rgb24")).save(
                        directory / relative, compress_level=1
                    )
                    row[name] = relative
                for name in ("depth_top", "depth_hand"):
                    row[name] = None
                row["joint_pos"] = native_row["observation.state"]
                row["action"] = native_row["action"]
                row["success"] = int(frame == count - 1)
                for key in ("joint_pos", "action"):
                    if len(row[key]) != 17 or not np.isfinite(row[key]).all():
                        raise ValueError("JSON requires finite 17-dimensional vectors")
                stream.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )
            if any(next(video, None) is not None for video in videos.values()):
                raise ValueError("JSON source video is longer than episode")
        total += count
    return {
        "episodes": len(episodes),
        "frames": total,
        "format": "JSONL",
        "fields": [
            "t",
            "image_top",
            "image_hand",
            "depth_top",
            "depth_hand",
            "joint_pos",
            "action",
            "success",
        ],
    }
