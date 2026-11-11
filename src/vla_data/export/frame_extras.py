"""Optional RGB-D/event sidecars; canonical LeRobot features stay unchanged.

Importable by the standalone Python 3.11 worker. PyArrow is needed only for
writing/reading sidecar tables in the existing LeRobot interpreter.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

SCHEMA = {
    "schema_name": "vla_frame_extras",
    "schema_version": 1,
    "join_keys": ["episode_index", "frame_index"],
    "asset_paths_relative_to": "episode_directory",
    "depth_storage": "uint16_png",
    "depth_units": "raw_value * scale_m_per_unit = meters",
    "depth_alignment": "source_rgb_before_export_rotation",
    "calibration_coordinates": "source_rgb_before_export_rotation",
    "success_semantics": "annotated_event_at_source_tick; never inferred from run end",
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _integer(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _asset(base, value):
    if not isinstance(value, str) or not value:
        raise ValueError("asset path must be a nonempty string")
    path = (base / value).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"asset is not a file: {path}")
    return {"source_path": str(path), "sha256": digest(path)}


def enrich_plan(plan, manifest_path):
    """Bind explicitly timestamped depth/events to selected physical source ticks."""
    if manifest_path is None:
        return plan
    manifest = Path(manifest_path).resolve(strict=True)
    entries = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not set(row) <= {
            "source_episode_id",
            "source_tick_index",
            "success",
            "success_source",
            "depth_top",
            "depth_hand",
        }:
            raise ValueError("unsupported frame-extras input fields")
        source = row.get("source_episode_id")
        if not isinstance(source, str) or not source:
            raise ValueError("source_episode_id is required")
        tick = _integer(row.get("source_tick_index"), "source_tick_index")
        key = (source, tick)
        if key in entries:
            raise ValueError(f"duplicate frame-extras source tick: {key}")
        success = row.get("success")
        if success is not None and (type(success) is not int or success not in (0, 1)):
            raise ValueError("success must be 0, 1 or null (unknown)")
        if success is not None and not (
            isinstance(row.get("success_source"), str) and row["success_source"].strip()
        ):
            raise ValueError("known success requires success_source evidence")
        entries[key] = row

    selected = set()
    extras = {}
    for run in plan["runs"]:
        with np.load(
            Path(run["curated_path"]) / "trajectory.npz", allow_pickle=False
        ) as data:
            rows = []
            for local, index in enumerate(
                range(run["source_curated_start"], run["source_curated_end"] + 1)
            ):
                tick = int(data["source_tick_index"][index])
                key = (run["source_episode_id"], tick)
                selected.add(key)
                entry = entries.get(key, {})
                frame = {
                    "episode_index": run["lerobot_episode_index"],
                    "frame_index": local,
                    "t": local / plan["fps"],
                    "source_episode_id": key[0],
                    "source_tick_index": tick,
                    "source_state_timestamp_ns": int(data["state_timestamp_ns"][index]),
                    "source_action_timestamp_ns": int(
                        data["action_timestamp_ns"][index]
                    ),
                    "image_top": Path(
                        run["images"]["observation.images.head"][local]
                    ).name,
                    "image_hand": Path(
                        run["images"]["observation.images.wrist"][local]
                    ).name,
                    "success": entry.get("success") or 0,
                    "success_valid": entry.get("success") is not None,
                    "success_source": entry.get("success_source")
                    if entry.get("success") is not None
                    else None,
                }
                for name, camera in (("depth_top", "head"), ("depth_hand", "wrist")):
                    spec = entry.get(name)
                    frame[name] = None
                    if spec is None:
                        continue
                    if not isinstance(spec, dict) or set(spec) != {
                        "path",
                        "timestamp_ns",
                        "rgb_timestamp_ns",
                        "rgb_frame_index",
                        "max_skew_ns",
                        "scale_m_per_unit",
                        "invalid_value",
                        "calibration",
                        "aligned_to_rgb",
                    }:
                        raise ValueError(
                            f"{name}: incomplete or unsupported depth specification"
                        )
                    if spec["aligned_to_rgb"] is not True:
                        raise ValueError("depth must already be aligned to source RGB")
                    role = "head" if camera == "head" else "right_wrist"
                    if _integer(spec["rgb_frame_index"], "rgb_frame_index") != int(
                        data[f"{role}_rgb_frame_index"][index]
                    ):
                        raise ValueError(
                            "depth references a different source RGB frame"
                        )
                    ts = _integer(spec["timestamp_ns"], "depth timestamp_ns")
                    rgb_ts = _integer(spec["rgb_timestamp_ns"], "rgb_timestamp_ns")
                    skew = _integer(spec["max_skew_ns"], "max_skew_ns")
                    if (
                        not (
                            0 < ts < frame["source_action_timestamp_ns"]
                            and 0 < rgb_ts < frame["source_action_timestamp_ns"]
                        )
                        or abs(ts - rgb_ts) > skew
                    ):
                        raise ValueError(
                            "depth/RGB timestamp skew or pre-action causality failed"
                        )
                    scale = spec["scale_m_per_unit"]
                    invalid = _integer(spec["invalid_value"], "invalid_value")
                    if (
                        isinstance(scale, bool)
                        or not isinstance(scale, (int, float))
                        or not math.isfinite(scale)
                        or scale <= 0
                        or invalid > 65535
                    ):
                        raise ValueError("invalid depth scale or invalid_value")
                    asset = _asset(manifest.parent, spec["path"])
                    calibration = _asset(manifest.parent, spec["calibration"])
                    calibration_data = json.loads(
                        Path(calibration["source_path"]).read_text()
                    )
                    if not isinstance(calibration_data, dict):
                        raise TypeError("calibration must be a JSON object")
                    intrinsics = np.asarray(calibration_data.get("K"), dtype=float)
                    if (
                        intrinsics.shape != (3, 3)
                        or not np.isfinite(intrinsics).all()
                        or intrinsics[0, 0] <= 0
                        or intrinsics[1, 1] <= 0
                        or not np.array_equal(intrinsics[2], [0, 0, 1])
                    ):
                        raise ValueError(
                            "calibration K must be finite source-RGB 3x3 intrinsics"
                        )
                    with Image.open(asset["source_path"]) as image:
                        pixels = np.asarray(image)
                        expected = plan["features"][
                            "observation.images.head"
                            if camera == "head"
                            else "observation.images.wrist"
                        ]["shape"][:2]
                        if (
                            image.format != "PNG"
                            or pixels.dtype != np.uint16
                            or list(pixels.shape) != expected
                        ):
                            raise ValueError(
                                "depth must be uint16 PNG at the source RGB resolution"
                            )
                    frame[name] = {
                        **spec,
                        **asset,
                        "calibration": calibration,
                        "rotation_deg": run["camera_transforms"][camera][
                            "rotation_deg"
                        ],
                    }
                rows.append(frame)
        extras[run["export_run_id"]] = rows
    ignored = sorted(set(entries) - selected)
    extension = {
        "manifest_path": str(manifest),
        "manifest_sha256": digest(manifest),
        "ignored_source_ticks": [list(key) for key in ignored],
        "runs": extras,
    }
    return {
        **plan,
        "frame_extras": extension,
        "fingerprint": _fingerprint([plan["fingerprint"], extension]),
    }


def _episode_dir(root, index):
    return Path(root) / "extras" / "episodes" / f"episode_{index:06d}"


def _write_manifest(root, episodes):
    extra = Path(root) / "extras"
    extra.mkdir(exist_ok=True)
    (extra / "schema.json").write_text(
        json.dumps(SCHEMA, sort_keys=True, indent=2) + "\n"
    )
    files = {
        path.relative_to(extra).as_posix(): digest(path)
        for path in sorted(extra.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    (extra / "manifest.json").write_text(
        json.dumps({"episodes": episodes, "files": files}, sort_keys=True, indent=2)
        + "\n"
    )


def write_extras(plan, output):
    import pyarrow as pa
    import pyarrow.parquet as pq

    extension = plan.get("frame_extras")
    if extension is None:
        return
    for split in ("train", "val"):
        episodes = []
        root = Path(output) / split
        for run in (r for r in plan["runs"] if r["split"] == split):
            index = run["lerobot_episode_index"]
            directory = _episode_dir(root, index)
            directory.mkdir(parents=True)
            frames = []
            source_rows = extension["runs"][run["export_run_id"]]
            for row in source_rows:
                frame = dict(row)
                for name in ("depth_top", "depth_hand"):
                    spec = row[name]
                    frame[name] = None
                    frame[f"{name}_valid"] = spec is not None
                    frame[f"{name}_metadata"] = None
                    if spec is None:
                        continue
                    for asset in (spec, spec["calibration"]):
                        if digest(asset["source_path"]) != asset["sha256"]:
                            raise ValueError(
                                "frame-extras source changed during export"
                            )
                    relative = f"{name}/{row['frame_index']:06d}.png"
                    target = directory / relative
                    target.parent.mkdir(exist_ok=True)
                    with Image.open(spec["source_path"]) as image:
                        pixels = np.asarray(image)
                        if spec["rotation_deg"] == 180:
                            pixels = np.rot90(pixels, 2)
                        Image.fromarray(pixels).save(target)
                    calibration = f"{name}/{row['frame_index']:06d}.calibration.json"
                    shutil.copyfile(
                        spec["calibration"]["source_path"], directory / calibration
                    )
                    frame[name] = relative
                    frame[f"{name}_metadata"] = json.dumps(
                        {
                            key: value
                            for key, value in spec.items()
                            if key not in {"path", "source_path", "calibration"}
                        }
                        | {"calibration": calibration},
                        sort_keys=True,
                    )
                frames.append(frame)
            pq.write_table(pa.Table.from_pylist(frames), directory / "frames.parquet")
            # Exclude split-local IDs, retaining exact source rows/assets for duplicate checks.
            identity = [
                {k: v for k, v in row.items() if k != "episode_index"} for row in frames
            ]
            episodes.append(
                {
                    "episode_index": index,
                    "frame_count": len(frames),
                    "source_fingerprint": _fingerprint(identity),
                }
            )
        if episodes:
            _write_manifest(root, episodes)


def _contained(extra, name):
    if (
        not isinstance(name, str)
        or Path(name).is_absolute()
        or ".." in Path(name).parts
    ):
        raise ValueError("unsafe frame-extras relative path")
    path = extra / name
    if path.is_symlink() or not path.resolve().is_relative_to(extra.resolve()):
        raise ValueError("frame-extras path escapes dataset")
    return path


def validate_package(root):
    """Dependency-light integrity check, also used by publication and fresh reload."""
    root = Path(root)
    extra = root / "extras"
    if not extra.exists():
        return {}
    if extra.is_symlink() or json.loads((extra / "schema.json").read_text()) != SCHEMA:
        raise ValueError("unsupported frame-extras schema")
    manifest = json.loads((extra / "manifest.json").read_text())
    if any(p.is_symlink() for p in extra.rglob("*")):
        raise ValueError("frame-extras symlinks are prohibited")
    actual = {
        p.relative_to(extra).as_posix()
        for p in extra.rglob("*")
        if p.is_file() and p != extra / "manifest.json"
    }
    if actual != set(manifest["files"]):
        raise ValueError("frame-extras file inventory mismatch")
    for name, expected in manifest["files"].items():
        if digest(_contained(extra, name)) != expected:
            raise ValueError(f"frame-extras checksum mismatch: {name}")
    lengths = {
        row["episode_index"]: row["length"]
        for row in map(
            json.loads, (root / "meta/episodes.jsonl").read_text().splitlines()
        )
    }
    episodes = {}
    for entry in manifest["episodes"]:
        index = _integer(entry["episode_index"], "episode_index")
        if index in episodes or lengths.get(index) != entry["frame_count"]:
            raise ValueError("frame-extras episode coverage/count mismatch")
        table = f"episodes/episode_{index:06d}/frames.parquet"
        if table not in actual:
            raise ValueError("frame-extras episode table missing")
        episodes[index] = entry
    if any(
        name.startswith("episodes/")
        and not any(name.startswith(f"episodes/episode_{i:06d}/") for i in episodes)
        for name in actual
    ):
        raise ValueError("orphan frame-extras episode assets")
    return episodes


def validate_tables(root):
    import pyarrow.parquet as pq

    episodes = validate_package(root)
    info = json.loads((Path(root) / "meta/info.json").read_text()) if episodes else {}
    for index, entry in episodes.items():
        directory = _episode_dir(root, index)
        rows = pq.read_table(directory / "frames.parquet").to_pylist()
        if len(rows) != entry["frame_count"] or any(
            row["episode_index"] != index or row["frame_index"] != frame
            for frame, row in enumerate(rows)
        ):
            raise ValueError("frame-extras table join keys/count mismatch")
        identity = [
            {k: v for k, v in row.items() if k != "episode_index"} for row in rows
        ]
        if _fingerprint(identity) != entry["source_fingerprint"]:
            raise ValueError("frame-extras table source fingerprint mismatch")
        chunk = index // info["chunks_size"]
        native_path = Path(root) / info["data_path"].format(
            episode_chunk=chunk, episode_index=index
        )
        native = pq.read_table(native_path).to_pylist()
        if len(native) != len(rows) or any(
            not math.isclose(
                row["t"], float(native[frame]["timestamp"]), abs_tol=1e-6, rel_tol=1e-6
            )
            for frame, row in enumerate(rows)
        ):
            raise ValueError("frame-extras timestamps differ from canonical rows")
        for row in rows:
            if row["success"] not in (0, 1) or (
                not row["success_valid"] and row["success"] != 0
            ):
                raise ValueError("invalid frame-extras success")
            for name in ("depth_top", "depth_hand"):
                if row[f"{name}_valid"] != (row[name] is not None):
                    raise ValueError("depth validity/path mismatch")
                if row[name] is not None:
                    _contained(directory, row[name]).resolve(strict=True)
                    metadata = json.loads(row[f"{name}_metadata"])
                    _contained(directory, metadata["calibration"]).resolve(strict=True)
    return {
        "episodes": len(episodes),
        "frames": sum(e["frame_count"] for e in episodes.values()),
    }


def validate_against_plan(plan, output):
    import pyarrow.parquet as pq

    extension = plan.get("frame_extras")
    evidence = {}
    for split in ("train", "val"):
        root = Path(output) / split
        if not root.is_dir():
            continue
        evidence[split] = validate_tables(root)
        runs = [r for r in plan["runs"] if r["split"] == split]
        if evidence[split]["episodes"] != (len(runs) if extension else 0):
            raise ValueError("frame-extras coverage differs from source plan")
        if extension is None:
            continue
        for run in runs:
            directory = _episode_dir(root, run["lerobot_episode_index"])
            rows = pq.read_table(directory / "frames.parquet").to_pylist()
            expected = extension["runs"][run["export_run_id"]]
            for row, source in zip(rows, expected, strict=True):
                for key, value in source.items():
                    if key not in ("depth_top", "depth_hand") and row[key] != value:
                        raise ValueError(
                            f"frame-extras source correspondence failed: {key}"
                        )
                for name in ("depth_top", "depth_hand"):
                    spec = source[name]
                    if (spec is not None) != row[f"{name}_valid"]:
                        raise ValueError("frame-extras source depth validity mismatch")
                    if spec is None:
                        continue
                    with (
                        Image.open(spec["source_path"]) as original,
                        Image.open(directory / row[name]) as stored,
                    ):
                        pixels = np.asarray(original)
                        if spec["rotation_deg"] == 180:
                            pixels = np.rot90(pixels, 2)
                        if not np.array_equal(pixels, np.asarray(stored)):
                            raise ValueError(
                                "frame-extras depth pixels differ from transformed source"
                            )
                    metadata = json.loads(row[f"{name}_metadata"])
                    expected_metadata = {
                        k: v
                        for k, v in spec.items()
                        if k not in {"path", "source_path", "calibration"}
                    }
                    if {
                        k: v for k, v in metadata.items() if k != "calibration"
                    } != expected_metadata or digest(
                        directory / metadata["calibration"]
                    ) != spec["calibration"]["sha256"]:
                        raise ValueError(
                            "frame-extras depth metadata/calibration mismatch"
                        )
    return evidence


def frame_view(request):
    """Join sidecar and canonical data into the requested eight-field view."""
    import pyarrow.parquet as pq

    root = Path(request["dataset_root"])
    index = _integer(request["episode_index"], "episode_index")
    frame = _integer(request["frame_index"], "frame_index")
    validate_tables(root)
    directory = _episode_dir(root, index)
    sidecar = pq.read_table(directory / "frames.parquet").to_pylist()[frame]
    info = json.loads((root / "meta/info.json").read_text())
    chunk = index // info["chunks_size"]
    path = root / info["data_path"].format(episode_chunk=chunk, episode_index=index)
    native = pq.read_table(path).to_pylist()[frame]
    return {
        **sidecar,
        "joint_pos": native["observation.state"],
        "action": native["action"],
        "depth_base_directory": str(directory.resolve()),
        "rgb_videos": {
            alias: info["video_path"].format(
                episode_chunk=chunk, episode_index=index, video_key=key
            )
            for alias, key in (
                ("image_top", "observation.images.head"),
                ("image_hand", "observation.images.wrist"),
            )
        },
    }


def merge_extras(request):
    """Copy optional episodes and rebase table IDs after the logical merge."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = Path(request["output_root"])
    episodes = []
    sources = {
        root: validate_package(root)
        for root in {m["root"] for m in request["episodes"]}
    }
    for mapping in request["episodes"]:
        root, old, new = Path(mapping["root"]), mapping["old"], mapping["new"]
        entries = sources[mapping["root"]]
        if old not in entries:
            continue
        directory = _episode_dir(output, new)
        shutil.copytree(_episode_dir(root, old), directory)
        rows = pq.read_table(directory / "frames.parquet").to_pylist()
        for row in rows:
            row["episode_index"] = new
        pq.write_table(pa.Table.from_pylist(rows), directory / "frames.parquet")
        episodes.append({**entries[old], "episode_index": new})
    if episodes:
        _write_manifest(output, episodes)
    return validate_tables(output)
