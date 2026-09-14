"""Manifest-authorized contiguous-run planning; no LeRobot dependency."""

import hashlib
import json
import math
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image

from vla_data.io.curated_episode import CuratedEpisode
from vla_data.manifest.schema import OUTPUT_FILES
from vla_data.manifest.validator import validate_training_manifest

CODEBASE_VERSION = "v2.1"
CAMERAS = {"observation.images.head": "head", "observation.images.wrist": "right_wrist"}


def contiguous_runs(
    offsets: tuple[int, ...], mask: np.ndarray
) -> list[tuple[int, int, int, int]]:
    """(segment, local run number, inclusive start, exclusive end)."""
    if mask.dtype != np.dtype(bool) or mask.ndim != 1:
        raise ValueError("quality mask must be 1D bool")
    if (
        not offsets
        or offsets[0] != 0
        or offsets[-1] != len(mask)
        or any(a >= b for a, b in pairwise(offsets))
    ):
        raise ValueError("invalid D2 segment offsets")
    result = []
    for segment, (start, end) in enumerate(pairwise(offsets)):
        number = 0
        index = start
        while index < end:
            if not mask[index]:
                index += 1
                continue
            begin = index
            while index < end and mask[index]:
                index += 1
            result.append((segment, number, begin, index))
            number += 1
    return result


def storage_vector(values: np.ndarray) -> tuple[np.ndarray, float]:
    if values.ndim != 2 or values.shape[1] != 17 or not np.all(np.isfinite(values)):
        raise ValueError("source vectors must be finite [N,17]")
    converted = values.astype(np.float32)
    if not np.all(np.isfinite(converted)):
        raise ValueError("float32 cast overflow")
    error = (
        float(np.max(np.abs(converted.astype(np.float64) - values)))
        if values.size
        else 0.0
    )
    return converted, error


def make_plan(manifest_root: str | Path, dataset_name: str = "vla-local") -> dict:
    root = Path(manifest_root).resolve()
    if not dataset_name or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for c in dataset_name
    ):
        raise ValueError(
            "dataset-name must contain only letters, digits, hyphen or underscore"
        )
    checked = validate_training_manifest(root)
    if not checked.passed:
        raise ValueError(f"invalid training manifest: {checked.errors}")
    rows = [
        json.loads(line)
        for line in (root / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    manifest_summary = json.loads((root / "dataset_manifest_summary.json").read_text())
    features = {
        "observation.state": {"dtype": "float32", "shape": [17], "names": ["state"]},
        "action": {"dtype": "float32", "shape": [17], "names": ["action"]},
    }
    runs, fps_values, sources, identities = [], set(), [], []
    split_indices = {"train": 0, "val": 0}
    for row in rows:
        episode = CuratedEpisode.load(row["curated_path"])
        rate = episode.metadata.get("dataset_hz")
        if (
            isinstance(rate, bool)
            or not isinstance(rate, (float, int))
            or not math.isfinite(rate)
            or rate <= 0
            or rate != int(rate)
        ):
            raise ValueError(
                "Curated dataset_hz must specify a positive integer FPS; no fallback"
            )
        fps_values.add(int(rate))
        if (
            episode.metadata.get("state_unit") != "rad"
            or episode.metadata.get("action_unit") != "rad"
        ):
            raise ValueError("state/action units must remain rad")
        mask = np.load(row["training_transition_mask_path"], allow_pickle=False)
        derived = contiguous_runs(episode.segment_offsets, mask)
        if (
            sum(end - start for _, _, start, end in derived)
            != row["transition_count_selected"]
        ):
            raise ValueError("run accounting differs from manifest selection")
        sources.append(
            {
                "episode_id": row["episode_id"],
                "split": row["split"],
                "d2_segments": len(episode.segment_offsets) - 1,
                "selected_transitions": row["transition_count_selected"],
                "export_runs": len(derived),
            }
        )
        # Lightweight identities plus headers, not another full image-byte hash.
        image_paths = set()
        for segment, number, start, end in derived:
            state, state_error = storage_vector(episode.state[start:end])
            action, action_error = storage_vector(episode.action[start:end])
            del state, action
            ticks = episode.trajectory["source_tick_index"][start:end]
            if len(ticks) > 1 and not np.all(np.diff(ticks) == 1):
                raise ValueError("run is not contiguous on the source sampler timeline")
            images = {}
            for key, role in CAMERAS.items():
                indices = episode.trajectory[f"{role}_rgb_frame_index"][start:end]
                paths = [episode.media.rgb_path(role, int(i)) for i in indices]
                images[key] = [str(p) for p in paths]
                for path in paths:
                    if path in image_paths:
                        continue
                    with Image.open(path) as image:
                        if image.mode != "RGB":
                            raise ValueError(f"source image must be RGB: {path}")
                        shape = [image.height, image.width, 3]
                    feature = {
                        "dtype": "video",
                        "shape": shape,
                        "names": ["height", "width", "channels"],
                    }
                    if key in features and features[key] != feature:
                        raise ValueError(f"inconsistent native resolution for {key}")
                    features[key] = feature
                    image_paths.add(path)
            split = row["split"]
            runs.append(
                {
                    "lerobot_episode_index": split_indices[split],
                    "export_run_id": f"{row['episode_id']}__segment_{segment:03d}__run_{number:03d}",
                    "source_episode_id": row["episode_id"],
                    "source_segment_id": segment,
                    "source_curated_start": start,
                    "source_curated_end": end - 1,
                    "transition_count": end - start,
                    "split": split,
                    "final_instruction": row["final_instruction"],
                    "curated_path": str(episode.episode_dir),
                    "images": images,
                    "max_abs_state_cast_error": state_error,
                    "max_abs_action_cast_error": action_error,
                }
            )
            split_indices[split] += 1
        for path in sorted(image_paths | {episode.episode_dir / "trajectory.npz"}):
            stat = path.stat()
            identities.append(
                [str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
            )
        for path in (
            episode.episode_dir / "metadata.json",
            Path(row["training_transition_mask_path"]),
        ):
            identities.append(
                [str(path), hashlib.sha256(path.read_bytes()).hexdigest()]
            )
    if len(fps_values) > 1:
        raise ValueError("one export requires a consistent source FPS")
    plan = {
        "schema_name": "vla_lerobot_export",
        "schema_version": 1,
        "manifest_root": str(root),
        "manifest_schema_version": manifest_summary["schema_version"],
        "input_roots": manifest_summary["input_roots"],
        "dataset_name": dataset_name,
        "codebase_version": CODEBASE_VERSION,
        "features": features,
        "fps": next(iter(fps_values), None),
        "camera_names": list(CAMERAS),
        "sources": sources,
        "runs": runs,
        "manifest_hashes": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in OUTPUT_FILES
        },
        "source_identities": identities,
    }
    plan["fingerprint"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True).encode()
    ).hexdigest()
    return plan


def plan_summary(plan: dict) -> dict:
    return {
        "source_episodes": len(plan["sources"]),
        "export_runs": len(plan["runs"]),
        **{
            f"{split}_source_episodes": sum(
                s["split"] == split for s in plan["sources"]
            )
            for split in ("train", "val")
        },
        **{
            f"{split}_export_runs": sum(r["split"] == split for r in plan["runs"])
            for split in ("train", "val")
        },
        "selected_transitions": sum(r["transition_count"] for r in plan["runs"]),
        "camera_features": plan["camera_names"],
        "state_action_shape": [17],
        "task_count": len({r["final_instruction"] for r in plan["runs"]}),
        "fps": plan["fps"],
        "fingerprint": plan["fingerprint"],
    }
