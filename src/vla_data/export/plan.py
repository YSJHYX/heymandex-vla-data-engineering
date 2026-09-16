"""Contiguous-run planning for direct D2+D3 and optional semantic exports."""

import hashlib
import json
import math
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image

from vla_data.annotation.eligibility import EligibilityError, evaluate_eligibility
from vla_data.batch.discovery import discover_curated_episodes
from vla_data.io.curated_episode import CuratedEpisode
from vla_data.manifest import SplitConfig
from vla_data.manifest.schema import OUTPUT_FILES
from vla_data.manifest.split import assign_splits
from vla_data.manifest.validator import validate_training_manifest
from vla_data.validation.curated_v1 import validate_curated_episode

CODEBASE_VERSION = "v2.1"
CAMERAS = {"observation.images.head": "head", "observation.images.wrist": "right_wrist"}


def _validate_dataset_name(dataset_name: str) -> None:
    if not dataset_name or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for c in dataset_name
    ):
        raise ValueError(
            "dataset-name must contain only letters, digits, hyphen or underscore"
        )


def _task_instruction(episode: CuratedEpisode) -> str:
    """Map the frozen D2 field to the model-facing name without rewriting it."""

    value = episode.metadata.get("language_instruction")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("collection-time task instruction is missing or empty")
    return value


def _is_synthetic(episode: CuratedEpisode) -> bool:
    """Honor every existing explicit synthetic marker conservatively."""

    fields = (
        episode.metadata.get("source_dataset_status"),
        episode.metadata.get("expert_training_status"),
        episode.metadata.get("source_schema_name"),
    )
    return any(
        "SYNTHETIC" in str(value).upper() for value in fields if value is not None
    )


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


def make_mainline_plan(
    curated_root: str | Path,
    quality_root: str | Path,
    dataset_name: str = "vla-local",
    *,
    split_config: SplitConfig | None = None,
    episode: str | None = None,
) -> dict:
    """Plan the production path directly from validated D2 and D3 artifacts.

    One uninterrupted True run inside one D2 physical segment becomes one
    LeRobot episode. D4/D5 files are intentionally neither accepted nor read.
    """

    _validate_dataset_name(dataset_name)
    split_config = split_config or SplitConfig()
    curated = Path(curated_root).resolve()
    quality = Path(quality_root).resolve()
    discovered = discover_curated_episodes(curated, episode=episode)
    if discovered.issues:
        raise ValueError(
            "invalid Curated discovery: "
            + "; ".join(f"{issue.path}: {issue.message}" for issue in discovered.issues)
        )

    features = {
        "observation.state": {"dtype": "float32", "shape": [17], "names": ["state"]},
        "action": {"dtype": "float32", "shape": [17], "names": ["action"]},
    }
    candidates: list[dict] = []
    sources: list[dict] = []
    identities: list[list[object]] = []
    fps_values: set[int] = set()

    for item in discovered.episodes:
        source = {
            "episode_id": item.episode_id,
            "split": None,
            "eligibility": "EXCLUDED",
            "reasons": [],
            "d2_segments": 0,
            "selected_transitions": 0,
            "export_runs": 0,
            "source_dataset_status": None,
            "expert_training_status": None,
            "training_use_status": "NOT_FOR_REAL_MODEL_TRAINING",
        }
        sources.append(source)
        try:
            loaded = CuratedEpisode.load(item.episode_path)
            source["source_dataset_status"] = loaded.metadata.get(
                "source_dataset_status"
            )
            source["expert_training_status"] = loaded.metadata.get(
                "expert_training_status"
            )
            source["d2_segments"] = len(loaded.segment_offsets) - 1
            validation = validate_curated_episode(loaded)
            if not validation.passed:
                source["reasons"].append("D2_INVALID")
            if _is_synthetic(loaded):
                source["reasons"].append("SYNTHETIC_TEST_ONLY")
            if (
                loaded.metadata.get("expert_training_status")
                == "EXCLUDE_FROM_EXPERT_TRAINING"
            ):
                source["reasons"].append("EXCLUDE_FROM_EXPERT_TRAINING")
            try:
                instruction = _task_instruction(loaded)
            except ValueError:
                source["reasons"].append("TASK_INSTRUCTION_MISSING")
                instruction = None

            report_path = quality / item.episode_id / "quality_report.json"
            mask_path = quality / item.episode_id / "quality_mask.npy"
            decision = evaluate_eligibility(
                item.episode_id,
                transition_count=loaded.transition_count,
                quality_report_path=report_path,
                quality_mask_path=mask_path,
            )
            if not decision.eligible:
                source["reasons"].append("D3_NOT_CLEAN")
            report = json.loads(report_path.read_text())
            if (
                report.get("transition_count") != loaded.transition_count
                or report.get("clean_transition_count")
                != decision.clean_transition_count
            ):
                source["reasons"].append("D3_COUNT_MISMATCH")
            mask = np.load(mask_path, allow_pickle=False)

            if not source["reasons"]:
                rate = loaded.metadata.get("dataset_hz")
                if (
                    isinstance(rate, bool)
                    or not isinstance(rate, (float, int))
                    or not math.isfinite(rate)
                    or rate <= 0
                    or rate != int(rate)
                ):
                    source["reasons"].append("INVALID_DATASET_FPS")
                elif (
                    loaded.metadata.get("state_unit") != "rad"
                    or loaded.metadata.get("action_unit") != "rad"
                ):
                    source["reasons"].append("INVALID_PHYSICAL_UNIT")
                else:
                    derived = contiguous_runs(loaded.segment_offsets, mask)
                    if not derived:
                        source["reasons"].append("NO_CLEAN_CONTIGUOUS_RUN")
                    else:
                        fps_values.add(int(rate))
                        candidates.append(
                            {
                                "episode": loaded,
                                "source": source,
                                "instruction": instruction,
                                "derived": derived,
                                "mask_path": mask_path,
                                "report_path": report_path,
                            }
                        )
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            EligibilityError,
        ) as exc:
            source["reasons"].append(f"INPUT_INVALID: {exc}")

    if len(fps_values) > 1:
        raise ValueError("one export requires a consistent source FPS")

    splits = assign_splits(
        (candidate["source"]["episode_id"] for candidate in candidates), split_config
    )
    runs: list[dict] = []
    split_indices = {"train": 0, "val": 0}
    for candidate in candidates:
        loaded = candidate["episode"]
        source = candidate["source"]
        source_id = source["episode_id"]
        split = splits[source_id]
        source.update(
            split=split,
            eligibility="ELIGIBLE",
            training_use_status=(
                "TRAINING_ELIGIBLE_SOURCE"
                if source["expert_training_status"] == "APPROVED_FOR_EXPERT_TRAINING"
                else "INTEGRATION_ELIGIBLE_REVIEW_REQUIRED"
            ),
        )
        instruction = candidate["instruction"]
        instruction_hash = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        image_paths: set[Path] = set()
        for segment, number, start, end in candidate["derived"]:
            state, state_error = storage_vector(loaded.state[start:end])
            action, action_error = storage_vector(loaded.action[start:end])
            del state, action
            ticks = loaded.trajectory["source_tick_index"][start:end]
            if len(ticks) > 1 and not np.all(np.diff(ticks) == 1):
                raise ValueError("run is not contiguous on the source sampler timeline")
            images: dict[str, list[str]] = {}
            for key, role in CAMERAS.items():
                indices = loaded.trajectory[f"{role}_rgb_frame_index"][start:end]
                paths = [loaded.media.rgb_path(role, int(index)) for index in indices]
                images[key] = [str(path) for path in paths]
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
            runs.append(
                {
                    "lerobot_episode_index": split_indices[split],
                    "export_run_id": (
                        f"{source_id}__segment_{segment:03d}__run_{number:03d}"
                    ),
                    "source_episode_id": source_id,
                    "source_segment_id": segment,
                    "source_start_index": start,
                    "source_end_index": end,
                    # Writer compatibility; the explicit fields above use [start, end).
                    "source_curated_start": start,
                    "source_curated_end": end - 1,
                    "transition_count": end - start,
                    "split": split,
                    "task_instruction": instruction,
                    "task_instruction_sha256": instruction_hash,
                    "source_dataset_status": source["source_dataset_status"],
                    "expert_training_status": source["expert_training_status"],
                    "training_use_status": source["training_use_status"],
                    "curated_path": str(loaded.episode_dir),
                    "images": images,
                    "max_abs_state_cast_error": state_error,
                    "max_abs_action_cast_error": action_error,
                }
            )
            split_indices[split] += 1
        source["selected_transitions"] = sum(
            end - start for _, _, start, end in candidate["derived"]
        )
        source["export_runs"] = len(candidate["derived"])
        for path in sorted(image_paths | {loaded.episode_dir / "trajectory.npz"}):
            stat = path.stat()
            identities.append(
                [str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
            )
        for path in (
            loaded.episode_dir / "metadata.json",
            candidate["report_path"],
            candidate["mask_path"],
        ):
            identities.append(
                [str(path), hashlib.sha256(path.read_bytes()).hexdigest()]
            )

    plan = {
        "schema_name": "vla_lerobot_export",
        "schema_version": 1,
        "source_mode": "DIRECT_D2_D3",
        "input_roots": {"curated": str(curated), "quality": str(quality)},
        "episode": episode,
        "split_config": {
            "validation_fraction": split_config.validation_fraction,
            "seed": split_config.seed,
        },
        "dataset_name": dataset_name,
        "codebase_version": CODEBASE_VERSION,
        "features": features,
        "fps": next(iter(fps_values), None),
        "camera_names": list(CAMERAS),
        "sources": sources,
        "runs": runs,
        "source_identities": identities,
    }
    plan["fingerprint"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True).encode()
    ).hexdigest()
    return plan


def make_plan(manifest_root: str | Path, dataset_name: str = "vla-local") -> dict:
    root = Path(manifest_root).resolve()
    _validate_dataset_name(dataset_name)
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
    runs, fps_values, source_map, identities = [], set(), {}, []
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
        semantic_segment_id = row.get("semantic_segment_id")
        if semantic_segment_id is not None:
            start, end = row["start_curated_index"], row["end_curated_index"]
            if not (0 <= start < end <= episode.transition_count) or not bool(
                mask[start:end].all()
            ):
                raise ValueError("semantic segment is outside the D3 clean mask")
            matches = [
                number
                for number, (left, right) in enumerate(
                    pairwise(episode.segment_offsets)
                )
                if left <= start < end <= right
            ]
            if len(matches) != 1:
                raise ValueError("semantic segment crosses a D2 physical boundary")
            derived = [(matches[0], 0, start, end)]
        else:
            derived = contiguous_runs(episode.segment_offsets, mask)
        if (
            sum(end - start for _, _, start, end in derived)
            != row["transition_count_selected"]
        ):
            raise ValueError("run accounting differs from manifest selection")
        source_id = row.get("source_episode_id", row["episode_id"])
        source = source_map.setdefault(
            source_id,
            {
                "episode_id": source_id,
                "split": row["split"],
                "d2_segments": len(episode.segment_offsets) - 1,
                "selected_transitions": 0,
                "export_runs": 0,
                "source_dataset_status": episode.metadata.get("source_dataset_status"),
                "training_use_status": (
                    "NOT_FOR_REAL_MODEL_TRAINING"
                    if episode.metadata.get("source_dataset_status")
                    == "SYNTHETIC_TEST_ONLY"
                    else "TRAINING_ELIGIBLE_SOURCE"
                ),
            },
        )
        if source["split"] != row["split"]:
            raise ValueError("semantic segments from one source cross train/val split")
        source["selected_transitions"] += row["transition_count_selected"]
        source["export_runs"] += len(derived)
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
                    "export_run_id": (
                        row.get("training_unit_id")
                        or f"{row['episode_id']}__segment_{segment:03d}__run_{number:03d}"
                    ),
                    "source_episode_id": source_id,
                    "source_segment_id": segment,
                    "semantic_segment_id": semantic_segment_id,
                    "episode_task": row.get("episode_task"),
                    "episode_task_model_instruction": row.get(
                        "episode_task_model_instruction"
                    ),
                    "episode_task_verification_status": row.get(
                        "episode_task_verification_status"
                    ),
                    "training_use_status": source["training_use_status"],
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
        "sources": [source_map[key] for key in sorted(source_map)],
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
    direct = plan.get("source_mode") == "DIRECT_D2_D3"
    return {
        "source_episodes": len(plan["sources"]),
        "source_episodes_discovered": len(plan["sources"]),
        "eligible_source_episodes": sum(
            source.get("eligibility") == "ELIGIBLE" for source in plan["sources"]
        )
        if direct
        else len(plan["sources"]),
        "export_runs": len(plan["runs"]),
        "clean_runs_generated": len(plan["runs"]),
        "rejected_runs": 0,
        "minimum_run_length": None,
        "short_runs_rejected": 0,
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
        "task_count": len(
            {
                r.get("task_instruction", r.get("final_instruction"))
                for r in plan["runs"]
            }
        ),
        "excluded_source_episodes": sum(
            s.get("eligibility") == "EXCLUDED" for s in plan["sources"]
        ),
        "synthetic_source_episodes_excluded": sum(
            "SYNTHETIC_TEST_ONLY" in s.get("reasons", []) for s in plan["sources"]
        ),
        "missing_task_source_episodes_excluded": sum(
            "TASK_INSTRUCTION_MISSING" in s.get("reasons", []) for s in plan["sources"]
        ),
        "fps": plan["fps"],
        "fingerprint": plan["fingerprint"],
    }
