"""Production RAW-to-Curated-v1 episode builder."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from vla_data.cleaning.causal_sync import (
    CausalSyncResult,
    CausalTransition,
    synchronize_episode,
)
from vla_data.contracts.curated_v1 import (
    ACTION_REPRESENTATION,
    ACTION_UNIT,
    CURATED_DATASET_STATUS_READY,
    CURATED_SCHEMA_NAME,
    CURATED_SCHEMA_VERSION,
    STATE_REPRESENTATION,
    STATE_UNIT,
)
from vla_data.io.curated_episode import CuratedEpisode
from vla_data.io.raw_episode import RawEpisode
from vla_data.morphology.rm65_sg100 import (
    ARM_SLICE,
    HAND_SLICE,
    ROBOT_JOINT_NAMES,
)
from vla_data.validation.curated_v1 import validate_curated_episode


def build_curated_v1(
    raw_path: str | Path,
    output_root: str | Path,
    *,
    media_root: str | Path | None = None,
    expert_training_status: str = "REVIEW_REQUIRED",
) -> Path:
    """Build, independently validate, and atomically publish one episode."""

    raw = RawEpisode.load(raw_path, media_root=media_root)
    sync = synchronize_episode(raw)
    if not sync.transitions:
        raise ValueError("D2 rejected every candidate transition; no output published")

    destination_root = Path(output_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / raw.episode_id
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")

    staging = Path(
        tempfile.mkdtemp(prefix=f".{raw.episode_id}.staging-", dir=destination_root)
    )
    try:
        trajectory = _trajectory(sync.transitions)
        metadata = _metadata(
            raw,
            sync.transitions,
            expert_training_status=expert_training_status,
        )
        report = _cleaning_report(
            raw,
            sync,
            trajectory,
            expert_training_status=expert_training_status,
        )

        _write_json(staging / "metadata.json", metadata)
        np.savez(staging / "trajectory.npz", **trajectory)
        _copy_selected_media(raw, staging, sync.transitions)
        _write_json(staging / "cleaning_report.json", report)

        validation = validate_curated_episode(CuratedEpisode.load(staging))
        if not validation.passed:
            raise ValueError(
                "independent Curated v1 validation failed: "
                + "; ".join(validation.errors)
            )

        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return destination.resolve()


def _trajectory(
    transitions: Sequence[CausalTransition],
) -> dict[str, np.ndarray]:
    return {
        "source_tick_index": _int_array(
            transition.source_tick_index for transition in transitions
        ),
        "next_state_tick_index": _int_array(
            transition.next_state_tick_index for transition in transitions
        ),
        "state_timestamp_ns": _int_array(
            max(transition.arm_pre_timestamp_ns, transition.hand_pre_timestamp_ns)
            for transition in transitions
        ),
        "action_timestamp_ns": _int_array(
            transition.action_end_ns for transition in transitions
        ),
        "next_state_timestamp_ns": _int_array(
            min(transition.arm_post_timestamp_ns, transition.hand_post_timestamp_ns)
            for transition in transitions
        ),
        "arm_qpos_source_timestamp_ns": _int_array(
            transition.arm_pre_timestamp_ns for transition in transitions
        ),
        "hand_qpos_source_timestamp_ns": _int_array(
            transition.hand_pre_timestamp_ns for transition in transitions
        ),
        "arm_qcmd_source_timestamp_ns": _int_array(
            transition.arm_action_timestamp_ns for transition in transitions
        ),
        "hand_qcmd_source_timestamp_ns": _int_array(
            transition.hand_action_timestamp_ns for transition in transitions
        ),
        "robot_qpos_17d_rad": np.stack(
            [transition.state17 for transition in transitions]
        ).astype(np.float64, copy=False),
        "robot_qcmd_17d_rad": np.stack(
            [transition.action17 for transition in transitions]
        ).astype(np.float64, copy=False),
        "head_rgb_frame_index": _int_array(
            transition.head_rgb_frame_index for transition in transitions
        ),
        "right_wrist_rgb_frame_index": _int_array(
            transition.wrist_rgb_frame_index for transition in transitions
        ),
    }


def _metadata(
    raw: RawEpisode,
    transitions: Sequence[CausalTransition],
    *,
    expert_training_status: str,
) -> dict[str, object]:
    source_schema_name = _optional_scalar(raw, "schema_name", "UNKNOWN")
    source_schema_version = _optional_scalar(raw, "schema_version", -1)
    language = _optional_scalar(raw, "language_instruction", "")
    dataset_hz = _optional_scalar(raw, "dataset_hz", 0.0)
    hardware_execution = bool(_optional_scalar(raw, "hardware_execution", False))

    return {
        "schema_name": CURATED_SCHEMA_NAME,
        "schema_version": CURATED_SCHEMA_VERSION,
        "episode_id": raw.episode_id,
        "language_instruction": str(language),
        "dataset_hz": float(dataset_hz),
        "joint_names": list(ROBOT_JOINT_NAMES),
        "arm_slice": list(ARM_SLICE),
        "hand_slice": list(HAND_SLICE),
        "camera_roles": ["head", "right_wrist"],
        "source_schema_name": str(source_schema_name),
        "source_schema_version": int(source_schema_version),
        "source_episode_path": str(raw.path),
        # Source-level historical provenance only; D2 readiness is evidence-based.
        "hardware_execution": hardware_execution,
        "training_ready": True,
        "dataset_status": CURATED_DATASET_STATUS_READY,
        "action_representation": ACTION_REPRESENTATION,
        "action_unit": ACTION_UNIT,
        "state_representation": STATE_REPRESENTATION,
        "state_unit": STATE_UNIT,
        "segment_offsets": _segment_offsets(transitions),
        "expert_training_status": expert_training_status,
        "source_dataset_status": str(
            _optional_scalar(raw, "dataset_status", "UNKNOWN")
        ),
        "source_training_ready": bool(_optional_scalar(raw, "training_ready", False)),
        "source_sg100_feedback_map_status": str(
            _optional_scalar(raw, "sg100_feedback_map_status", "UNKNOWN")
        ),
    }


def _cleaning_report(
    raw: RawEpisode,
    sync: CausalSyncResult,
    trajectory: dict[str, np.ndarray],
    *,
    expert_training_status: str,
) -> dict[str, object]:
    transitions = sync.transitions
    metrics = {
        "state_component_skew_ns": [
            abs(t.arm_pre_timestamp_ns - t.hand_pre_timestamp_ns) for t in transitions
        ],
        "action_component_skew_ns": [
            abs(t.arm_action_timestamp_ns - t.hand_action_timestamp_ns)
            for t in transitions
        ],
        "post_state_component_skew_ns": [
            abs(t.arm_post_timestamp_ns - t.hand_post_timestamp_ns) for t in transitions
        ],
        "arm_state_age_at_action_ns": [
            t.action_start_ns - t.arm_pre_timestamp_ns for t in transitions
        ],
        "hand_state_age_at_action_ns": [
            t.action_start_ns - t.hand_pre_timestamp_ns for t in transitions
        ],
        "head_frame_age_at_action_ns": [
            t.action_start_ns - t.head_timestamp_ns for t in transitions
        ],
        "wrist_frame_age_at_action_ns": [
            t.action_start_ns - t.wrist_timestamp_ns for t in transitions
        ],
        "head_wrist_frame_skew_ns": [
            abs(t.head_timestamp_ns - t.wrist_timestamp_ns) for t in transitions
        ],
    }
    return {
        "curated_schema_name": CURATED_SCHEMA_NAME,
        "curated_schema_version": CURATED_SCHEMA_VERSION,
        "source_episode_path": str(raw.path),
        "source_provenance": {
            "hardware_execution": bool(
                _optional_scalar(raw, "hardware_execution", False)
            ),
            "training_ready": bool(_optional_scalar(raw, "training_ready", False)),
            "dataset_status": str(_optional_scalar(raw, "dataset_status", "UNKNOWN")),
            "sg100_feedback_map_status": str(
                _optional_scalar(raw, "sg100_feedback_map_status", "UNKNOWN")
            ),
        },
        "authority": {
            "state_arm": "arm_qpos_rad measured feedback",
            "state_hand": "hand_feedback_sdk_rad measured mode7 feedback",
            "action": "robot_qcmd_17d_rad exact effective command",
            "excluded_state": "hand_qpos_canonical_rad",
            "excluded_action": "desired/retargeted commands",
        },
        "expert_training_status": expert_training_status,
        "candidate_count": sync.candidate_count,
        "accepted_count": len(transitions),
        "rejected_count": len(sync.rejected),
        "rejection_counts": sync.rejection_counts,
        "rejection_group_counts": sync.rejection_group_counts,
        "trajectory_shapes": {
            key: list(value.shape) for key, value in sorted(trajectory.items())
        },
        "segment_offsets": _segment_offsets(transitions),
        "segment_count": len(_segment_offsets(transitions)) - 1,
        "total_media_references": {
            "head": len(transitions),
            "right_wrist": len(transitions),
        },
        "unique_media_references": {
            "head": len({t.head_rgb_frame_index for t in transitions}),
            "right_wrist": len({t.wrist_rgb_frame_index for t in transitions}),
        },
        "synchronization_metrics": {
            key: _summary(values) for key, values in metrics.items()
        },
        # Auxiliary D3 input. These values are deliberately kept out of the
        # training trajectory and preserve per-transition thresholdability.
        "transition_temporal_metrics": metrics,
    }


def _copy_selected_media(
    raw: RawEpisode,
    staging: Path,
    transitions: Sequence[CausalTransition],
) -> None:
    roles = (
        ("head", "head", {t.head_rgb_frame_index for t in transitions}),
        (
            "right_wrist",
            "right_wrist",
            {t.wrist_rgb_frame_index for t in transitions},
        ),
    )
    for source_role, target_role, indices in roles:
        target_dir = staging / "media" / target_role / "rgb"
        target_dir.mkdir(parents=True)
        for index in sorted(indices):
            source = raw.media_root / source_role / "rgb" / f"{index:06d}.jpg"
            _validate_rgb_jpeg(source)
            shutil.copy2(source, target_dir / source.name)


def _validate_rgb_jpeg(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.convert("RGB").getbands() != ("R", "G", "B"):
                raise ValueError(f"media cannot be decoded as RGB: {path}")
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid RGB JPEG: {path}") from exc


def _segment_offsets(transitions: Sequence[CausalTransition]) -> list[int]:
    if not transitions:
        return []
    offsets = [0]
    for index in range(1, len(transitions)):
        if (
            transitions[index].source_tick_index
            != transitions[index - 1].next_state_tick_index
        ):
            offsets.append(index)
    offsets.append(len(transitions))
    return offsets


def _summary(values: Sequence[int]) -> dict[str, int | float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": int(np.max(array)),
    }


def _int_array(values) -> np.ndarray:
    return np.fromiter(values, dtype=np.int64)


def _optional_scalar(raw: RawEpisode, key: str, default: object) -> object:
    if key not in raw.arrays:
        return default
    return raw.scalar(key)


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_path", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument(
        "--expert-training-status",
        default="REVIEW_REQUIRED",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = build_curated_v1(
        args.raw_path,
        args.output_root,
        media_root=args.media_root,
        expert_training_status=args.expert_training_status,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
