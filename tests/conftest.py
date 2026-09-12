from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vla_data.contracts.curated_v1 import (
    ACTION_REPRESENTATION,
    ACTION_UNIT,
    CURATED_DATASET_STATUS_READY,
    CURATED_SCHEMA_NAME,
    CURATED_SCHEMA_VERSION,
    STATE_REPRESENTATION,
    STATE_UNIT,
)
from vla_data.morphology.rm65_sg100 import (
    ARM_SLICE,
    HAND_SLICE,
    ROBOT_JOINT_NAMES,
)


@pytest.fixture
def synthetic_raw_episode(tmp_path: Path) -> Path:
    path = tmp_path / "episode_000007.npz"
    media_root = tmp_path / "episode_000007_media"
    for role, value in (("head", 40), ("right_wrist", 80)):
        rgb_dir = media_root / role / "rgb"
        rgb_dir.mkdir(parents=True)
        image = np.full((6, 8, 3), value, dtype=np.uint8)
        Image.fromarray(image).save(rgb_dir / "000002.jpg")

    n = 8
    arm_qpos = np.arange(n * 6, dtype=np.float64).reshape(n, 6) / 100.0
    hand_feedback = np.arange(n * 11, dtype=np.float64).reshape(n, 11) / 1000.0
    arm_effective = np.full((n, 6), np.nan, dtype=np.float64)
    hand_effective = np.full((n, 11), np.nan, dtype=np.float64)
    robot_effective = np.full((n, 17), np.nan, dtype=np.float64)
    for row in (3, 4):
        arm_effective[row] = row + np.arange(6) / 10.0
        hand_effective[row] = row + np.arange(11) / 100.0
        robot_effective[row] = np.concatenate((arm_effective[row], hand_effective[row]))

    action_valid = np.zeros(n, dtype=bool)
    action_valid[[3, 4]] = True
    arm_action_ts = np.zeros(n, dtype=np.int64)
    hand_action_ts = np.zeros(n, dtype=np.int64)
    arm_action_ts[[3, 4]] = [30, 40]
    hand_action_ts[[3, 4]] = [32, 42]

    arrays = {
        "timestamp_ns": np.arange(n, dtype=np.int64) + 1,
        "arm_qpos_rad": arm_qpos,
        "arm_qpos_source_timestamp_ns": np.asarray(
            [5, 15, 31, 35, 45, 55, 65, 75], dtype=np.int64
        ),
        "arm_feedback_valid": np.ones(n, dtype=bool),
        "hand_feedback_sdk_rad": hand_feedback,
        "hand_feedback_modes": np.full((n, 11), 7, dtype=np.int64),
        "hand_feedback_source_timestamp_ns": np.asarray(
            [4, 18, 29, 33, 43, 53, 63, 73], dtype=np.int64
        ),
        "hand_feedback_valid": np.ones(n, dtype=bool),
        "arm_qcmd_sent_rad": arm_effective,
        "arm_qcmd_source_timestamp_ns": arm_action_ts,
        "arm_command_valid": action_valid.copy(),
        "hand_qcmd_effective_canonical_rad": hand_effective,
        "hand_qcmd_source_timestamp_ns": hand_action_ts,
        "hand_command_valid": action_valid.copy(),
        "robot_qcmd_17d_rad": robot_effective,
        "robot_qcmd_17d_valid": action_valid.copy(),
        "head_camera_host_timestamp_ns": np.asarray(
            [0, 0, 20, 0, 0, 0, 0, 0], dtype=np.int64
        ),
        "head_camera_valid": np.asarray(
            [False, False, True, False, False, False, False, False]
        ),
        "head_rgb_frame_index": np.asarray(
            [-1, -1, 2, -1, -1, -1, -1, -1], dtype=np.int64
        ),
        "wrist_camera_host_timestamp_ns": np.asarray(
            [0, 0, 19, 0, 0, 0, 0, 0], dtype=np.int64
        ),
        "wrist_camera_valid": np.asarray(
            [False, False, True, False, False, False, False, False]
        ),
        "wrist_rgb_frame_index": np.asarray(
            [-1, -1, 2, -1, -1, -1, -1, -1], dtype=np.int64
        ),
        # Deliberate sentinels: D2 must never use these as authority.
        "hand_qpos_canonical_rad": np.full((n, 11), 999.0),
        "hand_qcmd_retargeted_canonical_rad": np.full((n, 11), 777.0),
        "schema_name": np.asarray("synthetic_raw"),
        "schema_version": np.asarray(3),
        "language_instruction": np.asarray("synthetic task"),
        "dataset_hz": np.asarray(30.0),
        "hardware_execution": np.asarray(False),
        "training_ready": np.asarray(False),
        "dataset_status": np.asarray("RAW_CAPTURE_QUARANTINED"),
        "sg100_feedback_map_status": np.asarray("F_FB_PENDING"),
    }
    np.savez(path, **arrays)
    return path


@pytest.fixture
def curated_v1_episode(tmp_path: Path) -> Path:
    episode_dir = tmp_path / "episode_000123"
    media_root = episode_dir / "media"

    head_dir = media_root / "head" / "rgb"
    wrist_dir = media_root / "right_wrist" / "rgb"

    head_dir.mkdir(parents=True)
    wrist_dir.mkdir(parents=True)

    # Curated media is deduplicated:
    # three transitions reference only two physical JPEGs per camera.
    for index, value in enumerate((32, 96)):
        image = np.full((8, 10, 3), value, dtype=np.uint8)
        Image.fromarray(image).save(head_dir / f"{index:06d}.jpg")

    for index, value in enumerate((64, 128)):
        image = np.full((8, 10, 3), value, dtype=np.uint8)
        Image.fromarray(image).save(wrist_dir / f"{index:06d}.jpg")

    metadata = {
        "schema_name": CURATED_SCHEMA_NAME,
        "schema_version": CURATED_SCHEMA_VERSION,
        "episode_id": "episode_000123",
        "language_instruction": "Pick up the object and place it into the fixture.",
        "dataset_hz": 30.0,
        "joint_names": list(ROBOT_JOINT_NAMES),
        "arm_slice": list(ARM_SLICE),
        "hand_slice": list(HAND_SLICE),
        "camera_roles": ["head", "right_wrist"],
        "source_schema_name": "synthetic_raw_fixture",
        "source_schema_version": 3,
        "source_episode_path": "/synthetic/episode_000123.npz",
        "hardware_execution": True,
        "training_ready": True,
        "dataset_status": CURATED_DATASET_STATUS_READY,
        "action_representation": ACTION_REPRESENTATION,
        "action_unit": ACTION_UNIT,
        "state_representation": STATE_REPRESENTATION,
        "state_unit": STATE_UNIT,
        "segment_offsets": [0, 2, 3],
    }

    with (episode_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)

    state = np.arange(3 * 17, dtype=np.float64).reshape(3, 17) * 0.001
    action = state + 0.01

    trajectory = {
        "source_tick_index": np.asarray([0, 1, 3], dtype=np.int64),
        "next_state_tick_index": np.asarray([1, 2, 4], dtype=np.int64),
        "state_timestamp_ns": np.asarray([105, 205, 405], dtype=np.int64),
        "action_timestamp_ns": np.asarray([115, 215, 415], dtype=np.int64),
        "next_state_timestamp_ns": np.asarray([205, 305, 505], dtype=np.int64),
        "arm_qpos_source_timestamp_ns": np.asarray([100, 200, 400], dtype=np.int64),
        "hand_qpos_source_timestamp_ns": np.asarray([105, 205, 405], dtype=np.int64),
        "arm_qcmd_source_timestamp_ns": np.asarray([110, 210, 410], dtype=np.int64),
        "hand_qcmd_source_timestamp_ns": np.asarray([115, 215, 415], dtype=np.int64),
        "robot_qpos_17d_rad": state,
        "robot_qcmd_17d_rad": action,
        "head_rgb_frame_index": np.asarray([0, 0, 1], dtype=np.int64),
        "right_wrist_rgb_frame_index": np.asarray([0, 1, 1], dtype=np.int64),
    }

    np.savez(episode_dir / "trajectory.npz", **trajectory)

    return episode_dir
