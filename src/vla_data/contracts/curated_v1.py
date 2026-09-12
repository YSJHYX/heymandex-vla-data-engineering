"""RM65 + SG100 Curated VLA trajectory v1 contract.

This module mirrors the persisted Curated v1 data contract produced by
VisionProTeleop, while remaining independent from the teleoperation runtime.

Curated v1 is:
- physical robot data;
- 17-DoF state/action only;
- transition-oriented;
- RGB-only;
- external-media based;
- training-valid-transition oriented.

This repository does not define model-space padding or model architecture.
"""

from __future__ import annotations

from vla_data.morphology.rm65_sg100 import (
    ARM_SLICE,
    HAND_SLICE,
    PHYSICAL_DOF,
)

CURATED_SCHEMA_NAME = "rm65_sg100_dual_realsense_vla_curated"
CURATED_SCHEMA_VERSION = 1

CURATED_DATASET_STATUS_READY = "CURATED_TRAINING_READY"
CURATED_DATASET_STATUS_NOT_READY = "CURATED_NOT_TRAINING_READY"

ACTION_REPRESENTATION = "FINAL_EFFECTIVE_BACKEND_COMMAND"
ACTION_UNIT = "rad"

STATE_REPRESENTATION = "REAL_MEASURED_HARDWARE_JOINT_FEEDBACK"
STATE_UNIT = "rad"


TRAJECTORY_CORE_KEYS: tuple[str, ...] = (
    "source_tick_index",
    "next_state_tick_index",
    "state_timestamp_ns",
    "action_timestamp_ns",
    "next_state_timestamp_ns",
    "arm_qpos_source_timestamp_ns",
    "hand_qpos_source_timestamp_ns",
    "arm_qcmd_source_timestamp_ns",
    "hand_qcmd_source_timestamp_ns",
    "robot_qpos_17d_rad",
    "robot_qcmd_17d_rad",
    "head_rgb_frame_index",
    "right_wrist_rgb_frame_index",
)

TRAJECTORY_OPTIONAL_KEYS: tuple[str, ...] = ()

TRAJECTORY_VECTOR_KEYS: dict[str, int] = {
    "robot_qpos_17d_rad": PHYSICAL_DOF,
    "robot_qcmd_17d_rad": PHYSICAL_DOF,
}


METADATA_REQUIRED_KEYS: tuple[str, ...] = (
    "schema_name",
    "schema_version",
    "episode_id",
    "language_instruction",
    "dataset_hz",
    "joint_names",
    "arm_slice",
    "hand_slice",
    "camera_roles",
    "source_schema_name",
    "source_schema_version",
    "source_episode_path",
    "hardware_execution",
    "training_ready",
    "dataset_status",
    "action_representation",
    "action_unit",
    "state_representation",
    "state_unit",
    "segment_offsets",
)


FORBIDDEN_TRAJECTORY_KEYS = frozenset(
    {
        # Duplicate decomposed physical representations.
        "arm_qcmd_6d_rad",
        "hand_qcmd_11d_rad",
        "arm_qpos_6d_rad",
        "hand_qpos_11d_rad",
        "arm_replay",
        "hand_replay",
        # Raw component / SDK / wire-domain data.
        "arm_q_goal_rad",
        "arm_qcmd_sent_rad",
        "arm_qpos_rad",
        "hand_qcmd_retargeted_canonical_rad",
        "hand_qcmd_effective_canonical_rad",
        "hand_qcmd_sdk_deg",
        "hand_feedback_sdk_rad",
        "hand_feedback_modes",
        "hand_qpos_canonical_rad",
        "robot_desired_qcmd_17d_rad",
        # Telemetry / writer / sampler diagnostics.
        "telemetry_publish_sequence",
        "telemetry_age_ns",
        "telemetry_valid",
        "telemetry_publish_timestamp_ns",
        "writer_backlog",
        "writer_drop_count",
        "media_writer_drop_count",
        "media_writer_error_count",
        "sampler_wakeup_timestamp_ns",
        "sampler_jitter_ns",
        "arm_qcmd_sequence",
        "arm_feedback_sequence",
        "hand_command_sequence",
        "hand_feedback_sequence",
        # Derived timing / quality data.
        "timestamp_relative_ns",
        "arm_hand_state_skew_ns",
        "arm_hand_action_skew_ns",
        "state_action_dt_ns",
        "action_next_state_dt_ns",
        "desired_effective_l2",
        "desired_effective_max_abs",
        "arm_command_dt_mean_ns",
        "hand_feedback_dt_p95_ns",
        # Runtime camera diagnostics.
        "head_camera_host_timestamp_ns",
        "head_camera_device_timestamp_ms",
        "head_camera_frame_number",
        "head_camera_age_ns",
        "head_camera_valid",
        "wrist_camera_host_timestamp_ns",
        "wrist_camera_device_timestamp_ms",
        "wrist_camera_frame_number",
        "wrist_camera_age_ns",
        "wrist_camera_valid",
        # Curated contains valid transitions only.
        "transition_valid",
        "invalid_reason_mask",
        # Curated v1 is RGB-only.
        "head_depth_frame_index",
        "wrist_depth_frame_index",
        "head_depth_valid",
        "wrist_depth_valid",
        # Deliberately not persisted in Curated v1.
        "next_arm_qpos_source_timestamp_ns",
        "next_hand_qpos_source_timestamp_ns",
        "next_state_source_timestamp_ns",
    }
)


INVALID_REASONS: tuple[str, ...] = (
    "STATE_QPOS_INVALID",
    "ACTION_QCMD_INVALID",
    "NEXT_STATE_QPOS_INVALID",
    "TICK_TIMESTAMP_ORDER_INVALID",
    "ARM_QPOS_SOURCE_TIMESTAMP_INVALID",
    "HAND_QPOS_SOURCE_TIMESTAMP_INVALID",
    "ARM_QCMD_SOURCE_TIMESTAMP_INVALID",
    "HAND_QCMD_SOURCE_TIMESTAMP_INVALID",
    "ARM_CAUSALITY_INVALID",
    "HAND_CAUSALITY_INVALID",
    "UNIFIED_CAUSALITY_INVALID",
    "HEAD_RGB_INVALID",
    "HEAD_RGB_MEDIA_MISSING",
    "WRIST_RGB_INVALID",
    "WRIST_RGB_MEDIA_MISSING",
)

INVALID_REASON_BITS: dict[str, int] = {
    name: 1 << index for index, name in enumerate(INVALID_REASONS)
}


def valid_arm_slice() -> tuple[int, int]:
    return ARM_SLICE


def valid_hand_slice() -> tuple[int, int]:
    return HAND_SLICE
