from vla_data.contracts.curated_v1 import (
    ACTION_REPRESENTATION,
    ACTION_UNIT,
    CURATED_SCHEMA_NAME,
    CURATED_SCHEMA_VERSION,
    FORBIDDEN_TRAJECTORY_KEYS,
    METADATA_REQUIRED_KEYS,
    STATE_REPRESENTATION,
    STATE_UNIT,
    TRAJECTORY_CORE_KEYS,
    TRAJECTORY_OPTIONAL_KEYS,
    TRAJECTORY_VECTOR_KEYS,
)


def test_schema_identity() -> None:
    assert CURATED_SCHEMA_NAME == "rm65_sg100_dual_realsense_vla_curated"
    assert CURATED_SCHEMA_VERSION == 1


def test_state_action_semantics() -> None:
    assert STATE_REPRESENTATION == "REAL_MEASURED_HARDWARE_JOINT_FEEDBACK"
    assert ACTION_REPRESENTATION == "FINAL_EFFECTIVE_BACKEND_COMMAND"

    assert STATE_UNIT == "rad"
    assert ACTION_UNIT == "rad"


def test_physical_vectors_are_17d() -> None:
    assert TRAJECTORY_VECTOR_KEYS["robot_qpos_17d_rad"] == 17
    assert TRAJECTORY_VECTOR_KEYS["robot_qcmd_17d_rad"] == 17
    assert "robot_desired_qcmd_17d_rad" not in TRAJECTORY_VECTOR_KEYS


def test_core_state_action_fields_exist() -> None:
    assert "robot_qpos_17d_rad" in TRAJECTORY_CORE_KEYS
    assert "robot_qcmd_17d_rad" in TRAJECTORY_CORE_KEYS

    assert "robot_desired_qcmd_17d_rad" not in TRAJECTORY_CORE_KEYS
    assert "robot_desired_qcmd_17d_rad" not in TRAJECTORY_OPTIONAL_KEYS
    assert "robot_desired_qcmd_17d_rad" in FORBIDDEN_TRAJECTORY_KEYS


def test_external_media_indices_are_core_contract() -> None:
    assert "head_rgb_frame_index" in TRAJECTORY_CORE_KEYS
    assert "right_wrist_rgb_frame_index" in TRAJECTORY_CORE_KEYS


def test_segments_are_metadata_contract() -> None:
    assert "segment_offsets" in METADATA_REQUIRED_KEYS


def test_duplicate_arm_hand_storage_is_forbidden() -> None:
    assert "arm_qpos_6d_rad" in FORBIDDEN_TRAJECTORY_KEYS
    assert "hand_qpos_11d_rad" in FORBIDDEN_TRAJECTORY_KEYS

    assert "arm_qcmd_6d_rad" in FORBIDDEN_TRAJECTORY_KEYS
    assert "hand_qcmd_11d_rad" in FORBIDDEN_TRAJECTORY_KEYS


def test_depth_is_not_part_of_curated_v1() -> None:
    assert "head_depth_frame_index" in FORBIDDEN_TRAJECTORY_KEYS
    assert "wrist_depth_frame_index" in FORBIDDEN_TRAJECTORY_KEYS
