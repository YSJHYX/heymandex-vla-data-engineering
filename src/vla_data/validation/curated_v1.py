"""Independent read-only validator for RM65 + SG100 Curated v1 episodes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from vla_data.contracts.curated_v1 import (
    ACTION_REPRESENTATION,
    ACTION_UNIT,
    CURATED_DATASET_STATUS_NOT_READY,
    CURATED_DATASET_STATUS_READY,
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
from vla_data.io.curated_episode import CuratedEpisode
from vla_data.morphology.rm65_sg100 import (
    ARM_SLICE,
    HAND_SLICE,
    ROBOT_JOINT_NAMES,
)


@dataclass(frozen=True)
class CuratedValidationReport:
    """Result of validating one persisted Curated v1 episode."""

    quality: str
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    transition_count: int

    @property
    def passed(self) -> bool:
        return self.quality in {"PASS", "WARN"}


def validate_curated_episode(
    episode: CuratedEpisode,
    *,
    allow_empty: bool = False,
) -> CuratedValidationReport:
    """Validate Curated v1 schema, timing, trajectory, and media integrity."""

    metadata = episode.metadata
    trajectory = episode.trajectory

    errors: list[str] = []
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # Metadata contract
    # ------------------------------------------------------------------

    if metadata.get("schema_name") != CURATED_SCHEMA_NAME:
        errors.append(f"invalid schema_name: {metadata.get('schema_name')!r}")

    if metadata.get("schema_version") != CURATED_SCHEMA_VERSION:
        errors.append(f"invalid schema_version: {metadata.get('schema_version')!r}")

    # D10.1: these are provenance/summaries, not authority for a training field.
    diagnostic_keys = {
        "language_instruction",
        "source_schema_name",
        "source_schema_version",
        "source_episode_path",
        "hardware_execution",
        "training_ready",
        "dataset_status",
    }
    missing_metadata = sorted(
        key
        for key in METADATA_REQUIRED_KEYS
        if key not in metadata and key not in diagnostic_keys
    )
    if missing_metadata:
        errors.append(f"missing metadata keys: {missing_metadata}")
    missing_diagnostics = sorted(diagnostic_keys - metadata.keys())
    if missing_diagnostics:
        warnings.append(
            f"DIAGNOSTIC_WARNING: missing provenance: {missing_diagnostics}"
        )

    joint_names = tuple(metadata.get("joint_names", ()))
    if joint_names != ROBOT_JOINT_NAMES:
        errors.append("joint_names do not match canonical 17D ordering")

    if tuple(metadata.get("arm_slice", ())) != ARM_SLICE:
        errors.append(f"arm_slice must be {ARM_SLICE}")

    if tuple(metadata.get("hand_slice", ())) != HAND_SLICE:
        errors.append(f"hand_slice must be {HAND_SLICE}")

    camera_roles = tuple(metadata.get("camera_roles", ()))
    allowed_camera_roles = {"head", "right_wrist"}

    if "head" not in camera_roles:
        errors.append("camera_roles must include head")

    unknown_camera_roles = set(camera_roles) - allowed_camera_roles
    if unknown_camera_roles:
        errors.append(f"unsupported camera roles: {sorted(unknown_camera_roles)}")

    if metadata.get("action_representation") != ACTION_REPRESENTATION:
        errors.append("invalid action_representation")

    if metadata.get("action_unit") != ACTION_UNIT:
        errors.append("invalid action_unit")

    if metadata.get("state_representation") != STATE_REPRESENTATION:
        errors.append("invalid state_representation")

    if metadata.get("state_unit") != STATE_UNIT:
        errors.append("invalid state_unit")

    training_ready = metadata.get("training_ready")
    dataset_status = metadata.get("dataset_status")

    if training_ready is True:
        if dataset_status != CURATED_DATASET_STATUS_READY:
            warnings.append("DIAGNOSTIC_WARNING: training_ready/status disagree")
    elif training_ready is False:
        if dataset_status != CURATED_DATASET_STATUS_NOT_READY:
            warnings.append("DIAGNOSTIC_WARNING: training_ready/status disagree")
    else:
        warnings.append("DIAGNOSTIC_WARNING: training_ready is not boolean")

    # ------------------------------------------------------------------
    # Trajectory field contract
    # ------------------------------------------------------------------

    present = set(trajectory)

    missing_trajectory = sorted(set(TRAJECTORY_CORE_KEYS) - present)
    if missing_trajectory:
        errors.append(f"missing trajectory keys: {missing_trajectory}")

    allowed_keys = set(TRAJECTORY_CORE_KEYS) | set(TRAJECTORY_OPTIONAL_KEYS)

    unknown_keys = sorted(present - allowed_keys)
    if unknown_keys:
        errors.append(f"unknown trajectory keys: {unknown_keys}")

    forbidden_keys = sorted(present & FORBIDDEN_TRAJECTORY_KEYS)
    if forbidden_keys:
        errors.append(f"forbidden trajectory keys: {forbidden_keys}")

    # Do not continue field-dependent checks when core data is absent.
    if missing_trajectory:
        return _report(
            errors=errors,
            warnings=warnings,
            transition_count=0,
        )

    # ------------------------------------------------------------------
    # Shapes and common transition length
    # ------------------------------------------------------------------

    lengths: dict[str, int] = {}

    for key in sorted(present):
        array = np.asarray(trajectory[key])

        if key in TRAJECTORY_VECTOR_KEYS:
            expected_width = TRAJECTORY_VECTOR_KEYS[key]

            if array.ndim != 2 or array.shape[1] != expected_width:
                errors.append(
                    f"{key} shape={array.shape} expected=(M, {expected_width})"
                )
                continue

            lengths[key] = int(array.shape[0])

        else:
            if array.ndim != 1:
                errors.append(f"{key} must be one-dimensional, got {array.shape}")
                continue

            lengths[key] = int(array.shape[0])

    length_values = set(lengths.values())

    if len(length_values) > 1:
        errors.append(f"trajectory row counts disagree: {sorted(length_values)}")
        transition_count = 0
    elif length_values:
        transition_count = next(iter(length_values))
    else:
        transition_count = 0

    if transition_count == 0 and not allow_empty:
        errors.append("trajectory contains no transitions")

    # Shape errors may make later indexing unsafe.
    required_shapes_valid = all(key in lengths for key in TRAJECTORY_CORE_KEYS)

    if not required_shapes_valid:
        return _report(
            errors=errors,
            warnings=warnings,
            transition_count=transition_count,
        )

    m = transition_count

    # ------------------------------------------------------------------
    # Segment contract
    # ------------------------------------------------------------------

    offsets = tuple(int(value) for value in metadata.get("segment_offsets", ()))

    if m == 0:
        if offsets:
            errors.append("empty trajectory requires empty segment_offsets")
    else:
        if not offsets:
            errors.append("segment_offsets are missing")
        elif (
            offsets[0] != 0
            or offsets[-1] != m
            or any(later <= earlier for earlier, later in pairwise(offsets))
        ):
            errors.append(
                "segment_offsets must start at 0, end at M, and be strictly increasing"
            )

    boundaries = set(offsets[:-1]) if offsets else set()

    # ------------------------------------------------------------------
    # Tick relationship
    # ------------------------------------------------------------------

    source_tick = np.asarray(trajectory["source_tick_index"], dtype=np.int64)
    next_tick = np.asarray(trajectory["next_state_tick_index"], dtype=np.int64)

    if not np.all(next_tick == source_tick + 1):
        errors.append("next_state_tick_index must equal source_tick_index + 1")

    if m > 1 and np.any(np.diff(source_tick) <= 0):
        errors.append("source_tick_index must be strictly increasing")

    for index in range(1, m):
        continuous = source_tick[index] == next_tick[index - 1]

        if not continuous and index not in boundaries:
            errors.append(f"tick discontinuity must split segment at row {index}")

    # ------------------------------------------------------------------
    # Native timestamps and causality
    # ------------------------------------------------------------------

    arm_state_ts = np.asarray(
        trajectory["arm_qpos_source_timestamp_ns"],
        dtype=np.int64,
    )
    hand_state_ts = np.asarray(
        trajectory["hand_qpos_source_timestamp_ns"],
        dtype=np.int64,
    )
    arm_action_ts = np.asarray(
        trajectory["arm_qcmd_source_timestamp_ns"],
        dtype=np.int64,
    )
    hand_action_ts = np.asarray(
        trajectory["hand_qcmd_source_timestamp_ns"],
        dtype=np.int64,
    )

    native_timestamps = (
        arm_state_ts,
        hand_state_ts,
        arm_action_ts,
        hand_action_ts,
    )

    if any(np.any(values <= 0) for values in native_timestamps):
        errors.append("native source timestamps must all be > 0")

    state_ts = np.asarray(trajectory["state_timestamp_ns"], dtype=np.int64)
    action_ts = np.asarray(trajectory["action_timestamp_ns"], dtype=np.int64)
    next_state_ts = np.asarray(trajectory["next_state_timestamp_ns"], dtype=np.int64)

    expected_state_ts = np.maximum(
        arm_state_ts,
        hand_state_ts,
    )
    if not np.array_equal(state_ts, expected_state_ts):
        errors.append("state_timestamp_ns must equal max(arm_state, hand_state)")

    expected_action_ts = np.maximum(
        arm_action_ts,
        hand_action_ts,
    )
    if not np.array_equal(action_ts, expected_action_ts):
        errors.append("action_timestamp_ns must equal max(arm_action, hand_action)")

    action_start_ts = np.minimum(arm_action_ts, hand_action_ts)
    if np.any(state_ts >= action_start_ts):
        errors.append(
            "PRE causality requires max(component state timestamps) "
            "< min(component action timestamps)"
        )

    if np.any(action_ts >= next_state_ts):
        errors.append("unified causality requires action < next_state")

    # ------------------------------------------------------------------
    # Physical state/action integrity
    # ------------------------------------------------------------------

    for key in (
        "robot_qpos_17d_rad",
        "robot_qcmd_17d_rad",
    ):
        values = np.asarray(trajectory[key], dtype=np.float64)

        if not np.all(np.isfinite(values)):
            errors.append(f"{key} contains non-finite values")

    # ------------------------------------------------------------------
    # Camera frame references
    # ------------------------------------------------------------------

    head_indices = np.asarray(
        trajectory["head_rgb_frame_index"],
        dtype=np.int64,
    )

    if np.any(head_indices < 0):
        errors.append("head_rgb_frame_index must be >= 0")

    wrist_indices = np.asarray(
        trajectory["right_wrist_rgb_frame_index"],
        dtype=np.int64,
    )

    wrist_enabled = "right_wrist" in camera_roles

    if wrist_enabled:
        if np.any(wrist_indices < 0):
            errors.append(
                "right_wrist frame indices must be >= 0 when wrist camera is enabled"
            )
    elif np.any(wrist_indices != -1):
        errors.append(
            "right_wrist frame indices must be -1 when wrist camera is disabled"
        )

    # ------------------------------------------------------------------
    # External JPG media integrity
    # ------------------------------------------------------------------

    _validate_media_role(
        episode=episode,
        role="head",
        indices=head_indices,
        errors=errors,
    )

    if wrist_enabled:
        _validate_media_role(
            episode=episode,
            role="right_wrist",
            indices=wrist_indices,
            errors=errors,
        )
    else:
        wrist_role_dir = episode.episode_dir / "media" / "right_wrist"
        if wrist_role_dir.exists():
            errors.append(
                "right_wrist media directory exists but camera role is disabled"
            )

    for role in camera_roles:
        depth_dir = episode.episode_dir / "media" / role / "depth"

        if depth_dir.exists():
            errors.append(f"Curated v1 forbids depth directory: {role}/depth")

    return _report(
        errors=errors,
        warnings=warnings,
        transition_count=m,
    )


def _validate_media_role(
    *,
    episode: CuratedEpisode,
    role: str,
    indices: np.ndarray,
    errors: list[str],
) -> None:
    referenced = {int(value) for value in indices.tolist() if int(value) >= 0}

    try:
        existing = set(episode.media.existing_rgb_indices(role))
    except ValueError as exc:
        errors.append(str(exc))
        return

    if referenced != existing:
        errors.append(
            f"{role}/rgb reference mismatch: "
            f"referenced={sorted(referenced)} "
            f"files={sorted(existing)}"
        )


def _report(
    *,
    errors: list[str],
    warnings: list[str],
    transition_count: int,
) -> CuratedValidationReport:
    quality = "FAIL" if errors else ("WARN" if warnings else "PASS")

    return CuratedValidationReport(
        quality=quality,
        errors=tuple(errors),
        warnings=tuple(warnings),
        transition_count=transition_count,
    )
