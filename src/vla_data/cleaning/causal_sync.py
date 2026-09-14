"""Strict native-timestamp synchronization for Curated v1 transitions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from vla_data.io.raw_episode import RawEpisode


@dataclass(frozen=True)
class CausalTransition:
    """One causally synchronized physical transition."""

    source_tick_index: int
    next_state_tick_index: int
    arm_pre_index: int
    hand_pre_index: int
    arm_post_index: int
    hand_post_index: int
    head_rgb_frame_index: int
    wrist_rgb_frame_index: int
    arm_pre_timestamp_ns: int
    hand_pre_timestamp_ns: int
    arm_action_timestamp_ns: int
    hand_action_timestamp_ns: int
    arm_post_timestamp_ns: int
    hand_post_timestamp_ns: int
    head_timestamp_ns: int
    wrist_timestamp_ns: int
    state17: np.ndarray
    action17: np.ndarray
    next_state17: np.ndarray

    @property
    def action_start_ns(self) -> int:
        return min(self.arm_action_timestamp_ns, self.hand_action_timestamp_ns)

    @property
    def action_end_ns(self) -> int:
        return max(self.arm_action_timestamp_ns, self.hand_action_timestamp_ns)


@dataclass(frozen=True)
class RejectedCandidate:
    source_tick_index: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CausalSyncResult:
    candidate_count: int
    transitions: tuple[CausalTransition, ...]
    rejected: tuple[RejectedCandidate, ...]
    rejection_counts: dict[str, int]
    rejection_group_counts: dict[str, int]


REQUIRED_FIELDS: tuple[str, ...] = (
    "timestamp_ns",
    "arm_qpos_rad",
    "arm_qpos_source_timestamp_ns",
    "arm_feedback_valid",
    "hand_feedback_sdk_rad",
    "hand_feedback_modes",
    "hand_feedback_source_timestamp_ns",
    "hand_feedback_valid",
    "arm_qcmd_sent_rad",
    "arm_qcmd_source_timestamp_ns",
    "arm_command_valid",
    "hand_qcmd_effective_canonical_rad",
    "hand_qcmd_source_timestamp_ns",
    "hand_command_valid",
    "robot_qcmd_17d_rad",
    "robot_qcmd_17d_valid",
    "head_camera_host_timestamp_ns",
    "head_camera_valid",
    "head_rgb_frame_index",
    "wrist_camera_host_timestamp_ns",
    "wrist_camera_valid",
    "wrist_rgb_frame_index",
)


def synchronize_episode(raw: RawEpisode) -> CausalSyncResult:
    """Build strict PRE -> ACTION -> POST transitions from one RAW episode.

    State authority is measured hardware evidence only: RM65 ``arm_qpos_rad``
    plus SG100 mode-7 ``hand_feedback_sdk_rad``. Action authority is the
    recorded final effective ``robot_qcmd_17d_rad``, accepted only when it is
    exactly equal to the recorded arm/hand effective component concatenation.

    Component validity certifies recorder acceptance/freshness, not generic
    device health. Finite held values can be STALE; mode != 7 is
    SEMANTICALLY_INVALID; timestamp ordering failures are CAUSALLY_INVALID.
    Diagnostic status/rate/readiness fields never override these authorities.
    """

    fields = {key: raw.require(key) for key in REQUIRED_FIELDS}
    n = _validate_raw_shapes(fields)

    arm_feedback_valid = (
        np.asarray(fields["arm_feedback_valid"], dtype=bool)
        & (fields["arm_qpos_source_timestamp_ns"] > 0)
        & np.all(np.isfinite(fields["arm_qpos_rad"]), axis=1)
    )
    hand_feedback_valid = (
        np.asarray(fields["hand_feedback_valid"], dtype=bool)
        & (fields["hand_feedback_source_timestamp_ns"] > 0)
        & np.all(fields["hand_feedback_modes"] == 7, axis=1)
        & np.all(np.isfinite(fields["hand_feedback_sdk_rad"]), axis=1)
    )

    transitions: list[CausalTransition] = []
    rejected: list[RejectedCandidate] = []
    reason_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    media_integrity: dict[Path, bool] = {}

    # The persisted transition contract keeps next_state_tick_index=k+1.
    # Native POST component samples may originate at different RAW row indices.
    for tick in range(max(0, n - 1)):
        reasons: list[str] = []

        arm_action_ts = int(fields["arm_qcmd_source_timestamp_ns"][tick])
        hand_action_ts = int(fields["hand_qcmd_source_timestamp_ns"][tick])
        arm_action = np.asarray(fields["arm_qcmd_sent_rad"][tick])
        hand_action = np.asarray(fields["hand_qcmd_effective_canonical_rad"][tick])
        recorded_action = np.asarray(fields["robot_qcmd_17d_rad"][tick])

        if not bool(fields["arm_command_valid"][tick]):
            reasons.append("ACTION_ARM_COMMAND_FLAG_INVALID")
        if not bool(fields["hand_command_valid"][tick]):
            reasons.append("ACTION_HAND_COMMAND_FLAG_INVALID")
        if not bool(fields["robot_qcmd_17d_valid"][tick]):
            reasons.append("ACTION_ROBOT_COMMAND_FLAG_INVALID")
        if arm_action_ts <= 0 or hand_action_ts <= 0:
            reasons.append("ACTION_TIMESTAMP_NONPOSITIVE")
        action_is_finite = (
            np.all(np.isfinite(arm_action))
            and np.all(np.isfinite(hand_action))
            and np.all(np.isfinite(recorded_action))
        )
        if not action_is_finite:
            reasons.append("ACTION_NONFINITE")

        expected_action = np.concatenate((arm_action, hand_action))
        if action_is_finite and not np.array_equal(recorded_action, expected_action):
            reasons.append("ACTION_COMPONENT_MISMATCH")

        if reasons:
            _record_rejection(tick, reasons, rejected, reason_counts, group_counts)
            continue

        action_start = min(arm_action_ts, hand_action_ts)
        action_end = max(arm_action_ts, hand_action_ts)

        arm_pre = _select_feedback(
            fields["arm_qpos_source_timestamp_ns"],
            arm_feedback_valid,
            boundary=action_start,
            before=True,
            component="ARM_PRE",
            reasons=reasons,
        )
        hand_pre = _select_feedback(
            fields["hand_feedback_source_timestamp_ns"],
            hand_feedback_valid,
            boundary=action_start,
            before=True,
            component="HAND_PRE",
            reasons=reasons,
        )
        arm_post = _select_feedback(
            fields["arm_qpos_source_timestamp_ns"],
            arm_feedback_valid,
            boundary=action_end,
            before=False,
            component="ARM_POST",
            reasons=reasons,
        )
        hand_post = _select_feedback(
            fields["hand_feedback_source_timestamp_ns"],
            hand_feedback_valid,
            boundary=action_end,
            before=False,
            component="HAND_POST",
            reasons=reasons,
        )

        head_selection = _select_camera(
            timestamps=fields["head_camera_host_timestamp_ns"],
            valid=fields["head_camera_valid"],
            frame_indices=fields["head_rgb_frame_index"],
            boundary=action_start,
            media_dir=raw.media_root / "head" / "rgb",
            component="HEAD_CAMERA",
            reasons=reasons,
            media_integrity=media_integrity,
        )
        wrist_selection = _select_camera(
            timestamps=fields["wrist_camera_host_timestamp_ns"],
            valid=fields["wrist_camera_valid"],
            frame_indices=fields["wrist_rgb_frame_index"],
            boundary=action_start,
            media_dir=raw.media_root / "right_wrist" / "rgb",
            component="WRIST_CAMERA",
            reasons=reasons,
            media_integrity=media_integrity,
        )

        if reasons:
            _record_rejection(tick, reasons, rejected, reason_counts, group_counts)
            continue

        assert arm_pre is not None
        assert hand_pre is not None
        assert arm_post is not None
        assert hand_post is not None
        assert head_selection is not None
        assert wrist_selection is not None
        head_row, head_index = head_selection
        wrist_row, wrist_index = wrist_selection

        arm_pre_ts = int(fields["arm_qpos_source_timestamp_ns"][arm_pre])
        hand_pre_ts = int(fields["hand_feedback_source_timestamp_ns"][hand_pre])
        arm_post_ts = int(fields["arm_qpos_source_timestamp_ns"][arm_post])
        hand_post_ts = int(fields["hand_feedback_source_timestamp_ns"][hand_post])

        if max(arm_pre_ts, hand_pre_ts) >= action_start:
            reasons.append("CAUSALITY_PRE_NOT_STRICT")
        if action_end >= min(arm_post_ts, hand_post_ts):
            reasons.append("CAUSALITY_POST_NOT_STRICT")

        if reasons:
            _record_rejection(tick, reasons, rejected, reason_counts, group_counts)
            continue

        state = np.concatenate(
            (
                np.asarray(fields["arm_qpos_rad"][arm_pre], dtype=np.float64),
                np.asarray(fields["hand_feedback_sdk_rad"][hand_pre], dtype=np.float64),
            )
        )
        action = np.asarray(recorded_action, dtype=np.float64).copy()
        next_state = np.concatenate(
            (
                np.asarray(fields["arm_qpos_rad"][arm_post], dtype=np.float64),
                np.asarray(
                    fields["hand_feedback_sdk_rad"][hand_post], dtype=np.float64
                ),
            )
        )
        state.setflags(write=False)
        action.setflags(write=False)
        next_state.setflags(write=False)

        transitions.append(
            CausalTransition(
                source_tick_index=tick,
                next_state_tick_index=tick + 1,
                arm_pre_index=arm_pre,
                hand_pre_index=hand_pre,
                arm_post_index=arm_post,
                hand_post_index=hand_post,
                head_rgb_frame_index=head_index,
                wrist_rgb_frame_index=wrist_index,
                arm_pre_timestamp_ns=arm_pre_ts,
                hand_pre_timestamp_ns=hand_pre_ts,
                arm_action_timestamp_ns=arm_action_ts,
                hand_action_timestamp_ns=hand_action_ts,
                arm_post_timestamp_ns=arm_post_ts,
                hand_post_timestamp_ns=hand_post_ts,
                head_timestamp_ns=int(
                    fields["head_camera_host_timestamp_ns"][head_row]
                ),
                wrist_timestamp_ns=int(
                    fields["wrist_camera_host_timestamp_ns"][wrist_row]
                ),
                state17=state,
                action17=action,
                next_state17=next_state,
            )
        )

    return CausalSyncResult(
        candidate_count=max(0, n - 1),
        transitions=tuple(transitions),
        rejected=tuple(rejected),
        rejection_counts=dict(sorted(reason_counts.items())),
        rejection_group_counts={
            group: int(group_counts.get(group, 0))
            for group in ("ACTION", "FEEDBACK", "CAUSALITY", "CAMERA")
        },
    )


def _validate_raw_shapes(fields: dict[str, np.ndarray]) -> int:
    timestamp = fields["timestamp_ns"]
    if timestamp.ndim != 1:
        raise ValueError(f"timestamp_ns must have shape (N,), got {timestamp.shape}")
    n = int(timestamp.shape[0])

    widths = {
        "arm_qpos_rad": 6,
        "hand_feedback_sdk_rad": 11,
        "hand_feedback_modes": 11,
        "arm_qcmd_sent_rad": 6,
        "hand_qcmd_effective_canonical_rad": 11,
        "robot_qcmd_17d_rad": 17,
    }
    for key, array in fields.items():
        expected = (n, widths[key]) if key in widths else (n,)
        if array.shape != expected:
            raise ValueError(f"{key} must have shape {expected}, got {array.shape}")
    return n


def _select_feedback(
    timestamps: np.ndarray,
    valid: np.ndarray,
    *,
    boundary: int,
    before: bool,
    component: str,
    reasons: list[str],
) -> tuple[int, int] | None:
    positive = timestamps > 0
    temporal = timestamps < boundary if before else timestamps > boundary
    temporal_candidates = np.flatnonzero(positive & temporal)
    if temporal_candidates.size == 0:
        reasons.append(f"CAUSALITY_{component}_UNAVAILABLE")
        return None

    candidates = np.flatnonzero(valid & temporal)
    if candidates.size == 0:
        reasons.append(f"FEEDBACK_{component}_INVALID")
        return None

    candidate_ts = timestamps[candidates]
    selected_ts = np.max(candidate_ts) if before else np.min(candidate_ts)
    tied = candidates[candidate_ts == selected_ts]
    return int(tied[-1] if before else tied[0])


def _select_camera(
    *,
    timestamps: np.ndarray,
    valid: np.ndarray,
    frame_indices: np.ndarray,
    boundary: int,
    media_dir: Path,
    component: str,
    reasons: list[str],
    media_integrity: dict[Path, bool],
) -> int | None:
    temporal = (timestamps > 0) & (timestamps < boundary)
    valid_mask = np.asarray(valid, dtype=bool) & (frame_indices >= 0)
    candidates = np.flatnonzero(temporal & valid_mask)
    if candidates.size == 0:
        reasons.append(f"CAMERA_{component}_UNAVAILABLE")
        return None

    candidate_ts = timestamps[candidates]
    selected_ts = np.max(candidate_ts)
    tied = candidates[candidate_ts == selected_ts]
    row = int(tied[-1])
    frame_index = int(frame_indices[row])
    path = media_dir / f"{frame_index:06d}.jpg"
    if not path.is_file():
        reasons.append(f"CAMERA_{component}_MEDIA_MISSING")
        return None
    # MISSING / SEMANTICALLY_INVALID RGB rejects this reference's transitions,
    # not unrelated good rows. No fallback to an older image; ordering is frozen.
    if path not in media_integrity:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.convert("RGB").load()
            media_integrity[path] = True
        except (OSError, ValueError):
            media_integrity[path] = False
    if not media_integrity[path]:
        reasons.append(f"CAMERA_{component}_DECODE_INVALID")
        return None
    return row, frame_index


def _record_rejection(
    tick: int,
    reasons: list[str],
    rejected: list[RejectedCandidate],
    reason_counts: Counter[str],
    group_counts: Counter[str],
) -> None:
    unique = tuple(sorted(set(reasons)))
    rejected.append(RejectedCandidate(tick, unique))
    reason_counts.update(unique)
    group_counts.update({reason.split("_", 1)[0] for reason in unique})
