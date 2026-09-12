"""Physical 17D trajectory and activity quality metrics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from vla_data.io.curated_episode import CuratedEpisode
from vla_data.morphology.rm65_sg100 import ARM_SLICE, HAND_SLICE
from vla_data.quality.metrics import summary, vector_summary


@dataclass(frozen=True)
class ActivityConfig:
    motion_range_epsilon_rad: float = 1e-6

    def __post_init__(self) -> None:
        if self.motion_range_epsilon_rad < 0:
            raise ValueError("motion_range_epsilon_rad must be >= 0")


def evaluate_trajectory(
    episode: CuratedEpisode,
    activity_config: ActivityConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    """Compute real-dt velocity metrics and domain-independent activity."""

    state = np.asarray(episode.state, dtype=np.float64)
    action = np.asarray(episode.action, dtype=np.float64)
    if state.shape[1:] != (17,) or action.shape[1:] != (17,):
        raise ValueError("D3 trajectory evaluation requires physical 17D state/action")

    arm_start, arm_end = ARM_SLICE
    hand_start, hand_end = HAND_SLICE
    arm_state_velocity, arm_state_velocity_timing = _velocity(
        state[:, arm_start:arm_end],
        np.asarray(episode.trajectory["arm_qpos_source_timestamp_ns"], dtype=np.int64),
        episode.segment_offsets,
    )
    hand_state_velocity, hand_state_velocity_timing = _velocity(
        state[:, hand_start:hand_end],
        np.asarray(episode.trajectory["hand_qpos_source_timestamp_ns"], dtype=np.int64),
        episode.segment_offsets,
    )
    arm_action_velocity, arm_action_velocity_timing = _velocity(
        action[:, arm_start:arm_end],
        np.asarray(episode.trajectory["arm_qcmd_source_timestamp_ns"], dtype=np.int64),
        episode.segment_offsets,
    )
    hand_action_velocity, hand_action_velocity_timing = _velocity(
        action[:, hand_start:hand_end],
        np.asarray(episode.trajectory["hand_qcmd_source_timestamp_ns"], dtype=np.int64),
        episode.segment_offsets,
    )

    trajectory = {
        "velocity_uses_real_dt": True,
        "velocity_timestamp_authority": {
            "arm_state": "arm_qpos_source_timestamp_ns",
            "hand_state": "hand_qpos_source_timestamp_ns",
            "arm_action": "arm_qcmd_source_timestamp_ns",
            "hand_action": "hand_qcmd_source_timestamp_ns",
        },
        "arm_state_velocity_timing": arm_state_velocity_timing,
        "hand_state_velocity_timing": hand_state_velocity_timing,
        "arm_action_velocity_timing": arm_action_velocity_timing,
        "hand_action_velocity_timing": hand_action_velocity_timing,
        "arm_state_velocity_rad_s": vector_summary(arm_state_velocity),
        "hand_state_velocity_rad_s": vector_summary(hand_state_velocity),
        "arm_action_velocity_rad_s": vector_summary(arm_action_velocity),
        "hand_action_velocity_rad_s": vector_summary(hand_action_velocity),
        "state_step_rad": vector_summary(
            _segment_steps(state, episode.segment_offsets)
        ),
        "action_step_rad": vector_summary(
            _segment_steps(action, episode.segment_offsets)
        ),
    }

    state_range = np.ptp(state, axis=0) if state.size else np.zeros(17)
    action_range = np.ptp(action, axis=0) if action.size else np.zeros(17)
    state_active = state_range > activity_config.motion_range_epsilon_rad
    action_active = action_range > activity_config.motion_range_epsilon_rad
    active = state_active | action_active
    arm_active_count = int(np.count_nonzero(active[arm_start:arm_end]))
    hand_active_count = int(np.count_nonzero(active[hand_start:hand_end]))
    arm_action_active_count = int(np.count_nonzero(action_active[arm_start:arm_end]))
    hand_action_active_count = int(np.count_nonzero(action_active[hand_start:hand_end]))
    arm_state_active_count = int(np.count_nonzero(state_active[arm_start:arm_end]))
    hand_state_active_count = int(np.count_nonzero(state_active[hand_start:hand_end]))
    activity = {
        "motion_range_epsilon_rad": activity_config.motion_range_epsilon_rad,
        "active_joint_count": int(np.count_nonzero(active)),
        "active_joint_mask": active.tolist(),
        "state_active_joint_mask": state_active.tolist(),
        "action_active_joint_mask": action_active.tolist(),
        "state_active_joint_count": int(np.count_nonzero(state_active)),
        "action_active_joint_count": int(np.count_nonzero(action_active)),
        "state_motion_range_per_joint_rad": state_range.astype(float).tolist(),
        "action_motion_range_per_joint_rad": action_range.astype(float).tolist(),
        "arm_active_joint_count": arm_active_count,
        "hand_active_joint_count": hand_active_count,
        "arm_state_active_joint_count": arm_state_active_count,
        "hand_state_active_joint_count": hand_state_active_count,
        "arm_action_active_joint_count": arm_action_active_count,
        "hand_action_active_joint_count": hand_action_active_count,
        "single_active_hand_action_joint": hand_action_active_count == 1,
        "hand_action_activity": (
            "STATIC"
            if hand_action_active_count == 0
            else (
                "SINGLE_JOINT"
                if hand_action_active_count == 1
                else (
                    "FULL_HAND"
                    if hand_action_active_count == hand_end - hand_start
                    else "PARTIAL_HAND"
                )
            )
        ),
        "static_episode": not bool(np.any(active)),
        "single_active_joint": int(np.count_nonzero(active)) == 1,
        "partial_hand_activity": 0 < hand_active_count < hand_end - hand_start,
        "full_hand_activity": hand_active_count == hand_end - hand_start,
        "full_arm_hand_activity": arm_active_count > 0
        and hand_active_count == hand_end - hand_start,
    }
    return trajectory, activity


def _velocity(
    values: np.ndarray,
    timestamps_ns: np.ndarray,
    segment_offsets: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, object]]:
    steps = _pair_indices(segment_offsets)
    if not steps:
        return np.empty((0, values.shape[1]), dtype=np.float64), {
            "candidate_pair_count": 0,
            "computed_pair_count": 0,
            "nonpositive_dt_pair_count": 0,
            "positive_dt_s": summary([]),
        }
    previous = np.asarray([left for left, _ in steps], dtype=np.int64)
    current = np.asarray([right for _, right in steps], dtype=np.int64)
    dt_s = (timestamps_ns[current] - timestamps_ns[previous]).astype(np.float64) / 1e9
    positive = dt_s > 0
    velocity = (values[current[positive]] - values[previous[positive]]) / dt_s[
        positive, None
    ]
    return velocity, {
        "candidate_pair_count": len(steps),
        "computed_pair_count": int(np.count_nonzero(positive)),
        "nonpositive_dt_pair_count": int(np.count_nonzero(~positive)),
        "positive_dt_s": summary(dt_s[positive]),
    }


def _segment_steps(values: np.ndarray, offsets: tuple[int, ...]) -> np.ndarray:
    pairs = _pair_indices(offsets)
    if not pairs:
        return np.empty((0, values.shape[1]), dtype=np.float64)
    return np.stack([values[right] - values[left] for left, right in pairs])


def _pair_indices(offsets: tuple[int, ...]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for start, end in pairwise(offsets):
        pairs.extend((index - 1, index) for index in range(start + 1, end))
    return pairs
