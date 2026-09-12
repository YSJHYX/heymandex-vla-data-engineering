from __future__ import annotations

import numpy as np

from vla_data.io.curated_episode import CuratedEpisode
from vla_data.quality.trajectory import ActivityConfig, evaluate_trajectory


def _rewrite_trajectory(path, mutate) -> CuratedEpisode:
    trajectory_path = path / "trajectory.npz"
    with np.load(trajectory_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    mutate(arrays)
    np.savez(trajectory_path, **arrays)
    return CuratedEpisode.load(path)


def test_velocity_uses_real_dt(curated_v1_episode) -> None:
    episode = CuratedEpisode.load(curated_v1_episode)
    trajectory, _ = evaluate_trajectory(episode, ActivityConfig())
    dt_s = (
        episode.trajectory["arm_qpos_source_timestamp_ns"][1]
        - episode.trajectory["arm_qpos_source_timestamp_ns"][0]
    ) / 1e9
    expected = abs((episode.state[1, 0] - episode.state[0, 0]) / dt_s)

    observed = trajectory["arm_state_velocity_rad_s"]["per_joint_max"][0]
    assert observed == expected


def test_each_subsystem_velocity_uses_its_native_source_timestamp(
    curated_v1_episode,
) -> None:
    def mutate(arrays) -> None:
        arrays["state_timestamp_ns"][:] = [1, 2, 3]
        arrays["action_timestamp_ns"][:] = [4, 5, 6]
        arrays["arm_qpos_source_timestamp_ns"][:] = [10, 110, 210]
        arrays["hand_qpos_source_timestamp_ns"][:] = [20, 220, 420]
        arrays["arm_qcmd_source_timestamp_ns"][:] = [30, 330, 630]
        arrays["hand_qcmd_source_timestamp_ns"][:] = [40, 440, 840]

    episode = _rewrite_trajectory(curated_v1_episode, mutate)
    trajectory, _ = evaluate_trajectory(episode, ActivityConfig())

    expected = {
        "arm_state_velocity_rad_s": 0.017 / (100 / 1e9),
        "hand_state_velocity_rad_s": 0.017 / (200 / 1e9),
        "arm_action_velocity_rad_s": 0.017 / (300 / 1e9),
        "hand_action_velocity_rad_s": 0.017 / (400 / 1e9),
    }
    for key, value in expected.items():
        assert trajectory[key]["per_joint_max"][0] == value
    assert trajectory["velocity_timestamp_authority"] == {
        "arm_state": "arm_qpos_source_timestamp_ns",
        "hand_state": "hand_qpos_source_timestamp_ns",
        "arm_action": "arm_qcmd_source_timestamp_ns",
        "hand_action": "hand_qcmd_source_timestamp_ns",
    }


def test_static_episode_detected(curated_v1_episode) -> None:
    def mutate(arrays) -> None:
        arrays["robot_qpos_17d_rad"][:] = 0.0
        arrays["robot_qcmd_17d_rad"][:] = 100.0

    episode = _rewrite_trajectory(curated_v1_episode, mutate)
    _, activity = evaluate_trajectory(episode, ActivityConfig())

    assert activity["static_episode"] is True
    assert activity["active_joint_count"] == 0


def test_single_active_joint_characterized(curated_v1_episode) -> None:
    def mutate(arrays) -> None:
        arrays["robot_qpos_17d_rad"][:] = 0.0
        arrays["robot_qcmd_17d_rad"][:] = 10.0
        arrays["robot_qcmd_17d_rad"][:, 7] = [10.0, 10.1, 10.2]

    episode = _rewrite_trajectory(curated_v1_episode, mutate)
    _, activity = evaluate_trajectory(episode, ActivityConfig())

    assert activity["single_active_joint"] is True
    assert activity["active_joint_count"] == 1
    assert activity["active_joint_mask"][7] is True
    assert activity["single_active_hand_action_joint"] is True
    assert activity["hand_action_activity"] == "SINGLE_JOINT"


def test_state_action_domain_offset_is_not_tracking_error(
    curated_v1_episode,
) -> None:
    def mutate(arrays) -> None:
        arrays["robot_qcmd_17d_rad"] = arrays["robot_qpos_17d_rad"] + 100.0

    episode = _rewrite_trajectory(curated_v1_episode, mutate)
    trajectory, activity = evaluate_trajectory(episode, ActivityConfig())

    assert activity["active_joint_count"] == 17
    assert "tracking_error" not in trajectory
