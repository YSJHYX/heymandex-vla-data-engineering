"""Deterministic keyframe selection over clean transitions only."""

from __future__ import annotations

import numpy as np

from vla_data.annotation.keyframes import (
    keyframe_image_items,
    select_keyframes,
)
from vla_data.annotation.prompts import PROMPT_VERSION


def _motion(episode, peak: int, magnitude: float = 5.0) -> None:
    """Deterministically place one strong motion phase at ``peak``."""

    state = np.zeros_like(episode.trajectory["robot_qpos_17d_rad"])
    state[peak, 0] = magnitude
    episode.trajectory["robot_qpos_17d_rad"][:] = state


def test_only_clean_transitions_can_become_keyframes(fake_curated_episode) -> None:
    episode = fake_curated_episode(20)
    mask = np.ones(20, dtype=bool)
    mask[[0, 1, 2, 18, 19]] = False
    mask[[7, 8]] = False
    selection = select_keyframes(episode, clean_mask=mask)
    chosen = {point.transition_index for point in selection.points}
    assert chosen
    assert chosen <= set(np.flatnonzero(mask))


def test_selection_is_deterministic(fake_curated_episode) -> None:
    episode = fake_curated_episode(30)
    mask = np.ones(30, dtype=bool)
    first = select_keyframes(episode, clean_mask=mask)
    second = select_keyframes(episode, clean_mask=mask)
    assert [p.transition_index for p in first.points] == [
        p.transition_index for p in second.points
    ]


def test_first_and_last_clean_transitions_anchor_initial_and_end_state(
    fake_curated_episode,
) -> None:
    episode = fake_curated_episode(24)
    mask = np.ones(24, dtype=bool)
    mask[[0, 1, 22, 23]] = False
    selection = select_keyframes(episode, clean_mask=mask)
    indices = [point.transition_index for point in selection.points]
    assert indices[0] == 2
    assert indices[-1] == 21


def test_motion_aware_adjustment_prefers_manipulation_peaks(
    fake_curated_episode,
) -> None:
    episode = fake_curated_episode(60)
    _motion(episode, peak=25)
    mask = np.ones(60, dtype=bool)
    selection = select_keyframes(episode, clean_mask=mask)
    indices = [point.transition_index for point in selection.points]
    assert len(indices) == 6
    # The strongest motion phase must be covered by some anchor.
    assert any(abs(index - 25) <= 4 for index in indices)


def test_dominant_peak_yields_distinct_anchors_not_deduplicated(
    fake_curated_episode,
) -> None:
    # A single dominant peak must not swallow two neighbouring anchors.
    episode = fake_curated_episode(60)
    _motion(episode, peak=35)
    mask = np.ones(60, dtype=bool)
    selection = select_keyframes(episode, clean_mask=mask)
    indices = [point.transition_index for point in selection.points]
    assert len(indices) == len(set(indices)) == 6
    assert any(abs(index - 35) <= 5 for index in indices)


def test_point_count_is_bounded_by_max(fake_curated_episode) -> None:
    episode = fake_curated_episode(50)
    mask = np.ones(50, dtype=bool)
    selection = select_keyframes(episode, clean_mask=mask, max_temporal_points=3)
    assert len(selection.points) == 3


def test_short_episodes_use_all_clean_transitions(fake_curated_episode) -> None:
    episode = fake_curated_episode(3)
    mask = np.array([True, True, True])
    selection = select_keyframes(episode, clean_mask=mask)
    assert [point.transition_index for point in selection.points] == [0, 1, 2]


def test_head_then_wrist_order_within_each_point(fake_curated_episode) -> None:
    episode = fake_curated_episode(12)
    mask = np.ones(12, dtype=bool)
    selection = select_keyframes(episode, clean_mask=mask)
    items = keyframe_image_items(selection)
    assert selection.prompt_version == PROMPT_VERSION
    for point in selection.points:
        head = next(
            item
            for item in items
            if item.temporal_point == point.point and item.camera == "head"
        )
        wrist = next(
            item
            for item in items
            if item.temporal_point == point.point and item.camera == "right_wrist"
        )
        assert items.index(head) < items.index(wrist)
        assert head.source_path.is_file()
        assert wrist.source_path.is_file()


def test_selection_records_frame_indices_paths_and_timestamps(
    fake_curated_episode,
) -> None:
    episode = fake_curated_episode(10)
    mask = np.ones(10, dtype=bool)
    selection = select_keyframes(episode, clean_mask=mask)
    document = selection.as_dict()
    assert document["episode_id"] == "episode_000001"
    assert document["prompt_version"] == PROMPT_VERSION
    for point, record in zip(
        selection.points, document["temporal_points"], strict=True
    ):
        assert record["transition_index"] == point.transition_index
        assert record["timestamp_ns"] == int(
            episode.trajectory["action_timestamp_ns"][point.transition_index]
        )
        assert record["head"]["frame_index"] == int(
            episode.trajectory["head_rgb_frame_index"][point.transition_index]
        )
        assert str(point.head_source_path) == record["head"]["source_path"]
        # No base64 payloads in keyframe provenance.
        assert "base64" not in str(record)
