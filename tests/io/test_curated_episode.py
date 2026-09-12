from __future__ import annotations

import numpy as np
import pytest

from vla_data.io.curated_episode import CuratedEpisode


def test_load_curated_episode(curated_v1_episode) -> None:
    episode = CuratedEpisode.load(curated_v1_episode)

    assert episode.transition_count == 3

    assert episode.state.shape == (3, 17)
    assert episode.action.shape == (3, 17)

    assert episode.arm_state.shape == (3, 6)
    assert episode.hand_state.shape == (3, 11)

    assert episode.arm_action.shape == (3, 6)
    assert episode.hand_action.shape == (3, 11)


def test_loaded_trajectory_arrays_are_read_only(curated_v1_episode) -> None:
    episode = CuratedEpisode.load(curated_v1_episode)

    assert not episode.state.flags.writeable
    assert not episode.action.flags.writeable

    with pytest.raises(ValueError):
        episode.state[0, 0] = 123.0


def test_episode_metadata_accessors(curated_v1_episode) -> None:
    episode = CuratedEpisode.load(curated_v1_episode)

    assert episode.segment_offsets == (0, 2, 3)
    assert episode.camera_roles == ("head", "right_wrist")

    assert (
        episode.language_instruction
        == "Pick up the object and place it into the fixture."
    )


def test_external_head_media(curated_v1_episode) -> None:
    episode = CuratedEpisode.load(curated_v1_episode)

    assert episode.media.existing_rgb_indices("head") == (0, 1)

    image = episode.media.read_rgb("head", 0)

    assert image.shape == (8, 10, 3)
    assert image.dtype == np.uint8


def test_external_wrist_media(curated_v1_episode) -> None:
    episode = CuratedEpisode.load(curated_v1_episode)

    assert episode.media.existing_rgb_indices("right_wrist") == (0, 1)

    image = episode.media.read_rgb("right_wrist", 1)

    assert image.shape == (8, 10, 3)
    assert image.dtype == np.uint8


def test_repeated_media_reference_is_supported(curated_v1_episode) -> None:
    episode = CuratedEpisode.load(curated_v1_episode)

    indices = episode.trajectory["head_rgb_frame_index"]

    assert indices.tolist() == [0, 0, 1]

    first = episode.media.rgb_path("head", int(indices[0]))
    second = episode.media.rgb_path("head", int(indices[1]))

    assert first == second


def test_invalid_camera_role_rejected(curated_v1_episode) -> None:
    episode = CuratedEpisode.load(curated_v1_episode)

    with pytest.raises(ValueError, match="unsupported camera role"):
        episode.media.rgb_path("left_wrist", 0)


def test_missing_episode_directory_rejected(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        CuratedEpisode.load(tmp_path / "does_not_exist")
