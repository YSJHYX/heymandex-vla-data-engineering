from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from vla_data.cleaning.build_curated_v1 import build_curated_v1
from vla_data.cleaning.causal_sync import synchronize_episode
from vla_data.io.curated_episode import CuratedEpisode
from vla_data.io.raw_episode import RawEpisode
from vla_data.validation.curated_v1 import validate_curated_episode


def _rewrite(path: Path, mutate) -> RawEpisode:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    mutate(arrays)
    np.savez(path, **arrays)
    return RawEpisode.load(path)


def test_strict_pre_action_post_and_component_indices(
    synthetic_raw_episode,
) -> None:
    result = synchronize_episode(RawEpisode.load(synthetic_raw_episode))

    assert [t.source_tick_index for t in result.transitions] == [3, 4]
    first = result.transitions[0]
    assert (first.arm_pre_index, first.hand_pre_index) == (1, 2)
    assert (first.arm_post_index, first.hand_post_index) == (3, 3)
    assert max(first.arm_pre_timestamp_ns, first.hand_pre_timestamp_ns) < 30
    assert 32 < min(first.arm_post_timestamp_ns, first.hand_post_timestamp_ns)
    raw = RawEpisode.load(synthetic_raw_episode)
    expected_post = np.concatenate(
        (raw.require("arm_qpos_rad")[3], raw.require("hand_feedback_sdk_rad")[3])
    )
    assert np.array_equal(first.next_state17, expected_post)
    assert first.head_timestamp_ns < first.action_start_ns
    assert first.wrist_timestamp_ns < first.action_start_ns


def test_future_arm_feedback_is_not_used_as_pre(synthetic_raw_episode) -> None:
    raw = RawEpisode.load(synthetic_raw_episode)
    first = synchronize_episode(raw).transitions[0]

    assert int(raw.require("arm_qpos_source_timestamp_ns")[2]) == 31
    assert first.action_start_ns == 30
    assert first.arm_pre_index == 1


def test_physical_state_is_6_plus_11_measured_feedback(
    synthetic_raw_episode,
) -> None:
    raw = RawEpisode.load(synthetic_raw_episode)
    transition = synchronize_episode(raw).transitions[0]
    expected = np.concatenate(
        (raw.require("arm_qpos_rad")[1], raw.require("hand_feedback_sdk_rad")[2])
    )

    assert transition.state17.shape == (17,)
    assert np.array_equal(transition.state17, expected)


def test_action_is_exact_recorded_effective_17d(synthetic_raw_episode) -> None:
    raw = RawEpisode.load(synthetic_raw_episode)
    transition = synchronize_episode(raw).transitions[0]
    expected = np.concatenate(
        (
            raw.require("arm_qcmd_sent_rad")[3],
            raw.require("hand_qcmd_effective_canonical_rad")[3],
        )
    )

    assert transition.action17.shape == (17,)
    assert np.array_equal(transition.action17, raw.require("robot_qcmd_17d_rad")[3])
    assert np.array_equal(transition.action17, expected)


def test_retargeted_and_canonical_fields_are_never_authority(
    synthetic_raw_episode,
) -> None:
    raw = RawEpisode.load(synthetic_raw_episode)
    transition = synchronize_episode(raw).transitions[0]

    assert not np.any(transition.state17 == 999.0)
    assert not np.any(transition.action17 == 777.0)


def test_action_component_mismatch_is_rejected(synthetic_raw_episode) -> None:
    raw = _rewrite(
        synthetic_raw_episode,
        lambda arrays: arrays["robot_qcmd_17d_rad"].__setitem__((3, 0), -123.0),
    )
    result = synchronize_episode(raw)

    assert [t.source_tick_index for t in result.transitions] == [4]
    assert result.rejection_counts["ACTION_COMPONENT_MISMATCH"] == 1


def test_non_mode7_hand_feedback_is_rejected(synthetic_raw_episode) -> None:
    raw = _rewrite(
        synthetic_raw_episode,
        lambda arrays: arrays["hand_feedback_modes"].fill(6),
    )
    result = synchronize_episode(raw)

    assert result.transitions == ()
    assert result.rejection_group_counts["FEEDBACK"] >= 1


def test_nonfinite_hand_feedback_is_rejected(synthetic_raw_episode) -> None:
    raw = _rewrite(
        synthetic_raw_episode,
        lambda arrays: arrays["hand_feedback_sdk_rad"].fill(np.nan),
    )
    result = synchronize_episode(raw)

    assert result.transitions == ()
    assert result.rejection_group_counts["FEEDBACK"] >= 1


def test_missing_strict_post_is_causality_rejection(synthetic_raw_episode) -> None:
    def mutate(arrays) -> None:
        arrays["hand_feedback_source_timestamp_ns"][3:] = 0
        arrays["hand_feedback_valid"][3:] = False

    result = synchronize_episode(_rewrite(synthetic_raw_episode, mutate))

    assert result.transitions == ()
    assert any(
        "CAUSALITY_HAND_POST_UNAVAILABLE" in rejected.reasons
        for rejected in result.rejected
    )


def test_invalid_camera_is_rejected(synthetic_raw_episode) -> None:
    raw = _rewrite(
        synthetic_raw_episode,
        lambda arrays: arrays["head_camera_valid"].fill(False),
    )
    result = synchronize_episode(raw)

    assert result.transitions == ()
    assert result.rejection_group_counts["CAMERA"] >= 1


def test_missing_camera_media_is_rejected(synthetic_raw_episode) -> None:
    media = synthetic_raw_episode.with_name("episode_000007_media")
    (media / "right_wrist" / "rgb" / "000002.jpg").unlink()
    result = synchronize_episode(RawEpisode.load(synthetic_raw_episode))

    assert result.transitions == ()
    assert result.rejection_counts["CAMERA_WRIST_CAMERA_MEDIA_MISSING"] >= 1


def test_builder_deduplicates_jpegs_and_validates(
    synthetic_raw_episode, tmp_path
) -> None:
    output = build_curated_v1(
        synthetic_raw_episode,
        tmp_path / "curated",
        expert_training_status="SYNTHETIC_TEST_ONLY",
    )
    episode = CuratedEpisode.load(output)
    report = validate_curated_episode(episode)

    assert report.passed
    assert episode.transition_count == 2
    assert episode.task_instruction == "synthetic task"
    assert episode.trajectory["head_rgb_frame_index"].tolist() == [2, 2]
    assert episode.trajectory["right_wrist_rgb_frame_index"].tolist() == [2, 2]
    assert episode.media.existing_rgb_indices("head") == (2,)
    assert episode.media.existing_rgb_indices("right_wrist") == (2,)
    assert set(episode.trajectory) == {
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
    }

    with (output / "cleaning_report.json").open(encoding="utf-8") as stream:
        cleaning_report = json.load(stream)
    assert cleaning_report["accepted_count"] == 2
    assert (
        cleaning_report["synchronization_metrics"]["state_component_skew_ns"]["count"]
        == 2
    )
    assert set(cleaning_report["transition_temporal_metrics"]) == {
        "state_component_skew_ns",
        "action_component_skew_ns",
        "post_state_component_skew_ns",
        "arm_state_age_at_action_ns",
        "hand_state_age_at_action_ns",
        "head_frame_age_at_action_ns",
        "wrist_frame_age_at_action_ns",
        "head_wrist_frame_skew_ns",
    }
    assert cleaning_report["segment_count"] == 1
    assert cleaning_report["rejection_group_counts"] == {
        "ACTION": 5,
        "FEEDBACK": 0,
        "CAUSALITY": 0,
        "CAMERA": 0,
    }


def test_camera_files_are_real_rgb_jpegs(synthetic_raw_episode) -> None:
    media = synthetic_raw_episode.with_name("episode_000007_media")
    for role in ("head", "right_wrist"):
        with Image.open(media / role / "rgb" / "000002.jpg") as image:
            assert image.convert("RGB").getbands() == ("R", "G", "B")
