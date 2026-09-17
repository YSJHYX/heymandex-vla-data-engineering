from __future__ import annotations

import hashlib
import json

import numpy as np

from vla_data.cleaning.build_curated_v1 import build_curated_v1
from vla_data.quality.evaluator import (
    QualityConfig,
    evaluate_curated_episode,
    evaluate_episode,
)
from vla_data.quality.report import aggregate_quality_reports
from vla_data.quality.temporal import TemporalQualityConfig
from vla_data.quality.visual import VisualQualityConfig


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build(synthetic_raw_episode, tmp_path):
    return build_curated_v1(
        synthetic_raw_episode,
        tmp_path / "curated",
        expert_training_status="SYNTHETIC_TEST_ONLY",
    )


def test_evaluate_episode_is_public_alias() -> None:
    assert evaluate_episode is evaluate_curated_episode


def test_none_thresholds_measure_without_rejecting(
    synthetic_raw_episode, tmp_path
) -> None:
    episode = _build(synthetic_raw_episode, tmp_path)
    result = evaluate_curated_episode(episode, tmp_path / "quality")

    assert result.status == "ACCEPT"
    assert result.clean_transition_count == 2
    mask = np.load(result.mask_path, allow_pickle=False)
    assert mask.tolist() == [True, True]


def test_configured_temporal_threshold_can_warn(
    synthetic_raw_episode, tmp_path
) -> None:
    episode = _build(synthetic_raw_episode, tmp_path)
    config = QualityConfig(temporal=TemporalQualityConfig(max_camera_age_ns=0))
    result = evaluate_curated_episode(episode, tmp_path / "quality", config=config)

    assert result.status == "ACCEPT_WITH_WARNING"
    assert result.clean_transition_count == 2


def test_configured_temporal_threshold_can_exclude(
    synthetic_raw_episode, tmp_path
) -> None:
    episode = _build(synthetic_raw_episode, tmp_path)
    config = QualityConfig(
        temporal=TemporalQualityConfig(
            max_camera_age_ns=0,
            exceedance_policy="exclude",
        )
    )
    result = evaluate_curated_episode(episode, tmp_path / "quality", config=config)

    assert result.status == "EXCLUDE_FROM_EXPERT_TRAINING"
    assert result.clean_transition_count == 0


def test_quality_evaluation_never_changes_trajectory_or_dimensions(
    synthetic_raw_episode, tmp_path
) -> None:
    episode = _build(synthetic_raw_episode, tmp_path)
    trajectory = episode / "trajectory.npz"
    before = _digest(trajectory)

    config = QualityConfig(
        visual=VisualQualityConfig(
            max_same_reference_run=1,
            anomaly_policy="exclude",
        )
    )
    result = evaluate_curated_episode(episode, tmp_path / "quality", config=config)

    assert _digest(trajectory) == before
    with np.load(trajectory, allow_pickle=False) as archive:
        assert archive["robot_qpos_17d_rad"].shape == (2, 17)
        assert archive["robot_qcmd_17d_rad"].shape == (2, 17)
    assert result.report_path.parent != episode
    assert result.clean_transition_count == 0


def test_hard_rgb_failure_is_reject(synthetic_raw_episode, tmp_path) -> None:
    episode = _build(synthetic_raw_episode, tmp_path)
    (episode / "media" / "head" / "rgb" / "000002.jpg").unlink()

    result = evaluate_curated_episode(episode, tmp_path / "quality")

    assert result.status == "REJECT"
    assert result.clean_transition_count == 0


def test_expert_status_is_diagnostic_only(synthetic_raw_episode, tmp_path) -> None:
    episode = build_curated_v1(
        synthetic_raw_episode,
        tmp_path / "curated",
        expert_training_status="EXCLUDE_FROM_EXPERT_TRAINING",
    )

    result = evaluate_curated_episode(episode, tmp_path / "quality")

    assert result.status == "ACCEPT"
    assert result.clean_transition_count == 2


def test_state_action_offset_does_not_reject(synthetic_raw_episode, tmp_path) -> None:
    episode = _build(synthetic_raw_episode, tmp_path)
    trajectory_path = episode / "trajectory.npz"
    with np.load(trajectory_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    arrays["robot_qcmd_17d_rad"] += 100.0
    np.savez(trajectory_path, **arrays)

    result = evaluate_curated_episode(episode, tmp_path / "quality")

    assert result.status == "ACCEPT"


def test_dataset_quality_aggregator(synthetic_raw_episode, tmp_path) -> None:
    episode = _build(synthetic_raw_episode, tmp_path)
    result = evaluate_curated_episode(episode, tmp_path / "quality")
    output = aggregate_quality_reports(
        [result.report_path], tmp_path / "dataset_quality_summary.json"
    )

    with output.open(encoding="utf-8") as stream:
        dataset_summary = json.load(stream)
    assert dataset_summary["episode_count"] == 1
    assert dataset_summary["transition_count"] == 2
    assert dataset_summary["clean_transition_count"] == 2
    assert dataset_summary["status_counts"] == {"ACCEPT": 1}
