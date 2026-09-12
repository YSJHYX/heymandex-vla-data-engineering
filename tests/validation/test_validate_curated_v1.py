from __future__ import annotations

import json

import numpy as np

from vla_data.io.curated_episode import CuratedEpisode
from vla_data.validation.curated_v1 import (
    validate_curated_episode,
)


def _load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _rewrite_npz(path, arrays) -> None:
    np.savez(path, **arrays)


def _load_metadata(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _rewrite_metadata(path, metadata) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)


def test_valid_episode_passes(curated_v1_episode) -> None:
    episode = CuratedEpisode.load(curated_v1_episode)
    report = validate_curated_episode(episode)

    assert report.quality == "PASS"
    assert report.passed
    assert report.errors == ()
    assert report.transition_count == 3


def test_wrong_schema_rejected(curated_v1_episode) -> None:
    metadata_path = curated_v1_episode / "metadata.json"
    metadata = _load_metadata(metadata_path)

    metadata["schema_name"] = "wrong_schema"
    _rewrite_metadata(metadata_path, metadata)

    report = validate_curated_episode(CuratedEpisode.load(curated_v1_episode))

    assert report.quality == "FAIL"
    assert any("schema_name" in error for error in report.errors)


def test_wrong_joint_order_rejected(curated_v1_episode) -> None:
    metadata_path = curated_v1_episode / "metadata.json"
    metadata = _load_metadata(metadata_path)

    metadata["joint_names"][0], metadata["joint_names"][1] = (
        metadata["joint_names"][1],
        metadata["joint_names"][0],
    )

    _rewrite_metadata(metadata_path, metadata)

    report = validate_curated_episode(CuratedEpisode.load(curated_v1_episode))

    assert report.quality == "FAIL"
    assert any("canonical 17D ordering" in error for error in report.errors)


def test_32d_state_rejected(curated_v1_episode) -> None:
    trajectory_path = curated_v1_episode / "trajectory.npz"
    arrays = _load_npz(trajectory_path)

    arrays["robot_qpos_17d_rad"] = np.zeros(
        (3, 32),
        dtype=np.float64,
    )

    _rewrite_npz(trajectory_path, arrays)

    report = validate_curated_episode(CuratedEpisode.load(curated_v1_episode))

    assert report.quality == "FAIL"
    assert any("expected=(M, 17)" in error for error in report.errors)


def test_32d_action_rejected(curated_v1_episode) -> None:
    trajectory_path = curated_v1_episode / "trajectory.npz"
    arrays = _load_npz(trajectory_path)
    arrays["robot_qcmd_17d_rad"] = np.zeros((3, 32), dtype=np.float64)
    _rewrite_npz(trajectory_path, arrays)

    report = validate_curated_episode(CuratedEpisode.load(curated_v1_episode))

    assert report.quality == "FAIL"
    assert any("robot_qcmd_17d_rad shape=(3, 32)" in error for error in report.errors)


def test_nonfinite_state_rejected(curated_v1_episode) -> None:
    trajectory_path = curated_v1_episode / "trajectory.npz"
    arrays = _load_npz(trajectory_path)

    arrays["robot_qpos_17d_rad"][1, 4] = np.nan

    _rewrite_npz(trajectory_path, arrays)

    report = validate_curated_episode(CuratedEpisode.load(curated_v1_episode))

    assert report.quality == "FAIL"
    assert any("non-finite" in error for error in report.errors)


def test_invalid_segment_offsets_rejected(
    curated_v1_episode,
) -> None:
    metadata_path = curated_v1_episode / "metadata.json"
    metadata = _load_metadata(metadata_path)

    metadata["segment_offsets"] = [0, 3]

    _rewrite_metadata(metadata_path, metadata)

    report = validate_curated_episode(CuratedEpisode.load(curated_v1_episode))

    assert report.quality == "FAIL"
    assert any("tick discontinuity" in error for error in report.errors)


def test_invalid_unified_causality_rejected(
    curated_v1_episode,
) -> None:
    trajectory_path = curated_v1_episode / "trajectory.npz"
    arrays = _load_npz(trajectory_path)

    arrays["action_timestamp_ns"][1] = arrays["next_state_timestamp_ns"][1]

    _rewrite_npz(trajectory_path, arrays)

    report = validate_curated_episode(CuratedEpisode.load(curated_v1_episode))

    assert report.quality == "FAIL"
    assert any("action < next_state" in error for error in report.errors)


def test_non_strict_pre_causality_rejected(curated_v1_episode) -> None:
    trajectory_path = curated_v1_episode / "trajectory.npz"
    arrays = _load_npz(trajectory_path)
    arrays["hand_qpos_source_timestamp_ns"][1] = 210
    arrays["state_timestamp_ns"][1] = 210
    _rewrite_npz(trajectory_path, arrays)

    report = validate_curated_episode(CuratedEpisode.load(curated_v1_episode))

    assert report.quality == "FAIL"
    assert any("PRE causality" in error for error in report.errors)


def test_missing_referenced_head_media_rejected(
    curated_v1_episode,
) -> None:
    (curated_v1_episode / "media" / "head" / "rgb" / "000001.jpg").unlink()

    report = validate_curated_episode(CuratedEpisode.load(curated_v1_episode))

    assert report.quality == "FAIL"
    assert any("head/rgb reference mismatch" in error for error in report.errors)


def test_extra_unreferenced_media_rejected(
    curated_v1_episode,
) -> None:
    source = curated_v1_episode / "media" / "head" / "rgb" / "000000.jpg"
    extra = curated_v1_episode / "media" / "head" / "rgb" / "000002.jpg"

    extra.write_bytes(source.read_bytes())

    report = validate_curated_episode(CuratedEpisode.load(curated_v1_episode))

    assert report.quality == "FAIL"
    assert any("head/rgb reference mismatch" in error for error in report.errors)


def test_forbidden_duplicate_representation_rejected(
    curated_v1_episode,
) -> None:
    trajectory_path = curated_v1_episode / "trajectory.npz"
    arrays = _load_npz(trajectory_path)

    arrays["arm_qpos_6d_rad"] = np.zeros(
        (3, 6),
        dtype=np.float64,
    )

    _rewrite_npz(trajectory_path, arrays)

    report = validate_curated_episode(CuratedEpisode.load(curated_v1_episode))

    assert report.quality == "FAIL"
    assert any("forbidden trajectory keys" in error for error in report.errors)


def test_depth_directory_rejected(
    curated_v1_episode,
) -> None:
    depth_dir = curated_v1_episode / "media" / "head" / "depth"

    depth_dir.mkdir(parents=True)

    report = validate_curated_episode(CuratedEpisode.load(curated_v1_episode))

    assert report.quality == "FAIL"
    assert any("forbids depth" in error for error in report.errors)
