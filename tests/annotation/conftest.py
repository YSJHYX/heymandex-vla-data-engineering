"""Shared fixtures for the D4 annotation test suite."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vla_data.batch.runner import build_curated_dataset, quality_dataset
from vla_data.io.curated_episode import CuratedEpisode


def _write_jpg(path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path, format="JPEG")


@pytest.fixture
def fake_curated_episode(tmp_path):
    """Build an in-memory CuratedEpisode with a controllable transition count."""

    def factory(
        transitions: int,
        *,
        episode_id: str = "episode_000001",
    ) -> CuratedEpisode:
        root = tmp_path / "fake_media" / episode_id
        head_indices = np.arange(transitions, dtype=np.int64)
        wrist_indices = np.arange(transitions, dtype=np.int64) + 1000
        for index in head_indices:
            _write_jpg(
                root / "media" / "head" / "rgb" / f"{index:06d}.jpg", (200, 40, 40)
            )
        for index in wrist_indices:
            _write_jpg(
                root / "media" / "right_wrist" / "rgb" / f"{index:06d}.jpg",
                (40, 40, 200),
            )
        timestamps = np.arange(
            1_000_000_000,
            1_000_000_000 + transitions * 10_000_000,
            10_000_000,
            dtype=np.int64,
        )
        trajectory = {
            "robot_qpos_17d_rad": np.zeros((transitions, 17), dtype=np.float64),
            "robot_qcmd_17d_rad": np.zeros((transitions, 17), dtype=np.float64),
            "action_timestamp_ns": timestamps,
            "state_timestamp_ns": timestamps,
            "head_rgb_frame_index": head_indices,
            "right_wrist_rgb_frame_index": wrist_indices,
        }
        return CuratedEpisode(
            episode_dir=root,
            metadata={
                "episode_id": episode_id,
                "schema_name": "rm65_sg100_dual_realsense_vla_curated",
                "schema_version": 1,
                "language_instruction": "test",
            },
            trajectory=trajectory,
        )

    return factory


@pytest.fixture
def annotated_tree(raw_dataset_factory, synthetic_raw_episode, tmp_path):
    """A synthetic curated+quality tree: ACCEPT, ACCEPT_WITH_WARNING, EXCLUDED.

    Episode outcomes after D3 evaluation:
      episode_000000 -> ACCEPT
      episode_000001 -> ACCEPT_WITH_WARNING (duplicate qcmd timestamps)
      episode_000002 -> EXCLUDE_FROM_EXPERT_TRAINING (explicit flag)
    """

    raw = raw_dataset_factory(tmp_path / "raw", (0, 1, 2))
    warning_path = raw / "episode_000001.npz"
    with np.load(warning_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    arrays["arm_qcmd_source_timestamp_ns"][4] = arrays["arm_qcmd_source_timestamp_ns"][
        3
    ]
    arrays["hand_qcmd_source_timestamp_ns"][4] = arrays[
        "hand_qcmd_source_timestamp_ns"
    ][3]
    np.savez(warning_path, **arrays)

    curated = tmp_path / "curated"
    quality = tmp_path / "quality"
    build_curated_dataset(raw, curated, expert_exclude=("episode_000002",))
    quality_dataset(curated, quality)

    return SimpleNamespace(
        raw=raw,
        curated=curated,
        quality=quality,
        output=tmp_path / "annotations",
    )
