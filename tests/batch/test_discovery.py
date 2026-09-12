from __future__ import annotations

import shutil

import numpy as np
import pytest

from vla_data.batch.discovery import (
    DiscoveryError,
    canonical_episode_id,
    discover_raw_episodes,
)


def test_raw_discovery_is_canonical_numeric_and_deterministic(
    raw_dataset_factory, tmp_path
) -> None:
    root = raw_dataset_factory(tmp_path / "raw", (10, 2, 1))
    np.savez(root / "unrelated.npz", value=np.asarray(1))
    np.savez(root / "episode_bad.npz", value=np.asarray(1))

    result = discover_raw_episodes(root)

    assert [item.episode_id for item in result.episodes] == [
        "episode_000001",
        "episode_000002",
        "episode_000010",
    ]
    assert [path.name for path in result.ignored_paths] == ["unrelated.npz"]
    assert [issue.error_type for issue in result.issues] == ["INVALID_EPISODE_NAME"]


def test_missing_media_is_reported_and_not_dispatched(
    synthetic_raw_episode, tmp_path
) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    shutil.copy2(synthetic_raw_episode, root / "episode_000004.npz")

    result = discover_raw_episodes(root)

    assert result.episodes == ()
    assert result.issues[0].episode_id == "episode_000004"
    assert result.issues[0].error_type == "MISSING_MEDIA_DIRECTORY"


def test_single_episode_selection_uses_same_discovery(raw_dataset_factory, tmp_path):
    root = raw_dataset_factory(tmp_path / "raw", (1, 2))

    result = discover_raw_episodes(root, episode="episode_000002")

    assert [item.episode_id for item in result.episodes] == ["episode_000002"]


@pytest.mark.parametrize("value", ["episode_1", "1", "episode_000001.npz"])
def test_noncanonical_episode_selector_is_rejected(value) -> None:
    with pytest.raises(DiscoveryError):
        canonical_episode_id(value)
