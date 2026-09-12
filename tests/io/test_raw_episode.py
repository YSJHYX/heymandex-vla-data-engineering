from __future__ import annotations

import numpy as np
import pytest

from vla_data.io.raw_episode import RawEpisode


def test_raw_episode_is_read_only(synthetic_raw_episode) -> None:
    raw = RawEpisode.load(synthetic_raw_episode)

    assert raw.episode_id == "episode_000007"
    assert raw.require("arm_qpos_rad").shape == (8, 6)
    assert not raw.require("arm_qpos_rad").flags.writeable

    with pytest.raises(ValueError):
        raw.require("arm_qpos_rad")[0, 0] = 1.0


def test_raw_episode_scalar_and_missing_field(synthetic_raw_episode) -> None:
    raw = RawEpisode.load(synthetic_raw_episode)

    assert "arm_qpos_rad" in raw.keys
    assert raw.scalar("schema_version") == 3
    assert raw.metadata["schema_name"] == "synthetic_raw"
    with pytest.raises(KeyError, match="missing required field"):
        raw.require("does_not_exist")


def test_raw_loader_disables_pickle(tmp_path) -> None:
    path = tmp_path / "object.npz"
    media = tmp_path / "object_media"
    media.mkdir()
    np.savez(path, unsafe=np.asarray([object()], dtype=object))

    with pytest.raises(ValueError, match="Object arrays"):
        RawEpisode.load(path)
