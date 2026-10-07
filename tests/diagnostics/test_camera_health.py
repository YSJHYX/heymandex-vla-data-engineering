from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vla_data.diagnostics.camera_health import camera_health


def _timeline(rows: int = 120) -> np.ndarray:
    return 2_000_000_000_000 + np.arange(rows, dtype=np.int64) * 33_000_000


def test_healthy_camera() -> None:
    master = _timeline()
    result = camera_health(
        "head",
        master,
        master - 18_000_000,
        np.ones(master.size, dtype=bool),
        np.arange(master.size, dtype=np.int64),
    )

    assert result.status == "HEALTHY"
    assert result.timestamp_unique_count == master.size
    assert result.master_minus_camera_ns["median"] == 18_000_000


def test_frozen_camera() -> None:
    master = _timeline()
    frozen = np.full(master.size, master[0] - 975_000_000_000, dtype=np.int64)
    result = camera_health(
        "head",
        master,
        frozen,
        np.zeros(master.size, dtype=bool),
        np.arange(master.size, dtype=np.int64),
    )

    assert result.status == "FROZEN"
    assert result.timestamp_unique_count == 1
    assert result.max_consecutive_identical_timestamp == master.size
    assert result.first_stale_row == 0
    assert result.last_timestamp_change_row == 0
    assert result.estimated_frozen_duration_ns is not None
    assert result.estimated_frozen_duration_ns > 975_000_000_000


def test_stale_camera() -> None:
    master = _timeline()
    result = camera_health(
        "head",
        master,
        master - 5_000_000_000,
        np.ones(master.size, dtype=bool),
        np.arange(master.size, dtype=np.int64),
    )

    assert result.status == "STALE"


def test_missing_camera() -> None:
    master = _timeline()
    result = camera_health(
        "head",
        master,
        np.zeros(master.size, dtype=np.int64),
        np.zeros(master.size, dtype=bool),
        np.full(master.size, -1, dtype=np.int64),
    )

    assert result.status == "MISSING"


def test_future_clock_domain_is_clock_mismatch() -> None:
    master = _timeline()
    result = camera_health(
        "head",
        master,
        master + 120_000_000_000,
        np.ones(master.size, dtype=bool),
        np.arange(master.size, dtype=np.int64),
    )

    assert result.status == "CLOCK_MISMATCH"


def test_huge_fixed_past_offset_is_conservative_unknown() -> None:
    master = _timeline()
    result = camera_health(
        "head",
        master,
        master - 120_000_000_000,
        np.ones(master.size, dtype=bool),
        np.arange(master.size, dtype=np.int64),
    )

    assert result.status == "UNKNOWN"


REAL_ROOT = Path("/data/vla_runs/biyi_real_test_dataset/raw")


@pytest.mark.skipif(
    not (REAL_ROOT / "episode_000010.npz").is_file(),
    reason="real episodes 000007-000010 unavailable",
)
def test_real_episodes_000007_through_000010_head_is_frozen() -> None:
    for number in range(7, 11):
        with np.load(
            REAL_ROOT / f"episode_{number:06d}.npz", allow_pickle=False
        ) as archive:
            result = camera_health(
                "head",
                archive["timestamp_ns"],
                archive["head_camera_host_timestamp_ns"],
                archive["head_camera_valid"],
                archive["head_rgb_frame_index"],
            )
        assert result.status == "FROZEN", number
        assert result.timestamp_unique_count == 1, number
