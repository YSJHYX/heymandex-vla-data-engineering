"""Append rebasing must match Parquet without altering physical statistics."""

import copy

import pytest

from vla_data.export.bookkeeping_stats import (
    refresh_bookkeeping_stats,
    scalar_stats,
    validate_bookkeeping_stats,
)


def test_append_rebases_nested_stats_and_preserves_physical_fields():
    old_columns = {
        "episode_index": [0] * 3,
        "index": [0, 1, 2],
        "task_index": [0, 1, 0],
    }
    row = {
        "episode_index": 57,
        "stats": {
            **{key: scalar_stats(value) for key, value in old_columns.items()},
            "action": {"min": [0.0] * 17, "count": [3]},
            "observation.state": {"max": [1.0] * 17, "count": [3]},
            "observation.images.head": {"count": [2]},
        },
    }
    original = copy.deepcopy(row)
    actual = {
        "episode_index": [57] * 3,
        "index": [39727, 39728, 39729],
        "task_index": [1, 1, 1],
    }
    with pytest.raises(ValueError, match="stale"):
        validate_bookkeeping_stats(row, actual)
    fixed = refresh_bookkeeping_stats(row, actual)
    validate_bookkeeping_stats(fixed, actual)
    assert row == original
    for key in ("action", "observation.state", "observation.images.head"):
        assert fixed["stats"][key] == original["stats"][key]
    assert fixed["stats"]["index"]["std"] == row["stats"]["index"]["std"]
    assert fixed["stats"]["task_index"]["std"] == [0.0]
    assert refresh_bookkeeping_stats(fixed, actual) == fixed


def test_wrong_parquet_episode_is_rejected():
    columns = {"episode_index": [0], "index": [0], "task_index": [0]}
    row = {
        "episode_index": 57,
        "stats": {key: scalar_stats(value) for key, value in columns.items()},
    }
    with pytest.raises(ValueError, match="Parquet episode_index"):
        refresh_bookkeeping_stats(row, columns)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True])
def test_nonfinite_or_boolean_stats_rejected(bad):
    columns = {"episode_index": [0], "index": [0], "task_index": [0]}
    row = {
        "episode_index": 0,
        "stats": {key: scalar_stats(value) for key, value in columns.items()},
    }
    row["stats"]["index"]["mean"] = [bad]
    with pytest.raises(ValueError, match="stale"):
        validate_bookkeeping_stats(row, columns)
