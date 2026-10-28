"""LeRobot scalar bookkeeping statistics, checked against actual Parquet rows."""

from __future__ import annotations

import copy
import math

BOOKKEEPING = ("episode_index", "index", "task_index")


def scalar_stats(values: list[int]) -> dict:
    if not values:
        raise ValueError("empty bookkeeping column")
    count = len(values)
    mean = math.fsum(values) / count
    return {
        "count": [count],
        "min": [min(values)],
        "max": [max(values)],
        "mean": [mean],
        "std": [math.sqrt(math.fsum((x - mean) ** 2 for x in values) / count)],
    }


def _matches(actual: dict, expected: dict) -> bool:
    for field, value in expected.items():
        found = actual.get(field)
        if hasattr(found, "tolist"):
            found = found.tolist()
        if (
            not isinstance(found, list)
            or len(found) != 1
            or not isinstance(found[0], (int, float))
            or isinstance(found[0], bool)
            or not math.isfinite(found[0])
            or not math.isclose(found[0], value[0], rel_tol=1e-9, abs_tol=1e-6)
        ):
            return False
    return True


def validate_bookkeeping_stats(row: dict, columns: dict[str, list[int]]) -> None:
    if set(columns["episode_index"]) != {row["episode_index"]}:
        raise ValueError("Parquet episode_index differs from metadata")
    for key in BOOKKEEPING:
        if not _matches(row["stats"][key], scalar_stats(columns[key])):
            raise ValueError(f"episode {row['episode_index']} stale {key} statistics")


def refresh_bookkeeping_stats(row: dict, columns: dict[str, list[int]]) -> dict:
    """Preserve physical/image stats and already-correct bookkeeping exactly."""
    result = copy.deepcopy(row)
    for key in BOOKKEEPING:
        expected = scalar_stats(columns[key])
        if not _matches(result["stats"].get(key, {}), expected):
            result["stats"][key] = expected
    validate_bookkeeping_stats(result, columns)
    return result
