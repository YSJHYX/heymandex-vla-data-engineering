"""Small dependency-free statistical helpers for D3 quality reports."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def summary(values: Iterable[float] | np.ndarray) -> dict[str, int | float | None]:
    """Describe finite scalar values without inventing values for empty input."""

    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values)
    finite = np.asarray(array, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def vector_summary(values: np.ndarray) -> dict[str, object]:
    """Summarize absolute vector values globally and per joint."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"vector metrics require a 2D array, got {array.shape}")
    width = int(array.shape[1])
    if array.shape[0] == 0:
        per_joint_p99: list[float | None] = [None] * width
        per_joint_max: list[float | None] = [None] * width
    else:
        absolute = np.abs(array)
        per_joint_p99 = np.percentile(absolute, 99, axis=0).astype(float).tolist()
        per_joint_max = np.max(absolute, axis=0).astype(float).tolist()
    return {
        "sample_count": int(array.shape[0]),
        "summary": summary(np.abs(array)),
        "per_joint_p99": per_joint_p99,
        "per_joint_max": per_joint_max,
    }
