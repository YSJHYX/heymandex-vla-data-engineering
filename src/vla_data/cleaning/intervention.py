"""Validate RAW operator intervals and classify the existing sampler timeline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


def intervention_fingerprint(metadata: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                key: metadata[key]
                for key in ("operator_intervention", "operator_interventions")
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class InterventionIntervals:
    present: bool
    intervals: tuple[tuple[int, int | None], ...]
    invalid_mask: np.ndarray
    clean_domain: np.ndarray
    error: str | None = None

    def domain_at(self, timestamp_ns: int) -> int | None:
        """Return the clean interval number, or None inside an intervention."""
        domain = 0
        for start, end in self.intervals:
            if timestamp_ns < start:
                return domain
            if end is None or timestamp_ns <= end:
                return None
            domain += 1
        return domain

    def origin(self, domain: int) -> str:
        if not self.intervals:
            return "full_clean"
        if domain == 0:
            return "pre_intervention"
        if domain == len(self.intervals):
            return "post_intervention"
        return "between_interventions"


def classify_interventions(
    metadata: Mapping[str, object], timestamps: np.ndarray
) -> InterventionIntervals:
    """Missing legacy keys mean clean; inconsistent new metadata fails closed."""
    present = any(
        key in metadata for key in ("operator_intervention", "operator_interventions")
    )
    empty_mask = np.zeros(timestamps.shape, dtype=bool)
    empty_domain = np.zeros(timestamps.shape, dtype=np.int64)
    if not present:
        return InterventionIntervals(False, (), empty_mask, empty_domain)

    try:
        if {"operator_intervention", "operator_interventions"} - metadata.keys():
            raise ValueError("both operator intervention fields are required")
        flag = metadata["operator_intervention"]
        raw_intervals = metadata["operator_interventions"]
        if type(flag) is not bool or not isinstance(raw_intervals, list):
            raise ValueError("operator intervention flag/list has invalid type")
        if flag != bool(raw_intervals):
            raise ValueError("operator intervention flag disagrees with intervals")
        intervals: list[tuple[int, int | None]] = []
        for position, item in enumerate(raw_intervals):
            if not isinstance(item, dict):
                raise TypeError("operator intervention interval must be an object")
            start = item.get("start_timestamp_ns")
            end = item.get("end_timestamp_ns")
            if (
                type(start) is not int
                or start < 0
                or (end is not None and (type(end) is not int or end <= start))
            ):
                raise ValueError("operator intervention interval has invalid bounds")
            if position and (intervals[-1][1] is None or start <= intervals[-1][1]):
                raise ValueError(
                    "operator intervention intervals overlap or are unordered"
                )
            intervals.append((start, end))
    except (TypeError, ValueError) as exc:
        return InterventionIntervals(True, (), empty_mask, empty_domain, str(exc))

    invalid = np.zeros(timestamps.shape, dtype=bool)
    domains = np.zeros(timestamps.shape, dtype=np.int64)
    for index, (start, end) in enumerate(intervals):
        invalid |= (timestamps >= start) & (True if end is None else timestamps <= end)
        if end is not None:
            domains[timestamps > end] = index + 1
    invalid.setflags(write=False)
    domains.setflags(write=False)
    return InterventionIntervals(True, tuple(intervals), invalid, domains)
