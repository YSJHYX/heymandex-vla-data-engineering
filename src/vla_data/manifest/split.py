"""Seeded episode assignment, independent of filesystem and Python hash order."""

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class SplitConfig:
    validation_fraction: float = 0.0
    seed: int = 0
    strategy: str = "episode_random"
    group_key: str | None = None

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.validation_fraction)
            or not 0 <= self.validation_fraction <= 1
        ):
            raise ValueError("validation_fraction must be finite and within [0, 1]")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.strategy != "episode_random" or self.group_key is not None:
            raise ValueError(
                "only episode_random is supported; group-aware split is future work"
            )


def assign_splits(episode_ids: Iterable[str], config: SplitConfig) -> dict[str, str]:
    ids = sorted(episode_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate episode IDs")
    ranked = sorted(
        ids,
        key=lambda value: (
            hashlib.sha256(f"{config.seed}:{value}".encode()).digest(),
            value,
        ),
    )
    # Explicit floor; no forced validation episode for tiny datasets.
    validation = set(ranked[: math.floor(len(ids) * config.validation_fraction)])
    return {value: "val" if value in validation else "train" for value in ids}
