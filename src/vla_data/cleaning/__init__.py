"""RAW-to-Curated cleaning primitives."""

from vla_data.cleaning.build_curated_v1 import build_curated_v1
from vla_data.cleaning.causal_sync import (
    CausalSyncResult,
    CausalTransition,
    synchronize_episode,
)

# Stable task-oriented spelling for callers; the versioned implementation remains
# available for code that intentionally pins Curated v1.
build_curated_episode = build_curated_v1

__all__ = [
    "CausalSyncResult",
    "CausalTransition",
    "build_curated_episode",
    "build_curated_v1",
    "synchronize_episode",
]
