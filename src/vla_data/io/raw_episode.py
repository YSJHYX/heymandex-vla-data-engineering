"""Read-only access to one native RAW recorder episode."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True)
class RawEpisode:
    """Immutable in-memory view of a RAW ``.npz`` and its media directory."""

    path: Path
    media_root: Path
    arrays: Mapping[str, np.ndarray]

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        media_root: str | Path | None = None,
    ) -> RawEpisode:
        raw_path = Path(path)
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)

        resolved_media = (
            Path(media_root)
            if media_root is not None
            else raw_path.with_name(f"{raw_path.stem}_media")
        )
        if not resolved_media.is_dir():
            raise FileNotFoundError(resolved_media)

        with np.load(raw_path, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}

        for array in arrays.values():
            array.setflags(write=False)

        return cls(
            path=raw_path.resolve(),
            media_root=resolved_media.resolve(),
            arrays=MappingProxyType(arrays),
        )

    def require(self, key: str) -> np.ndarray:
        try:
            return self.arrays[key]
        except KeyError as exc:
            raise KeyError(f"RAW episode is missing required field {key!r}") from exc

    def scalar(self, key: str) -> object:
        value = self.require(key)
        if value.ndim != 0:
            raise ValueError(f"RAW field {key!r} must be scalar, got {value.shape}")
        return value.item()

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self.arrays)

    @property
    def metadata(self) -> Mapping[str, object]:
        values = {
            key: array.item() for key, array in self.arrays.items() if array.ndim == 0
        }
        return MappingProxyType(values)

    @property
    def episode_id(self) -> str:
        return self.path.stem
