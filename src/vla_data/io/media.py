"""Read-only access to Curated v1 external RGB media."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

CAMERA_ROLES = ("head", "right_wrist")


@dataclass(frozen=True)
class CuratedMedia:
    """Resolve and decode Curated v1 RGB frame references."""

    media_root: Path

    def rgb_dir(self, role: str) -> Path:
        if role not in CAMERA_ROLES:
            raise ValueError(f"unsupported camera role: {role!r}")
        return self.media_root / role / "rgb"

    def rgb_path(self, role: str, frame_index: int) -> Path:
        if frame_index < 0:
            raise ValueError(f"RGB frame index must be >= 0, got {frame_index}")

        return self.rgb_dir(role) / f"{frame_index:06d}.jpg"

    def existing_rgb_indices(self, role: str) -> tuple[int, ...]:
        rgb_dir = self.rgb_dir(role)
        if not rgb_dir.is_dir():
            return ()

        indices: list[int] = []
        for path in rgb_dir.glob("*.jpg"):
            try:
                indices.append(int(path.stem))
            except ValueError as exc:
                raise ValueError(
                    f"non-numeric Curated RGB filename: {path.name!r}"
                ) from exc

        return tuple(sorted(indices))

    def read_rgb(self, role: str, frame_index: int) -> np.ndarray:
        path = self.rgb_path(role, frame_index)
        if not path.is_file():
            raise FileNotFoundError(path)

        with Image.open(path) as image:
            rgb = image.convert("RGB")
            array = np.asarray(rgb, dtype=np.uint8).copy()

        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(
                f"decoded RGB image must have shape (H, W, 3), got {array.shape}"
            )

        return array
