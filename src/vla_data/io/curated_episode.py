"""Read-only loader for persisted Curated v1 episodes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from vla_data.io.media import CuratedMedia
from vla_data.morphology.rm65_sg100 import ARM_SLICE, HAND_SLICE


@dataclass(frozen=True)
class CuratedEpisode:
    """In-memory read-only view of one Curated v1 episode."""

    episode_dir: Path
    metadata: Mapping[str, object]
    trajectory: Mapping[str, np.ndarray]

    @classmethod
    def load(cls, episode_dir: str | Path) -> CuratedEpisode:
        root = Path(episode_dir)

        if not root.is_dir():
            raise FileNotFoundError(root)

        metadata_path = root / "metadata.json"
        trajectory_path = root / "trajectory.npz"
        media_root = root / "media"

        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        if not trajectory_path.is_file():
            raise FileNotFoundError(trajectory_path)
        if not media_root.is_dir():
            raise FileNotFoundError(media_root)

        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata_raw = json.load(stream)

        if not isinstance(metadata_raw, dict):
            raise TypeError("metadata.json root must be a JSON object")

        with np.load(trajectory_path, allow_pickle=False) as archive:
            trajectory_raw = {key: np.asarray(archive[key]) for key in archive.files}

        # Prevent accidental mutation of the persisted-data view.
        for array in trajectory_raw.values():
            array.setflags(write=False)

        return cls(
            episode_dir=root.resolve(),
            metadata=MappingProxyType(metadata_raw),
            trajectory=MappingProxyType(trajectory_raw),
        )

    @property
    def media(self) -> CuratedMedia:
        return CuratedMedia(self.episode_dir / "media")

    @property
    def transition_count(self) -> int:
        state = self.trajectory.get("robot_qpos_17d_rad")
        if state is None:
            return 0
        return int(state.shape[0])

    @property
    def state(self) -> np.ndarray:
        return self.trajectory["robot_qpos_17d_rad"]

    @property
    def action(self) -> np.ndarray:
        return self.trajectory["robot_qcmd_17d_rad"]

    @property
    def arm_state(self) -> np.ndarray:
        start, end = ARM_SLICE
        return self.state[:, start:end]

    @property
    def hand_state(self) -> np.ndarray:
        start, end = HAND_SLICE
        return self.state[:, start:end]

    @property
    def arm_action(self) -> np.ndarray:
        start, end = ARM_SLICE
        return self.action[:, start:end]

    @property
    def hand_action(self) -> np.ndarray:
        start, end = HAND_SLICE
        return self.action[:, start:end]

    @property
    def segment_offsets(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.metadata["segment_offsets"])

    @property
    def camera_roles(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.metadata["camera_roles"])

    @property
    def language_instruction(self) -> str:
        return str(self.metadata["language_instruction"])
