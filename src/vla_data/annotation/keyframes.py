"""Deterministic, episode-bound keyframe selection from clean transitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vla_data.annotation.prompts import PROMPT_VERSION
from vla_data.annotation.provider import ImageItem
from vla_data.io.curated_episode import CuratedEpisode

DEFAULT_MAX_TEMPORAL_POINTS = 6


class KeyframeError(ValueError):
    """The episode cannot yield a valid keyframe selection."""


@dataclass(frozen=True)
class TemporalKeyframe:
    point: int
    transition_index: int
    timestamp_ns: int
    head_frame_index: int
    head_source_path: Path
    wrist_frame_index: int
    wrist_source_path: Path

    def as_dict(self) -> dict[str, object]:
        return {
            "temporal_point": self.point,
            "transition_index": int(self.transition_index),
            "timestamp_ns": int(self.timestamp_ns),
            "head": {
                "frame_index": int(self.head_frame_index),
                "source_path": str(self.head_source_path),
            },
            "wrist": {
                "frame_index": int(self.wrist_frame_index),
                "source_path": str(self.wrist_source_path),
            },
        }


@dataclass(frozen=True)
class KeyframeSelection:
    episode_id: str
    prompt_version: str
    points: tuple[TemporalKeyframe, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_name": "vla_episode_annotation_keyframes",
            "schema_version": 1,
            "episode_id": self.episode_id,
            "prompt_version": self.prompt_version,
            "temporal_points": [point.as_dict() for point in self.points],
        }


def select_keyframes(
    episode: CuratedEpisode,
    *,
    clean_mask: np.ndarray,
    max_temporal_points: int = DEFAULT_MAX_TEMPORAL_POINTS,
) -> KeyframeSelection:
    """Select deterministic keyframes over clean transitions only.

    Uniform temporal anchors keep start/middle/end coverage; interior anchors
    are then adjusted to the locally strongest observed motion so that the
    actual manipulation phases are represented, and the first/last clean
    transitions always pin the initial scene and the end state.
    """

    if max_temporal_points < 1:
        raise KeyframeError("max_temporal_points must be >= 1")

    clean_indices = np.flatnonzero(np.asarray(clean_mask, dtype=bool))
    if clean_indices.size == 0:
        raise KeyframeError("no clean transitions to select keyframes from")

    transition_count = episode.transition_count
    if clean_indices.size > transition_count or clean_indices[-1] >= transition_count:
        raise KeyframeError("clean mask does not match the trajectory length")

    chosen = _anchor_indices(
        clean_indices, episode.state, max_temporal_points=max_temporal_points
    )

    media = episode.media
    trajectory = episode.trajectory
    points: list[TemporalKeyframe] = []
    for number, index in enumerate(chosen, start=1):
        index = int(index)
        head_index = int(trajectory["head_rgb_frame_index"][index])
        wrist_index = int(trajectory["right_wrist_rgb_frame_index"][index])
        points.append(
            TemporalKeyframe(
                point=number,
                transition_index=index,
                timestamp_ns=int(trajectory["action_timestamp_ns"][index]),
                head_frame_index=head_index,
                head_source_path=media.rgb_path("head", head_index),
                wrist_frame_index=wrist_index,
                wrist_source_path=media.rgb_path("right_wrist", wrist_index),
            )
        )
    for point in points:
        _require_readable(point)
    return KeyframeSelection(
        episode_id=str(episode.metadata["episode_id"]),
        prompt_version=PROMPT_VERSION,
        points=tuple(points),
    )


def _anchor_indices(
    clean_indices: np.ndarray,
    state: np.ndarray,
    *,
    max_temporal_points: int,
) -> tuple[int, ...]:
    """Uniform anchors adjusted by observed motion; deterministic tie-breaks."""

    total = clean_indices.size
    if total <= max_temporal_points:
        return tuple(int(value) for value in clean_indices)

    anchor_count = max_temporal_points
    uniform_positions = [
        round((anchor / (anchor_count - 1)) * (total - 1))
        for anchor in range(anchor_count)
    ]

    motion = _motion_magnitude(state)
    chosen: list[int] = [int(clean_indices[uniform_positions[0]])]
    for anchor in range(1, anchor_count - 1):
        # Non-overlapping uniform segments around each anchor keep the six
        # picks distinct even when one motion peak dominates the episode.
        lower = (uniform_positions[anchor - 1] + uniform_positions[anchor]) // 2
        upper = (uniform_positions[anchor] + uniform_positions[anchor + 1]) // 2 - 1
        if upper < lower:
            candidate = int(clean_indices[uniform_positions[anchor]])
        else:
            window = clean_indices[lower : upper + 1]
            # Deterministic: max motion, earliest transition wins ties.
            candidate = int(window[int(np.argmax(motion[window]))])
        if candidate not in chosen:
            chosen.append(candidate)
    final_anchor = int(clean_indices[uniform_positions[-1]])
    if final_anchor not in chosen:
        chosen.append(final_anchor)
    return tuple(sorted(chosen))


def _motion_magnitude(state: np.ndarray) -> np.ndarray:
    """Per-transition observed motion magnitude (zero for the last row)."""

    motion = np.zeros(state.shape[0], dtype=np.float64)
    if state.shape[0] > 1:
        motion[:-1] = np.linalg.norm(np.diff(state, axis=0), axis=1)
    return motion


def _require_readable(point: TemporalKeyframe) -> None:
    for path in (point.head_source_path, point.wrist_source_path):
        if not Path(path).is_file():
            raise KeyframeError(f"referenced Curated image is missing: {path}")


def keyframe_image_items(selection: KeyframeSelection) -> tuple[ImageItem, ...]:
    """Build ordered (head, wrist) image items preserving temporal order."""

    items: list[ImageItem] = []
    for point in selection.points:
        items.append(
            ImageItem(
                temporal_point=point.point,
                camera="head",
                label=f"Temporal point {point.point} — head camera view",
                source_path=point.head_source_path,
            )
        )
        items.append(
            ImageItem(
                temporal_point=point.point,
                camera="right_wrist",
                label=f"Temporal point {point.point} — wrist camera view",
                source_path=point.wrist_source_path,
            )
        )
    return tuple(items)
