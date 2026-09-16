"""Benchmark annotation episode source (in-memory, never production Curated).

A benchmark episode dir contains ``benchmark_episode.json`` (the marker) plus
a ``media/`` JPEG cache laid out exactly like Curated media. The loader here
constructs a CuratedEpisode-compatible object IN MEMORY ONLY: no fake
``trajectory.npz`` is ever written, and the trajectory mapping carries only
honest visual-index/timestamp data — never fabricated physical17 vectors.

Index contract (critical for human review):
    benchmark_curated_index == LIBERO source_frame_index == inspection video
    frame index, one-to-one, for every selected episode.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from vla_data.io.curated_episode import CuratedEpisode

BENCHMARK_EPISODE_MARKER = "benchmark_episode.json"
INDEX_SEMANTICS = "benchmark_curated_index_equals_source_frame_index"
BENCHMARK_SCHEMA_NAME = "vla_public_annotation_benchmark_episode"

TRAJECTORY_KEYS = (
    "head_rgb_frame_index",
    "right_wrist_rgb_frame_index",
    "action_timestamp_ns",
)


class BenchmarkEpisode(CuratedEpisode):
    """CuratedEpisode-compatible view over public benchmark observations.

    ``transition_count`` is the source frame count (no physical vector is
    fabricated to derive it). RGB frame indices are the identity over
    source frames, so media paths are ``frame_<source_index>``.
    """

    @property
    def transition_count(self) -> int:  # type: ignore[override]
        return int(self.trajectory["head_rgb_frame_index"].shape[0])


def is_benchmark_episode_dir(episode_dir: str | Path) -> bool:
    return (Path(episode_dir) / BENCHMARK_EPISODE_MARKER).is_file()


def load_benchmark_episode(episode_dir: str | Path) -> BenchmarkEpisode:
    """Build the in-memory benchmark episode from the marker + JPEG cache."""

    root = Path(episode_dir)
    marker_path = root / BENCHMARK_EPISODE_MARKER
    marker = json.loads(marker_path.read_text())
    frame_count = int(marker["source_frame_count"])
    timestamps_ns = np.asarray(marker["timestamps_ns"], dtype=np.int64)
    if timestamps_ns.shape[0] != frame_count:
        raise ValueError(
            f"{marker_path}: {timestamps_ns.shape[0]} timestamps for "
            f"{frame_count} source frames"
        )
    available = tuple(marker.get("available_views", ("head", "right_wrist")))
    identity = np.arange(frame_count, dtype=np.int64)
    trajectory: dict[str, np.ndarray] = {
        "head_rgb_frame_index": identity,
        "right_wrist_rgb_frame_index": identity.copy(),
        "action_timestamp_ns": timestamps_ns,
    }
    for role in available:
        first = root / "media" / role / "rgb" / f"{identity[0]:06d}.jpg"
        last = root / "media" / role / "rgb" / f"{identity[-1]:06d}.jpg"
        if not first.is_file() or not last.is_file():
            raise FileNotFoundError(
                f"{marker_path}: view {role!r} JPEG cache incomplete "
                f"(checked {first.name}..{last.name})"
            )
    metadata: dict[str, Any] = {
        "episode_id": marker["episode_id"],
        "schema_name": BENCHMARK_SCHEMA_NAME,
        "schema_version": 1,
        "camera_roles": list(available),
        "segment_offsets": [0, frame_count],
        "dataset_role": marker.get("dataset_role", "PUBLIC_ANNOTATION_BENCHMARK"),
        "production_training_eligible": False,
        "physical17_compatible": False,
        "index_semantics": INDEX_SEMANTICS,
        "source_dataset": marker.get("source_dataset"),
        "source_episode_index": marker.get("source_episode_index"),
        "fps": marker.get("fps"),
        "source_task_instruction_withheld": True,
        "language_instruction": None,
    }
    episode = BenchmarkEpisode(
        episode_dir=root.resolve(),
        metadata=metadata,  # type: ignore[arg-type]
        trajectory=trajectory,  # type: ignore[arg-type]
    )
    for array in episode.trajectory.values():
        array.setflags(write=False)
    return episode


def write_annotation_quality_gate(
    quality_root: str | Path, episode: BenchmarkEpisode
) -> Path:
    """Benchmark eligibility gate: every source frame is annotatable.

    This is NOT a D3 quality evaluation — public benchmark frames receive no
    production RGB-cleaning exclusion, so the mask is uniformly valid and the
    report records that explicitly.
    """

    episode_root = Path(quality_root) / episode.metadata["episode_id"]
    episode_root.mkdir(parents=True, exist_ok=True)
    count = episode.transition_count
    np.save(episode_root / "quality_mask.npy", np.ones(count, dtype=bool))
    report = {
        "schema_name": "vla_quality_report",
        "schema_version": 1,
        "episode_id": episode.metadata["episode_id"],
        "status": "ACCEPT",
        "transition_count": count,
        "clean_transition_count": count,
        "benchmark_note": (
            "PUBLIC_ANNOTATION_BENCHMARK gate: no D3 production RGB policy "
            "applied; all source frames annotatable by construction"
        ),
        "dataset_role": "PUBLIC_ANNOTATION_BENCHMARK",
        "production_training_eligible": False,
    }
    (episode_root / "quality_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return episode_root
