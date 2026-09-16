"""Public-dataset D4 semantic annotation benchmark (isolated, non-production).

These adapters evaluate the annotation pipeline on standardized public
manipulation data. They NEVER feed production D2/D3/D6: benchmark outputs
carry ``production_training_eligible=false`` and live in a namespace fully
separate from production manifests/exports. Public state/action vectors
are never reshaped into the RM65B+SG100 physical17 contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BENCHMARK_ROLE = "PUBLIC_ANNOTATION_BENCHMARK"
PRODUCTION_TRAINING_ELIGIBLE = False
BENCHMARK_INDEX_NAME = "benchmark_index.json"


@dataclass(frozen=True)
class BenchmarkObservation:
    """One sampled dual/single-view observation from a public episode."""

    benchmark_index: int
    source_frame_index: int
    timestamp_s: float | None
    head_path: Path | None = None
    right_wrist_path: Path | None = None


@dataclass(frozen=True)
class BenchmarkEpisodeManifest:
    """Non-production metadata for one benchmark episode."""

    episode_id: str
    source_dataset: str
    source_episode_index: int
    robot_embodiment: str
    end_effector: str
    camera_configuration: str
    available_views: tuple[str, ...]
    source_instruction: str | None
    observation_count: int
    production_training_eligible: bool = PRODUCTION_TRAINING_ELIGIBLE
    dataset_role: str = BENCHMARK_ROLE
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_role": self.dataset_role,
            "production_training_eligible": self.production_training_eligible,
            "source_dataset": self.source_dataset,
            "episode_id": self.episode_id,
            "source_episode_index": self.source_episode_index,
            "robot_embodiment": self.robot_embodiment,
            "end_effector": self.end_effector,
            "camera_configuration": self.camera_configuration,
            "available_views": list(self.available_views),
            "source_instruction": self.source_instruction,
            "has_ground_truth_language": self.source_instruction is not None,
            "observation_count": self.observation_count,
            **self.extra,
        }


def write_benchmark_index(
    output_root: Path,
    manifests: list[BenchmarkEpisodeManifest],
    observations: dict[str, list[BenchmarkObservation]],
) -> Path:
    """Persist the benchmark index (view paths + held-out source labels)."""

    document = {
        "schema_name": "vla_public_annotation_benchmark_index",
        "schema_version": 1,
        "dataset_role": BENCHMARK_ROLE,
        "production_training_eligible": PRODUCTION_TRAINING_ELIGIBLE,
        "episodes": [
            {
                **manifest.as_dict(),
                "observations": [
                    {
                        "benchmark_index": observation.benchmark_index,
                        "source_frame_index": observation.source_frame_index,
                        "timestamp_s": observation.timestamp_s,
                        "head": (
                            str(observation.head_path)
                            if observation.head_path
                            else None
                        ),
                        "right_wrist": (
                            str(observation.right_wrist_path)
                            if observation.right_wrist_path
                            else None
                        ),
                    }
                    for observation in observations.get(manifest.episode_id, [])
                ],
            }
            for manifest in manifests
        ],
    }
    destination = Path(output_root) / BENCHMARK_INDEX_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return destination


def assert_benchmark_never_production(metadata: dict[str, Any]) -> None:
    """Deterministic guard: benchmark inputs are never production-exported."""

    if metadata.get("dataset_role") != BENCHMARK_ROLE:
        raise ValueError("benchmark guard: missing PUBLIC_ANNOTATION_BENCHMARK role")
    if metadata.get("production_training_eligible") is not False:
        raise ValueError("benchmark guard: production_training_eligible must be false")
