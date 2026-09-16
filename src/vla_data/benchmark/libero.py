"""LIBERO semantic benchmark adapter (physical-intelligence/libero).

Canonical identifiers:
    dataset_id     = physical-intelligence/libero
    benchmark_name = libero_semantic_v32
    dataset_role   = PUBLIC_ANNOTATION_BENCHMARK

The previously referenced ``dex_ur5e`` dataset does not exist on this
machine; every artifact here is named after the actual source (LIBERO).

Robot identity comes from the LOCAL dataset metadata (README robot_type:
"panda"); the end effector is not further specified by the metadata, so the
generic "parallel gripper" is used. Nothing is guessed.

Source task labels are HELD OUT of the annotation prompt: they are stored
under ``source_index/`` for post-annotation evaluation only.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from vla_data.annotation.platform_context import PlatformContext
from vla_data.benchmark.episode_source import (
    BENCHMARK_EPISODE_MARKER,
    BENCHMARK_SCHEMA_NAME,
    INDEX_SEMANTICS,
)
from vla_data.benchmark.public_dataset import (
    BENCHMARK_ROLE,
    BenchmarkEpisodeManifest,
    BenchmarkObservation,
    write_benchmark_index,
)

SOURCE_DATASET = "physical-intelligence/libero"
BENCHMARK_NAME = "libero_semantic_v32"

# Local dataset evidence: README robot_type "panda"; timestamps step 0.1 s
# (verified from parquet) => 10 fps.
LIBERO_FPS = 10

LIBERO_PLATFORM_CONTEXT = PlatformContext(
    robot_arm="Panda (LIBERO dataset metadata: robot_type=panda)",
    robot_arm_dof=7,
    end_effector="parallel gripper",
    end_effector_dof=1,
    head_camera_role="external_global_agentview_camera",
    head_camera_device="LIBERO agentview (rendered)",
    right_wrist_camera_role="robot_mounted_wrist_camera",
    right_wrist_camera_device="LIBERO wrist camera (rendered)",
    collection_mode="offline rendered manipulation demonstrations",
)

WORKER_SOURCE = Path(__file__).with_name("_bench_worker.py")
DEFAULT_BENCH_PYTHON = "/data/projects/vla_ws/openpi/.venv/bin/python"


def _episode_records(root: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    for line in (root / "meta" / "episodes.jsonl").read_text().splitlines():
        record = json.loads(line)
        records[int(record["episode_index"])] = record
    return records


def _parquet_path(root: Path, episode_index: int) -> Path:
    for chunk in sorted((root / "data").glob("chunk-*")):
        candidate = chunk / f"episode_{episode_index:06d}.parquet"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"LIBERO episode {episode_index} parquet not found under {root}/data"
    )


def _even_frame_indices(frame_count: int, sample_count: int) -> list[int]:
    if frame_count <= sample_count:
        return list(range(frame_count))
    return sorted(
        {
            round(step * (frame_count - 1) / (sample_count - 1))
            for step in range(sample_count)
        }
    )


def _run_extraction_worker(specs: list[dict], bench_python: str | None) -> list[dict]:
    """Extract full-frame JPEG caches in the pyarrow-capable interpreter."""

    interpreter = bench_python or DEFAULT_BENCH_PYTHON
    completed = subprocess.run(
        [interpreter, str(WORKER_SOURCE)],
        input=json.dumps({"episodes": specs}),
        capture_output=True,
        text=True,
        check=True,
        timeout=1800,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list):
        raise TypeError("benchmark extraction worker returned no list")
    return payload


def prepare_libero_benchmark(
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    episode_indices: list[int],
    max_temporal_points: int = 6,
    bench_python: str | None = None,
) -> dict[str, object]:
    """Prepare the full libero_semantic_v32 benchmark root.

    Layout (the filesystem is self-explanatory):
        observations/episode_XXXXXX/media/{head,right_wrist}/rgb/NNNNNN.jpg
        observations/episode_XXXXXX/benchmark_episode.json   (marker+metadata)
        source_index/benchmark_index.json                    (labels held out)
        source_index/episode_XXXXXX/quality_{mask.npy,report.json}
    """

    root = Path(dataset_root)
    out = Path(output_root)
    records = _episode_records(root)

    specs: list[dict] = []
    for episode_index in sorted(episode_indices):
        record = records.get(episode_index)
        if record is None:
            raise FileNotFoundError(f"LIBERO episode {episode_index} not in meta")
        frame_count = int(record["length"])
        episode_id = f"episode_{episode_index:06d}"
        episode_dir = out / "observations" / episode_id
        view_dirs = {
            role: episode_dir / "media" / role / "rgb"
            for role in ("head", "right_wrist")
        }
        for view_dir in view_dirs.values():
            view_dir.mkdir(parents=True, exist_ok=True)
        specs.append(
            {
                "episode_index": episode_index,
                "parquet": str(_parquet_path(root, episode_index)),
                "frame_count": frame_count,
                "view_dirs": {role: str(path) for role, path in view_dirs.items()},
            }
        )

    extracted = _run_extraction_worker(specs, bench_python)
    timestamps_by_index = {
        int(payload["episode_index"]): payload["timestamps_ns"] for payload in extracted
    }

    manifests: list[BenchmarkEpisodeManifest] = []
    observations: dict[str, list[BenchmarkObservation]] = {}
    for spec in specs:
        episode_index = int(spec["episode_index"])
        record = records[episode_index]
        frame_count = int(record["length"])
        episode_id = f"episode_{episode_index:06d}"
        episode_dir = out / "observations" / episode_id
        timestamps_ns = timestamps_by_index[episode_index]
        sampled = _even_frame_indices(frame_count, max_temporal_points)
        source_task = record["tasks"][0] if record.get("tasks") else None
        marker = {
            "schema_name": BENCHMARK_SCHEMA_NAME,
            "schema_version": 1,
            "dataset_role": BENCHMARK_ROLE,
            "production_training_eligible": False,
            "physical17_compatible": False,
            "index_semantics": INDEX_SEMANTICS,
            "episode_id": episode_id,
            "source_dataset": SOURCE_DATASET,
            "source_episode_index": episode_index,
            "source_frame_count": frame_count,
            "available_views": ["head", "right_wrist"],
            "camera_view_mapping": {
                "head": "LIBERO image (external global agentview)",
                "right_wrist": "LIBERO wrist_image (robot-mounted wrist)",
            },
            "fps": LIBERO_FPS,
            "timestamps_ns": timestamps_ns,
            "source_task_instruction": source_task,
            "source_task_instruction_withheld_from_prompt": True,
            "pass_a_sampled_source_frame_indices": sampled,
        }
        (episode_dir / BENCHMARK_EPISODE_MARKER).write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n"
        )
        manifests.append(
            BenchmarkEpisodeManifest(
                episode_id=episode_id,
                source_dataset=SOURCE_DATASET,
                source_episode_index=episode_index,
                robot_embodiment="Panda (LIBERO dataset metadata)",
                end_effector="parallel gripper",
                camera_configuration="external agentview + wrist-mounted",
                available_views=("head", "right_wrist"),
                source_instruction=source_task,
                observation_count=frame_count,
                extra={
                    "benchmark_name": BENCHMARK_NAME,
                    "source_frame_count": frame_count,
                    "pass_a_sampled_source_frame_indices": sampled,
                    "fps": LIBERO_FPS,
                    "state_dim": 8,
                    "action_dim": 7,
                    "physical17_compatible": False,
                    "index_semantics": INDEX_SEMANTICS,
                },
            )
        )
        observations[episode_id] = [
            BenchmarkObservation(
                benchmark_index=source_index,
                source_frame_index=source_index,
                timestamp_s=timestamps_ns[source_index] / 1e9,
                head_path=episode_dir
                / "media"
                / "head"
                / "rgb"
                / f"{source_index:06d}.jpg",
                right_wrist_path=episode_dir
                / "media"
                / "right_wrist"
                / "rgb"
                / f"{source_index:06d}.jpg",
            )
            for source_index in range(frame_count)
        ]

    source_index = out / "source_index"
    index_path = write_benchmark_index(source_index, manifests, observations)

    from vla_data.benchmark.episode_source import (
        load_benchmark_episode,
        write_annotation_quality_gate,
    )

    for episode_index in sorted(episode_indices):
        episode_id = f"episode_{episode_index:06d}"
        episode = load_benchmark_episode(out / "observations" / episode_id)
        write_annotation_quality_gate(source_index, episode)

    return {
        "benchmark_name": BENCHMARK_NAME,
        "output_root": str(out),
        "source_index": str(index_path),
        "episodes": [
            {
                "episode_id": manifest.episode_id,
                "source_frame_count": manifest.observation_count,
                "pass_a_sampled_source_frame_indices": manifest.extra[
                    "pass_a_sampled_source_frame_indices"
                ],
                "source_task_instruction": manifest.source_instruction,
            }
            for manifest in manifests
        ],
    }
