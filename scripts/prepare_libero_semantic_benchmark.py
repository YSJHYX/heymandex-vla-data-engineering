"""Prepare the libero_semantic_v32 public annotation benchmark.

One short command: full-frame JPEG caches, benchmark markers, quality gates,
held-out source index, and dual-view inspection videos (frame count and fps
verified) for the five selected LIBERO episodes. Zero provider calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_data.benchmark.inspection_video import build_dual_view_inspection_video
from vla_data.benchmark.libero import (
    BENCHMARK_NAME,
    LIBERO_FPS,
    prepare_libero_benchmark,
)

DEFAULT_DATASET_ROOT = "/data/huggingface_cache/lerobot/physical-intelligence/libero"
DEFAULT_OUTPUT_ROOT = f"/data/vla_runs/public_benchmarks/{BENCHMARK_NAME}"
DEFAULT_EPISODES = "823,388,379,385,1280"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--episodes", default=DEFAULT_EPISODES)
    parser.add_argument(
        "--bench-python",
        default="/data/projects/vla_ws/openpi/.venv/bin/python",
        help="interpreter with pyarrow for parquet extraction",
    )
    args = parser.parse_args()
    episode_indices = [int(value) for value in args.episodes.split(",")]

    summary = prepare_libero_benchmark(
        args.dataset_root,
        args.output_root,
        episode_indices=episode_indices,
        bench_python=args.bench_python,
    )
    output_root = Path(args.output_root)
    videos = []
    for record in summary["episodes"]:
        video = build_dual_view_inspection_video(
            output_root / "observations" / record["episode_id"],
            fps=LIBERO_FPS,
            output_path=(
                output_root
                / "inspection_videos"
                / f"{record['episode_id']}_dual_view.mp4"
            ),
        )
        videos.append(video)

    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "preparation_summary.json").write_text(
        json.dumps({**summary, "inspection_videos": videos}, indent=2) + "\n"
    )
    for record, video in zip(summary["episodes"], videos, strict=True):
        (reports / f"{record['episode_id']}_review.json").write_text(
            json.dumps(
                {
                    "episode_id": record["episode_id"],
                    "inspection_video": video["video"],
                    "inspection_frame_map": video["frame_map"],
                    "source_task_instruction": record["source_task_instruction"],
                    "predicted_episode_task": None,
                    "semantic_segments": None,
                    "index_semantics": "benchmark_curated_index_equals_source_frame_index",
                    "diagnostics": None,
                },
                indent=2,
            )
            + "\n"
        )

    print(json.dumps(summary, indent=2))
    for video in videos:
        print(
            f"{video['episode_id']}: {video['encoded_frame_count']} frames @ "
            f"{video['encoded_fps']}fps -> {video['video']}"
        )
    print("source task loaded for evaluation: yes")
    print("source task included in model request: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
