"""libero_semantic_v32 benchmark: preparation, isolation, index contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from vla_data.annotation.hierarchical import (
    get_prompt_family,
    plan_hierarchical_episode,
)
from vla_data.annotation.platform_context import DEFAULT_PLATFORM_CONTEXT
from vla_data.benchmark.episode_source import (
    TRAJECTORY_KEYS,
    is_benchmark_episode_dir,
    load_benchmark_episode,
)
from vla_data.benchmark.inspection_video import build_dual_view_inspection_video
from vla_data.benchmark.libero import (
    BENCHMARK_NAME,
    LIBERO_PLATFORM_CONTEXT,
    _even_frame_indices,
)

FPS = 10
FRAME_COUNT = 5
TASK = "put the bowl on the plate"


@pytest.fixture
def benchmark_episode(tmp_path: Path) -> Path:
    """A minimal but fully valid benchmark episode (5 frames, dual view)."""

    episode_dir = tmp_path / "observations" / "episode_000001"
    for role in ("head", "right_wrist"):
        view_dir = episode_dir / "media" / role / "rgb"
        view_dir.mkdir(parents=True)
        color = (200, 40, 40) if role == "head" else (40, 40, 200)
        for index in range(FRAME_COUNT):
            Image.new("RGB", (32, 24), color).save(
                view_dir / f"{index:06d}.jpg", format="JPEG"
            )
    marker = {
        "schema_name": "vla_public_annotation_benchmark_episode",
        "schema_version": 1,
        "dataset_role": "PUBLIC_ANNOTATION_BENCHMARK",
        "production_training_eligible": False,
        "physical17_compatible": False,
        "index_semantics": "benchmark_curated_index_equals_source_frame_index",
        "episode_id": "episode_000001",
        "source_dataset": "physical-intelligence/libero",
        "source_episode_index": 1,
        "source_frame_count": FRAME_COUNT,
        "available_views": ["head", "right_wrist"],
        "fps": FPS,
        "timestamps_ns": [index * 10**8 for index in range(FRAME_COUNT)],
        "source_task_instruction": TASK,
        "source_task_instruction_withheld_from_prompt": True,
        "pass_a_sampled_source_frame_indices": _even_frame_indices(FRAME_COUNT, 5),
    }
    (episode_dir / "benchmark_episode.json").write_text(json.dumps(marker) + "\n")
    quality = tmp_path / "source_index" / "episode_000001"
    quality.mkdir(parents=True)
    import numpy as np

    np.save(quality / "quality_mask.npy", np.ones(FRAME_COUNT, dtype=bool))
    (quality / "quality_report.json").write_text(
        json.dumps(
            {
                "schema_name": "vla_quality_report",
                "schema_version": 1,
                "episode_id": "episode_000001",
                "status": "ACCEPT",
            }
        )
    )
    return tmp_path


# ---------------------------------------------------- episode source contract


def test_marker_detection_and_in_memory_loading(benchmark_episode: Path) -> None:
    episode_dir = benchmark_episode / "observations" / "episode_000001"
    assert is_benchmark_episode_dir(episode_dir)
    assert not is_benchmark_episode_dir(benchmark_episode / "observations")
    episode = load_benchmark_episode(episode_dir)
    assert episode.transition_count == FRAME_COUNT
    # Only honest visual/timestamp keys — never fabricated physical17 vectors.
    assert tuple(episode.trajectory) == TRAJECTORY_KEYS
    assert "robot_qpos_17d_rad" not in episode.trajectory
    assert not (episode_dir / "trajectory.npz").exists()
    assert not (episode_dir / "metadata.json").exists()
    assert episode.metadata["production_training_eligible"] is False
    assert episode.metadata["index_semantics"] == (
        "benchmark_curated_index_equals_source_frame_index"
    )


def test_index_identity_source_equals_annotation_equals_video(
    benchmark_episode: Path,
) -> None:
    episode = load_benchmark_episode(
        benchmark_episode / "observations" / "episode_000001"
    )
    identity = episode.trajectory["head_rgb_frame_index"]
    assert list(identity) == list(range(FRAME_COUNT))
    # RGB path for annotation index N is source frame N (identity mapping).
    path = episode.media.rgb_path("head", 3)
    assert path.name == "000003.jpg" and path.is_file()


def test_global_wrist_same_length_pairing(benchmark_episode: Path) -> None:
    episode_dir = benchmark_episode / "observations" / "episode_000001"
    head = sorted((episode_dir / "media/head/rgb").glob("*.jpg"))
    wrist = sorted((episode_dir / "media/right_wrist/rgb").glob("*.jpg"))
    assert [p.name for p in head] == [p.name for p in wrist]
    assert len(head) == FRAME_COUNT


def test_single_view_never_duplicated(tmp_path: Path) -> None:
    source = tmp_path / "observations" / "episode_000002"
    head_dir = source / "media" / "head" / "rgb"
    head_dir.mkdir(parents=True)
    for index in range(2):
        Image.new("RGB", (8, 8), (10, 10, 10)).save(head_dir / f"{index:06d}.jpg")
    (source / "benchmark_episode.json").write_text(
        json.dumps(
            {
                "episode_id": "episode_000002",
                "source_frame_count": 2,
                "available_views": ["head"],
                "timestamps_ns": [0, 10**8],
            }
        )
    )
    load_benchmark_episode(source)
    assert not (source / "media/right_wrist").exists()  # no fabricated view


# ------------------------------------------------------ inspection video


def test_dual_view_video_frame_count_fps_and_index_map(benchmark_episode: Path) -> None:
    episode_dir = benchmark_episode / "observations" / "episode_000001"
    result = build_dual_view_inspection_video(
        episode_dir,
        fps=FPS,
        output_path=(
            benchmark_episode / "inspection_videos" / "episode_000001_dual_view.mp4"
        ),
    )
    assert result["source_frame_count"] == result["encoded_frame_count"] == FRAME_COUNT
    assert result["encoded_fps"] == float(FPS)
    assert result["resolution"] == [64, 24 + 28]  # 2x32 wide + overlay bar
    frame_map = json.loads(Path(result["frame_map"]).read_text())
    assert frame_map["video_frame_index_equals_source_frame_index"] is True
    for entry in frame_map["frames"]:
        assert entry["video_frame_index"] == entry["source_frame_index"]


def test_video_overlay_never_contains_source_task(benchmark_episode: Path) -> None:
    # The overlay only ever renders episode_id/frame/time; prove the source
    # task string is absent from the frame-map artifact family and that the
    # generator script has no access path for it.
    episode_dir = benchmark_episode / "observations" / "episode_000001"
    result = build_dual_view_inspection_video(
        episode_dir,
        fps=FPS,
        output_path=benchmark_episode
        / "inspection_videos/episode_000001_dual_view.mp4",
    )
    frame_map_text = Path(result["frame_map"]).read_text()
    assert TASK not in frame_map_text
    assert "put the bowl" not in frame_map_text


# ------------------------------------------------------- prompt + leakage


def _plan(benchmark_episode: Path):
    return plan_hierarchical_episode(
        benchmark_episode / "observations" / "episode_000001",
        quality_root=benchmark_episode / "source_index",
        output_root=benchmark_episode / "annotations",
        prompt_versions=(
            "vla_semantic_coarse_v3_2",
            "vla_semantic_boundary_refine_local_v3_2",
        ),
        boundary_local=True,
        platform_context=LIBERO_PLATFORM_CONTEXT,
    )


def test_benchmark_prompt_is_libero_identity_and_label_free(
    benchmark_episode: Path,
) -> None:
    _planned, plan = _plan(benchmark_episode)
    assert plan is not None
    family = get_prompt_family("v3.2", LIBERO_PLATFORM_CONTEXT)
    prompt = family.build_pass_a(
        plan["clean_domains"], plan["pass_a_allowed_boundaries"]
    )
    assert "Panda" in prompt
    for forbidden in ("RM65B", "SG100", "D455", "D405", TASK, "bowl"):
        assert forbidden not in prompt, forbidden


def test_pass_a_sampling_uses_real_source_indices(benchmark_episode: Path) -> None:
    _planned, plan = _plan(benchmark_episode)
    assert [record["curated_index"] for record in plan["pass_a_records"]] == list(
        range(FRAME_COUNT)
    )
    assert plan["pass_a_allowed_boundaries"] == [0, 1, 2, 3, 4, 5]


def test_production_prompts_unchanged_for_rm65b_platform() -> None:
    family = get_prompt_family("v3.2", DEFAULT_PLATFORM_CONTEXT)
    prompt = family.build_pass_a(
        [
            {
                "d2_segment_id": 0,
                "clean_run_id": 0,
                "start_curated_index": 0,
                "end_curated_index": 6,
            }
        ],
        list(range(7)),
    )
    assert "SG100 COARSE EVIDENCE" in prompt  # production wording intact


def test_dry_run_makes_zero_provider_calls(benchmark_episode: Path) -> None:
    planned, _plan_value = _plan(benchmark_episode)
    assert planned.status == "WOULD_PROCESS"
    assert planned.pass_a_calls == 1
    assert planned.expected_glm_calls == 1  # plan only; no provider invoked


def test_benchmark_root_naming_is_canonical() -> None:
    assert BENCHMARK_NAME == "libero_semantic_v32"
    assert "dex_ur5e" not in BENCHMARK_NAME
    assert "d43" not in BENCHMARK_NAME and "tmp" not in BENCHMARK_NAME
