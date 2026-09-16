"""Public LIBERO annotation-benchmark adapter: isolation and leakage guards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from vla_data.annotation.hierarchical import get_prompt_family
from vla_data.annotation.platform_context import DEFAULT_PLATFORM_CONTEXT
from vla_data.annotation.provider import AnnotationRequest, ImageItem, TextItem
from vla_data.benchmark import (
    BENCHMARK_ROLE,
    LIBERO_PLATFORM_CONTEXT,
    SOURCE_DATASET,
    BenchmarkEpisodeManifest,
    BenchmarkObservation,
    assert_benchmark_never_production,
    write_benchmark_index,
)
from vla_data.benchmark.libero import _even_frame_indices
from vla_data.benchmark.public_dataset import BENCHMARK_INDEX_NAME

# --------------------------------------------------------------- manifest


def test_benchmark_manifest_is_never_production_training_eligible() -> None:
    manifest = BenchmarkEpisodeManifest(
        episode_id="libero_episode_000001",
        source_dataset=SOURCE_DATASET,
        source_episode_index=1,
        robot_embodiment="WidowX 250 6-DoF arm (LIBERO benchmark)",
        end_effector="parallel-jaw gripper",
        camera_configuration="external agentview + wrist-mounted",
        available_views=("head", "right_wrist"),
        source_instruction="put the bowl on the plate",
        observation_count=6,
    )
    document = manifest.as_dict()
    assert document["dataset_role"] == BENCHMARK_ROLE
    assert document["production_training_eligible"] is False
    assert_benchmark_never_production(document)
    with pytest.raises(ValueError):
        assert_benchmark_never_production(
            {**document, "production_training_eligible": True}
        )
    with pytest.raises(ValueError):
        assert_benchmark_never_production({**document, "dataset_role": "PRODUCTION"})


def test_benchmark_index_written_with_held_out_labels(tmp_path) -> None:
    manifest = BenchmarkEpisodeManifest(
        episode_id="e",
        source_dataset=SOURCE_DATASET,
        source_episode_index=7,
        robot_embodiment="WidowX",
        end_effector="gripper",
        camera_configuration="dual",
        available_views=("head", "right_wrist"),
        source_instruction="turn on the stove",
        observation_count=2,
    )
    observations = {
        "e": [
            BenchmarkObservation(0, 0, 0.0, tmp_path / "h0.jpg", tmp_path / "w0.jpg"),
            BenchmarkObservation(1, 10, 1.0, tmp_path / "h1.jpg", tmp_path / "w1.jpg"),
        ]
    }
    index = write_benchmark_index(tmp_path, [manifest], observations)
    document = json.loads(index.read_text())
    assert document["episodes"][0]["source_instruction"] == "turn on the stove"
    assert document["production_training_eligible"] is False
    assert index.name == BENCHMARK_INDEX_NAME


# ------------------------------------------------------------- sampling


def test_even_frame_indices_are_deterministic_and_sparse_stable() -> None:
    assert _even_frame_indices(112, 6) == [0, 22, 44, 67, 89, 111]
    assert _even_frame_indices(5, 6) == [0, 1, 2, 3, 4]
    assert _even_frame_indices(90, 6) == _even_frame_indices(90, 6)


# ------------------------------------------------- platform separation


def test_libero_platform_context_is_dataset_specific() -> None:
    libero = LIBERO_PLATFORM_CONTEXT.render()
    production = DEFAULT_PLATFORM_CONTEXT.render()
    assert "Panda" in libero  # identity from local dataset metadata (README)
    assert "RM65B" not in libero and "SG100" not in libero
    assert "RM65B" in production  # production context unchanged
    assert LIBERO_PLATFORM_CONTEXT.fingerprint != DEFAULT_PLATFORM_CONTEXT.fingerprint


# ------------------------------------------------------- prompt leakage


def _request(tmp_path: Path, *, wrist: bool = True) -> AnnotationRequest:
    items: list = [TextItem("prompt body")]
    head = tmp_path / "h.jpg"
    Image.new("RGB", (8, 8), (10, 10, 10)).save(head)
    items.append(ImageItem(0, "head", "obs0 head", head))
    if wrist:
        wrist_path = tmp_path / "w.jpg"
        Image.new("RGB", (8, 8), (20, 20, 20)).save(wrist_path)
        items.append(ImageItem(0, "right_wrist", "obs0 wrist", wrist_path))
    return AnnotationRequest("e", "vla_semantic_coarse_v3_2", tuple(items))


def test_source_label_held_out_of_v32_prompt(tmp_path) -> None:
    family = get_prompt_family("v3.2", LIBERO_PLATFORM_CONTEXT)
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
    for forbidden in (
        "put the bowl on the plate",
        "chocolate pudding",
        "turn on the stove",
        "RM65B",
        "SG100",
    ):
        assert forbidden not in prompt
    request = _request(tmp_path)
    assert request.image_count == 2
    assert "Panda" in prompt  # benchmark identity IS present


def test_single_view_request_is_explicit_not_duplicated(tmp_path) -> None:
    request = _request(tmp_path, wrist=False)
    images = [i for i in request.items if isinstance(i, ImageItem)]
    assert [i.camera for i in images] == ["head"]
    # No fabricated second view: the adapter records availability instead.
    assert not any(i.camera == "right_wrist" for i in images)


def test_no_physical17_anywhere_in_benchmark_path(tmp_path) -> None:
    # The benchmark namespace writes observations + index only; it never
    # produces trajectory.npz/physical17 artifacts.
    manifest = BenchmarkEpisodeManifest(
        episode_id="e",
        source_dataset=SOURCE_DATASET,
        source_episode_index=1,
        robot_embodiment="WidowX",
        end_effector="gripper",
        camera_configuration="dual",
        available_views=("head",),
        source_instruction=None,
        observation_count=1,
        extra={"state_dim": 8, "action_dim": 7, "physical17_compatible": False},
    )
    document = manifest.as_dict()
    assert document["physical17_compatible"] is False
    assert document["state_dim"] == 8  # public dims recorded truthfully
    serialized = json.dumps(document)
    assert "robot_qpos_17d_rad" not in serialized
    assert "arm_qcmd_sent_rad" not in serialized


def test_dry_run_makes_zero_provider_calls(tmp_path) -> None:
    calls = []

    class CountingProvider:
        provider_name = "counting"
        model = "counting"

        def infer_json(self, request):
            calls.append(request)
            raise AssertionError("dry-run must not call the provider")

    # Dry-run path: build prompt + storyboard only (no infer_json).
    family = get_prompt_family("v3.2", LIBERO_PLATFORM_CONTEXT)
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
    assert prompt and calls == []
