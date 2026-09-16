"""D4.2.3 adaptive annotation-storyboard transport budget."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.annotation.test_mcp_provider import FakeMCPClient, _ok_result, _tool_schema
from vla_data.annotation.mcp_provider import (
    MCP_IMAGE_SIZE_REJECTED,
    TARGET_MCP_IMAGE_BYTES,
    CodingPlanVisionMCPProvider,
    MCPProviderConfig,
)
from vla_data.annotation.provider import (
    AnnotationRequest,
    ImageItem,
    ProviderError,
    TextItem,
)
from vla_data.annotation.storyboard import (
    DEFAULT_JPEG_QUALITY,
    STORYBOARD_SIZE_LIMIT,
    MultiViewObservation,
    fit_storyboard_to_budget,
)

PAYLOAD = json.dumps(
    {
        "episode_task": {
            "instruction": "Place the cable into the tray.",
            "confidence": 0.9,
        },
        "semantic_segments": [],
        "non_training_intervals": [],
    }
)


def _entropy_observations(
    tmp_path: Path, count: int, *, seed: int = 42
) -> list[MultiViewObservation]:
    rng = np.random.default_rng(seed)
    observations = []
    for curated_index in range(count):
        paths = []
        for role in ("head", "right_wrist"):
            path = tmp_path / f"{role}_{curated_index}.png"
            pixels = rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(path)
            paths.append(path)
        observations.append(
            MultiViewObservation(curated_index, curated_index * 10, *paths)
        )
    return observations


def _solid_observations(tmp_path: Path, count: int) -> list[MultiViewObservation]:
    observations = []
    for curated_index in range(count):
        paths = []
        for role, color in (("head", (200, 40, 40)), ("right_wrist", (40, 40, 200))):
            path = tmp_path / f"{role}_{curated_index}.jpg"
            Image.new("RGB", (640, 360), color).save(path, format="JPEG")
            paths.append(path)
        observations.append(MultiViewObservation(curated_index, 0, *paths))
    return observations


def _source_hashes(observations: list[MultiViewObservation]) -> dict[Path, str]:
    paths = [
        path
        for observation in observations
        for path in (observation.head_path, observation.right_wrist_path)
    ]
    return {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _request(observations: list[MultiViewObservation]) -> AnnotationRequest:
    items: list[TextItem | ImageItem] = [TextItem("Pass B prompt")]
    for observation in observations:
        items.extend(
            (
                ImageItem(
                    observation.curated_index,
                    "head",
                    "head",
                    observation.head_path,
                ),
                ImageItem(
                    observation.curated_index,
                    "right_wrist",
                    "right wrist",
                    observation.right_wrist_path,
                ),
            )
        )
    return AnnotationRequest(
        episode_id="episode_000002",
        prompt_version="vla_semantic_boundary_refine_v2",
        items=tuple(items),
    )


def _provider(tmp_path: Path, observations, responses, **config):
    client = FakeMCPClient(
        [
            _tool_schema(
                {"image_source": {"type": "string"}, "prompt": {"type": "string"}},
                name="analyze_image",
            )
        ],
        responses,
    )
    provider = CodingPlanVisionMCPProvider(
        MCPProviderConfig(storyboard_root=tmp_path / "boards", **config),
        client=client,
        api_key="team-key",
    )
    return provider, _request(observations)


def test_exact_515mb_regression_fits_without_changing_temporal_evidence(
    tmp_path,
) -> None:
    observations = _entropy_observations(tmp_path, 12)
    identity_before = [
        (item.curated_index, item.head_path, item.right_wrist_path)
        for item in observations
    ]
    hashes_before = _source_hashes(observations)

    artifact = fit_storyboard_to_budget(
        observations,
        tmp_path / "boards",
        target_max_bytes=TARGET_MCP_IMAGE_BYTES,
        episode_id="episode_000002",
        storyboard_pass="pass_b",
    )
    metadata = artifact.transport_metadata

    assert 5_000_000 < metadata["storyboard_initial_bytes"] < 5_300_000
    assert metadata["storyboard_final_bytes"] <= TARGET_MCP_IMAGE_BYTES
    assert metadata["storyboard_observation_count"] == 12
    assert metadata["storyboard_scale"] == 1.0
    assert metadata["storyboard_jpeg_quality"] < DEFAULT_JPEG_QUALITY
    assert identity_before == [
        (item.curated_index, item.head_path, item.right_wrist_path)
        for item in observations
    ]
    assert hashes_before == _source_hashes(observations)


def test_under_budget_storyboard_keeps_default_encoding_path(tmp_path) -> None:
    observations = _solid_observations(tmp_path, 2)
    artifact = fit_storyboard_to_budget(
        observations,
        tmp_path / "boards",
        target_max_bytes=TARGET_MCP_IMAGE_BYTES,
        storyboard_pass="pass_a",
    )
    metadata = artifact.transport_metadata
    assert artifact.within_budget
    assert metadata["storyboard_initial_bytes"] == metadata["storyboard_final_bytes"]
    assert metadata["storyboard_jpeg_quality"] == DEFAULT_JPEG_QUALITY
    assert metadata["storyboard_scale"] == 1.0
    assert metadata["storyboard_adaptive_compression_applied"] is False


def test_quality_only_fitting_precedes_resolution_reduction(tmp_path) -> None:
    observations = _entropy_observations(tmp_path, 12)
    artifact = fit_storyboard_to_budget(
        observations,
        tmp_path / "boards",
        target_max_bytes=TARGET_MCP_IMAGE_BYTES,
        storyboard_pass="pass_b",
    )
    metadata = artifact.transport_metadata
    assert artifact.within_budget
    assert metadata["storyboard_jpeg_quality"] == 88
    assert metadata["storyboard_scale"] == 1.0


def test_resolution_fallback_activates_only_after_quality_floor(tmp_path) -> None:
    observations = _entropy_observations(tmp_path, 4)
    artifact = fit_storyboard_to_budget(
        observations,
        tmp_path / "boards",
        target_max_bytes=800_000,
        storyboard_pass="pass_b",
    )
    metadata = artifact.transport_metadata
    assert artifact.within_budget
    assert metadata["storyboard_jpeg_quality"] == 60
    assert metadata["storyboard_scale"] == 0.9
    assert metadata["storyboard_final_bytes"] <= 800_000
    assert metadata["storyboard_observation_count"] == 4
    assert [item.curated_index for item in observations] == [0, 1, 2, 3]


def test_unfit_storyboard_fails_locally_without_tools_call(tmp_path) -> None:
    observations = _entropy_observations(tmp_path, 1)
    provider, request = _provider(
        tmp_path,
        observations,
        [_ok_result(PAYLOAD)],
        storyboard_target_max_bytes=100,
    )
    with pytest.raises(ProviderError) as error:
        provider.infer_json(request)
    assert error.value.category == STORYBOARD_SIZE_LIMIT
    assert provider.mcp_tools_call_attempted == 0
    assert provider._client.calls == []
    assert provider._last_failure_metadata["storyboard_final_bytes"] > 100
    assert provider._last_failure_metadata["storyboard_scale"] == 0.7


def test_local_fitting_does_not_increment_tools_call_accounting(tmp_path) -> None:
    observations = _entropy_observations(tmp_path, 4)
    provider, request = _provider(
        tmp_path,
        observations,
        [_ok_result(PAYLOAD)],
        storyboard_target_max_bytes=1_000_000,
    )
    response = provider.infer_json(request)
    assert response.provider_metadata["storyboard_adaptive_compression_applied"]
    assert provider.planned_provider_calls == 1
    assert provider.completed_provider_calls == 1
    assert provider.failed_provider_calls == 0
    assert provider.mcp_tools_call_attempted == 1
    assert len(provider._client.calls) == 1


def test_remote_image_too_large_is_specific_after_real_tools_call(tmp_path) -> None:
    observations = _solid_observations(tmp_path, 1)
    provider, request = _provider(
        tmp_path,
        observations,
        [
            {
                "content": [
                    {
                        "type": "text",
                        "text": "Image file too large: 5.15MB. Maximum allowed: 5MB",
                    }
                ],
                "isError": True,
            }
        ],
    )
    with pytest.raises(ProviderError) as error:
        provider.infer_json(request)
    assert error.value.category == MCP_IMAGE_SIZE_REJECTED
    assert "Image file too large" in error.value.provider_error_message
    assert provider.mcp_tools_call_attempted == 1
    assert provider._last_failure_metadata["error_category"] == (
        MCP_IMAGE_SIZE_REJECTED
    )


def test_success_metadata_records_pass_specific_transport_facts(tmp_path) -> None:
    observations = _solid_observations(tmp_path, 2)
    provider, request = _provider(tmp_path, observations, [_ok_result(PAYLOAD)])
    response = provider.infer_json(request)
    metadata = response.provider_metadata
    assert metadata["storyboard_pass"] == "pass_b"
    assert metadata["storyboard_observation_count"] == 2
    assert metadata["storyboard_target_max_bytes"] == TARGET_MCP_IMAGE_BYTES
    assert metadata["storyboard_final_bytes"] <= TARGET_MCP_IMAGE_BYTES
    assert metadata["storyboard_final_width"] == metadata["storyboard_original_width"]
    assert metadata["storyboard_final_height"] == metadata["storyboard_original_height"]
    assert Path(metadata["storyboard_path"]).name.startswith("pass_b_storyboard_")
