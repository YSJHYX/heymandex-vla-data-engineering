"""D4.3.3 raw-response persistence, validation ordering, and replay safety."""

from __future__ import annotations

import json
from pathlib import Path

from tests.annotation.test_boundary_local import (
    _domain,
    _load_single_domain_episode,
    _local,
    _write_eligible_quality,
)
from vla_data.annotation.boundary_local import build_boundary_transitions
from vla_data.annotation.glm_provider import DeterministicStructuredMockProvider
from vla_data.annotation.hierarchical import annotate_hierarchical_episode
from vla_data.annotation.platform_context import (
    DEFAULT_PLATFORM_CONTEXT,
    PlatformContext,
)
from vla_data.annotation.provider import (
    AnnotationRequest,
    StructuredModelResponse,
    TextItem,
)
from vla_data.annotation.provider_raw import (
    load_compatible_provider_response,
    persist_provider_response,
)
from vla_data.annotation.schema import load_annotation


def _bad_canonical(end: int) -> dict:
    return {
        "episode_task": {
            "instruction": "Grasp the butter cookies box and lift it",
            "confidence": 0.9,
            "paraphrases": [],
        },
        "semantic_segments": [
            {
                "segment_id": "merged",
                "start_curated_index": 0,
                "end_curated_index": end,
                "instruction": "Grasp the butter cookies box and lift it",
                "confidence": 0.8,
                "paraphrases": [],
            }
        ],
        "non_training_intervals": [],
    }


def _split_canonical(end: int) -> dict:
    return {
        "episode_task": {
            "instruction": "Lift the butter cookies box",
            "confidence": 0.9,
            "paraphrases": [],
        },
        "semantic_segments": [
            {
                "segment_id": "grasp",
                "start_curated_index": 0,
                "end_curated_index": 1,
                "instruction": "Grasp the butter cookies box",
                "confidence": 0.8,
                "paraphrases": [],
            },
            {
                "segment_id": "lift",
                "start_curated_index": 1,
                "end_curated_index": end,
                "instruction": "Lift the butter cookies box",
                "confidence": 0.8,
                "paraphrases": [],
            },
        ],
        "non_training_intervals": [],
    }


def _single_canonical(end: int, *, paraphrases: list[str] | None = None) -> dict:
    return {
        "episode_task": {
            "instruction": "Move the butter cookies box toward the basket",
            "confidence": 0.9,
            "paraphrases": paraphrases or [],
        },
        "semantic_segments": [
            {
                "segment_id": "move",
                "start_curated_index": 0,
                "end_curated_index": end,
                "instruction": "Move the butter cookies box toward the basket",
                "confidence": 0.8,
                "paraphrases": [],
            }
        ],
        "non_training_intervals": [],
    }


def _fixture(curated_v1_episode: Path, tmp_path: Path):
    episode = _load_single_domain_episode(curated_v1_episode)
    quality = _write_eligible_quality(tmp_path, episode)
    return episode, quality, tmp_path / "annotations"


def test_raw_pass_a_is_persisted_before_canonical_hard_failure(
    curated_v1_episode, tmp_path
) -> None:
    episode, quality, output = _fixture(curated_v1_episode, tmp_path)
    provider = DeterministicStructuredMockProvider(
        _bad_canonical(episode.transition_count)
    )
    result = annotate_hierarchical_episode(
        episode.episode_dir,
        quality_root=quality,
        output_root=output,
        provider=provider,
        prompt_family="v3.3",
    )
    assert result.status == "FAILED"
    assert result.annotation_status == "VALIDATION_FAILED"
    assert result.failure_reason == "MODEL_OUTPUT_SCHEMA_ERROR"
    assert result.semantic_evaluated is False
    assert result.logical_provider_calls == 1
    assert result.actual_provider_calls == 1
    assert result.replayed_provider_responses == 0
    raw = output / episode.episode_dir.name / "provider_raw/pass_a_response.json"
    document = json.loads(raw.read_text())
    assert document["raw_structured_payload"] == _bad_canonical(
        episode.transition_count
    )
    assert document["prompt_version"] == "vla_semantic_coarse_v3_3"


def test_same_contract_replays_failed_pass_a_without_provider_call(
    curated_v1_episode, tmp_path
) -> None:
    episode, quality, output = _fixture(curated_v1_episode, tmp_path)
    first = DeterministicStructuredMockProvider(
        _bad_canonical(episode.transition_count)
    )
    assert (
        annotate_hierarchical_episode(
            episode.episode_dir,
            quality_root=quality,
            output_root=output,
            provider=first,
            prompt_family="v3.3",
        ).status
        == "FAILED"
    )
    replay_only = DeterministicStructuredMockProvider()
    result = annotate_hierarchical_episode(
        episode.episode_dir,
        quality_root=quality,
        output_root=output,
        provider=replay_only,
        prompt_family="v3.3",
    )
    assert result.status == "FAILED"
    assert result.failure_reason == "MODEL_OUTPUT_SCHEMA_ERROR"
    assert replay_only.calls == []
    assert result.logical_provider_calls == 1
    assert result.actual_provider_calls == 0
    assert result.replayed_provider_responses == 1


def test_valid_split_output_passes_and_persists_each_pass(
    curated_v1_episode, tmp_path
) -> None:
    episode, quality, output = _fixture(curated_v1_episode, tmp_path)
    coarse = _split_canonical(episode.transition_count)
    transitions = build_boundary_transitions(
        coarse, _domain(episode.transition_count), radius=1
    )
    provider = DeterministicStructuredMockProvider(
        coarse, *[_local(transition) for transition in transitions]
    )
    result = annotate_hierarchical_episode(
        episode.episode_dir,
        quality_root=quality,
        output_root=output,
        provider=provider,
        boundary_radius=1,
        prompt_family="v3.3",
    )
    assert result.status == "SUCCESS", result.message
    assert result.actual_provider_calls == 2
    root = output / episode.episode_dir.name
    annotation = load_annotation(root / "annotation.json")
    assert [item["instruction"] for item in annotation["semantic_segments"]] == [
        "Grasp the butter cookies box",
        "Lift the butter cookies box",
    ]
    assert (root / "provider_raw/pass_a_response.json").is_file()
    assert (root / "provider_raw/pass_b_transition_boundary_000.json").is_file()


def test_valid_explicit_single_instruction_and_bad_optional_paraphrase(
    curated_v1_episode, tmp_path
) -> None:
    episode, quality, output = _fixture(curated_v1_episode, tmp_path)
    provider = DeterministicStructuredMockProvider(
        _single_canonical(episode.transition_count, paraphrases=["Pick it up"])
    )
    result = annotate_hierarchical_episode(
        episode.episode_dir,
        quality_root=quality,
        output_root=output,
        provider=provider,
        prompt_family="v3.3",
    )
    assert result.status == "SUCCESS", result.message
    annotation = load_annotation(output / episode.episode_dir.name / "annotation.json")
    assert annotation["episode_task"]["instruction"] == (
        "Move the butter cookies box toward the basket"
    )
    assert annotation["episode_task"]["paraphrases"] == []
    assert (
        "REJECTED_OPTIONAL_PARAPHRASE" in annotation["diagnostics"]["diagnostic_codes"]
    )


def test_raw_replay_compatibility_invalidates_every_required_key(tmp_path) -> None:
    request = AnnotationRequest("episode_000823", "prompt-a", (TextItem("hello"),))
    response = StructuredModelResponse(
        provider="provider-a",
        model="model-a",
        prompt_version="prompt-a",
        payload={"ok": True},
        request_id="request-a",
        attempt_count=1,
        finish_reason="stop",
        usage=None,
    )
    artifact = tmp_path / "pass_a_response.json"
    base = {
        "pass_name": "A",
        "transition_id": None,
        "input_fingerprint": "input-a",
        "prompt_family": "v3.3",
        "provider_identity": "provider-a",
        "platform_context_fingerprint": "platform-a",
        "pass_b_mode": "boundary_local",
    }
    persist_provider_response(
        artifact,
        response,
        request,
        logical_provider_call_index=1,
        **{key: value for key, value in base.items() if key != "provider_identity"},
    )
    assert load_compatible_provider_response(artifact, request, **base) is not None

    changed_request = AnnotationRequest(
        "episode_000823", "prompt-b", (TextItem("hello"),)
    )
    assert load_compatible_provider_response(artifact, changed_request, **base) is None
    for key, value in (
        ("input_fingerprint", "input-b"),
        ("prompt_family", "v3.2"),
        ("provider_identity", "provider-b"),
        ("platform_context_fingerprint", "platform-b"),
        ("pass_b_mode", "consolidated"),
    ):
        changed = {**base, key: value}
        assert load_compatible_provider_response(artifact, request, **changed) is None


def test_raw_artifact_uses_metadata_allowlist_and_contains_no_secrets(tmp_path) -> None:
    request = AnnotationRequest("episode_000823", "prompt-a", (TextItem("hello"),))
    response = StructuredModelResponse(
        provider="provider-a",
        model="model-a",
        prompt_version="prompt-a",
        payload={"episode_task": {"instruction": "Grasp the component"}},
        request_id="request-a",
        attempt_count=1,
        finish_reason="stop",
        usage=None,
        provider_metadata={
            "mcp_server_name": "safe-server",
            "Z_AI_API_KEY": "TEAM KEY",
            "authorization": "Bearer token",
            "secret": "hidden",
        },
    )
    artifact = tmp_path / "pass_a_response.json"
    persist_provider_response(
        artifact,
        response,
        request,
        pass_name="A",
        transition_id=None,
        input_fingerprint="input-a",
        prompt_family="v3.3",
        platform_context_fingerprint=DEFAULT_PLATFORM_CONTEXT.fingerprint,
        pass_b_mode="boundary_local",
        logical_provider_call_index=1,
    )
    serialized = artifact.read_text()
    assert "safe-server" in serialized
    for secret in ("Z_AI_API_KEY", "TEAM KEY", "authorization", "Bearer token"):
        assert secret not in serialized


def test_platform_fingerprint_change_prevents_replay(tmp_path) -> None:
    altered = PlatformContext(
        **{
            **DEFAULT_PLATFORM_CONTEXT.as_dict(),
            "collection_mode": "different_mode",
        }
    )
    assert altered.fingerprint != DEFAULT_PLATFORM_CONTEXT.fingerprint
