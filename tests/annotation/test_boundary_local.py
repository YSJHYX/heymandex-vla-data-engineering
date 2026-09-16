"""D4.3 boundary-local planning, merging, calls, and regressions."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from tests.annotation.test_mcp_provider import FakeMCPClient, _ok_result, _tool_schema
from vla_data.annotation.boundary_local import (
    build_boundary_transitions,
    merge_local_boundary_results,
    validate_local_boundary_result,
)
from vla_data.annotation.glm_provider import DeterministicStructuredMockProvider
from vla_data.annotation.hierarchical import (
    annotate_hierarchical_dataset,
    annotate_hierarchical_episode,
)
from vla_data.annotation.mcp_provider import (
    MCP_TIMEOUT,
    CodingPlanVisionMCPProvider,
    MCPProviderConfig,
)
from vla_data.annotation.platform_context import DEFAULT_PLATFORM_CONTEXT
from vla_data.annotation.provider import ImageItem, ProviderError
from vla_data.annotation.schema import AnnotationSchemaError, load_annotation
from vla_data.io.curated_episode import CuratedEpisode


def _coarse(end: int) -> dict:
    return {
        "episode_task": {
            "instruction": "Take the connector from the operator",
            "confidence": 0.8,
            "paraphrases": ["Grasp the connector"],
        },
        "semantic_segments": [
            {
                "segment_id": "reach",
                "start_curated_index": 1,
                "end_curated_index": 2,
                "instruction": "Reach toward the connector",
                "confidence": 0.75,
                "paraphrases": [],
            },
            {
                "segment_id": "grasp",
                "start_curated_index": 2,
                "end_curated_index": end,
                "instruction": "Grasp the connector",
                "confidence": 0.65,
                "paraphrases": [],
            },
        ],
        "non_training_intervals": [
            {
                "start_curated_index": 0,
                "end_curated_index": 1,
                "reason": "PRE_TASK_IDLE",
            }
        ],
    }


def _local(
    transition,
    *,
    status="SUPPORTED",
    boundary=None,
    confidence=0.7,
    semantic_correction=None,
) -> dict:
    return {
        "transition_id": transition["transition_id"],
        "previous_instruction": transition["previous"]["instruction"],
        "next_instruction": transition["next"]["instruction"],
        "boundary_curated_index": (
            transition["coarse_boundary_curated_index"]
            if boundary is None
            else boundary
        ),
        "confidence": confidence,
        "evidence_status": status,
        "semantic_correction": semantic_correction,
    }


def _domain(end: int) -> list[dict]:
    return [
        {
            "d2_segment_id": 0,
            "clean_run_id": 0,
            "start_curated_index": 0,
            "end_curated_index": end,
        }
    ]


def _write_eligible_quality(tmp_path, episode: CuratedEpisode):
    quality_root = tmp_path / "quality"
    episode_id = episode.episode_dir.name
    episode_root = quality_root / episode_id
    episode_root.mkdir(parents=True)
    (episode_root / "quality_report.json").write_text(
        json.dumps(
            {
                "schema_name": "vla_quality_report",
                "schema_version": 1,
                "episode_id": episode_id,
                "status": "ACCEPT",
            }
        )
    )
    np.save(
        episode_root / "quality_mask.npy",
        np.ones(episode.transition_count, dtype=bool),
    )
    return quality_root


def _load_single_domain_episode(episode_dir) -> CuratedEpisode:
    metadata_path = episode_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["segment_offsets"] = [0, 3]
    metadata_path.write_text(json.dumps(metadata))
    return CuratedEpisode.load(episode_dir)


def _load_four_transition_episode(episode_dir) -> CuratedEpisode:
    metadata_path = episode_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["segment_offsets"] = [0, 4]
    metadata_path.write_text(json.dumps(metadata))
    trajectory_path = episode_dir / "trajectory.npz"
    with np.load(trajectory_path, allow_pickle=False) as archive:
        arrays = {}
        for key in archive.files:
            value = np.asarray(archive[key])
            if value.ndim == 2 or key.endswith("rgb_frame_index"):
                extra = value[-1:]
            else:
                step = value[-1] - value[-2]
                extra = np.asarray([value[-1] + step], dtype=value.dtype)
            arrays[key] = np.concatenate((value, extra), axis=0)
    np.savez(trajectory_path, **arrays)
    return CuratedEpisode.load(episode_dir)


def _coarse_four() -> dict:
    payload = _coarse(3)
    payload["semantic_segments"].append(
        {
            "segment_id": "lift",
            "start_curated_index": 3,
            "end_curated_index": 4,
            "instruction": "Lift the connector",
            "confidence": 0.7,
            "paraphrases": [],
        }
    )
    return payload


def test_one_transition_plan_per_boundary_with_exact_dense_evidence() -> None:
    coarse = {
        "semantic_segments": [
            {
                "segment_id": "a",
                "start_curated_index": 0,
                "end_curated_index": 20,
                "instruction": "Reach toward the object",
            },
            {
                "segment_id": "b",
                "start_curated_index": 20,
                "end_curated_index": 50,
                "instruction": "Grasp the object",
            },
            {
                "segment_id": "c",
                "start_curated_index": 50,
                "end_curated_index": 80,
                "instruction": "Lift the object",
            },
            {
                "segment_id": "d",
                "start_curated_index": 80,
                "end_curated_index": 100,
                "instruction": "Place the object",
            },
        ],
        "non_training_intervals": [],
    }
    transitions = build_boundary_transitions(coarse, _domain(100), radius=6)
    assert len(transitions) == 3
    assert [item["candidate_indices"] for item in transitions] == [
        list(range(14, 27)),
        list(range(44, 57)),
        list(range(74, 87)),
    ]
    assert not (
        set(transitions[0]["candidate_indices"])
        & set(transitions[1]["candidate_indices"])
    )


def test_ambiguous_boundary_stays_local_and_keeps_positive_segments() -> None:
    coarse = _coarse(3)
    transitions = build_boundary_transitions(coarse, _domain(3), radius=1)
    results = [
        validate_local_boundary_result(_local(transition), transition)
        for transition in transitions
    ]
    results[1] = validate_local_boundary_result(
        _local(transitions[1], status="AMBIGUOUS"), transitions[1]
    )
    refined = merge_local_boundary_results(coarse, transitions, results)
    assert [item["instruction"] for item in refined["semantic_segments"]] == [
        "Reach toward the connector",
        "Grasp the connector",
    ]
    assert refined["non_training_intervals"] == coarse["non_training_intervals"]


def test_middle_ambiguous_boundary_cannot_erase_other_positive_phases() -> None:
    coarse = _coarse_four()
    transitions = build_boundary_transitions(coarse, _domain(4), radius=1)
    statuses = ("SUPPORTED", "AMBIGUOUS", "SUPPORTED")
    results = [
        validate_local_boundary_result(_local(transition, status=status), transition)
        for transition, status in zip(transitions, statuses, strict=True)
    ]
    refined = merge_local_boundary_results(coarse, transitions, results)
    assert [item["instruction"] for item in refined["semantic_segments"]] == [
        "Reach toward the connector",
        "Grasp the connector",
        "Lift the connector",
    ]
    assert refined["non_training_intervals"] == coarse["non_training_intervals"]


def test_ambiguous_result_cannot_move_boundary() -> None:
    transition = build_boundary_transitions(_coarse(3), _domain(3), radius=1)[0]
    with pytest.raises(AnnotationSchemaError, match="retain the coarse boundary"):
        validate_local_boundary_result(
            _local(transition, status="AMBIGUOUS", boundary=0), transition
        )


def test_supported_boundary_can_move_but_ambiguous_and_insufficient_cannot() -> None:
    coarse = {
        "semantic_segments": [
            {
                "segment_id": "reach",
                "start_curated_index": 0,
                "end_curated_index": 5,
                "instruction": "Reach toward the component",
            },
            {
                "segment_id": "grasp",
                "start_curated_index": 5,
                "end_curated_index": 10,
                "instruction": "Grasp the component",
            },
        ],
        "non_training_intervals": [],
    }
    transition = build_boundary_transitions(coarse, _domain(10), radius=2)[0]
    supported = validate_local_boundary_result(
        _local(transition, status="SUPPORTED", boundary=4), transition
    )
    refined = merge_local_boundary_results(coarse, [transition], [supported])
    assert refined["semantic_segments"][0]["end_curated_index"] == 4
    assert refined["semantic_segments"][1]["start_curated_index"] == 4
    partial = validate_local_boundary_result(
        _local(transition, status="PARTIALLY_SUPPORTED", boundary=4), transition
    )
    assert partial["boundary_curated_index"] == 4
    for status in ("AMBIGUOUS", "INSUFFICIENT_VISUAL_EVIDENCE"):
        with pytest.raises(AnnotationSchemaError, match="retain the coarse boundary"):
            validate_local_boundary_result(
                _local(transition, status=status, boundary=4), transition
            )


def test_semantic_correction_is_thresholded_and_never_silent() -> None:
    coarse = {
        "semantic_segments": [
            {
                "segment_id": "reach",
                "start_curated_index": 0,
                "end_curated_index": 5,
                "instruction": "Reach toward the connector",
            },
            {
                "segment_id": "grasp",
                "start_curated_index": 5,
                "end_curated_index": 10,
                "instruction": "Grasp the connector",
            },
        ],
        "non_training_intervals": [],
    }
    transition = build_boundary_transitions(coarse, _domain(10), radius=1)[0]
    silent_drift = _local(transition)
    silent_drift["next_instruction"] = "Grasp the device"
    with pytest.raises(AnnotationSchemaError, match="preserve the supplied coarse"):
        validate_local_boundary_result(silent_drift, transition)
    correction = {
        "previous_instruction": None,
        "next_instruction": "Grasp the device",
        "reason": "Local evidence supports a different neutral object category",
        "confidence": 0.6,
    }
    low = validate_local_boundary_result(
        _local(transition, semantic_correction=correction), transition
    )
    assert low["semantic_correction"] is None
    assert low["rejected_semantic_correction"]["minimum_confidence"] == 0.7
    unchanged = merge_local_boundary_results(coarse, [transition], [low])
    assert unchanged["semantic_segments"][1]["instruction"] == "Grasp the connector"

    correction["confidence"] = 0.8
    high = validate_local_boundary_result(
        _local(transition, confidence=0.8, semantic_correction=correction), transition
    )
    changed = merge_local_boundary_results(coarse, [transition], [high])
    assert changed["semantic_segments"][1]["instruction"] == "Grasp the device"


def test_003_like_positive_uses_one_call_per_local_transition(
    curated_v1_episode, tmp_path
) -> None:
    episode = _load_single_domain_episode(curated_v1_episode)
    quality_root = _write_eligible_quality(tmp_path, episode)
    end = episode.transition_count
    assert end >= 3
    coarse = _coarse(end)
    transitions = build_boundary_transitions(coarse, _domain(end), radius=1)
    provider = DeterministicStructuredMockProvider(
        coarse, *[_local(transition) for transition in transitions]
    )
    output = tmp_path / "d43"
    result = annotate_hierarchical_episode(
        episode.episode_dir,
        quality_root=quality_root,
        output_root=output,
        provider=provider,
        boundary_radius=1,
        prompt_family="v3",
    )
    assert result.status == "SUCCESS"
    assert result.pass_b_mode == "boundary_local"
    assert result.pass_a_calls == 1 and result.pass_b_calls == 2
    assert result.actual_planned_provider_calls == 3
    assert result.completed_provider_calls == 3
    annotation = load_annotation(output / episode.episode_dir.name / "annotation.json")
    assert len(annotation["semantic_segments"]) == 2
    assert annotation["diagnostics"]["operator_context_present"] is True
    assert annotation["diagnostics"]["semantic_recall_review"] is False
    assert annotation["model_provenance"]["pass_a"]["coarse_output"] == coarse
    keyframes = json.loads(
        (output / episode.episode_dir.name / "keyframes.json").read_text()
    )
    assert len(keyframes["pass_b"]["boundaries"]) == 2
    for request, transition in zip(provider.calls[1:], transitions, strict=True):
        indices = sorted(
            {
                item.temporal_point
                for item in request.items
                if isinstance(item, ImageItem)
            }
        )
        assert indices == transition["candidate_indices"]


@pytest.mark.parametrize(
    ("reason", "confidence", "review"),
    [
        ("NO_TASK_RELEVANT_ACTIVITY", 0.9, False),
        ("AMBIGUOUS_VISUAL_EVIDENCE", 0.2, True),
    ],
)
def test_zero_segment_negative_and_recall_review_are_both_legal(
    annotated_tree, tmp_path, reason, confidence, review
) -> None:
    episode = CuratedEpisode.load(annotated_tree.curated / "episode_000000")
    end = episode.transition_count
    coarse = {
        "episode_task": {
            "instruction": "Observe the object interaction",
            "confidence": confidence,
            "paraphrases": [],
        },
        "semantic_segments": [],
        "non_training_intervals": [
            {"start_curated_index": 0, "end_curated_index": end, "reason": reason}
        ],
    }
    provider = DeterministicStructuredMockProvider(coarse)
    output = tmp_path / reason
    result = annotate_hierarchical_episode(
        episode.episode_dir,
        quality_root=annotated_tree.quality,
        output_root=output,
        provider=provider,
        prompt_family="v3",
    )
    assert result.status == "SUCCESS"
    assert len(provider.calls) == 1
    assert result.pass_b_calls == 0
    annotation = load_annotation(output / episode.episode_dir.name / "annotation.json")
    assert annotation["semantic_segments"] == []
    assert annotation["diagnostics"]["semantic_recall_review"] is review


def test_fake_mcp_creates_one_distinct_storyboard_per_boundary(
    curated_v1_episode, tmp_path
) -> None:
    episode = _load_four_transition_episode(curated_v1_episode)
    quality_root = _write_eligible_quality(tmp_path, episode)
    coarse = _coarse_four()
    transitions = build_boundary_transitions(
        coarse, _domain(episode.transition_count), radius=1
    )
    client = FakeMCPClient(
        [
            _tool_schema(
                {"image_source": {"type": "string"}, "prompt": {"type": "string"}},
                name="analyze_image",
            )
        ],
        [
            _ok_result(json.dumps(coarse)),
            *[_ok_result(json.dumps(_local(item))) for item in transitions],
        ],
    )
    provider = CodingPlanVisionMCPProvider(
        MCPProviderConfig(storyboard_root=tmp_path / "boards"),
        client=client,
        api_key="team-key",
    )
    result = annotate_hierarchical_episode(
        episode.episode_dir,
        quality_root=quality_root,
        output_root=tmp_path / "annotations",
        provider=provider,
        boundary_radius=1,
        prompt_family="v3",
    )
    assert result.status == "SUCCESS"
    assert result.actual_planned_provider_calls == 4
    assert result.pass_a_calls == 1 and result.pass_b_calls == 3
    assert result.completed_provider_calls == 4
    assert len(client.calls) == 4
    paths = [call[1]["image_source"] for call in client.calls]
    assert "pass_a_storyboard" in paths[0]
    assert "pass_b_boundary_000_storyboard" in paths[1]
    assert "pass_b_boundary_001_storyboard" in paths[2]
    assert "pass_b_boundary_002_storyboard" in paths[3]
    assert len(set(paths)) == 4


def test_v3_dry_run_reports_dynamic_pass_b_without_provider_calls(
    annotated_tree, tmp_path
) -> None:
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    result = annotate_hierarchical_dataset(
        annotated_tree.curated,
        annotated_tree.quality,
        tmp_path / "d43-dry",
        None,
        dry_run=True,
        prompt_family="v3",
        baseline_annotation_root=baseline,
    )
    assert result.summary["episodes_discovered"] == 3
    assert result.summary["prompt_versions"] == [
        "vla_semantic_coarse_v3",
        "vla_semantic_boundary_refine_local_v3",
    ]
    assert result.summary["pass_b_mode"] == "boundary_local"
    assert result.summary["pass_a_calls"] == 2
    assert result.summary["pass_b_calls"] == "dynamic_after_pass_a"
    assert result.summary["minimum_planned_provider_calls"] == 2
    assert result.summary["actual_planned_provider_calls"] is None
    assert not (tmp_path / "d43-dry").exists()


def test_v3_dataset_requires_separate_existing_baseline(annotated_tree, tmp_path):
    with pytest.raises(ValueError, match="requires --baseline-annotation-root"):
        annotate_hierarchical_dataset(
            annotated_tree.curated,
            annotated_tree.quality,
            tmp_path / "d43",
            None,
            dry_run=True,
            prompt_family="v3",
        )
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    with pytest.raises(ValueError, match="non-overlapping"):
        annotate_hierarchical_dataset(
            annotated_tree.curated,
            annotated_tree.quality,
            baseline / "nested",
            None,
            dry_run=True,
            prompt_family="v3",
            baseline_annotation_root=baseline,
        )


def test_v31_optional_paraphrase_is_quarantined_not_episode_fatal(
    annotated_tree, tmp_path
) -> None:
    episode = CuratedEpisode.load(annotated_tree.curated / "episode_000000")
    end = episode.transition_count
    coarse = {
        "episode_task": {
            "instruction": "Reach toward the object",
            "confidence": 0.6,
            "paraphrases": ["Move the robot arm toward the object"],
        },
        "semantic_segments": [
            {
                "segment_id": "reach",
                "start_curated_index": 0,
                "end_curated_index": end,
                "instruction": "Reach toward the object",
                "confidence": 0.6,
                "paraphrases": ["Move the robotic hand closer"],
            }
        ],
        "non_training_intervals": [],
    }
    output = tmp_path / "soft-paraphrase"
    result = annotate_hierarchical_episode(
        episode.episode_dir,
        quality_root=annotated_tree.quality,
        output_root=output,
        provider=DeterministicStructuredMockProvider(coarse),
        prompt_family="v3.1",
    )
    assert result.status == "SUCCESS"
    annotation = load_annotation(output / episode.episode_dir.name / "annotation.json")
    assert annotation["episode_task"]["instruction"] == "Reach toward the object"
    assert annotation["episode_task"]["paraphrases"] == []
    assert annotation["semantic_segments"][0]["paraphrases"] == []
    rejected = annotation["diagnostics"]["rejected_paraphrases"]
    assert len(rejected) == 2
    assert {item["reason"] for item in rejected} == {"EMBODIMENT_CENTRIC_LANGUAGE"}
    assert annotation["model_provenance"]["platform_context"] == (
        DEFAULT_PLATFORM_CONTEXT.as_dict()
    )
    assert annotation["model_provenance"]["platform_context_fingerprint"] == (
        DEFAULT_PLATFORM_CONTEXT.fingerprint
    )


@pytest.mark.parametrize(
    ("evidence_status", "diagnostic_code"),
    [
        ("AMBIGUOUS", "BOUNDARY_EVIDENCE_AMBIGUOUS"),
        (
            "INSUFFICIENT_VISUAL_EVIDENCE",
            "BOUNDARY_EVIDENCE_INSUFFICIENT",
        ),
    ],
)
def test_v31_003_like_uncertain_boundary_preserves_grasp_phase(
    curated_v1_episode, tmp_path, evidence_status, diagnostic_code
) -> None:
    episode = _load_single_domain_episode(curated_v1_episode)
    quality_root = _write_eligible_quality(tmp_path, episode)
    coarse = _coarse(episode.transition_count)
    transitions = build_boundary_transitions(
        coarse, _domain(episode.transition_count), radius=1
    )
    provider = DeterministicStructuredMockProvider(
        coarse,
        _local(transitions[0], status="SUPPORTED"),
        _local(transitions[1], status=evidence_status),
    )
    output = tmp_path / "semantic-preservation"
    result = annotate_hierarchical_episode(
        episode.episode_dir,
        quality_root=quality_root,
        output_root=output,
        provider=provider,
        boundary_radius=1,
        prompt_family="v3.1",
    )
    assert result.status == "SUCCESS"
    annotation = load_annotation(output / episode.episode_dir.name / "annotation.json")
    assert [segment["instruction"] for segment in annotation["semantic_segments"]] == [
        "Reach toward the connector",
        "Grasp the connector",
    ]
    assert annotation["semantic_segments"][1]["start_curated_index"] == 2
    assert diagnostic_code in annotation["diagnostics"]["diagnostic_codes"]
    assert (
        "VERY_SHORT_NONTRAINING_INTERVAL"
        in annotation["diagnostics"]["diagnostic_codes"]
    )
    assert annotation["diagnostics"]["boundary_uncertainty"] == [
        {
            "transition_id": "boundary_001",
            "evidence_status": evidence_status,
            "confidence": 0.7,
        }
    ]


def test_v31_canonical_embodiment_language_still_fails_closed(
    annotated_tree, tmp_path
) -> None:
    episode = CuratedEpisode.load(annotated_tree.curated / "episode_000000")
    end = episode.transition_count
    coarse = {
        "episode_task": {
            "instruction": "Move the robot arm toward the object",
            "confidence": 0.6,
            "paraphrases": [],
        },
        "semantic_segments": [],
        "non_training_intervals": [
            {
                "start_curated_index": 0,
                "end_curated_index": end,
                "reason": "AMBIGUOUS_VISUAL_EVIDENCE",
            }
        ],
    }
    output = tmp_path / "bad-canonical"
    result = annotate_hierarchical_episode(
        episode.episode_dir,
        quality_root=annotated_tree.quality,
        output_root=output,
        provider=DeterministicStructuredMockProvider(coarse),
        prompt_family="v3.1",
    )
    assert result.status == "FAILED"
    assert result.annotation_status == "VALIDATION_FAILED"
    assert "EMBODIMENT_CENTRIC_LANGUAGE" in result.message
    failure = json.loads(
        (output / episode.episode_dir.name / "annotation_result.json").read_text()
    )
    assert failure["semantic_evaluated"] is False


def test_v31_task_sequence_mismatch_is_review_only(annotated_tree, tmp_path) -> None:
    episode = CuratedEpisode.load(annotated_tree.curated / "episode_000000")
    coarse = {
        "episode_task": {
            "instruction": "Plug the connector into the device",
            "confidence": 0.6,
            "paraphrases": [],
        },
        "semantic_segments": [
            {
                "segment_id": "reach_device",
                "start_curated_index": 0,
                "end_curated_index": 1,
                "instruction": "Reach toward the device",
                "confidence": 0.5,
                "paraphrases": [],
            },
            {
                "segment_id": "reach_connector",
                "start_curated_index": 1,
                "end_curated_index": 2,
                "instruction": "Reach toward the connector",
                "confidence": 0.5,
                "paraphrases": [],
            },
        ],
        "non_training_intervals": [],
    }
    transition = build_boundary_transitions(coarse, _domain(2), radius=1)[0]
    output = tmp_path / "mismatch"
    result = annotate_hierarchical_episode(
        episode.episode_dir,
        quality_root=annotated_tree.quality,
        output_root=output,
        provider=DeterministicStructuredMockProvider(coarse, _local(transition)),
        boundary_radius=1,
        prompt_family="v3.1",
    )
    assert result.status == "SUCCESS"
    annotation = load_annotation(output / episode.episode_dir.name / "annotation.json")
    assert (
        "SEMANTIC_TASK_SEQUENCE_MISMATCH"
        in annotation["diagnostics"]["diagnostic_codes"]
    )
    assert annotation["diagnostics"]["review_required"] is True
    assert len(annotation["semantic_segments"]) == 2


def test_v31_platform_fingerprint_controls_resume(annotated_tree, tmp_path) -> None:
    episode = CuratedEpisode.load(annotated_tree.curated / "episode_000000")
    end = episode.transition_count
    coarse = {
        "episode_task": {
            "instruction": "Reach toward the object",
            "confidence": 0.6,
            "paraphrases": [],
        },
        "semantic_segments": [
            {
                "segment_id": "reach",
                "start_curated_index": 0,
                "end_curated_index": end,
                "instruction": "Reach toward the object",
                "confidence": 0.6,
                "paraphrases": [],
            }
        ],
        "non_training_intervals": [],
    }
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    output = tmp_path / "resume"
    first_provider = DeterministicStructuredMockProvider(coarse)
    first = annotate_hierarchical_dataset(
        annotated_tree.curated,
        annotated_tree.quality,
        output,
        first_provider,
        episode="episode_000000",
        prompt_family="v3.1",
        baseline_annotation_root=baseline,
    )
    assert first.results[0].status == "SUCCESS"
    same_provider = DeterministicStructuredMockProvider()
    same = annotate_hierarchical_dataset(
        annotated_tree.curated,
        annotated_tree.quality,
        output,
        same_provider,
        episode="episode_000000",
        prompt_family="v3.1",
        baseline_annotation_root=baseline,
    )
    assert same.results[0].status == "SKIPPED"
    assert same_provider.calls == []

    changed_context = replace(DEFAULT_PLATFORM_CONTEXT, robot_arm="Another robot")
    changed_provider = DeterministicStructuredMockProvider(coarse)
    changed = annotate_hierarchical_dataset(
        annotated_tree.curated,
        annotated_tree.quality,
        output,
        changed_provider,
        episode="episode_000000",
        prompt_family="v3.1",
        baseline_annotation_root=baseline,
        platform_context=changed_context,
    )
    assert changed.results[0].status == "SUCCESS"
    assert len(changed_provider.calls) == 1
    assert (
        changed.results[0].platform_context_fingerprint == changed_context.fingerprint
    )


def test_v31_timeout_is_provider_failure_not_zero_semantics(
    annotated_tree, tmp_path
) -> None:
    episode = CuratedEpisode.load(annotated_tree.curated / "episode_000000")
    client = FakeMCPClient(
        [_tool_schema({"images": {"type": "array", "items": {"type": "string"}}})],
        [
            ProviderError(MCP_TIMEOUT, "tools/call timed out after 300s"),
            ProviderError(MCP_TIMEOUT, "tools/call timed out after 300s"),
        ],
    )
    provider = CodingPlanVisionMCPProvider(client=client, api_key="team-key")
    output = tmp_path / "timeout"
    result = annotate_hierarchical_episode(
        episode.episode_dir,
        quality_root=annotated_tree.quality,
        output_root=output,
        provider=provider,
        prompt_family="v3.1",
    )
    assert result.status == "FAILED"
    assert result.annotation_status == "PROVIDER_FAILED"
    assert result.failure_reason == "MCP_TIMEOUT"
    assert result.semantic_evaluated is False
    assert result.logical_provider_calls == 1
    assert result.actual_tools_call_attempts == 2
    assert result.retry_count == 1
    assert result.semantic_segments is None
    assert not (output / episode.episode_dir.name / "annotation.json").exists()
    failure = json.loads(
        (output / episode.episode_dir.name / "annotation_result.json").read_text()
    )
    assert failure["annotation_status"] == "PROVIDER_FAILED"
    assert failure["semantic_evaluated"] is False
