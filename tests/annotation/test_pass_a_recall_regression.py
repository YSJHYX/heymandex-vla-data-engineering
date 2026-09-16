"""D4.3.2 Pass-A recall behavior regressions (003/005/002-like, merge)."""

from __future__ import annotations

# Reuse the single-domain loader pattern (episode → one [0,N) clean domain).
from tests.annotation.test_boundary_local import (
    _domain,
    _load_single_domain_episode,
    _local,
    _write_eligible_quality,
)
from vla_data.annotation.glm_provider import DeterministicStructuredMockProvider
from vla_data.annotation.hierarchical import annotate_hierarchical_episode
from vla_data.annotation.schema import load_annotation


def _run(curated_v1_episode, tmp_path, coarse, *, local_results=None, name):
    episode = _load_single_domain_episode(curated_v1_episode)
    quality_root = _write_eligible_quality(tmp_path, episode)
    end = episode.transition_count
    from vla_data.annotation.boundary_local import build_boundary_transitions

    transitions = (
        build_boundary_transitions(coarse, _domain(end), radius=1)
        if local_results is None
        else []
    )
    provider = DeterministicStructuredMockProvider(
        coarse,
        *(local_results or [_local(transition) for transition in transitions]),
    )
    output = tmp_path / name
    result = annotate_hierarchical_episode(
        episode.episode_dir,
        quality_root=quality_root,
        output_root=output,
        provider=provider,
        boundary_radius=1,
        prompt_family="v3.2",
    )
    annotation = (
        load_annotation(output / episode.episode_dir.name / "annotation.json")
        if result.status == "SUCCESS"
        else None
    )
    return result, annotation


def _coarse_003_like(end: int) -> dict:
    # Sparse evidence: idle, approach to operator-held device, then object
    # enclosed/displaced — coarse Reach + acquisition without exact contact.
    return {
        "episode_task": {
            "instruction": "Take the device from the operator",
            "confidence": 0.82,
            "paraphrases": [],
        },
        "semantic_segments": [
            {
                "segment_id": "reach",
                "start_curated_index": 0,
                "end_curated_index": 2,
                "instruction": "Reach toward the device held by the operator",
                "confidence": 0.71,
                "paraphrases": [],
            },
            {
                "segment_id": "acquire",
                "start_curated_index": 2,
                "end_curated_index": end,
                "instruction": "Take the device from the operator",
                "confidence": 0.74,
                "paraphrases": [],
            },
        ],
        "non_training_intervals": [],
    }


def test_003_like_sparse_sequence_keeps_positive_phases(curated_v1_episode, tmp_path):
    """No exact-contact sample must not collapse the episode to insufficient."""

    result, annotation = _run(
        curated_v1_episode, tmp_path, _coarse_003_like(3), name="d432-003"
    )
    assert result.status == "SUCCESS"
    assert result.pass_a_calls == 1
    assert len(annotation["semantic_segments"]) == 2
    codes = annotation["diagnostics"]["diagnostic_codes"]
    assert "PASS_A_INTERNAL_INCONSISTENCY" not in codes
    assert "SEMANTIC_RECALL_REVIEW" not in codes


def test_005_like_partial_positive_with_localized_uncertainty(
    curated_v1_episode,
    tmp_path,
) -> None:
    coarse = {
        "episode_task": {
            "instruction": "Plug the connector into the device",
            "confidence": 0.78,
            "paraphrases": [],
        },
        "semantic_segments": [
            {
                "segment_id": "reach",
                "start_curated_index": 1,
                "end_curated_index": 3,
                "instruction": "Reach toward the connector",
                "confidence": 0.7,
                "paraphrases": [],
            }
        ],
        "non_training_intervals": [
            {
                "start_curated_index": 0,
                "end_curated_index": 1,
                "reason": "PRE_TASK_IDLE",
            }
        ],
    }
    result, annotation = _run(curated_v1_episode, tmp_path, coarse, name="d432-005")
    assert result.status == "SUCCESS"
    assert len(annotation["semantic_segments"]) == 1
    assert annotation["non_training_intervals"][0]["reason"] == "PRE_TASK_IDLE"
    assert (
        "PASS_A_INTERNAL_INCONSISTENCY"
        not in (annotation["diagnostics"]["diagnostic_codes"])
    )


def test_concrete_task_whole_domain_no_task_flags_inconsistency(
    curated_v1_episode,
    tmp_path,
) -> None:
    """005 D4.3.1 failure pattern: must NOT be a silent clean success."""

    end = 3
    coarse = {
        "episode_task": {
            "instruction": "Grasp the cable",
            "confidence": 0.85,
            "paraphrases": [],
        },
        "semantic_segments": [],
        "non_training_intervals": [
            {
                "start_curated_index": 0,
                "end_curated_index": end,
                "reason": "NO_TASK_RELEVANT_ACTIVITY",
            }
        ],
    }
    result, annotation = _run(
        curated_v1_episode, tmp_path, coarse, name="d432-contradiction"
    )
    assert result.status == "SUCCESS"  # evidence-derived, not silently valid
    diagnostics = annotation["diagnostics"]
    assert "PASS_A_INTERNAL_INCONSISTENCY" in diagnostics["diagnostic_codes"]
    assert diagnostics["review_required"] is True


def test_generic_no_task_negative_baseline_stays_legal(
    curated_v1_episode,
    tmp_path,
) -> None:
    """002 negative baseline: no hallucinated phases, no diagnostics."""
    end = 3
    coarse = {
        "episode_task": {
            "instruction": "Manipulate the object",
            "confidence": 0.2,
            "paraphrases": [],
        },
        "semantic_segments": [],
        "non_training_intervals": [
            {
                "start_curated_index": 0,
                "end_curated_index": end,
                "reason": "NO_TASK_RELEVANT_ACTIVITY",
            }
        ],
    }
    result, annotation = _run(curated_v1_episode, tmp_path, coarse, name="d432-002")
    assert result.status == "SUCCESS"
    assert annotation["semantic_segments"] == []
    codes = annotation["diagnostics"]["diagnostic_codes"]
    assert "PASS_A_INTERNAL_INCONSISTENCY" not in codes


def test_generic_task_whole_domain_insufficient_stays_legal(
    curated_v1_episode,
    tmp_path,
) -> None:
    end = 3
    coarse = {
        "episode_task": {
            "instruction": "Manipulate the object",
            "confidence": 0.2,
            "paraphrases": [],
        },
        "semantic_segments": [],
        "non_training_intervals": [
            {
                "start_curated_index": 0,
                "end_curated_index": end,
                "reason": "INSUFFICIENT_VISUAL_EVIDENCE",
            }
        ],
    }
    result, annotation = _run(
        curated_v1_episode, tmp_path, coarse, name="d432-insufficient"
    )
    assert result.status == "SUCCESS"
    assert annotation["semantic_segments"] == []


def test_concrete_task_whole_domain_insufficient_flags_recall_review(
    curated_v1_episode,
    tmp_path,
) -> None:
    end = 3
    coarse = {
        "episode_task": {
            "instruction": "Take the device from the operator",
            "confidence": 0.75,
            "paraphrases": [],
        },
        "semantic_segments": [],
        "non_training_intervals": [
            {
                "start_curated_index": 0,
                "end_curated_index": end,
                "reason": "INSUFFICIENT_VISUAL_EVIDENCE",
            }
        ],
    }
    result, annotation = _run(
        curated_v1_episode, tmp_path, coarse, name="d432-003-review"
    )
    assert result.status == "SUCCESS"
    assert "SEMANTIC_RECALL_REVIEW" in annotation["diagnostics"]["diagnostic_codes"]


def test_d431_merge_regression_ambiguous_boundary_keeps_both_phases(
    curated_v1_episode,
    tmp_path,
) -> None:
    """D4.3.1 fix must hold under v3.2: AMBIGUOUS boundary never deletes
    a coarse positive phase (no return to D4.3's Grasp→non-training)."""

    from vla_data.annotation.boundary_local import build_boundary_transitions

    episode = _load_single_domain_episode(curated_v1_episode)
    quality_root = _write_eligible_quality(tmp_path, episode)
    end = episode.transition_count
    coarse = _coarse_003_like(end)
    transitions = build_boundary_transitions(coarse, _domain(end), radius=1)
    provider = DeterministicStructuredMockProvider(
        coarse,
        *[
            _local(transition, status="AMBIGUOUS", confidence=0.4)
            for transition in transitions
        ],
    )
    output = tmp_path / "d432-merge"
    result = annotate_hierarchical_episode(
        episode.episode_dir,
        quality_root=quality_root,
        output_root=output,
        provider=provider,
        boundary_radius=1,
        prompt_family="v3.2",
    )
    assert result.status == "SUCCESS"
    annotation = load_annotation(output / episode.episode_dir.name / "annotation.json")
    instructions = [s["instruction"] for s in annotation["semantic_segments"]]
    assert any("Reach" in instruction for instruction in instructions)
    assert any("Take the device" in instruction for instruction in instructions)
    assert annotation["non_training_intervals"] == []
