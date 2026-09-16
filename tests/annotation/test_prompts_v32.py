"""D4.3.2 stage-specific prompts, recall calibration, consistency diagnostics."""

from __future__ import annotations

import json

from vla_data.annotation.hierarchical import get_prompt_family
from vla_data.annotation.platform_context import DEFAULT_PLATFORM_CONTEXT
from vla_data.annotation.prompts_v32 import (
    PASS_A_PROMPT_VERSION,
    PASS_B_PROMPT_VERSION,
    build_pass_a_prompt_v32,
)
from vla_data.annotation.semantic_diagnostics import (
    concrete_task_with_insufficient_domain,
    pass_a_internal_inconsistency,
)

DOMAINS = [
    {
        "d2_segment_id": 0,
        "clean_run_id": 0,
        "start_curated_index": 0,
        "end_curated_index": 343,
    }
]
ALLOWED = [0, 68, 136, 205, 273, 342]


def pass_a() -> str:
    return build_pass_a_prompt_v32(
        DOMAINS, ALLOWED, platform_context=DEFAULT_PLATFORM_CONTEXT
    )


def pass_b() -> str:
    family = get_prompt_family("v3.2", DEFAULT_PLATFORM_CONTEXT)
    return family.build_pass_b(
        {
            "transition_id": "t0",
            "previous_instruction": "Reach toward the cable",
            "next_instruction": "Grasp the cable",
        },
        ALLOWED,
    )


# ------------------------------------------------------- family registration


def test_v32_family_registered_with_boundary_local() -> None:
    family = get_prompt_family("v3.2", DEFAULT_PLATFORM_CONTEXT)
    assert family.pass_a_version == PASS_A_PROMPT_VERSION == "vla_semantic_coarse_v3_2"
    assert (
        family.pass_b_version
        == PASS_B_PROMPT_VERSION
        == ("vla_semantic_boundary_refine_local_v3_2")
    )
    assert family.boundary_local is True
    assert family.platform_context is DEFAULT_PLATFORM_CONTEXT


def test_sampling_and_visual_representation_unchanged() -> None:
    # D4.3.2 must not change sampling: the prompt builder only receives the
    # same domain/allowed inputs as prior families.
    prompt = pass_a()
    assert json.dumps(DOMAINS, sort_keys=True) in prompt
    assert f"allowed_boundary_indices={ALLOWED}" in prompt


# --------------------------------------------------- Pass-A purity (spec 37)


def test_pass_a_contains_platform_and_sparse_calibration() -> None:
    prompt = pass_a()
    for required in (
        "RM65B",
        "SG100",
        "external/global",
        "wrist-mounted",
        "sparse observations",
        "coarse semantic phases",
        "does not need to appear in the sparse Pass-A samples",
    ):
        assert required in prompt, required


def test_pass_a_excludes_pass_b_merge_vocabulary() -> None:
    prompt = pass_a()
    for forbidden in (
        "semantic_correction",
        "correction confidence",
        ">= 0.7",
        "previous_instruction",
        "next_instruction",
        "Pass B boundary ambiguity",
        "boundary ambiguity must not",
        "retain the coarse boundary",
        "coarse semantic preservation",
        "Pass B must preserve",
    ):
        assert forbidden not in prompt, forbidden


def test_pass_a_rejects_direct_frame_proof_only_reading() -> None:
    prompt = pass_a()
    assert "Do not require the exact contact or transition frame" in prompt
    assert "does NOT imply absence of a manipulation phase" in prompt
    assert "before/after state changes" in prompt


def test_pass_a_sg100_rule_is_coarse_not_strict() -> None:
    prompt = pass_a()
    assert "SG100 COARSE EVIDENCE" in prompt
    assert "finger motion alone is not evidence of acquisition" in prompt
    # The strict Pass-B wording must NOT gate Pass A.
    assert "Require persistent hand-object attachment" not in prompt


def test_pass_a_keeps_hallucination_safeguards() -> None:
    prompt = pass_a()
    assert "expected workflow, task text, and robot" in prompt
    assert "capability are still not event evidence" in prompt
    assert "Do not auto-create semantic segments from episode_task" in prompt


def test_pass_a_partial_positive_preference_and_no_forced_positive() -> None:
    prompt = pass_a()
    assert "preserve the supported intervals" in prompt
    assert "localize uncertainty to the narrowest justified interval" in prompt
    assert "whole-domain NO_TASK_RELEVANT_ACTIVITY interval remains valid" in prompt


def test_pass_a_object_identity_has_no_pass_b_wording() -> None:
    prompt = pass_a()
    assert "most specific object identity supported by the views" in prompt
    assert "Pass B must preserve supplied object wording" not in prompt


# --------------------------------------------------- Pass-B purity (spec 38)


def test_pass_b_preserves_merge_contract_and_threshold() -> None:
    prompt = pass_b()
    for required in (
        "COARSE SEMANTIC PRESERVATION",
        "AMBIGUOUS means the exact transition position is uncertain",
        "INSUFFICIENT_VISUAL_EVIDENCE also retains coarse semantics",
        "semantic_correction",
        ">= 0.7",
        "previous_instruction",
        "next_instruction",
        "candidate_indices",
    ):
        assert required in prompt, required


def test_pass_b_version_provenance_is_v32_family() -> None:
    # Even though Pass B text is inherited from v3.1, the family pins v3.2
    # provenance so experiments stay reproducible.
    assert PASS_B_PROMPT_VERSION == "vla_semantic_boundary_refine_local_v3_2"


# --------------------------------------------- consistency diagnostics (33)


def _annotation(task: str, segments: list, reasons: list[str]) -> dict:
    return {
        "episode_task": {"instruction": task, "confidence": 0.9},
        "semantic_segments": [
            {
                "instruction": instruction,
                "start_curated_index": 0,
                "end_curated_index": 10,
                "segment_id": "s",
                "confidence": 0.8,
            }
            for instruction in segments
        ],
        "non_training_intervals": [
            {"reason": reason, "start_curated_index": 10, "end_curated_index": 20}
            for reason in reasons
        ],
    }


def test_concrete_task_whole_domain_no_task_is_internal_inconsistency() -> None:
    # 005 pattern: Grasp the cable + whole domain NO_TASK_RELEVANT_ACTIVITY.
    assert pass_a_internal_inconsistency(
        _annotation("Grasp the cable", [], ["NO_TASK_RELEVANT_ACTIVITY"])
    )


def test_generic_task_whole_domain_no_task_stays_legal() -> None:
    # 002 negative baseline: generic/unresolved task, no segments.
    assert not pass_a_internal_inconsistency(
        _annotation("Manipulate the object", [], ["NO_TASK_RELEVANT_ACTIVITY"])
    )
    assert not pass_a_internal_inconsistency(
        _annotation("Interact with the scene", [], ["NO_TASK_RELEVANT_ACTIVITY"])
    )


def test_concrete_task_with_segments_is_consistent() -> None:
    assert not pass_a_internal_inconsistency(
        _annotation("Grasp the cable", ["Reach toward the cable"], ["PRE_TASK_IDLE"])
    )


def test_concrete_task_whole_domain_insufficient_flags_recall_review() -> None:
    # 003 pattern: concrete task + whole-domain INSUFFICIENT_VISUAL_EVIDENCE.
    assert concrete_task_with_insufficient_domain(
        _annotation(
            "Take the device from the operator", [], ["INSUFFICIENT_VISUAL_EVIDENCE"]
        )
    )


def test_generic_low_confidence_insufficient_is_not_forced_positive() -> None:
    assert not concrete_task_with_insufficient_domain(
        _annotation("Manipulate the object", [], ["INSUFFICIENT_VISUAL_EVIDENCE"])
    )


def test_partial_insufficient_with_positive_segments_not_flagged() -> None:
    assert not concrete_task_with_insufficient_domain(
        _annotation(
            "Grasp the cable", ["Reach toward the cable"], ["AMBIGUOUS_VISUAL_EVIDENCE"]
        )
    )
