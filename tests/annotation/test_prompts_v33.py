"""D4.3.3 prompt contract: canonical compliance without v3.2 regression."""

from __future__ import annotations

from vla_data.annotation.hierarchical import get_prompt_family
from vla_data.annotation.platform_context import DEFAULT_PLATFORM_CONTEXT
from vla_data.annotation.prompts_v33 import (
    PASS_A_PROMPT_VERSION,
    PASS_B_PROMPT_VERSION,
)
from vla_data.benchmark.libero import LIBERO_PLATFORM_CONTEXT

DOMAINS = [
    {
        "d2_segment_id": 0,
        "clean_run_id": 0,
        "start_curated_index": 0,
        "end_curated_index": 20,
    }
]
ALLOWED = [0, 4, 8, 12, 16, 20]


def _prompt(context=DEFAULT_PLATFORM_CONTEXT) -> str:
    return get_prompt_family("v3.3", context).build_pass_a(DOMAINS, ALLOWED)


def test_v33_versions_and_boundary_local_registration() -> None:
    family = get_prompt_family("v3.3", DEFAULT_PLATFORM_CONTEXT)
    assert family.pass_a_version == PASS_A_PROMPT_VERSION == "vla_semantic_coarse_v3_3"
    assert (
        family.pass_b_version
        == (PASS_B_PROMPT_VERSION)
        == "vla_semantic_boundary_refine_local_v3_3"
    )
    assert family.boundary_local is True


def test_pass_a_requires_explicit_object_nouns_and_objective_separation() -> None:
    prompt = _prompt()
    for required in (
        "must explicitly name manipulated objects",
        "unnamed object pronouns",
        "Repeat the explicit object noun",
        "different temporal phases",
        "separate semantic segments",
        "manipulation objective changes",
        "Do not over-segment minor kinematic corrections",
    ):
        assert required in prompt, required
    for pronoun in ("it", "them", "this", "that", "one", "ones"):
        assert pronoun in prompt


def test_pass_a_keeps_v32_recall_and_excludes_pass_b_only_authority() -> None:
    prompt = _prompt()
    assert "SPARSE PASS-A ROLE" in prompt
    assert "PASS-A EVIDENCE CALIBRATION" in prompt
    assert "COARSE RECALL RULES" in prompt
    for forbidden in (
        "semantic_correction",
        "correction confidence",
        ">= 0.7",
        "previous_instruction",
        "next_instruction",
    ):
        assert forbidden not in prompt, forbidden


def test_pass_b_semantics_remain_v32_boundary_local_contract() -> None:
    family = get_prompt_family("v3.3", DEFAULT_PLATFORM_CONTEXT)
    prompt = family.build_pass_b(
        {
            "transition_id": "boundary_000",
            "coarse_boundary_curated_index": 8,
            "previous": {
                "collection": "semantic_segments",
                "index": 0,
                "instruction": "Grasp the component",
            },
            "next": {
                "collection": "semantic_segments",
                "index": 1,
                "instruction": "Lift the component",
            },
            "candidate_indices": [7, 8, 9],
        },
        [7, 8, 9],
    )
    assert "COARSE SEMANTIC PRESERVATION" in prompt
    assert "semantic_correction" in prompt
    assert ">= 0.7" in prompt


def test_libero_context_stays_panda_and_production_context_stays_rm65_sg100() -> None:
    libero = _prompt(LIBERO_PLATFORM_CONTEXT)
    assert "Panda" in libero and "parallel gripper" in libero
    for forbidden in ("RM65B", "SG100", "D455", "D405"):
        assert forbidden not in libero

    production = _prompt(DEFAULT_PLATFORM_CONTEXT)
    assert "RM65B" in production and "SG100" in production
    assert "SG100 COARSE EVIDENCE" in production


def test_examples_are_generic_not_libero_task_specific() -> None:
    prompt = _prompt(LIBERO_PLATFORM_CONTEXT).lower()
    for forbidden in ("butter cookies", "bowl", "basket", "plate"):
        assert forbidden not in prompt
