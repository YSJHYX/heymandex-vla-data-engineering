"""Prompt v3.1 platform and semantic-preservation contracts."""

from vla_data.annotation.platform_context import DEFAULT_PLATFORM_CONTEXT
from vla_data.annotation.prompts_v31 import (
    PASS_A_PROMPT_VERSION,
    PASS_B_PROMPT_VERSION,
    build_pass_a_prompt_v31,
    build_pass_b_prompt_v31,
)

DOMAINS = [
    {
        "d2_segment_id": 0,
        "clean_run_id": 0,
        "start_curated_index": 0,
        "end_curated_index": 100,
    }
]
TRANSITION = {
    "transition_id": "boundary_000",
    "coarse_boundary_curated_index": 50,
    "previous": {
        "collection": "semantic_segments",
        "index": 0,
        "instruction": "Reach toward the component",
    },
    "next": {
        "collection": "semantic_segments",
        "index": 1,
        "instruction": "Lift the component",
    },
    "candidate_indices": list(range(44, 57)),
}


def test_v31_versions_are_explicit() -> None:
    assert PASS_A_PROMPT_VERSION == "vla_semantic_coarse_v3_1"
    assert PASS_B_PROMPT_VERSION == "vla_semantic_boundary_refine_local_v3_1"


def test_both_passes_receive_identical_platform_context() -> None:
    rendered = DEFAULT_PLATFORM_CONTEXT.render()
    pass_a = build_pass_a_prompt_v31(
        DOMAINS, [0, 50, 100], platform_context=DEFAULT_PLATFORM_CONTEXT
    )
    pass_b = build_pass_b_prompt_v31(
        TRANSITION,
        TRANSITION["candidate_indices"],
        platform_context=DEFAULT_PLATFORM_CONTEXT,
    )
    assert pass_a.startswith(rendered)
    assert pass_b.startswith(rendered)
    for prompt in (pass_a, pass_b):
        assert "finger" in prompt.lower() and "alone" in prompt.lower()
        assert "camera ego-motion" in prompt
        assert "WHAT" in prompt


def test_pass_b_separates_boundary_uncertainty_from_semantic_validity() -> None:
    prompt = build_pass_b_prompt_v31(
        TRANSITION,
        TRANSITION["candidate_indices"],
        platform_context=DEFAULT_PLATFORM_CONTEXT,
    )
    assert "Boundary uncertainty is not semantic-interval unusability" in prompt
    assert "must retain the coarse boundary and coarse phases" in prompt
    assert "correction confidence >= 0.7" in prompt
    assert "cannot delete a phase" in prompt
