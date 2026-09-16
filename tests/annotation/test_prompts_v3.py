"""D4.3 prompt-version and positive-recall safety contracts."""

from vla_data.annotation.prompts_v3 import (
    PASS_A_PROMPT_VERSION,
    PASS_B_PROMPT_VERSION,
    build_pass_a_prompt_v3,
    build_pass_b_prompt_v3,
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
        "instruction": "Reach toward the connector",
    },
    "next": {
        "collection": "semantic_segments",
        "index": 1,
        "instruction": "Grasp the connector",
    },
    "candidate_indices": list(range(44, 57)),
}


def test_v3_versions_are_new_and_explicit() -> None:
    assert PASS_A_PROMPT_VERSION == "vla_semantic_coarse_v3"
    assert PASS_B_PROMPT_VERSION == "vla_semantic_boundary_refine_local_v3"


def test_pass_a_remains_global_and_preserves_negative_safety() -> None:
    prompt = build_pass_a_prompt_v3(DOMAINS, [0, 25, 50, 75, 100])
    assert "episode_task" in prompt and "semantic_segments" in prompt
    assert "Do NOT infer contact/grasp" in prompt
    assert "fingers closing" in prompt
    assert "NO_TASK_RELEVANT_ACTIVITY" in prompt
    assert "Do not auto-create semantic segments from episode_task" in prompt
    assert "Local uncertainty must remain local" in prompt


def test_v3_temporal_evidence_and_embodiment_neutral_rules() -> None:
    prompt = build_pass_a_prompt_v3(DOMAINS, [0, 50, 100])
    assert "before/after temporal evidence" in prompt
    assert "exact contact frame is partially occluded" in prompt
    assert "Finger closing alone never proves grasp" in prompt
    assert "Describe WHAT" in prompt
    assert "Move the robot arm" in prompt
    assert "Take the device from the operator" in prompt


def test_pass_b_is_strictly_one_transition_and_boundary_only() -> None:
    prompt = build_pass_b_prompt_v3(TRANSITION, TRANSITION["candidate_indices"])
    assert "exactly ONE" in prompt
    assert "Do not re-analyze the full episode" in prompt
    assert "boundary_curated_index" in prompt
    assert "evidence_status" in prompt
    assert "semantic_correction" in prompt
    assert "Confidence describes this boundary evidence only" in prompt
    assert "AMBIGUOUS/INSUFFICIENT_VISUAL_EVIDENCE must retain" in prompt
