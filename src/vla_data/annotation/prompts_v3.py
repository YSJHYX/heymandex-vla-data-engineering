"""D4.3 positive-recall prompts with boundary-local Pass B authority."""

from __future__ import annotations

import json
from typing import Any

from vla_data.annotation.prompts_v2 import (
    EVIDENCE_RULES,
    IDENTITY_PREAMBLE,
    LANGUAGE_RULES,
    MULTI_VIEW,
    OUTPUT_CONTRACT,
    OUTPUT_PURPOSE,
    SEGMENTATION_RULES,
    STATIC_RULES,
    WHAT_NOT_HOW,
)

PASS_A_PROMPT_VERSION = "vla_semantic_coarse_v3"
PASS_B_PROMPT_VERSION = "vla_semantic_boundary_refine_local_v3"

PASS_A_SCHEMA = (
    '{"episode_task":{"instruction":str,"confidence":number,"task_type":str|null,'
    '"objects":[str],"paraphrases":[str]},'
    '"semantic_segments":[{"segment_id":str,"start_curated_index":int,'
    '"end_curated_index":int,"instruction":str,"confidence":number,'
    '"task_type":str|null,"objects":[str],"paraphrases":[str]}],'
    '"non_training_intervals":[{"start_curated_index":int,'
    '"end_curated_index":int,"reason":str}]}'
)

LOCAL_RECALL_RULES = """\
POSITIVE RECALL WITHOUT PRECISION LOSS

- Local uncertainty must remain local.
- Do not expand one uncertain transition into an entire clean-domain ambiguous \
interval when other task-relevant behavior is observable.
- Use the narrowest temporal interval that is actually ambiguous.
- Local ambiguity at one transition must not invalidate an otherwise observable \
task interval.
- If an episode task is identifiable and some phases are observable, preserve \
those supported semantic phases; lower confidence or isolate only the uncertain \
interval instead of erasing all positives.
- These rules do NOT authorize creating a segment from the episode-task text. \
Every positive interval still requires visual/temporal evidence."""

TEMPORAL_ACQUISITION_RULES = """\
BOUNDARY-LOCAL TEMPORAL EVIDENCE

A manipulation transition may be supported by before/after temporal evidence \
even when the exact contact frame is partially occluded. For acquisition, use \
the ordered evidence: object independent from hand before, partial/occluded \
contact during, then persistent hand-object relation or object moving \
consistently with the hand after.

Finger closing alone never proves grasp. Do not emit grasp without persistent \
hand-object relation, object-following-hand evidence, stable visible enclosure, \
or a sufficiently supported combination of those cues."""

EMBODIMENT_NEUTRAL_RULES = """\
EMBODIMENT-NEUTRAL TRAINING LANGUAGE

Instructions describe manipulation goals, not embodiment movement. Do not use \
"Move the robot arm", "Move the robotic hand", "Rotate the wrist", \
"Move the end effector", or "Position the robot arm" as task instructions. \
Prefer observable goal language such as "Reach toward the connector", \
"Grasp the cable", or "Take the device from the operator".

An operator's hand is valid object/source context. Do not remove operator \
context when it is task-essential, and do not confuse the operator's action \
with the robot task."""


def build_pass_a_prompt_v3(domains: list[dict[int, int]], allowed: list[int]) -> str:
    """Global sparse Pass A; D4.2 sampling remains unchanged."""

    return "\n\n".join(
        [
            IDENTITY_PREAMBLE,
            OUTPUT_PURPOSE,
            WHAT_NOT_HOW,
            MULTI_VIEW,
            LANGUAGE_RULES,
            EVIDENCE_RULES,
            TEMPORAL_ACQUISITION_RULES,
            LOCAL_RECALL_RULES,
            EMBODIMENT_NEUTRAL_RULES,
            STATIC_RULES,
            SEGMENTATION_RULES,
            OUTPUT_CONTRACT,
            f"JSON SHAPE (exact):\n{PASS_A_SCHEMA}",
            (
                "All boundaries are half-open [start,end), must come from "
                "allowed_boundary_indices, must remain within one clean_domain, "
                "and semantic_segments plus non_training_intervals must exactly "
                "classify every clean-domain index with no gap and no overlap. "
                "Do not auto-create semantic segments from episode_task."
            ),
            f"clean_domains={json.dumps(domains, sort_keys=True)}",
            f"allowed_boundary_indices={allowed}",
        ]
    )


def build_pass_b_prompt_v3(transition: dict[str, Any], allowed: list[int]) -> str:
    """One local call that may update only one supplied coarse transition."""

    schema = (
        '{"transition_id":str,"previous_instruction":str,'
        '"next_instruction":str,"boundary_curated_index":int,'
        '"confidence":number,"evidence_status":"SUPPORTED|PARTIALLY_SUPPORTED|'
        'AMBIGUOUS|INSUFFICIENT_VISUAL_EVIDENCE",'
        '"semantic_correction":null|{"previous_instruction":str|null,'
        '"next_instruction":str|null,"reason":str}}'
    )
    return "\n\n".join(
        [
            IDENTITY_PREAMBLE,
            """\
STRICTLY LOCAL BOUNDARY ROLE

This call handles exactly ONE already-defined coarse semantic transition. \
Do not re-analyze the full episode. Do not invent an episode task, add/remove \
unrelated phases, create unrelated objects, or globally reclassify intervals.""",
            MULTI_VIEW,
            TEMPORAL_ACQUISITION_RULES,
            LOCAL_RECALL_RULES,
            EMBODIMENT_NEUTRAL_RULES,
            """\
LOCAL AUTHORITY

- Select exactly one boundary_curated_index from candidate_indices.
- Preserve previous_instruction and next_instruction exactly in their top-level \
fields.
- SUPPORTED/PARTIALLY_SUPPORTED may move the boundary when temporal evidence \
supports that choice.
- AMBIGUOUS/INSUFFICIENT_VISUAL_EVIDENCE must retain the coarse boundary; \
uncertainty stays local and does not erase other phases.
- A semantic rename is forbidden unless semantic_correction is a non-null, \
explicit object with a reason. It may refer only to these two adjacent semantic \
objectives. Never rename a non-training reason.
- Confidence describes this boundary evidence only; it is separate from final \
semantic-segment confidence.""",
            OUTPUT_CONTRACT,
            f"JSON SHAPE (exact):\n{schema}",
            f"transition={json.dumps(transition, sort_keys=True)}",
            f"candidate_indices={allowed}",
        ]
    )
