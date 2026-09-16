"""D4.3.1 platform-aware prompts with semantic-preserving local refinement."""

from __future__ import annotations

import json
from typing import Any

from vla_data.annotation.platform_context import PlatformContext
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
from vla_data.annotation.prompts_v3 import (
    EMBODIMENT_NEUTRAL_RULES,
    LOCAL_RECALL_RULES,
    PASS_A_SCHEMA,
    TEMPORAL_ACQUISITION_RULES,
)

PASS_A_PROMPT_VERSION = "vla_semantic_coarse_v3_1"
PASS_B_PROMPT_VERSION = "vla_semantic_boundary_refine_local_v3_1"

COARSE_PRESERVATION_RULES = """\
COARSE SEMANTIC PRESERVATION

- Pass A defines the episode task, coarse semantic phases, and coarse boundaries.
- Boundary uncertainty is not semantic-interval unusability.
- Pass B boundary ambiguity must not delete, replace, or convert an otherwise valid coarse semantic phase into a non-training interval.
- AMBIGUOUS means the exact transition position is uncertain; retain the coarse boundary and both coarse phases.
- INSUFFICIENT_VISUAL_EVIDENCE also retains coarse semantics by default and records review evidence.
- Only an explicit, sufficiently supported semantic_correction may rename one adjacent coarse semantic objective. Silent deletion, replacement, or object renaming is forbidden."""

OBJECT_IDENTITY_RULES = """\
OBJECT IDENTITY CONSISTENCY

- Keep object naming internally consistent across episode_task and coarse phases.
- When the visual category is uncertain, use a stable neutral name such as object, component, or item.
- Do not drift among more specific object names across phases without visual evidence.
- Pass B must preserve supplied object wording unless an explicit high-confidence semantic_correction requests a local rename."""

SG100_EVIDENCE_RULES = """\
SG100 VISUAL EVIDENCE

- SG100 is a multi-finger dexterous robot hand capable of reaching, enclosure, grasping, holding, and releasing.
- Capability is not event evidence: finger closing alone is never proof of acquisition.
- Require persistent hand-object attachment, visible enclosure/contact, subsequent object motion with the SG100, or equivalent before/after temporal evidence."""


def build_pass_a_prompt_v31(
    domains: list[dict[int, int]],
    allowed: list[int],
    *,
    platform_context: PlatformContext,
) -> str:
    """Global Pass A with stable embodiment/camera interpretation context."""

    return "\n\n".join(
        [
            platform_context.render(),
            IDENTITY_PREAMBLE,
            OUTPUT_PURPOSE,
            WHAT_NOT_HOW,
            MULTI_VIEW,
            LANGUAGE_RULES,
            EVIDENCE_RULES,
            TEMPORAL_ACQUISITION_RULES,
            LOCAL_RECALL_RULES,
            EMBODIMENT_NEUTRAL_RULES,
            COARSE_PRESERVATION_RULES,
            OBJECT_IDENTITY_RULES,
            _strict_local_rules(platform_context),
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


GENERIC_STRICT_LOCAL_RULES = """\
END-EFFECTOR STRICT LOCAL EVIDENCE

- The end effector is capable of reaching, enclosure, grasping, holding, and releasing.
- Capability is not event evidence: finger/gripper closing alone is never proof of acquisition.
- For exact local boundary placement require strict local evidence: \
persistent end-effector-object attachment, visible enclosure/contact at the \
sampled boundary neighborhood, subsequent object motion with the end \
effector, or equivalent before/after temporal evidence at this transition."""


def _strict_local_rules(platform_context: PlatformContext) -> str:
    if "SG100" in platform_context.end_effector:
        return SG100_EVIDENCE_RULES
    return GENERIC_STRICT_LOCAL_RULES


def build_pass_b_prompt_v31(
    transition: dict[str, Any],
    allowed: list[int],
    *,
    platform_context: PlatformContext,
) -> str:
    """One boundary-only Pass B call with no silent semantic authority."""

    schema = (
        '{"transition_id":str,"previous_instruction":str,'
        '"next_instruction":str,"boundary_curated_index":int,'
        '"confidence":number,"evidence_status":"SUPPORTED|PARTIALLY_SUPPORTED|'
        'AMBIGUOUS|INSUFFICIENT_VISUAL_EVIDENCE",'
        '"semantic_correction":null|{"previous_instruction":str|null,'
        '"next_instruction":str|null,"reason":str,"confidence":number}}'
    )
    return "\n\n".join(
        [
            platform_context.render(),
            IDENTITY_PREAMBLE,
            """\
STRICTLY LOCAL BOUNDARY ROLE

This call handles exactly ONE already-defined coarse semantic transition. Do not re-analyze the full episode. Do not invent an episode task, add/remove phases, create unrelated objects, or globally reclassify intervals.""",
            MULTI_VIEW,
            TEMPORAL_ACQUISITION_RULES,
            EMBODIMENT_NEUTRAL_RULES,
            COARSE_PRESERVATION_RULES,
            OBJECT_IDENTITY_RULES,
            _strict_local_rules(platform_context),
            """\
LOCAL AUTHORITY

- Select one boundary_curated_index from candidate_indices.
- Preserve previous_instruction and next_instruction exactly in their top-level fields.
- SUPPORTED may use the refined boundary.
- PARTIALLY_SUPPORTED may use a valid refined boundary; otherwise retain the coarse boundary.
- AMBIGUOUS and INSUFFICIENT_VISUAL_EVIDENCE must retain the coarse boundary and coarse phases.
- semantic_correction is optional and local. It is applied only with sufficient evidence and correction confidence >= 0.7; otherwise it is rejected and coarse semantics remain unchanged.
- semantic_correction can rename only an adjacent semantic objective. It cannot delete a phase, replace a phase with non-training, rename a non-training reason, or silently change object identity.
- Confidence describes local evidence; it does not grant global semantic authority.""",
            OUTPUT_CONTRACT,
            f"JSON SHAPE (exact):\n{schema}",
            f"transition={json.dumps(transition, sort_keys=True)}",
            f"candidate_indices={allowed}",
        ]
    )
