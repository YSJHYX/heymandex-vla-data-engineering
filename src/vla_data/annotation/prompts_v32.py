"""D4.3.2 stage-specific prompts: Pass-A coarse recall, Pass-B local precision.

Root cause of D4.3.1 over-conservatism: COARSE_SEMANTIC_PRESERVATION (a
Pass-B merge/refinement contract) was injected into Pass A, so Pass A
reasoned about how Pass B would preserve its phases instead of proposing
coarse phases from sparse temporal evidence. This family splits the rule
sets per stage and calibrates Pass-A evidence for SPARSE observations.

Stage purity is asserted by tests: Pass A must never see Pass-B merge
vocabulary (semantic_correction, correction confidence, previous/next
instruction, boundary ambiguity retention); Pass B keeps the full D4.3.1
merge contract unchanged.
"""

from __future__ import annotations

import json
from typing import Any

from vla_data.annotation.platform_context import PlatformContext
from vla_data.annotation.prompts_v2 import (
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
    PASS_A_SCHEMA,
)

PASS_A_PROMPT_VERSION = "vla_semantic_coarse_v3_2"
PASS_B_PROMPT_VERSION = "vla_semantic_boundary_refine_local_v3_2"

# ---------------------------------------------------------------- Pass A --

SPARSE_PASS_A_ROLE = """\
SPARSE PASS-A ROLE

Pass A receives sparse observations sampled from the full trajectory, \
not every frame. Its job is global sparse semantic hypothesis generation:

- one episode-level task hypothesis,
- coarse semantic phases (observable over the ordered samples),
- coarse temporal boundaries between those phases,
- pre-task / post-task idle,
- obviously unusable intervals.

Pass A is NOT a contact verifier, boundary verifier, fine-grained grasp \
detector, or the final semantic authority. Exact contact timing and local \
transition confidence are refined later by boundary-local Pass B."""

PASS_A_EVIDENCE_CALIBRATION = """\
PASS-A EVIDENCE CALIBRATION

Pass A operates on sparse global observations. Infer coarse task phases \
from the ordered temporal sequence, multi-view spatial relationships, \
before/after state changes, and persistent object/hand relationships.

Do not require the exact contact or transition frame to be visible in one \
sampled observation before proposing a coarse semantic phase. Absence of an \
exact contact frame in the sparse samples does NOT imply absence of a \
manipulation phase.

You may propose a visually supported coarse semantic hypothesis from \
combined temporal evidence, but you may not invent an unsupported phase \
from the expected task procedure: expected workflow, task text, and robot \
capability are still not event evidence."""

PASS_A_SG100_RULES = """\
SG100 COARSE EVIDENCE

SG100 finger motion alone is not evidence of acquisition.

For coarse phase inference, use the temporal combination of:
- approach,
- changing hand-object spatial relationship,
- visible enclosure/contact when available,
- persistent proximity/attachment,
- later object displacement with the SG100.

The exact acquisition frame does not need to appear in the sparse Pass-A \
samples. For example: an early sample showing the SG100 approaching an \
operator-held object, followed by a later sample where the object appears \
enclosed, attached, or displaced with the SG100, supports a coarse grasp \
hypothesis even if no sample shows the exact first-contact moment."""

PASS_A_RECALL_RULES = """\
COARSE RECALL RULES

- Pass A optimizes coarse semantic recall under evidence constraints; \
boundary precision belongs to Pass B.
- If some task-relevant intervals are visually supported but another \
interval is uncertain, preserve the supported intervals and localize \
uncertainty to the narrowest justified interval.
- Do not expand one uncertain transition into a whole-domain ambiguous or \
insufficient interval while other task-relevant behavior is observable.
- If genuinely no usable task evidence exists, an empty semantic_segments \
list with a whole-domain NO_TASK_RELEVANT_ACTIVITY interval remains valid. \
This decision must be evidence-derived: a concrete manipulation \
episode_task (Grasp/Take/Insert/Place/...) is inconsistent with claiming \
the whole domain contains no task-relevant activity.
- Whole-domain INSUFFICIENT_VISUAL_EVIDENCE requires that the observations \
actually obstruct or conflict, not merely that no exact contact frame was \
sampled."""

PASS_A_OBJECT_IDENTITY = """\
OBJECT IDENTITY CONSISTENCY

- Use the most specific object identity supported by the views.
- If identity is uncertain, use a stable neutral noun such as object, \
component, or item.
- Do not alternate between incompatible object identities without new \
visual evidence."""

# ---------------------------------------------------------------- Pass B --

PASS_B_COARSE_PRESERVATION_RULES = """\
COARSE SEMANTIC PRESERVATION

- Pass A defines the episode task, coarse semantic phases, and coarse boundaries.
- Boundary uncertainty is not semantic-interval unusability.
- Pass B boundary ambiguity must not delete, replace, or convert an otherwise valid coarse semantic phase into a non-training interval.
- AMBIGUOUS means the exact transition position is uncertain; retain the coarse boundary and both coarse phases.
- INSUFFICIENT_VISUAL_EVIDENCE also retains coarse semantics by default and records review evidence.
- Only an explicit, sufficiently supported semantic_correction may rename one adjacent coarse semantic objective. Silent deletion, replacement, or object renaming is forbidden."""

PASS_B_OBJECT_IDENTITY = """\
OBJECT IDENTITY CONSISTENCY

- Keep object naming internally consistent across episode_task and coarse phases.
- When the visual category is uncertain, use a stable neutral name such as object, component, or item.
- Do not drift among more specific object names across phases without visual evidence.
- Pass B must preserve supplied object wording unless an explicit high-confidence semantic_correction requests a local rename."""

PASS_B_SG100_RULES = """\
SG100 STRICT LOCAL EVIDENCE

- SG100 is a multi-finger dexterous robot hand capable of reaching, enclosure, grasping, holding, and releasing.
- Capability is not event evidence: finger closing alone is never proof of acquisition.
- For exact local boundary placement require strict local evidence: \
persistent hand-object attachment, visible enclosure/contact at the \
sampled boundary neighborhood, subsequent object motion with the SG100, or \
equivalent before/after temporal evidence at this transition."""


GENERIC_COARSE_EVIDENCE_RULES = """\
END-EFFECTOR COARSE EVIDENCE

End-effector finger/gripper motion alone is not evidence of acquisition.

For coarse phase inference, use the temporal combination of:
- approach,
- changing end-effector-object spatial relationship,
- visible enclosure/contact when available,
- persistent proximity/attachment,
- later object displacement with the end effector.

The exact acquisition frame does not need to appear in the sparse Pass-A
samples. For example: an early sample showing the end effector approaching
an object, followed by a later sample where the object appears enclosed,
attached, or displaced with the end effector, supports a coarse grasp
hypothesis even if no sample shows the exact first-contact moment."""


def _coarse_evidence_rules(platform_context: PlatformContext) -> str:
    """SG100 wording only on SG100 platforms; generic elsewhere (spec: the
    LIBERO benchmark prompt must not mention our production hand)."""

    if "SG100" in platform_context.end_effector:
        return PASS_A_SG100_RULES
    return GENERIC_COARSE_EVIDENCE_RULES


def build_pass_a_prompt_v32(
    domains: list[dict[int, int]],
    allowed: list[int],
    *,
    platform_context: PlatformContext,
) -> str:
    """Global sparse Pass A: coarse recall, no Pass-B merge vocabulary."""

    return "\n\n".join(
        [
            platform_context.render(),
            IDENTITY_PREAMBLE,
            OUTPUT_PURPOSE,
            WHAT_NOT_HOW,
            MULTI_VIEW,
            LANGUAGE_RULES,
            EMBODIMENT_NEUTRAL_RULES,
            SPARSE_PASS_A_ROLE,
            PASS_A_EVIDENCE_CALIBRATION,
            PASS_A_RECALL_RULES,
            _coarse_evidence_rules(platform_context),
            PASS_A_OBJECT_IDENTITY,
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


def build_pass_b_prompt_v32(
    transition: dict[str, Any],
    allowed: list[int],
    *,
    platform_context: PlatformContext,
) -> str:
    """One boundary-only Pass B call; D4.3.1 merge contract unchanged."""

    from vla_data.annotation.prompts_v31 import build_pass_b_prompt_v31

    return build_pass_b_prompt_v31(
        transition,
        allowed,
        platform_context=platform_context,
    )
