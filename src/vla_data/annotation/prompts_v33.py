"""D4.3.3 canonical-compliant coarse semantics over the frozen v3.2 design.

Only Pass A changes: the v3.2 sparse-recall prompt is extended with narrow
canonical-language and coarse-objective-separation rules. Pass B retains the
v3.2 boundary-local contract unchanged.
"""

from __future__ import annotations

from typing import Any

from vla_data.annotation.platform_context import PlatformContext
from vla_data.annotation.prompts_v32 import (
    build_pass_a_prompt_v32,
    build_pass_b_prompt_v32,
)

PASS_A_PROMPT_VERSION = "vla_semantic_coarse_v3_3"
PASS_B_PROMPT_VERSION = "vla_semantic_boundary_refine_local_v3_3"

CANONICAL_OBJECT_NAMING_RULES = """\
CANONICAL TRAINING INSTRUCTION COMPLIANCE

- Canonical episode-task and semantic-segment instructions must explicitly name manipulated objects.
- Do not use unnamed object pronouns such as it, them, this, that, one, or ones when the pronoun refers to a manipulated object.
- Repeat the explicit object noun when needed; do not make the reader resolve an unnamed object pronoun.
- This restriction applies to canonical training instructions. Optional natural-language paraphrases remain metadata and need not use awkward noun repetition.

CANONICAL EXAMPLES (generic examples only):
BAD: Grasp the component and lift it
BAD: Pick it up
BAD: Move it toward the container
GOOD: Grasp the component
GOOD: Lift the component
GOOD: Move the component toward the container"""

COARSE_OBJECTIVE_SEPARATION_RULES = """\
COARSE OBJECTIVE SEPARATION

- When two sequential manipulation objectives occur at different temporal phases, represent them as separate semantic segments instead of combining them with "and" in one segment.
- Split when the manipulation objective changes, not for every visible motion change.
- Do not over-segment minor kinematic corrections, continuous execution of one objective, or incidental motion.

SEPARATION EXAMPLE (only when visually supported as temporally distinct):
BAD single segment: Grasp the component and lift the component
GOOD sequence: Grasp the component -> Lift the component"""


def build_pass_a_prompt_v33(
    domains: list[dict[int, int]],
    allowed: list[int],
    *,
    platform_context: PlatformContext,
) -> str:
    """Extend v3.2 Pass A without changing its sparse-recall calibration."""

    return "\n\n".join(
        [
            build_pass_a_prompt_v32(
                domains,
                allowed,
                platform_context=platform_context,
            ),
            CANONICAL_OBJECT_NAMING_RULES,
            COARSE_OBJECTIVE_SEPARATION_RULES,
        ]
    )


def build_pass_b_prompt_v33(
    transition: dict[str, Any],
    allowed: list[int],
    *,
    platform_context: PlatformContext,
) -> str:
    """Keep the frozen v3.2 boundary-local prompt semantics."""

    return build_pass_b_prompt_v32(
        transition,
        allowed,
        platform_context=platform_context,
    )
