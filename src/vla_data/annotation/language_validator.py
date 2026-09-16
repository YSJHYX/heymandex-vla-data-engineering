"""Deterministic controlled-language validation for training instructions.

Form constraints only — this never whitelists specific task verbs, so new
manipulation domains (press, insert, pull, push, turn, open, close, seat,
align, remove, transfer, ...) stay expressible. Violations are classified
for observability; the pipeline maps them to MODEL_OUTPUT_SCHEMA_ERROR.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_INSTRUCTION_LENGTH = 160
MIN_INSTRUCTION_LENGTH = 4

# Narrative/speculative prefixes that betray captioning, not task labeling.
FORBIDDEN_PREFIXES = (
    "the robot",
    "the robotic",
    "the arm",
    "the hand is",
    "it seems",
    "it appears",
    "we can see",
    "in this image",
    "in this frame",
    "in the image",
    "according to",
    "i think",
    "i guess",
    "maybe",
    "perhaps",
    "probably",
    "possibly",
    "looks like",
    "seems like",
)

SPECULATIVE_WORDS = (
    "appears to",
    "seems to",
    "seems like",
    "might be",
    "may be",
)

ACTUATOR_PATTERNS = (
    r"\bjoint\s*\d+\b",
    r"\bqpos\b",
    r"\bqcmd\b",
    r"\bjoints?\b.*\b(rotat|move|flex|bend|extend)\w*",
    r"\bservo\b",
    r"\bmotor\b",
    r"\bactuator\b",
    r"\bgripper (command|position|target)\b",
    r"\b(degrees?|deg|radians?)\b",
    r"\b17d\b",
    r"\b1[76][- ]?d(im(ensional)?)?\b",
)

# Task instructions must express the external manipulation goal, not command
# the robot body. Anchoring at the imperative subject keeps human context such
# as "Take the object from the operator's hand" legal.
EMBODIMENT_CENTRIC_PATTERNS = (
    r"^(?:move|rotate|position|orient|raise|lower)\s+(?:the\s+)?(?:robot(?:ic)?\s+)?(?:arm|hand|wrist|end[- ]effector|gripper)\b",
    r"^(?:reach|extend)\s+(?:the\s+)?(?:robot(?:ic)?\s+)?(?:arm|hand|wrist|end[- ]effector|gripper)\b",
)

EVALUATIVE_WORDS = (
    "successfully",
    "correctly",
    "perfectly",
    "carefully",
    "slowly",
    "precisely",
)

PRONOUN_OBJECT_PATTERNS = (
    r"\b(?:pick|move|insert|place|grasp|lift|hold|release|press)\s+(?:it|them|this|that)\b",
)

MARKDOWN_PATTERN = re.compile(r"[*#`_{}\[\]]|^\s*[-+]\s")

CLASS_NARRATIVE_PREFIX = "NARRATIVE_OR_SPECULATIVE_PREFIX"
CLASS_SPECULATIVE = "SPECULATIVE_WORDING"
CLASS_ACTUATOR = "ACTUATOR_NARRATION"
CLASS_EMBODIMENT = "EMBODIMENT_CENTRIC_LANGUAGE"
CLASS_EVALUATIVE = "EVALUATIVE_WORDING"
CLASS_PRONOUN = "UNNAMED_PRONOUN_OBJECT"
CLASS_MARKDOWN = "MARKDOWN_OR_STRUCTURE"
CLASS_LENGTH = "INSTRUCTION_LENGTH"
CLASS_EMPTY = "EMPTY_INSTRUCTION"


@dataclass(frozen=True)
class LanguageViolation:
    instruction: str
    reason: str
    category: str


def normalize_instruction(text: str) -> str:
    """Deterministic light normalization only; never semantic rewriting."""

    return " ".join(text.strip().rstrip(".").split())


def validate_instruction(text: str) -> list[LanguageViolation]:
    """Return every deterministic violation; empty list means acceptable."""

    if not isinstance(text, str):
        return [LanguageViolation(str(text), "not a string", CLASS_EMPTY)]
    normalized = text.strip()
    if not normalized:
        return [LanguageViolation(text, "empty", CLASS_EMPTY)]
    violations: list[LanguageViolation] = []
    lowered = normalized.lower()
    if lowered.startswith(FORBIDDEN_PREFIXES):
        violations.append(
            LanguageViolation(
                normalized, "narrative/speculative prefix", CLASS_NARRATIVE_PREFIX
            )
        )
    for phrase in SPECULATIVE_WORDS:
        if phrase in lowered:
            violations.append(
                LanguageViolation(normalized, f"contains {phrase!r}", CLASS_SPECULATIVE)
            )
            break
    for pattern in ACTUATOR_PATTERNS:
        if re.search(pattern, lowered):
            violations.append(
                LanguageViolation(
                    normalized, "actuator/joint narration", CLASS_ACTUATOR
                )
            )
            break
    for pattern in EMBODIMENT_CENTRIC_PATTERNS:
        if re.search(pattern, lowered):
            violations.append(
                LanguageViolation(
                    normalized,
                    "robot embodiment motion used as task language",
                    CLASS_EMBODIMENT,
                )
            )
            break
    for word in EVALUATIVE_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            violations.append(
                LanguageViolation(
                    normalized, f"evaluative word {word!r}", CLASS_EVALUATIVE
                )
            )
            break
    for pattern in PRONOUN_OBJECT_PATTERNS:
        if re.search(pattern, lowered):
            violations.append(
                LanguageViolation(normalized, "pronoun object", CLASS_PRONOUN)
            )
            break
    if MARKDOWN_PATTERN.search(normalized):
        violations.append(
            LanguageViolation(normalized, "markdown/structured text", CLASS_MARKDOWN)
        )
    if len(normalized) > MAX_INSTRUCTION_LENGTH:
        violations.append(LanguageViolation(normalized, "too long", CLASS_LENGTH))
    if len(normalized) < MIN_INSTRUCTION_LENGTH:
        violations.append(LanguageViolation(normalized, "too short", CLASS_LENGTH))
    return violations


def is_reviewable_speculative(text: str) -> bool:
    """Low-confidence-speculation variants (e.g. 'Maybe grasp the object')."""

    return bool(validate_instruction(text)) and any(
        violation.category
        in {CLASS_NARRATIVE_PREFIX, CLASS_SPECULATIVE, CLASS_EVALUATIVE}
        for violation in validate_instruction(text)
    )
