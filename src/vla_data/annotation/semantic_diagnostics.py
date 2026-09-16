"""Review-only diagnostics for hierarchical semantic annotations."""

from __future__ import annotations

import re
from typing import Any

SEMANTIC_TASK_SEQUENCE_MISMATCH = "SEMANTIC_TASK_SEQUENCE_MISMATCH"
PASS_A_INTERNAL_INCONSISTENCY = "PASS_A_INTERNAL_INCONSISTENCY"
VERY_SHORT_NONTRAINING_INTERVAL = "VERY_SHORT_NONTRAINING_INTERVAL"
BOUNDARY_EVIDENCE_AMBIGUOUS = "BOUNDARY_EVIDENCE_AMBIGUOUS"
BOUNDARY_EVIDENCE_INSUFFICIENT = "BOUNDARY_EVIDENCE_INSUFFICIENT"
REJECTED_OPTIONAL_PARAPHRASE = "REJECTED_OPTIONAL_PARAPHRASE"

_INTENT_PATTERNS = {
    "APPROACH": (r"\breach\b", r"\bapproach\b"),
    "ACQUIRE": (
        r"\bgrasp\b",
        r"\bpick(?: up)?\b",
        r"\btake\b",
        r"\breceive\b",
        r"\bhold\b",
    ),
    "TRANSPORT": (r"\bmove\b", r"\bcarry\b", r"\btransfer\b", r"\blift\b"),
    "ALIGN": (r"\balign\b", r"\bposition\b", r"\borient\b"),
    "INSERT_CONNECT": (
        r"\binsert\b",
        r"\bplug\b",
        r"\bconnect\b",
        r"\bseat\b",
    ),
    "RELEASE": (r"\brelease\b", r"\bplace\b", r"\bset down\b"),
    "ACTUATE": (
        r"\bpress\b",
        r"\bpush\b",
        r"\bpull\b",
        r"\bturn\b",
        r"\bopen\b",
        r"\bclose\b",
    ),
}


def normalized_intents(text: str) -> set[str]:
    """Map wording to broad observable intent classes, never a fixed plan."""

    lowered = text.lower()
    return {
        intent
        for intent, patterns in _INTENT_PATTERNS.items()
        if any(re.search(pattern, lowered) for pattern in patterns)
    }


def task_sequence_mismatch(annotation: dict[str, Any]) -> bool:
    """Flag only an obvious goal-versus-approach mismatch for human review."""

    task_intents = normalized_intents(annotation["episode_task"]["instruction"])
    segment_intents = set().union(
        *(
            normalized_intents(segment["instruction"])
            for segment in annotation["semantic_segments"]
        ),
        set(),
    )
    if not segment_intents:
        return False
    if "ACQUIRE" in task_intents:
        return not bool(segment_intents & {"ACQUIRE", "TRANSPORT"})
    if "INSERT_CONNECT" in task_intents:
        return not bool(
            segment_intents & {"ACQUIRE", "TRANSPORT", "ALIGN", "INSERT_CONNECT"}
        )
    return False


_CONCRETE_MANIPULATION_PATTERN = re.compile(
    r"\b(?:grasp|take|pick|place|insert|plug|connect|press|push|pull|turn|"
    r"open|close|lift|move|transfer|release|align|seat|remove|hold)\b",
    re.IGNORECASE,
)


def pass_a_internal_inconsistency(annotation: dict[str, Any]) -> bool:
    """A concrete manipulation episode_task cannot coexist with a fully
    non-task domain and zero semantic segments (D4.3.1 005 pattern)."""

    task = annotation.get("episode_task")
    if not isinstance(task, dict):
        return False
    instruction = str(task.get("instruction", ""))
    if not _CONCRETE_MANIPULATION_PATTERN.search(instruction):
        return False
    if annotation.get("semantic_segments"):
        return False
    intervals = annotation.get("non_training_intervals", [])
    reasons = {
        str(interval.get("reason"))
        for interval in intervals
        if isinstance(interval, dict)
    }
    return reasons == {"NO_TASK_RELEVANT_ACTIVITY"}


def concrete_task_with_insufficient_domain(annotation: dict[str, Any]) -> bool:
    """A concrete manipulation episode_task plus a whole-domain
    INSUFFICIENT_VISUAL_EVIDENCE interval warrants recall review (003 pattern)."""

    task = annotation.get("episode_task")
    if not isinstance(task, dict):
        return False
    if not _CONCRETE_MANIPULATION_PATTERN.search(str(task.get("instruction", ""))):
        return False
    if annotation.get("semantic_segments"):
        return False
    intervals = annotation.get("non_training_intervals", [])
    reasons = {
        str(interval.get("reason"))
        for interval in intervals
        if isinstance(interval, dict)
    }
    return "INSUFFICIENT_VISUAL_EVIDENCE" in reasons and len(reasons) == 1


def very_short_nontraining_intervals(
    annotation: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return one-transition non-training spans as review warnings."""

    return [
        {
            "start_curated_index": interval["start_curated_index"],
            "end_curated_index": interval["end_curated_index"],
            "reason": interval["reason"],
            "length": interval["end_curated_index"] - interval["start_curated_index"],
        }
        for interval in annotation["non_training_intervals"]
        if interval["end_curated_index"] - interval["start_curated_index"] <= 1
    ]
