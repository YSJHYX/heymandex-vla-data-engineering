"""Versioned episode annotation schema and the human review state machine."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

ANNOTATION_SCHEMA_NAME = "vla_episode_annotation"
ANNOTATION_SCHEMA_VERSION = 1
HIERARCHICAL_ANNOTATION_SCHEMA_VERSION = 2

INSTRUCTION_MAX_LENGTH = 240

REVIEW_AUTO_LABELED = "AUTO_LABELED"
REVIEW_AUTO_ACCEPTED = "AUTO_ACCEPTED"
REVIEW_HUMAN_VERIFIED = "HUMAN_VERIFIED"
REVIEW_HUMAN_CORRECTED = "HUMAN_CORRECTED"
REVIEW_REJECTED = "REJECTED"

REVIEW_STATES = (
    REVIEW_AUTO_LABELED,
    REVIEW_AUTO_ACCEPTED,
    REVIEW_HUMAN_VERIFIED,
    REVIEW_HUMAN_CORRECTED,
    REVIEW_REJECTED,
)

# Model-first: humans may accept, verify, correct, or reject — never author.
ALLOWED_REVIEW_TRANSITIONS = {
    REVIEW_AUTO_LABELED: (
        REVIEW_AUTO_ACCEPTED,
        REVIEW_HUMAN_VERIFIED,
        REVIEW_HUMAN_CORRECTED,
        REVIEW_REJECTED,
    ),
    REVIEW_AUTO_ACCEPTED: (
        REVIEW_HUMAN_VERIFIED,
        REVIEW_HUMAN_CORRECTED,
        REVIEW_REJECTED,
    ),
    REVIEW_HUMAN_VERIFIED: (REVIEW_HUMAN_CORRECTED, REVIEW_REJECTED),
    REVIEW_HUMAN_CORRECTED: (REVIEW_REJECTED, REVIEW_HUMAN_VERIFIED),
    REVIEW_REJECTED: (REVIEW_HUMAN_VERIFIED, REVIEW_HUMAN_CORRECTED),
}


class AnnotationSchemaError(ValueError):
    """The candidate annotation does not satisfy the versioned schema."""


_ACTUATOR_WORDS = frozenset(
    {
        "joint",
        "joints",
        "motor",
        "motors",
        "actuator",
        "servo",
        "qpos",
        "qcmd",
        "radian",
        "关节",
        "电机",
        "执行器",
    }
)


def validate_task_instruction(value: object, field: str) -> str:
    """Require semantic task language, never actuator-level narration."""

    if not isinstance(value, str) or not value.strip():
        raise AnnotationSchemaError(f"{field} must be a non-empty string")
    text = value.strip()
    if len(text) > INSTRUCTION_MAX_LENGTH:
        raise AnnotationSchemaError(
            f"{field} must be at most {INSTRUCTION_MAX_LENGTH} characters"
        )
    words = {token.strip('.,:;!?()[]{}"').lower() for token in text.split()}
    if words & _ACTUATOR_WORDS or any(
        word in text for word in ("关节", "电机", "执行器")
    ):
        raise AnnotationSchemaError(f"{field} must describe WHAT, not actuator motion")
    return text


def _bounded_index(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnnotationSchemaError(f"{field} must be an integer")
    if value < 0:
        raise AnnotationSchemaError(f"{field} must be nonnegative")
    return value


def _validate_instruction_block(block: object, field: str) -> dict[str, Any]:
    if not isinstance(block, dict):
        raise AnnotationSchemaError(f"{field} must be an object")
    validate_task_instruction(block.get("instruction"), f"{field}.instruction")
    confidence = block.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise AnnotationSchemaError(f"{field}.confidence must be a number")
    if not 0 <= float(confidence) <= 1:
        raise AnnotationSchemaError(f"{field}.confidence must be within [0, 1]")
    paraphrases = block.get("paraphrases", [])
    if not isinstance(paraphrases, list):
        raise AnnotationSchemaError(f"{field}.paraphrases must be a list")
    for number, text in enumerate(paraphrases):
        validate_task_instruction(text, f"{field}.paraphrases[{number}]")
    task_type = block.get("task_type")
    if task_type is not None and not isinstance(task_type, str):
        raise AnnotationSchemaError(f"{field}.task_type must be a string or null")
    objects = block.get("objects", [])
    if not isinstance(objects, list) or not all(
        isinstance(item, str) for item in objects
    ):
        raise AnnotationSchemaError(f"{field}.objects must be a list of strings")
    uncertainty = block.get("uncertainty")
    if uncertainty is not None and not isinstance(uncertainty, str):
        raise AnnotationSchemaError(f"{field}.uncertainty must be a string or null")
    return block


def validate_hierarchical_annotation(annotation: object) -> dict[str, Any]:
    """Validate annotation v2, including temporal-domain containment."""

    if not isinstance(annotation, dict):
        raise AnnotationSchemaError("annotation must be a JSON object")
    if annotation.get("schema_name") != ANNOTATION_SCHEMA_NAME:
        raise AnnotationSchemaError(f"schema_name must be {ANNOTATION_SCHEMA_NAME!r}")
    if annotation.get("schema_version") != HIERARCHICAL_ANNOTATION_SCHEMA_VERSION:
        raise AnnotationSchemaError("hierarchical annotation schema_version must be 2")
    if annotation.get("annotation_schema_version") != 2:
        raise AnnotationSchemaError("annotation_schema_version must be 2")
    if (
        not isinstance(annotation.get("episode_id"), str)
        or not annotation["episode_id"]
    ):
        raise AnnotationSchemaError("episode_id must be a non-empty string")
    _validate_instruction_block(annotation.get("episode_task"), "episode_task")

    domains = annotation.get("clean_domains")
    if not isinstance(domains, list) or not domains:
        raise AnnotationSchemaError("clean_domains must be a non-empty list")
    normalized_domains: list[tuple[int, int]] = []
    for number, domain in enumerate(domains):
        if not isinstance(domain, dict):
            raise AnnotationSchemaError(f"clean_domains[{number}] must be an object")
        start = _bounded_index(domain.get("start_curated_index"), "domain start")
        end = _bounded_index(domain.get("end_curated_index"), "domain end")
        if start >= end:
            raise AnnotationSchemaError("clean domain must satisfy start < end")
        normalized_domains.append((start, end))
    if normalized_domains != sorted(normalized_domains) or any(
        left[1] > right[0] for left, right in pairwise(normalized_domains)
    ):
        raise AnnotationSchemaError("clean domains must be sorted and non-overlapping")

    segments = annotation.get("semantic_segments")
    intervals = annotation.get("non_training_intervals")
    if not isinstance(segments, list) or not isinstance(intervals, list):
        raise AnnotationSchemaError(
            "semantic_segments and non_training_intervals must be lists"
        )
    covered: list[tuple[int, int, str]] = []
    ids: set[str] = set()
    previous_start = -1
    for number, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise AnnotationSchemaError(
                f"semantic_segments[{number}] must be an object"
            )
        segment_id = segment.get("segment_id")
        if not isinstance(segment_id, str) or not segment_id or segment_id in ids:
            raise AnnotationSchemaError(
                "semantic segment IDs must be non-empty and unique"
            )
        ids.add(segment_id)
        start = _bounded_index(segment.get("start_curated_index"), "segment start")
        end = _bounded_index(segment.get("end_curated_index"), "segment end")
        if start >= end:
            raise AnnotationSchemaError("semantic segment must satisfy start < end")
        if start < previous_start:
            raise AnnotationSchemaError("semantic segments must be sorted")
        previous_start = start
        _validate_instruction_block(segment, f"semantic_segments[{number}]")
        covered.append((start, end, f"semantic segment {segment_id}"))
    previous_interval_start = -1
    for number, interval in enumerate(intervals):
        if not isinstance(interval, dict):
            raise AnnotationSchemaError(
                f"non_training_intervals[{number}] must be an object"
            )
        start = _bounded_index(interval.get("start_curated_index"), "interval start")
        end = _bounded_index(interval.get("end_curated_index"), "interval end")
        if start >= end:
            raise AnnotationSchemaError(
                "non-training interval must satisfy start < end"
            )
        if start < previous_interval_start:
            raise AnnotationSchemaError("non-training intervals must be sorted")
        previous_interval_start = start
        if (
            not isinstance(interval.get("reason"), str)
            or not interval["reason"].strip()
        ):
            raise AnnotationSchemaError("non-training interval requires a reason")
        covered.append((start, end, f"non-training interval {number}"))

    covered.sort()
    for start, end, label in covered:
        containing = [d for d in normalized_domains if d[0] <= start < end <= d[1]]
        if len(containing) != 1:
            raise AnnotationSchemaError(
                f"{label} crosses or lies outside a clean domain"
            )
    if any(left[1] > right[0] for left, right in pairwise(covered)):
        raise AnnotationSchemaError("semantic and non-training intervals overlap")
    expected = [(index, domain) for index, domain in enumerate(normalized_domains)]
    for _, (domain_start, domain_end) in expected:
        pieces = [
            (start, end)
            for start, end, _ in covered
            if domain_start <= start and end <= domain_end
        ]
        cursor = domain_start
        for start, end in pieces:
            if start != cursor:
                raise AnnotationSchemaError(
                    "clean domains must be explicitly and completely classified"
                )
            cursor = end
        if cursor != domain_end:
            raise AnnotationSchemaError(
                "clean domains must be explicitly and completely classified"
            )
    provenance = annotation.get("model_provenance")
    if not isinstance(provenance, dict):
        raise AnnotationSchemaError("model_provenance must be an object")
    for key in ("provider", "model", "pass_a_prompt_version", "pass_b_prompt_version"):
        if not isinstance(provenance.get(key), str) or not provenance[key]:
            raise AnnotationSchemaError(f"model_provenance.{key} is required")
    return annotation


def validate_model_payload(payload: object) -> dict[str, Any]:
    """Strictly validate the parsed model JSON payload and keep only known keys."""

    if not isinstance(payload, dict):
        raise AnnotationSchemaError("model payload must be a JSON object")

    instruction = payload.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise AnnotationSchemaError("instruction must be a non-empty string")
    if len(instruction) > INSTRUCTION_MAX_LENGTH:
        raise AnnotationSchemaError(
            f"instruction must be at most {INSTRUCTION_MAX_LENGTH} characters"
        )

    confidence = payload.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise AnnotationSchemaError("confidence must be a number")
    if not 0.0 <= float(confidence) <= 1.0:
        raise AnnotationSchemaError("confidence must be within [0, 1]")

    task_type = payload.get("task_type")
    if task_type is not None and not isinstance(task_type, str):
        raise AnnotationSchemaError("task_type must be a string or null")

    objects = payload.get("objects")
    if not isinstance(objects, list) or not all(
        isinstance(item, str) for item in objects
    ):
        raise AnnotationSchemaError("objects must be a list of strings")

    uncertainty = payload.get("uncertainty")
    if uncertainty is not None and not isinstance(uncertainty, str):
        raise AnnotationSchemaError("uncertainty must be a string or null")

    # Unknown model fields never enter the production annotation.
    return {
        "instruction": instruction.strip(),
        "confidence": float(confidence),
        "task_type": task_type,
        "objects": list(objects),
        "uncertainty": uncertainty,
    }


def build_annotation(
    *,
    episode_id: str,
    model_annotation: dict[str, Any],
    quality_outcome: str,
) -> dict[str, Any]:
    """Assemble a fresh vla_episode_annotation in the AUTO_LABELED state."""

    return {
        "schema_name": ANNOTATION_SCHEMA_NAME,
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "episode_id": episode_id,
        "quality_outcome": quality_outcome,
        "model_annotation": dict(model_annotation),
        "review": {
            "status": REVIEW_AUTO_LABELED,
            "reviewer": None,
            "corrected_instruction": None,
        },
        "final_instruction": None,
    }


def validate_annotation(annotation: object) -> dict[str, Any]:
    """Validate a persisted annotation document and return it unchanged."""

    if not isinstance(annotation, dict):
        raise AnnotationSchemaError("annotation must be a JSON object")
    if annotation.get("schema_version") == HIERARCHICAL_ANNOTATION_SCHEMA_VERSION:
        return validate_hierarchical_annotation(annotation)
    if annotation.get("schema_name") != ANNOTATION_SCHEMA_NAME:
        raise AnnotationSchemaError(f"schema_name must be {ANNOTATION_SCHEMA_NAME!r}")
    if annotation.get("schema_version") != ANNOTATION_SCHEMA_VERSION:
        raise AnnotationSchemaError(
            f"schema_version must be {ANNOTATION_SCHEMA_VERSION}"
        )
    episode_id = annotation.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise AnnotationSchemaError("episode_id must be a non-empty string")

    model_block = annotation.get("model_annotation")
    if not isinstance(model_block, dict):
        raise AnnotationSchemaError("model_annotation must be an object")
    for key in ("provider", "model", "prompt_version", "instruction", "confidence"):
        if key not in model_block:
            raise AnnotationSchemaError(f"model_annotation is missing {key!r}")
    validate_model_payload({**model_block, "objects": model_block.get("objects", [])})

    review = annotation.get("review")
    if not isinstance(review, dict):
        raise AnnotationSchemaError("review must be an object")
    if review.get("status") not in REVIEW_STATES:
        raise AnnotationSchemaError(f"unknown review status: {review.get('status')!r}")

    final_instruction = annotation.get("final_instruction")
    if final_instruction is not None and not isinstance(final_instruction, str):
        raise AnnotationSchemaError("final_instruction must be a string or null")

    resolved = resolve_final_instruction(annotation)
    if final_instruction != resolved:
        raise AnnotationSchemaError(
            "final_instruction does not match the review state machine"
        )
    return annotation


def resolve_final_instruction(annotation: dict[str, Any]) -> str | None:
    """Apply the review state machine to derive the final instruction."""

    status = annotation["review"]["status"]
    if status in {REVIEW_AUTO_ACCEPTED, REVIEW_HUMAN_VERIFIED}:
        return annotation["model_annotation"]["instruction"]
    if status == REVIEW_HUMAN_CORRECTED:
        corrected = annotation["review"]["corrected_instruction"]
        if not isinstance(corrected, str) or not corrected.strip():
            raise AnnotationSchemaError(
                "HUMAN_CORRECTED requires a non-empty corrected_instruction"
            )
        return corrected.strip()
    return None


def apply_review(
    annotation: dict[str, Any],
    *,
    status: str,
    reviewer: str | None = None,
    corrected_instruction: str | None = None,
) -> dict[str, Any]:
    """Return a new annotation with the review transition applied.

    The original model annotation block is never modified; corrections live in
    the review block only.
    """

    current = annotation["review"]["status"]
    if status not in REVIEW_STATES:
        raise AnnotationSchemaError(f"unknown review status: {status!r}")
    if status not in ALLOWED_REVIEW_TRANSITIONS[current]:
        raise AnnotationSchemaError(
            f"review transition {current} -> {status} is not allowed"
        )
    updated = json.loads(json.dumps(annotation))  # deep copy, no shared state
    if status == REVIEW_HUMAN_CORRECTED:
        if (
            not isinstance(corrected_instruction, str)
            or not corrected_instruction.strip()
        ):
            raise AnnotationSchemaError(
                "HUMAN_CORRECTED requires a non-empty corrected_instruction"
            )
        updated["review"]["corrected_instruction"] = corrected_instruction.strip()
    updated["review"]["status"] = status
    updated["review"]["reviewer"] = reviewer
    updated["final_instruction"] = resolve_final_instruction(updated)
    return validate_annotation(updated)


def write_annotation(path: str | Path, annotation: dict[str, Any]) -> Path:
    """Atomically persist one annotation document."""

    import os
    import tempfile

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    validate_annotation(annotation)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(annotation, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_annotation(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate one persisted annotation document."""

    with Path(path).open(encoding="utf-8") as stream:
        annotation = json.load(stream)
    return validate_annotation(annotation)
