"""Versioned episode annotation schema and the human review state machine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ANNOTATION_SCHEMA_NAME = "vla_episode_annotation"
ANNOTATION_SCHEMA_VERSION = 1

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
