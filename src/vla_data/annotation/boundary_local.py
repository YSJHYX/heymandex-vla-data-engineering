"""Pure planning and merge rules for D4.3 boundary-local refinement."""

from __future__ import annotations

import copy
from itertools import pairwise
from typing import Any

from vla_data.annotation.language_validator import validate_instruction
from vla_data.annotation.schema import AnnotationSchemaError

BOUNDARY_LOCAL_MODE = "boundary_local"
BOUNDARY_LOCAL_SAMPLING_STRATEGY = (
    "deterministic_per_transition_dense_boundary_neighborhood_v1"
)
EVIDENCE_STATUSES = frozenset(
    {
        "SUPPORTED",
        "PARTIALLY_SUPPORTED",
        "AMBIGUOUS",
        "INSUFFICIENT_VISUAL_EVIDENCE",
    }
)
UNCERTAIN_EVIDENCE_STATUSES = frozenset({"AMBIGUOUS", "INSUFFICIENT_VISUAL_EVIDENCE"})
SEMANTIC_CORRECTION_MIN_CONFIDENCE = 0.70


def build_boundary_transitions(
    coarse: dict[str, Any],
    domains: list[dict[str, int]],
    *,
    radius: int,
) -> list[dict[str, Any]]:
    """Create exactly one dense local plan per real coarse transition."""

    if radius < 0:
        raise AnnotationSchemaError("boundary radius must be nonnegative")
    timeline = _timeline(coarse)
    transitions: list[dict[str, Any]] = []
    for previous, following in pairwise(timeline):
        boundary = previous["end_curated_index"]
        if boundary != following["start_curated_index"]:
            continue
        domain = next(
            (
                domain
                for domain in domains
                if domain["start_curated_index"]
                < boundary
                < domain["end_curated_index"]
            ),
            None,
        )
        if domain is None:
            continue
        candidate_indices = list(
            range(
                max(domain["start_curated_index"], boundary - radius),
                min(domain["end_curated_index"], boundary + radius + 1),
            )
        )
        transitions.append(
            {
                "transition_id": f"boundary_{len(transitions):03d}",
                "coarse_boundary_curated_index": boundary,
                "previous": _transition_side(previous),
                "next": _transition_side(following),
                "candidate_indices": candidate_indices,
            }
        )
    return transitions


def validate_local_boundary_result(
    payload: object, transition: dict[str, Any]
) -> dict[str, Any]:
    """Validate the narrow Pass-B response without expanding its authority."""

    if not isinstance(payload, dict):
        raise AnnotationSchemaError("boundary-local result must be an object")
    transition_id = payload.get("transition_id")
    if transition_id != transition["transition_id"]:
        raise AnnotationSchemaError("boundary-local transition_id mismatch")
    for field, side in (
        ("previous_instruction", transition["previous"]),
        ("next_instruction", transition["next"]),
    ):
        if payload.get(field) != side["instruction"]:
            raise AnnotationSchemaError(
                f"{field} must preserve the supplied coarse objective"
            )
    boundary = payload.get("boundary_curated_index")
    if isinstance(boundary, bool) or not isinstance(boundary, int):
        raise AnnotationSchemaError("boundary_curated_index must be an integer")
    if boundary not in transition["candidate_indices"]:
        raise AnnotationSchemaError(
            "boundary_curated_index must come from candidate_indices"
        )
    confidence = payload.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise AnnotationSchemaError("boundary confidence must be a number")
    confidence = float(confidence)
    if not 0 <= confidence <= 1:
        raise AnnotationSchemaError("boundary confidence must be within [0, 1]")
    evidence_status = payload.get("evidence_status")
    if evidence_status not in EVIDENCE_STATUSES:
        raise AnnotationSchemaError(
            f"evidence_status must be one of {sorted(EVIDENCE_STATUSES)}"
        )
    if (
        evidence_status in UNCERTAIN_EVIDENCE_STATUSES
        and boundary != transition["coarse_boundary_curated_index"]
    ):
        raise AnnotationSchemaError(
            "ambiguous/insufficient local evidence must retain the coarse boundary"
        )
    correction = _validate_semantic_correction(
        payload.get("semantic_correction"), transition
    )
    if correction is not None and correction["confidence"] is None:
        correction["confidence"] = confidence
    rejected_correction = None
    if (
        correction is not None
        and correction["confidence"] < SEMANTIC_CORRECTION_MIN_CONFIDENCE
    ):
        rejected_correction = {
            **correction,
            "rejection_reason": ("SEMANTIC_CORRECTION_CONFIDENCE_BELOW_THRESHOLD"),
            "minimum_confidence": SEMANTIC_CORRECTION_MIN_CONFIDENCE,
        }
        correction = None
    return {
        "transition_id": transition_id,
        "previous_instruction": payload["previous_instruction"],
        "next_instruction": payload["next_instruction"],
        "boundary_curated_index": boundary,
        "confidence": confidence,
        "evidence_status": evidence_status,
        "semantic_correction": correction,
        "rejected_semantic_correction": rejected_correction,
    }


def merge_local_boundary_results(
    coarse: dict[str, Any],
    transitions: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Apply only adjacent boundary/correction updates to the coarse blocks."""

    if len(transitions) != len(results):
        raise AnnotationSchemaError("every planned boundary requires one result")
    refined = {
        "semantic_segments": copy.deepcopy(coarse.get("semantic_segments", [])),
        "non_training_intervals": copy.deepcopy(
            coarse.get("non_training_intervals", [])
        ),
    }
    for transition, result in zip(transitions, results, strict=True):
        previous = _resolve_side(refined, transition["previous"])
        following = _resolve_side(refined, transition["next"])
        selected = result["boundary_curated_index"]
        previous["end_curated_index"] = selected
        following["start_curated_index"] = selected
        correction = result.get("semantic_correction")
        if correction:
            if correction.get("previous_instruction") is not None:
                previous["instruction"] = correction["previous_instruction"]
            if correction.get("next_instruction") is not None:
                following["instruction"] = correction["next_instruction"]
    for blocks in refined.values():
        blocks.sort(key=lambda block: block["start_curated_index"])
    return refined


def _timeline(payload: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = []
    for collection in ("semantic_segments", "non_training_intervals"):
        values = payload.get(collection, [])
        if not isinstance(values, list):
            raise AnnotationSchemaError(f"{collection} must be a list")
        for index, block in enumerate(values):
            if not isinstance(block, dict):
                raise AnnotationSchemaError(f"{collection}[{index}] must be an object")
            blocks.append({**block, "_collection": collection, "_index": index})
    return sorted(blocks, key=lambda block: block["start_curated_index"])


def _transition_side(block: dict[str, Any]) -> dict[str, Any]:
    collection = block["_collection"]
    instruction = (
        block["instruction"]
        if collection == "semantic_segments"
        else f"NON_TRAINING:{block['reason']}"
    )
    return {
        "collection": collection,
        "index": block["_index"],
        "instruction": instruction,
    }


def _resolve_side(
    refined: dict[str, list[dict[str, Any]]], side: dict[str, Any]
) -> dict[str, Any]:
    return refined[side["collection"]][side["index"]]


def _validate_semantic_correction(
    value: object, transition: dict[str, Any]
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise AnnotationSchemaError("semantic_correction must be an object or null")
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise AnnotationSchemaError("semantic_correction requires a reason")
    confidence = value.get("confidence")
    if confidence is not None and (
        isinstance(confidence, bool) or not isinstance(confidence, (int, float))
    ):
        raise AnnotationSchemaError(
            "semantic_correction.confidence must be numeric or omitted"
        )
    if confidence is not None and not 0 <= float(confidence) <= 1:
        raise AnnotationSchemaError(
            "semantic_correction.confidence must be within [0, 1]"
        )
    normalized: dict[str, Any] = {
        "previous_instruction": None,
        "next_instruction": None,
        "reason": reason.strip(),
        "confidence": float(confidence) if confidence is not None else None,
    }
    for field, side in (
        ("previous_instruction", transition["previous"]),
        ("next_instruction", transition["next"]),
    ):
        candidate = value.get(field)
        if candidate is None:
            continue
        if side["collection"] != "semantic_segments":
            raise AnnotationSchemaError(
                "semantic correction cannot rename a non-training interval"
            )
        violations = validate_instruction(candidate)
        if violations:
            raise AnnotationSchemaError(
                f"semantic_correction.{field} violates controlled language: "
                f"{violations[0].category}"
            )
        normalized[field] = candidate.strip()
    if not normalized["previous_instruction"] and not normalized["next_instruction"]:
        raise AnnotationSchemaError(
            "semantic_correction must explicitly change an adjacent instruction"
        )
    return normalized
