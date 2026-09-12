"""Schema validation and the human review state machine."""

from __future__ import annotations

import pytest

from vla_data.annotation.schema import (
    ANNOTATION_SCHEMA_VERSION,
    REVIEW_AUTO_LABELED,
    REVIEW_HUMAN_CORRECTED,
    REVIEW_HUMAN_VERIFIED,
    REVIEW_REJECTED,
    AnnotationSchemaError,
    apply_review,
    build_annotation,
    load_annotation,
    resolve_final_instruction,
    validate_model_payload,
    write_annotation,
)

VALID_PAYLOAD = {
    "instruction": "Pick up the black component and place it into the tray.",
    "confidence": 0.91,
    "task_type": "pick_and_place",
    "objects": ["component", "tray"],
    "uncertainty": None,
}


def _annotation() -> dict:
    return build_annotation(
        episode_id="episode_000001",
        model_annotation={
            "provider": "p",
            "model": "m",
            "prompt_version": "task_instruction_v1",
            **validate_model_payload(VALID_PAYLOAD),
            "request_id": "req-1",
            "attempt_count": 1,
        },
        quality_outcome="ACCEPT",
    )


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ("not-a-dict", "object"),
        ({**VALID_PAYLOAD, "instruction": " "}, "non-empty"),
        ({**VALID_PAYLOAD, "instruction": "x" * 241}, "at most"),
        ({**VALID_PAYLOAD, "confidence": 1.5}, "within"),
        ({**VALID_PAYLOAD, "confidence": -0.1}, "within"),
        ({**VALID_PAYLOAD, "confidence": True}, "number"),
        ({**VALID_PAYLOAD, "objects": "tray"}, "list"),
        ({**VALID_PAYLOAD, "objects": [1]}, "strings"),
        ({**VALID_PAYLOAD, "task_type": 3}, "string or null"),
        ({**VALID_PAYLOAD, "uncertainty": 5}, "string or null"),
    ],
)
def test_invalid_model_payloads_are_rejected(payload, reason) -> None:
    with pytest.raises(AnnotationSchemaError, match=reason):
        validate_model_payload(payload)


def test_valid_payload_keeps_only_known_fields() -> None:
    kept = validate_model_payload({**VALID_PAYLOAD, "secret": "leak"})
    assert set(kept) == {
        "instruction",
        "confidence",
        "task_type",
        "objects",
        "uncertainty",
    }
    assert kept["confidence"] == pytest.approx(0.91)


def test_fresh_annotation_is_auto_labeled_with_null_final() -> None:
    annotation = _annotation()
    assert annotation["schema_version"] == ANNOTATION_SCHEMA_VERSION
    assert annotation["review"]["status"] == REVIEW_AUTO_LABELED
    assert annotation["final_instruction"] is None
    assert resolve_final_instruction(annotation) is None


def test_auto_labeled_rejects_direct_final_instruction_mismatch() -> None:
    annotation = _annotation()
    annotation["final_instruction"] = "smuggled"
    with pytest.raises(AnnotationSchemaError, match="state machine"):
        load_annotation_if_written(annotation)


def load_annotation_if_written(annotation: dict) -> dict:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "annotation.json"
        write_annotation(path, annotation)
        return load_annotation(path)


def test_human_verified_resolves_to_model_instruction() -> None:
    annotation = apply_review(
        _annotation(), status=REVIEW_HUMAN_VERIFIED, reviewer="alice"
    )
    assert annotation["final_instruction"] == VALID_PAYLOAD["instruction"]
    assert annotation["model_annotation"]["instruction"] == VALID_PAYLOAD["instruction"]


def test_human_corrected_preserves_original_model_label() -> None:
    annotation = apply_review(
        _annotation(),
        status=REVIEW_HUMAN_CORRECTED,
        reviewer="bob",
        corrected_instruction="Pick up the component and insert it into the slot.",
    )
    assert annotation["final_instruction"].startswith("Pick up the component")
    # The original model output is preserved verbatim for provenance.
    assert annotation["model_annotation"]["instruction"] == VALID_PAYLOAD["instruction"]
    assert annotation["review"]["corrected_instruction"].startswith("Pick up the")


def test_rejected_resolves_to_null_final_instruction() -> None:
    annotation = apply_review(_annotation(), status=REVIEW_REJECTED, reviewer="carol")
    assert annotation["final_instruction"] is None
    assert annotation["model_annotation"]["instruction"] == VALID_PAYLOAD["instruction"]


def test_illegal_review_transitions_are_rejected() -> None:
    with pytest.raises(AnnotationSchemaError, match="not allowed"):
        apply_review(_annotation(), status=REVIEW_AUTO_LABELED)
    with pytest.raises(AnnotationSchemaError, match="unknown review status"):
        apply_review(_annotation(), status="NOT_A_STATE")
    corrected = apply_review(
        _annotation(), status=REVIEW_HUMAN_CORRECTED, corrected_instruction="Fix it."
    )
    with pytest.raises(AnnotationSchemaError, match="not allowed"):
        apply_review(corrected, status=REVIEW_AUTO_LABELED, reviewer="dave")
