"""Versioned verification records, distinct from immutable D4 annotations."""

import hashlib
import json
from pathlib import Path

from vla_data.batch.discovery import canonical_episode_id
from vla_data.batch.summary import write_summary
from vla_data.verification.policy import VerificationPolicy, probability

SCHEMA_NAME = "vla_annotation_verification"
SCHEMA_VERSION = 1
APPROVED = {"AUTO_VERIFIED", "HUMAN_VERIFIED", "HUMAN_CORRECTED"}
HUMAN_STATES = {"HUMAN_VERIFIED", "HUMAN_CORRECTED", "REJECTED"}
STATES = APPROVED | {"NEEDS_HUMAN_REVIEW", "REJECTED", "INELIGIBLE"}


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def seal(record: dict) -> dict:
    record["fingerprint"] = digest(
        {k: v for k, v in record.items() if k != "fingerprint"}
    )
    return record


def validate_verification(record: object) -> dict:
    if not isinstance(record, dict):
        raise TypeError("verification must be an object")
    if (
        record.get("schema_name") != SCHEMA_NAME
        or record.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("unsupported verification schema")
    canonical_episode_id(record["episode_id"])
    policy = VerificationPolicy(**record["policy"])
    status = record["verification_status"]
    if status not in STATES:
        raise ValueError("invalid verification status")
    if record["fingerprint"] != digest(
        {k: v for k, v in record.items() if k != "fingerprint"}
    ):
        raise ValueError("verification fingerprint mismatch")
    if record["input_fingerprint"] != digest(
        [SCHEMA_VERSION, record["policy"], record["input_identity"]]
    ):
        raise ValueError("verification input fingerprint mismatch")
    review = record["human_review"]
    if not isinstance(review, dict):
        raise TypeError("human_review must be an object")
    if status == "INELIGIBLE":
        if not record["reason"] or record["final_instruction"] is not None:
            raise ValueError(
                "ineligible verification must carry reason and null instruction"
            )
        return record
    if record["quality_eligible"] is not True or record["reason"]:
        raise ValueError("verification cannot override quality/input ineligibility")
    confidence = probability(record["model_confidence"], "model_confidence")
    instruction = record["model_instruction"]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("model instruction must be non-empty")
    if status in HUMAN_STATES:
        if record["verification_source"] != "HUMAN_REVIEW":
            raise ValueError("human state requires HUMAN_REVIEW source")
    else:
        expected = (
            "AUTO_VERIFIED"
            if confidence >= policy.confidence_threshold
            else "NEEDS_HUMAN_REVIEW"
        )
        if (
            status != expected
            or record["verification_source"] != "MODEL_CONFIDENCE_POLICY"
        ):
            raise ValueError("verification status disagrees with confidence policy")
    final = instruction if status in {"AUTO_VERIFIED", "HUMAN_VERIFIED"} else None
    if status == "HUMAN_CORRECTED":
        final = review["corrected_instruction"]
        if not isinstance(final, str) or not final.strip():
            raise ValueError("HUMAN_CORRECTED requires corrected_instruction")
    if record["final_instruction"] != final:
        raise ValueError("final instruction does not match verification resolution")
    return record


def load_verification(path: str | Path, *, check_current: bool = True) -> dict:
    record = validate_verification(json.loads(Path(path).read_text()))
    if check_current:
        from vla_data.verification.evaluator import evaluate_verification

        current = evaluate_verification(
            record["episode_id"],
            Path(record["annotation_path"]),
            Path(record["quality_path"]),
            VerificationPolicy(**record["policy"]),
        )
        if current["input_fingerprint"] != record["input_fingerprint"]:
            raise ValueError("verification sources changed; rerun verify-annotations")
        for key in (
            "model_instruction",
            "model_confidence",
            "quality_eligible",
            "reason",
        ):
            if current[key] != record[key]:
                raise ValueError(f"verification source resolution mismatch: {key}")
    return record


def write_verification(path: Path, record: dict) -> None:
    write_summary(path, validate_verification(record))
