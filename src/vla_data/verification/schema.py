"""Versioned verification records, distinct from immutable D4 annotations."""

import hashlib
import json
from pathlib import Path

from vla_data.batch.discovery import canonical_episode_id
from vla_data.batch.summary import write_summary
from vla_data.verification.policy import VerificationPolicy, probability

SCHEMA_NAME = "vla_annotation_verification"
SCHEMA_VERSION = 1
HIERARCHICAL_SCHEMA_VERSION = 2
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
    if record.get("schema_version") == HIERARCHICAL_SCHEMA_VERSION:
        return validate_hierarchical_verification(record)
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


def hierarchical_status(record: dict) -> str:
    statuses = [record["episode_task"]["verification_status"]] + [
        segment["verification_status"] for segment in record["semantic_segments"]
    ]
    if not statuses:
        return "INELIGIBLE"
    if any(status == "REJECTED" for status in statuses):
        return "REJECTED"
    if any(status == "NEEDS_HUMAN_REVIEW" for status in statuses):
        return "NEEDS_HUMAN_REVIEW"
    if any(status == "HUMAN_CORRECTED" for status in statuses):
        return "HUMAN_CORRECTED"
    if any(status == "HUMAN_VERIFIED" for status in statuses):
        return "HUMAN_VERIFIED"
    return "AUTO_VERIFIED"


def validate_hierarchical_verification(record: object) -> dict:
    if not isinstance(record, dict):
        raise TypeError("verification must be an object")
    if record.get("schema_name") != SCHEMA_NAME:
        raise ValueError("unsupported verification schema")
    canonical_episode_id(record["episode_id"])
    policy = VerificationPolicy(**record["policy"])
    segments = record.get("semantic_segments")
    if not isinstance(segments, list):
        raise TypeError("semantic_segments must be a list")
    seen: set[str] = set()
    task = record.get("episode_task")
    if not isinstance(task, dict):
        raise TypeError("episode_task verification must be an object")
    task_instruction = task.get("model_instruction")
    if not isinstance(task_instruction, str) or not task_instruction.strip():
        raise ValueError("episode task model instruction must be non-empty")
    task_confidence = probability(
        task.get("model_confidence"), "episode_task.model_confidence"
    )
    task_status = task.get("verification_status")
    if task_status not in STATES - {"INELIGIBLE"}:
        raise ValueError("invalid episode task verification status")
    if task_status in HUMAN_STATES:
        if task.get("verification_source") != "HUMAN_REVIEW":
            raise ValueError("human episode task state requires HUMAN_REVIEW source")
    else:
        expected = (
            "AUTO_VERIFIED"
            if task_confidence >= policy.confidence_threshold
            else "NEEDS_HUMAN_REVIEW"
        )
        if (
            task_status != expected
            or task.get("verification_source") != "MODEL_CONFIDENCE_POLICY"
        ):
            raise ValueError("episode task status disagrees with confidence policy")
    task_review = task.get("human_review")
    if not isinstance(task_review, dict):
        raise TypeError("episode task human_review must be an object")
    task_final = (
        task_instruction if task_status in {"AUTO_VERIFIED", "HUMAN_VERIFIED"} else None
    )
    if task_status == "HUMAN_CORRECTED":
        task_final = task_review.get("corrected_instruction")
        if not isinstance(task_final, str) or not task_final.strip():
            raise ValueError("corrected episode task instruction must be non-empty")
    if task.get("final_instruction") != task_final:
        raise ValueError("episode task final instruction disagrees with review state")
    previous_end = -1
    for segment in segments:
        segment_id = segment.get("semantic_segment_id")
        if not isinstance(segment_id, str) or not segment_id or segment_id in seen:
            raise ValueError("semantic segment IDs must be unique")
        seen.add(segment_id)
        instruction = segment.get("model_instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("model instruction must be non-empty")
        confidence = probability(segment.get("model_confidence"), "model_confidence")
        status = segment.get("verification_status")
        if status not in STATES - {"INELIGIBLE"}:
            raise ValueError("invalid segment verification status")
        model_start, model_end = (
            segment.get("model_start_curated_index"),
            segment.get("model_end_curated_index"),
        )
        final_start, final_end = (
            segment.get("final_start_curated_index"),
            segment.get("final_end_curated_index"),
        )
        if any(
            isinstance(v, bool) or not isinstance(v, int)
            for v in (model_start, model_end, final_start, final_end)
        ):
            raise TypeError("segment boundaries must be integers")
        if not 0 <= final_start < final_end or final_start < previous_end:
            raise ValueError(
                "final segment boundaries must be sorted and non-overlapping"
            )
        if not any(
            domain["start_curated_index"]
            <= final_start
            < final_end
            <= domain["end_curated_index"]
            for domain in record["clean_domains"]
        ):
            raise ValueError("final segment boundaries leave a D2/D3 clean domain")
        previous_end = final_end
        review = segment.get("human_review")
        if not isinstance(review, dict):
            raise TypeError("segment human_review must be an object")
        if status in HUMAN_STATES:
            if segment.get("verification_source") != "HUMAN_REVIEW":
                raise ValueError("human segment state requires HUMAN_REVIEW source")
        else:
            expected = (
                "AUTO_VERIFIED"
                if confidence >= policy.confidence_threshold
                else "NEEDS_HUMAN_REVIEW"
            )
            if (
                status != expected
                or segment.get("verification_source") != "MODEL_CONFIDENCE_POLICY"
            ):
                raise ValueError("segment status disagrees with confidence policy")
            if (final_start, final_end) != (model_start, model_end):
                raise ValueError("automatic verification cannot alter boundaries")
        final = instruction if status in {"AUTO_VERIFIED", "HUMAN_VERIFIED"} else None
        if status == "HUMAN_CORRECTED":
            final = review.get("corrected_instruction") or instruction
        if segment.get("final_instruction") != final:
            raise ValueError("segment final instruction does not match review state")
    if record.get("verification_status") != hierarchical_status(record):
        raise ValueError("aggregate verification status disagrees with segments")
    if record["fingerprint"] != digest(
        {k: v for k, v in record.items() if k != "fingerprint"}
    ):
        raise ValueError("verification fingerprint mismatch")
    if record["input_fingerprint"] != digest(
        [HIERARCHICAL_SCHEMA_VERSION, record["policy"], record["input_identity"]]
    ):
        raise ValueError("verification input fingerprint mismatch")
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
        if record["schema_version"] == HIERARCHICAL_SCHEMA_VERSION:
            return record
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
