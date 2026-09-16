"""Quality-first confidence evaluation; no VLM provider or image decoding."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from zipfile import BadZipFile

from vla_data.annotation.eligibility import evaluate_eligibility
from vla_data.annotation.schema import load_annotation
from vla_data.quality.report import QUALITY_SCHEMA_VERSION
from vla_data.verification.policy import VerificationPolicy
from vla_data.verification.schema import (
    HIERARCHICAL_SCHEMA_VERSION,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    digest,
    hierarchical_status,
    seal,
)


def artifact_identity(path: Path) -> list[str]:
    try:
        content = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        content = type(exc).__name__
    return [str(path.resolve()), content]


def evaluate_verification(
    episode_id: str,
    annotation_path: Path,
    quality_path: Path,
    policy: VerificationPolicy,
) -> dict:
    identity = [
        artifact_identity(path)
        for path in (
            annotation_path,
            quality_path / "quality_report.json",
            quality_path / "quality_mask.npy",
        )
    ]
    record = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "policy": asdict(policy),
        "input_identity": identity,
        "input_fingerprint": digest([SCHEMA_VERSION, asdict(policy), identity]),
        "annotation_path": str(annotation_path.resolve()),
        "quality_path": str(quality_path.resolve()),
        "keyframes_path": str(annotation_path.with_name("keyframes.json").resolve()),
        "annotation_present": annotation_path.is_file(),
        "quality_eligible": False,
        "model_instruction": None,
        "model_confidence": None,
        "verification_status": "INELIGIBLE",
        "verification_source": "QUALITY_OR_INPUT_GATE",
        "human_review": {"reviewer": None, "corrected_instruction": None},
        "final_instruction": None,
        "reason": [],
    }
    try:
        report = json.loads((quality_path / "quality_report.json").read_text())
        if report["schema_version"] != QUALITY_SCHEMA_VERSION:
            raise ValueError("unsupported quality schema version")
        count = report["transition_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("quality transition_count must be a nonnegative integer")
        decision = evaluate_eligibility(
            episode_id,
            transition_count=count,
            quality_report_path=quality_path / "quality_report.json",
            quality_mask_path=quality_path / "quality_mask.npy",
        )
        if report["clean_transition_count"] != decision.clean_transition_count:
            raise ValueError("quality clean count disagrees with mask")
        if not decision.eligible:
            record["reason"] = [decision.quality_outcome, decision.reason]
            return seal(record)
        record["quality_eligible"] = True
    except (OSError, ValueError, TypeError, KeyError, BadZipFile) as exc:
        record["reason"] = ["QUALITY_INVALID_OR_MISSING", str(exc)]
        return seal(record)
    try:
        annotation = load_annotation(annotation_path)
        if annotation["episode_id"] != episode_id:
            raise ValueError("annotation episode_id mismatch")
        if annotation["schema_version"] == 2:
            return _evaluate_hierarchical(
                record, annotation, policy, identity, annotation_path, quality_path
            )
        model = annotation["model_annotation"]
        record["model_instruction"] = model["instruction"]
        record["model_confidence"] = model["confidence"]
        automatic = model["confidence"] >= policy.confidence_threshold
        record["verification_status"] = (
            "AUTO_VERIFIED" if automatic else "NEEDS_HUMAN_REVIEW"
        )
        record["verification_source"] = "MODEL_CONFIDENCE_POLICY"
        record["final_instruction"] = model["instruction"] if automatic else None
    except (OSError, ValueError, TypeError, KeyError) as exc:
        record["reason"] = ["ANNOTATION_INVALID_OR_MISSING", str(exc)]
    return seal(record)


def _evaluate_hierarchical(
    legacy_record: dict,
    annotation: dict,
    policy: VerificationPolicy,
    identity: list,
    annotation_path: Path,
    quality_path: Path,
) -> dict:
    task_source = annotation["episode_task"]
    task_automatic = task_source["confidence"] >= policy.confidence_threshold
    episode_task = {
        "model_instruction": task_source["instruction"],
        "model_confidence": task_source["confidence"],
        "model_paraphrases": list(task_source.get("paraphrases", [])),
        "verification_status": (
            "AUTO_VERIFIED" if task_automatic else "NEEDS_HUMAN_REVIEW"
        ),
        "verification_source": "MODEL_CONFIDENCE_POLICY",
        "human_review": {"reviewer": None, "corrected_instruction": None},
        "final_instruction": task_source["instruction"] if task_automatic else None,
    }
    segments = []
    for source in annotation["semantic_segments"]:
        automatic = source["confidence"] >= policy.confidence_threshold
        status = "AUTO_VERIFIED" if automatic else "NEEDS_HUMAN_REVIEW"
        segments.append(
            {
                "semantic_segment_id": source["segment_id"],
                "model_start_curated_index": source["start_curated_index"],
                "model_end_curated_index": source["end_curated_index"],
                "model_instruction": source["instruction"],
                "model_confidence": source["confidence"],
                "model_paraphrases": list(source.get("paraphrases", [])),
                "verification_status": status,
                "verification_source": "MODEL_CONFIDENCE_POLICY",
                "human_review": {
                    "reviewer": None,
                    "corrected_instruction": None,
                    "corrected_start_curated_index": None,
                    "corrected_end_curated_index": None,
                },
                "final_start_curated_index": source["start_curated_index"],
                "final_end_curated_index": source["end_curated_index"],
                "final_instruction": source["instruction"] if automatic else None,
            }
        )
    record = {
        "schema_name": SCHEMA_NAME,
        "schema_version": HIERARCHICAL_SCHEMA_VERSION,
        "episode_id": legacy_record["episode_id"],
        "policy": asdict(policy),
        "input_identity": identity,
        "input_fingerprint": digest(
            [HIERARCHICAL_SCHEMA_VERSION, asdict(policy), identity]
        ),
        "annotation_path": str(annotation_path.resolve()),
        "quality_path": str(quality_path.resolve()),
        "keyframes_path": str(annotation_path.with_name("keyframes.json").resolve()),
        "annotation_present": True,
        "quality_eligible": True,
        "episode_task": episode_task,
        "clean_domains": annotation["clean_domains"],
        "semantic_segments": segments,
        "non_training_intervals": annotation["non_training_intervals"],
        "reason": [],
    }
    record["verification_status"] = hierarchical_status(record)
    return seal(record)
