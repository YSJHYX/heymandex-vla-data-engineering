"""Consume frozen episode validators and D3/D4 artifacts without changing them."""

import json
from pathlib import Path
from zipfile import BadZipFile

from vla_data.annotation.eligibility import evaluate_eligibility
from vla_data.annotation.schema import load_annotation
from vla_data.io.curated_episode import CuratedEpisode
from vla_data.manifest.schema import SCHEMA_NAME, SCHEMA_VERSION
from vla_data.quality.report import QUALITY_SCHEMA_VERSION
from vla_data.validation.curated_v1 import validate_curated_episode
from vla_data.verification.schema import APPROVED, load_verification


def inspect_episode(
    episode_id: str, curated: Path, quality: Path, annotation: Path, verification: Path
) -> dict:
    record = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "curated_path": str(curated.resolve()),
        "quality_path": str(quality.resolve()),
        "annotation_path": str(annotation.resolve()),
        "verification_path": str(verification.resolve()),
        "verification_status": None,
        "verification_policy": None,
        "training_transition_mask_path": str((quality / "quality_mask.npy").resolve()),
        "quality_outcome": None,
        "annotation_status": None,
        "final_instruction": None,
        "transition_count_total": 0,
        "transition_count_selected": 0,
        "split": None,
        "task_type": None,
        "session_id": None,
        "collection_date": None,
        "collection_group": None,
    }
    reasons, details = [], []
    loaded = None
    try:
        loaded = CuratedEpisode.load(curated)
        record["transition_count_total"] = loaded.transition_count
        if loaded.metadata.get("episode_id") != episode_id:
            reasons.append("CURATED_EPISODE_ID_MISMATCH")
        if (
            loaded.metadata.get("expert_training_status")
            == "EXCLUDE_FROM_EXPERT_TRAINING"
        ):
            reasons.append("EXCLUDE_FROM_EXPERT_TRAINING")
        for key in ("session_id", "collection_date", "collection_group"):
            record[key] = loaded.metadata.get(key)
        if any(
            array.ndim != 2 or array.shape[-1] != 17
            for array in (loaded.state, loaded.action)
        ):
            reasons.append("CURATED_DIMENSION_NOT_17")
        validation = validate_curated_episode(loaded)
        if not validation.passed:
            reasons.append("CURATED_VALIDATION_FAILED")
            details.extend(validation.errors)
    except (OSError, ValueError, TypeError, KeyError, IndexError, BadZipFile) as exc:
        reasons.append("CURATED_INVALID_OR_MISSING")
        details.append(str(exc))
    try:
        report = json.loads((quality / "quality_report.json").read_text())
        if not isinstance(report, dict):
            raise TypeError("quality report must be an object")
        if not isinstance(report.get("status"), str):
            raise TypeError("quality status must be a string")
        record["quality_outcome"] = report.get("status")
        if report.get("status") == "EXCLUDE_FROM_EXPERT_TRAINING":
            reasons.append("EXCLUDE_FROM_EXPERT_TRAINING")
        if report.get("schema_version") != QUALITY_SCHEMA_VERSION:
            raise ValueError("unsupported quality schema version")
        if loaded is not None:
            decision = evaluate_eligibility(
                episode_id,
                transition_count=loaded.transition_count,
                quality_report_path=quality / "quality_report.json",
                quality_mask_path=quality / "quality_mask.npy",
            )
            record["transition_count_selected"] = decision.clean_transition_count
            if (
                report.get("transition_count") != loaded.transition_count
                or report.get("clean_transition_count")
                != decision.clean_transition_count
            ):
                raise ValueError("quality report counts disagree with Curated/mask")
            if decision.quality_outcome not in {"ACCEPT", "ACCEPT_WITH_WARNING"}:
                reasons.append(f"QUALITY_{decision.quality_outcome}")
            if not decision.clean_transition_count:
                reasons.append("NO_CLEAN_TRANSITIONS")
    except (OSError, ValueError, TypeError, KeyError, BadZipFile) as exc:
        reasons.append("QUALITY_INVALID_OR_MISSING")
        details.append(str(exc))
    try:
        document = load_annotation(annotation)
        if document["episode_id"] != episode_id:
            raise ValueError("annotation episode_id mismatch")
        status = document["review"]["status"]
        record["annotation_status"] = status
        record["task_type"] = document["model_annotation"].get("task_type")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        reasons.append("ANNOTATION_INVALID_OR_MISSING")
        details.append(str(exc))
    try:
        verified = load_verification(verification)
        if (
            verified["episode_id"] != episode_id
            or Path(verified["annotation_path"]).resolve() != annotation.resolve()
            or Path(verified["quality_path"]).resolve() != quality.resolve()
        ):
            raise ValueError(
                "verification refers to different episode/source artifacts"
            )
        record["verification_status"] = verified["verification_status"]
        record["verification_policy"] = verified["policy"]
        record["final_instruction"] = verified["final_instruction"]
        if verified["verification_status"] not in APPROVED:
            reasons.append(verified["verification_status"])
    except (OSError, ValueError, TypeError, KeyError) as exc:
        reasons.append("VERIFICATION_INVALID_OR_MISSING")
        details.append(str(exc))
    record["reason"] = sorted(set(reasons))
    record["details"] = details
    record["eligibility"] = (
        "TRAINING_ELIGIBLE"
        if not reasons
        else "NEEDS_REVIEW"
        if set(reasons) == {"NEEDS_HUMAN_REVIEW"}
        else "EXCLUDED"
    )
    return record
