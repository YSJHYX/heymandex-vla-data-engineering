"""Annotation eligibility gate backed by the frozen D3 quality artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vla_data.quality.report import QUALITY_SCHEMA_NAME, QUALITY_STATUSES

ELIGIBLE_OUTCOMES = frozenset({"ACCEPT", "ACCEPT_WITH_WARNING"})
INELIGIBLE_OUTCOMES = frozenset({"EXCLUDE_FROM_EXPERT_TRAINING", "REJECT", "FAILED"})


class EligibilityError(ValueError):
    """The quality artifacts are unreadable or internally inconsistent."""


@dataclass(frozen=True)
class EligibilityDecision:
    episode_id: str
    eligible: bool
    quality_outcome: str | None
    clean_transition_count: int
    transition_count: int
    reason: str | None


def evaluate_eligibility(
    episode_id: str,
    *,
    transition_count: int,
    quality_report_path: str | Path,
    quality_mask_path: str | Path,
) -> EligibilityDecision:
    """Decide annotation eligibility without touching any provider."""

    report_path = Path(quality_report_path)
    mask_path = Path(quality_mask_path)
    try:
        with report_path.open(encoding="utf-8") as stream:
            report = json.load(stream)
    except (OSError, ValueError) as exc:
        raise EligibilityError(
            f"{episode_id}: unreadable quality report {report_path}: {exc}"
        ) from exc

    if not isinstance(report, dict):
        raise EligibilityError(f"{episode_id}: quality report must be an object")
    if report.get("schema_name") != QUALITY_SCHEMA_NAME:
        raise EligibilityError(
            f"{episode_id}: unexpected quality schema {report.get('schema_name')!r}"
        )
    outcome = str(report.get("status"))
    if outcome not in QUALITY_STATUSES:
        raise EligibilityError(f"{episode_id}: unknown quality status {outcome!r}")
    if str(report.get("episode_id", episode_id)) != episode_id:
        raise EligibilityError(f"{episode_id}: quality report episode_id mismatch")

    try:
        mask = np.load(mask_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise EligibilityError(
            f"{episode_id}: unreadable quality mask {mask_path}: {exc}"
        ) from exc
    if mask.dtype != np.dtype(bool) or mask.ndim != 1:
        raise EligibilityError(f"{episode_id}: quality mask must be a 1-D bool array")
    if mask.shape[0] != transition_count:
        raise EligibilityError(
            f"{episode_id}: quality mask length {mask.shape[0]} does not match "
            f"transition count {transition_count}"
        )

    clean = int(np.count_nonzero(mask))
    if outcome in INELIGIBLE_OUTCOMES or outcome not in ELIGIBLE_OUTCOMES:
        return EligibilityDecision(
            episode_id,
            False,
            outcome,
            clean,
            transition_count,
            f"quality outcome {outcome} is not annotation-eligible",
        )
    if clean == 0:
        return EligibilityDecision(
            episode_id,
            False,
            outcome,
            clean,
            transition_count,
            "quality mask contains no clean transitions",
        )
    return EligibilityDecision(episode_id, True, outcome, clean, transition_count, None)
