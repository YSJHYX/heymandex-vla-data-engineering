"""Annotation eligibility gate over D3 quality artifacts."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vla_data.annotation.eligibility import (
    EligibilityError,
    evaluate_eligibility,
)


def _write_quality(
    tmp_path, outcome: str, mask: np.ndarray, *, episode_id="episode_000001"
):
    episode_dir = tmp_path / "quality" / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / "quality_report.json").write_text(
        json.dumps(
            {
                "schema_name": "vla_quality_report",
                "schema_version": 1,
                "episode_id": episode_id,
                "status": outcome,
            }
        )
    )
    np.save(episode_dir / "quality_mask.npy", mask)
    return episode_dir / "quality_report.json", episode_dir / "quality_mask.npy"


@pytest.mark.parametrize("outcome", ["ACCEPT", "ACCEPT_WITH_WARNING"])
def test_eligible_outcomes_with_clean_transitions(tmp_path, outcome) -> None:
    report, mask = _write_quality(tmp_path, outcome, np.array([True, False, True]))
    decision = evaluate_eligibility(
        "episode_000001",
        transition_count=3,
        quality_report_path=report,
        quality_mask_path=mask,
    )
    assert decision.eligible
    assert decision.quality_outcome == outcome
    assert decision.clean_transition_count == 2


@pytest.mark.parametrize(
    "outcome",
    ["EXCLUDE_FROM_EXPERT_TRAINING", "REJECT"],
)
def test_ineligible_outcomes_never_annotate(tmp_path, outcome) -> None:
    report, mask = _write_quality(tmp_path, outcome, np.array([True, True]))
    decision = evaluate_eligibility(
        "episode_000001",
        transition_count=2,
        quality_report_path=report,
        quality_mask_path=mask,
    )
    assert not decision.eligible
    assert outcome in decision.reason


def test_eligible_outcome_without_clean_transitions_is_ineligible(tmp_path) -> None:
    report, mask = _write_quality(tmp_path, "ACCEPT", np.array([False, False]))
    decision = evaluate_eligibility(
        "episode_000001",
        transition_count=2,
        quality_report_path=report,
        quality_mask_path=mask,
    )
    assert not decision.eligible
    assert "no clean transitions" in decision.reason


def test_mask_length_mismatch_is_an_error(tmp_path) -> None:
    report, mask = _write_quality(tmp_path, "ACCEPT", np.array([True]))
    with pytest.raises(EligibilityError, match="does not match"):
        evaluate_eligibility(
            "episode_000001",
            transition_count=4,
            quality_report_path=report,
            quality_mask_path=mask,
        )


def test_unknown_quality_schema_is_an_error(tmp_path) -> None:
    report, mask = _write_quality(tmp_path, "ACCEPT", np.array([True]))
    document = json.loads(report.read_text())
    document["schema_name"] = "something_else"
    report.write_text(json.dumps(document))
    with pytest.raises(EligibilityError, match="unexpected quality schema"):
        evaluate_eligibility(
            "episode_000001",
            transition_count=1,
            quality_report_path=report,
            quality_mask_path=mask,
        )


def test_missing_quality_report_is_an_error(tmp_path) -> None:
    with pytest.raises(EligibilityError, match="unreadable quality report"):
        evaluate_eligibility(
            "episode_000001",
            transition_count=1,
            quality_report_path=tmp_path / "missing.json",
            quality_mask_path=tmp_path / "missing.npy",
        )
