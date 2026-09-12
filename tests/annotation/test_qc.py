"""QC sampling plans and non-destructive review application."""

from __future__ import annotations

import pytest

from vla_data.annotation.pipeline import ANNOTATION_FILENAME
from vla_data.annotation.qc import (
    QCSamplingConfig,
    build_review_plan,
    review_annotation,
)
from vla_data.annotation.schema import (
    REVIEW_HUMAN_CORRECTED,
    REVIEW_HUMAN_VERIFIED,
    build_annotation,
    validate_model_payload,
    write_annotation,
)


def _records():
    return [
        {
            "episode_id": "episode_000002",
            "confidence": 0.99,
            "quality_outcome": "ACCEPT",
        },
        {
            "episode_id": "episode_000000",
            "confidence": 0.42,
            "quality_outcome": "ACCEPT",
        },
        {
            "episode_id": "episode_000001",
            "confidence": 0.85,
            "quality_outcome": "ACCEPT_WITH_WARNING",
        },
    ]


def test_below_confidence_sampling_is_deterministic() -> None:
    config = QCSamplingConfig(review_below_confidence=0.6)
    first = build_review_plan(_records(), config)
    second = build_review_plan(list(reversed(_records())), config)
    assert [r.episode_id for r in first.selected] == ["episode_000000"]
    assert [r.episode_id for r in second.selected] == ["episode_000000"]


def test_quality_warning_sampling() -> None:
    plan = build_review_plan(_records(), QCSamplingConfig(review_quality_warnings=True))
    assert [r.episode_id for r in plan.selected] == ["episode_000001"]
    assert plan.selected[0].selected_reason == "quality_warning"


def test_random_sampling_is_seed_stable() -> None:
    config = QCSamplingConfig(random_review_fraction=1.0 / 3, seed=7)
    first = build_review_plan(_records(), config)
    second = build_review_plan(_records(), config)
    assert [r.episode_id for r in first.selected] == [
        r.episode_id for r in second.selected
    ]
    assert all(r.selected_reason == "random_sample" for r in first.selected)


def test_sampling_config_validates_ranges() -> None:
    with pytest.raises(ValueError, match="review_below_confidence"):
        QCSamplingConfig(review_below_confidence=1.5)
    with pytest.raises(ValueError, match="random_review_fraction"):
        QCSamplingConfig(random_review_fraction=-0.1)


@pytest.fixture
def annotation_path(tmp_path):
    annotation = build_annotation(
        episode_id="episode_000001",
        model_annotation={
            "provider": "p",
            "model": "m",
            "prompt_version": "task_instruction_v1",
            **validate_model_payload(
                {
                    "instruction": "Pick up the block.",
                    "confidence": 0.8,
                    "task_type": None,
                    "objects": ["block"],
                    "uncertainty": None,
                }
            ),
            "request_id": None,
            "attempt_count": 1,
        },
        quality_outcome="ACCEPT",
    )
    path = tmp_path / "annotation.json"
    write_annotation(path, annotation)
    return path


def test_review_annotation_verifies_and_resolves(annotation_path, tmp_path) -> None:
    reviewed = review_annotation(
        annotation_path, status=REVIEW_HUMAN_VERIFIED, reviewer="alice"
    )
    assert reviewed["review"]["status"] == REVIEW_HUMAN_VERIFIED
    assert reviewed["final_instruction"] == "Pick up the block."
    # The model block is untouched.
    assert reviewed["model_annotation"]["instruction"] == "Pick up the block."


def test_review_annotation_corrects_without_overwriting_model(
    annotation_path, tmp_path
) -> None:
    copy = tmp_path / "copy" / ANNOTATION_FILENAME
    copy.parent.mkdir()
    copy.write_bytes(annotation_path.read_bytes())
    reviewed = review_annotation(
        copy,
        status=REVIEW_HUMAN_CORRECTED,
        reviewer="bob",
        corrected_instruction="Pick up the red block.",
    )
    assert reviewed["final_instruction"] == "Pick up the red block."
    assert reviewed["model_annotation"]["instruction"] == "Pick up the block."
    # Original artifact on disk is preserved when reviewing a copy.
    import json

    assert json.loads(annotation_path.read_text())["review"]["status"] == "AUTO_LABELED"
