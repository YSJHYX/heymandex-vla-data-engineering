"""Human QC sampling and review-book helpers (model-first, non-destructive)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from vla_data.annotation.schema import (
    apply_review,
    load_annotation,
    write_annotation,
)


@dataclass(frozen=True)
class QCSamplingConfig:
    """Deterministic sampling plan for human review; thresholds not locked yet."""

    review_below_confidence: float | None = None
    review_quality_warnings: bool = False
    random_review_fraction: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.review_below_confidence is not None and not (
            0.0 <= self.review_below_confidence <= 1.0
        ):
            raise ValueError("review_below_confidence must be within [0, 1]")
        if not 0.0 <= self.random_review_fraction <= 1.0:
            raise ValueError("random_review_fraction must be within [0, 1]")


@dataclass(frozen=True)
class QCRecord:
    episode_id: str
    annotation_path: Path
    confidence: float
    quality_outcome: str
    selected_reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "annotation_path": str(self.annotation_path),
            "confidence": self.confidence,
            "quality_outcome": self.quality_outcome,
            "selected_reason": self.selected_reason,
        }


@dataclass(frozen=True)
class QCReviewPlan:
    config: QCSamplingConfig
    selected: tuple[QCRecord, ...] = field(default=())

    def as_dict(self) -> dict[str, object]:
        return {
            "sampling": {
                "review_below_confidence": self.config.review_below_confidence,
                "review_quality_warnings": self.config.review_quality_warnings,
                "random_review_fraction": self.config.random_review_fraction,
                "seed": self.config.seed,
            },
            "selected": [record.as_dict() for record in self.selected],
        }


def build_review_plan(
    annotation_records: tuple[dict[str, object], ...],
    config: QCSamplingConfig,
) -> QCReviewPlan:
    """Select episodes for human review deterministically.

    Each record needs: episode_id, confidence, quality_outcome. Selection is
    stable for a fixed seed and independent of input ordering.
    """

    ordered = sorted(annotation_records, key=lambda record: str(record["episode_id"]))
    reasons: dict[str, str] = {}
    for record in ordered:
        episode_id = str(record["episode_id"])
        confidence = float(record["confidence"])
        outcome = str(record["quality_outcome"])
        if (
            config.review_below_confidence is not None
            and confidence < config.review_below_confidence
        ):
            reasons[episode_id] = "confidence_below_threshold"
        elif config.review_quality_warnings and outcome == "ACCEPT_WITH_WARNING":
            reasons[episode_id] = "quality_warning"
    if config.random_review_fraction > 0:
        population = [
            record for record in ordered if str(record["episode_id"]) not in reasons
        ]
        count = round(len(population) * config.random_review_fraction)
        if count:
            generator = random.Random(config.seed)
            for record in generator.sample(population, count):
                reasons[str(record["episode_id"])] = "random_sample"
    return QCReviewPlan(
        config=config,
        selected=tuple(
            QCRecord(
                episode_id=str(record["episode_id"]),
                annotation_path=Path(str(record.get("annotation_path", ""))),
                confidence=float(record["confidence"]),
                quality_outcome=str(record["quality_outcome"]),
                selected_reason=reasons[str(record["episode_id"])],
            )
            for record in ordered
            if str(record["episode_id"]) in reasons
        ),
    )


def review_annotation(
    annotation_path: str | Path,
    *,
    status: str,
    reviewer: str | None = None,
    corrected_instruction: str | None = None,
    out_path: str | Path | None = None,
) -> dict[str, object]:
    """Apply one review transition; the model block is preserved verbatim."""

    annotation = load_annotation(annotation_path)
    updated = apply_review(
        annotation,
        status=status,
        reviewer=reviewer,
        corrected_instruction=corrected_instruction,
    )
    write_annotation(out_path or annotation_path, updated)
    return updated
