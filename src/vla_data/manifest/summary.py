"""Compact dataset counts and transparent split limitations."""

from collections import Counter

from vla_data.manifest.schema import (
    SCHEMA_VERSION,
    SPLIT_LIMITATION,
    SUMMARY_SCHEMA_NAME,
)


def summarize(records: list[dict]) -> dict:
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "val"]
    sources = {r.get("source_episode_id", r["episode_id"]) for r in records}
    return {
        "schema_name": SUMMARY_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "episodes_discovered": len(sources),
        "training_eligible": len(train) + len(val),
        "training_units": len(train) + len(val),
        "needs_review": sum(r["eligibility"] == "NEEDS_REVIEW" for r in records),
        "excluded": sum(r["eligibility"] == "EXCLUDED" for r in records),
        "train_episodes": len(
            {r.get("source_episode_id", r["episode_id"]) for r in train}
        ),
        "validation_episodes": len(
            {r.get("source_episode_id", r["episode_id"]) for r in val}
        ),
        "train_transitions": sum(r["transition_count_selected"] for r in train),
        "validation_transitions": sum(r["transition_count_selected"] for r in val),
        "annotation_status_counts": dict(
            sorted(
                Counter(
                    r["annotation_status"] or "MISSING_OR_INVALID" for r in records
                ).items()
            )
        ),
        "verification_status_counts": dict(
            sorted(
                Counter(
                    r["verification_status"] or "MISSING_OR_INVALID" for r in records
                ).items()
            )
        ),
        "quality_status_counts": dict(
            sorted(
                Counter(
                    r["quality_outcome"] or "MISSING_OR_INVALID" for r in records
                ).items()
            )
        ),
        "split_limitation": SPLIT_LIMITATION,
    }
