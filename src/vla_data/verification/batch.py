"""Dataset verification and resume without annotation mutation or provider calls."""

from dataclasses import dataclass
from pathlib import Path

from vla_data.batch.discovery import canonical_episode_id, discover_curated_episodes
from vla_data.verification.evaluator import evaluate_verification
from vla_data.verification.policy import AutoVerificationAuditConfig, VerificationPolicy
from vla_data.verification.schema import (
    HIERARCHICAL_SCHEMA_VERSION,
    HUMAN_STATES,
    hierarchical_status,
    load_verification,
    seal,
    write_verification,
)
from vla_data.verification.summary import publish_index, summarize


@dataclass(frozen=True)
class VerificationBatchResult:
    records: tuple[dict, ...]
    summary: dict
    processed: int
    skipped: int


def verify_annotations(
    annotation_root: str | Path,
    quality_root: str | Path,
    output_root: str | Path,
    *,
    policy: VerificationPolicy,
    episode: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    audit: AutoVerificationAuditConfig | None = None,
) -> VerificationBatchResult:
    annotation, quality, output = (
        Path(p).resolve() for p in (annotation_root, quality_root, output_root)
    )
    if any(
        output == root or output in root.parents or root in output.parents
        for root in (annotation, quality)
    ):
        raise ValueError("verification output must be separate from inputs")
    roots = [annotation, quality] + ([output] if output.is_dir() else [])
    discovered = [discover_curated_episodes(root) for root in roots]
    if any(d.issues for d in discovered):
        raise ValueError("noncanonical episode directory")
    ids = sorted({item.episode_id for d in discovered for item in d.episodes})
    selected = canonical_episode_id(episode) if episode is not None else None
    if selected is not None and selected not in ids:
        raise ValueError(f"selected episode not found: {selected}")
    audit = audit or AutoVerificationAuditConfig()
    records, pending = [], []
    skipped = 0
    for eid in ids:
        path = output / eid / "verification.json"
        if selected is not None and eid != selected:
            if path.is_file():
                prior = load_verification(path)
                if prior["policy"] != {
                    "confidence_threshold": policy.confidence_threshold,
                    "policy_version": policy.policy_version,
                }:
                    raise ValueError(
                        "mixed policies in output root; rerun whole dataset"
                    )
                records.append(prior)
            continue
        candidate = evaluate_verification(
            eid, annotation / eid / "annotation.json", quality / eid, policy
        )
        try:
            prior = load_verification(path, check_current=False)
        except (OSError, ValueError, TypeError, KeyError):
            prior = None
        # A human decision survives force/threshold changes on identical source
        # artifacts. Source changes invalidate it; D4 is never re-annotated here.
        if (
            prior is not None
            and prior.get("schema_version") == HIERARCHICAL_SCHEMA_VERSION
            and candidate.get("schema_version") == HIERARCHICAL_SCHEMA_VERSION
            and prior["input_identity"] == candidate["input_identity"]
            and not candidate["reason"]
        ):
            previous_task = prior["episode_task"]
            current_task = candidate["episode_task"]
            if (
                previous_task["verification_status"] in HUMAN_STATES
                and previous_task["model_instruction"]
                == current_task["model_instruction"]
                and previous_task["model_confidence"]
                == current_task["model_confidence"]
            ):
                for key in (
                    "verification_status",
                    "verification_source",
                    "human_review",
                    "final_instruction",
                ):
                    current_task[key] = previous_task[key]
            previous = {
                segment["semantic_segment_id"]: segment
                for segment in prior["semantic_segments"]
            }
            for segment in candidate["semantic_segments"]:
                old = previous.get(segment["semantic_segment_id"])
                if old is None or old["verification_status"] not in HUMAN_STATES:
                    continue
                immutable = (
                    "model_start_curated_index",
                    "model_end_curated_index",
                    "model_instruction",
                    "model_confidence",
                )
                if any(old[key] != segment[key] for key in immutable):
                    continue
                for key in (
                    "verification_status",
                    "verification_source",
                    "human_review",
                    "final_start_curated_index",
                    "final_end_curated_index",
                    "final_instruction",
                ):
                    segment[key] = old[key]
            candidate["verification_status"] = hierarchical_status(candidate)
            seal(candidate)
        elif (
            prior is not None
            and prior["verification_status"] in HUMAN_STATES
            and prior["input_identity"] == candidate["input_identity"]
            and not candidate["reason"]
        ):
            for key in (
                "verification_status",
                "verification_source",
                "human_review",
                "final_instruction",
            ):
                candidate[key] = prior[key]
            seal(candidate)
        if not force and prior == candidate:
            skipped += 1
        else:
            pending.append((path, candidate))
        records.append(candidate)
    if not dry_run:
        # Resolve all decisions before publication. A source change invalidates
        # the batch instead of allowing a partially stale confidence decision.
        for record in records:
            current = evaluate_verification(
                record["episode_id"],
                Path(record["annotation_path"]),
                Path(record["quality_path"]),
                policy,
            )
            if current["input_fingerprint"] != record["input_fingerprint"]:
                raise ValueError("verification inputs changed during run; retry")
        for path, record in pending:
            write_verification(path, record)
        summary = publish_index(output, records, policy, audit)
    else:
        summary = summarize(records, policy, audit)
    return VerificationBatchResult(tuple(records), summary, len(pending), skipped)
