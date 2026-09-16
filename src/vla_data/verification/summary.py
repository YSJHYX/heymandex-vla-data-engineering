"""Confidence distributions and analysis-only review workload projections."""

import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np

from vla_data.batch.summary import write_summary
from vla_data.verification.policy import AutoVerificationAuditConfig, VerificationPolicy


def summarize(
    records: list[dict], policy: VerificationPolicy, audit: AutoVerificationAuditConfig
) -> dict:
    units = [
        segment
        for record in records
        for segment in (
            record["semantic_segments"]
            if record.get("schema_version") == 2
            else [record]
        )
    ]
    confidence = [
        u["model_confidence"] for u in units if u.get("model_confidence") is not None
    ]
    percentiles = (0, 10, 25, 50, 75, 90, 95, 100)
    names = ("min", "p10", "p25", "median", "p75", "p90", "p95", "max")
    stats = (
        dict(
            zip(names, map(float, np.percentile(confidence, percentiles)), strict=True)
        )
        if confidence
        else dict.fromkeys(names)
    )
    stats["mean"] = float(np.mean(confidence)) if confidence else None
    projected = [
        {
            "threshold": threshold,
            "count": sum(c < threshold for c in confidence),
            "fraction": sum(c < threshold for c in confidence) / len(confidence)
            if confidence
            else None,
        }
        for threshold in (0.70, 0.80, 0.90, 0.95)
    ]
    summary = {
        "schema_name": "vla_dataset_verification_summary",
        "schema_version": 1,
        "episodes_discovered": len(records),
        "annotation_count": sum(r["annotation_present"] for r in records),
        "quality_eligible_count": sum(r["quality_eligible"] for r in records),
        "threshold": policy.confidence_threshold,
        "policy_version": policy.policy_version,
        "confidence_statistics": stats,
        "confidence_population_count": len(confidence),
        "confidence_population": "valid annotations after D3 quality eligibility",
        "projected_review_workload": projected,
        "projection_scope": "ANALYSIS_ONLY; confidence below threshold, before human review; not a production threshold recommendation",
        "audit_config": asdict(audit),
    }
    for status in (
        "AUTO_VERIFIED",
        "NEEDS_HUMAN_REVIEW",
        "HUMAN_VERIFIED",
        "HUMAN_CORRECTED",
        "REJECTED",
        "INELIGIBLE",
    ):
        summary[f"{status.lower()}_count"] = sum(
            unit["verification_status"] == status for unit in units
        )
        summary[f"episode_task_{status.lower()}_count"] = sum(
            record.get("schema_version") == 2
            and record["episode_task"]["verification_status"] == status
            for record in records
        )
    auto = sorted(
        [
            r["episode_id"]
            for r in records
            if r["verification_status"] == "AUTO_VERIFIED"
        ],
        key=lambda value: hashlib.sha256(f"{audit.seed}:{value}".encode()).digest(),
    )
    summary["optional_auto_audit_episode_ids"] = sorted(
        auto[: int(len(auto) * audit.random_audit_fraction)]
    )
    summary["episode_fingerprints"] = {
        r["episode_id"]: r["fingerprint"] for r in records
    }
    return summary


def publish_index(
    root: Path,
    records: list[dict],
    policy: VerificationPolicy,
    audit: AutoVerificationAuditConfig,
) -> dict:
    import json
    import os
    import tempfile

    queue = []
    for record in records:
        if record.get("schema_version") == 2:
            task = record["episode_task"]
            if task["verification_status"] == "NEEDS_HUMAN_REVIEW":
                queue.append(
                    {
                        "episode_id": record["episode_id"],
                        "review_target": "episode_task",
                        "semantic_segment_id": None,
                        "model_instruction": task["model_instruction"],
                        "confidence": task["model_confidence"],
                        "annotation_path": record["annotation_path"],
                        "keyframes_path": record["keyframes_path"],
                    }
                )
            for segment in record["semantic_segments"]:
                if segment["verification_status"] == "NEEDS_HUMAN_REVIEW":
                    queue.append(
                        {
                            "episode_id": record["episode_id"],
                            "review_target": "semantic_segment",
                            "semantic_segment_id": segment["semantic_segment_id"],
                            "model_instruction": segment["model_instruction"],
                            "confidence": segment["model_confidence"],
                            "start_curated_index": segment["model_start_curated_index"],
                            "end_curated_index": segment["model_end_curated_index"],
                            "annotation_path": record["annotation_path"],
                            "keyframes_path": record["keyframes_path"],
                        }
                    )
        elif record["verification_status"] == "NEEDS_HUMAN_REVIEW":
            queue.append(
                {
                    "episode_id": record["episode_id"],
                    "model_instruction": record["model_instruction"],
                    "confidence": record["model_confidence"],
                    "annotation_path": record["annotation_path"],
                    "keyframes_path": record["keyframes_path"],
                }
            )
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=root, delete=False, encoding="utf-8"
    ) as stream:
        temporary = Path(stream.name)
        for row in queue:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    try:
        os.replace(temporary, root / "human_review_queue.jsonl")
    finally:
        temporary.unlink(missing_ok=True)
    summary = summarize(records, policy, audit)
    write_summary(root / "dataset_verification_summary.json", summary)
    return summary
