"""Explicit human review of verification artifacts; D4 remains immutable."""

import json
from pathlib import Path

from vla_data.batch.discovery import canonical_episode_id
from vla_data.verification.policy import AutoVerificationAuditConfig, VerificationPolicy
from vla_data.verification.schema import (
    HIERARCHICAL_SCHEMA_VERSION,
    hierarchical_status,
    load_verification,
    seal,
    write_verification,
)
from vla_data.verification.summary import publish_index

TRANSITIONS = {
    "NEEDS_HUMAN_REVIEW": {"HUMAN_VERIFIED", "HUMAN_CORRECTED", "REJECTED"},
    "AUTO_VERIFIED": {"HUMAN_VERIFIED", "HUMAN_CORRECTED", "REJECTED"},
    "HUMAN_VERIFIED": {"HUMAN_CORRECTED", "REJECTED"},
    "HUMAN_CORRECTED": {"HUMAN_VERIFIED", "REJECTED"},
    "REJECTED": {"HUMAN_VERIFIED", "HUMAN_CORRECTED"},
    "INELIGIBLE": set(),
}


def review_episode(
    verification_root: str | Path,
    episode: str,
    *,
    status: str,
    instruction: str | None = None,
    reviewer: str | None = None,
    semantic_segment_id: str | None = None,
    start_curated_index: int | None = None,
    end_curated_index: int | None = None,
) -> dict:
    root = Path(verification_root).resolve()
    path = root / canonical_episode_id(episode) / "verification.json"
    record = load_verification(path)
    if record["schema_version"] == HIERARCHICAL_SCHEMA_VERSION:
        return _review_segment(
            root,
            record,
            status=status,
            instruction=instruction,
            reviewer=reviewer,
            semantic_segment_id=semantic_segment_id,
            start_curated_index=start_curated_index,
            end_curated_index=end_curated_index,
        )
    if (
        semantic_segment_id is not None
        or start_curated_index is not None
        or end_curated_index is not None
    ):
        raise ValueError("segment/boundary flags require annotation schema v2")
    if status not in TRANSITIONS[record["verification_status"]]:
        raise ValueError(
            f"invalid review transition: {record['verification_status']} -> {status}"
        )
    if status == "HUMAN_CORRECTED":
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("HUMAN_CORRECTED requires --instruction")
    elif instruction is not None:
        raise ValueError("--instruction only applies to HUMAN_CORRECTED")
    # Validate the whole index before changing anything, keeping queue/summary
    # consistent with the same root-wide policy.
    records = [
        load_verification(p) for p in sorted(root.glob("episode_*/verification.json"))
    ]
    if any(r["policy"] != record["policy"] for r in records):
        raise ValueError("mixed policies; rerun verify-annotations before review")
    summary = json.loads((root / "dataset_verification_summary.json").read_text())
    record["verification_status"] = status
    record["verification_source"] = "HUMAN_REVIEW"
    record["human_review"] = {
        "reviewer": reviewer,
        "corrected_instruction": instruction if status == "HUMAN_CORRECTED" else None,
    }
    record["final_instruction"] = (
        instruction
        if status == "HUMAN_CORRECTED"
        else record["model_instruction"]
        if status == "HUMAN_VERIFIED"
        else None
    )
    seal(record)
    write_verification(path, record)
    records = [record if r["episode_id"] == episode else r for r in records]
    publish_index(
        root,
        records,
        VerificationPolicy(**record["policy"]),
        AutoVerificationAuditConfig(**summary["audit_config"]),
    )
    return record


def _review_segment(
    root: Path,
    record: dict,
    *,
    status: str,
    instruction: str | None,
    reviewer: str | None,
    semantic_segment_id: str | None,
    start_curated_index: int | None,
    end_curated_index: int | None,
) -> dict:
    if semantic_segment_id is None:
        if start_curated_index is not None or end_curated_index is not None:
            raise ValueError("boundary correction requires --semantic-segment-id")
        task = record["episode_task"]
        current = task["verification_status"]
        if status not in TRANSITIONS[current]:
            raise ValueError(f"invalid review transition: {current} -> {status}")
        if status == "HUMAN_CORRECTED":
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("HUMAN_CORRECTED episode task requires --instruction")
        elif instruction is not None:
            raise ValueError("--instruction only applies to HUMAN_CORRECTED")
        task["verification_status"] = status
        task["verification_source"] = "HUMAN_REVIEW"
        task["human_review"] = {
            "reviewer": reviewer,
            "corrected_instruction": instruction,
        }
        task["final_instruction"] = (
            None if status == "REJECTED" else instruction or task["model_instruction"]
        )
        record["verification_status"] = hierarchical_status(record)
        seal(record)
        path = root / record["episode_id"] / "verification.json"
        write_verification(path, record)
        records = [
            load_verification(item)
            for item in sorted(root.glob("episode_*/verification.json"))
        ]
        summary = json.loads((root / "dataset_verification_summary.json").read_text())
        publish_index(
            root,
            records,
            VerificationPolicy(**record["policy"]),
            AutoVerificationAuditConfig(**summary["audit_config"]),
        )
        return record
    matches = [
        segment
        for segment in record["semantic_segments"]
        if segment["semantic_segment_id"] == semantic_segment_id
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown semantic segment: {semantic_segment_id}")
    segment = matches[0]
    current = segment["verification_status"]
    if status not in TRANSITIONS[current]:
        raise ValueError(f"invalid review transition: {current} -> {status}")
    boundary_requested = (
        start_curated_index is not None or end_curated_index is not None
    )
    if boundary_requested and (
        start_curated_index is None or end_curated_index is None
    ):
        raise ValueError(
            "boundary correction requires both --start-curated-index and --end-curated-index"
        )
    if status != "HUMAN_CORRECTED" and (instruction is not None or boundary_requested):
        raise ValueError("instruction/boundary corrections require HUMAN_CORRECTED")
    if status == "HUMAN_CORRECTED" and instruction is None and not boundary_requested:
        raise ValueError(
            "HUMAN_CORRECTED requires instruction and/or boundary correction"
        )
    final_start = segment["model_start_curated_index"]
    final_end = segment["model_end_curated_index"]
    if boundary_requested:
        assert start_curated_index is not None and end_curated_index is not None
        final_start, final_end = start_curated_index, end_curated_index
        containing = [
            domain
            for domain in record["clean_domains"]
            if domain["start_curated_index"]
            <= final_start
            < final_end
            <= domain["end_curated_index"]
        ]
        if len(containing) != 1:
            raise ValueError("corrected boundaries cross or leave a D2/D3 clean domain")
        for other in record["semantic_segments"]:
            if other is segment:
                continue
            if max(final_start, other["final_start_curated_index"]) < min(
                final_end, other["final_end_curated_index"]
            ):
                raise ValueError(
                    "corrected boundaries overlap another semantic segment"
                )
    segment["verification_status"] = status
    segment["verification_source"] = "HUMAN_REVIEW"
    segment["human_review"] = {
        "reviewer": reviewer,
        "corrected_instruction": instruction,
        "corrected_start_curated_index": final_start if boundary_requested else None,
        "corrected_end_curated_index": final_end if boundary_requested else None,
    }
    segment["final_start_curated_index"] = final_start
    segment["final_end_curated_index"] = final_end
    segment["final_instruction"] = (
        None if status == "REJECTED" else instruction or segment["model_instruction"]
    )
    record["verification_status"] = hierarchical_status(record)
    seal(record)
    path = root / record["episode_id"] / "verification.json"
    write_verification(path, record)
    records = [
        load_verification(item)
        for item in sorted(root.glob("episode_*/verification.json"))
    ]
    summary = json.loads((root / "dataset_verification_summary.json").read_text())
    publish_index(
        root,
        records,
        VerificationPolicy(**record["policy"]),
        AutoVerificationAuditConfig(**summary["audit_config"]),
    )
    return record
