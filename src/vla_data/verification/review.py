"""Explicit human review of verification artifacts; D4 remains immutable."""

import json
from pathlib import Path

from vla_data.batch.discovery import canonical_episode_id
from vla_data.verification.policy import AutoVerificationAuditConfig, VerificationPolicy
from vla_data.verification.schema import load_verification, seal, write_verification
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
) -> dict:
    root = Path(verification_root).resolve()
    path = root / canonical_episode_id(episode) / "verification.json"
    record = load_verification(path)
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
