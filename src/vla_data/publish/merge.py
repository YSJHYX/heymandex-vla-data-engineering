"""Plan and rebuild cumulative LeRobot datasets by logical episode."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from vla_data.publish.local import InvalidDatasetRootError, validate_dataset_root

WORKER = Path(__file__).with_name("_merge_worker.py")


class UnverifiableProvenanceError(InvalidDatasetRootError):
    """A source run cannot be identified without inventing provenance."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_key(row: dict) -> tuple[str, str, str, str]:
    """Use immutable RAW bytes plus the exact clean source run and task."""

    raw_hash = row.get("source_raw_sha256")
    raw_kind = row.get("source_raw_kind")
    original_task_hash = row.get("source_original_task_sha256")
    curated = row.get("curated_path")
    if original_task_hash is None:
        if not isinstance(curated, str):
            raise UnverifiableProvenanceError("source has no original task evidence")
        try:
            metadata = json.loads((Path(curated) / "metadata.json").read_text())
            instruction = metadata["language_instruction"]
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("empty original task")
            original_task_hash = hashlib.sha256(instruction.encode()).hexdigest()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise UnverifiableProvenanceError("cannot verify original task") from exc
    if original_task_hash != row["task_instruction_sha256"]:
        raise UnverifiableProvenanceError(
            "exported task differs from original instruction"
        )
    if raw_kind is None:
        if not isinstance(curated, str):
            raise UnverifiableProvenanceError("source has no captured RAW kind")
        try:
            report = json.loads((Path(curated) / "cleaning_report.json").read_text())
            raw_kind = report["source_provenance"]["dataset_status"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise UnverifiableProvenanceError(
                "cannot verify captured RAW kind"
            ) from exc
    if not isinstance(raw_kind, str) or not raw_kind or "SYNTHETIC" in raw_kind.upper():
        raise UnverifiableProvenanceError("synthetic or unknown RAW source kind")
    if raw_hash is None:
        if not isinstance(curated, str):
            raise UnverifiableProvenanceError("source has no captured RAW identity")
        metadata_path = Path(curated) / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text())
            raw_path = Path(metadata["source_episode_path"])
            raw_hash = _sha256(raw_path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise UnverifiableProvenanceError(
                f"cannot verify captured RAW identity for {row.get('source_episode_id')}"
            ) from exc
    if not isinstance(raw_hash, str) or len(raw_hash) != 64:
        raise UnverifiableProvenanceError("invalid captured RAW SHA-256")
    identity = {
        "source_raw_sha256": raw_hash,
        "source_episode_id": row["source_episode_id"],
        "source_start_index": row["source_start_index"],
        "source_end_index": row["source_end_index"],
        "task_instruction_sha256": row["task_instruction_sha256"],
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    stored = row.get("source_run_key")
    if stored is not None and stored != key:
        raise UnverifiableProvenanceError(
            "stored source run key differs from RAW evidence"
        )
    return key, raw_hash, raw_kind, original_task_hash


def plan_logical_merge(
    incoming_root: str | Path,
    baseline_root: str | Path | None,
    *,
    baseline_revision: str | None = None,
    source_export_fingerprint: str,
) -> dict:
    """Return an auditable plan; never conflate task identity with episode identity."""

    incoming = validate_dataset_root(incoming_root)
    baseline = validate_dataset_root(baseline_root) if baseline_root else None
    if baseline is not None and (
        baseline["fps"] != incoming["fps"]
        or baseline["info"]["features"] != incoming["info"]["features"]
    ):
        raise InvalidDatasetRootError(
            "baseline/incoming LeRobot FPS or features differ"
        )
    prior = baseline["provenance"] if baseline else []
    existing: set[str] = set()
    rows = []
    for merged_index, source in enumerate(prior):
        key, raw_hash, raw_kind, original_task_hash = _source_key(source)
        if key in existing:
            raise UnverifiableProvenanceError("baseline contains duplicate source runs")
        existing.add(key)
        rows.append(
            {
                **source,
                "lerobot_episode_index": merged_index,
                "source_run_key": key,
                "source_raw_sha256": raw_hash,
                "source_raw_kind": raw_kind,
                "source_original_task_sha256": original_task_hash,
            }
        )
    append_indices = []
    duplicate_indices = []
    incoming_seen: set[str] = set()
    for local_index, source in enumerate(incoming["provenance"]):
        key, raw_hash, raw_kind, original_task_hash = _source_key(source)
        if key in incoming_seen:
            raise UnverifiableProvenanceError("incoming batch repeats a source run")
        incoming_seen.add(key)
        if key in existing:
            duplicate_indices.append(local_index)
            continue
        existing.add(key)
        append_indices.append(local_index)
        rows.append(
            {
                **source,
                "lerobot_episode_index": len(rows),
                "source_run_key": key,
                "source_raw_sha256": raw_hash,
                "source_raw_kind": raw_kind,
                "source_original_task_sha256": original_task_hash,
                "upload_batch_id": source_export_fingerprint,
            }
        )
    incoming_lengths = [row["transition_count"] for row in incoming["provenance"]]
    old_frames = sum(row["transition_count"] for row in prior)
    appended_frames = sum(incoming_lengths[index] for index in append_indices)
    return {
        "baseline_revision": baseline_revision,
        "baseline_episodes": len(prior),
        "baseline_frames": old_frames,
        "incoming_episodes": len(incoming_lengths),
        "incoming_frames": sum(incoming_lengths),
        "already_present": len(duplicate_indices),
        "duplicate_indices": duplicate_indices,
        "to_append": len(append_indices),
        "append_indices": append_indices,
        "appended_frames": appended_frames,
        "merged_episodes": len(rows),
        "merged_frames": old_frames + appended_frames,
        "provenance": rows,
    }


def rebuild_logical_dataset(
    incoming_root: str | Path,
    baseline_root: str | Path | None,
    plan: dict,
    output_root: str | Path,
    *,
    lerobot_python: str,
) -> dict:
    """Use LeRobot's writer for a complete, isolated dataset-level rebuild."""

    output = Path(output_root)
    if output.exists():
        raise InvalidDatasetRootError("merged output root already exists")
    request = {
        "incoming_root": str(Path(incoming_root).resolve()),
        "baseline_root": str(Path(baseline_root).resolve()) if baseline_root else None,
        "append_indices": plan["append_indices"],
        "output_root": str(output.resolve()),
    }
    env = dict(os.environ)
    env["HF_DATASETS_CACHE"] = str(output.parent / "hf_datasets_cache")
    completed = subprocess.run(
        [lerobot_python, "-B", str(WORKER)],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        check=False,
        timeout=3600,
        env=env,
    )
    if completed.returncode or not completed.stdout.strip():
        raise InvalidDatasetRootError(
            f"LeRobot merge worker failed: {(completed.stderr or completed.stdout)[-500:]}"
        )
    result = json.loads(completed.stdout)
    if not result.get("ok"):
        raise InvalidDatasetRootError(f"LeRobot merge failed: {result.get('error')}")
    provenance = output / "meta" / "source_provenance.jsonl"
    provenance.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in plan["provenance"])
    )
    if (
        result["episodes"] != plan["merged_episodes"]
        or result["frames"] != plan["merged_frames"]
    ):
        raise InvalidDatasetRootError("merged counts differ from logical plan")
    validate_dataset_root(output)
    return result
