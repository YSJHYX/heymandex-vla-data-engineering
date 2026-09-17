"""Local dataset-root validation and canonical SHA-256 manifest construction."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

CANONICAL_PARQUET = "**/*.parquet"
CANONICAL_MP4 = "**/*.mp4"
METADATA_DIR = "meta"

REPO_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")

# HF-managed files that are not part of the canonical training payload.
NON_CANONICAL_REMOTE_FILES = frozenset({".gitattributes", "README.md"})
CAMERA_FEATURES = ("observation.images.head", "observation.images.wrist")


class InvalidRepoIdError(ValueError):
    """Malformed Hugging Face repo id."""


class InvalidDatasetRootError(ValueError):
    """The folder is not a validated LeRobot v2.1 canonical dataset root."""


def validate_repo_id(repo_id: str) -> str:
    """Validate ``owner/name`` form without touching the network."""

    if not isinstance(repo_id, str) or not REPO_ID_PATTERN.fullmatch(repo_id.strip()):
        raise InvalidRepoIdError(f"repo id must look like owner/name, got {repo_id!r}")
    return repo_id.strip()


def validate_dataset_root(dataset_root: str | Path) -> dict:
    """Validate the canonical LeRobot layout and 17D float32 contract."""

    root = Path(dataset_root)
    if not root.is_dir():
        raise InvalidDatasetRootError(f"dataset root is not a directory: {root}")

    info_path = root / METADATA_DIR / "info.json"
    if not info_path.is_file():
        raise InvalidDatasetRootError(f"missing {METADATA_DIR}/info.json under {root}")
    info = json.loads(info_path.read_text())
    if str(info.get("codebase_version")) != "v2.1":
        raise InvalidDatasetRootError(
            f"expected LeRobot codebase_version v2.1, got {info.get('codebase_version')!r}"
        )
    features = info.get("features", {})
    for key in ("observation.state", "action"):
        feature = features.get(key)
        if not isinstance(feature, dict):
            raise InvalidDatasetRootError(f"missing feature {key}")
        if feature.get("dtype") != "float32" or list(feature.get("shape", [])) != [17]:
            raise InvalidDatasetRootError(
                f"feature {key} must be float32 [17], got {feature.get('dtype')} "
                f"{feature.get('shape')}"
            )
    for key in CAMERA_FEATURES:
        feature = features.get(key)
        shape = feature.get("shape") if isinstance(feature, dict) else None
        if (
            not isinstance(feature, dict)
            or feature.get("dtype") != "video"
            or not isinstance(shape, list)
            or len(shape) != 3
            or any(
                isinstance(d, bool) or not isinstance(d, int) or d <= 0 for d in shape
            )
            or shape[2] != 3
        ):
            raise InvalidDatasetRootError(f"missing or invalid RGB video feature {key}")

    tasks_path = root / METADATA_DIR / "tasks.jsonl"
    if not tasks_path.is_file():
        raise InvalidDatasetRootError(f"missing {METADATA_DIR}/tasks.jsonl")
    tasks = [
        json.loads(line)["task"]
        for line in tasks_path.read_text().splitlines()
        if line.strip()
    ]
    if not tasks or not all(isinstance(task, str) and task for task in tasks):
        raise InvalidDatasetRootError("tasks.jsonl must contain at least one task")

    provenance_path = root / METADATA_DIR / "source_provenance.jsonl"
    if not provenance_path.is_file():
        raise InvalidDatasetRootError(f"missing {METADATA_DIR}/source_provenance.jsonl")
    provenance = [
        json.loads(line)
        for line in provenance_path.read_text().splitlines()
        if line.strip()
    ]
    if not provenance:
        raise InvalidDatasetRootError("source provenance must not be empty")
    required = {
        "source_episode_id",
        "source_start_index",
        "source_end_index",
        "transition_count",
        "task_instruction",
        "task_instruction_sha256",
    }
    for row in provenance:
        if not isinstance(row, dict) or not required <= row.keys():
            raise InvalidDatasetRootError("source provenance fields are incomplete")
        task = row["task_instruction"]
        start, end = row["source_start_index"], row["source_end_index"]
        if not isinstance(task, str) or not task.strip():
            raise InvalidDatasetRootError("source task instruction is empty")
        if (
            row["task_instruction_sha256"]
            != hashlib.sha256(task.encode("utf-8")).hexdigest()
        ):
            raise InvalidDatasetRootError("source task instruction hash mismatch")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or end - start != row["transition_count"]
        ):
            raise InvalidDatasetRootError("invalid source provenance row range")

    parquet_files = sorted(root.glob(f"data/{CANONICAL_PARQUET}"))
    if not parquet_files:
        raise InvalidDatasetRootError("no Parquet episodes under data/")
    video_files = sorted(root.glob(f"videos/{CANONICAL_MP4}"))
    if not video_files or len(video_files) != 2 * len(parquet_files):
        raise InvalidDatasetRootError(
            "every derived episode needs exactly one head and one wrist MP4"
        )
    if len(provenance) != len(parquet_files):
        raise InvalidDatasetRootError(
            "source provenance row count must equal LeRobot episode count"
        )
    if {row["task_instruction"] for row in provenance} != set(tasks):
        raise InvalidDatasetRootError(
            "source provenance tasks differ from LeRobot tasks"
        )
    for episode_index in range(len(parquet_files)):
        for key in CAMERA_FEATURES:
            matches = list(
                root.glob(f"videos/**/{key}/episode_{episode_index:06d}.mp4")
            )
            if len(matches) != 1:
                raise InvalidDatasetRootError(
                    f"expected one {key} MP4 for episode {episode_index}"
                )
    return {
        "episodes": len(parquet_files),
        "tasks": tasks,
        "provenance": provenance,
        "fps": info.get("fps"),
        "info": info,
        "expert_training_statuses": sorted(
            {str(row.get("expert_training_status")) for row in provenance}
        ),
    }


def build_canonical_manifest(dataset_root: str | Path) -> dict:
    """Deterministic SHA-256 manifest over the canonical training payload."""

    root = Path(dataset_root)
    entries = []
    paths = sorted(
        [
            *(root / METADATA_DIR).rglob("*"),
            *root.glob(f"data/{CANONICAL_PARQUET}"),
            *root.glob(f"videos/{CANONICAL_MP4}"),
        ],
        key=lambda p: p.relative_to(root).as_posix(),
    )
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        entries.append(
            {
                "relative_path": relative,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    if not entries:
        raise InvalidDatasetRootError(f"no canonical files found under {root}")
    fingerprint = hashlib.sha256(
        json.dumps(entries, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "schema_name": "vla_hf_publish_manifest",
        "schema_version": 1,
        "dataset_root": str(root.resolve()),
        "files": entries,
        "file_count": len(entries),
        "total_bytes": sum(entry["size_bytes"] for entry in entries),
        "fingerprint": fingerprint,
    }


def build_readme(tasks: list[str], fps: object) -> str:
    """Minimal production dataset card; never a training feature."""

    return (
        "---\n"
        f"tags:\n- lerobot\n- robotics\nfps: {fps}\n---\n\n"
        "# HeymanDex RM65B + SG100 VLA Dataset\n\n"
        "Status: PRIVATE_TRAINING_DATASET\n\n"
        "Cumulative dataset: each clean rollout is one separate episode, even when "
        "tasks repeat. Physical storage: measured 17D state, effective 17D action, "
        "head RGB, wrist RGB, and exact collection-time task instructions.\n\n"
        f"Tasks: {tasks}\n"
    )
