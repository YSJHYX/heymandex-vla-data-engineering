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

    parquet_files = sorted(root.glob(f"data/{CANONICAL_PARQUET}"))
    if not parquet_files:
        raise InvalidDatasetRootError("no Parquet episodes under data/")
    video_files = sorted(root.glob(f"videos/{CANONICAL_MP4}"))
    if not video_files or len(video_files) != 2 * len(parquet_files):
        raise InvalidDatasetRootError(
            "every derived episode needs exactly one head and one wrist MP4"
        )
    return {
        "episodes": len(parquet_files),
        "tasks": tasks,
        "fps": info.get("fps"),
        "info": info,
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
    """TEST_THRESHOLD dataset card; never a training feature."""

    return (
        "---\n"
        f"tags:\n- lerobot\n- robotics\nfps: {fps}\n---\n\n"
        "# HeymanDex VLA Data — Pipeline Test Dataset\n\n"
        "Status: TEST_THRESHOLD / NOT_PRODUCTION_DATASET\n\n"
        "Purpose: validate LeRobot v2.1 → Hugging Face → OpenPI data "
        "compatibility.\n\n"
        "Current data are pipeline bring-up samples and are not representative "
        "of production demonstration quality.\n\n"
        f"Tasks: {tasks}\n"
    )
