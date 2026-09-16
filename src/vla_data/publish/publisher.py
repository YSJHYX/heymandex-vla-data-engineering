"""Publication orchestration: safe upload, clean download, remote validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from vla_data.publish.local import (
    APPROVED_EXPERT_STATUS,
    NON_CANONICAL_REMOTE_FILES,
    InvalidDatasetRootError,
    build_canonical_manifest,
    build_readme,
    validate_dataset_root,
    validate_repo_id,
)
from vla_data.publish.remote import HFWorkerError, hf_call

STATUS_UPLOADED = "UPLOADED"
STATUS_SKIPPED = "SKIPPED"
STATUS_DRY_RUN = "DRY_RUN"
FULL_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class AuthRequiredError(RuntimeError):
    """No valid Hugging Face credential; give the official login command."""


class RemoteNotEmptyError(RuntimeError):
    """The remote repository holds files outside the canonical local payload."""


class RemotePrivacyError(RuntimeError):
    """Production training datasets must not be published to a public repo."""


class RemoteValidationError(RuntimeError):
    """The remote snapshot does not match the local canonical dataset."""


class PublicationEligibilityError(RuntimeError):
    """The export is valid for integration but not approved for production HF."""


def publish_dataset(
    dataset_root: str | Path,
    repo_id: str,
    *,
    hf_python: str | None = None,
    lerobot_python: str | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """Publish a validated D6 LeRobot root to the HF dataset repo root.

    Publication only: canonical files are copied byte-for-byte into a staging
    directory (plus a production dataset card when the remote has none) and
    uploaded in a single commit. The D6 dataset itself is never mutated.
    """

    started = time.monotonic()
    validate_repo_id(repo_id)
    layout = validate_dataset_root(dataset_root)
    manifest = build_canonical_manifest(dataset_root)
    source_export_fingerprint = _source_export_fingerprint(dataset_root)
    if lerobot_python:
        local_reload = _official_reload(
            local_root=str(Path(dataset_root).resolve()),
            remote_root=str(Path(dataset_root).resolve()),
            lerobot_python=lerobot_python,
            expected_tasks=layout["tasks"],
        )
        if not local_reload.get("reload_pass"):
            raise InvalidDatasetRootError(
                f"official local LeRobot reload failed: {local_reload.get('errors')}"
            )
    else:
        local_reload = None

    if not layout["publication_approved"]:
        reason = (
            "production publication requires explicit "
            f"expert_training_status={APPROVED_EXPERT_STATUS} on every source run; "
            f"observed {layout['expert_training_statuses']}"
        )
        if not dry_run:
            raise PublicationEligibilityError(reason)
        return {
            "schema_name": "vla_hf_publication",
            "schema_version": 1,
            "repo_id": repo_id,
            "action": "BLOCKED",
            "would_action": "BLOCKED_EXPERT_APPROVAL",
            "reason": reason,
            "local_fingerprint": manifest["fingerprint"],
            "local_file_count": manifest["file_count"],
            "local_bytes": manifest["total_bytes"],
            "local_official_reload": local_reload,
            "expert_training_statuses": layout["expert_training_statuses"],
            "wall_time_s": time.monotonic() - started,
        }
    if source_export_fingerprint is None:
        reason = "missing valid source export fingerprint beside the split root"
        if not dry_run:
            raise PublicationEligibilityError(reason)
        return {
            "schema_name": "vla_hf_publication",
            "schema_version": 1,
            "repo_id": repo_id,
            "action": "BLOCKED",
            "would_action": "BLOCKED_SOURCE_EXPORT_FINGERPRINT",
            "reason": reason,
            "local_fingerprint": manifest["fingerprint"],
            "local_file_count": manifest["file_count"],
            "local_bytes": manifest["total_bytes"],
            "local_official_reload": local_reload,
            "wall_time_s": time.monotonic() - started,
        }

    state = _authenticated_repo_state(repo_id, hf_python)
    action, would_action = _decide(state, manifest, dry_run=dry_run, force=force)

    evidence = {
        "schema_name": "vla_hf_publication",
        "schema_version": 1,
        "repo_id": repo_id,
        "repo_type": "dataset",
        "account": _account(hf_python),
        "local_fingerprint": manifest["fingerprint"],
        "local_file_count": manifest["file_count"],
        "local_bytes": manifest["total_bytes"],
        "source_export_fingerprint": source_export_fingerprint,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "expert_training_statuses": layout["expert_training_statuses"],
        "local_official_reload": local_reload,
        "remote_before": {
            "private": state["private"],
            "sha": state["sha"],
            "v2_1_tag_sha": state.get("v2_1_tag_sha"),
            "file_count": len(state["files"]),
        },
        "action": action,
        "would_action": would_action,
        "tasks": layout["tasks"],
        "fps": layout["fps"],
    }

    if action in {STATUS_DRY_RUN, "BLOCKED", STATUS_SKIPPED}:
        return {**evidence, "wall_time_s": time.monotonic() - started}

    readme_needed = not any(file["path"] == "README.md" for file in state["files"])
    staging = _stage(dataset_root, manifest, readme_needed, layout)
    try:
        commit = hf_call(
            "upload", {"repo_id": repo_id, "staging_dir": str(staging)}, hf_python
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {
        **evidence,
        "commit_sha": commit["commit_sha"],
        "commit_url": commit["commit_url"],
        "codebase_tag": commit["codebase_tag"],
        "uploaded_file_count": manifest["file_count"] + int(readme_needed),
        "uploaded_bytes": manifest["total_bytes"],
        "wall_time_s": time.monotonic() - started,
    }


def validate_remote(
    dataset_root: str | Path,
    repo_id: str,
    revision: str,
    *,
    hf_python: str | None,
    lerobot_python: str,
    cache_dir: str | Path,
) -> dict:
    """Clean-cache download, SHA-256 equality, and official LeRobot reload."""

    started = time.monotonic()
    validate_repo_id(repo_id)
    if not isinstance(revision, str) or not FULL_SHA_PATTERN.fullmatch(revision):
        raise RemoteValidationError(
            "remote verification requires an exact 40-character commit SHA"
        )
    manifest = build_canonical_manifest(dataset_root)
    cache = Path(cache_dir)
    if cache.exists() and (not cache.is_dir() or any(cache.iterdir())):
        raise RemoteValidationError(
            "fresh download requires an absent or empty cache directory"
        )
    downloaded = hf_call(
        "download",
        {"repo_id": repo_id, "revision": revision, "cache_dir": str(cache_dir)},
        hf_python,
    )
    snapshot = Path(downloaded["snapshot_path"])
    if downloaded.get("private") is not True:
        raise RemoteValidationError("downloaded revision is not confirmed private")
    if downloaded.get("resolved_sha") != revision:
        raise RemoteValidationError(
            "downloaded revision does not match pinned commit SHA"
        )

    mismatches = []
    for entry in manifest["files"]:
        remote = snapshot / entry["relative_path"]
        if not remote.is_file():
            mismatches.append(
                {"relative_path": entry["relative_path"], "error": "missing"}
            )
            continue
        data = remote.read_bytes()
        if (
            len(data) != entry["size_bytes"]
            or hashlib.sha256(data).hexdigest() != entry["sha256"]
        ):
            mismatches.append(
                {"relative_path": entry["relative_path"], "error": "hash_mismatch"}
            )
    if mismatches:
        raise RemoteValidationError(
            f"canonical file mismatches: {json.dumps(mismatches[:5])}"
        )

    lerobot_evidence = _official_reload(
        local_root=str(Path(dataset_root).resolve()),
        remote_root=str(snapshot),
        lerobot_python=lerobot_python,
        expected_tasks=manifest_tasks(dataset_root),
    )
    if not lerobot_evidence.get("reload_pass"):
        raise RemoteValidationError(
            f"official remote reload failed: {lerobot_evidence.get('errors')}"
        )
    return {
        "schema_name": "vla_hf_remote_validation",
        "schema_version": 1,
        "repo_id": repo_id,
        "revision": revision,
        "resolved_sha": downloaded["resolved_sha"],
        "private": True,
        "snapshot_path": str(snapshot),
        "source_export_fingerprint": _source_export_fingerprint(dataset_root),
        "local_fingerprint": manifest["fingerprint"],
        "canonical_file_mismatches": 0,
        "canonical_file_count": manifest["file_count"],
        "lerobot": lerobot_evidence,
        "wall_time_s": time.monotonic() - started,
    }


def manifest_tasks(dataset_root: str | Path) -> list[str]:
    return validate_dataset_root(dataset_root)["tasks"]


def _source_export_fingerprint(dataset_root: str | Path) -> str | None:
    path = Path(dataset_root).parent / "export_summary.json"
    if not path.is_file():
        return None
    summary = json.loads(path.read_text())
    value = summary.get("fingerprint")
    return (
        value
        if isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
        else None
    )


def _authenticated_repo_state(repo_id: str, hf_python: str | None) -> dict:
    try:
        state = hf_call("repo_state", {"repo_id": repo_id}, hf_python)
    except HFWorkerError as exc:
        if (
            "Auth" in exc.error_type
            or "Credential" in exc.error_type
            or "401" in exc.message
        ):
            raise AuthRequiredError(
                "Hugging Face authentication failed; run: hf auth login"
            ) from exc
        raise
    if state.get("exists") is not True:
        raise RemotePrivacyError(
            "HF publication requires an existing dataset repository"
        )
    if state.get("private") not in {True, False}:
        raise RemoteValidationError("remote repository visibility is unknown")
    if not isinstance(state.get("files"), list) or not state.get("sha"):
        raise RemoteValidationError("remote repository metadata is incomplete")
    return state


def _account(hf_python: str | None) -> str | None:
    try:
        return hf_call("whoami", {}, hf_python).get("account")
    except HFWorkerError:
        return None


def _decide(
    state: dict, manifest: dict, *, dry_run: bool, force: bool
) -> tuple[str, str]:
    """Return (reported_action, would_action); BLOCKED only stays silent in dry-run."""

    if state.get("private") is not True:
        if dry_run:
            return "BLOCKED", "BLOCKED_PUBLIC_REPOSITORY"
        raise RemotePrivacyError(
            "HF publication requires an existing private dataset repository; "
            "refusing to upload to a public repository"
        )

    canonical = {file["relative_path"]: file for file in manifest["files"]}
    unknown = [
        file["path"]
        for file in state["files"]
        if file["path"] not in canonical
        and file["path"] not in NON_CANONICAL_REMOTE_FILES
    ]
    if unknown:
        if dry_run:
            return "BLOCKED", "UPLOAD"
        raise RemoteNotEmptyError(
            "D7 BLOCKED — REMOTE REPOSITORY NOT EMPTY: "
            f"{len(unknown)} unknown remote file(s), first: {unknown[:5]}"
        )
    matches = _remote_matches(state["files"], canonical)
    tag_matches_head = state.get("v2_1_tag_sha") == state.get("sha")
    would = (
        STATUS_UPLOADED
        if (force or not matches or not tag_matches_head)
        else STATUS_SKIPPED
    )
    if dry_run:
        return STATUS_DRY_RUN, would
    return would, would


def _remote_matches(remote_files: list[dict], canonical: dict[str, dict]) -> bool:
    remote_by_path = {file["path"]: file for file in remote_files}
    if set(remote_by_path) < set(canonical):
        return False
    for path, entry in canonical.items():
        remote = remote_by_path.get(path)
        if remote is None or remote.get("size") != entry["size_bytes"]:
            return False
        lfs_sha = remote.get("lfs_sha256")
        if lfs_sha is not None and lfs_sha != entry["sha256"]:
            return False
    return True


def _stage(dataset_root, manifest: dict, readme_needed: bool, layout: dict) -> Path:
    """Copy canonical files unchanged plus the production dataset card."""

    base = Path(tempfile.mkdtemp(prefix="vla_d7_staging_", dir=_staging_parent()))
    root = Path(dataset_root)
    for entry in manifest["files"]:
        destination = base / entry["relative_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / entry["relative_path"], destination)
    if readme_needed:
        (base / "README.md").write_text(build_readme(layout["tasks"], layout["fps"]))
    return base


def _staging_parent() -> str | None:
    for candidate in ("/data", None):
        if candidate is None:
            return None
        try:
            probe = Path(candidate)
            if probe.is_dir() and probe.stat().st_mode & 0o200:
                return candidate
        except OSError:
            continue
    return None


_RELOAD_SCRIPT = r"""
import hashlib
import json
import numpy as np
from pathlib import Path

request = json.loads(input())
local_root = request["local_root"]
remote_root = request["remote_root"]
expected_tasks = request["expected_tasks"]

evidence = {"reload_pass": False, "errors": []}

def fail(message):
    evidence["errors"].append(message)
    print(json.dumps(evidence))
    raise SystemExit(0)

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    remote = LeRobotDataset("local/d7-remote", root=remote_root)
    local = LeRobotDataset("local/d6-local", root=local_root)
    info = remote.meta.info
    if str(info.get("codebase_version")) != "v2.1":
        fail(f"codebase {info.get('codebase_version')!r}")
    for key in ("observation.state", "action"):
        feature = remote.features[key]
        if feature["dtype"] != "float32" or list(feature["shape"]) != [17]:
            fail(f"feature {key} is {feature['dtype']} {feature['shape']}")
    camera_shapes = {}
    for key in ("observation.images.head", "observation.images.wrist"):
        feature = remote.features[key]
        shape = list(feature["shape"])
        if feature["dtype"] != "video" or len(shape) != 3 or shape[2] != 3:
            fail(f"invalid RGB camera feature {key}")
        if local.features[key] != feature:
            fail(f"camera feature {key} differs")
        camera_shapes[key] = shape
    tasks = [
        json.loads(line)["task"]
        for line in Path(remote_root, "meta/tasks.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if tasks != expected_tasks:
        fail(f"remote tasks {tasks}")
    provenance = [
        json.loads(line)
        for line in Path(remote_root, "meta/source_provenance.jsonl").read_text().splitlines()
        if line.strip()
    ]
    local_provenance = [
        json.loads(line)
        for line in Path(local_root, "meta/source_provenance.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if provenance != local_provenance or len(provenance) != remote.num_episodes:
        fail("source provenance differs or episode count mismatches")
    for row in provenance:
        task = row["task_instruction"]
        if row["task_instruction_sha256"] != hashlib.sha256(task.encode("utf-8")).hexdigest():
            fail("source task hash mismatch")
        if row["source_end_index"] - row["source_start_index"] != row["transition_count"]:
            fail("source episode range mismatch")
    if remote.num_episodes != local.num_episodes:
        fail(f"episode count {remote.num_episodes} != {local.num_episodes}")
    if remote.num_frames != local.num_frames:
        fail(f"frame count {remote.num_frames} != {local.num_frames}")
    lengths = [
        int(remote.episode_data_index["to"][i] - remote.episode_data_index["from"][i])
        for i in range(remote.num_episodes)
    ]
    state_remote = np.stack(remote.hf_dataset["observation.state"])
    action_remote = np.stack(remote.hf_dataset["action"])
    state_local = np.stack(local.hf_dataset["observation.state"])
    action_local = np.stack(local.hf_dataset["action"])
    if state_remote.dtype != np.float32 or action_remote.dtype != np.float32:
        fail("remote arrays are not float32")
    if state_remote.shape != (remote.num_frames, 17) or action_remote.shape != (remote.num_frames, 17):
        fail("remote state/action shape mismatch")
    if lengths != [row["transition_count"] for row in provenance]:
        fail("source range lengths differ from LeRobot episodes")
    if not np.array_equal(state_remote, state_local):
        fail("state arrays differ between local and remote")
    if not np.array_equal(action_remote, action_local):
        fail("action arrays differ between local and remote")
    import av
    for episode_index in range(remote.num_episodes):
        parquet_rows = lengths[episode_index]
        for camera in ("head", "wrist"):
            matches = sorted(
                Path(remote_root, "videos").rglob(
                    f"observation.images.{camera}/episode_{episode_index:06d}.mp4"
                )
            )
            if len(matches) != 1:
                fail(f"expected one {camera} MP4 for episode {episode_index}")
            with av.open(str(matches[0])) as container:
                frames = container.streams.video[0].frames
            if frames != parquet_rows:
                fail(
                    f"episode {episode_index}:{camera} frames {frames} != rows "
                    f"{parquet_rows}"
                )
    evidence.update(
        reload_pass=True,
        episodes=remote.num_episodes,
        frames=remote.num_frames,
        episode_lengths=lengths,
        fps=remote.fps,
        tasks=tasks,
        task_hashes=[row["task_instruction_sha256"] for row in provenance],
        source_ranges=[
            [row["source_episode_id"], row["source_start_index"], row["source_end_index"]]
            for row in provenance
        ],
        expert_training_statuses=[row.get("expert_training_status") for row in provenance],
        camera_shapes=camera_shapes,
        state_shape=list(state_remote.shape),
        action_shape=list(action_remote.shape),
        state_action_bitwise_equal=True,
    )
except Exception as exc:  # noqa: BLE001
    fail(f"{type(exc).__name__}: {exc}")
print(json.dumps(evidence))
"""


def _official_reload(
    *, local_root: str, remote_root: str, lerobot_python: str, expected_tasks: list[str]
) -> dict:
    request = json.dumps(
        {
            "local_root": local_root,
            "remote_root": remote_root,
            "expected_tasks": expected_tasks,
        }
    )
    with tempfile.TemporaryDirectory(prefix="vla_lerobot_reload_cache_") as cache:
        environment = os.environ.copy()
        environment["HF_DATASETS_CACHE"] = cache
        completed = subprocess.run(
            [lerobot_python, "-c", _RELOAD_SCRIPT],
            input=request,
            capture_output=True,
            text=True,
            check=False,
            timeout=1800,
            env=environment,
        )
    lines = [line for line in completed.stdout.strip().splitlines() if line.strip()]
    if not lines:
        return {
            "reload_pass": False,
            "errors": [f"reload worker failed: {completed.stderr[-300:]}"],
        }
    return json.loads(lines[-1])
