"""Publication orchestration: safe upload, clean download, remote validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Collection
from datetime import UTC, datetime
from pathlib import Path

from vla_data.export.validator import validate_lerobot_export
from vla_data.publish.local import (
    NON_CANONICAL_REMOTE_FILES,
    InvalidDatasetRootError,
    build_canonical_manifest,
    build_readme,
    validate_dataset_root,
    validate_repo_id,
)
from vla_data.publish.merge import plan_logical_merge, rebuild_logical_dataset
from vla_data.publish.remote import HFWorkerError, hf_call

STATUS_UPLOADED = "UPLOADED"
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


def publish_dataset(
    dataset_root: str | Path,
    repo_id: str,
    *,
    hf_python: str | None = None,
    lerobot_python: str | None = None,
    dry_run: bool = False,
    force: bool = False,
    verification_cache_dir: str | Path | None = None,
    legacy_camera_baseline_revisions: Collection[str] | None = None,
) -> dict:
    """Publish a complete, locally rebuilt cumulative LeRobot revision."""

    started = time.monotonic()
    validate_repo_id(repo_id)
    layout = validate_dataset_root(dataset_root)
    manifest = build_canonical_manifest(dataset_root)
    if lerobot_python is None:
        raise InvalidDatasetRootError(
            "publication requires --lerobot-python for independent validate-lerobot"
        )
    validation = validate_lerobot_export(
        Path(dataset_root).parent, lerobot_python=lerobot_python
    )
    if not validation.passed:
        raise InvalidDatasetRootError(
            f"independent validate-lerobot failed: {validation.errors}"
        )
    source_export_fingerprint = _source_export_fingerprint(dataset_root)
    if source_export_fingerprint is None:
        raise InvalidDatasetRootError(
            "missing valid direct D2/D3 source export fingerprint"
        )

    state = _authenticated_repo_state(repo_id, hf_python)
    if state["private"] is not True:
        if dry_run:
            return {
                "repo_id": repo_id,
                "action": "BLOCKED",
                "would_action": "BLOCKED_PUBLIC_REPOSITORY",
                "local_file_count": manifest["file_count"],
                "local_bytes": manifest["total_bytes"],
                "local_fingerprint": manifest["fingerprint"],
            }
        raise RemotePrivacyError(
            "HF publication requires an existing private dataset repository"
        )

    with tempfile.TemporaryDirectory(prefix="vla_hf_cumulative_") as temporary:
        work = Path(temporary)
        try:
            baseline, baseline_manifest = _download_valid_baseline(
                state, repo_id, hf_python, lerobot_python, work
            )
            plan = plan_logical_merge(
                dataset_root,
                baseline,
                baseline_revision=state["sha"] if baseline else None,
                source_export_fingerprint=source_export_fingerprint,
                legacy_camera_baseline_revisions=legacy_camera_baseline_revisions,
            )
        except (
            RemoteNotEmptyError,
            InvalidDatasetRootError,
            RemoteValidationError,
        ) as exc:
            if not dry_run:
                raise
            return {
                "repo_id": repo_id,
                "action": "BLOCKED",
                "would_action": "BLOCKED_INVALID_BASELINE_OR_PROVENANCE",
                "reason": str(exc),
                "local_file_count": manifest["file_count"],
                "local_bytes": manifest["total_bytes"],
                "local_fingerprint": manifest["fingerprint"],
            }
        evidence = {
            "schema_name": "vla_hf_publication",
            "schema_version": 2,
            "repo_id": repo_id,
            "repo_type": "dataset",
            "account": _account(hf_python),
            "local_fingerprint": manifest["fingerprint"],
            "local_file_count": manifest["file_count"],
            "local_bytes": manifest["total_bytes"],
            "source_export_fingerprint": source_export_fingerprint,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "expert_training_statuses": layout["expert_training_statuses"],
            "local_validation": validation.evidence,
            "remote_before": {
                "private": True,
                "sha": state["sha"],
                "file_count": len(state["files"]),
            },
            "merge_plan": {
                key: value for key, value in plan.items() if key != "provenance"
            },
            "tasks": layout["tasks"],
            "fps": layout["fps"],
        }
        if not plan["to_append"] and not plan["requires_camera_transform_migration"]:
            return {
                **evidence,
                "action": STATUS_DRY_RUN if dry_run else "NO_NEW_EPISODES",
                "would_action": "NO_NEW_EPISODES",
                "wall_time_s": time.monotonic() - started,
            }
        if dry_run:
            return {
                **evidence,
                "action": STATUS_DRY_RUN,
                "would_action": STATUS_UPLOADED,
                "wall_time_s": time.monotonic() - started,
            }
        rebuilt = work / "rebuilt"
        merged = rebuild_logical_dataset(
            dataset_root, baseline, plan, rebuilt, lerobot_python=lerobot_python
        )
        merged_layout = validate_dataset_root(rebuilt)
        reload = _official_reload(
            local_root=str(rebuilt),
            remote_root=str(rebuilt),
            lerobot_python=lerobot_python,
            expected_tasks=merged_layout["tasks"],
        )
        if not reload.get("reload_pass"):
            raise InvalidDatasetRootError(
                f"merged official LeRobot reload failed: {reload.get('errors')}"
            )
        merged_manifest = build_canonical_manifest(rebuilt)
        old_paths = (
            {entry["relative_path"] for entry in baseline_manifest["files"]}
            if baseline_manifest
            else set()
        )
        new_paths = {entry["relative_path"] for entry in merged_manifest["files"]}
        obsolete_managed = sorted(old_paths - new_paths)
        (rebuilt / "README.md").write_text(
            build_readme(merged_layout["tasks"], merged_layout["fps"])
        )
        commit = hf_call(
            "upload",
            {
                "repo_id": repo_id,
                "staging_dir": str(rebuilt),
                "parent_commit": state["sha"],
                "delete_paths": obsolete_managed,
            },
            hf_python,
        )
        remote_check = validate_remote(
            rebuilt,
            repo_id,
            revision=commit["commit_sha"],
            hf_python=hf_python,
            lerobot_python=lerobot_python,
            cache_dir=verification_cache_dir or work / "verification_cache",
        )
        return {
            **evidence,
            "action": STATUS_UPLOADED,
            "would_action": STATUS_UPLOADED,
            "commit_sha": commit["commit_sha"],
            "commit_url": commit["commit_url"],
            "codebase_tag": commit["codebase_tag"],
            "merged_validation": merged,
            "remote_validation": remote_check,
            "uploaded_file_count": merged_manifest["file_count"] + 1,
            "uploaded_bytes": merged_manifest["total_bytes"],
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
    if summary.get("source_mode") != "DIRECT_D2_D3":
        return None
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
    if (
        not isinstance(state.get("files"), list)
        or not isinstance(state.get("sha"), str)
        or not FULL_SHA_PATTERN.fullmatch(state["sha"])
    ):
        raise RemoteValidationError("remote repository metadata is incomplete")
    return state


def _download_valid_baseline(
    state: dict,
    repo_id: str,
    hf_python: str | None,
    lerobot_python: str,
    work: Path,
) -> tuple[Path | None, dict | None]:
    """Pin and validate the complete old dataset before any merge or upload."""

    paths = {file["path"] for file in state["files"]}
    managed = {path for path in paths if path.startswith(("meta/", "data/", "videos/"))}
    if not managed:
        unknown = paths - NON_CANONICAL_REMOTE_FILES
        if unknown:
            raise RemoteNotEmptyError(
                f"REMOTE REPOSITORY NOT EMPTY: {sorted(unknown)[:5]}"
            )
        return None, None
    if "meta/info.json" not in managed:
        raise InvalidDatasetRootError(
            "remote has managed files but no LeRobot info.json"
        )
    downloaded = hf_call(
        "download",
        {
            "repo_id": repo_id,
            "revision": state["sha"],
            "cache_dir": str(work / "baseline_cache"),
        },
        hf_python,
    )
    if (
        downloaded.get("resolved_sha") != state["sha"]
        or downloaded.get("private") is not True
    ):
        raise RemoteValidationError("baseline download is not pinned and private")
    snapshot = Path(downloaded["snapshot_path"])
    layout = validate_dataset_root(snapshot)
    manifest = build_canonical_manifest(snapshot)
    canonical = {entry["relative_path"] for entry in manifest["files"]}
    known_meta = {
        "meta/info.json",
        "meta/tasks.jsonl",
        "meta/episodes.jsonl",
        "meta/episodes_stats.jsonl",
        "meta/source_provenance.jsonl",
    }
    unknown = paths - canonical - NON_CANONICAL_REMOTE_FILES
    unknown |= {
        path
        for path in canonical
        if path.startswith("meta/") and path not in known_meta
    }
    if unknown:
        raise RemoteNotEmptyError(f"REMOTE REPOSITORY NOT EMPTY: {sorted(unknown)[:5]}")
    if not canonical <= paths:
        raise RemoteValidationError(
            "baseline file listing differs from pinned snapshot"
        )
    reload = _official_reload(
        local_root=str(snapshot),
        remote_root=str(snapshot),
        lerobot_python=lerobot_python,
        expected_tasks=layout["tasks"],
    )
    if not reload.get("reload_pass"):
        raise RemoteValidationError(
            f"baseline official LeRobot reload failed: {reload.get('errors')}"
        )
    return snapshot, manifest


def _account(hf_python: str | None) -> str | None:
    try:
        return hf_call("whoami", {}, hf_python).get("account")
    except HFWorkerError:
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
    for episode_index, row in enumerate(provenance):
        start = int(remote.episode_data_index["from"][episode_index])
        stop = int(remote.episode_data_index["to"][episode_index])
        task_indices = remote.hf_dataset.select(range(start, stop))["task_index"]
        if any(remote.meta.tasks[int(index)] != row["task_instruction"] for index in task_indices):
            fail(f"episode {episode_index} task differs from source provenance")
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
