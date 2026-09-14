"""Subprocess entry point for Hugging Face Hub operations.

Runs inside an interpreter that has ``huggingface_hub`` installed (the project
venv deliberately does not). Communicates one JSON request on stdin and one
JSON response on stdout; never prints credentials.
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    request = json.loads(sys.stdin.read())
    mode = request["mode"]
    try:
        response = dispatch(mode, request)
        response["ok"] = True
    except Exception as exc:  # noqa: BLE001 - classify for the parent process
        response = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }
    sys.stdout.write(json.dumps(response))
    return 0


def dispatch(mode: str, request: dict) -> dict:
    if mode == "whoami":
        return whoami()
    if mode == "repo_state":
        return repo_state(request["repo_id"])
    if mode == "upload":
        return upload(request["repo_id"], request["staging_dir"])
    if mode == "download":
        return download(request["repo_id"], request["revision"], request["cache_dir"])
    raise ValueError(f"unknown hf worker mode {mode!r}")


def whoami() -> dict:
    from huggingface_hub import HfApi

    info = HfApi().whoami()
    token = info.get("auth", {}).get("accessToken", {})
    return {
        "account": info.get("name"),
        "account_type": info.get("type"),
        "token_role": token.get("role"),
    }


def repo_state(repo_id: str) -> dict:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.repo_info(repo_id=repo_id, repo_type="dataset", files_metadata=True)
    files = []
    for item in api.list_repo_tree(
        repo_id=repo_id, repo_type="dataset", recursive=True, expand=True
    ):
        if type(item).__name__ == "RepoFolder":  # directory entries have no blob
            continue
        entry = {"path": item.path, "size": getattr(item, "size", None)}
        lfs = getattr(item, "lfs", None)
        if lfs is not None:
            # Hub >=0.30 renamed BlobLfsInfo.oid to .sha256.
            sha = getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)
            if sha:
                entry["lfs_sha256"] = sha
        files.append(entry)
    return {
        "exists": True,
        "private": bool(info.private),
        "sha": info.sha,
        "files": files,
    }


def upload(repo_id: str, staging_dir: str) -> dict:
    from huggingface_hub import HfApi

    commit = HfApi().upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=staging_dir,
        commit_message="D7 publish TEST_THRESHOLD dataset (LeRobot v2.1)",
    )
    return {"commit_sha": commit.oid, "commit_url": commit.commit_url}


def download(repo_id: str, revision: str, cache_dir: str) -> dict:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        cache_dir=cache_dir,
    )
    return {"snapshot_path": path}


if __name__ == "__main__":
    raise SystemExit(main())
