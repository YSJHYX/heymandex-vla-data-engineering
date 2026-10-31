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
        return repo_state(request["repo_id"], request.get("revision"))
    if mode == "upload":
        return upload(
            request["repo_id"],
            request["staging_dir"],
            request["parent_commit"],
            request.get("delete_paths", []),
        )
    if mode == "download":
        return download(request["repo_id"], request["revision"], request["cache_dir"])
    if mode == "verify_files":
        return verify_files(request["repo_id"], request["revision"], request["files"])
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


def repo_state(repo_id: str, revision: str | None = None) -> dict:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.repo_info(
        repo_id=repo_id, repo_type="dataset", files_metadata=True, revision=revision
    )
    files = []
    for item in api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        revision=info.sha,
        recursive=True,
        expand=True,
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
    refs = api.list_repo_refs(repo_id=repo_id, repo_type="dataset")
    version_tag_sha = next(
        (
            getattr(tag, "target_commit", None)
            for tag in refs.tags
            if getattr(tag, "name", None) == "v2.1"
        ),
        None,
    )
    return {
        "exists": True,
        "private": info.private,
        "sha": info.sha,
        "files": files,
        "v2_1_tag_sha": version_tag_sha,
    }


def upload(
    repo_id: str, staging_dir: str, parent_commit: str, delete_paths: list[str]
) -> dict:
    from huggingface_hub import HfApi

    api = HfApi()
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=staging_dir,
        parent_commit=parent_commit,
        delete_patterns=delete_paths or None,
        commit_message="Publish RM65B+SG100 dataset (LeRobot v2.1)",
    )
    refs = api.list_repo_refs(repo_id=repo_id, repo_type="dataset")
    if any(getattr(tag, "name", None) == "v2.1" for tag in refs.tags):
        api.delete_tag(repo_id, tag="v2.1", repo_type="dataset")
    api.create_tag(
        repo_id,
        tag="v2.1",
        revision=commit.oid,
        repo_type="dataset",
    )
    return {
        "commit_sha": commit.oid,
        "commit_url": commit.commit_url,
        "codebase_tag": "v2.1",
    }


def download(repo_id: str, revision: str, cache_dir: str) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().repo_info(repo_id=repo_id, repo_type="dataset", revision=revision)
    if info.sha != revision:
        raise ValueError("pinned revision could not be verified")
    if not isinstance(info.private, bool):
        raise TypeError("repository visibility could not be verified")

    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        cache_dir=cache_dir,
    )
    return {"snapshot_path": path, "resolved_sha": info.sha, "private": info.private}


def verify_files(repo_id: str, revision: str, files: list) -> dict:
    """Byte-verify small non-LFS remote files against expected SHA-256 digests.

    Listing-based remote verification covers LFS files through the server-side
    content SHA; plain git-blob files (meta/*.jsonl, README) carry no sha256 in
    the tree listing, so they are downloaded individually and hashed. Only
    small canonical files are ever passed here.
    """

    import hashlib
    import tempfile
    from pathlib import Path

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.repo_info(repo_id=repo_id, repo_type="dataset", revision=revision)
    if info.sha != revision:
        raise ValueError("pinned revision could not be verified")
    if not isinstance(info.private, bool):
        raise TypeError("repository visibility could not be verified")

    mismatches = []
    with tempfile.TemporaryDirectory(prefix="vla_hf_verify_files_") as temporary:
        root = Path(temporary)
        for item in files:
            path = item["path"]
            try:
                local = hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    revision=revision,
                    filename=path,
                    local_dir=str(root),
                )
                data = Path(local).read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                if digest != item["sha256"] or len(data) != item.get("size", -1):
                    mismatches.append({"path": path, "error": "hash_mismatch"})
            except Exception as exc:  # noqa: BLE001 - report per-file, keep going
                mismatches.append(
                    {"path": path, "error": f"download_failed: {type(exc).__name__}"}
                )
    return {"mismatches": mismatches, "checked": len(files)}


if __name__ == "__main__":
    raise SystemExit(main())
