"""Publisher orchestration with fully mocked Hub network calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import vla_data.publish.publisher as publisher_module
from tests.publish.conftest import manifest_entries
from vla_data.publish import (
    AuthRequiredError,
    RemoteNotEmptyError,
    RemoteValidationError,
    build_canonical_manifest,
    publish_dataset,
    validate_remote,
)
from vla_data.publish.remote import HFWorkerError

REPO = "PPPPPilot/VLADexData"


class FakeHub:
    """Records every worker call; answers from scripted state."""

    def __init__(self, state: dict) -> None:
        self.state = state
        self.calls: list[str] = []
        self.uploads: list[str] = []
        self.uploaded_files: list[str] = []

    def __call__(self, mode: str, payload: dict, hf_python=None) -> dict:
        self.calls.append(mode)
        if mode == "whoami":
            return {"account": "PPPPPilot", "token_role": "write"}
        if mode == "repo_state":
            return self.state
        if mode == "upload":
            staging = Path(payload["staging_dir"])
            self.uploaded_files = sorted(
                path.relative_to(staging).as_posix()
                for path in staging.rglob("*")
                if path.is_file()
            )
            self.uploads.append(payload["staging_dir"])
            return {
                "commit_sha": "deadbeef" * 5,
                "commit_url": f"https://huggingface.co/datasets/{REPO}/tree/abc",
            }
        if mode == "download":
            return {"snapshot_path": str(self.state["snapshot"])}
        raise AssertionError(mode)


def _state(files: list[dict], snapshot: Path | None = None) -> dict:
    return {
        "exists": True,
        "private": True,
        "sha": "initial",
        "files": files,
        "snapshot": snapshot,
    }


def _matching_remote(root: Path) -> list[dict]:
    return [
        {
            "path": entry["relative_path"],
            "size": entry["size_bytes"],
            "lfs_sha256": entry["sha256"],
        }
        for entry in build_canonical_manifest(root)["files"]
    ]


@pytest.fixture
def hub(hf_dataset_root, monkeypatch):
    fake = FakeHub(_state(_matching_remote(hf_dataset_root)))
    monkeypatch.setattr(publisher_module, "hf_call", fake)
    return fake


# --------------------------------------------------------------- publish


def test_same_local_remote_fingerprint_skips_upload(hf_dataset_root, hub) -> None:
    result = publish_dataset(hf_dataset_root, REPO, hf_python="python")
    assert result["action"] == "SKIPPED"
    assert "upload" not in hub.calls
    assert result["local_file_count"] == 6


def test_empty_remote_uploads_once(hf_dataset_root, monkeypatch, tmp_path) -> None:
    hub = FakeHub(_state([], snapshot=tmp_path))
    monkeypatch.setattr(publisher_module, "hf_call", hub)
    result = publish_dataset(hf_dataset_root, REPO)
    assert result["action"] == "UPLOADED"
    assert result["commit_sha"] == "deadbeef" * 5
    assert result["uploaded_file_count"] == 7  # canonical + new README
    assert len(hub.uploads) == 1
    # The staged payload is the canonical tree plus a TEST_THRESHOLD README.
    assert "README.md" in hub.uploaded_files
    assert len(hub.uploaded_files) == 7
    assert not Path(hub.uploads[0]).exists()  # staging cleaned after upload
    # The D6 dataset itself was never touched.
    assert not (hf_dataset_root / "README.md").exists()


def test_changed_local_fingerprint_requires_upload(hf_dataset_root, hub) -> None:
    parquet = hf_dataset_root / "data" / "chunk-000" / "episode_000000.parquet"
    parquet.write_bytes(parquet.read_bytes() + b"changed")
    result = publish_dataset(hf_dataset_root, REPO)
    assert result["action"] == "UPLOADED"


def test_force_reuploads_matching_data(hf_dataset_root, hub) -> None:
    result = publish_dataset(hf_dataset_root, REPO, force=True)
    assert result["action"] == "UPLOADED"
    assert hub.calls.count("upload") == 1


def test_dry_run_performs_zero_upload(hf_dataset_root, hub) -> None:
    result = publish_dataset(hf_dataset_root, REPO, dry_run=True)
    assert result["action"] == "DRY_RUN"
    assert result["would_action"] == "SKIPPED"
    assert hub.calls == ["repo_state", "whoami"]

    parquet = hf_dataset_root / "data" / "chunk-000" / "episode_000000.parquet"
    parquet.write_bytes(b"different")
    result = publish_dataset(hf_dataset_root, REPO, dry_run=True)
    assert result["would_action"] == "UPLOADED"
    assert hub.calls == ["repo_state", "whoami", "repo_state", "whoami"]


def test_unknown_remote_files_block_destructive_upload(hf_dataset_root, hub) -> None:
    hub.state["files"] = [
        *_matching_remote(hf_dataset_root),
        {"path": "user_production.parquet", "size": 999},
    ]
    with pytest.raises(RemoteNotEmptyError, match="NOT EMPTY"):
        publish_dataset(hf_dataset_root, REPO)
    assert "upload" not in hub.calls


def test_unknown_remote_files_report_blocked_in_dry_run(hf_dataset_root, hub) -> None:
    hub.state["files"] = [
        *_matching_remote(hf_dataset_root),
        {"path": "user_production.parquet", "size": 999},
    ]
    result = publish_dataset(hf_dataset_root, REPO, dry_run=True)
    assert result["action"] == "BLOCKED"
    assert "upload" not in hub.calls


def test_non_canonical_remote_files_do_not_block(hf_dataset_root, hub) -> None:
    hub.state["files"] = [
        *_matching_remote(hf_dataset_root),
        {"path": ".gitattributes", "size": 100},
        {"path": "README.md", "size": 200},
    ]
    result = publish_dataset(hf_dataset_root, REPO)
    assert result["action"] == "SKIPPED"


def test_auth_failure_is_actionable(hf_dataset_root, monkeypatch) -> None:
    def unauthenticated(mode, payload, hf_python=None):
        if mode == "repo_state":
            raise HFWorkerError("AuthenticationError", "401 Unauthorized")
        raise AssertionError(mode)

    monkeypatch.setattr(publisher_module, "hf_call", unauthenticated)
    with pytest.raises(AuthRequiredError, match="huggingface-cli login"):
        publish_dataset(hf_dataset_root, REPO)


def test_no_token_persisted_in_evidence(hf_dataset_root, hub) -> None:
    result = publish_dataset(hf_dataset_root, REPO)
    serialized = json.dumps(result).lower()
    # No credential-shaped key or value may appear in publication evidence.
    for secret_hint in ("token", "secret", "password", "authorization"):
        assert secret_hint not in serialized, secret_hint
    assert set(result).isdisjoint({"token", "api_key", "credentials"})


# ------------------------------------------------------- remote validation


def _reload_ok() -> dict:
    return {
        "reload_pass": True,
        "episodes": 1,
        "frames": 3,
        "episode_lengths": [3],
        "fps": 30,
        "tasks": ["Place the cable on the table."],
        "state_shape": [3, 17],
        "action_shape": [3, 17],
        "state_action_bitwise_equal": True,
        "errors": [],
    }


def _prepare_snapshot(hf_dataset_root, tmp_path) -> Path:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for entry in manifest_entries(hf_dataset_root):
        target = snapshot / entry["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((hf_dataset_root / entry["relative_path"]).read_bytes())
    return snapshot


def test_validate_remote_full_pass(hf_dataset_root, monkeypatch, tmp_path) -> None:
    snapshot = _prepare_snapshot(hf_dataset_root, tmp_path)
    hub = FakeHub(_state([], snapshot=snapshot))
    monkeypatch.setattr(publisher_module, "hf_call", hub)
    monkeypatch.setattr(publisher_module, "_official_reload", lambda **kw: _reload_ok())
    result = validate_remote(
        hf_dataset_root,
        REPO,
        "rev123",
        hf_python="python",
        lerobot_python="lerobot",
        cache_dir=tmp_path / "cache",
    )
    assert result["canonical_file_mismatches"] == 0
    assert result["lerobot"]["reload_pass"] is True


def test_remote_hash_mismatch_fails_validation(
    hf_dataset_root, monkeypatch, tmp_path
) -> None:
    snapshot = _prepare_snapshot(hf_dataset_root, tmp_path)
    target = snapshot / "data" / "chunk-000" / "episode_000000.parquet"
    target.write_bytes(target.read_bytes() + b"corrupted")
    hub = FakeHub(_state([], snapshot=snapshot))
    monkeypatch.setattr(publisher_module, "hf_call", hub)
    monkeypatch.setattr(publisher_module, "_official_reload", lambda **kw: _reload_ok())
    with pytest.raises(RemoteValidationError, match="hash_mismatch"):
        validate_remote(
            hf_dataset_root,
            REPO,
            "rev123",
            hf_python="python",
            lerobot_python="lerobot",
            cache_dir=tmp_path,
        )


def test_missing_remote_mp4_fails_validation(
    hf_dataset_root, monkeypatch, tmp_path
) -> None:
    snapshot = _prepare_snapshot(hf_dataset_root, tmp_path)
    (
        snapshot
        / "videos"
        / "chunk-000"
        / "observation.images.wrist"
        / "episode_000000.mp4"
    ).unlink()
    hub = FakeHub(_state([], snapshot=snapshot))
    monkeypatch.setattr(publisher_module, "hf_call", hub)
    monkeypatch.setattr(publisher_module, "_official_reload", lambda **kw: _reload_ok())
    with pytest.raises(RemoteValidationError, match="missing"):
        validate_remote(
            hf_dataset_root,
            REPO,
            "rev123",
            hf_python="python",
            lerobot_python="lerobot",
            cache_dir=tmp_path,
        )


@pytest.mark.parametrize(
    "reload_errors",
    [
        ["remote tasks ['test']"],
        ["feature observation.state is float32 [32]"],
        ["reload worker failed: import error"],
    ],
)
def test_remote_reload_failures_fail_validation(
    hf_dataset_root, monkeypatch, tmp_path, reload_errors
) -> None:
    snapshot = _prepare_snapshot(hf_dataset_root, tmp_path)
    hub = FakeHub(_state([], snapshot=snapshot))
    monkeypatch.setattr(publisher_module, "hf_call", hub)
    broken = {"reload_pass": False, "errors": reload_errors}
    monkeypatch.setattr(publisher_module, "_official_reload", lambda **kw: broken)
    with pytest.raises(RemoteValidationError):
        validate_remote(
            hf_dataset_root,
            REPO,
            "rev123",
            hf_python="python",
            lerobot_python="lerobot",
            cache_dir=tmp_path,
        )


def test_publish_module_never_imports_openpi_or_lerobot() -> None:
    """This process must stay free of OpenPI/LeRobot imports (D8 boundary)."""

    import ast

    package = Path(publisher_module.__file__).parent
    for source in package.glob("*.py"):
        tree = ast.parse(source.read_text())
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module.split(".")[0])
        assert "openpi" not in modules, source.name
        assert "lerobot" not in modules, source.name
        # LeRobot/OpenPI execution happens only in the external reload worker
        # script, never in this package's own import graph.
        assert modules <= {
            "json",
            "__future__",
            "hashlib",
            "re",
            "pathlib",
            "sys",
            "subprocess",
            "tempfile",
            "time",
            "shutil",
            "huggingface_hub",
            "vla_data",
        }, (source.name, modules)
