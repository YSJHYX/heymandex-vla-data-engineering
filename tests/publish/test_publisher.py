"""Publisher orchestration with fully mocked Hub network calls."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import vla_data.publish.publisher as publisher_module
from tests.publish.conftest import manifest_entries
from vla_data.publish import (
    AuthRequiredError,
    InvalidDatasetRootError,
    PublicationEligibilityError,
    RemoteNotEmptyError,
    RemotePrivacyError,
    RemoteValidationError,
    build_canonical_manifest,
    publish_dataset,
    validate_remote,
)
from vla_data.publish.remote import HFWorkerError

REPO = "PPPPPilot/VLADexData"
REVISION = "deadbeef" * 5


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
                "codebase_tag": "v2.1",
            }
        if mode == "download":
            return {
                "snapshot_path": str(self.state["snapshot"]),
                "resolved_sha": self.state.get("resolved_sha", payload["revision"]),
                "private": self.state["private"],
            }
        raise AssertionError(mode)


def _state(files: list[dict], snapshot: Path | None = None) -> dict:
    return {
        "exists": True,
        "private": True,
        "sha": "initial",
        "files": files,
        "v2_1_tag_sha": "initial",
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
    assert result["local_file_count"] == 7


def test_empty_remote_uploads_once(hf_dataset_root, monkeypatch, tmp_path) -> None:
    hub = FakeHub(_state([], snapshot=tmp_path))
    monkeypatch.setattr(publisher_module, "hf_call", hub)
    result = publish_dataset(hf_dataset_root, REPO)
    assert result["action"] == "UPLOADED"
    assert result["commit_sha"] == "deadbeef" * 5
    assert result["uploaded_file_count"] == 8  # canonical + new README
    assert len(hub.uploads) == 1
    # The staged payload is the canonical tree plus a production dataset card.
    assert "README.md" in hub.uploaded_files
    assert len(hub.uploaded_files) == 8
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


def test_missing_or_stale_v21_tag_requires_publication(hf_dataset_root, hub) -> None:
    hub.state["v2_1_tag_sha"] = None
    result = publish_dataset(hf_dataset_root, REPO, dry_run=True)
    assert result["would_action"] == "UPLOADED"

    hub.state["v2_1_tag_sha"] = "old-commit"
    result = publish_dataset(hf_dataset_root, REPO)
    assert result["action"] == "UPLOADED"
    assert result["codebase_tag"] == "v2.1"


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


def test_dry_run_requires_official_local_reload_when_interpreter_supplied(
    hf_dataset_root, hub, monkeypatch
) -> None:
    monkeypatch.setattr(
        publisher_module,
        "_official_reload",
        lambda **kw: {"reload_pass": False, "errors": ["invalid video"]},
    )
    with pytest.raises(InvalidDatasetRootError, match="official local LeRobot reload"):
        publish_dataset(hf_dataset_root, REPO, lerobot_python="lerobot", dry_run=True)
    assert hub.calls == []


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


def test_public_remote_is_a_hard_safety_stop(hf_dataset_root, hub) -> None:
    hub.state["private"] = False
    with pytest.raises(RemotePrivacyError, match="private dataset repository"):
        publish_dataset(hf_dataset_root, REPO)
    assert "upload" not in hub.calls

    result = publish_dataset(hf_dataset_root, REPO, dry_run=True)
    assert result["action"] == "BLOCKED"
    assert result["would_action"] == "BLOCKED_PUBLIC_REPOSITORY"


@pytest.mark.parametrize("status", ["REVIEW_REQUIRED", None])
def test_review_or_missing_expert_approval_blocks_production(
    hf_dataset_root, hub, status
) -> None:
    path = hf_dataset_root / "meta" / "source_provenance.jsonl"
    row = json.loads(path.read_text())
    if status is None:
        row.pop("expert_training_status")
    else:
        row["expert_training_status"] = status
    path.write_text(json.dumps(row) + "\n")
    result = publish_dataset(hf_dataset_root, REPO, dry_run=True)
    assert result["action"] == "BLOCKED"
    assert result["would_action"] == "BLOCKED_EXPERT_APPROVAL"
    assert hub.calls == []
    with pytest.raises(PublicationEligibilityError, match="explicit"):
        publish_dataset(hf_dataset_root, REPO)
    assert "upload" not in hub.calls


def test_missing_source_export_fingerprint_blocks_publication(
    hf_dataset_root, hub
) -> None:
    (hf_dataset_root.parent / "export_summary.json").unlink()
    result = publish_dataset(hf_dataset_root, REPO, dry_run=True)
    assert result["would_action"] == "BLOCKED_SOURCE_EXPORT_FINGERPRINT"
    with pytest.raises(PublicationEligibilityError, match="source export fingerprint"):
        publish_dataset(hf_dataset_root, REPO)
    assert "upload" not in hub.calls


def test_cli_reports_local_policy_block_without_remote_account(
    monkeypatch, capsys
) -> None:
    import vla_data.publish as publish_package
    from vla_data.cli import main

    monkeypatch.setattr(
        publish_package,
        "publish_dataset",
        lambda *args, **kwargs: {
            "repo_id": REPO,
            "local_file_count": 7,
            "local_bytes": 100,
            "local_fingerprint": "a" * 64,
            "action": "BLOCKED",
            "would_action": "BLOCKED_EXPERT_APPROVAL",
            "reason": "review required",
        },
    )
    result = main(
        [
            "publish-hf",
            "--dataset-root",
            "unused",
            "--repo-id",
            REPO,
            "--lerobot-python",
            "python",
            "--dry-run",
        ]
    )
    assert result == 2
    assert "account not queried" in capsys.readouterr().out


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
    with pytest.raises(AuthRequiredError, match="hf auth login"):
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
        REVISION,
        hf_python="python",
        lerobot_python="lerobot",
        cache_dir=tmp_path / "cache",
    )
    assert result["canonical_file_mismatches"] == 0
    assert result["lerobot"]["reload_pass"] is True
    assert result["resolved_sha"] == REVISION


def test_remote_verification_requires_fresh_cache_and_pinned_private_revision(
    hf_dataset_root, monkeypatch, tmp_path
) -> None:
    snapshot = _prepare_snapshot(hf_dataset_root, tmp_path)
    hub = FakeHub(_state([], snapshot=snapshot))
    monkeypatch.setattr(publisher_module, "hf_call", hub)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "old").write_text("stale")
    with pytest.raises(RemoteValidationError, match="empty cache"):
        validate_remote(
            hf_dataset_root,
            REPO,
            REVISION,
            hf_python="python",
            lerobot_python="lerobot",
            cache_dir=occupied,
        )
    assert "download" not in hub.calls

    hub.state["resolved_sha"] = "wrong"
    with pytest.raises(RemoteValidationError, match="pinned commit"):
        validate_remote(
            hf_dataset_root,
            REPO,
            REVISION,
            hf_python="python",
            lerobot_python="lerobot",
            cache_dir=tmp_path / "fresh",
        )


@pytest.mark.parametrize("revision", ["", "main", "v2.1", "deadbeef"])
def test_remote_verification_requires_full_commit_sha(
    hf_dataset_root, monkeypatch, tmp_path, revision
) -> None:
    hub = FakeHub(_state([]))
    monkeypatch.setattr(publisher_module, "hf_call", hub)
    with pytest.raises(RemoteValidationError, match="40-character commit SHA"):
        validate_remote(
            hf_dataset_root,
            REPO,
            revision,
            hf_python="python",
            lerobot_python="lerobot",
            cache_dir=tmp_path / "fresh",
        )
    assert hub.calls == []


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
            REVISION,
            hf_python="python",
            lerobot_python="lerobot",
            cache_dir=tmp_path / "fresh",
        )


@pytest.mark.parametrize(
    "relative", ["meta/tasks.jsonl", "meta/source_provenance.jsonl"]
)
def test_remote_task_or_provenance_change_fails_validation(
    hf_dataset_root, monkeypatch, tmp_path, relative
) -> None:
    snapshot = _prepare_snapshot(hf_dataset_root, tmp_path)
    target = snapshot / relative
    target.write_bytes(target.read_bytes() + b"changed")
    hub = FakeHub(_state([], snapshot=snapshot))
    monkeypatch.setattr(publisher_module, "hf_call", hub)
    with pytest.raises(RemoteValidationError, match="hash_mismatch"):
        validate_remote(
            hf_dataset_root,
            REPO,
            REVISION,
            hf_python="python",
            lerobot_python="lerobot",
            cache_dir=tmp_path / "fresh",
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
            REVISION,
            hf_python="python",
            lerobot_python="lerobot",
            cache_dir=tmp_path / "fresh",
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
            REVISION,
            hf_python="python",
            lerobot_python="lerobot",
            cache_dir=tmp_path / "fresh",
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
            "datetime",
            "os",
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
