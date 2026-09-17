"""Publisher orchestration with fully mocked Hub network calls."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import vla_data.publish.publisher as publisher_module
from tests.publish.conftest import manifest_entries
from vla_data.export.validator import ExportValidation
from vla_data.publish import (
    AuthRequiredError,
    InvalidDatasetRootError,
    RemoteNotEmptyError,
    RemotePrivacyError,
    RemoteValidationError,
    build_canonical_manifest,
    validate_remote,
)
from vla_data.publish import publish_dataset as _publish_dataset
from vla_data.publish.remote import HFWorkerError

REPO = "PPPPPilot/VLADexData"
REVISION = "deadbeef" * 5


def publish_dataset(dataset_root, repo_id, **kwargs):
    kwargs.setdefault("lerobot_python", "lerobot")
    return _publish_dataset(dataset_root, repo_id, **kwargs)


@pytest.fixture(autouse=True)
def validated_export(monkeypatch):
    monkeypatch.setattr(
        publisher_module,
        "validate_lerobot_export",
        lambda *args, **kwargs: ExportValidation(
            (), {"official_reload": "PASS", "total_frames": 3}
        ),
    )
    monkeypatch.setattr(publisher_module, "_official_reload", lambda **kw: _reload_ok())


class FakeHub:
    """Records every worker call; answers from scripted state."""

    def __init__(self, state: dict) -> None:
        self.state = state
        self.calls: list[str] = []
        self.uploads: list[str] = []
        self.uploaded_files: list[str] = []
        self.upload_payloads: list[dict] = []
        self.download_revisions: list[str] = []

    def __call__(self, mode: str, payload: dict, hf_python=None) -> dict:
        self.calls.append(mode)
        if mode == "whoami":
            return {"account": "PPPPPilot", "token_role": "write"}
        if mode == "repo_state":
            return self.state
        if mode == "upload":
            self.upload_payloads.append(payload)
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
            self.download_revisions.append(payload["revision"])
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
        "sha": "a" * 40,
        "files": files,
        "v2_1_tag_sha": "a" * 40,
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


def _fake_rebuild(incoming, baseline, plan, output, *, lerobot_python):
    assert baseline is None
    shutil.copytree(incoming, output)
    (output / "meta/source_provenance.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in plan["provenance"])
    )
    return {
        "official_reload": "PASS",
        "episodes": plan["merged_episodes"],
        "frames": plan["merged_frames"],
    }


def _mock_upload_merge(monkeypatch):
    monkeypatch.setattr(publisher_module, "rebuild_logical_dataset", _fake_rebuild)
    monkeypatch.setattr(
        publisher_module,
        "validate_remote",
        lambda *args, **kwargs: {"revision": REVISION, "lerobot": _reload_ok()},
    )


@pytest.fixture
def hub(hf_dataset_root, monkeypatch, tmp_path):
    fake = FakeHub(
        _state(
            _matching_remote(hf_dataset_root),
            snapshot=_prepare_snapshot(hf_dataset_root, tmp_path),
        )
    )
    monkeypatch.setattr(publisher_module, "hf_call", fake)
    return fake


# --------------------------------------------------------------- publish


def test_same_local_remote_fingerprint_skips_upload(hf_dataset_root, hub) -> None:
    result = publish_dataset(hf_dataset_root, REPO, hf_python="python")
    assert result["action"] == "NO_NEW_EPISODES"
    assert result["merge_plan"]["already_present"] == 1
    assert "upload" not in hub.calls
    assert result["local_file_count"] == 7


def test_valid_current_head_is_only_baseline_no_historical_discovery(
    hf_dataset_root, hub
) -> None:
    hub.state["sha"] = "2347702eed03bb3b4a54c1f4598c47c98e804e45"
    hub.state["historical_revisions"] = ["b786109f06c985069c57118f5a815eedef684a2e"]
    result = publish_dataset(hf_dataset_root, REPO, dry_run=True)
    assert result["merge_plan"]["baseline_revision"] == hub.state["sha"]
    assert hub.download_revisions == [hub.state["sha"]]
    assert "upload" not in hub.calls


def test_empty_remote_uploads_once(hf_dataset_root, monkeypatch, tmp_path) -> None:
    hub = FakeHub(_state([], snapshot=tmp_path))
    monkeypatch.setattr(publisher_module, "hf_call", hub)
    _mock_upload_merge(monkeypatch)
    result = publish_dataset(hf_dataset_root, REPO)
    assert result["action"] == "UPLOADED"
    assert result["commit_sha"] == "deadbeef" * 5
    assert result["uploaded_file_count"] == 8  # canonical + new README
    assert len(hub.uploads) == 1
    assert hub.upload_payloads[0]["parent_commit"] == "a" * 40
    assert hub.upload_payloads[0]["delete_paths"] == []
    # The staged payload is the canonical tree plus a production dataset card.
    assert "README.md" in hub.uploaded_files
    assert len(hub.uploaded_files) == 8
    assert not Path(hub.uploads[0]).exists()  # staging cleaned after upload
    # The D6 dataset itself was never touched.
    assert not (hf_dataset_root / "README.md").exists()


def test_existing_logical_episode_is_not_overwritten_by_file_collision(
    hf_dataset_root, hub
) -> None:
    parquet = hf_dataset_root / "data" / "chunk-000" / "episode_000000.parquet"
    parquet.write_bytes(parquet.read_bytes() + b"changed")
    result = publish_dataset(hf_dataset_root, REPO)
    assert result["action"] == "NO_NEW_EPISODES"
    assert "upload" not in hub.calls


def test_force_cannot_duplicate_matching_data(hf_dataset_root, hub) -> None:
    result = publish_dataset(hf_dataset_root, REPO, force=True)
    assert result["action"] == "NO_NEW_EPISODES"
    assert hub.calls.count("upload") == 0


def test_tag_state_does_not_create_duplicate_episode(hf_dataset_root, hub) -> None:
    hub.state["v2_1_tag_sha"] = None
    result = publish_dataset(hf_dataset_root, REPO, dry_run=True)
    assert result["would_action"] == "NO_NEW_EPISODES"

    hub.state["v2_1_tag_sha"] = "old-commit"
    result = publish_dataset(hf_dataset_root, REPO)
    assert result["action"] == "NO_NEW_EPISODES"


def test_dry_run_performs_zero_upload(hf_dataset_root, hub) -> None:
    result = publish_dataset(hf_dataset_root, REPO, dry_run=True)
    assert result["action"] == "DRY_RUN"
    assert result["would_action"] == "NO_NEW_EPISODES"
    assert hub.calls == ["repo_state", "download", "whoami"]

    parquet = hf_dataset_root / "data" / "chunk-000" / "episode_000000.parquet"
    parquet.write_bytes(b"different")
    result = publish_dataset(hf_dataset_root, REPO, dry_run=True)
    assert result["would_action"] == "NO_NEW_EPISODES"
    assert hub.calls == [
        "repo_state",
        "download",
        "whoami",
        "repo_state",
        "download",
        "whoami",
    ]


@pytest.mark.parametrize("failure", ["D2_INVALID", "D3_REJECT", "SYNTHETIC_TEST_ONLY"])
def test_dry_run_requires_independent_export_validation(
    hf_dataset_root, hub, monkeypatch, failure
) -> None:
    monkeypatch.setattr(
        publisher_module,
        "validate_lerobot_export",
        lambda *args, **kwargs: ExportValidation((failure,), {}),
    )
    with pytest.raises(InvalidDatasetRootError, match="independent validate-lerobot"):
        publish_dataset(hf_dataset_root, REPO, dry_run=True)
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


@pytest.mark.parametrize(
    "status", ["REVIEW_REQUIRED", "EXCLUDE_FROM_EXPERT_TRAINING", "ARBITRARY", None]
)
def test_expert_status_does_not_gate_private_publication(
    hf_dataset_root, hub, monkeypatch, status
) -> None:
    path = hf_dataset_root / "meta" / "source_provenance.jsonl"
    row = json.loads(path.read_text())
    if status is None:
        row.pop("expert_training_status")
    else:
        row["expert_training_status"] = status
    path.write_text(json.dumps(row) + "\n")
    hub.state["files"] = []
    result = publish_dataset(hf_dataset_root, REPO, dry_run=True)
    assert result["action"] == "DRY_RUN"
    assert result["would_action"] == "UPLOADED"
    assert "upload" not in hub.calls
    _mock_upload_merge(monkeypatch)
    uploaded = publish_dataset(hf_dataset_root, REPO)
    assert uploaded["action"] == "UPLOADED"
    assert "meta/source_provenance.jsonl" in hub.uploaded_files


def test_missing_source_export_fingerprint_blocks_publication(
    hf_dataset_root, hub
) -> None:
    (hf_dataset_root.parent / "export_summary.json").unlink()
    with pytest.raises(InvalidDatasetRootError, match="source export fingerprint"):
        publish_dataset(hf_dataset_root, REPO, dry_run=True)
    assert "upload" not in hub.calls


def test_optional_manifest_export_is_not_the_direct_publication_path(
    hf_dataset_root, hub
) -> None:
    summary_path = hf_dataset_root.parent / "export_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["source_mode"] = "SEMANTIC_MANIFEST"
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(InvalidDatasetRootError, match="direct D2/D3"):
        publish_dataset(hf_dataset_root, REPO, dry_run=True)
    assert hub.calls == []


def test_non_canonical_remote_files_do_not_block(hf_dataset_root, hub) -> None:
    hub.state["files"] = [
        *_matching_remote(hf_dataset_root),
        {"path": ".gitattributes", "size": 100},
        {"path": "README.md", "size": 200},
    ]
    result = publish_dataset(hf_dataset_root, REPO)
    assert result["action"] == "NO_NEW_EPISODES"


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
        if source.name == "_merge_worker.py":
            continue  # separate existing LeRobot interpreter, not package import graph
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
