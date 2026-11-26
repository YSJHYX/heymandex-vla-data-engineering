"""Regression coverage for upload completion, resumable audits and metadata checks."""

import json

import pytest

from tests.publish.test_publisher import (
    REPO,
    REVISION,
    FakeHub,
    _mock_upload_merge,
    _prepare_snapshot,
    _reload_ok,
    _state,
    publish_dataset,
    publisher_module,
    validated_export,  # noqa: F401 - pytest autouse fixture
)
from vla_data.publish import retry_verification
from vla_data.publish.publisher import RemoteValidationError, validate_remote
from vla_data.publish.remote import HFWorkerError


def setup_publication(monkeypatch, tmp_path):
    hub = FakeHub(_state([], snapshot=tmp_path))
    monkeypatch.setattr(publisher_module, "hf_call", hub)
    _mock_upload_merge(monkeypatch)
    return hub


def test_default_verification_does_not_download_payload(
    hf_dataset_root, monkeypatch, tmp_path
):
    hub = setup_publication(monkeypatch, tmp_path)
    monkeypatch.setattr(
        publisher_module,
        "validate_remote",
        lambda *a, **k: pytest.fail("full audit invoked"),
    )
    result = publish_dataset(hf_dataset_root, REPO)
    assert result["action"] == "UPLOADED"
    assert "download" not in hub.calls
    assert "dataset_summary" in hub.calls
    assert result["remote_validation"]["remote_summary"]["total_frames"] == 3


def test_timeout_retains_commit_and_retry_never_uploads(
    hf_dataset_root, monkeypatch, tmp_path
):
    hub = setup_publication(monkeypatch, tmp_path)
    cache = tmp_path / "cache"
    root = cache / REPO.replace("/", "__")
    root.mkdir(parents=True)
    counter = root / ".verification_counter.json"
    counter.write_text('{"successful_publishes": 10}')
    paths = []

    def timeout(*args, **kwargs):
        paths.append(kwargs["cache_dir"])
        raise HFWorkerError("HFWorkerTimeout", "HF download exceeded 43200 seconds")

    monkeypatch.setattr(publisher_module, "validate_remote", timeout)
    pending = publish_dataset(
        hf_dataset_root, REPO, verification_mode="full", baseline_cache_dir=cache
    )
    assert pending["action"] == "UPLOADED_VERIFICATION_PENDING"
    assert pending["commit_sha"] == REVISION
    assert json.loads(counter.read_text())["successful_publishes"] == 10
    assert (root / "publications" / REVISION / "expected").is_dir()
    assert len(hub.uploads) == 1

    def success(*args, **kwargs):
        assert kwargs["cache_dir"] == paths[0]
        assert kwargs["resume"] is True
        return {"resolved_sha": REVISION, "lerobot": _reload_ok()}

    monkeypatch.setattr(publisher_module, "validate_remote", success)
    result = retry_verification(
        REPO,
        REVISION,
        hf_python="python",
        lerobot_python="lerobot",
        baseline_cache_dir=cache,
    )
    assert result["action"] == "UPLOADED"
    assert json.loads(counter.read_text())["successful_publishes"] == 11
    retry_verification(
        REPO,
        REVISION,
        hf_python="python",
        lerobot_python="lerobot",
        baseline_cache_dir=cache,
    )
    assert json.loads(counter.read_text())["successful_publishes"] == 11
    assert len(hub.uploads) == 1


def test_summary_mismatch_is_not_success(hf_dataset_root, monkeypatch, tmp_path):
    hub = setup_publication(monkeypatch, tmp_path)

    def wrong_counts(mode, payload, hf_python=None):
        if mode == "dataset_summary":
            return {
                "resolved_sha": REVISION,
                "total_episodes": 1,
                "total_frames": 999,
                "total_videos": 2,
            }
        return hub(mode, payload, hf_python)

    monkeypatch.setattr(publisher_module, "hf_call", wrong_counts)
    result = publish_dataset(hf_dataset_root, REPO)
    assert result["action"] == "UPLOADED_VERIFICATION_PENDING"
    assert result["commit_sha"] == REVISION
    assert result["remote_validation"]["status"] == "PENDING"


def test_resume_cache_is_bound_to_repo_revision_and_payload(
    hf_dataset_root, monkeypatch, tmp_path
):
    snapshot = _prepare_snapshot(hf_dataset_root, tmp_path)
    hub = FakeHub(_state([], snapshot=snapshot))
    monkeypatch.setattr(publisher_module, "hf_call", hub)
    cache = tmp_path / "audit-cache"
    kwargs = {
        "hf_python": "python",
        "lerobot_python": "lerobot",
        "cache_dir": cache,
        "resume": True,
    }
    validate_remote(hf_dataset_root, REPO, REVISION, **kwargs)
    (cache / "partial-download").write_bytes(b"incomplete")
    validate_remote(hf_dataset_root, REPO, REVISION, **kwargs)
    with pytest.raises(RemoteValidationError, match="another payload"):
        validate_remote(hf_dataset_root, REPO, "b" * 40, **kwargs)


def test_changed_retained_reference_never_passes(
    hf_dataset_root, monkeypatch, tmp_path
):
    setup_publication(monkeypatch, tmp_path)
    monkeypatch.setattr(
        publisher_module,
        "validate_remote",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("interrupted")),
    )
    cache = tmp_path / "cache"
    publish_dataset(
        hf_dataset_root, REPO, baseline_cache_dir=cache, verification_mode="full"
    )
    expected = cache / REPO.replace("/", "__") / "publications" / REVISION / "expected"
    (expected / "data/chunk-000/episode_000000.parquet").write_bytes(b"changed")
    monkeypatch.setattr(
        publisher_module,
        "validate_remote",
        lambda *a, **k: pytest.fail("changed reference accepted"),
    )
    result = retry_verification(
        REPO,
        REVISION,
        hf_python="python",
        lerobot_python="lerobot",
        baseline_cache_dir=cache,
    )
    assert result["action"] == "UPLOADED_VERIFICATION_PENDING"
