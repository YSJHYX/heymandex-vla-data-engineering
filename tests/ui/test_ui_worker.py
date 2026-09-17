"""UI worker delegates to frozen APIs; every HF call here is mocked."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from vla_data.ui import worker


@pytest.fixture
def worker_request(tmp_path: Path) -> dict:
    return {
        "stage": "hf-dry-run",
        "paths": {
            "raw": str(tmp_path / "raw"),
            "curated": str(tmp_path / "curated"),
            "quality": str(tmp_path / "quality"),
            "export": str(tmp_path / "export"),
        },
        "repo_id": "operator/private-dataset",
        "hf_python": "/existing/venv/bin/python",
        "lerobot_python": "/existing/venv/bin/python",
        "baseline_sha": "a" * 40,
    }


def test_no_new_dry_run_calls_existing_publisher_read_only(
    worker_request: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = {}

    def fake_publish(root, repo_id, **kwargs):
        observed.update({"root": root, "repo_id": repo_id, **kwargs})
        return {
            "action": "DRY_RUN",
            "would_action": "NO_NEW_EPISODES",
            "merge_plan": {"to_append": 0},
        }

    monkeypatch.setattr(worker, "publish_dataset", fake_publish)
    result = worker.execute(worker_request)
    assert result["dry_run"]["would_action"] == "NO_NEW_EPISODES"
    assert observed["dry_run"] is True
    assert observed["root"] == Path(worker_request["paths"]["export"]) / "train"


def test_publish_rechecks_private_head_before_existing_publisher(
    worker_request: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker_request["stage"] = "hf-publish"
    calls = []
    monkeypatch.setattr(
        worker, "hf_call", lambda *_: {"private": True, "sha": "b" * 40}
    )
    monkeypatch.setattr(
        worker, "publish_dataset", lambda *_args, **_kwargs: calls.append(1)
    )
    with pytest.raises(RuntimeError, match="HEAD"):
        worker.execute(worker_request)
    assert calls == []

    monkeypatch.setattr(
        worker, "hf_call", lambda *_: {"private": True, "sha": "a" * 40}
    )
    monkeypatch.setattr(
        worker,
        "publish_dataset",
        lambda *_args, **_kwargs: {
            "action": "UPLOADED",
            "commit_sha": "b" * 40,
            "remote_validation": {
                "resolved_sha": "b" * 40,
                "canonical_file_mismatches": 0,
                "lerobot": {"reload_pass": True},
            },
        },
    )
    assert worker.execute(worker_request)["publish"]["remote_validation"]["lerobot"][
        "reload_pass"
    ]


def test_validation_failure_is_hard_failure(
    worker_request: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker_request["stage"] = "validate"
    monkeypatch.setattr(
        worker,
        "validate_lerobot_export",
        lambda *_args, **_kwargs: SimpleNamespace(passed=False, errors=("bad wrist",)),
    )
    with pytest.raises(RuntimeError, match="bad wrist"):
        worker.execute(worker_request)
