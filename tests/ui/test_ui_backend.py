"""Operator UI safety and persistence contracts (no real HF writes)."""

from __future__ import annotations

import io
import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from vla_data.ui import jobs
from vla_data.ui import server as ui_server
from vla_data.ui.config import HFTarget, UIConfig, load_config
from vla_data.ui.jobs import PipelineJobRunner, sanitize
from vla_data.ui.server import UIServer

TEST_TARGETS = (
    HFTarget(
        id="test",
        label="测试",
        dataset_name="private-dataset",
        repo_id="operator/private-dataset",
    ),
    HFTarget(
        id="production",
        label="正式数采",
        dataset_name="production-dataset",
        repo_id="operator/production-dataset",
    ),
)


@pytest.fixture
def config(tmp_path: Path) -> UIConfig:
    root = tmp_path / "runs"
    run = root / "sample"
    (run / "raw").mkdir(parents=True)
    (run / "work" / "curated").mkdir(parents=True)
    (run / "work" / "quality").mkdir(parents=True)
    export = run / "work" / "lerobot"
    (export / "train").mkdir(parents=True)
    (export / "export_summary.json").write_text('{"fingerprint":"source"}')
    return UIConfig(
        allowed_data_roots=(root.resolve(),),
        allowed_raw_roots=(root.resolve(),),
        state_dir=root / ".ui",
        default_hf_repo="operator/private-dataset",
        hf_targets=TEST_TARGETS,
        default_hf_target="test",
        uv_cache_dir=tmp_path / "uv-cache",
        lerobot_python=Path("/bin/false"),
        hf_python=Path("/bin/false"),
        host="127.0.0.1",
        port=0,
        run_overrides={},
    )


def _seed(runner: PipelineJobRunner, stage: str, status: str, summary: dict) -> str:
    job_id = f"{len(runner.list()) + 1:032x}"
    context = {
        "fingerprints": {
            "source_export_fingerprint": "source",
            "local_fingerprint": "local",
        },
        "repo_id": "operator/private-dataset",
    }
    with runner._connection() as db:
        db.execute(
            "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                job_id,
                "sample",
                stage,
                status,
                "2026-01-01",
                "2026-01-01",
                "[]",
                0,
                json.dumps(summary),
                None,
                json.dumps(context),
                str(runner.logs_dir / f"{job_id}.log"),
            ),
        )
    return job_id


def _ready_runner(
    config: UIConfig, monkeypatch: pytest.MonkeyPatch
) -> PipelineJobRunner:
    runner = PipelineJobRunner(config)
    monkeypatch.setattr(
        runner,
        "_fingerprints",
        lambda _: {"source_export_fingerprint": "source", "local_fingerprint": "local"},
    )
    return runner


def test_run_discovery_traversal_and_symlink(config: UIConfig, tmp_path: Path) -> None:
    assert config.list_runs() == ["sample"]
    with pytest.raises(ValueError):
        config.run("../../etc")
    external = tmp_path / "outside"
    external.mkdir()
    (config.allowed_data_roots[0] / "escape").symlink_to(
        external, target_is_directory=True
    )
    assert "escape" not in config.list_runs()
    with pytest.raises(ValueError):
        config.run("escape")
    (config.allowed_data_roots[0] / "sample" / "work" / "curated").rmdir()
    (config.allowed_data_roots[0] / "sample" / "work" / "curated").symlink_to(
        external, target_is_directory=True
    )
    with pytest.raises(ValueError, match="超出允许范围"):
        config.run("sample")


def test_virtual_environment_entry_path_is_not_resolved(
    config: UIConfig, tmp_path: Path
) -> None:
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to("/bin/false")
    source = tmp_path / "ui.toml"
    source.write_text(
        f'allowed_data_roots = ["{config.allowed_data_roots[0]}"]\n'
        f'allowed_raw_roots = ["{config.allowed_raw_roots[0]}"]\n'
        f'state_dir = "{config.state_dir}"\n'
        'default_hf_repo = "operator/private-dataset"\n'
        'default_hf_target = "test"\n'
        f'uv_cache_dir = "{config.uv_cache_dir}"\n'
        f'lerobot_python = "{venv_python}"\n'
        "\n[[hf_targets]]\n"
        'id = "test"\n'
        'label = "测试"\n'
        'dataset_name = "private-dataset"\n'
        'repo_id = "operator/private-dataset"\n'
    )
    loaded = load_config(source)
    assert loaded.lerobot_python == venv_python
    assert loaded.hf_python == venv_python


def test_invalid_repo_and_command_injection_rejected(
    config: UIConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _ready_runner(config, monkeypatch)
    with pytest.raises(ValueError):
        runner.submit("sample;echo PWNED", "validate")
    with pytest.raises(ValueError):
        runner.submit("sample", "validate", {"repo_id": "owner/repo;echo PWNED"})
    with pytest.raises(ValueError):
        runner.submit("sample", "export", {"dataset_name": "dataset;echo PWNED"})
    with pytest.raises(ValueError):
        runner.submit("sample", "shell")


def test_shell_false_and_job_persistence(
    config: UIConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _ready_runner(config, monkeypatch)
    seen: dict = {}

    class FakePopen:
        def __init__(self, argv, **kwargs):
            seen["argv"] = argv
            seen["shell"] = kwargs["shell"]
            self.stdin = io.StringIO()
            self.stdout = io.StringIO(
                "UI_EVENT "
                + json.dumps(
                    {
                        "kind": "result",
                        "ok": True,
                        "summary": {
                            "validation": {"passed": True, "evidence": "x" * 5000}
                        },
                    }
                )
                + "\n"
            )
            self.pid = 12345

        def wait(self):
            return 0

    monkeypatch.setattr(jobs.subprocess, "Popen", FakePopen)
    submitted = runner.submit("sample", "validate")
    for _ in range(100):
        result = runner.get(submitted["job_id"])
        if result["status"] == "PASS":
            break
        time.sleep(0.01)
    assert result["status"] == "PASS"
    assert seen["shell"] is False
    assert seen["argv"][1:] == ["-u", "-m", "vla_data.ui.worker"]
    assert len(result["summary"]["validation"]["evidence"]) == 5000
    assert PipelineJobRunner(config).get(submitted["job_id"])["status"] == "PASS"


def test_validation_dry_run_and_publish_gates(
    config: UIConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _ready_runner(config, monkeypatch)
    with pytest.raises(ValueError, match="本地验证"):
        runner.submit("sample", "hf-dry-run")
    _seed(runner, "validate", "FAILED", {})
    with pytest.raises(ValueError, match="本地验证"):
        runner.submit("sample", "hf-dry-run")
    _seed(runner, "validate", "PASS", {})
    with pytest.raises(ValueError, match="Dry Run"):
        runner.submit("sample", "hf-publish", {"confirmation": "我已检查 Dry Run 结果"})
    no_new_id = _seed(
        runner,
        "hf-dry-run",
        "PASS",
        {
            "dry_run": {
                "merge_plan": {"to_append": 0},
                "remote_before": {"sha": "a" * 40},
            }
        },
    )
    with pytest.raises(ValueError, match="无需重复上传"):
        runner.submit(
            "sample",
            "hf-publish",
            {"confirmation": "我已检查 Dry Run 结果", "dry_run_job_id": no_new_id},
        )
    append_id = _seed(
        runner,
        "hf-dry-run",
        "PASS",
        {
            "dry_run": {
                "merge_plan": {"to_append": 2},
                "remote_before": {"sha": "a" * 40},
            }
        },
    )
    with pytest.raises(ValueError, match="HF 仓库已变化"):
        runner.submit(
            "sample",
            "hf-publish",
            {
                "confirmation": "我已检查 Dry Run 结果",
                "dry_run_job_id": append_id,
                "repo_id": "someone/else",
            },
        )
    monkeypatch.setattr(
        runner, "_fingerprints", lambda _: {"source_export_fingerprint": "changed"}
    )
    with pytest.raises(ValueError, match="本地导出已变化"):
        runner.submit(
            "sample",
            "hf-publish",
            {"confirmation": "我已检查 Dry Run 结果", "dry_run_job_id": append_id},
        )
    _seed(runner, "hf-dry-run", "FAILED", {})
    with pytest.raises(ValueError, match="Dry Run"):
        runner.submit(
            "sample",
            "hf-publish",
            {"confirmation": "我已检查 Dry Run 结果", "dry_run_job_id": append_id},
        )


def test_camera_migration_only_dry_run_can_reach_publish_gate(
    config: UIConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _ready_runner(config, monkeypatch)
    _seed(runner, "validate", "PASS", {})
    dry_run_id = _seed(
        runner,
        "hf-dry-run",
        "PASS",
        {
            "dry_run": {
                "merge_plan": {
                    "to_append": 0,
                    "requires_camera_transform_migration": True,
                },
                "remote_before": {"sha": "a" * 40},
            }
        },
    )
    monkeypatch.setattr(runner, "_execute", lambda *_: None)

    job = runner.submit(
        "sample",
        "hf-publish",
        {
            "confirmation": "我已检查 Dry Run 结果",
            "dry_run_job_id": dry_run_id,
        },
    )

    assert job["status"] == "PENDING"


def test_global_publish_lock_and_read_only(
    config: UIConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _ready_runner(config, monkeypatch)
    _seed(runner, "validate", "PASS", {})
    append_id = _seed(
        runner,
        "hf-dry-run",
        "PASS",
        {
            "dry_run": {
                "merge_plan": {"to_append": 2},
                "remote_before": {"sha": "a" * 40},
            }
        },
    )
    monkeypatch.setattr(runner, "_execute", lambda *_: None)
    job = runner.submit(
        "sample",
        "hf-publish",
        {"confirmation": "我已检查 Dry Run 结果", "dry_run_job_id": append_id},
    )
    assert job["status"] == "PENDING"
    with pytest.raises(ValueError, match="已有任务"):
        runner.submit(
            "sample",
            "hf-publish",
            {"confirmation": "我已检查 Dry Run 结果", "dry_run_job_id": append_id},
        )
    with pytest.raises(ValueError, match="不能取消"):
        runner.cancel(job["job_id"])
    restarted = PipelineJobRunner(config)
    assert restarted.get(job["job_id"])["status"] == "FAILED"
    readonly = replace(config, run_overrides={"sample": {"read_only": True}})
    with pytest.raises(ValueError, match="只读"):
        PipelineJobRunner(readonly).submit("sample", "hf-publish")


def test_log_sanitization() -> None:
    safe = sanitize("Authorization: Bearer hf_abcdefghijklmnopqrstuvwxyz PASSWORD=abc")
    assert "hf_" not in safe and "PASSWORD=abc" not in safe


def test_http_routes_csrf_and_static(config: UIConfig) -> None:
    try:
        server = UIServer(config)
    except PermissionError:
        pytest.skip("当前 sandbox 不允许创建 loopback socket")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/runs") as response:
            assert json.load(response)["runs"] == ["sample"]
        with urlopen(base + "/") as response:
            html = response.read().decode()
            assert "HeymanDex VLA Data Engineering" in html
            assert 'href="/help"' in html
            assert server.csrf in html
            assert (
                "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
            )
        with urlopen(base + "/help") as response:
            guide = response.read().decode()
            assert response.status == 200
            assert "每日标准 SOP" in guide
            assert "数采人员操作与新电脑部署说明" in guide
        for asset in ("app.js", "style.css", "diagnostics.css", "help.css"):
            with urlopen(base + f"/static/{asset}") as response:
                assert response.status == 200
                assert response.read()
        with pytest.raises(HTTPError) as bad_host:
            urlopen(Request(base + "/api/runs", headers={"Host": "evil.example"}))
        assert bad_host.value.code == 403
        payload = json.dumps({"run_name": "sample"}).encode()
        request = Request(
            base + "/api/jobs/validate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as denied:
            urlopen(request)
        assert denied.value.code == 403
        with pytest.raises(HTTPError) as invalid:
            urlopen(
                Request(
                    base + "/api/jobs/validate",
                    data=json.dumps({"run_name": "../etc"}).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "X-UI-Token": server.csrf,
                    },
                    method="POST",
                )
            )
        assert invalid.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_episode_diagnostics_and_comparison_routes(
    config: UIConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple] = []

    def diagnostic(paths, episode_id, *, reason=None):
        seen.append(("diagnostic", paths.name, episode_id, reason))
        return {"episode_id": episode_id, "selected_reason": reason}

    def comparison(paths, episode_id, baseline):
        seen.append(("comparison", paths.name, episode_id, baseline))
        return {"current_episode": episode_id, "baseline_episode": baseline}

    monkeypatch.setattr(ui_server, "episode_diagnostics", diagnostic)
    monkeypatch.setattr(ui_server, "episode_diagnostics_comparison", comparison)
    try:
        server = UIServer(config)
    except PermissionError:
        pytest.skip("当前 sandbox 不允许创建 loopback socket")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        path = "/api/runs/sample/episodes/episode_000007/diagnostics"
        with urlopen(
            base + path + "?reason=CAMERA_HEAD_CAMERA_UNAVAILABLE"
        ) as response:
            value = json.load(response)
        assert value["selected_reason"] == "CAMERA_HEAD_CAMERA_UNAVAILABLE"

        path = "/api/runs/sample/episodes/episode_000007/comparison"
        with urlopen(base + path + "?baseline=episode_000006") as response:
            value = json.load(response)
        assert value == {
            "current_episode": "episode_000007",
            "baseline_episode": "episode_000006",
        }
        assert seen == [
            (
                "diagnostic",
                "sample",
                "episode_000007",
                "CAMERA_HEAD_CAMERA_UNAVAILABLE",
            ),
            ("comparison", "sample", "episode_000007", "episode_000006"),
        ]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
