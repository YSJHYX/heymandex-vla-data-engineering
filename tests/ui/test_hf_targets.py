"""Controlled HF dataset target selector: config, API whitelist, UI contract."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from vla_data.ui import server as ui_server
from vla_data.ui.config import HFTarget, UIConfig, load_config

TEST_REPO = "operator/private-dataset"
PRODUCTION_REPO = "operator/production-dataset"


@pytest.fixture
def config(tmp_path: Path) -> UIConfig:
    root = tmp_path / "runs"
    run = root / "sample"
    (run / "raw").mkdir(parents=True)
    (run / "work" / "curated").mkdir(parents=True)
    (run / "work" / "quality").mkdir(parents=True)
    (run / "work" / "lerobot" / "train").mkdir(parents=True)
    (run / "work" / "lerobot" / "export_summary.json").write_text('{"f":"s"}')
    return UIConfig(
        allowed_data_roots=(root.resolve(),),
        allowed_raw_roots=(root.resolve(),),
        state_dir=root / ".ui",
        default_hf_repo=TEST_REPO,
        hf_targets=(
            HFTarget("test", "测试", "private-dataset", TEST_REPO),
            HFTarget("production", "正式数采", "production-dataset", PRODUCTION_REPO),
        ),
        default_hf_target="test",
        uv_cache_dir=tmp_path / "uv",
        lerobot_python=Path("/bin/false"),
        hf_python=Path("/bin/false"),
        host="127.0.0.1",
        port=0,
        run_overrides={},
    )


def _write_toml(tmp_path: Path, body: str) -> Path:
    source = tmp_path / "ui.toml"
    source.write_text(body)
    return source


BASE_TOML = """
allowed_data_roots = ["{root}"]
allowed_raw_roots = ["{root}"]
state_dir = "{root}/.ui"
default_hf_repo = "operator/private-dataset"
default_hf_target = "test"
uv_cache_dir = "{tmp}/uv"
lerobot_python = "/bin/false"

[[hf_targets]]
id = "test"
label = "测试"
dataset_name = "private-dataset"
repo_id = "operator/private-dataset"

[[hf_targets]]
id = "production"
label = "正式数采"
dataset_name = "production-dataset"
repo_id = "operator/production-dataset"
"""


def _toml(tmp_path: Path) -> Path:
    return _write_toml(tmp_path, BASE_TOML.format(root=tmp_path / "runs", tmp=tmp_path))


def test_config_parses_two_hf_targets(tmp_path: Path) -> None:
    config = load_config(_toml(tmp_path))
    assert [target.id for target in config.hf_targets] == ["test", "production"]
    assert [target.repo_id for target in config.hf_targets] == [
        TEST_REPO,
        PRODUCTION_REPO,
    ]
    assert config.default_hf_target == "test"


def test_duplicate_target_id_rejected(tmp_path: Path) -> None:
    source = _toml(tmp_path)
    text = source.read_text() + (
        '\n[[hf_targets]]\nid = "test"\nlabel = "重复"\n'
        'dataset_name = "x"\nrepo_id = "operator/other"\n'
    )
    source.write_text(text)
    with pytest.raises(ValueError, match="id 重复"):
        load_config(source)


def test_duplicate_repo_id_rejected(tmp_path: Path) -> None:
    source = _toml(tmp_path)
    text = source.read_text() + (
        '\n[[hf_targets]]\nid = "clone"\nlabel = "克隆"\n'
        f'dataset_name = "x"\nrepo_id = "{TEST_REPO}"\n'
    )
    source.write_text(text)
    with pytest.raises(ValueError, match="repo_id 重复"):
        load_config(source)


def test_missing_default_target_rejected(tmp_path: Path) -> None:
    source = _toml(tmp_path)
    source.write_text(source.read_text().replace('default_hf_target = "test"', ""))
    with pytest.raises(ValueError, match="default_hf_target"):
        load_config(source)


def test_invalid_repo_id_rejected(tmp_path: Path) -> None:
    source = _toml(tmp_path)
    source.write_text(source.read_text().replace(TEST_REPO, "not a valid repo id"))
    with pytest.raises(ValueError):
        load_config(source)


def test_no_targets_rejected(tmp_path: Path) -> None:
    source = _toml(tmp_path)
    text = source.read_text().split("[[hf_targets]]")[0]
    source.write_text(text)
    with pytest.raises(ValueError, match="至少配置一个 HF 数据集目标"):
        load_config(source)


@pytest.fixture
def live_server(config, monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, dict]] = []

    def fake_hf_call(action, payload, interpreter):
        calls.append((action, payload))
        if action == "whoami":
            return {"account": "operator"}
        if action == "repo_state":
            return {"private": True, "sha": "a" * 40, "files": []}
        raise AssertionError(f"unexpected hf action {action}")

    monkeypatch.setattr(ui_server, "hf_call", fake_hf_call)
    try:
        server = ui_server.UIServer(config)
    except PermissionError:
        pytest.skip("当前 sandbox 不允许创建 loopback socket")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", server, calls
    server.shutdown()


def _post(base: str, path: str, token: str, payload: dict, port: int):
    return Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "X-UI-Token": token,
            "Origin": f"http://127.0.0.1:{port}",
        },
        method="POST",
    )


def test_system_status_returns_targets(live_server) -> None:
    base, _, _ = live_server
    with urlopen(base + "/api/system/status") as response:
        value = json.load(response)
    assert value["default_hf_target"] == "test"
    assert [(target["id"], target["repo_id"]) for target in value["hf_targets"]] == [
        ("test", TEST_REPO),
        ("production", PRODUCTION_REPO),
    ]
    assert value["hf_targets"][0]["dataset_name"] == "private-dataset"


def test_hf_status_queries_selected_repo(live_server) -> None:
    base, _, calls = live_server
    with urlopen(
        base + f"/api/hf/status?repo_id={PRODUCTION_REPO.replace('/', '%2F')}"
    ) as response:
        value = json.load(response)
    assert value["repo_id"] == PRODUCTION_REPO
    assert ("repo_state", {"repo_id": PRODUCTION_REPO}) in calls


def test_hf_status_rejects_repo_outside_targets(live_server) -> None:
    base, _, calls = live_server
    with pytest.raises(HTTPError) as denied:
        urlopen(base + "/api/hf/status?repo_id=operator%2Funknown")
    assert denied.value.code == 403
    assert all(payload.get("repo_id") != "operator/unknown" for _, payload in calls)


def test_hf_job_rejects_repo_outside_targets(live_server, monkeypatch) -> None:
    base, server, _ = live_server
    request = _post(
        base,
        "/api/jobs/hf-dry-run",
        server.csrf,
        {"run_name": "sample", "params": {"repo_id": "operator/unknown"}},
        server.server_port,
    )
    with pytest.raises(HTTPError) as denied:
        urlopen(request)
    assert denied.value.code == 403


def test_hf_job_accepts_production_target(live_server, monkeypatch) -> None:
    base, server, _ = live_server
    submitted: dict = {}

    class FakeRunner:
        def submit(self, run_name, stage, params):
            submitted.update({"run_name": run_name, "stage": stage, "params": params})
            return {
                "job_id": "0" * 32,
                "run_name": run_name,
                "stage": stage,
                "status": "PENDING",
            }

    monkeypatch.setattr(live_server[1], "runner", FakeRunner(), raising=False)
    request = _post(
        base,
        "/api/jobs/hf-dry-run",
        server.csrf,
        {"run_name": "sample", "params": {"repo_id": PRODUCTION_REPO}},
        server.server_port,
    )
    with urlopen(request) as response:
        assert response.status == 202
    assert submitted["params"]["repo_id"] == PRODUCTION_REPO


def test_publish_modal_and_selector_frontend_contract() -> None:
    static = Path(__file__).resolve().parents[2] / "src/vla_data/ui/static"
    html = (static / "index.html").read_text()
    app = (static / "app.js").read_text()
    # Select-only target control: no free-text repo input remains.
    assert '<select id="hf-repo"></select>' in html
    assert 'input id="hf-repo"' not in html
    # Options are rendered from system status with repo_id values.
    assert "renderHFTargets" in app
    assert "option.value=target.repo_id" in app
    assert "default_hf_target" in app
    # Production target shows an explicit warning note and modal title.
    assert "确认上传正式数据？" in app
    assert "正式数据集" in app
    # Publish stays gated on the dry-run repo matching the selected repo.
    assert 'dry.context?.repo_id!==$("hf-repo").value' in app
    # Switching the target invalidates the cached dry-run state.
    assert '"change",()=>{state.hf=null;state.dry=null' in app
