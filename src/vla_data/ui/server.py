"""Loopback-only JSON API and static operator console."""

from __future__ import annotations

import json
import mimetypes
import secrets
import subprocess
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from vla_data.publish.local import validate_repo_id
from vla_data.publish.remote import hf_call
from vla_data.ui.config import UIConfig
from vla_data.ui.jobs import PipelineJobRunner, sanitize
from vla_data.ui.readers import (
    episode_diagnostics,
    episode_diagnostics_comparison,
    run_snapshot,
)

STATIC = Path(__file__).with_name("static")
GUIDE = Path(__file__).resolve().parents[3] / "docs" / "OPERATOR_UI_GUIDE_CN.md"


def lerobot_version(interpreter: Path) -> str | None:
    if not interpreter.is_file():
        return None
    try:
        result = subprocess.run(
            [
                str(interpreter),
                "-c",
                "import importlib.metadata; print(importlib.metadata.version('lerobot'))",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


class UIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self, config: UIConfig, runner: PipelineJobRunner | None = None
    ) -> None:
        if config.host not in {"127.0.0.1", "localhost"}:
            raise ValueError("UI 只能监听 loopback 地址")
        self.config = config
        self.runner = runner or PipelineJobRunner(config)
        self.csrf = secrets.token_urlsafe(32)
        super().__init__((config.host, config.port), UIHandler)


class UIHandler(BaseHTTPRequestHandler):
    server: UIServer

    def log_message(self, format: str, *args: object) -> None:
        print("UI HTTP: " + sanitize(format % args), flush=True)

    def _headers(self, status: int, content_type: str, size: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.end_headers()

    def _json(self, status: int, value: dict | list) -> None:
        data = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(data))
        self.wfile.write(data)

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": sanitize(message)})

    def _host_allowed(self) -> bool:
        host = self.headers.get("Host", "")
        return host in {
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
        }

    def _serve_static(self, name: str) -> None:
        if name not in {
            "index.html",
            "app.js",
            "style.css",
            "diagnostics.css",
            "help.css",
        }:
            self._error(HTTPStatus.NOT_FOUND, "页面不存在")
            return
        data = (STATIC / name).read_bytes()
        if name == "index.html":
            data = data.replace(b"__UI_CSRF__", self.server.csrf.encode("ascii"))
        content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        self._headers(HTTPStatus.OK, f"{content_type}; charset=utf-8", len(data))
        self.wfile.write(data)

    def _serve_help(self) -> None:
        try:
            guide = GUIDE.read_text(encoding="utf-8")
        except OSError:
            self._error(HTTPStatus.NOT_FOUND, "本地使用说明不存在，请联系数据工程人员")
            return
        page = (
            '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            "<title>数采人员操作与新电脑部署说明</title>"
            '<link rel="stylesheet" href="/static/help.css"></head>'
            '<body class="help-page"><main><a href="/">← 返回操作界面</a>'
            "<h1>数采人员操作与新电脑部署说明</h1>"
            f'<pre class="guide-text">{escape(guide)}</pre></main></body></html>'
        ).encode()
        self._headers(HTTPStatus.OK, "text/html; charset=utf-8", len(page))
        self.wfile.write(page)

    def do_GET(self) -> None:
        if not self._host_allowed():
            self._error(HTTPStatus.FORBIDDEN, "Host 不允许")
            return
        parsed = urlsplit(self.path)
        path = parsed.path
        try:
            if path == "/":
                self._serve_static("index.html")
            elif path == "/help":
                self._serve_help()
            elif path in {
                "/static/app.js",
                "/static/style.css",
                "/static/diagnostics.css",
                "/static/help.css",
            }:
                self._serve_static(path.rsplit("/", 1)[-1])
            elif path in {"/api/system", "/api/system/status"}:
                version = lerobot_version(self.server.config.lerobot_python)
                self._json(
                    HTTPStatus.OK,
                    {
                        "host": self.server.config.host,
                        "port": self.server.server_port,
                        "state_dir": str(self.server.config.state_dir),
                        "uv_cache_dir": str(self.server.config.uv_cache_dir),
                        "lerobot_python": str(self.server.config.lerobot_python),
                        "hf_python": str(self.server.config.hf_python),
                        "lerobot_ready": version is not None,
                        "lerobot_distribution_version": version,
                        "hf_python_exists": self.server.config.hf_python.is_file(),
                        "default_hf_repo": self.server.config.default_hf_repo,
                        "default_hf_target": self.server.config.default_hf_target,
                        "hf_targets": [
                            {
                                "id": target.id,
                                "label": target.label,
                                "dataset_name": target.dataset_name,
                                "repo_id": target.repo_id,
                            }
                            for target in self.server.config.hf_targets
                        ],
                    },
                )
            elif path == "/api/runs":
                self._json(HTTPStatus.OK, {"runs": self.server.config.list_runs()})
            elif path.startswith("/api/runs/") and "/episodes/" in path:
                parts = path.strip("/").split("/")
                if (
                    len(parts) != 6
                    or parts[:2] != ["api", "runs"]
                    or parts[3] != "episodes"
                ):
                    self._error(HTTPStatus.NOT_FOUND, "路径不存在")
                    return
                run_name, episode_id, action = parts[2], parts[4], parts[5]
                run_paths = self.server.config.run(run_name)
                query = parse_qs(parsed.query)
                if action == "diagnostics":
                    reason = query.get("reason", [None])[0]
                    self._json(
                        HTTPStatus.OK,
                        episode_diagnostics(run_paths, episode_id, reason=reason),
                    )
                elif action == "comparison":
                    baseline = query.get("baseline", [None])[0]
                    if not baseline:
                        raise ValueError("comparison 需要 baseline Episode")
                    self._json(
                        HTTPStatus.OK,
                        episode_diagnostics_comparison(run_paths, episode_id, baseline),
                    )
                else:
                    self._error(HTTPStatus.NOT_FOUND, "路径不存在")
            elif path.startswith("/api/runs/") and path.count("/") == 3:
                name = path.rsplit("/", 1)[-1]
                self._json(HTTPStatus.OK, run_snapshot(self.server.config.run(name)))
            elif path == "/api/jobs":
                query = parse_qs(parsed.query)
                run_name = query.get("run", [None])[0]
                if run_name:
                    self.server.config.run(run_name)
                self._json(HTTPStatus.OK, {"jobs": self.server.runner.list(run_name)})
            elif path.startswith("/api/jobs/"):
                parts = path.split("/")
                if len(parts) not in {4, 5}:
                    self._error(HTTPStatus.NOT_FOUND, "路径不存在")
                    return
                job_id = parts[3]
                if len(parts) == 5 and parts[4] == "logs":
                    self._json(HTTPStatus.OK, {"logs": self.server.runner.logs(job_id)})
                elif len(parts) == 4:
                    job = self.server.runner.get(job_id)
                    if job is None:
                        self._error(HTTPStatus.NOT_FOUND, "Job 不存在")
                    else:
                        self._json(HTTPStatus.OK, job)
                else:
                    self._error(HTTPStatus.NOT_FOUND, "路径不存在")
            elif path == "/api/hf/status":
                repo_id = parse_qs(parsed.query).get(
                    "repo_id", [self.server.config.default_hf_repo]
                )[0]
                validate_repo_id(repo_id)
                if not self.server.config.is_allowed_hf_repo(repo_id):
                    self._error(HTTPStatus.FORBIDDEN, "该 HF 仓库不在允许的目标列表中")
                    return
                account = hf_call("whoami", {}, str(self.server.config.hf_python))
                state = hf_call(
                    "repo_state",
                    {"repo_id": repo_id},
                    str(self.server.config.hf_python),
                )
                self._json(
                    HTTPStatus.OK,
                    {
                        "repo_id": repo_id,
                        "account": account.get("account"),
                        "private": state.get("private"),
                        "sha": state.get("sha"),
                        "file_count": len(state.get("files", [])),
                    },
                )
            else:
                self._error(HTTPStatus.NOT_FOUND, "路径不存在")
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # noqa: BLE001 - do not leak worker output
            self._error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"读取失败：{type(exc).__name__}: {exc}",
            )

    def do_POST(self) -> None:
        if not self._host_allowed():
            self._error(HTTPStatus.FORBIDDEN, "Host 不允许")
            return
        origin = self.headers.get("Origin")
        if origin and origin not in {
            f"http://127.0.0.1:{self.server.server_port}",
            f"http://localhost:{self.server.server_port}",
        }:
            self._error(HTTPStatus.FORBIDDEN, "Origin 不允许")
            return
        if self.headers.get("X-UI-Token") != self.server.csrf:
            self._error(HTTPStatus.FORBIDDEN, "CSRF token 无效")
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
            self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "只接受 JSON")
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 65536:
                raise ValueError("请求大小无效")
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise TypeError("请求必须是对象")
            path = urlsplit(self.path).path
            if path.startswith("/api/jobs/") and path.endswith("/cancel"):
                job_id = path.split("/")[3]
                self._json(HTTPStatus.OK, self.server.runner.cancel(job_id))
            elif path.startswith("/api/jobs/") and path.count("/") == 3:
                stage = path.rsplit("/", 1)[-1]
                run_name = payload.get("run_name")
                if not isinstance(run_name, str):
                    raise ValueError("缺少 Run 名称")
                params = payload.get("params", {})
                if not isinstance(params, dict):
                    raise ValueError("params 必须是对象")
                if stage in {"hf-dry-run", "hf-publish"}:
                    repo_id = params.get("repo_id")
                    try:
                        validate_repo_id(repo_id if isinstance(repo_id, str) else "")
                    except ValueError as exc:
                        self._error(HTTPStatus.BAD_REQUEST, str(exc))
                        return
                    if not self.server.config.is_allowed_hf_repo(repo_id):
                        self._error(
                            HTTPStatus.FORBIDDEN,
                            "HF 上传目标必须从允许的数据集目标中选择",
                        )
                        return
                self._json(
                    HTTPStatus.ACCEPTED,
                    self.server.runner.submit(run_name, stage, params),
                )
            else:
                self._error(HTTPStatus.NOT_FOUND, "路径不存在")
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # noqa: BLE001 - persist failure in API response
            self._error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"提交失败：{type(exc).__name__}: {exc}",
            )
