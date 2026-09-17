"""Persistent, single-writer UI job runner for the frozen pipeline APIs."""

from __future__ import annotations

import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from vla_data.publish.local import build_canonical_manifest, validate_repo_id
from vla_data.ui.config import DATASET_NAME, UIConfig

STAGES = {
    "clean",
    "curate",
    "quality",
    "export",
    "validate",
    "hf-dry-run",
    "hf-publish",
}
MUTATING = {"clean", "curate", "quality", "export", "hf-publish"}
TOKEN = re.compile(r"hf_[A-Za-z0-9]{10,}")
BEARER = re.compile(r"(?i)Bearer\s+[^\s\"']+")
AUTH = re.compile(r"(?i)(Authorization\s*[:=]\s*)[^\s\"']+")
SECRET = re.compile(r"(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|API_KEY))\s*=\s*[^\s]+")


def sanitize(value: str) -> str:
    value = TOKEN.sub("[REDACTED]", value)
    value = BEARER.sub("Bearer [REDACTED]", value)
    value = AUTH.sub(r"\1[REDACTED]", value)
    return SECRET.sub(r"\1=[REDACTED]", value)


def redact_data(value):
    """Redact secrets without destroying the worker's structured JSON result."""
    if isinstance(value, str):
        return sanitize(value)
    if isinstance(value, list):
        return [redact_data(item) for item in value]
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if SECRET.fullmatch(str(key)) else redact_data(item)
            for key, item in value.items()
        }
    return value


def now() -> str:
    return datetime.now(UTC).isoformat()


class PipelineJobRunner:
    def __init__(self, config: UIConfig) -> None:
        self.config = config
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.config.state_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        self.database = self.config.state_dir / "jobs.sqlite3"
        self._lock = threading.RLock()
        self._processes: dict[str, subprocess.Popen] = {}
        with self._connection() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY, run_name TEXT NOT NULL,
                    stage TEXT NOT NULL, status TEXT NOT NULL,
                    start_time TEXT, end_time TEXT, command TEXT NOT NULL,
                    return_code INTEGER, summary TEXT, error TEXT,
                    context TEXT NOT NULL, log_path TEXT NOT NULL
                )"""
            )
            db.execute(
                "UPDATE jobs SET status='FAILED', end_time=?, "
                "error='UI 服务重启；此前 job 结果未确认' "
                "WHERE status IN ('PENDING','RUNNING')",
                (now(),),
            )

    def _connection(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def _fingerprints(self, export: Path) -> dict:
        try:
            summary = json.loads((export / "export_summary.json").read_text())
            manifest = build_canonical_manifest(export / "train")
            return {
                "source_export_fingerprint": summary["fingerprint"],
                "local_fingerprint": manifest["fingerprint"],
            }
        except (OSError, ValueError, KeyError):
            return {}

    def _latest(self, run_name: str, stage: str) -> dict | None:
        with self._connection() as db:
            row = db.execute(
                "SELECT * FROM jobs WHERE run_name=? AND stage=? ORDER BY rowid DESC LIMIT 1",
                (run_name, stage),
            ).fetchone()
        return self._row(row) if row else None

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        value = dict(row)
        for key in ("command", "summary", "context"):
            value[key] = json.loads(value[key]) if value[key] else None
        return value

    def get(self, job_id: str) -> dict | None:
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            return None
        with self._connection() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return self._row(row) if row else None

    def list(self, run_name: str | None = None, limit: int = 50) -> list[dict]:
        with self._connection() as db:
            if run_name is None:
                rows = db.execute(
                    "SELECT * FROM jobs ORDER BY rowid DESC LIMIT ?", (min(limit, 100),)
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT * FROM jobs WHERE run_name=? ORDER BY rowid DESC LIMIT ?",
                    (run_name, min(limit, 100)),
                ).fetchall()
        return [self._row(row) for row in rows]

    def logs(self, job_id: str) -> str:
        job = self.get(job_id)
        if job is None:
            raise ValueError("Job 不存在")
        path = self.logs_dir / f"{job_id}.log"
        if not path.is_file():
            return ""
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 100_000))
            return stream.read().decode("utf-8", errors="replace")

    def submit(self, run_name: str, stage: str, params: dict | None = None) -> dict:
        params = params or {}
        if stage not in STAGES:
            raise ValueError("未知的处理阶段")
        paths = self.config.run(run_name)
        if paths.read_only and stage in MUTATING:
            raise ValueError("该历史验收 Run 为只读，不能重新处理或上传")
        if stage in {"clean", "curate"} and not paths.raw_root.is_dir():
            raise ValueError("RAW 数据目录不存在")
        if stage in {"quality", "export"} and not paths.curated_root.is_dir():
            raise ValueError("尚无 Curated 数据，请先执行数据清洗")
        if stage == "export" and not paths.quality_root.is_dir():
            raise ValueError("尚无 D3 质量结果，请先执行数据清洗")
        if (
            stage in {"validate", "hf-dry-run", "hf-publish"}
            and not (paths.export_root / "export_summary.json").is_file()
        ):
            raise ValueError("尚无 LeRobot 导出")
        force = bool(params.get("force", False))
        if force and stage not in {"clean", "curate", "quality", "export"}:
            raise ValueError("该阶段不允许强制重跑")
        if force and params.get("rerun_confirmed") is not True:
            raise ValueError("重新运行前必须二次确认")
        dataset_name = str(params.get("dataset_name", f"rm65_sg100_{run_name}"))
        if stage == "export" and not DATASET_NAME.fullmatch(dataset_name):
            raise ValueError("数据集名称只允许字母、数字、下划线、短横线和点")
        fraction = params.get("validation_fraction", 0.1)
        seed = params.get("split_seed", 17)
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (float, int))
            or not 0 <= fraction <= 1
        ):
            raise ValueError("Validation fraction 必须在 0 到 1 之间")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("Split seed 必须为整数")
        repo_id = str(params.get("repo_id", self.config.default_hf_repo))
        validate_repo_id(repo_id)
        fingerprint = self._fingerprints(paths.export_root)
        if stage == "hf-dry-run":
            validation = self._latest(run_name, "validate")
            if (
                validation is None
                or validation["status"] != "PASS"
                or validation["context"].get("fingerprints") != fingerprint
            ):
                raise ValueError("请先对当前 LeRobot 导出执行本地验证")
        baseline_sha = None
        if stage == "hf-publish":
            if params.get("confirmation") != "我已检查 Dry Run 结果":
                raise ValueError("正式上传需要确认 Dry Run 结果")
            dry = self.get(str(params.get("dry_run_job_id", "")))
            newest_dry = self._latest(run_name, "hf-dry-run")
            if (
                dry is None
                or newest_dry is None
                or newest_dry["job_id"] != dry["job_id"]
                or dry["run_name"] != run_name
                or dry["stage"] != "hf-dry-run"
                or dry["status"] != "PASS"
            ):
                raise ValueError("没有可用于本次上传的成功 Dry Run")
            evidence = (dry["summary"] or {}).get("dry_run", {})
            plan = evidence.get("merge_plan", {})
            if not plan.get("to_append", 0):
                raise ValueError("当前批次已全部存在于 HF，无需重复上传")
            if dry["context"].get("fingerprints") != fingerprint:
                raise ValueError("本地导出已变化，请重新执行验证和 Dry Run")
            if dry["context"].get("repo_id") != repo_id:
                raise ValueError("HF 仓库已变化，请重新执行 Dry Run")
            validation = self._latest(run_name, "validate")
            if (
                validation is None
                or validation["status"] != "PASS"
                or validation["context"].get("fingerprints") != fingerprint
            ):
                raise ValueError("本地验证已过期")
            baseline_sha = evidence.get("remote_before", {}).get("sha")
            if not baseline_sha:
                raise ValueError("Dry Run 缺少远端 baseline SHA")
        request = {
            "stage": stage,
            "paths": {
                "raw": str(paths.raw_root),
                "curated": str(paths.curated_root),
                "quality": str(paths.quality_root),
                "export": str(paths.export_root),
            },
            "workers": 4,
            "force": force,
            "dataset_name": dataset_name,
            "validation_fraction": float(fraction),
            "split_seed": seed,
            "repo_id": repo_id,
            "lerobot_python": str(self.config.lerobot_python),
            "hf_python": str(self.config.hf_python),
            "baseline_sha": baseline_sha,
        }
        context = {"fingerprints": fingerprint, "repo_id": repo_id}
        job_id = uuid.uuid4().hex
        log_path = self.logs_dir / f"{job_id}.log"
        with self._lock:
            with self._connection() as db:
                active = db.execute(
                    "SELECT job_id FROM jobs WHERE status IN ('PENDING','RUNNING') LIMIT 1"
                ).fetchone()
                if active:
                    raise ValueError("已有任务正在运行，请等待其完成")
                db.execute(
                    "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        job_id,
                        run_name,
                        stage,
                        "PENDING",
                        now(),
                        None,
                        json.dumps([sys.executable, "-m", "vla_data.ui.worker", stage]),
                        None,
                        None,
                        None,
                        json.dumps(context),
                        str(log_path),
                    ),
                )
            thread = threading.Thread(
                target=self._execute,
                args=(job_id, request, paths.export_root),
                daemon=True,
            )
            thread.start()
        return self.get(job_id)

    def _update(self, job_id: str, **fields) -> None:
        values = list(fields.values()) + [job_id]
        with self._connection() as db:
            db.execute(
                "UPDATE jobs SET "
                + ",".join(f"{key}=?" for key in fields)
                + " WHERE job_id=?",
                values,
            )

    def _execute(self, job_id: str, request: dict, export_root: Path) -> None:
        output = None
        code = -1
        error = None
        summary = None
        try:
            self._update(job_id, status="RUNNING")
            environment = os.environ.copy()
            environment["UV_CACHE_DIR"] = str(self.config.uv_cache_dir)
            process = subprocess.Popen(
                [sys.executable, "-u", "-m", "vla_data.ui.worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                shell=False,
                start_new_session=True,
                env=environment,
            )
            with self._lock:
                self._processes[job_id] = process
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(json.dumps(request))
            process.stdin.close()
            with (self.logs_dir / f"{job_id}.log").open("w", encoding="utf-8") as log:
                for line in process.stdout:
                    raw = line.rstrip("\n")
                    if raw.startswith("UI_EVENT "):
                        try:
                            event = json.loads(raw.removeprefix("UI_EVENT "))
                        except ValueError:
                            event = None
                        if event and event.get("kind") == "result":
                            if event.get("ok"):
                                summary = redact_data(event.get("summary"))
                            else:
                                error = sanitize(str(event.get("error", "处理失败")))[
                                    :4000
                                ]
                        elif event and event.get("kind") == "log":
                            log.write(
                                sanitize(str(event.get("message", "")))[:4000] + "\n"
                            )
                    else:
                        log.write(sanitize(raw)[:4000] + "\n")
                    log.flush()
            code = process.wait()
            output = summary
            if code or summary is None:
                error = error or f"处理失败（退出码 {code}），请查看详细日志"
        except Exception as exc:  # noqa: BLE001 - persist failure of runner itself
            error = sanitize(f"UI job 执行异常：{type(exc).__name__}: {exc}")[:4000]
        finally:
            with self._lock:
                self._processes.pop(job_id, None)
                current = self.get(job_id)
                if current and current["status"] == "CANCELLED":
                    pass
                elif error:
                    self._update(
                        job_id,
                        status="FAILED",
                        end_time=now(),
                        return_code=code,
                        error=error,
                    )
                else:
                    if request["stage"] == "validate":
                        context = current["context"] if current else {}
                        context["fingerprints"] = self._fingerprints(export_root)
                        self._update(job_id, context=json.dumps(context))
                    warning = (
                        request["stage"] in {"clean", "quality"}
                        and (output or {})
                        .get("d3", {})
                        .get("eligibility_counts", {})
                        .get("ACCEPT_WITH_WARNING", 0)
                        > 0
                    )
                    self._update(
                        job_id,
                        status="WARNING" if warning else "PASS",
                        end_time=now(),
                        return_code=code,
                        summary=json.dumps(output, ensure_ascii=False),
                    )

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            job = self.get(job_id)
            if job is None:
                raise ValueError("Job 不存在")
            if job["stage"] == "hf-publish":
                raise ValueError("上传期间不能取消；请等待远端验收结果")
            if job["status"] not in {"PENDING", "RUNNING"}:
                raise ValueError("该任务已结束")
            process = self._processes.get(job_id)
            if process:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            self._update(
                job_id, status="CANCELLED", end_time=now(), error="操作人员取消"
            )
            return self.get(job_id)
