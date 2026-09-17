"""Isolated UI job worker; delegates to the same public APIs as the CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from vla_data.batch.runner import build_curated_dataset, quality_dataset
from vla_data.export import export_lerobot, validate_lerobot_export
from vla_data.publish import publish_dataset
from vla_data.publish.remote import hf_call


def emit(kind: str, value: dict) -> None:
    print(
        "UI_EVENT " + json.dumps({"kind": kind, **value}, ensure_ascii=False),
        flush=True,
    )


def batch_evidence(result) -> dict:
    for row in result.results:
        emit(
            "log",
            {
                "message": f"{row.episode_id}: {row.status}"
                + (f" — {row.message}" if row.message else "")
            },
        )
    return result.summary


def execute(request: dict) -> dict:
    stage = request["stage"]
    paths = request["paths"]
    curated = Path(paths["curated"])
    quality = Path(paths["quality"])
    export = Path(paths["export"])
    raw = Path(paths["raw"])
    if stage in {"clean", "curate"}:
        emit("log", {"message": "开始 D2 物理/因果清洗"})
        d2 = build_curated_dataset(
            raw,
            curated,
            workers=request.get("workers", 4),
            force=request.get("force", False),
        )
        d2_summary = batch_evidence(d2)
        if d2.failed_count:
            raise RuntimeError(f"D2 有 {d2.failed_count} 个失败 episode；请查看报告")
        if stage == "curate":
            return {"d2": d2_summary}
    if stage in {"clean", "quality"}:
        emit("log", {"message": "开始 D3 数据质量检查"})
        d3 = quality_dataset(
            curated,
            quality,
            workers=request.get("workers", 4),
            force=request.get("force", False),
        )
        d3_summary = batch_evidence(d3)
        if d3.failed_count:
            raise RuntimeError(f"D3 有 {d3.failed_count} 个失败 episode；请查看报告")
        return {"d2": d2_summary if stage == "clean" else None, "d3": d3_summary}
    if stage == "export":
        emit("log", {"message": "开始 LeRobot v2.1 导出"})
        result = export_lerobot(
            None,
            export,
            curated_root=curated,
            quality_root=quality,
            dataset_name=request["dataset_name"],
            validation_fraction=request["validation_fraction"],
            split_seed=request["split_seed"],
            lerobot_python=request["lerobot_python"],
            force=request.get("force", False),
        )
        return {"export": result}
    if stage == "validate":
        emit("log", {"message": "开始本地独立验证和官方 LeRobot 回读"})
        result = validate_lerobot_export(
            export, lerobot_python=request["lerobot_python"]
        )
        if not result.passed:
            raise RuntimeError("本地验证失败：" + "; ".join(result.errors))
        return {"validation": {"passed": True, "evidence": result.evidence}}
    if stage == "hf-dry-run":
        emit("log", {"message": "读取私有 HF HEAD 并计算累计计划（不会上传）"})
        result = publish_dataset(
            export / "train",
            request["repo_id"],
            hf_python=request["hf_python"],
            lerobot_python=request["lerobot_python"],
            dry_run=True,
        )
        if result["action"] == "BLOCKED":
            raise RuntimeError(
                "HF Dry Run 被阻断：" + result.get("reason", result["would_action"])
            )
        return {"dry_run": result}
    if stage == "hf-publish":
        state = hf_call(
            "repo_state", {"repo_id": request["repo_id"]}, request["hf_python"]
        )
        if (
            state.get("private") is not True
            or state.get("sha") != request["baseline_sha"]
        ):
            raise RuntimeError("HF HEAD 或私有状态已变化，请重新执行 Dry Run")
        emit("log", {"message": "开始累计发布；将自动进行远端固定 revision 验收"})
        result = publish_dataset(
            export / "train",
            request["repo_id"],
            hf_python=request["hf_python"],
            lerobot_python=request["lerobot_python"],
        )
        if result["action"] != "UPLOADED":
            raise RuntimeError("发布未产生新 revision：" + result["action"])
        return {"publish": result}
    raise ValueError("未知的 UI job stage")


def main() -> int:
    try:
        request = json.loads(sys.stdin.read())
        result = execute(request)
    except Exception as exc:  # noqa: BLE001 - report job failure without traceback/secrets
        emit("result", {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    emit("result", {"ok": True, "summary": result})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
