"""Read-only projections of existing RAW, D2, D3, and LeRobot artifacts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from vla_data.diagnostics.d2_episode import (
    compare_episode_diagnostics,
    diagnose_episode,
)
from vla_data.ui.config import EPISODE_ID, RunPaths, inside
from vla_data.ui.diagnostic_labels import (
    localize_diagnostics_comparison,
    localize_episode_diagnostics,
)
from vla_data.ui.reasons import translate_reason


def _json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _rows(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line]
    except (OSError, ValueError):
        return []


def raw_preview(paths: RunPaths) -> dict:
    root = paths.raw_root
    ids = paths.episode_ids
    episodes = (
        [root / f"{episode_id}.npz" for episode_id in ids]
        if ids
        else sorted(root.glob("episode_*.npz"))
        if root.is_dir()
        else []
    )
    tasks: Counter[str] = Counter()
    details = []
    files = 0
    bytes_total = 0
    for episode in episodes:
        if not episode.resolve().is_relative_to(root.resolve()):
            details.append(
                {"episode_id": episode.stem, "error": "RAW 路径超出允许范围"}
            )
            continue
        if not episode.is_file():
            details.append({"episode_id": episode.stem, "error": "RAW 文件不存在"})
            continue
        try:
            with np.load(episode, allow_pickle=False) as archive:
                value = archive.get("language_instruction", None)
                task = (
                    str(value.item()) if value is not None and value.ndim == 0 else None
                )
                cameras = {
                    role: f"{role}_camera_present" in archive.files
                    and bool(archive[f"{role}_camera_present"].item())
                    for role in ("head", "wrist")
                }
        except (OSError, ValueError, KeyError) as exc:
            task, cameras = None, {"head": False, "wrist": False}
            details.append({"episode_id": episode.stem, "error": type(exc).__name__})
        else:
            details.append(
                {"episode_id": episode.stem, "task": task, "cameras": cameras}
            )
        if task:
            tasks[task] += 1
        files += 1
        bytes_total += episode.stat().st_size
        media = root / f"{episode.stem}_media"
        if media.is_dir() and media.resolve().is_relative_to(root.resolve()):
            for item in media.rglob("*"):
                if item.is_file() and item.resolve().is_relative_to(root.resolve()):
                    files += 1
                    bytes_total += item.stat().st_size
    return {
        "root": str(root),
        "episodes": len(episodes),
        "files": files,
        "bytes": bytes_total,
        "tasks": [{"task": task, "episodes": count} for task, count in tasks.items()],
        "details": details,
    }


def run_snapshot(paths: RunPaths) -> dict:
    raw = raw_preview(paths)
    build = _json(paths.curated_root / "dataset_build_summary.json")
    quality = _json(paths.quality_root / "dataset_quality_summary.json")
    export = _json(paths.export_root / "export_summary.json")
    validation = _json(paths.export_root / "export_validation.json")
    provenance = _rows(paths.export_root / "export_provenance.jsonl")
    build_rows = {r.get("episode_id"): r for r in (build or {}).get("results", [])}
    quality_rows = {r.get("episode_id"): r for r in (quality or {}).get("results", [])}
    episode_ids = sorted(
        set(build_rows) | set(quality_rows) | {r["episode_id"] for r in raw["details"]}
    )
    episodes = []
    for episode_id in episode_ids:
        d2 = build_rows.get(episode_id, {})
        d3 = quality_rows.get(episode_id, {})
        report = _json(paths.quality_root / episode_id / "quality_report.json") or {}
        d2_raw = d2.get("message")
        d3_report_reasons = report.get("exclusion_reasons", []) + report.get(
            "warnings", []
        )
        d3_raw = d3.get("message") or "; ".join(d3_report_reasons)
        d3_reasons = d3_report_reasons or (d3_raw.split("; ") if d3_raw else [])
        episodes.append(
            {
                "episode_id": episode_id,
                "d2": d2.get("status", "PENDING"),
                "d2_reason": d2_raw,
                "d2_reasons": [
                    translate_reason(value)
                    for value in d2_raw.split("; ")
                    if value.strip()
                ]
                if d2_raw
                else [],
                "d3": d3.get("outcome", d3.get("status", "PENDING")),
                "d3_reason": d3_raw,
                "d3_reasons": [translate_reason(value) for value in d3_reasons],
                "transitions": d3.get(
                    "transitions_total", report.get("transition_count")
                ),
                "clean_transitions": report.get(
                    "clean_transition_count", d3.get("transitions_valid")
                ),
            }
        )
    return {
        "run": paths.name,
        "read_only": paths.read_only,
        "paths": {
            "raw": str(paths.raw_root),
            "work": str(paths.work_root),
            "curated": str(paths.curated_root),
            "quality": str(paths.quality_root),
            "export": str(paths.export_root),
        },
        "raw": raw,
        "d2": {
            "present": build is not None,
            "discovered": (build or {}).get("episodes_discovered", 0),
            "valid": (build or {}).get("episodes_succeeded", 0),
            "rejected": (build or {}).get("episodes_failed", 0),
        },
        "d3": {
            "present": quality is not None,
            "counts": (quality or {}).get("eligibility_counts", {}),
            "clean_transitions": sum(
                int(r.get("clean_transitions") or 0) for r in episodes
            ),
            "rejected_transitions": sum(
                max(
                    0,
                    int(r.get("transitions") or 0)
                    - int(r.get("clean_transitions") or 0),
                )
                for r in episodes
            ),
        },
        "episodes": episodes,
        "export": {
            "present": export is not None,
            "summary": export,
            "validation_record": validation,
            "runs": [
                {
                    "source_episode_id": row.get("source_episode_id"),
                    "start": row.get("source_start_index"),
                    "end": row.get("source_end_index"),
                    "split": row.get("split"),
                    "task": row.get("task_instruction"),
                }
                for row in provenance
            ],
        },
    }


def episode_diagnostics(
    paths: RunPaths, episode_id: str, *, reason: str | None = None
) -> dict:
    """Load one Episode's diagnostics without changing the Run snapshot contract."""

    raw_path, status = _diagnostic_episode(paths, episode_id)
    report_path = paths.curated_root / episode_id / "cleaning_report.json"
    return localize_episode_diagnostics(
        diagnose_episode(
            run_name=paths.name,
            raw_path=raw_path,
            cleaning_report_path=report_path,
            d2_status=status,
            reason=reason,
        )
    )


def episode_diagnostics_comparison(
    paths: RunPaths, episode_id: str, baseline_id: str
) -> dict:
    """Compare two Episodes from the same allowlisted Run."""

    current = episode_diagnostics(paths, episode_id)
    baseline = episode_diagnostics(paths, baseline_id)
    return localize_diagnostics_comparison(
        compare_episode_diagnostics(current, baseline)
    )


def _diagnostic_episode(paths: RunPaths, episode_id: str) -> tuple[Path, str | None]:
    if not EPISODE_ID.fullmatch(episode_id):
        raise ValueError("无效的 Episode ID")
    if paths.episode_ids and episode_id not in paths.episode_ids:
        raise ValueError("Episode 不属于当前 Run")
    raw_path = (paths.raw_root / f"{episode_id}.npz").resolve()
    if not inside(raw_path, (paths.raw_root,)) or not raw_path.is_file():
        raise ValueError("Episode RAW 不存在或超出允许范围")
    build = _json(paths.curated_root / "dataset_build_summary.json") or {}
    status = next(
        (
            str(row.get("status"))
            for row in build.get("results", [])
            if row.get("episode_id") == episode_id and row.get("status")
        ),
        None,
    )
    return raw_path, status
