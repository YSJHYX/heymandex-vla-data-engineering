"""Chinese diagnostics presentation preserves canonical evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from vla_data.diagnostics.d2_episode import clear_diagnostics_cache, diagnose_episode
from vla_data.ui.diagnostic_labels import (
    COMPARISON_CHANGE_LABELS,
    COMPARISON_LABELS,
    D2_GROUP_LABELS,
    D2_STATUS_LABELS,
    HEALTH_STATUS_LABELS,
    ROOT_CAUSE_LABELS,
    SEVERITY_LABELS,
    UI_ACTION_LABELS,
    UNKNOWN_DIAGNOSTIC_LABEL,
    diagnostic_reason_label,
    label_for,
    localize_diagnostics_comparison,
    localize_episode_diagnostics,
)

REAL_RAW = Path("/data/vla_runs/biyi_real_test_dataset/raw/episode_000007.npz")


def test_translation_tables_cover_every_supported_canonical_value() -> None:
    assert set(HEALTH_STATUS_LABELS) == {
        "HEALTHY",
        "DEGRADED",
        "FROZEN",
        "STALE",
        "MISSING",
        "CLOCK_MISMATCH",
        "UNKNOWN",
    }
    assert set(SEVERITY_LABELS) == {"INFO", "WARNING", "CRITICAL", "UNKNOWN"}
    assert {"FAILED", "PASSED", "SKIPPED", "WARNING"} <= set(D2_STATUS_LABELS)
    assert set(D2_GROUP_LABELS) == {"ACTION", "FEEDBACK", "CAUSALITY", "CAMERA"}
    assert {
        "HEAD_CAMERA_FROZEN",
        "HEAD_CAMERA_MISSING",
        "HEAD_CAMERA_STALE",
        "HEAD_CAMERA_CLOCK_MISMATCH",
        "WRIST_CAMERA_FROZEN",
        "WRIST_CAMERA_MISSING",
        "WRIST_CAMERA_STALE",
        "WRIST_CAMERA_CLOCK_MISMATCH",
        "UNKNOWN",
    } == set(ROOT_CAUSE_LABELS)
    assert {
        "VIEW_DIAGNOSTICS",
        "COPY_REPORT",
        "COMPARE",
        "CLOSE",
        "LOADING",
        "LOAD_FAILED",
        "NO_EVIDENCE",
        "NO_REJECTED_TRANSITIONS",
    } == set(UI_ACTION_LABELS)
    for mapping in (
        HEALTH_STATUS_LABELS,
        SEVERITY_LABELS,
        D2_STATUS_LABELS,
        D2_GROUP_LABELS,
        ROOT_CAUSE_LABELS,
        UI_ACTION_LABELS,
        COMPARISON_LABELS,
        COMPARISON_CHANGE_LABELS,
    ):
        assert all(
            any("\u4e00" <= char <= "\u9fff" for char in label)
            for label in mapping.values()
        )


def test_unknown_diagnostic_fallback_does_not_promote_code_to_ui_label() -> None:
    technical_code = "NEW_UNKNOWN_CODE"
    assert label_for(ROOT_CAUSE_LABELS, technical_code) == UNKNOWN_DIAGNOSTIC_LABEL
    assert diagnostic_reason_label(technical_code) == UNKNOWN_DIAGNOSTIC_LABEL
    assert technical_code == "NEW_UNKNOWN_CODE"


def test_comparison_labels_are_chinese_and_keep_technical_label() -> None:
    result = localize_diagnostics_comparison(
        {
            "rows": [
                {
                    "label": "HEAD valid",
                    "kind": "ratio",
                    "baseline": 1.0,
                    "current": 0.0,
                    "regression": True,
                }
            ]
        }
    )
    row = result["rows"][0]
    assert row["label"] == "头部相机有效率"
    assert row["technical_label"] == "HEAD valid"
    assert row["change_label"] == "指标下降"


def test_synthetic_diagnostic_projection_has_chinese_display_copy(
    synthetic_raw_episode: Path,
) -> None:
    clear_diagnostics_cache()
    canonical = diagnose_episode(
        run_name="synthetic", raw_path=synthetic_raw_episode, d2_status="FAILED"
    )
    result = localize_episode_diagnostics(canonical)

    assert result["d2_status"] == "FAILED"
    assert result["d2_evidence"]["status_label"] == "失败"
    assert result["raw_diagnostics"]["camera_health"]["head"]["status_label"]
    assert result["raw_diagnostics"]["inferred_root_cause"]["severity_label"]
    assert "[D2 数据诊断报告]" in result["diagnostic_report_text"]
    assert "Diagnostic Report" not in result["diagnostic_report_text"]


@pytest.mark.skipif(not REAL_RAW.is_file(), reason="real episode_000007 unavailable")
def test_real_episode_000007_chinese_ui_projection() -> None:
    clear_diagnostics_cache()
    canonical = diagnose_episode(
        run_name="biyi_real_test_dataset",
        raw_path=REAL_RAW,
        d2_status="FAILED",
    )
    result = localize_episode_diagnostics(canonical)
    d2 = result["d2_evidence"]
    raw = result["raw_diagnostics"]
    cause = raw["inferred_root_cause"]

    # Canonical contract remains intact.
    assert result["d2_status"] == "FAILED"
    assert d2["primary_failure_reason"] == "CAMERA_HEAD_CAMERA_UNAVAILABLE"
    assert raw["camera_health"]["head"]["status"] == "FROZEN"
    assert cause["code"] == "HEAD_CAMERA_FROZEN"
    assert cause["severity"] == "CRITICAL"

    # Chinese labels are the primary UI copy.
    assert d2["status_label"] == "失败"
    assert d2["primary_failure_group_label"] == "相机"
    assert d2["primary_failure_reason_label"] == "头部相机不可用"
    assert raw["camera_health"]["head"]["status_label"] == "冻结"
    assert raw["camera_health"]["wrist"]["status_label"] == "正常"
    assert cause["label"] == "头部相机冻结"
    assert cause["severity_label"] == "严重"
    assert cause["suggested_subsystem"] == "D455 / 头部相机数据生产链路"
    assert raw["why_episode_rejected"] == [
        "共发现 1667 个候选 Transition。",
        "其中 1048 个未通过动作有效性检查。",
        "剩余 619 个通过了动作有效性检查。",
        "这 619 个 Transition 均需要可用的头部相机观测。",
        "当前 Episode 中不存在可用的头部相机观测。",
        "头部相机时间戳在整个 Episode 内未继续更新。",
        "因此最终没有任何 Transition 可以进入 Curated 数据。",
    ]
    report = result["diagnostic_report_text"]
    assert "[D2 数据诊断报告]" in report
    assert "D2 状态：失败" in report
    assert "D2 状态：失败（FAILED）" not in report
    assert "主要根因：头部相机冻结" in report
    assert "时间戳中位延迟：1003.71 秒" in report
    assert "不应通过降低 D2 清洗要求来绕过该问题。" in report
    for forbidden in (
        "Diagnostic Report",
        "Primary Root Cause",
        "Suggested subsystem",
        "Suggested checks",
        "Therefore no transition",
        "timestamp stopped advancing",
    ):
        assert forbidden not in report
