"""Operator translations keep diagnostics intact without changing pipeline codes."""

from pathlib import Path

import pytest

from vla_data.contracts.curated_v1 import INVALID_REASONS
from vla_data.ui.config import load_config
from vla_data.ui.readers import run_snapshot
from vla_data.ui.reasons import translate_reason


@pytest.mark.parametrize(
    ("code", "summary"),
    [
        ("MISSING_MEASURED_HAND_FEEDBACK", "缺少灵巧手实测反馈"),
        ("MISSING_ARM_FEEDBACK", "缺少机械臂实测反馈"),
        ("MISSING_HEAD_CAMERA", "缺少头部相机数据"),
        ("MISSING_WRIST_CAMERA", "缺少腕部相机数据"),
        ("NO_CLEAN_TRANSITION", "没有可用于训练的有效数据区间"),
        ("INVALID_CAUSAL_ORDER", "状态和动作的时间顺序异常"),
        ("STALE_ACTION_COMMAND", "动作命令时间过期或不同步"),
        ("TASK_MISSING", "缺少任务语言指令"),
        ("INVALID_STATE_DIMENSION", "状态向量维度不符合要求"),
        ("INVALID_ACTION_DIMENSION", "动作向量维度不符合要求"),
        ("SYNTHETIC_SOURCE", "当前数据属于测试或合成数据"),
    ],
)
def test_operator_reason_aliases(code: str, summary: str) -> None:
    result = translate_reason(code)
    assert result == {"summary": summary, "technical_code": code, "original": code}


def test_all_curated_validity_bits_translated() -> None:
    for code in INVALID_REASONS:
        result = translate_reason(code)
        assert result["summary"] != "未知数据问题", code
        assert result["original"] == code


@pytest.mark.parametrize(
    "code",
    [
        "CAUSALITY_ARM_PRE_UNAVAILABLE",
        "CAUSALITY_HAND_POST_UNAVAILABLE",
        "FEEDBACK_ARM_PRE_INVALID",
        "FEEDBACK_HAND_POST_INVALID",
        "CAMERA_HEAD_CAMERA_UNAVAILABLE",
        "CAMERA_WRIST_CAMERA_MEDIA_MISSING",
        "CAMERA_HEAD_CAMERA_DECODE_INVALID",
        "JPEG_DECODE_FAILURE",
        "MISSING_MEDIA_DIRECTORY",
        "D3_NOT_CLEAN",
        "SYNTHETIC_TEST_ONLY",
        "NO_CLEAN_CONTIGUOUS_RUN",
        "ANNOTATION_CAMERA_VIEW_MISSING",
        "MCP_INVALID_RESPONSE",
    ],
)
def test_current_dynamic_and_diagnostic_codes_translated(code: str) -> None:
    assert translate_reason(code)["summary"] != "未知数据问题"


def test_unknown_reason_fallback_preserves_evidence() -> None:
    result = translate_reason("NEW_UNKNOWN_CODE: engineer detail")
    assert result == {
        "summary": "未知数据问题",
        "technical_code": "NEW_UNKNOWN_CODE",
        "original": "NEW_UNKNOWN_CODE: engineer detail",
    }
    assert translate_reason("unrecognized free text")["summary"] == "未知数据问题"


def test_real_d3_warning_pattern() -> None:
    result = translate_reason(
        "hand state velocity skipped 52 pairs with non-positive dt"
    )
    assert result["summary"] == "灵巧手状态速度检查跳过 52 对无效时间间隔"
    assert (
        result["original"]
        == "hand state velocity skipped 52 pairs with non-positive dt"
    )


def test_historical_003_006_displayed_under_single_read_only_run() -> None:
    config_file = Path(__file__).resolve().parents[2] / "config" / "ui.toml"
    config = load_config(config_file)
    if "batch4_acceptance" not in config.list_runs():
        pytest.skip("历史验收工件在当前机器不可用")
    snapshot = run_snapshot(config.run("batch4_acceptance"))
    assert snapshot["read_only"] is True
    assert snapshot["run"] == "batch4_acceptance"
    assert [row["episode_id"] for row in snapshot["episodes"]] == [
        f"episode_{index:06d}" for index in range(3, 7)
    ]
    assert all(
        reason["summary"] != "未知数据问题"
        for row in snapshot["episodes"]
        for reason in row["d3_reasons"]
    )
