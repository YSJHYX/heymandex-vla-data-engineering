"""Centralized Chinese presentation labels for read-only D2 diagnostics.

Canonical diagnostic values and API keys remain unchanged.  This module adds
operator-facing labels and replaces only natural-language presentation text.
"""

from __future__ import annotations

import copy
from typing import Any

from vla_data.ui.reasons import translate_reason

UNKNOWN_DIAGNOSTIC_LABEL = "未知诊断类型"

HEALTH_STATUS_LABELS = {
    "HEALTHY": "正常",
    "DEGRADED": "性能下降",
    "FROZEN": "冻结",
    "STALE": "数据陈旧",
    "MISSING": "数据缺失",
    "CLOCK_MISMATCH": "时钟异常",
    "UNKNOWN": "未知",
}

SEVERITY_LABELS = {
    "INFO": "信息",
    "WARNING": "警告",
    "CRITICAL": "严重",
    "UNKNOWN": "未知",
}

D2_STATUS_LABELS = {
    "FAILED": "失败",
    "FAIL": "失败",
    "SUCCESS": "通过",
    "PASSED": "通过",
    "PASS": "通过",
    "SKIPPED": "已跳过",
    "WARNING": "警告",
    "PENDING": "待处理",
    "UNKNOWN": "未知",
}

ROOT_CAUSE_LABELS = {
    "HEAD_CAMERA_FROZEN": "头部相机冻结",
    "HEAD_CAMERA_MISSING": "头部相机数据缺失",
    "HEAD_CAMERA_STALE": "头部相机数据陈旧",
    "HEAD_CAMERA_CLOCK_MISMATCH": "头部相机时钟异常",
    "WRIST_CAMERA_FROZEN": "腕部相机冻结",
    "WRIST_CAMERA_MISSING": "腕部相机数据缺失",
    "WRIST_CAMERA_STALE": "腕部相机数据陈旧",
    "WRIST_CAMERA_CLOCK_MISMATCH": "腕部相机时钟异常",
    "UNKNOWN": "未知",
}

D2_GROUP_LABELS = {
    "ACTION": "动作",
    "FEEDBACK": "反馈",
    "CAUSALITY": "因果时序",
    "CAMERA": "相机",
}

SUBSYSTEM_LABELS = {
    "HEAD_CAMERA": "D455 / 头部相机数据生产链路",
    "WRIST_CAMERA": "腕部相机数据生产链路",
    "UNKNOWN": "未知子系统",
}

UI_ACTION_LABELS = {
    "VIEW_DIAGNOSTICS": "查看 D2 诊断",
    "COPY_REPORT": "复制诊断报告",
    "COMPARE": "对比",
    "CLOSE": "关闭",
    "LOADING": "正在加载诊断信息……",
    "LOAD_FAILED": "诊断信息加载失败",
    "NO_EVIDENCE": "暂无可用诊断证据",
    "NO_REJECTED_TRANSITIONS": "没有被拒绝的 Transition",
}

COMPARISON_LABELS = {
    "HEAD valid": "头部相机有效率",
    "HEAD unique timestamps": "头部相机唯一时间戳数量",
    "HEAD median age": "头部相机时间戳中位延迟",
    "WRIST valid": "腕部相机有效率",
    "WRIST median age": "腕部相机时间戳中位延迟",
    "ACTION READY": "可用于 D2 的动作",
    "D2 accepted": "D2 接受数量",
}

COMPARISON_CHANGE_LABELS = {
    "REGRESSION": "指标下降",
    "IMPROVEMENT": "指标改善",
    "NO_SIGNIFICANT_CHANGE": "无明显变化",
}


def label_for(mapping: dict[str, str], canonical: str | None) -> str:
    """Return a Chinese label without exposing a new code as primary copy."""

    return mapping.get(str(canonical), UNKNOWN_DIAGNOSTIC_LABEL)


def localize_episode_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    """Add Chinese UI labels and presentation copy to an Episode diagnostic."""

    localized = copy.deepcopy(result)
    d2 = localized["d2_evidence"]
    raw = localized["raw_diagnostics"]
    cameras = raw["camera_health"]
    cause = raw["inferred_root_cause"]

    reason_codes = set(d2["rejection_counts"])
    reason_labels = {code: diagnostic_reason_label(code) for code in reason_codes}
    d2["reason_labels"] = reason_labels
    d2["group_labels"] = D2_GROUP_LABELS.copy()
    d2["status_label"] = label_for(D2_STATUS_LABELS, d2["status"])
    d2["primary_failure_group_label"] = label_for(
        D2_GROUP_LABELS, d2["primary_failure_group"]
    )
    d2["primary_failure_reason_label"] = diagnostic_reason_label(
        d2["primary_failure_reason"]
    )
    d2["primary_failure_basis"] = (
        "唯一覆盖全部动作有效候选数据的拒绝原因"
        if d2["primary_failure_basis"]
        == "unique reason covering every action-valid candidate"
        else "出现次数最多的 D2 拒绝原因"
    )
    d2["exclusive_funnel_note"] = (
        "未显示互斥漏斗：D2 的拒绝原因分组可能重叠，诊断层不会推断互斥划分。"
    )
    _localize_representatives(d2, reason_labels)

    for role, camera in cameras.items():
        camera["status_label"] = label_for(HEALTH_STATUS_LABELS, camera["status"])
        findings, evidence, checks = _camera_explanation(role, camera)
        camera["findings"] = findings
        camera["evidence"] = evidence
        camera["suggested_checks"] = checks

    raw["action_health"]["action_ready_definition_source"] = (
        "候选 Transition 数减去被 D2 动作分组拒绝的数量；D2 证据来自 "
        "cleaning_report 或只读同步计算。"
    )
    raw["causality_metrics"]["status_label"] = label_for(
        HEALTH_STATUS_LABELS, raw["causality_metrics"]["status"]
    )
    raw["causality_metrics"]["note"] = (
        "满足严格时序的接受数量和拒绝原因来自 D2 证据；主时间线单调性统计来自 RAW 只读诊断。"
    )

    _localize_root_cause(cause, d2, raw)
    raw["why_episode_rejected"] = _why_rejected(d2, raw)
    localized["scope_label"] = "仅用于只读诊断，不作为 D2/D3 数据资格判定依据"
    localized["display"] = {
        "d2_status": d2["status_label"],
        "primary_failure_group": d2["primary_failure_group_label"],
        "primary_failure_reason": d2["primary_failure_reason_label"],
        "root_cause": cause["label"],
        "severity": cause["severity_label"],
    }
    localized["diagnostic_report_text"] = _diagnostic_report(localized)
    return localized


def localize_diagnostics_comparison(result: dict[str, Any]) -> dict[str, Any]:
    """Localize presentation labels in an Episode comparison."""

    localized = copy.deepcopy(result)
    for item in localized["rows"]:
        technical_label = str(item["label"])
        item["technical_label"] = technical_label
        item["label"] = label_for(COMPARISON_LABELS, technical_label)
        change = _comparison_change(item)
        item["change"] = change
        item["change_label"] = COMPARISON_CHANGE_LABELS[change]
    return localized


def diagnostic_reason_label(code: str | None) -> str:
    """Return a Chinese D2 reason label with a closed unknown fallback."""

    if not code:
        return "无"
    translated = translate_reason(code)
    summary = str(translated["summary"])
    return UNKNOWN_DIAGNOSTIC_LABEL if summary == "未知数据问题" else summary


def _localize_representatives(
    evidence: dict[str, Any], reason_labels: dict[str, str]
) -> None:
    position_labels = {"first": "首个", "middle": "中间", "last": "末尾"}
    collections = [evidence["representative_rejections"]]
    collections.extend(evidence["representative_rejections_by_reason"].values())
    for rows in collections:
        for sample in rows:
            sample["sample_position_label"] = position_labels.get(
                sample["sample_position"], "代表"
            )
            sample["reason_labels"] = [
                reason_labels.get(code, diagnostic_reason_label(code))
                for code in sample["reasons"]
            ]


def _camera_name(role: str) -> str:
    return "头部相机" if role == "head" else "腕部相机"


def _camera_explanation(
    role: str, camera: dict[str, Any]
) -> tuple[list[str], list[str], list[str]]:
    name = _camera_name(role)
    status = camera["status"]
    repeat = camera["max_consecutive_identical_timestamp"]
    findings_by_status = {
        "FROZEN": [
            f"{name}时间戳在主时间线继续推进时停止更新。",
            f"最长连续重复时间戳覆盖 {repeat} 行。",
        ],
        "MISSING": [f"{name}没有可用的时间戳或帧索引证据。"],
        "STALE": [f"{name}时间戳仍在更新，但观测数据持续陈旧。"],
        "CLOCK_MISMATCH": [f"{name}时间戳来自不兼容或超前的时钟域。"],
        "HEALTHY": [f"{name}的时间戳、有效标记、帧索引和 freshness 均正常。"],
        "DEGRADED": [f"{name}存在部分信号丢失或有效率下降。"],
        "UNKNOWN": [f"{name}现有证据不足，无法进行保守分类。"],
    }
    checks_by_status = {
        "FROZEN": [
            "检查相机采集线程是否仍在运行。",
            "检查 latest-frame 是否持续更新。",
            "检查相机数据流及 USB / 设备连接状态。",
        ],
        "MISSING": ["检查相机初始化、Recorder 接线和媒体文件引用。"],
        "STALE": ["检查数据生产延迟、缓冲队列和 Recorder freshness 状态。"],
        "CLOCK_MISMATCH": ["检查时间戳来源以及设备时钟到主机时钟的转换。"],
        "HEALTHY": [],
        "DEGRADED": ["检查无效数据行和采集连续性。"],
        "UNKNOWN": ["在判断根因前检查 RAW 时间戳与数据生产端日志。"],
    }
    delta = camera["master_minus_camera_ns"]
    evidence = [
        f"{name}唯一时间戳：{camera['timestamp_unique_count']} / {camera['timestamp_positive_count']}",
        f"{name}有效数据：{camera['valid_count']} / {camera['total_rows']}",
        f"{name}有效帧索引：{camera['frame_index_valid_count']} / {camera['total_rows']}",
    ]
    if delta["median"] is not None:
        evidence.append(f"主时间线与{name}时间戳中位差：{delta['median']:.0f} ns")
    frozen_duration = camera["estimated_frozen_duration_ns"]
    if frozen_duration is not None:
        evidence.append(f"诊断估算冻结时长：{frozen_duration} ns")
    return (
        findings_by_status.get(status, findings_by_status["UNKNOWN"]),
        evidence,
        checks_by_status.get(status, checks_by_status["UNKNOWN"]),
    )


def _localize_root_cause(
    cause: dict[str, Any], d2: dict[str, Any], raw: dict[str, Any]
) -> None:
    code = str(cause["code"])
    cause["label"] = label_for(ROOT_CAUSE_LABELS, code)
    cause["severity_label"] = label_for(SEVERITY_LABELS, cause["severity"])
    cause["inference_source_label"] = "根据 RAW 信号推断"
    if code == "UNKNOWN":
        cause["findings"] = ["现有 RAW 证据无法保守解释主要 D2 拒绝原因。"]
        cause["evidence"] = []
        cause["suggested_subsystem"] = SUBSYSTEM_LABELS["UNKNOWN"]
        cause["suggested_checks"] = ["检查 D2 拒绝证据与 RAW 数据生产端日志。"]
    else:
        role = "head" if code.startswith("HEAD_") else "wrist"
        camera_name = _camera_name(role)
        reason = d2["primary_failure_reason"]
        rejected = int(d2["rejection_counts"].get(reason, 0))
        action_ready = int(raw["action_health"]["action_ready_count"])
        finding = (
            f"{rejected} 个通过动作有效性检查的候选 Transition "
            f"因缺少可用的{camera_name}观测而被拒绝。"
        )
        if cause.get("all_action_ready_lost_to_camera"):
            finding = (
                f"所有 {action_ready} 个通过动作有效性检查的 Transition "
                f"均因缺少可用的{camera_name}观测而被拒绝。"
            )
        camera = raw["camera_health"][role]
        other_role = "wrist" if role == "head" else "head"
        other = raw["camera_health"][other_role]
        cause["findings"] = [finding, *camera["findings"]]
        cause["evidence"] = [
            *camera["evidence"],
            f"D2 技术代码 {reason}：{rejected}",
            f"通过动作有效性检查的候选 Transition：{action_ready}",
            f"另一相机状态：{other['status_label']}",
        ]
        cause["suggested_subsystem"] = SUBSYSTEM_LABELS[
            "HEAD_CAMERA" if role == "head" else "WRIST_CAMERA"
        ]
        cause["suggested_checks"] = [
            "检查 D455 RealSense 数据流是否正常。"
            if role == "head"
            else "检查腕部相机数据流是否正常。",
            f"检查{camera_name}采集线程是否仍在运行。",
            "检查 latest-frame 是否持续更新。",
            "检查 USB / 设备连接状态及断连日志。",
            "检查 Recorder 中的相机 freshness 状态。",
        ]
    cause["safety_note"] = "不应通过降低 D2 清洗要求来绕过该问题。"


def _why_rejected(d2: dict[str, Any], raw: dict[str, Any]) -> list[str]:
    if d2["rejected_count"] == 0:
        return ["没有被拒绝的 D2 候选 Transition。"]
    action_ready = int(raw["action_health"]["action_ready_count"])
    steps = [
        f"共发现 {d2['candidate_count']} 个候选 Transition。",
        f"其中 {d2['rejection_group_counts']['ACTION']} 个未通过动作有效性检查。",
        f"剩余 {action_ready} 个通过了动作有效性检查。",
    ]
    cause = raw["inferred_root_cause"]
    if cause.get("all_action_ready_lost_to_camera"):
        role = "head" if cause["code"].startswith("HEAD_") else "wrist"
        camera_name = _camera_name(role)
        steps.extend(
            [
                f"这 {action_ready} 个 Transition 均需要可用的{camera_name}观测。",
                f"当前 Episode 中不存在可用的{camera_name}观测。",
                f"{camera_name}时间戳在整个 Episode 内未继续更新。",
                "因此最终没有任何 Transition 可以进入 Curated 数据。",
            ]
        )
    else:
        steps.append(f"主要 D2 拒绝原因是：{d2['primary_failure_reason_label']}。")
    return steps


def _format_age(value: float | None) -> str:
    if value is None:
        return "—"
    numeric = float(value)
    if abs(numeric) >= 1_000_000_000:
        return f"{numeric / 1_000_000_000:.2f} 秒"
    return f"{numeric / 1_000_000:.2f} 毫秒"


def _diagnostic_report(result: dict[str, Any]) -> str:
    d2 = result["d2_evidence"]
    raw = result["raw_diagnostics"]
    cause = raw["inferred_root_cause"]
    groups = d2["rejection_group_counts"]
    lines = [
        "[D2 数据诊断报告]",
        "",
        f"Episode：{result['episode_id']}",
        f"D2 状态：{d2['status_label']}",
        f"候选 Transition：{d2['candidate_count']}",
        f"接受：{d2['accepted_count']}",
        f"拒绝：{d2['rejected_count']}",
        f"主要失败分组：{d2['primary_failure_group_label']}",
        f"主要失败原因：{d2['primary_failure_reason_label']}",
        f"技术代码：{d2['primary_failure_reason'] or '无'}",
        f"主要根因：{cause['label']}",
        f"根因代码：{cause['code']}",
        f"严重程度：{cause['severity_label']}",
        "",
        "D2 拒绝分组（可能重叠）：",
        *(f"{D2_GROUP_LABELS[group]}：{groups[group]}" for group in D2_GROUP_LABELS),
    ]
    for role in ("head", "wrist"):
        camera = raw["camera_health"][role]
        lines.extend(
            [
                "",
                f"{_camera_name(role)}：",
                f"状态：{camera['status_label']}",
                f"有效率：{camera['valid_ratio']:.2%}",
                (
                    "唯一时间戳："
                    f"{camera['timestamp_unique_count']} / "
                    f"{camera['timestamp_positive_count']}"
                ),
                f"有效帧索引：{camera['frame_index_valid_ratio']:.2%}",
                (
                    "时间戳中位延迟："
                    f"{_format_age(camera['master_minus_camera_ns']['median'])}"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "结论：",
            *cause["findings"],
            "",
            "建议排查：",
            f"可能故障子系统：{cause['suggested_subsystem']}",
            *(f"- {check}" for check in cause["suggested_checks"]),
            "",
            "注意：",
            cause["safety_note"],
        ]
    )
    return "\n".join(lines)


def _comparison_change(item: dict[str, Any]) -> str:
    if item["regression"]:
        return "REGRESSION"
    baseline = item["baseline"]
    current = item["current"]
    if baseline is None or current is None or baseline == current:
        return "NO_SIGNIFICANT_CHANGE"
    lower_is_better = item["technical_label"].endswith("median age")
    if lower_is_better:
        improved = current < baseline * 0.5 and baseline - current > 100_000_000
    else:
        improved = current > baseline * 1.2 if baseline else current > 0
    return "IMPROVEMENT" if improved else "NO_SIGNIFICANT_CHANGE"
