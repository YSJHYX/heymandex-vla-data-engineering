"""Chinese operator summaries for immutable pipeline diagnostics.

The original reason is never replaced in an artifact or API response. Unknown
diagnostics fail closed to a generic operator summary and retain raw evidence.
"""

from __future__ import annotations

import re

REASONS = {
    # Operator-facing aliases and current D2/Curated validity bits.
    "MISSING_MEASURED_HAND_FEEDBACK": "缺少灵巧手实测反馈",
    "MISSING_ARM_FEEDBACK": "缺少机械臂实测反馈",
    "MISSING_HEAD_CAMERA": "缺少头部相机数据",
    "MISSING_WRIST_CAMERA": "缺少腕部相机数据",
    "NO_CLEAN_TRANSITION": "没有可用于训练的有效数据区间",
    "NO_CLEAN_TRANSITIONS": "没有可用于训练的有效数据区间",
    "INVALID_CAUSAL_ORDER": "状态和动作的时间顺序异常",
    "STALE_ACTION_COMMAND": "动作命令时间过期或不同步",
    "TASK_MISSING": "缺少任务语言指令",
    "TASK_INSTRUCTION_MISSING": "缺少任务语言指令",
    "INVALID_STATE_DIMENSION": "状态向量维度不符合要求",
    "INVALID_ACTION_DIMENSION": "动作向量维度不符合要求",
    "SYNTHETIC_SOURCE": "当前数据属于测试或合成数据",
    "SYNTHETIC_TEST_ONLY": "当前数据属于测试或合成数据",
    "STATE_QPOS_INVALID": "机器人实测状态无效",
    "ACTION_QCMD_INVALID": "机器人有效动作命令无效",
    "NEXT_STATE_QPOS_INVALID": "下一帧机器人实测状态无效",
    "TICK_TIMESTAMP_ORDER_INVALID": "采集时间顺序异常",
    "ARM_QPOS_SOURCE_TIMESTAMP_INVALID": "机械臂实测状态时间戳无效",
    "HAND_QPOS_SOURCE_TIMESTAMP_INVALID": "灵巧手实测状态时间戳无效",
    "ARM_QCMD_SOURCE_TIMESTAMP_INVALID": "机械臂命令时间戳无效",
    "HAND_QCMD_SOURCE_TIMESTAMP_INVALID": "灵巧手命令时间戳无效",
    "ARM_CAUSALITY_INVALID": "机械臂状态与动作的时间关系无效",
    "HAND_CAUSALITY_INVALID": "灵巧手状态与动作的时间关系无效",
    "UNIFIED_CAUSALITY_INVALID": "机器人状态与动作的整体时间关系无效",
    "HEAD_RGB_INVALID": "头部相机画面无效",
    "HEAD_RGB_MEDIA_MISSING": "缺少头部相机图像文件",
    "WRIST_RGB_INVALID": "腕部相机画面无效",
    "WRIST_RGB_MEDIA_MISSING": "缺少腕部相机图像文件",
    # D2 transition, discovery and image-integrity diagnostics.
    "ACTION_ARM_COMMAND_FLAG_INVALID": "机械臂命令有效标记异常",
    "ACTION_HAND_COMMAND_FLAG_INVALID": "灵巧手命令有效标记异常",
    "ACTION_ROBOT_COMMAND_FLAG_INVALID": "机器人命令有效标记异常",
    "ACTION_TIMESTAMP_NONPOSITIVE": "动作时间戳无效",
    "ACTION_NONFINITE": "动作包含非有限数值",
    "ACTION_COMPONENT_MISMATCH": "机械臂与灵巧手动作不一致",
    "CAUSALITY_PRE_NOT_STRICT": "动作前状态的时间顺序异常",
    "CAUSALITY_POST_NOT_STRICT": "动作后状态的时间顺序异常",
    "INVALID_EPISODE_NAME": "Episode 文件名不符合规则",
    "MISSING_MEDIA_DIRECTORY": "缺少 Episode 图像目录",
    "MISSING_JPG": "缺少相机图像",
    "JPEG_DECODE_FAILURE": "相机图像无法解码",
    "ZERO_SIZE_IMAGE": "相机图像为空",
    "WRONG_CHANNEL_COUNT": "相机图像通道数错误",
    "RGB_CONVERSION_FAILURE": "相机图像 RGB 转换失败",
    "UNEXPECTED_RESOLUTION": "相机图像分辨率不符合要求",
    "QUALITY_INVALID": "部分数据未通过质量检查",
    "EXCLUDE_FROM_EXPERT_TRAINING": "当前 Episode 不适合作为专家训练示范",
    # Export and manifest eligibility diagnostics.
    "D2_INVALID": "物理数据检查未通过",
    "D3_NOT_CLEAN": "数据质量检查未通过",
    "D3_COUNT_MISMATCH": "清洗区间数量与报告不一致",
    "INVALID_DATASET_FPS": "数据采样频率无效",
    "INVALID_PHYSICAL_UNIT": "机器人数据物理单位无效",
    "NO_CLEAN_CONTIGUOUS_RUN": "没有连续有效的训练区间",
    "INPUT_INVALID": "输入数据无效",
    "CURATED_EPISODE_ID_MISMATCH": "整理后 Episode 编号不一致",
    "CURATED_DIMENSION_NOT_17": "整理后状态或动作不是 17 维",
    "CURATED_VALIDATION_FAILED": "整理后数据校验失败",
    "CURATED_INVALID_OR_MISSING": "整理后数据缺失或无效",
    "QUALITY_INVALID_OR_MISSING": "质量报告缺失或无效",
    "ANNOTATION_INVALID_OR_MISSING": "标注缺失或无效",
    "VERIFICATION_INVALID_OR_MISSING": "验证结果缺失或无效",
    "HIERARCHICAL_INPUT_INVALID": "层级标注输入无效",
    "NO_SEMANTIC_TRAINING_SEGMENTS": "没有可训练的语义区间",
    "SEMANTIC_BOUNDARY_OUTSIDE_D3_CLEAN_RUN": "语义边界超出有效清洗区间",
    "SEMANTIC_BOUNDARY_CROSSES_D2_SEGMENT": "语义边界跨越物理数据分段",
    "INPUT_FINGERPRINT_CHANGED": "输入数据指纹发生变化",
    "INVALID_SPLIT": "训练集或验证集划分无效",
    "MIXED_VERIFICATION_POLICIES": "数据混用了不同的验证策略",
    "TRAIN_VAL_EPISODE_OVERLAP": "训练集与验证集存在 Episode 重叠",
    "NEEDS_HUMAN_REVIEW": "需要人工复核",
    "INELIGIBLE": "当前数据不符合使用条件",
    "REJECTED": "当前数据已被拒绝",
    "PROVIDER_FAILED": "标注服务处理失败",
    "VALIDATION_FAILED": "数据验证失败",
    "NO_TASK_RELEVANT_ACTIVITY": "没有与任务相关的有效动作",
    "INSUFFICIENT_VISUAL_EVIDENCE": "相机画面证据不足",
    # Optional annotation/provider diagnostics found in src/.
    "ANNOTATION_CAMERA_VIEW_MISSING": "标注所需的相机画面缺失",
    "ANNOTATION_INPUT_ERROR": "标注输入无效",
    "PASS_A_INTERNAL_INCONSISTENCY": "初步标注内部不一致",
    "SEMANTIC_RECALL_REVIEW": "语义标注需要人工复核",
    "SEMANTIC_TASK_SEQUENCE_MISMATCH": "语义任务顺序不一致",
    "VERY_SHORT_NONTRAINING_INTERVAL": "非训练区间过短",
    "BOUNDARY_EVIDENCE_AMBIGUOUS": "区间边界证据不明确",
    "BOUNDARY_EVIDENCE_INSUFFICIENT": "区间边界证据不足",
    "REJECTED_OPTIONAL_PARAPHRASE": "可选任务改写未通过",
    "NETWORK_ERROR": "标注服务网络异常",
    "HTTP_RATE_LIMIT": "标注服务请求过于频繁",
    "HTTP_SERVER_ERROR": "标注服务内部错误",
    "HTTP_CLIENT_ERROR": "标注服务请求错误",
    "HTTP_AUTH": "标注服务认证失败",
    "HTTP_PERMISSION": "标注服务权限不足",
    "HTTP_QUOTA": "标注服务额度不足",
    "HTTP_PROVIDER_OVERLOAD": "标注服务繁忙",
    "HTTP_OTHER": "标注服务返回其他 HTTP 错误",
    "INVALID_JSON": "标注服务返回的数据格式无效",
    "INVALID_SCHEMA": "标注服务返回的数据结构无效",
    "EMPTY_RESPONSE": "标注服务返回空内容",
    "MODEL_OUTPUT_SCHEMA_ERROR": "模型输出结构不符合要求",
    "MCP_STARTUP_ERROR": "标注工具启动失败",
    "MCP_INITIALIZE_ERROR": "标注工具初始化失败",
    "MCP_TOOL_NOT_FOUND": "标注工具不可用",
    "MCP_TOOL_SCHEMA_ERROR": "标注工具参数结构无效",
    "MCP_TOOL_CALL_ERROR": "标注工具调用失败",
    "MCP_TIMEOUT": "标注工具调用超时",
    "MCP_PROCESS_EXITED": "标注工具意外退出",
    "MCP_INVALID_RESPONSE": "标注工具返回无效内容",
    "MCP_IMAGE_SIZE_REJECTED": "标注图像超过服务大小限制",
}

_CODE = re.compile(r"^([A-Z][A-Z0-9_]+)(?::|$)")
_VELOCITY = re.compile(
    r"^(arm|hand) (state|action) velocity skipped (\d+) pairs with non-positive dt$"
)
_TEMPORAL = re.compile(r"^(.+): (\d+) transitions exceed (\d+) ns$")
_VISUAL = re.compile(r"^(head|right_wrist): (\d+) frames below configured blur score$")
_VALIDATION_PREFIXES = {
    "invalid schema_name": "数据结构名称无效",
    "invalid schema_version": "数据结构版本无效",
    "missing metadata keys": "数据元信息缺失",
    "language_instruction must": "任务语言指令缺失或无效",
    "joint_names do not match": "关节顺序不符合 17 维要求",
    "arm_slice must": "机械臂状态切片无效",
    "hand_slice must": "灵巧手状态切片无效",
    "camera_roles must": "缺少必要相机角色",
    "unsupported camera roles": "相机角色不受支持",
    "invalid action_": "动作表示或单位无效",
    "invalid state_": "状态表示或单位无效",
    "missing trajectory keys": "轨迹数据字段缺失",
    "unknown trajectory keys": "轨迹包含未知字段",
    "forbidden trajectory keys": "轨迹包含禁止字段",
    "trajectory row counts disagree": "轨迹各字段的帧数不一致",
    "trajectory contains no transitions": "轨迹没有有效数据帧",
    "segment_offsets": "轨迹分段信息无效",
    "native source timestamps": "原生来源时间戳无效",
    "state_timestamp_ns": "状态时间戳无效",
    "action_timestamp_ns": "动作时间戳无效",
    "unified causality": "状态与动作的时间顺序异常",
    "head_rgb_frame_index": "头部相机帧索引无效",
    "Curated v1 forbids depth": "整理后数据包含不支持的深度目录",
}


def translate_reason(original: str) -> dict[str, str | None]:
    """Return a safe summary plus unchanged technical evidence for the UI."""
    raw = str(original).strip()
    match = _CODE.match(raw)
    code = match.group(1) if match else None
    summary = REASONS.get(code) if code else None
    if not summary:
        if code and re.fullmatch(r"CAUSALITY_(ARM|HAND)_(PRE|POST)_UNAVAILABLE", code):
            summary = "机械臂或灵巧手的前后状态时间证据缺失"
        elif code and re.fullmatch(r"FEEDBACK_(ARM|HAND)_(PRE|POST)_INVALID", code):
            summary = "机械臂或灵巧手的实测反馈无效"
        elif code and re.fullmatch(
            r"CAMERA_(HEAD_CAMERA|WRIST_CAMERA)_(UNAVAILABLE|MEDIA_MISSING|DECODE_INVALID)",
            code,
        ):
            camera = "头部" if "HEAD_CAMERA" in code else "腕部"
            summary = f"{camera}相机数据缺失或无效"
        elif code and re.fullmatch(
            r"QUALITY_(REJECT|EXCLUDE_FROM_EXPERT_TRAINING)", code
        ):
            summary = "质量检查拒绝该 Episode 用于训练"
        elif match := _VELOCITY.fullmatch(raw):
            component = "机械臂" if match.group(1) == "arm" else "灵巧手"
            kind = "状态" if match.group(2) == "state" else "动作"
            summary = f"{component}{kind}速度检查跳过 {match.group(3)} 对无效时间间隔"
        elif match := _TEMPORAL.fullmatch(raw):
            summary = f"时间同步检查发现 {match.group(2)} 个超出阈值的数据点"
        elif match := _VISUAL.fullmatch(raw):
            camera = "头部" if match.group(1) == "head" else "腕部"
            summary = f"{camera}相机有 {match.group(2)} 帧清晰度低于阈值"
        elif re.fullmatch(
            r"(head|right_wrist): (same-reference|near-static) run exceeds configured maximum",
            raw,
        ):
            summary = "相机画面长期重复或变化过小"
        else:
            phrases = {
                "required RGB hard-integrity failure": "必要相机画面存在严重完整性错误",
                "static episode is unsuitable as expert BC demonstration": "轨迹基本静止，不适合作为训练示范",
                "configured quality thresholds excluded all transitions": "全部数据都未通过质量阈值",
            }
            summary = phrases.get(raw) or next(
                (
                    value
                    for prefix, value in _VALIDATION_PREFIXES.items()
                    if raw.startswith(prefix)
                ),
                "未知数据问题",
            )
    return {"summary": summary, "technical_code": code, "original": raw}
