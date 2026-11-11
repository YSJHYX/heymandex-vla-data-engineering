# RGB-D sidecar / π0.5 数据管线验收（2026-11-11）

## 结论与范围

用户提供的真实 `episode_000009` 已完成 RAW → D2 → D3 → LeRobot v2.1 →
官方 OpenPI PI05 归一化、tokenizer、模型输入转换与 one-batch 数据读取。
未加载模型权重，未执行模型 forward、训练、动作推理或真机控制。

该 RAW 明确记录 `record_depth=false`、`camera_modalities=rgb`，没有深度图片或 success
标注。因此真实回读中的深度保持 null/invalid，success 保持 unknown；RGB-D 像素、成功事件
切段及累计合并使用明确标为 fixture 的合成数据验证，不能据此声称真实深度采集已经验收。
RAW 的 `hardware_execution=false`、`training_ready=false` 被保留为来源信息；本次数据读取
验证不认证采集硬件配置或专家示范质量。

## 实际输入与结果

| 项目 | 结果 |
| --- | --- |
| RAW | 1212 个采样点，1211 个候选 transition，双路 640×480 RGB，30 Hz |
| 任务原文 | `Place the motor support bracket into the fixture.` |
| D2 | 1043 accepted / 168 rejected，1 个物理连续片段 |
| D3 | `ACCEPT_WITH_WARNING`，1043 / 1043 clean |
| LeRobot | 1 episode，1043 帧，float32[17] state/action，两路标准视频 |
| 扩展表 | 1043 行，原始 tick 对齐，深度缺失，success 全部 unknown |
| PI05 state batch | `[2, 32]`，额外维度为零 |
| PI05 action batch | `[2, 50, 32]`，额外维度为零 |
| prompt tokens | `[2, 200]`，使用真实 PaliGemma tokenizer |
| 图像 | 各 `[2, 224, 224, 3]`；真实 head/right wrist，缺失 left wrist 被 mask |
| 动作边界 | 每个 episode 首末帧的 50 步读取与本 episode 的命令序列一致，不跨界 |
| 原始数据完整性 | 2426 个文件（包括 ZIP）SHA-256 未变；解压内容与 ZIP 内容一致 |

D3 warning 为实测/命令时间戳重复导致部分速度差分跳过：arm state 93 对、hand state
187 对、arm action 2 对、hand action 2 对。未降低清洗阈值，也未删除 warning。

归一化统计由官方 `RunningStats` 在本次数据上计算，输入为 17 维；官方 quantile
`Normalize`、`ModelTransformFactory` 和官方 JAX data loader 在 CPU 上实际执行。
统计量只用于本次验收，没有覆盖现有模型训练资产。

## 环境

机器提供 4× RTX 3090（各 24 GiB），本次数据验收不使用 GPU。
仓库独立 `.venv` 通过 `uv sync --frozen --group dev` 建立，未更改 lockfile：
Python 3.12.14、NumPy 2.5.3、Pillow 12.3.0、pytest 8.4.2、Ruff 0.16.7。

既有 LeRobot/OpenPI interpreter：
`/share/zhuweiwen-local/envs/pi05-libero-train-py311/bin/python`。

| 依赖 | 实际版本 |
| --- | --- |
| Python | 3.11.16 |
| LeRobot | distribution 0.1.0，`CODEBASE_VERSION=v2.1` |
| LeRobot Git revision | `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` |
| OpenPI Git revision | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| Torch / torchvision / TorchCodec | 2.7.1 / 0.22.1 / 0.4.0 |
| JAX / jaxlib | 0.5.3 / 0.5.3 |
| PyArrow / PyAV | 20.0.0 / 17.0.0 |

默认 TorchCodec 最初找不到 FFmpeg。通过本机已有 FFmpeg 7.1.1 的共享库目录
`/share/zhuweiwen-local/envs/lerobot-mujoco-imitation/lib` 设置进程级 `LD_LIBRARY_PATH`
后解决，没有安装或升级既有 OpenPI 环境。系统失效的代理变量在联网下载依赖及 GitHub
操作时被移除；测试的 loopback 请求同样使用无代理环境。

## 合成数据验收

`tests/export/test_frame_extras.py` 包括：

- 16 位深度无损回读、腕部旋转 180° 后逐像素比较。
- 拒绝重复 tick、无来源的已知 success、错误 RGB 帧索引、非因果时间戳和无效比例尺。
- D3 中间切段后，success 事件仅按原始 tick 归属；被过滤的事件不会迁移至片段末尾。
- 缺失标签保持 unknown，标定资产变化使来源指纹失效。
- 官方 LeRobot 回读及 OpenPI horizon 检查；可选真实 tokenizer 的完整 one-batch 检查。
- 累计合并重新编号后，扩展行与深度相对路径仍对应；历史无扩展 episode 可共存。
- 重复来源的不同扩展拒绝静默去重，损坏的深度资产拒绝通过验证。
- 扩展文件包含在发布 SHA-256 清单中；本次没有上传任何数据集到 Hugging Face。

## 本地复现

所有数据和日志产物位于 `/share/zhuweiwen-local/tmp/vla-rgbd-validation/`，不纳入 Git。
`source/` 是从用户 `/home/zhuweiwen/Downloads/raw.zip` 安全解压出的只读输入。

```bash
CHECK_ROOT=/share/zhuweiwen-local/tmp/vla-rgbd-validation
LEROBOT_PYTHON=/share/zhuweiwen-local/envs/pi05-libero-train-py311/bin/python
TOKENIZER=/share/zhuweiwen-local/projects/pi05-libero-experiment/stage4/cache/openpi/big_vision/paligemma_tokenizer.model
export LD_LIBRARY_PATH=/share/zhuweiwen-local/envs/lerobot-mujoco-imitation/lib

.venv/bin/vla-data validate-lerobot \
  --output-root "$CHECK_ROOT/export" --lerobot-python "$LEROBOT_PYTHON"
.venv/bin/vla-data validate-pi05 \
  --dataset-root "$CHECK_ROOT/export/train" \
  --tokenizer-path "$TOKENIZER" --lerobot-python "$LEROBOT_PYTHON"
.venv/bin/vla-data inspect-frame \
  --dataset-root "$CHECK_ROOT/export/train" --episode-index 0 --frame-index 1042 \
  --lerobot-python "$LEROBOT_PYTHON"
```

最后一帧对应 source tick 1103，深度为 null，`success=0, success_valid=false`。
详细证据包括 `pi05-batch.json`、`real-validation-final.json`、`last-frame-final.json`、
`source-integrity.json`、`environment.json` 及测试 JUnit XML。

## 最终检查

- 锁定依赖环境的完整测试（启用官方 LeRobot/OpenPI 与真实 tokenizer 集成）：
  **813 passed, 4 skipped**，300.41 秒。4 个 skip 均因历史指定位置的旧数据/验收工件不存在。
- 最终扩展专项测试（含完整 PI05 batch、历史数据混合合并）：**9 passed**，61.09 秒。
- `ruff check src tests`、变更 Python 文件的 `ruff format --check`、`git diff --check`：通过。
- `uv build --offline`：wheel 和 sdist 通过；wheel 确认包含 `frame_extras.py`。
- 最终重新计算 2426 个输入文件 SHA-256：全部未变。

完整测试需设置 `VLA_LEROBOT_PYTHON` 和 `VLA_PI05_TOKENIZER_PATH`，分别指向上述已有
interpreter 与 tokenizer；命令行可用的 FFmpeg/ffprobe 来自已有环境的 `bin/`。

## 独立八字段 JSONL 导出验收

`export-frame-json` 已从上述真实 LeRobot 数据另行导出
`frame-json/episode_000000/frames.jsonl`，共 **1043 行**。逐行检查严格包含
`t, image_top, image_hand, depth_top, depth_hand, joint_pos, action, success` 八字段，
state/action 均为有限的 17 维数组；2086 个 RGB 相对路径全部存在。
首末帧各路图片均为 640×480；缺失深度与未知 success 均为 JSON null。
额外交付文件没有改写标准 LeRobot 数据。图片从标准视频解码为 PNG。

包含新增 JSONL 导出和无 extras 数据导出的专项集成测试 **10 passed**（62.71 秒）；
CLI、导出规划、基础发布校验相关回归测试 **68 passed, 1 skipped**。
