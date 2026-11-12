# JSONL 导出与 π0.5 数据验证记录（2026-11-11）

本 PR 最终仅保留原有 `export-lerobot` 的可选 `--frame-json-root`，用于同时生成
标准 LeRobot v2.1 和独立八字段 JSONL/图片。使用说明见 [frame_json.md](frame_json.md)。

已删除此次新增的外部标注导入、extras Parquet/深度存储、旋转与成功事件处理、
扩展校验和累计合并、`inspect-frame`、`validate-pi05`。原有 D2、D3、LeRobot 校验、
原有腕部 RGB 方向修正及 HF 发布代码保持不变。

## 真实输入

用户提供 `/home/zhuweiwen/Downloads/raw.zip`，其中 `episode_000009` 有 1212 个采样点。
D2 共 1211 个候选 transition，保留 1043 个、拒绝 168 个；D3 为
`ACCEPT_WITH_WARNING`，1043 / 1043 clean，导出一个连续 episode。
任务原文为 `Place the motor support bracket into the fixture.`。

D3 warning 来自重复时间戳导致的速度差分跳过：arm state 93 对、hand state 187 对、
arm action 2 对、hand action 2 对。未修改处理阈值。

该数据明确记录 `record_depth=false`，没有成功标签。JSONL 中两路 depth 和 success
保留 null；不把输出末帧自动当作成功。RAW 的 `hardware_execution=false` 和
`training_ready=false` 保留为来源信息，读取验证不认证采集硬件或专家示范质量。

## 环境与 π0.5 验证

独立开发环境按原 lockfile 建立：Python 3.12.14、NumPy 2.5.3、Pillow 12.3.0、
pytest 8.4.2、Ruff 0.16.7。没有修改 lockfile 或既有 OpenPI 环境。

已有运行解释器：`/share/zhuweiwen-local/envs/pi05-libero-train-py311/bin/python`。
LeRobot distribution 0.1.0，数据格式 v2.1，revision
`0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`；OpenPI revision
`215abfb217dbac7d5f1273282331b9b1866c0479`。
Torch 2.7.1、torchvision 0.22.1、TorchCodec 0.4.0、JAX/jaxlib 0.5.3、PyArrow 20.0.0。

此前使用同一真实数据完成官方 OpenPI 离线 one-batch 验证，结果为：

- state `[2,32]`，action `[2,50,32]`，17 维以外均为零。
- 真实 tokenizer 输出 `[2,200]`，各路图像 `[2,224,224,3]`，缺失左腕视角正确 mask。
- 官方 RunningStats、quantile Normalize、PI05 model transforms 和 JAX data loader 实际运行。
- 每个 episode 首末帧的动作 horizon 不跨界，task 原文映射正确。

这些是本次开发的验收证据，没有作为新增命令保留，也没有加载模型 checkpoint、执行
forward、训练、动作推理或真机控制。临时统计量没有覆盖训练资产。
TorchCodec 使用进程级 `LD_LIBRARY_PATH=/share/zhuweiwen-local/envs/lerobot-mujoco-imitation/lib`
找到本机已有 FFmpeg 7.1.1 共享库。机器提供 4 张 RTX 3090，本次数据验证只用 CPU。

原始 ZIP 及解压数据共 2426 个文件的 SHA-256 未变，且解压内容与原 ZIP 一致。
所有数据、图片和完整日志位于 `/share/zhuweiwen-local/tmp/vla-rgbd-validation/`，未纳入 Git。

## 精简后重新验收

用一次 `export-lerobot` 命令及 `--frame-json-root`，生成了 `simplified/lerobot/`
和 `simplified/frame-json/` 两个独立目录。后者的
`train/episode_000000/frames.jsonl` 共 1043 行，八字段、时间、17 维状态和动作逐行与
标准 Parquet 一致，2086 个 RGB 路径全部存在。新 LeRobot 目录没有 `extras/`。
原始 2426 个文件重新计算 SHA-256，全部未变。

复用原有 worker 的 `openpi` smoke 对新 LeRobot 成果读取通过：1043 帧、50 步 horizon
不跨 episode、物理存储 17 维、读取补齐到 32 维。本次重验跳过归一化和 tokenizer，
此前完整 one-batch 结果仍作为历史验收记录，不为此保留新增命令。

完整回归（启用原有官方 LeRobot/OpenPI 集成与新增双输出集成）：
**807 passed, 4 skipped**，258.72 秒。4 个 skip 为历史指定位置的工件不可用。
Ruff、变更文件格式检查、diff check、离线 wheel/sdist 构建通过；wheel 中包含
`json_frames.py`，不包含已删除的 `frame_extras.py`。
