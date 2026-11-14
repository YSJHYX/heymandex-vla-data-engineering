# Biyi 分支 frame-json 验证（2026-11-14）

基线为 `Biyi@70e40b6`。本次移植 PR #1 的可选 `--frame-json-root` 和 PR #2 的
整数 success 规则：每条导出 episode 仅末帧为 `1`，此前各帧为 `0`；单帧为 `1`。
使用方式与字段定义见 [frame_json.md](frame_json.md)。

保留 Biyi 的纠错轨迹标签逻辑，未修改 D2/D3、export plan、HF 发布或合并实现。
双输出集成测试检查 `failure_recovery` / `dagger` 的 provenance 保留，并覆盖
train/val、单帧与多帧 clean run、路径隔离、dry-run、失败清理及复用已有 LeRobot。

## 真实 RAW 验证

输入为 `/home/zhuweiwen/Downloads/raw.zip`。逐文件 SHA-256 核对已有解压数据与
压缩包一致后，使用本分支从 RAW 重新运行 D2 → D3 → LeRobot + frame-json。
没有复用旧的 Curated、质量报告或导出数据。

- 来源 `episode_000009`；D2 成功，1043 条有效 transition。
- D3 `ACCEPT_WITH_WARNING`，1043 / 1043 clean；未放宽过滤阈值。
- 质量警告来自非正时间差的速度计算跳过：arm state 93 对、hand state 187 对，
  arm/hand action 各 2 对。
- LeRobot 官方独立回读 `PASS`；两路 RGB 均为 1043 帧、30 FPS、640×480。
- JSONL 严格八字段，共 1043 行，时间、17 维 state/action 与 Parquet 逐行一致。
- 2086 张 RGB PNG 全部可读；depth 字段为 null。
- success 类型逐行确认为整数；前 1042 行为 `0`，最后一行为 `1`。
- RAW 的 2425 个文件运行前后 SHA-256 一致；RAW 和原压缩包未被修改。

本机结果目录：`/share/zhuweiwen-local/datasets/heymandex-biyi-clean-20261114/`。
`commands.json` 保存完整运行参数，`result_checks.json` 保存逐行核对结果，
`raw_hashes_before.json` 保存 RAW 基线，`logs/` 保存运行日志。数据与图片不纳入 Git。

## 环境与检查

数据处理使用已有 Python 3.12 / NumPy 2 环境；LeRobot v2.1 worker 使用本机已有
Python 3.11、LeRobot 包与 FFmpeg 动态库。旧 OpenPI editable 路径失效，测试使用
本机已有 `openpi-source-215abfb.tar.gz` 解压的源码。未安装或更新依赖。
回归测试通过 PATH 显式选择已有 FFmpeg，并设置 NO_PROXY 避免本地 HTTP 请求走失效代理。

修改文件的 Ruff lint、format 和 `git diff --check` 通过。
全仓 Ruff 仍报告基线 `publish/_merge_worker.py` 中的 3 处 E402；该文件与 Biyi 基线一致。

启用官方 LeRobot v2.1 / OpenPI 集成的全量回归：**809 passed, 4 skipped**，260.54 秒。
4 项跳过均因历史验收工件不可用：camera health、D2 episode、UI diagnostic labels、
UI reasons。新增 frame-json 集成另行运行 **3 passed**，30.90 秒。
