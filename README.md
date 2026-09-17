# HeymanDex VLA Data Engineering

## 1. 项目用途与边界

本仓库把真实机器人 RAW 数据整理为经过验证、可累计发布的 LeRobot v2.1 数据集。主线到私有 Hugging Face 仓库的固定 revision 回读验证为止。

**本仓库不负责 π0.5 训练、不负责模型推理、不负责真机部署。** OpenPI/π0.5 只是下游消费者；normalization、17→32 padding、training、checkpoint 和 inference 都不属于这里的生产流程。D4/D5 也不是生产主线的前置条件。

## 2. 生产 Pipeline

```text
RAW
 │
 ▼
D2 物理 / 因果清洗
 │
 ▼
D3 质量过滤
 │
 ▼
最大连续 Clean Runs
 │
 ▼
LeRobot v2.1 导出
 │
 ▼
本地独立验证
 │
 ▼
私有 Hugging Face 累计发布
 │
 ▼
固定 Commit SHA 的 Fresh Download / Reload 验证
 │
 ▼
STOP
```

RAW 保持不变。只有 D2 物理/因果有效且 D3 clean 的连续区间才进入导出；D4/D5 不参与这条链路。

## 3. 数据合同

| LeRobot feature | 类型与来源 |
| --- | --- |
| `observation.state` | `float32[17]`：`arm_qpos_rad[6]` + `hand_feedback_sdk_rad[11]`，均为测量反馈 |
| `action` | `float32[17]`：`arm_qcmd_sent_rad[6]` + `hand_qcmd_effective_canonical_rad[11]`，均为实际发送/生效命令 |
| `observation.images.head` | RGB 视频，来自 RAW 相机角色 `head` |
| `observation.images.wrist` | RGB 视频，来自 RAW 相机角色 `right_wrist` |
| `task` | 采集时的原始 `language_instruction` |

语言必须逐字保持：`RAW language_instruction = D2 language_instruction = LeRobot task = HF task`。不 lowercase、不 trim、不 paraphrase，也不经过 D4/D5 改写；来源追踪保存原文及其 UTF-8 SHA-256。命令不能冒充测量状态，Hub 数据中也没有面向模型的 32D padding。

一个 LeRobot episode 是**一条连续的 clean physical rollout**。D2 boundary 或 D3 `False` 都会切断 trajectory。例如 `[0,100)` clean、`[100,120)` invalid、`[120,300)` clean，应导出两个 episode：`[0,100)` 和 `[120,300)`，绝不拼成 `[0,300)`。

相同 task 可以有多个独立 episode，例如 `grab the ball` 对应 episode 0、1、2、3；task 不是 episode 身份。

## 4. LeRobot / HF 数据格式

正式输出是 LeRobot v2.1：`data/` 存 Parquet，`videos/` 存 MP4，`meta/` 存 task、episode、统计与 `source_provenance.jsonl` 等 metadata。来源追踪包含 `source_episode_id`、`source_start_index`、`source_end_index`（右端不含）和 task 的 UTF-8 SHA-256。

导出根目录按非空 split 建立子目录：

```text
lerobot/
├── export_summary.json
├── export_provenance.jsonl
├── export_validation.json
├── train/
│   ├── data/
│   ├── videos/
│   └── meta/
└── val/                  # 仅在 val 有 episode 时存在
    ├── data/
    ├── videos/
    └── meta/
```

当前 003–006 实际导出只有 `train/`；`--validation-fraction 0.1` 不保证小批次一定产生 `val/`。分配按 `source_episode_id` 做 source-level split，同一来源的 derived runs 不会跨 train/val。发布时一次只指定一个 split root；如需发布 `val/`，应使用单独的私有 HF repo，不能假定它会自动并入 train repo。

### 为什么不是 HDF5

当前 pipeline 的正式输出就是 LeRobot v2.1 的 Parquet + MP4 + metadata。这里没有 HDF5 exporter 或 converter，也不要求训练前再转换成 HDF5；下游训练直接消费 LeRobot 数据结构。

## 5. 快速开始

1. 按下一节设置路径和现有解释器。
2. Step 1 运行 D2；Step 2 运行 D3。
3. Step 3 导出 LeRobot；Step 4 做本地独立验证。
4. Step 5 执行 HF dry-run，核对累计计划。
5. 经明确授权后执行 Step 6 真正发布；发布命令自动完成固定 revision 回读验证。
6. Step 7 说明验收项，以及在持有完整 merged 本地副本时如何额外独立复核。

## 6. 环境与路径配置

使用已有环境，不在数据处理时安装依赖。`LEROBOT_PYTHON` 只是已有 LeRobot v2.1 / `huggingface_hub` 解释器，不会把 OpenPI 加入本仓库的主流程。以下是 README 唯一的变量定义区；将 `REPLACE_*` 换成实际路径或仓库名：

```bash
cd /home/heymandex2025/heymandex-vla-data-engineering

export UV_CACHE_DIR=/data/uv-cache

RAW_ROOT=/data/vla_runs/REPLACE_RUN_NAME/raw
WORK_ROOT=/data/vla_runs/REPLACE_RUN_NAME/work

CURATED_ROOT="$WORK_ROOT/curated"
QUALITY_ROOT="$WORK_ROOT/quality"
EXPORT_ROOT="$WORK_ROOT/lerobot"

DATASET_NAME=rm65_sg100_mainline

HF_REPO=REPLACE_OWNER/REPLACE_DATASET

LEROBOT_PYTHON=/data/projects/vla_ws/openpi/.venv/bin/python
HF_PYTHON="$LEROBOT_PYTHON"
```

`RAW_ROOT` 指向待处理批次的真实 RAW episode 目录。HF dataset repo 必须预先存在且为 private；可用 `hf auth whoami` 检查当前认证，不要把 token 写进配置或产物。以下 `uv run --no-sync` 命令不会同步依赖。

## 7. Step 1 — D2 物理/因果清洗

用途：从 RAW 建立 Curated，检查 PRE/POST measured feedback、actual sent/effective action、causal timestamps，以及 head/wrist pre-action cameras。

输入：`$RAW_ROOT`。输出：`$CURATED_ROOT`。

运行命令：

```bash
uv run --no-sync vla-data build-curated \
  --input-root "$RAW_ROOT" \
  --output-root "$CURATED_ROOT" \
  --workers 4
```

成功标准：有效来源生成 Curated episode，RAW 未被改写，原始 `language_instruction` 和物理边界得到保留。

常见阻断条件：缺少测量反馈、有效命令或因果时间戳；不要用 command proxy 填补 measured state。

## 8. Step 2 — D3 质量过滤

用途：对 Curated transition 生成质量报告与 clean mask。

输入：`$CURATED_ROOT`。输出：`$QUALITY_ROOT`。

运行命令：

```bash
uv run --no-sync vla-data quality \
  --input-root "$CURATED_ROOT" \
  --output-root "$QUALITY_ROOT" \
  --workers 4
```

成功标准：来源状态为 `ACCEPT` 或 `ACCEPT_WITH_WARNING`，且至少有 1 条 clean transition，才可继续导出。

常见阻断条件：`REJECT` 或零条 clean transition；D3 `False` 不是可跨越的空洞。

## 9. Step 3 — LeRobot Export

用途：从 D2/D3 直接导出最大连续 clean runs，不读取 D4/D5 注释。

输入：`$CURATED_ROOT`、`$QUALITY_ROOT`。输出：`$EXPORT_ROOT`，其中 `train/` 总是发布候选 split；`val/` 仅在有验证 episode 时生成。

运行命令：

```bash
uv run --no-sync vla-data export-lerobot \
  --curated-root "$CURATED_ROOT" \
  --quality-root "$QUALITY_ROOT" \
  --output-root "$EXPORT_ROOT" \
  --dataset-name "$DATASET_NAME" \
  --validation-fraction 0.1 \
  --split-seed 17 \
  --lerobot-python "$LEROBOT_PYTHON"
```

成功标准：独立 clean runs 保持独立；state/action 为 17D，两路视频、精确 task 与 provenance 均存在。来源状态字段如 `expert_training_status`、`training_use_status`、`source_dataset_status` 仅作诊断 metadata，不是额外发布审批门槛。

常见阻断条件：显式 synthetic RAW、空 task、D2/D3 无效、没有 clean run，或同一次导出来源 FPS 不一致。

## 10. Step 4 — Local Validation

用途：独立重建来源计划并通过官方 LeRobot API 回读本地导出。

输入：`$EXPORT_ROOT`。输出：验证结果；不改写导出数据。

运行命令：

```bash
uv run --no-sync vla-data validate-lerobot \
  --output-root "$EXPORT_ROOT" \
  --lerobot-python "$LEROBOT_PYTHON"
```

成功标准：`passed=true`、`official_reload=PASS`；17D state/action、head/wrist、task/hash、episode 边界与 provenance 一致。

常见阻断条件：来源已变化，或 Parquet、MP4、metadata 与来源计划不一致；不得跳过验证直接发布。

## 11. Step 5 — HF Dry Run

用途：只读检查现有私有仓库与本次 incoming split，计算累计发布计划；**不会上传**。

输入：`$EXPORT_ROOT/train`、`$HF_REPO`。输出：`BASELINE`、`INCOMING`、`DUPLICATES`、`TO_APPEND`、`MERGED_TOTAL` 和拟执行动作。

运行命令：

```bash
uv run --no-sync vla-data publish-hf \
  --dataset-root "$EXPORT_ROOT/train" \
  --repo-id "$HF_REPO" \
  --hf-python "$HF_PYTHON" \
  --lerobot-python "$LEROBOT_PYTHON" \
  --dry-run
```

成功标准：本地导出独立验证通过，当前 private HEAD 可作有效 baseline，累计计划与预期相符；同批重跑可得到 `Would NO_NEW_EPISODES`。

常见阻断条件：HF repo 不存在或为 public、远端含未知文件、现有 HEAD 缺可核验 provenance、或本地导出未通过验证。不要用 `--force` 绕过。

## 12. Step 6 — HF Cumulative Publish

⚠️ **该操作会修改 HF 远端仓库。** 仅在 Step 5 dry-run 成功、检查计划并获得本次远端写入授权后执行。

用途：以当前有效 HEAD 为 baseline，按 logical episode 累计新 run，完整重建 LeRobot 数据集并发布新 revision；上传后自动做固定 SHA 的 fresh download、hash 比对和官方 reload。

输入：`$EXPORT_ROOT/train`、`$HF_REPO`。输出：新 commit SHA 与远端验证证据；原始 RAW、本地 incoming 导出和旧 HF revision 均不改写。

运行命令：

```bash
uv run --no-sync vla-data publish-hf \
  --dataset-root "$EXPORT_ROOT/train" \
  --repo-id "$HF_REPO" \
  --hf-python "$HF_PYTHON" \
  --lerobot-python "$LEROBOT_PYTHON"
```

成功标准：新 episode 已追加、旧 episode 保留，并得到通过回读验证的精确 commit SHA。若 `TO_APPEND=0`，返回 `NO_NEW_EPISODES`，不产生新 commit。

常见阻断条件：baseline 无效、未知远端文件、并发 HEAD 变化或上传后验收失败；不要把“上传请求返回”当成发布完成。

## 13. Step 7 — Pinned Remote Verification

用途：对精确 HF commit 做 private、文件 hash、LeRobot 结构和来源追踪验收。Step 6 已自动执行一次；以下命令只用于**额外的独立复核**。

输入：精确 `HF_REVISION`，以及与该 revision 上传前字节一致的**完整 merged dataset 本地副本**。输出：只读验收证据。

运行命令（仅在确实保留了完整 merged root 时）：

```bash
HF_REVISION=REPLACE_WITH_EXACT_COMMIT_SHA

VERIFY_CACHE=$(mktemp -d \
  -p "$WORK_ROOT" \
  hf_verify_XXXXXXXX)

uv run --no-sync vla-data publish-hf \
  --dataset-root REPLACE_WITH_RETAINED_MERGED_ROOT \
  --repo-id "$HF_REPO" \
  --hf-python "$HF_PYTHON" \
  --lerobot-python "$LEROBOT_PYTHON" \
  --revision "$HF_REVISION" \
  --cache-dir "$VERIFY_CACHE"
```

`REPLACE_WITH_RETAINED_MERGED_ROOT` 是本次发布实际使用、经过验证的完整 merged dataset，不是当天 incoming standalone dataset。当前 CLI **不会输出或持久保留**临时 merged staging 路径；若未另行保留字节一致的副本，就不要照抄这条独立复核命令，也不要用 incoming root 顶替。即使远端此前为空，发布器仍会完整重建，视频字节也可能变化。常规发布以 Step 6 自动执行并打印的 pinned 远端验收为准。

成功标准：

- `private=true`，`resolved_sha` 等于请求的 SHA，`canonical_file_mismatches=0`；
- 官方 LeRobot reload PASS，episode/frame 数正确；
- state/action 为 `float32[17]`，head/wrist 可加载；
- task/hash、`source_episode_id` 与 `[source_start_index, source_end_index)` 一致。

常见阻断条件：使用 `main` 或 tag 而非精确 SHA、cache 非空、把 standalone incoming 当作 merged root，或远端文件/结构不匹配。

## 14. HF 累计数据集语义

**append 单位是 logical episode，不是 chunk。** `chunk-000` 只是 LeRobot 的存储实现细节；两个独立导出都可能包含它，不能把今天的 `chunk-000` 直接覆盖昨天的文件，也不需要手工改名为 `chunk-001`。

```text
远端当前完整 LeRobot dataset + 今天的新 logical episodes
  → 本地完整 rebuild → validate → 上传新的完整 HF revision
```

同一个 task 可以跨许多独立 episode，重置边界不会因 task 相同而消失。日常例子：Day 1 有 100 episodes，Day 2 新增 50 后 latest 有 150，Day 3 再增 80 后 latest 有 230；旧 episode 不被覆盖。

重复上传同一 batch：`DUPLICATES=N`、`TO_APPEND=0`、`NO_NEW_EPISODES`，不产生新 commit。部分重叠例如 remote 10 episodes、incoming 4（其中 2 已有），结果是 2 duplicates、2 append、12 total。publisher 对已在远端的来源 run 自动去重；incoming 自身重复来源 run 则拒绝，避免含糊数据。

## 15. 当前已验证 Baseline

`PPPPPilot/VLADexData@2347702eed03bb3b4a54c1f4598c47c98e804e45` 是初始有效累计 baseline：4 episodes、1055 frames、task `grab the ball`。当前正常发布始终校验**现有有效 HEAD**，此 SHA 只是历史起点，不是写死的软件常量。

旧 commit `b786109f06c985069c57118f5a815eedef684a2e` 是早于当前 provenance/task contract 的历史测试数据，不参与当前累计 dataset。

## 16. 可选工具

D4 semantic annotation、D5 verification、LIBERO benchmark 与 OpenPI compatibility smoke 均可用于研究或诊断，**不是 RAW → HF 主线要求**。高级 CLI 参数与可选命令见 [CLI 参考](docs/cli.md)。

## 17. 常见问题

| 情况 | 处理 |
| --- | --- |
| `NO_NEW_EPISODES` | incoming 已全部存在于远端；这是正常幂等结果，不是错误。 |
| `unknown remote files` | 停止并核对仓库内容；不要用 `--force` 覆盖。 |
| HF repo 为 public | 必须先由仓库管理者设为 private；发布器不会向 public repo 写入。 |
| fresh cache not empty | 重新用 `mktemp -d -p "$WORK_ROOT" hf_verify_XXXXXXXX` 创建空目录。 |
| D3 no clean transition | 该 source 不导出；检查质量报告，不拼接无效区间。 |
| duplicate runs | 与远端重合会自动去重；incoming 内部重复同一来源 run 会拒绝。 |

## 18. 开发测试

```bash
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
git diff --check
uv run --no-sync pytest -q
```

本仓库不在此流程中执行模型训练或机器人动作。

## Web UI

本地中文操作界面是现有 CLI 的 operator-facing wrapper，**不重写 D2、D3、LeRobot exporter 或 HF publisher**。CLI 仍是工程师可审计、可脚本化的底层接口；UI 和 CLI 调用同一 Python pipeline，生成相同目录结构与报告。不安装前端工具链，也不自动安装/更新 Python 包。

在仓库根目录启动（使用现有 `.venv`，不执行 `uv sync`）：

```bash
.venv/bin/python -m vla_data.ui
```

打开 `http://127.0.0.1:8765`。服务默认仅监听本机 loopback；在启动终端按 `Ctrl-C` 关闭。浏览器关闭或刷新不会删除已提交的 backend job，处理记录保存在本机 SQLite。首次使用 HF 功能之前，由机器管理员在终端完成本机 `hf auth login`；UI 不接收、显示或保存 token。

配置集中在 [`config/ui.toml`](config/ui.toml)：允许的数据/RAW 根目录、UI 状态目录、现有 LeRobot/HF Python、UV cache、默认 private HF repo、监听地址/端口。Run 列表只能浏览 allowlist 内的已配置根目录，不能输入任意文件路径。`batch4_acceptance` 是 003–006 历史验收工件的**只读**映射：可看报告、重新执行本地验证和 HF Dry Run，但不可清洗、重导出或正式上传。当前 `PPPPPilot/VLADexData` 只是默认 repo ID，真实 HEAD 总是实时读取，不写死历史 SHA。

数采人员标准操作：

1. 选择 RAW Run，核对 episode、task、head/wrist 相机预览。
2. 点击「开始数据清洗」，等待物理数据检查和质量检查；`ACCEPT_WITH_WARNING` 可继续，但应查看警告。已有清洗结果可直接使用，重新运行需要二次确认。
3. 检查 episode 表，再点击「生成 LeRobot 数据集」。默认 validation fraction 为 `0.1`、split seed 为 `17`；小批次不保证产生 val split。
4. 点击「验证 LeRobot 数据集」。未通过时 HF 阶段被阻断。
5. 在 Hugging Face 页确认私有 repo 和账户，点击「检查 HF 上传计划」。这是只读 Dry Run，核对重复数、新增数、期望总数和 baseline SHA。`NO_NEW_EPISODES` 表示无需重复上传，上传按钮不可用。
6. 仅在确有新增 episode 时，点击「上传到 Hugging Face」，核对二次确认弹窗并勾选确认。随后等待固定 commit SHA 的 fresh download、hash 比对和官方 LeRobot 回读均通过；若远端验收失败，不能视作发布成功。

UI 不提供修改 RAW、task、provenance、HF 删除/覆盖、OpenPI 训练或机器人控制。页面请求只允许固定的 pipeline 动作，服务端校验路径与参数，且全局一次只执行一个 job。日志做 token/secret 基础脱敏；出错时保留报告与 job 日志供数据工程人员排查。UI metadata 位于配置的 `state_dir`，不写入 RAW。
