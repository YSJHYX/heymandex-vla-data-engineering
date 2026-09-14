# VLA Data Engineering Pipeline · Operator Runbook

RM65 + SG100 → **Native OpenPI π0.5**。本仓库负责数据工程，不控制机器人。

当前状态（2026-09-14）：**D1–D9 PASS**（历史本机 evidence，非本次重跑）；
**Fresh full end-to-end acceptance: PENDING VALID SOURCE**。
新采集 `episode_000002` 在 D2 正确拒绝，**不是 D10 full E2E PASS**。
D10.1 只审计训练契约、修正门禁粒度并交付操作手册；没有 GLM、HF、训练调用。

## 1. Authoritative training contract

| Native π0.5 payload | 唯一物理／语言 authority |
|---|---|
| `observation.images.head` | Curated causal head RGB reference |
| `observation.images.wrist` | Curated causal `right_wrist` RGB reference |
| `observation.state [17]` | RM65 measured qpos [6] + SG100 measured **mode-7** joint-position feedback [11] |
| `action [17]` | RM65 **actual-sent** qcmd [6] + SG100 **current final-effective accepted** qcmd [11] |
| `task` | `verification.final_instruction` → `manifest.final_instruction`，逐字一致 |

State/action 单位 **rad**。LeRobot/HF 存 **float32 [17]**，仅允许 dtype cast；
不做归一化、clamp、截断、desired/retargeted 替代。
**32D 只存在于 OpenPI model transforms，不存入数据集。**
RAW `language_instruction` 仅为 provenance，绝不是训练 task 的 fallback。

SG100 feedback packet 存在 ≠ 有效训练关节位置。所有 11 路 mode 必须为 7；
`hand_feedback_sdk_rad` 有限但 mode 非 7，不具有所需语义。
`hand_qcmd_effective_canonical_rad` 有限但已经 stale，也不是新的 action。
不使用 `hand_qpos_canonical_rad` / 未验证 F_fb 作为 measured state。

### 什么可以 hard-gate？

只允许训练 payload 和直接证明它的语义／因果 metadata：组件 timestamps、
component feedback/command validity、mode-7、camera timestamps/references、
D2 segment、D3 mask、verification status。缺失 authority 时 fail closed。

`arm_status`、`arm_controller_error`、`arm_sender_hz`、`hardware_execution`、
`training_ready`、`dataset_status`、generic health、telemetry rate、debug/operator
字段仅诊断，不得单独否决有效 payload。Curated readiness/status 不一致现在是 warning。
来源 schema/path/RAW language 缺失也是 provenance warning，不替代真实校验。

注意名字不能代替源码审计：当前 RAW `head_camera_valid/wrist_camera_valid`
实际认证 freshness、episode-start 边界、media enqueue，不是普通健康状态；继续 hard-gate。
组件 validity 同样包含 recorder 的 freshness/acceptance 证据。
PRE→ACTION→POST 和 segment 规则不变；本轮不发明 freshness 阈值或最短 run 长度。
完整修改前 inventory 和证据见 [D10.1 audit](docs/training_contract_audit.md)。

### 数采 SOP（Collection SOP）

正式采集顺序：

```text
C → stable → On → A → task → D → stable → Off → Y
```

| 步骤 | 含义 | 不是什么 |
|---|---|---|
| `C` | 重新锚定 AVP reference | 不是 recording start |
| `On` | 开始 dataset episode（recorder） | 不是 robot authority enable |
| `A` | 授权 active robot command | 不是 recording start |
| `task` | 执行演示任务 | — |
| `D` | 撤销 robot authority | 不是 episode stop；永远不会 enable |
| `Off` | **唯一正常的 episode stop** | — |
| `Y` | 保存 pending episode | — |

`A`/`D` **不是强制的 training segment boundary**。真正的 training segment 由
physical state/action validity、source timestamps、freshness、causality 与
D3 quality mask 决定。`A` 之前的 stationary rows 与 `D` 之后的 final
stationary rows 允许出现在 RAW 中，由 D2 transition-level 过滤处理。

### RAW ≠ training trajectory

Recorder `On`/`Off` 只决定 RAW episode 的 **byte boundary**；D2/D3 决定哪些
transition 真正成为 training trajectory。**不要把 `RAW row count == training
frame count` 当作预期**（见 §11 episode_000001 正例：3361 rows → 901
usable transitions）。

## 2. Architecture and responsibilities

```text
immutable RAW + external JPG
  ↓ D2 Causal Synchronization (component PRE → ACTION → POST)
Curated v1 (physical17, segment_offsets, causal RGB refs)
  ↓ D3 Quality (quality_report.json + quality_mask.npy)
  ↓ D4 GLM Annotation (model-first, quality-eligible only)
  ↓ D5 Confidence Verification / Human Review
Manifest / source-episode Split
  ↓ D6 LeRobot v2.1 (one contiguous clean run = one LeRobot episode)
  ↓ D7 Hugging Face publication + independent reload
OpenPI consumer: repack → physical17 quantile normalization → model transforms
  ↓ Native π0.5 (32D model width, horizon 50)
```

**Data Engineering responsibility ends at validated/published LeRobot dataset.**
OpenPI remains the consumer；norm stats、checkpoint、训练归 OpenPI 管理。
没有一个 `run` 命令贯穿所有阶段：当前 `run` 仅支持 `curated,validate,quality`。

## 3. Environment and verified CLI

项目 Python `>=3.12,<3.13`，安装后 import `vla_data`；wheel 提供 `vla-data`。
项目依赖 NumPy/Pillow；不会自动安装 LeRobot、torch、HF 或编码器。
本机现有 LeRobot/HF interpreter：`/data/projects/vla_ws/openpi/.venv/bin/python`
（Python 3.11，LeRobot distribution 0.1.0，实际 format **v2.1**）。
源码 commit `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`，不是最新版 v3。
不对 OpenPI 执行 `uv sync`。

下列 help 已在当前 checkout 执行核对：

```bash
vla-data --help
vla-data run --help
vla-data build-curated --help
vla-data validate-curated --help
vla-data quality --help
vla-data annotate --help
vla-data verify-annotations --help
vla-data review-annotation --help
vla-data build-manifest --help
vla-data export-lerobot --help
vla-data publish-hf --help
vla-data render-rgb-review --help
```

仓库内可用 `uv run --no-sync vla-data ...`。可控验证：
`uv run --no-sync pytest -v`、`uv run --no-sync ruff check src tests`、`uv build --offline`。
不应把本机已有环境路径当作云端必需路径；云端准备兼容环境后替换 interpreter。

## 4. NEW_EPISODE：正常 RAW → HF 操作示例

以下是**操作模板，不是本轮已执行的 production run**。先配置有效新源和已批准 policy。
真实 RAW 命名必须是 `episode_XXXXXX.npz` + `episode_XXXXXX_media/`。
`NEW_EPISODE` 是 run 的标签，不是合法 source episode ID。
所有输出 root 均由用户配置；下面 `/data/vla_runs/NEW_EPISODE` 是建议，尚未冻结或创建。

```bash
set -euo pipefail
DE_REPO=/home/heymandex2025/heymandex-vla-data-engineering
OPENPI_ROOT=/data/projects/vla_ws/openpi
RAW_ROOT=/home/heymandex2025/桌面/vla_raw
WORK_ROOT=/data/vla_runs/NEW_EPISODE
CURATED_ROOT="$WORK_ROOT/curated"
QUALITY_ROOT="$WORK_ROOT/quality"
ANNOTATION_ROOT="$WORK_ROOT/annotation"
VERIFICATION_ROOT="$WORK_ROOT/verification"
MANIFEST_ROOT="$WORK_ROOT/manifest"
LEROBOT_ROOT="$WORK_ROOT/lerobot"
QC_ROOT="$WORK_ROOT/rgb_review"
HF_CACHE=/data/vla_hf_cache
HF_REPO=PPPPPilot/VLADexData
LEROBOT_PYTHON="$OPENPI_ROOT/.venv/bin/python"
HF_PYTHON="$LEROBOT_PYTHON"
# 调用前由操作者设置；不在手册中发明 production N 或 split。
: "${EPISODE:?set a valid new episode_XXXXXX source id}"
: "${CONFIDENCE_THRESHOLD:?set the explicitly approved threshold}"
: "${SPLIT_SEED:?set the approved source split seed}"
: "${VAL_FRACTION:?set the approved validation fraction}"
```

### D2 / D3：发现、构建、校验、质量

```bash
# Discovery + execution plan only。不是逐样本 eligibility audit。
vla-data run --input-root "$RAW_ROOT" --work-root "$WORK_ROOT" \
  --episode "$EPISODE" --stages curated,validate,quality --dry-run

vla-data build-curated --input-root "$RAW_ROOT" --output-root "$CURATED_ROOT" \
  --episode "$EPISODE"
vla-data validate-curated --input-root "$CURATED_ROOT" \
  --output-root "$WORK_ROOT/validation" --episode "$EPISODE"
vla-data quality --input-root "$CURATED_ROOT" --output-root "$QUALITY_ROOT" \
  --episode "$EPISODE"
```

也可用 `vla-data run ... --stages curated,validate,quality` 编排这三步。
批量处理去掉 `--episode`；`--workers` 仅为本地 episode workers。
显式冻结 exclusion 用 `build-curated` / `run` 的 `--expert-exclude episode_XXXXXX`
（可重复）；若目标已经完成，修改 exclusion 必须显式 `--force` 重建，不能靠 resume 更新。
零有效 transition 不发布 Curated；先解决采集问题，不继续 D4/D7。

### D4 / D5：标注、verification、manifest

```bash
# GLM_API_KEY 优先于 ZHIPU_API_KEY；通过安全环境配置，不写入命令、仓库或日志。
vla-data annotate --curated-root "$CURATED_ROOT" --quality-root "$QUALITY_ROOT" \
  --output-root "$ANNOTATION_ROOT" --episode "$EPISODE" --dry-run
vla-data annotate --curated-root "$CURATED_ROOT" --quality-root "$QUALITY_ROOT" \
  --output-root "$ANNOTATION_ROOT" --episode "$EPISODE" --concurrency 1

vla-data verify-annotations --annotation-root "$ANNOTATION_ROOT" \
  --quality-root "$QUALITY_ROOT" --output-root "$VERIFICATION_ROOT" \
  --confidence-threshold "$CONFIDENCE_THRESHOLD"
```

当前 D4 模型 `glm-4.6v-flash`；quality-ineligible 不调用 GLM。
检查 verification/human-review queue；未批准的 task 不进入下面的 manifest。

```bash
vla-data build-manifest --curated-root "$CURATED_ROOT" --quality-root "$QUALITY_ROOT" \
  --annotation-root "$ANNOTATION_ROOT" --verification-root "$VERIFICATION_ROOT" \
  --output-root "$MANIFEST_ROOT" --seed "$SPLIT_SEED" --validation-fraction "$VAL_FRACTION"
vla-data export-lerobot --manifest-root "$MANIFEST_ROOT" --output-root "$LEROBOT_ROOT" \
  --dataset-name rm65-sg100 --lerobot-python "$LEROBOT_PYTHON" --dry-run
vla-data export-lerobot --manifest-root "$MANIFEST_ROOT" --output-root "$LEROBOT_ROOT" \
  --dataset-name rm65-sg100 --lerobot-python "$LEROBOT_PYTHON"
```

### D7：独立校验后发布一个非空 split

```bash
# train/ 是 canonical dataset root；不能把含 train/val 的父目录传给 publish-hf。
vla-data publish-hf --dataset-root "$LEROBOT_ROOT/train" --repo-id "$HF_REPO" \
  --hf-python "$HF_PYTHON" --lerobot-python "$LEROBOT_PYTHON" \
  --cache-dir "$HF_CACHE" --dry-run
# 检查 remote identity、private、would-action 后，明确批准发布才执行：
vla-data publish-hf --dataset-root "$LEROBOT_ROOT/train" --repo-id "$HF_REPO" \
  --hf-python "$HF_PYTHON" --lerobot-python "$LEROBOT_PYTHON" --cache-dir "$HF_CACHE"
```

**D7 dry-run 会读取远端和认证信息，不是 offline。** 本轮没有执行。
禁止忽略输出中的 `BLOCKED`；unknown remote files 不会自动删除，即使 `--force`。
当前 publisher 只验证/上传已有 repo；不创建 repo、不更改 private。
它能替换同路径文件，不是 append/merge 工具；新训练集不可盲目覆盖既有数据。
先确认整个目标 snapshot 和 source manifest，再授权更新。
val 若非空必须有另行批准的独立 repo，不能再传同一 `$HF_REPO` 覆盖 train；
空 split 无 canonical root，也不上传。当前单源 smoke 的 val 为空。

## 5. Resume / force / dry-run

| Stage | normal / resume | `--force` | `--dry-run` |
|---|---|---|---|
| build-curated | 已存在且当前校验完整则 skip | 重建单 episode，失败保留旧发布 | 发现／计划 |
| validate-curated | 完成记录和当前结构验证 | 重新校验 | 计划 |
| quality | 完成工件可复用 | 重评，需明确授权，不自动改阈值 | 计划 |
| run | 按所选 D2/D3 stages 编排 | 转交各 stage | 计划 |
| annotate | 复用有效匹配 annotation | **重新请求 GLM，可能收费** | 无 GLM |
| verify-annotations | identity + policy；保留来源未变的人工决定 | 重算 verification，不调用 GLM | 只读计划 |
| review-annotation | 写 verification 人工决定并更新 index | 不支持 | 不支持 |
| build-manifest | fingerprint / source eligibility | 重建 manifest | 只读计划 |
| export-lerobot | fingerprint + runtime + independent validation 才 skip | staging 重建后发布 | 不写 MP4/Parquet |
| publish-hf | remote/local 匹配则 skip | 强制重传；不是删除/合并权限 | **远端只读** |
| render-rgb-review | 新 diagnostic output root | 不支持；已有输出拒绝覆盖 | 不写视频/报告 |

早期 D2/D3 resume 不是全链路内容哈希失效机制：更改源码、源数据或策略后，
不能将“目录存在/校验通过”视作重新处理证据。经批准后从受影响 stage 显式 force，
再使下游重新验证；RAW 应始终不可变。D10.1 不自动重建历史工件。

## 6. Human review / task correction

### GLM annotation 是什么、不是什么

当前 D4 annotation 是 **episode-level task instruction**（例如
`"Place the cable on the table"`）+ confidence + task_type / objects + provenance
（request id、prompt 版本、keyframes）。**GLM 不做 YOLO bbox 标注**，
也不输出 segmentation mask；`objects` / `task_type` 主要用于 annotation/QC/
provenance，π0.5 的 language input 只消费 `final_instruction` / task。

`model_annotation.confidence` 是 **GLM 基于视觉 keyframes 做 task inference
的模型自报置信度**，不是 RAW `language_instruction` 与 GLM task 之间的
相似度，更不是数据质量分数。

低 confidence 不是坏数据，而是 `NEEDS_HUMAN_REVIEW`。查看

低 confidence 不是坏数据，而是 `NEEDS_HUMAN_REVIEW`。查看
`$VERIFICATION_ROOT/human_review_queue.jsonl`、D4 annotation/keyframes，再做以下三选一操作：

```bash
# 确认 GLM 原文正确
vla-data review-annotation --verification-root "$VERIFICATION_ROOT" \
  --episode "$EPISODE" --status HUMAN_VERIFIED --reviewer operator

# 人工修正（示例文字，必须与本 episode 的真实任务对应）
vla-data review-annotation --verification-root "$VERIFICATION_ROOT" \
  --episode "$EPISODE" --status HUMAN_CORRECTED \
  --instruction "Place the cable on the table" --reviewer operator

# 拒绝进入训练
vla-data review-annotation --verification-root "$VERIFICATION_ROOT" \
  --episode "$EPISODE" --status REJECTED --reviewer operator
```

这三条是替代操作，**不要依次全执行**。
人工修改后 **不重跑 D4**：review 已更新 D5 verification → 重建 manifest →
重导出 LeRobot → 经批准重新发布 HF，记录新 commit SHA。
旧 manifest/fingerprint 和 task 不再有效；D6 自动失效重建。
若改变 confidence threshold，重跑全体 `verify-annotations`（避免混合 policy root），
再走 manifest/D6/D7；**不调用 GLM**。相同 input identity 的人工决定保留；
源 annotation/quality 变化则必须重新核验。

## 7. RGB manual review：valid / invalid / all

```bash
vla-data render-rgb-review --curated-root "$CURATED_ROOT" --quality-root "$QUALITY_ROOT" \
  --episode "$EPISODE" --selection all --output-root "$QC_ROOT" --dry-run
for selection in valid invalid all; do
  vla-data render-rgb-review --curated-root "$CURATED_ROOT" --quality-root "$QUALITY_ROOT" \
    --episode "$EPISODE" --selection "$selection" --output-root "$QC_ROOT"
done
ffplay "$QC_ROOT/$EPISODE/valid_head_wrist.mp4"
ffplay "$QC_ROOT/$EPISODE/invalid_head_wrist.mp4"
ffplay "$QC_ROOT/$EPISODE/all_head_wrist.mp4"
```

要求已有 `ffmpeg`（libx264）；普通 unit tests 不依赖编码器。
仅人工 QC，左 head、右 wrist，同一 Curated row；保留源图尺寸，显示 canvas 可补边，
不写回 JPG、不改 mask、不生成训练 contact sheet。
`valid` 仅 D3=True；`invalid` 仅 D3=False；按 Curated row 确定性排序。
缺图／解码失败的 row 在同名 JSON 的 `skipped` 给出理由，不伪造任一相机。
空 selection 只生成 `EMPTY` JSON、没有 MP4，所以播放前检查报告。
selection 会压缩时间/略过 gaps；row/tick/segment/D3 状态显示在画面和 JSON，
**不能把 QC 视频的连续播放解释为连续机器人 trajectory**。

### 播放 LeRobot 原始双相机视频与检查编码

```bash
EP_INDEX=000000  # LeRobot derived index，不是 source id
HEAD_VIDEO="$LEROBOT_ROOT/train/videos/chunk-000/observation.images.head/episode_${EP_INDEX}.mp4"
WRIST_VIDEO="$LEROBOT_ROOT/train/videos/chunk-000/observation.images.wrist/episode_${EP_INDEX}.mp4"
ffplay -autoexit "$HEAD_VIDEO"
ffplay -autoexit "$WRIST_VIDEO"
ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=width,height,avg_frame_rate,nb_read_frames -of json "$HEAD_VIDEO"
# 设置为 ffprobe 的实际 nb_read_frames；输出只放 diagnostic 目录
: "${FRAME_COUNT:?set actual decoded frame count}"
FIRST=0
MIDDLE=$((FRAME_COUNT / 2))
LAST=$((FRAME_COUNT - 1))
mkdir -p "$QC_ROOT/extract"
for camera in head wrist; do
  VIDEO="$HEAD_VIDEO"
  if [ "$camera" = wrist ]; then VIDEO="$WRIST_VIDEO"; fi
  ffmpeg -n -i "$VIDEO" \
    -vf "select='eq(n,$FIRST)+eq(n,$MIDDLE)+eq(n,$LAST)'" -vsync 0 \
    "$QC_ROOT/extract/${camera}_%03d.png"
done
# 极短 run：循环、降低播放速度，或用播放器 s 键逐帧
ffplay -loop 0 -vf 'setpts=10*PTS' "$HEAD_VIDEO"
# 仅诊断的同步双路显示；不保存为训练输入
ffmpeg -n -i "$HEAD_VIDEO" -i "$WRIST_VIDEO" \
  -filter_complex '[0:v][1:v]hstack=inputs=2' "$QC_ROOT/extract/head_wrist_compare.mp4"
```

最后一条要求相同图像高度（本机历史 640×480）；不同高度请用 review 工具的补边 canvas。
first/middle/last 在 1/2 帧 run 中可重合；不能因此删掉 short episode。

## 8. Invalid-data lifecycle

### Segment / gap 规则

D2 segment boundary 或 D3 quality-mask hole 都是 **hard trajectory
boundary**。例如：

```text
quality_mask: T T T F T T
       导出:  run A = 前 3 帧
              run B = 后 2 帧
```

不能删除 `F` 后把两侧重新拼接成 `T T T T T`。多个 derived runs 在训练中
**共同训练同一套模型参数**（dataset-level jointly trained），但 action
horizon 永远不跨 derived episode boundary —— **jointly trained ≠
trajectory-level concatenated**。

| Stage | Invalid reason | What happens | Physically deleted? | Can it reach training? |
|---|---|---|---|---|
| D2 | 缺 physical state/action | 拒绝受影响 transition；0 个可用则不发布 | 否，RAW 保留 | 否 |
| D2 | stale state/action 或 acceptance 无效 | 拒绝候选，不用 held/desired proxy | 否 | 否 |
| D2 | SG100 非 mode-7 | 不能作为 measured joint-position 样本 | 否 | 否 |
| D2 | PRE/ACTION/POST 或 camera causality 错 | 拒绝候选，gaps 形成 segment | 否 | 否 |
| D2 | 引用 JPG 缺失/解码失败 | 拒绝引用它的 transition，不换成旧图 | 否 | 否 |
| D3 | `quality_mask=False` | Curated provenance 保留，训练 selection 排除 | 否 | 否 |
| D3 | 显式 expert exclusion / 冻结静态判定 / 0 clean | episode 不 eligible | 否 | 否 |
| D4 | quality-ineligible | 不请求 GLM | 否 | 否 |
| D5 | confidence 低 | **NEEDS_HUMAN_REVIEW，不等于 invalid** | 否 | 人工批准后可 |
| D5 | HUMAN REJECTED | 不 eligible，保留人工决定 | 否 | 否 |
| D6 | invalid/gap/stale manifest | 仅导出 eligible contiguous clean runs；不拼接 gap | 否 | 无效部分不能 |
| 任意 | diagnostic warning | 报告/provenance，不单独改变资格 | 否 | payload 有效则可 |

The pipeline generally does **NOT physically delete invalid RAW data**.
RAW remains immutable for provenance. 排除依赖 transition rejection、quality masks、
verification status、manifest eligibility、final export selection，而不是删掉原文件。
已有 Curated 的 schema/shape/timing/引用结构损坏是验证失败，不在 validator 中猜测修复。
D3 部分可定位 RGB 错误保持 mask=False，不再仅因一张坏图否决其它 clean rows；
结构已损坏或所有视觉均无效仍失败。metric、阈值和显式/静态 exclusion 不变。

| 术语 | 明确含义 |
|---|---|
| MISSING | 必需有限值不存在，包括字段缺失或 NaN sentinel |
| SEMANTICALLY_INVALID | 有数值但不是所需物理量，如 mode 非 7、非 accepted command |
| STALE | cache 有数值但已超出采集端有效期，不是新 state/action |
| CAUSALLY_INVALID | source timestamp 非正或 PRE/ACTION/POST 顺序不成立 |
| QUALITY_INVALID | D3 质量/完整性排除 |
| DIAGNOSTIC_WARNING | 本身不改变 training eligibility |

validity=False 可能是缺失、未接受或过期，不能只凭一个 flag 宣称具体 stale 根因；
联合 source timestamp/age/采集端声明取证。finite ≠ semantically valid。

## 9. Storage, HF location and publication configuration

| 工件 | 路径／责任 |
|---|---|
| RAW | 本机 `/home/heymandex2025/桌面/vla_raw`；不可变源 |
| Curated / Quality / Annotation / Verification / Manifest | 上述用户配置的各 `*_ROOT`，彼此分离 |
| Local LeRobot | `$LEROBOT_ROOT/train` 和非空 `$LEROBOT_ROOT/val`；provenance/summary 在父目录 |
| Diagnostic RGB / evidence | `$QC_ROOT`；本轮验证在 `/tmp/heymandex_vla_d101_*` |
| HF cache | 建议 `/data`；历史 `/data/heymandex_vla_d7_cache`、`/data/heymandex_vla_d8_lerobot_cache` |
| D7/D8/D9 evidence | `/data/heymandex_vla_d7_evidence`、`/data/heymandex_vla_d8_evidence`、`/data/heymandex_vla_d9_evidence` |
| D10 rejected-source evidence | `/data/heymandex_vla_d10_e2e/evidence/d10_lineage.json`，部分旧描述已由本轮 audit 纠正 |
| OpenPI TEST_ONLY norm stats | `/data/heymandex_vla_d8_evidence/assets/PPPPPilot/VLADexData/norm_stats.json` |
| OpenPI base weights | `/data/openpi_checkpoints/pi05_base_pytorch` |
| OpenPI TEST_ONLY checkpoint | `/data/heymandex_vla_d9_smoke/d9_training_smoke_TEST_ONLY/d9-smoke/1` |

这些是已核对本地路径或明确标注的模板，不是冻结的 production storage policy。
大产物优先 `/data`；本机根分区约 95%，先 `df -h / /data`，不要擅删用户数据。

HF dataset repo：**`PPPPPilot/VLADexData`**。历史已验证 commit：
`b786109f06c985069c57118f5a815eedef684a2e`，仅 TEST_THRESHOLD / NOT_PRODUCTION_DATASET。
本次没有联网复核 main；不能将历史 SHA 当作已确认当前 main。
HF episode 000000/000001/000002 是 **source episode_000001** 的三个 derived runs
505/2/394；**HF episode index ≠ source episode id**。

Dataset publication parameters（不是训练超参）：`repo_type=dataset`；private 保留既有；
canonical `meta/`、`data/`、`videos/`；format `codebase_version=v2.1`。
每次上传保存 exact commit SHA、local fingerprint、provenance 和 reload evidence。
当前 `_hf_worker.py` 用 `HfApi.upload_folder(repo_id, repo_type, folder_path, commit_message)`，
不暴露 upload concurrency/chunk-size，也没有写入 token 的 CLI 参数。
**当前 publisher 不创建/移动 `v2.1` Git tag**；format 声明不是 Git tag。
如消费者依赖 tag，发布协调时另行核验授权；不要把它写成已自动完成的功能。
当前生成的 dataset card/commit message 仍标 TEST_THRESHOLD，正式数据卡治理待 production 阶段。
`publish-hf --revision` 当前不传给 upload；上传到默认分支，上传后以返回 SHA 校验，
不要用这个 flag 误以为已指定发布分支。SKIPPED 不自动做 fresh remote reload。

## 10. OpenPI / π0.5 consumer and TEST_ONLY smoke

当前 OpenPI commit `5ffa00eecb1f3b86bc3e0dcf7d0ecccb74e517f7`。
以下构造来自本地 `training/config.py` 和 D8/D9 脚本，不是上游通用 config 名：

```python
from openpi.models.pi0_config import Pi0Config
from openpi.training import config

model = Pi0Config(pi05=True, action_dim=32, action_horizon=50)
data = config.LeRobotArmHandDataConfig(
    repo_id="PPPPPilot/VLADexData",
    physical_arm_action_dim=6,
    physical_hand_action_dim=11,
    base_config=config.DataConfig(prompt_from_task=True),
)
```

`action_sequence_keys=("action",)`；真实 loader 从 task_index 表生成 prompt。
ArmHandInputs 映射 head→base_0_rgb、wrist→right_wrist_0_rgb，未使用 left_wrist mask=False。
`repack(17D) → ArmHandInputs → quantile Normalize(17D stats) → resize/tokenize/pad(32D)`。
horizon 在 loader 形成，不存 50 个 action 于每行；LeRobot clamp 在当前 episode 内，
长度 1/2 也合法，不跨 D2 segment 或 D3 mask hole。
**Production norm stats must be recomputed from production train data**，不能从 val
或历史 TEST_ONLY 集计算/复用；本轮不计算、不冻结 stats。
DataConfig 的当前加载路径没有显式 revision 参数；底层 LeRobot 支持 revision。
历史 smoke 检查 main==pinned SHA；正式消费必须固定一致 cache/snapshot/revision，不能只写 repo_id 就声称复现。

### 已验证历史 smoke 的真实命令（不要在 D10.1 执行）

```bash
cd /data/projects/vla_ws/openpi
# D8: HF → loader → TEST_ONLY stats → one batch，不更新参数
.venv/bin/python scripts/d8_heymandex_compat.py
# D9: 一次真实 optimizer step + checkpoint save/reload（有写入）
.venv/bin/python scripts/d9_training_smoke.py
```

这两个脚本 **没有 argparse/Tyro，也不支持 `--help`**；附加 `--help` 仍会启动
主流程。本轮只读审阅源码和已有 JSON，未执行危险的伪 help。
它们固定 repo/SHA/输出路径、901 帧、task、D6 临时对照路径，不是 NEW_EPISODE
或 production 训练入口。再次执行会访问 HF，D8 可能计算 stats，D9 会训练和写 checkpoint；
须单独授权、核验路径/缓存/显存并准备生产适用入口，不得把历史命令直接推广到正式训练。

| 设置 | 历史 D9 TEST_ONLY SMOKE VALUES — NOT PRODUCTION HYPERPARAMETERS |
|---|---|
| batch / steps | 1 / 1（D8 one-batch 使用 batch 2） |
| LR | 2.5e-5 |
| precision / trainable mode | bf16 / native_action_interface |
| production batch/LR/trainable layers/LoRA/steps/scheduler/freeze | **TO BE DETERMINED ON FORMAL DATA** |

D9 smoke 环境记录：RTX 5090 Laptop 24GB、bf16、batch=1、
peak allocated ≈ 22GB。这只是 smoke evidence，不是性能结论。
Checkpoint 一律放 `/data`（如 `/data/heymandex_vla_d9_smoke/...`），不要写根分区。

## 11. What should I run?

| Situation | Resume from |
|---|---|
| 新 RAW episode | D2 |
| D2 rejected all | 回采集端排查所需物理字段、validity、timestamps；不能调低诊断门禁冒充修复 |
| D3 quality reject | RGB review / recollect；不自动改 mask/阈值 |
| GLM confidence low | D5 Human Review |
| Human corrected task | D5 verification 已更新 → manifest → D6 → 经批准 D7；不重新 GLM |
| 改 confidence threshold | 全集 D5 downstream；no GLM |
| LeRobot export failed | D6；检查 source/fingerprint/worker/encoder，保留旧有效 export |
| HF upload failed / unknown remote files | D7 publication audit；不盲删、不盲目 force |
| OpenPI load failed | OpenPI integration，核对版本、映射、snapshot/SHA |
| norm stats stale | 用 production train 重算 stats，不重采 RAW |
| training checkpoint interrupted | OpenPI checkpoint resume，遵循已批准训练配置 |
| GLM HTTP 429 rate limit | 限流窗口约 1–4 分钟；保持 `--concurrency 1`，稍后 resume（SKIPPED 不重复计费） |
| HF 认证失败 | 确认 cached login / `HF_TOKEN`；publisher 报 actionable 错误，不打印 token |
| LeRobot 远程加载要求 `v2.1` tag | LeRobot 0.1.0 remote loading 需要 repo 上存在 codebase tag；历史 D8 已打 `v2.1`，发布 checklist 应核验它仍在 |
| 磁盘接近满 | `df -h / /data`；大产物一律 `/data`；不擅自清理用户文件 |

### Accepted Source Example：episode_000001（正例）

RAW 共 **3361 rows**；D2 真正可用的是 **901 transitions**，对应三个
contiguous clean runs **505 / 2 / 394**（D2 offsets `[0, 505, 507, 901]`）。
这证明：**RAW 存在 invalid intervals ≠ 整个 episode invalid** ——
transition-level filtering 允许保留有效窗口，极短 run（如 2 帧）同样是
合法导出单元。该 source 成为 D6–D9 的测试数据集。

### Rejected Source Example：episode_000002

当前 RAW 814 行；`arm_qpos_rad [814,6]` 和 `arm_qcmd_sent_rad [814,6]`
均 **814 行全 NaN**，两路 arm source timestamp 全 0，无法构成 measured6/actual-sent6。
因此 physical17 state/action 不能形成。当前 D2 的候选数是 **813**，接受 **0**。
不是因为 `hardware_execution=False`，也不是因为 F_fb pending。

SG100 raw feedback 814 行有限，但只有 626 行全 mode-7，联合 feedback validity 后 625 行；
effective hand command 709 行有限，accepted/fresh validity 523 行，另有 186 行 age 超过
录制的 100 ms hand-command 界限。两个 camera validity 各 813 行。
这些局部证据不弥补缺失的 RM65 6D；没有导出手部 11D，也没有用 desired/零值补齐。
旧 lineage 的 “arm 全零”“F_fb 导致 D2 拒绝”“814 candidates” 不准确；新 audit
依据当前源码/数组纠正，旧 evidence 保留不修改。

## 12. Pre-flight collection checklist

正式数采前只检查与 training payload 直接相关的信号（不要求 diagnostic
status 字段完美）：

```text
□ RM65 measured feedback 正在推进（qpos 随时间变化、非 NaN）
□ RM65 sent command 路径可工作（qcmd_sent 有实际下发值）
□ SG100 mode-7 feedback 可用（11 路全部 mode==7）
□ SG100 effective command 可用（accepted、fresh）
□ head camera 可用（有帧、可解码）
□ wrist camera 可用（有帧、可解码）
□ source timestamps 正在推进（非 0、单调）
```

**Safety rule**：不要通过补零、插值或 `arm_q_goal_rad` 之类的 goal 值修复
缺失的 physical state/action。required physical data 缺失时的正确动作是
**reject / recollect**，不是"让数组看起来有数字"。

## 13. Further reference

[完整 CLI contract](docs/cli.md) · [installed LeRobot/OpenPI source audit](docs/lerobot_v21_source_audit.md)
· [D10.1 hard-gate audit](docs/training_contract_audit.md)

本手册没有授权发布、训练或硬件动作；每个有外部副作用的操作仍需对应阶段授权。
