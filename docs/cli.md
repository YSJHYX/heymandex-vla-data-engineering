# CLI reference

根目录 [README](../README.md) 是 production happy path。本页补充命令语义；
MAINLINE 与 OPTIONAL 不可混为一条 eligibility chain。
此机默认 `/data/uv-cache` 可能只读；照 README 设置 `UV_CACHE_DIR` 后运行
`uv run --no-sync`，不会执行依赖同步。

## Production CLI — stops at private HF verification

| Command | Role |
| --- | --- |
| `vla-data build-curated` | RAW → causally valid physical Curated v1 |
| `vla-data validate-curated` | Independent read-only Curated validation |
| `vla-data quality` | D3 reports and clean-transition masks |
| `vla-data export-lerobot` | Direct D2+D3 contiguous runs → LeRobot v2.1 |
| `vla-data validate-lerobot` | Reconstruct source plan and official reload |
| `vla-data publish-hf --dry-run` | Local official reload and production publication preflight |
| `vla-data publish-hf` | Publish one validated split to an existing private HF repo and verify it |
| `vla-data publish-hf --revision SHA` | Read-only fresh download/reload of a pinned revision |

### Curation and quality

```bash
uv run --no-sync vla-data build-curated \
  --input-root RAW \
  --output-root CURATED \
  --episode episode_000000

uv run --no-sync vla-data validate-curated \
  --input-root CURATED \
  --episode episode_000000

uv run --no-sync vla-data quality \
  --input-root CURATED \
  --output-root QUALITY \
  --episode episode_000000
```

省略 `--episode` 处理 discovery 得到的全部 canonical episode。batch stage 逐 episode
隔离；已验证且 fingerprint 相同的工件 skip。`--force` 请求 staging 重建，不会就地改
RAW 或 Curated。

`vla-data run --input-root RAW --work-root WORK --stages curated,validate,quality`
只是前三步的 batch orchestration；它不运行 annotation/export/HF。

### Direct production export

```bash
uv run --no-sync vla-data export-lerobot \
  --curated-root CURATED \
  --quality-root QUALITY \
  --output-root EXPORT \
  --dataset-name rm65_sg100_mainline \
  --validation-fraction 0.1 \
  --split-seed 17 \
  --lerobot-python /data/projects/vla_ws/openpi/.venv/bin/python
```

Production mode only reads:

```text
Curated metadata/trajectory/media
+ D2 cleaning_report.json (captured RAW source-kind marker)
+ quality_report.json
+ quality_mask.npy
```

It never discovers annotation or verification roots. Export eligibility is D2 valid, D3 accepted
with at least one clean row, non-empty collection instruction, and a non-synthetic RAW source-kind
marker. `hardware_execution` and expert/source status fields are diagnostic provenance, not
substitutes for physical transition evidence or separate approval gates.

Every maximal True run inside one D2 segment becomes one LeRobot episode. No minimum run
length is configured. `source_end_index` is exclusive. Split assignment hashes
`source_episode_id`; derived runs cannot leak across train/val.

The stored contract is exactly:

```text
observation.images.head   video
observation.images.wrist  video
observation.state         float32 [17]
action                    float32 [17]
task                      exact collection instruction
```

The exporter writes `export_summary.json`, `export_provenance.jsonl` and independent
`export_validation.json` beside split roots. It also writes the relevant rows to each
`train|val/meta/source_provenance.jsonl`, so source provenance is included in the Hub
canonical payload without becoming a model feature. A repeat skips only when source
fingerprint, runtime identity and independent reload all match.
The mainline additionally carries `expert_training_status` from Curated into every
source/run provenance row. Its value, including `REVIEW_REQUIRED`, does not affect publication.

```bash
uv run --no-sync vla-data validate-lerobot \
  --output-root EXPORT \
  --lerobot-python /data/projects/vla_ws/openpi/.venv/bin/python
```

### Private HF publication

The dataset repo must already exist and be private. Authentication stays outside this
repository (`hf auth login`, `hf auth whoami`); no token is persisted.

```bash
uv run --no-sync vla-data publish-hf \
  --dataset-root EXPORT/train \
  --repo-id OWNER/DATASET \
  --hf-python /data/projects/vla_ws/openpi/.venv/bin/python \
  --lerobot-python /data/projects/vla_ws/openpi/.venv/bin/python \
  --dry-run
```

`publish-hf` is cumulative: its normal baseline is the **current valid remote HEAD**. It
fresh-downloads and validates that pinned private revision, without walking backward to
older commits or automatically recovering historical/test episodes. The initial valid
baseline of `PPPPPilot/VLADexData` is
`2347702eed03bb3b4a54c1f4598c47c98e804e45` (4 episodes, 1055 frames).
The parent `b786109f06c985069c57118f5a815eedef684a2e` is legacy test history,
not part of this cumulative dataset. Historical recovery requires a separate explicit
maintenance/debug operation; these SHAs are documentation, not publisher constants.

From the valid HEAD, the command deduplicates exact source-run identities, then rebuilds
the complete LeRobot dataset by
logical episode with the official writer. Two standalone exports may both contain `chunk-000`;
their files are never directly overlaid. The same task string may occur in many separate
episodes. For example, 4 existing episodes plus 2 new ones become 6, while rerunning
the already-published 003–006 batch appends zero. Dry-run reports `BASELINE`, `INCOMING`,
`DUPLICATES`, `TO_APPEND`, and
`MERGED_TOTAL`. An exact rerun returns `NO_NEW_EPISODES`, including with `--force`.
No expert-status approval is required. Public repos, unknown files, missing baseline
provenance, 32D vectors, and missing tasks/videos fail closed. Remove `--dry-run` only
after inspecting the plan and authorizing the network mutation. A real upload is one
parent-SHA-guarded commit of the fully validated rebuilt dataset, followed by pinned fresh
download, SHA-256 comparison, and official LeRobot reload. The verifier's `--revision SHA`
mode never merges or uploads; to compare bytes independently, give it a retained **merged**
dataset root and an empty `--cache-dir`, not the standalone incoming root. `--force` never
duplicates source runs or deletes unknown remote files.

## Task field naming

Curated v1 keeps its stable persisted `language_instruction`. Mainline export calls the exact
same bytes-as-text value `task_instruction`, stores its exact UTF-8 SHA-256 in provenance, and
writes it as LeRobot `task`. This is not D4/D5 `final_instruction`.

## Optional semantic tools and downstream compatibility

These tools remain supported for research and diagnostics, but none is needed for direct
export or HF publication. OpenPI training/normalization, 17→32 padding and robot deployment
are downstream, not CLI stages here. `scripts/smoke_openpi_lerobot.py` is optional
downstream compatibility evidence and not a publication gate.

| Command | Optional role |
| --- | --- |
| `vla-data annotate` | D4 hierarchical semantic analysis |
| `vla-data verify-annotations` | D5 confidence/human-review policy |
| `vla-data review-annotation` | Human correction/rejection |
| `vla-data build-manifest` | Legacy semantic training manifest |
| `vla-data export-lerobot --manifest-root ...` | Legacy manifest-authorized export |
| `vla-data render-rgb-review` | Read-only visual diagnostic |
| `vla-data probe-provider` | Provider handshake without inference |

Example annotation run:

```bash
uv run --no-sync vla-data annotate \
  --curated-root CURATED \
  --quality-root QUALITY \
  --output-root ANNOTATION \
  --episode episode_000000 \
  --provider standard-glm \
  --prompt-version v3.3
```

Provider credentials and model calls are required unless `--dry-run` is used. Optional
annotation output may coexist with production inputs; direct export ignores it and its
fingerprint does not change.

LIBERO tooling under `src/vla_data/benchmark/` validates semantic behavior only. It is not
RM65B+SG100 production training data.

## Exit codes and help

```text
0  command completed with no episode/validation failures
1  one or more episode/validation failures
2  global configuration, input, safety or dependency error
```

The parser is the source of truth:

```bash
uv run --no-sync vla-data --help
uv run --no-sync vla-data export-lerobot --help
uv run --no-sync vla-data publish-hf --help
```
