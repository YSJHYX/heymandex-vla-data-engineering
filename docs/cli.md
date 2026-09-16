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
| `vla-data publish-hf` | Publish one approved split to an existing private HF repo and verify it |
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
+ quality_report.json
+ quality_mask.npy
```

It never discovers annotation or verification roots. Export eligibility is D2 valid, D3 accepted
with at least one clean row, non-empty collection instruction, and no synthetic/expert-exclude
marker. `hardware_execution` is provenance, not a substitute for physical transition evidence.
Export integration eligibility does not imply expert-training approval.

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
source/run provenance row. `REVIEW_REQUIRED` remains an integration-only status.

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

Remove `--dry-run` only after inspecting the remote state and confirming every source run
has explicit `expert_training_status=APPROVED_FOR_EXPERT_TRAINING`.
`REVIEW_REQUIRED` or missing approval blocks upload. Public repos, unknown remote files,
bad canonical layouts, missing source provenance, 32D physical vectors and missing tasks/videos
are hard failures. A real
upload is followed by exact-revision fresh download, SHA-256 comparison and official LeRobot
reload. Re-run verification independently with `--revision SHA --cache-dir EMPTY_PATH`;
the verifier refuses a nonempty cache and never uploads. The publisher refreshes the
existing `v2.1` tag policy to the uploaded commit; a missing or
stale tag prevents skip. `--force` permits a known payload re-upload but never deletes unknown files or disables
the privacy gate.

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
