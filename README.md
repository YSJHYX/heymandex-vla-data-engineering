# HeymanDex VLA Data Engineering

This repository prepares and publishes training data. It does not train π0.5 and does not participate in robot inference or deployment. OpenPI consumes the published dataset downstream, but is not part of this repository's production pipeline.

## Production boundary

```text
RAW → D2 physical/causal curation → D3 quality filtering
    → maximal clean runs → LeRobot v2.1 export → local validation
    → private Hugging Face publish → pinned fresh-download/reload verification
    → STOP
```

The production payload is measured `observation.state float32[17]` (`arm_qpos_rad[6] + hand_feedback_sdk_rad[11]`), effective `action float32[17]` (`arm_qcmd_sent_rad[6] + hand_qcmd_effective_canonical_rad[11]`), RGB `observation.images.head` and `observation.images.wrist` (source roles `head` and `right_wrist`), and an exact collection-time `task`. No command proxy as measured state; no 32D model padding in Curated, LeRobot, or Hub.

The language contract is exact: `RAW language_instruction = D2 language_instruction = LeRobot task = published task`. The exporter preserves the original string, including whitespace and case, and records its UTF-8 SHA-256. It never uses D4/D5 text to rewrite the task.

## Environment and inputs

Use the existing environments; do not install packages as part of data processing. The LeRobot/HF interpreter below is an existing interpreter with audited LeRobot v2.1 and `huggingface_hub`, not an OpenPI pipeline step. The destination dataset repo must already exist and be private; authentication is external (`hf auth login`, `hf auth whoami`). No token belongs in a config, artifact, or log.

```bash
cd /home/heymandex2025/heymandex-vla-data-engineering

RAW_ROOT=/data/vla_runs/REPLACE_RUN_NAME/raw
WORK_ROOT=/data/vla_runs/REPLACE_RUN_NAME/work
CURATED_ROOT="$WORK_ROOT/curated"
QUALITY_ROOT="$WORK_ROOT/quality"
EXPORT_ROOT="$WORK_ROOT/lerobot"
DATASET_NAME=rm65_sg100_mainline
HF_REPO=REPLACE_OWNER/REPLACE_DATASET
LEROBOT_PYTHON=/data/projects/vla_ws/openpi/.venv/bin/python
HF_PYTHON="$LEROBOT_PYTHON"
export UV_CACHE_DIR=/data/uv-cache
```

Run the commands below independently. `UV_CACHE_DIR` avoids this machine's read-only default `/data/uv-cache`; `uv run --no-sync` does not update the lockfile or install packages.

## 1. D2 curation

```bash
uv run --no-sync vla-data build-curated \
  --input-root "$RAW_ROOT" --output-root "$CURATED_ROOT" --workers 4
```

D2 enforces PRE and POST measured feedback, sent/effective action, causal timestamps, and both pre-action cameras. RAW is immutable. The Curated output preserves physical segmentation and the exact task.

## 2. D3 quality

```bash
uv run --no-sync vla-data quality \
  --input-root "$CURATED_ROOT" --output-root "$QUALITY_ROOT" --workers 4
```

Accepted production statuses are `ACCEPT` or `ACCEPT_WITH_WARNING` with at least one clean transition. A D2 segment and every D3 `False` row are hard boundaries: each maximal contiguous `D2 valid AND D3 clean` run becomes one LeRobot episode. Invalid gaps are never bridged.

## 3. Direct LeRobot v2.1 export

```bash
uv run --no-sync vla-data export-lerobot \
  --curated-root "$CURATED_ROOT" --quality-root "$QUALITY_ROOT" \
  --output-root "$EXPORT_ROOT" --dataset-name "$DATASET_NAME" \
  --validation-fraction 0.1 --split-seed 17 \
  --lerobot-python "$LEROBOT_PYTHON"
```

The direct exporter reads D2/D3 only. It rejects synthetic and expert-excluded sources, empty tasks, invalid D2, rejected D3, and zero clean runs. All derived runs of a `source_episode_id` remain in the same train/val split. `export_summary.json`, `export_provenance.jsonl`, and split-local `meta/source_provenance.jsonl` retain source episode ID, exclusive source range, exact task and SHA-256, source dataset status, and expert-training status. Annotation and D5 files are neither read nor required.

## 4. Independent local validation

```bash
uv run --no-sync vla-data validate-lerobot \
  --output-root "$EXPORT_ROOT" --lerobot-python "$LEROBOT_PYTHON"
```

Require `passed=true`, source-plan/provenance consistency, and official LeRobot reload before considering publication. The publication root below is one split, normally `$EXPORT_ROOT/train`; if a val split is published, it needs a separate private repo and the same acceptance.

## 5. Private-HF dry-run

```bash
uv run --no-sync vla-data publish-hf \
  --dataset-root "$EXPORT_ROOT/train" --repo-id "$HF_REPO" \
  --hf-python "$HF_PYTHON" --lerobot-python "$LEROBOT_PYTHON" \
  --dry-run
```

Dry-run performs local layout/17D/camera/task-hash/provenance checks and an official local reload, then (only for approval-eligible data) reads the existing repo's privacy, files, HEAD and `v2.1` tag. It never uploads. `BLOCKED` is a failure, not permission to remove `--dry-run`.

Production publication requires every source run to carry explicit `expert_training_status=APPROVED_FOR_EXPERT_TRAINING`. This is a future authorized source-metadata state, not an approval granted by this repository or this README. `REVIEW_REQUIRED`, missing status, synthetic, and expert-excluded sources do not authorize upload. Episodes 003–006 are integration fixtures with `REVIEW_REQUIRED`; their successful export/reload is **not** expert-training approval. No approval is inferred from `hardware_execution`, `training_use_status`, or a test PASS.

## 6. Private-HF publish — MUTATING NETWORK OPERATION

Only after human review of the dry-run and source approval metadata, run the same command without `--dry-run`. Do **not** run it merely because it appears in this README.

```bash
uv run --no-sync vla-data publish-hf \
  --dataset-root "$EXPORT_ROOT/train" --repo-id "$HF_REPO" \
  --hf-python "$HF_PYTHON" --lerobot-python "$LEROBOT_PYTHON"
```

The publisher rejects public/nonexistent repos and unknown remote files; `--force` cannot bypass either safety gate. It copies canonical metadata, Parquet and MP4 bytes unchanged, records the dataset fingerprint, source-export fingerprint (when available), UTC timestamp and exact commit SHA, and maintains the existing `v2.1` tag policy. A successful upload automatically attempts pinned fresh-download validation. Upload alone is not a completed publication if verification fails; preserve the commit SHA and investigate.

## 7. Independent fresh-download acceptance

Set `HF_REVISION` to the **exact commit SHA printed by the real publish**, not `main` or the moving tag. The verifier requires an absent or empty cache path. This command creates a new empty cache and performs no upload:

```bash
HF_REVISION=REPLACE_WITH_EXACT_COMMIT_SHA
uv run --no-sync vla-data publish-hf \
  --dataset-root "$EXPORT_ROOT/train" --repo-id "$HF_REPO" \
  --hf-python "$HF_PYTHON" --lerobot-python "$LEROBOT_PYTHON" \
  --revision "$HF_REVISION" \
  --cache-dir "$(mktemp -d -p "$WORK_ROOT" hf_verify_XXXXXXXX)"
```

Acceptance requires the pinned revision to resolve exactly and remain private, all canonical files to match the byte-preserved source manifest, and official LeRobot reload of the fresh snapshot. The reload compares episode/frame counts, 17D float32 arrays, both RGB camera features and per-episode video frames, exact task strings/hashes, provenance, and exclusive source ranges. Byte hashes are appropriate here because this publisher copies canonical files unchanged; any future rewriting publisher must use a stable logical fingerprint instead. Keep the fresh snapshot and evidence with the recorded revision.

## Eligibility, optional tooling, and boundary

Export integration eligibility is D2 valid, D3 accepted with a nonempty clean run, nonempty exact task, non-synthetic, and non-expert-excluded. Production HF publication adds explicit expert-training approval. Descriptive dataset statistics may accompany validation; OpenPI-specific normalization, model-side 17→32 padding, training, checkpoints, inference, and robot safety are downstream responsibilities. GPU availability is not a data-engineering gate.

D4 annotation, D5 verification, LIBERO benchmarks, and the legacy semantic-manifest export remain optional research/diagnostic tools. They are not inputs to direct export, validation, or production eligibility. See [CLI reference](docs/cli.md) and historical audit notes for details. `scripts/smoke_openpi_lerobot.py` remains an optional downstream compatibility check only; it is never a publication prerequisite. Its prior local evidence does not establish remote HF or training readiness.

For local development checks:

```bash
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync pytest -q
git diff --check
```
