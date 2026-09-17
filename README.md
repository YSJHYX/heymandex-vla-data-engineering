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
export UV_CACHE_DIR=/tmp/vla-data-uv-cache
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

The direct exporter reads D2/D3 only. It rejects explicitly synthetic sources (from captured RAW source-kind evidence in the D2 cleaning report), empty tasks, invalid D2, rejected D3, and zero clean runs. All derived runs of a `source_episode_id` remain in the same train/val split. `export_summary.json`, `export_provenance.jsonl`, and split-local `meta/source_provenance.jsonl` retain source episode ID, exclusive source range, exact task and SHA-256, source dataset status, and expert-training status. Those status fields are diagnostic only; annotation and D5 files are neither read nor required.

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

HF publication is cumulative. Dry-run reruns local `validate-lerobot`, pins and fresh-downloads the existing private repo, validates its complete LeRobot dataset, and reports `BASELINE`, `INCOMING`, `DUPLICATES`, `TO_APPEND`, and `MERGED_TOTAL`. It never uploads. Missing legacy source provenance, unknown remote files, or an invalid baseline fail closed. `BLOCKED` is not permission to remove `--dry-run`.

The current cumulative dataset is `PPPPPilot/VLADexData`. Its **initial valid cumulative baseline** is commit `2347702eed03bb3b4a54c1f4598c47c98e804e45`: four episodes, 1055 frames, lengths 343/116/286/310, with exact task `grab the ball` and source runs `episode_000003`–`episode_000006`. The earlier commit `b786109f06c985069c57118f5a815eedef684a2e` is `LEGACY_TEST_ONLY` and `NOT_PART_OF_CURRENT_CUMULATIVE_DATASET`: its three historical episodes are not restored or merged. Historical revisions can contain test datasets predating the current provenance/task contract; they are not automatically recovered into the current dataset. These SHAs document history, not a hardcoded publisher baseline.

Standalone exports usually both start at `chunk-000`; uploading today's files over yesterday's paths would overwrite episodes. The normal publisher uses the **current valid remote HEAD** as its only baseline, merges new logical episodes, and rebuilds a complete LeRobot dataset with the official writer. It does not walk backward through historical commits. Each maximal clean rollout remains a separate episode; the same exact task can belong to many episodes. Starting from four episodes, publishing ten new episodes yields 14; publishing 15 more yields 29, not 15. An exact rerun has `TO_APPEND=0`, returns `NO_NEW_EPISODES`, and makes no new commit, even with `--force`. Historical recovery is an exceptional, explicit maintenance/debug operation outside the normal publication path.

Any real dataset that completes D2, D3, LeRobot export, and local validation is eligible for private HF publication. `expert_training_status`, `training_use_status`, and `source_dataset_status` remain provenance only: `REVIEW_REQUIRED`, missing or arbitrary expert status, and `RAW_CAPTURE_QUARANTINED` do not veto valid data. The current 003–006 dataset follows this same path (4 episodes, 1055 frames); no separate approval or publication-purpose mode exists.

## 6. Private-HF publish — MUTATING NETWORK OPERATION

After inspecting a successful dry-run and obtaining authorization for the mutating network operation, run the same command without `--dry-run`. Do **not** run it merely because it appears in this README.

```bash
uv run --no-sync vla-data publish-hf \
  --dataset-root "$EXPORT_ROOT/train" --repo-id "$HF_REPO" \
  --hf-python "$HF_PYTHON" --lerobot-python "$LEROBOT_PYTHON"
```

The publisher rejects public/nonexistent repos, unknown remote files, and old datasets without verifiable source-run provenance; `--force` cannot bypass these gates or duplicate episodes. It logically reads old and new episodes, rebuilds all Parquet/video/metadata in isolated local staging, independently validates the merged result, then uploads it in one commit pinned to the observed parent SHA. The incoming export and downloaded baseline are never edited. Video bytes may change from re-encoding; state/action values, exact tasks, episode boundaries, and source identities must not. The new commit receives a pinned fresh-download/hash/official-reload check. Upload alone is not completed publication if that check fails.

## 7. Independent fresh-download acceptance

The real publish command performs this check automatically against the validated merged staging dataset. An independent `--revision` recheck remains read-only, but its `--dataset-root` must point to a retained copy of that **merged** dataset, not the standalone incoming export. Set `HF_REVISION` to the exact commit SHA, not `main` or the moving tag. The verifier requires an absent or empty cache path:

```bash
HF_REVISION=REPLACE_WITH_EXACT_COMMIT_SHA
uv run --no-sync vla-data publish-hf \
  --dataset-root REPLACE_WITH_RETAINED_MERGED_ROOT --repo-id "$HF_REPO" \
  --hf-python "$HF_PYTHON" --lerobot-python "$LEROBOT_PYTHON" \
  --revision "$HF_REVISION" \
  --cache-dir "$(mktemp -d -p "$WORK_ROOT" hf_verify_XXXXXXXX)"
```

Acceptance requires the pinned revision to resolve exactly and remain private, all canonical files to match the locally rebuilt merged dataset, and official LeRobot reload of the fresh snapshot. The reload compares episode/frame counts, 17D float32 arrays, both RGB camera features and per-episode video frames, exact task strings/hashes, provenance, and exclusive source ranges. Keep the exact commit SHA and validation evidence.

## Eligibility, optional tooling, and boundary

HF eligibility is D2 valid, D3 accepted with a nonempty clean run, nonempty exact task, non-synthetic, and independently validated LeRobot v2.1 with 17D state/action and both cameras. Descriptive dataset statistics may accompany validation; OpenPI-specific normalization, model-side 17→32 padding, training, checkpoints, inference, and robot safety are downstream responsibilities. GPU availability is not a data-engineering gate.

D4 annotation, D5 verification, LIBERO benchmarks, and the legacy semantic-manifest export remain optional research/diagnostic tools. They are not inputs to direct export, validation, or production eligibility. See [CLI reference](docs/cli.md) and historical audit notes for details. `scripts/smoke_openpi_lerobot.py` remains an optional downstream compatibility check only; it is never a publication prerequisite. Its prior local evidence does not establish remote HF or training readiness.

For local development checks:

```bash
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync pytest -q
git diff --check
```
