# `vla-data` CLI reference

Production batch orchestration over episode-isolated inputs. The runners own
only discovery, dispatch, resume, parallelism, failure isolation, and dataset
summaries; every episode is processed by the frozen episode-level
implementations (`build_curated_v1`, `validate_curated_episode`,
`evaluate_episode`). RAW inputs are never modified.

## Commands

| Command | Purpose |
| --- | --- |
| `vla-data build-curated` | RAW `.npz` + `_media` siblings to Curated v1 episodes |
| `vla-data validate-curated` | Independent re-validation of Curated v1 episodes |
| `vla-data quality` | D3 quality evaluation per Curated episode |
| `vla-data run` | Orchestrate the stages above in one dataset command |
| `vla-data annotate` | GLM-4.6V-Flash model-first task annotation over Curated episodes |
| `vla-data verify-annotations` | Apply an explicit confidence policy after D3 quality eligibility |
| `vla-data review-annotation` | Human review of a D5.1 verification artifact; D4 is immutable |
| `vla-data build-manifest` | Resolve D5 training eligibility and episode splits without copying trajectories |
| `vla-data export-lerobot` | Local v2.1 Parquet/MP4 export, isolated by D2 segments and D3 clean runs |
| `vla-data publish-hf` | Publish one canonical split root; remote reads even during dry-run |
| `vla-data render-rgb-review` | Read-only synchronized head/wrist QC video from existing D3 mask |

The full operator workflow and D10.1 training-contract-only policy are in
[README](../README.md). RGB review uses `--curated-root`, `--quality-root`,
`--output-root`, `--episode`, `--selection valid|invalid|all`, and optional
`--dry-run`. Existing outputs are not overwritten. Empty selections emit JSON
only; missing images are explicitly skipped, never fabricated. Playback compacts
selected rows and is diagnostic-only, not a continuous training trajectory.

Exit codes (stable contract):

- `0` — command succeeded; warnings and expert-training exclusions allowed
- `1` — one or more episode-level failures (batch still processes every episode)
- `2` — global/configuration/input failure (bad paths, invalid `--workers`, ...)

## Common options

| Option | Meaning |
| --- | --- |
| `--input-root PATH` | Dataset root to discover (RAW root, or Curated root for downstream stages) |
| `--output-root PATH` / `--work-root PATH` | Derived-output root |
| `--episode episode_000001` | Restrict processing to one canonical episode ID (single-episode mode) |
| `--workers N` | Parallel episode workers; `1` = serial. Results are sorted by episode ID regardless |
| `--force` | Reprocess episodes even if valid derived outputs already exist |
| `--dry-run` | Report the plan (`WOULD_PROCESS`) without writing any artifact |
| `--verbose` | Print per-episode messages |
| `--expert-exclude EPISODE_ID` | Repeatable. Persists an explicit `EXCLUDE_FROM_EXPERT_TRAINING` status into the Curated metadata; the D3 evaluator honors it deterministically |

## Dataset mode

`run` discovers every `episode_<six-digit>.npz` under `--input-root` (with its
`episode_<id>_media` sibling directory) and executes the stages in order:

```bash
uv run vla-data run \
  --input-root /path/to/raw \
  --work-root /path/to/processed \
  --workers 2 \
  --expert-exclude episode_000000
```

Stage selection/ordering is configurable but must stay in canonical order:

```bash
uv run vla-data run --input-root RAW --work-root WORK --stages curated,quality
```

Each stage writes a compact dataset summary under `--work-root`:

```text
processed/curated/dataset_build_summary.json
processed/validation/dataset_validation_summary.json
processed/quality/dataset_quality_summary.json
```

Summaries contain counts, per-episode status/outcome, core paths, and error
summaries only — never full per-episode quality metrics. Validation and quality
read from `processed/curated/`; run `build-curated` first when invoking the
stage commands individually.

## Resume, idempotency, and force

- Re-running a stage with valid existing outputs marks those episodes
  `SKIPPED` and performs no recomputation. Curated outputs are re-validated on
  load; a schema mismatch or corrupt artifact triggers reprocessing.
- `--force` reprocesses selected episodes unconditionally. Outputs are
  published through a staging directory with atomic replacement, so an
  interrupted publish cannot leave a partial episode behind.
- Combine with `--episode` to force exactly one episode:

```bash
uv run vla-data run --input-root RAW --work-root WORK --episode episode_000001 --force
```

## Dry run

```bash
uv run vla-data run --input-root RAW --work-root WORK --dry-run
```

Reports discovered episodes and, per stage, `WOULD_PROCESS` /
`WOULD_SKIP` counts. When the work root does not exist yet, downstream stages
project the episodes the build stage plans to publish. No directory or file is
created.

## Failure isolation and determinism

One malformed episode becomes a `FAILED` result for that episode only; the
remaining episodes still process, and the command exits `1`. Worker counts do
not affect episode IDs, statuses, summary counts, or ordering — every stage
sorts results by canonical episode ID, and episodes never share outputs.

## Expert-training exclusion provenance

`episode_000000` of the real capture set is regression data whose hand
effective-command channel is restricted to a single joint by a historical
collection bug. It must be excluded from expert training by the explicit,
auditable flag above — never by inferring the history from motion patterns at
run time. The persisted metadata status is the single deterministic source the
D3 evaluator honors.

## Model-first annotation (`vla-data annotate`)

Annotates each Curated episode with one GLM-4.6V-Flash task instruction.
Only episodes whose D3 quality outcome is `ACCEPT` or `ACCEPT_WITH_WARNING`
(with at least one clean transition in the quality mask) are eligible;
`EXCLUDE_FROM_EXPERT_TRAINING`, `REJECT`, and failed quality runs never reach
the provider.

Credentials are read from the environment — `GLM_API_KEY` takes precedence
over `ZHIPU_API_KEY` — and are never written to artifacts, summaries, or logs:

```bash
export GLM_API_KEY="..."
```

```bash
vla-data annotate \
  --curated-root /data/vla/processed/curated \
  --quality-root /data/vla/processed/quality \
  --output-root /data/vla/processed/annotations
```

Options:

- `--episode episode_000001` — single-episode mode
- `--force` — reprocess even when a valid matching annotation exists; an exact
  compatible `provider_raw` response may be replayed without a network call
- `--dry-run` — report `WOULD_PROCESS` / `WOULD_SKIP` / `INELIGIBLE` per
  episode without any provider call (works without an API key)
- `--concurrency N` — bounded provider concurrency; the default `1` sends one
  request at a time. Raise deliberately once the account rate limit is known.
  Observed on the current account (2026-09-12, `glm-4.6v-flash`): a 429 rate
  limit window of roughly 1–4 minutes after a request, independent of image
  count — a 12-image request (6 head+wrist pairs, ~5.1k prompt tokens) was
  accepted once the window passed. Keep concurrency at 1 for large batches and
  expect bounded-retry `FAILED` episodes to succeed on a later re-run.
- `--max-temporal-points N` — Pass A keyframes (default 6 head+wrist pairs,
  selected deterministically across every D2/D3 clean domain)
- `--prompt-version v3.1` — embodiment-aware coarse semantics plus one-transition
  boundary-local refinement; renders the same platform context into both passes
- `--prompt-version v3.3` — retains v3.2 sparse recall and boundary-local behavior,
  while requiring explicit object nouns in canonical instructions and separating
  temporally distinct manipulation objectives

Behavior:

- Pass A infers the episode task and coarse semantic segments. With prompt v3.1,
  Pass B remains one independent provider call and one storyboard per candidate
  transition, refining only that local `[start,end)` boundary. The implementation
  writes `vla_episode_annotation` v2 plus `keyframes.json`; Curated and RAW inputs
  are never modified.
- The legal domain is the intersection of each D2 segment with contiguous D3
  true runs. Model boundaries must be exact provided Curated indices. Segments
  plus explicit non-training intervals classify every clean domain without
  overlap; no minimum segment length is imposed.
- Resume: a valid annotation whose provider, model, and prompt version match
  the current run is `SKIPPED` with no new API spend. Invalid, corrupt, or
  version-mismatched annotations are re-annotated and flagged `stale` rather
  than silently skipped.
- Successful provider responses are atomically persisted under
  `episode_XXXXXX/provider_raw/` before local language/schema validation. Replay
  requires exact input, request, prompt, provider, platform-context, and Pass-B
  mode compatibility; replay increments `replayed_provider_responses`, not
  `actual_provider_calls`.
- One episode's provider failure fails only that episode; the batch continues and
  the command exits `1`. Coding Plan Vision MCP retries a transient timeout,
  overload, network reset, or retryable transport failure at most once (two
  actual attempts per logical call). Auth, quota, schema, controlled-language,
  local storyboard/camera, and invalid-result failures are not retried.
- A dataset summary lands at `annotations/dataset_annotation_summary.json`
  with eligibility counts, review-state counts, mean confidence, invalid
  response count, and retry count.
- Human QC: reviewers may accept, verify, correct, or reject annotations via
  the `vla_data.annotation.qc` API (`review_annotation`, `build_review_plan`).
  The original model annotation is always preserved; corrections resolve
  `final_instruction` per the review state machine.

## Confidence-gated verification and training manifests (D5.1)

D4 model annotations remain immutable. D5.1 is a separate authority: for every
semantic segment after D3 quality eligibility, `confidence >= N`
(including equality) produces `AUTO_VERIFIED`; `confidence < N` produces
`NEEDS_HUMAN_REVIEW`. Comparison uses the recorded confidence without rounding.
`AUTO_VERIFIED` means machine-policy verified, not human verified.

`--confidence-threshold` is required, with a finite non-bool number in [0, 1].
There is no default production threshold. Changing it never calls GLM or
re-annotates. Provider code is loaded only for the D4 `annotate` command.

```bash
vla-data verify-annotations --annotation-root ANNOTATIONS --quality-root QUALITY \
  --output-root VERIFICATION --confidence-threshold "$N" --dry-run
```

Remove `--dry-run` to write. Supports `--episode`, `--force`, and an optional
`--policy-version` label (default `confidence_gate_v1`). Dry run writes nothing
and reports `would_auto_verify`, `would_require_human_review`, and `ineligible`.
Quality rejection/expert exclusion/zero clean transitions prevent verification,
even at confidence 1.0. Missing or corrupt per-episode artifacts become explicit
`INELIGIBLE` records. Missing roots/threshold and mixed policies are global errors.

Outputs are `episode_XXXXXX/verification.json`, `human_review_queue.jsonl`, and
`dataset_verification_summary.json`. Verification v2 stores policy, source paths
and hashes, model confidence/instruction, status/source, human review, resolved
final instruction and fingerprint. Queue contains only `NEEDS_HUMAN_REVIEW`,
with instruction, confidence, annotation path and keyframes path; no image copy
or base64. Summary includes state counts, confidence min/p10/p25/median/p75/p90/
p95/max/mean and projected workload at 0.70/0.80/0.90/0.95. The statistical
population is valid annotations after D3 quality eligibility; workload is the
pre-human-review confidence count, not a recommendation to select production N.

A human may verify the original text, correct it, or reject the verification:

```bash
vla-data review-annotation --verification-root VERIFICATION \
  --episode episode_000001 --semantic-segment-id SEGMENT \
  --status HUMAN_VERIFIED --reviewer REVIEWER

vla-data review-annotation --verification-root VERIFICATION \
  --episode episode_000001 --semantic-segment-id SEGMENT \
  --status HUMAN_CORRECTED --instruction "REVIEWED TEXT"

vla-data review-annotation --verification-root VERIFICATION \
  --episode episode_000001 --semantic-segment-id SEGMENT --status REJECTED
```

These commands modify verification and refresh its queue/summary, never D4.
Omit `--semantic-segment-id` to review the independently stored episode task;
boundary flags apply only to a semantic segment.
`HUMAN_VERIFIED` resolves to the original model instruction; `HUMAN_CORRECTED`
requires `--instruction` and preserves its exact text, including whitespace;
`REJECTED` resolves to null. Invalid/self transitions are rejected. Optional
human audit of an auto-verified label is also supported. Never run real human
review commands without the user's explicit decision.

`AUTO_ACCEPTED` and `annotation_policy.auto_accept_enabled` are deprecated as
production authority. Legacy D4 artifacts remain readable, but D5.1 never emits
`AUTO_ACCEPTED` and never uses D4 review/final_instruction to select training data.
Old D5 manifest v1 is stale; the verification-backed manifest schema is v2.

```bash
vla-data build-manifest --curated-root CURATED --quality-root QUALITY \
  --annotation-root ANNOTATIONS --verification-root VERIFICATION --output-root MANIFEST \
  --seed 17 --validation-fraction 0.2 --dry-run
```

Remove `--dry-run` to write. The fraction above is an example, not a production
recommendation. Default fraction is `0` (all eligible episodes train); choose an
explicit fraction for an actual experiment. `--episode` restricts discovery;
`--force` bypasses resume. Dry run reports discovered, eligible, needs-review,
excluded, would-train (`train_episodes`) and would-validate (`validation_episodes`)
counts without creating files/directories.

Eligibility requires successful independent Curated validation, exact 17D
state/action, eligible D3 quality, at least one clean D3 transition, valid D4
annotation and current verification with a resolved non-empty instruction.
Explicit expert-training exclusion overrides verification. Approved verification
states are `AUTO_VERIFIED`, `HUMAN_VERIFIED`, and `HUMAN_CORRECTED`.
`AUTO_LABELED`, `NEEDS_HUMAN_REVIEW`, `REJECTED`, missing/stale verification and
ineligible inputs cannot enter training. Manifest instruction comes exclusively
from verification, with exact text preserved. `--verification-root` is required.

Outputs (all rows sorted by canonical episode ID):

- `episodes.jsonl`: eligible episodes, each wholly assigned to train or val.
- `train.jsonl`, `val.jsonl`: exact disjoint partitions of eligible episodes.
- `excluded.jsonl`: every other discovered episode, including `NEEDS_REVIEW`,
  with explicit reason codes and artifact-error details.
- `dataset_manifest_summary.json`: counts, status distributions, split config,
  input roots/fingerprint and split limitations.

The union of canonical episode directories in the four input roots is audited;
missing/corrupt per-episode inputs become exclusions. Missing dataset roots and
noncanonical episode directories are global input errors. Summary categories
are disjoint: discovered = training_eligible + needs_review + excluded.
Expected exclusions are a successful command (exit 0); global errors exit 2.

Each manifest row references Curated, quality, annotation, verification and the existing
`training_transition_mask_path` (D3 `quality_mask.npy`). That bool mask is the
exact training selection: no excluded transitions are restored. No trajectory,
images, encoded video or padded state/action are written. Optional `task_type`
comes from the existing model annotation, and session/date/group fields from
Curated metadata; missing values remain null.

Split ranking uses SHA-256 of `seed:episode_id`; `floor(N * validation_fraction)`
episodes go to val. No minimum validation episode is forced for tiny datasets.
Same episodes/config produce identical outputs across processes and filesystem
order. Episode-level random split does not guarantee scene/task independence.
`SplitConfig` reserves `strategy`/`group_key`, but rejects group-aware requests
until that strategy exists. Related sessions and near-duplicates still require
future handling.

Resume requires independently valid outputs and the same input/config
fingerprint. The fingerprint includes episode IDs, resolved roots, manifest
version, split config, content hashes of quality report/mask, annotation, verification
(including confidence threshold, policy version and human review state),
Curated metadata/cleaning report, and trajectory stat identity. It never hashes
all JPG/NPZ bytes. Inputs are revalidated on resume; annotation/review, quality,
config or episode changes rebuild. Output roots must be separate from inputs.
Files are staged and replaced individually, summary last; interrupted or mixed
sets fail validation and rebuild. This is not a concurrent-writer protocol;
run one manifest writer per output root.

Independent validation is available as a package API:

```python
from vla_data.manifest import validate_training_manifest

result = validate_training_manifest("MANIFEST")
assert result.passed, result.errors
```

It rereads referenced sources, checks unique IDs, complete accounting, split
partitions/no overlap, approval/instruction equality, 17D, mask/count agreement,
summary counts and fingerprint. No LeRobot/HF or model training runs in D5.

Verification has a source fingerprint (annotation/quality hashes, threshold,
policy version, schema version) and a full fingerprint including human state.
Threshold/version changes recompute auto decisions. Force recomputes but does
not erase a valid human decision over unchanged source artifacts; such decisions
are rebound to the new policy. Annotation/quality changes invalidate prior review
and resolve afresh under the explicit policy. Manifest then rebuilds on the new
verification hash. A single output root must not mix policies: threshold changes
require a full dataset run, not `--episode`. Use one writer per output root;
individual files are atomically replaced, but the whole dataset is not a database
transaction. Rerunning reconstructs the queue/summary after interruption.

`AutoVerificationAuditConfig(random_audit_fraction=0.0, seed=0)` reserves optional
seeded audit sampling. Nonzero values report audit episode IDs separately and
never block `AUTO_VERIFIED` or add them to the mandatory human review queue.

Without a user-chosen production threshold, real boundary demonstrations must
be labeled `TEST_THRESHOLD / NOT_PRODUCTION_POLICY`. A successful demonstration
does not freeze a production policy or production manifest.

## Local LeRobot v2.1 export (D6)

The audited writer/loader contract is recorded in
[lerobot_v21_source_audit.md](lerobot_v21_source_audit.md). The lightweight
Data Engineering package does not install LeRobot, torch or video dependencies.
Use an existing compatible interpreter; on this workstation the audited
environment belongs to OpenPI (Python 3.11), while this CLI uses Python 3.12.
A standalone worker bridges these environments without modifying either.

```bash
vla-data export-lerobot --manifest-root MANIFEST --output-root EXPORT \
  --dataset-name rm65-sg100 \
  --lerobot-python /data/projects/vla_ws/openpi/.venv/bin/python --dry-run
```

Dry run validates the manifest and inspects source metadata/image headers, then
reports source counts, split counts, derived run counts, selected transitions,
camera features, width 17, task count and FPS. It creates no output or worker
cache. Remove `--dry-run` to write. If no interpreter is supplied the current
Python is used; a missing/incompatible LeRobot dependency yields an actionable
error explaining `--lerobot-python`, never an automatic installation.

For v2 hierarchical annotations, each verified semantic segment is exactly one
LeRobot episode. D3 holes and D2 boundaries cannot be crossed. A split is first
assigned once to the source episode, then inherited by all of its semantic
training units. No minimum length is invented: the audited loader clamps horizons
to the semantic episode boundary and provides padding masks. Legacy v1 manifests
remain readable and retain their historical contiguous-clean-run export behavior.

Outputs:

```text
EXPORT/
  train/  # independent canonical LeRobot v2.1 dataset, if nonempty
  val/    # independent canonical LeRobot v2.1 dataset, if nonempty
  export_provenance.jsonl
  export_summary.json
  export_validation.json
```

Each nonempty split is created using official `LeRobotDataset.create`,
`add_frame`, `save_episode`. Canonical Parquet/video/metadata is never handwritten.
Empty splits have no dataset directory and are counted explicitly in the export
summary. The current OpenPI path has episode-index selection but no named split
selector, so two datasets are used rather than a custom split column.

Features are `observation.state` and `action` float32 `[17]`, plus two independent
video features `observation.images.head` / `observation.images.wrist`. Official
system fields remain intact. `task` is resolved by the official task table and
must equal `manifest.final_instruction` exactly. No fallback to RAW/Curated/D4
language is allowed; confidence/provider/verification fields never become model
features. The only numerical conversion is float64-to-float32 where needed;
cast errors are reported. Units stay rad; no normalization/clipping/padding.

FPS is required from each source's `dataset_hz` metadata and must be a consistent
positive integer for this writer. Missing or mixed FPS is rejected. Each run has
local frame index 0..N-1 and timestamps frame_index/FPS. Physical timestamps stay
in Curated, with exact row/segment references in provenance. Source image paths
come from the causal Curated references. Native per-camera dimensions are kept;
no source JPG is changed. The official encoder produces MP4s from temporary PNGs.

Resume requires unchanged manifest/config/source identities, matching writer
runtime and an independent validator PASS; otherwise it rebuilds. `--force`
bypasses skip. New outputs are fully written and independently validated in a
sibling staging directory before rename publication, with rollback around
replacement. Failed exports leave existing valid output intact. Output must be
separate from input trees; unmanaged directories and symlinks are not replaced.
Use one writer per output root. Fingerprints hash manifest/quality metadata and
record source trajectory/JPG stat identities, not every source JPG byte.

```python
from vla_data.export import validate_lerobot_export

result = validate_lerobot_export(
    "EXPORT", lerobot_python="/data/projects/vla_ws/openpi/.venv/bin/python"
)
assert result.passed, result.errors
```

The independent validator reconstructs authorized source runs, reloads each
dataset using the official library, checks every state/action/task row, canonical
features/counts, no source split leakage, and video counts/resolutions. Full
PyAV video decoding checks first/middle/last frame correspondence, recording MAE
and relative alternative-frame/camera differences without imposing an arbitrary
absolute visual-quality threshold. Official `__getitem__` is also exercised.

All worker execution is local with HF offline settings and network connections
disabled. No HF token, login, upload, push or OpenPI normalization/training occurs.
Before real exports check available disk; real smoke uses `/tmp/heymandex_vla_d6_*`
and carries `TEST_THRESHOLD / NOT_PRODUCTION_DATASET` until production N is chosen.

Optional full integration regression with the existing environment:

```bash
VLA_LEROBOT_PYTHON=/data/projects/vla_ws/openpi/.venv/bin/python uv run pytest -v tests/export
```

## Hugging Face publication (`vla-data publish-hf`)

Publication-only upload of a validated D6 LeRobot export to a Hugging Face
dataset repository. Canonical files are copied byte-for-byte (never re-encoded
or re-generated), staged, and committed in a single `upload_folder` commit;
a TEST_THRESHOLD README is added when the remote has none.

```bash
vla-data publish-hf \
  --dataset-root /data/vla/processed/export/train \
  --repo-id PPPPPilot/VLADexData \
  --hf-python /path/to/python-with-huggingface_hub \
  --lerobot-python /path/to/python-with-lerobot \
  --cache-dir /data/hf_cache
```

Behavior:

- `--dry-run` reports repo id, local file/byte counts, local fingerprint,
  remote state, and would-upload / would-skip / blocked — with zero writes.
- Authentication uses the official Hugging Face credential mechanisms
  (`HF_TOKEN` or cached login); no token argument exists, and tokens never
  appear in evidence or logs.
- Safety: an upload is refused when the remote repository contains files that
  are not part of this dataset (`REMOTE REPOSITORY NOT EMPTY`); unknown data
  is never deleted or overwritten. `.gitattributes` and `README.md` are
  ignored as HF-managed files.
- Resume: when every canonical remote file matches the local manifest
  (path, size, and LFS SHA-256), re-running reports `SKIPPED` with zero
  re-upload. A changed local fingerprint means `UPLOADED` again. `--force`
  re-uploads known same-test data only.
- After a successful upload the command re-downloads the exact commit SHA
  into a fresh cache and validates: per-file SHA-256 equality against the
  local canonical manifest plus an official `LeRobotDataset` reload
  (episodes/frames/task/17D float32 state and action bitwise-equal to the
  local dataset, head/wrist MP4 frame counts equal to Parquet rows).
  Pass `--skip-remote-validation` to disable.
