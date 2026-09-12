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
- `--force` — re-annotate even when a valid matching annotation exists
- `--dry-run` — report `WOULD_PROCESS` / `WOULD_SKIP` / `INELIGIBLE` per
  episode without any provider call (works without an API key)
- `--concurrency N` — bounded provider concurrency; the default `1` sends one
  request at a time. Raise deliberately once the account rate limit is known.
  Observed on the current account (2026-09-12, `glm-4.6v-flash`): a 429 rate
  limit window of roughly 1–4 minutes after a request, independent of image
  count — a 12-image request (6 head+wrist pairs, ~5.1k prompt tokens) was
  accepted once the window passed. Keep concurrency at 1 for large batches and
  expect bounded-retry `FAILED` episodes to succeed on a later re-run.
- `--max-temporal-points N` — keyframes per request (default 6 head+wrist
  pairs selected deterministically from clean transitions)

Behavior:

- Each annotation writes `annotations/<episode>/annotation.json` (schema
  `vla_episode_annotation` v1, review state `AUTO_LABELED`) plus
  `keyframes.json` provenance. Curated and RAW inputs are never modified.
- Resume: a valid annotation whose provider, model, and prompt version match
  the current run is `SKIPPED` with no new API spend. Invalid, corrupt, or
  version-mismatched annotations are re-annotated and flagged `stale` rather
  than silently skipped.
- One episode's provider failure (network, rate limit after bounded retries,
  invalid model output) fails only that episode; the batch continues and the
  command exits `1`. Retries are bounded (default 3 attempts, exponential
  backoff); 4xx configuration/auth errors are not retried.
- A dataset summary lands at `annotations/dataset_annotation_summary.json`
  with eligibility counts, review-state counts, mean confidence, invalid
  response count, and retry count.
- Human QC: reviewers may accept, verify, correct, or reject annotations via
  the `vla_data.annotation.qc` API (`review_annotation`, `build_review_plan`).
  The original model annotation is always preserved; corrections resolve
  `final_instruction` per the review state machine.
