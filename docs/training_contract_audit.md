# D10.1 — Training-contract-only validation audit

> **SUPERSEDED production guidance.** Historical audit snapshot. Since the simplified production mainline, D4/D5 and
> the semantic manifest are OPTIONAL only. Current production eligibility is
> D2 valid + D3 clean + exact collection task + non-synthetic source for export;
> private HF publication additionally requires validated LeRobot output and a private repo. See README.

Audit date: 2026-09-14. Data Engineering `main@90c2c68`; previous commit
`434a683`. D5–D7 uncommitted files pre-existed and are preserved. No README
existed at this baseline. No OpenPI, VisionProTeleop, RAW, HF or training writes.

Baseline: 299 passed, 1 opt-in skip, 3 failures. All three failures were mock D7
publication tests trying to create `/data/vla_d7_staging_*` on a read-only mount.
Only their test fixture now selects pytest's temporary root; production publisher
and HF behavior are unchanged. Baseline ruff and diff check passed.

## CURRENT HARD-GATE INVENTORY (before implementation)

Paths are relative to `src/vla_data/` unless otherwise noted. “Hard” means a
candidate/artifact cannot advance, not that source bytes are deleted.

| Field / condition | Where checked | Why checked | Before → D10.1 |
|---|---|---|---|
| Required RAW fields and common row shapes (6/11/17) | `cleaning/causal_sync.py:REQUIRED_FIELDS`, `_validate_raw_shapes` | Interpret physical data without guessing | hard → hard; missing whole field rejects episode, missing row rejects candidate |
| arm measured/hand SDK finite; positive source ts; component feedback flags | `synchronize_episode` | Measured/current state authority | hard → hard |
| All hand feedback modes == 7 | `synchronize_episode` | SDK numbers must mean joint positions | hard → hard |
| arm sent + hand effective + recorded robot command; finite/exact concatenation | `synchronize_episode` | Not desired/retargeted command | hard → hard |
| arm/hand command flags and robot composite validity | `synchronize_episode` | Accepted/fresh command proof; composite consistency | hard → hard |
| Component PRE strictly before earliest action; POST strictly after latest action | `_select_feedback`, `synchronize_episode` | Frozen causal transition definition, not a new model feature | hard → hard; no boundary/min-length policy change |
| Camera validity, timestamp > 0 and < action, frame index >= 0 | `_select_camera` | Actual freshness + enqueue/reference authority | hard → hard; not generic camera-health telemetry |
| Selected JPG exists | `_select_camera` | RGB source/reference integrity | hard candidate → hard candidate |
| Selected JPG decodes | `build_curated_v1._copy_selected_media` | Required training RGB | hard whole build → candidate-level check/cache first; publication still independently checks |
| No surviving transition | `build_curated_v1` | No trainable payload | no publication → unchanged |
| RAW hardware_execution/training_ready/dataset_status/F_fb status | builder `_metadata`, `_cleaning_report` | Historical provenance | diagnostic → diagnostic |
| arm_status/controller_error/sender_hz, generic health/rates/debug | not consumed by D2 sync | Not training-field authority | diagnostic/ignored → unchanged |
| Curated readiness/status consistency and presence | `validation/curated_v1.py` | Historical summary consistency, not proof of state/action | hard → DIAGNOSTIC_WARNING |
| RAW collection language instruction | builder + Curated validator | Exact training/inference task authority | hard non-empty; copied without rewrite |
| RAW source schema/path provenance presence | same metadata check | Historical source identity | diagnostic warning |
| Curated schema, field whitelist, widths/lengths, joint order/slices, units/representation | `validation/curated_v1.py` | Unambiguous persisted physical schema; prevent model/SDK payload pollution | hard → hard; no rewrite/salvage of corrupt persisted structures |
| Curated segment offsets/tick order/native PRE/ACTION/POST relationships | same validator | Temporal identity and no hidden discontinuity | hard → hard |
| Curated camera role/index/file correspondence; forbidden depth/extra trajectory payload | same validator | Frozen RGB-only storage schema and reference integrity | hard → hard |
| D3 temporal metric completeness/shape and configured exclusions | `quality/temporal.py` | Sample age/skew evidence; thresholds apply to training fields | hard if malformed/exclude configured; otherwise report/warn → unchanged |
| D3 RGB missing/decode/channel/resolution | `quality/visual.py` | Training image integrity | row mask hard; any false formerly rejected whole episode → partial mask can ACCEPT_WITH_WARNING |
| D3 visual anomaly thresholds | `quality/visual.py` | Configured training quality | warning or frozen episode exclusion by config → unchanged |
| D3 no visual-valid rows / D2 structural validation failure | `quality/evaluator.py` | No valid visual payload / cannot certify persisted contract | hard REJECT → unchanged |
| D3 explicit expert exclusion/static demonstration/no clean rows | `quality/evaluator.py`, `quality/trajectory.py` | Frozen episode-level expert-data exclusion | hard → unchanged |
| D4 annotation schema/provider completion | `annotation/eligibility.py`, `pipeline.py`, `schema.py` | Optional semantic research workflow only | hard inside D4; never a production-export gate |
| D5 confidence/final text | `verification/evaluator.py`, `schema.py`, `review.py` | Optional semantic review workflow only | hard inside D5; never a production-export gate |
| Legacy semantic manifest identities/split | `manifest/eligibility.py`, `validator.py`, `split.py` | Optional historical export compatibility | hard only when legacy `--manifest-root` mode is selected |
| Mainline D6 D2/D3/task/real-source/FPS/17D/run/split/reload | `export/plan.py`, `validator.py`, `_worker.py` | Correct storage and source correspondence | hard; no D4/D5 dependency |
| D7 canonical metadata/features/task/video layout, repo ID, remote conflict/reload | `publish/local.py`, `publisher.py` | Publication integrity; not RAW diagnostic policy | hard → unchanged |
| Paths/permissions/canonical IDs/unsafe overwrite/worker failure | batch discovery/runners, export/publish | Operational safety, not a data-quality label | operational failure → unchanged |

The frozen Curated schema's names remain unchanged. Presence-only provenance
checks no longer confer training eligibility; payload and temporal checks do.
No D3 thresholds, confidence N, camera arrangement, min run length, LR, batch or
freeze policy was changed. D3's mask computation is identical; only partial RGB
failure escalation was narrowed from episode to affected rows.

## Validity-field source evidence (read-only acquisition audit)

`/home/heymandex2025/VisionProTeleop/sg100_teleop/data_collection/synchronized_sampler.py`:

- `_age_and_valid`, lines 22–43: valid source is positive, within episode start
  and sampling tick, declared valid, age <= configured bound.
- Lines 147–170: arm/hand feedback/command validity uses this function. Dropping
  these flags would allow stale cached commands or pre-On samples.
- Lines 214–235: robot effective command/validity derived from actual arm sent
  and effective hand components. Canonical hand feedback and robot qpos validity
  are separate from the mode-7 SDK state used by this repo.
- Lines 247–299: camera validity includes bounded age and successful media enqueue.
  It is not equivalent to a generic device health flag.
- Lines 315–318: controller error/status/channel/sender rate are stored separately;
  none is used by this repo's D2 decision.

Thus `hand_feedback_modes`, component validity, camera validity remain semantic
hard gates. `hardware_execution=False` and F_fb pending are not rejection reasons.
No stale finite vector is promoted to a new sample, no command proxy is used as state.

## Read-only episode_000002 re-evaluation

Source `/home/heymandex2025/桌面/vla_raw/episode_000002.npz`, 814 rows.
NPZ SHA-256 `6285f887bccabe7dbbef38958c9e4ac6f821fe20e28cc9ea7fd63332d5547cc7`
matches the pre-existing D10 source hash. No downstream stage was rerun.

| Required evidence | Actual result |
|---|---|
| RM65 `arm_qpos_rad` | float64 [814,6], 0 finite rows, all NaN |
| RM65 `arm_qcmd_sent_rad` | float64 [814,6], 0 finite rows, all NaN |
| Arm state/action source timestamps | int64 [814], all 0 |
| Arm feedback/command validity | bool [814], 0 true |
| SG100 SDK feedback | float64 [814,11], 814 finite rows |
| All 11 modes == 7 | 626 rows; modes observed 0 and 7 |
| Mode-7 + feedback flag + positive timestamp | 625 rows, indices 1–625 |
| SG100 effective command | float64 [814,11], 709 finite rows, indices 105–813 |
| Hand accepted/fresh command flag | 523 true, indices 105–627 |
| Hand source timestamps positive | 709 rows |
| Stale held hand action | 186 rows age > recorded max_hand_command_age_ns=100000000 |
| Head / wrist valid flags | 813 / 813; indices 1–813 |
| Head / wrist source timestamps positive | 814 / 814 |
| D2 actual candidate/accepted | 813 / 0 (N-1 candidates, not 814) |

Current rejection counts: ACTION_ARM_COMMAND_FLAG_INVALID 813,
ACTION_HAND_COMMAND_FLAG_INVALID 290, ACTION_NONFINITE 813,
ACTION_ROBOT_COMMAND_FLAG_INVALID 813, ACTION_TIMESTAMP_NONPOSITIVE 813.
Counts overlap; these are not a sum of distinct rejected rows.

**Would TRAINING_CONTRACT_ONLY make this episode valid? No.** Required measured
arm6 and actual-sent arm6 do not exist. Hand/RGB evidence cannot fill that gap.
The old D10 lineage incorrectly described arm data as all-zero and attributed
hand rejection to F_fb pending; it also counted 814 candidates. This audit corrects
those descriptions without editing that historical artifact. `robot_qpos_17d_valid`
and `hand_feedback_canonical_valid` are not this D2 implementation's state authority.

## Documentation and execution safety findings

- README was absent; new README includes commands directly, not only links.
- Every Data Engineering subcommand help was executed. Existing RGB diagnostics
  did not provide synchronized valid/invalid videos; new `render-rgb-review` is
  read-only, deterministic, separate-output only, and never changes masks/JPGs.
- D8/D9 scripts lack argument parsers: `--help` is not safe. Source and existing
  local evidence were read, but neither script was executed. README explicitly
  labels their actual no-argument invocations historical, hardcoded TEST_ONLY.
- D7 upload uses default branch and does not manage Git tags/private/repo creation.
  `--revision` is not wired to publication; SKIPPED does not independently reload.
  These are documented limitations, not silently “fixed” in this stage.
- Production confidence threshold, dataset snapshot/card governance and production
  training hyperparameters remain decisions for formal data. D1–D9 historical PASS
  does not imply this fresh-source D10 end-to-end acceptance passed.

## Final local verification

- `ruff format src tests`: 101 files unchanged at final check.
- `ruff check src tests` and `git diff --check`: PASS.
- `pytest -v`: **317 passed, 1 skipped** (10.80 s). The skipped opt-in D6
  official LeRobot/OpenPI integration was intentionally not rerun in D10.1.
  GLM and publication unit tests use mocks, not live services.
- `uv build --offline`: wheel and sdist PASS. Fresh Python 3.12 wheel installed
  under `/tmp/heymandex_vla_d101_sD17yB/wheel-venv`, help/import run from `/tmp`.
- Actual ffmpeg/libx264 review smoke from synthetic Curated/D3 fixtures:
  valid/invalid/all decode to 2/1/3 frames respectively, 30 FPS; ffprobe PASS.
- Real RAW audit decoded all 814 referenced head and 814 wrist JPGs, each 640×480.
  All **1629 NPZ/JPG hashes unchanged** before/after read-only validation.
  Removing diagnostic fields in memory leaves the identical 813/0 D2 result.
- Evidence: `/tmp/heymandex_vla_d101_sD17yB/episode_000002_audit.json`,
  `final_tests.xml`, `video_smoke/episode_000123/*.json` and `*.mp4`.
- Temporary work/evidence/wheel/tests occupy about 128 MiB; root about 95% full
  (3.6 GiB available), `/data` about 402 GiB available. No user data was deleted.
- No git add/commit/push, no external API, no OpenPI training/stats/checkpoint
  mutation, no real downstream pipeline run.
