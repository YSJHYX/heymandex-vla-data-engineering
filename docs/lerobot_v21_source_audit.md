# D6 local source audit (2026-09-12)

This audit pins local evidence, not latest upstream documentation. Target:
Native OpenPI pi0.5; no HF operation, normalization-stat calculation or training.

## Environment

- Data Engineering baseline: `main`, `90c2c68`, with the uncommitted D5/D5.1
  delivery preserved; baseline 243 tests pass. Python 3.12, LeRobot absent.
- OpenPI: `/data/projects/vla_ws/openpi`, `main`,
  `5ffa00eecb1f3b86bc3e0dcf7d0ecccb74e517f7`. Pre-existing untracked benchmark,
  checkpoint and training helper files remain untouched.
- OpenPI interpreter: `.venv/bin/python`, Python 3.11.15; no environment sync.
- Installed LeRobot distribution: `0.1.0`; `direct_url.json` pins Git commit
  `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`.
- Implementation: `/data/projects/vla_ws/openpi/.venv/lib/python3.11/site-packages/lerobot/common/datasets/lerobot_dataset.py`.
  Line 77 defines `CODEBASE_VERSION = "v2.1"`.

## Writer and temporal contract

Evidence locations below are in that installed implementation unless specified.

| Evidence | Actual behavior | D6 decision |
| --- | --- | --- |
| `LeRobotDataset.create`, lines 988–1036 | `repo_id`, `root`, `fps`, `features`, `use_videos`; no finalize API required | Official create/add_frame/save_episode/stop_image_writer only |
| `add_frame`, lines 788–833 | Validates feature ndarray shape against tuple; defaults timestamp to `frame_index / fps` | Convert JSON feature shapes back to tuples for API; local frame indices and timestamps |
| `save_episode`, lines 835–910 | Resolves task strings into task indices, writes Parquet/stats/videos, checks timing, deletes its own temporary PNGs | One clean run per save_episode; no custom canonical metadata writer |
| `_get_query_indices`, lines 665–679 | Each delta index clamped to current episode start/end-1; supplies `*_is_pad` | Preserve all nonempty runs including length 1/2 |
| `__getitem__`, lines 724–749 | Resolves video from current episode, then `task_index` to `task` | Independent library reload checks task and both cameras |

`common/datasets/utils.py:47–57` sets chunk size 1000, and paths:

```text
meta/info.json
meta/tasks.jsonl
meta/episodes.jsonl
meta/episodes_stats.jsonl
data/chunk-000/episode_000000.parquet
videos/chunk-000/observation.images.head/episode_000000.mp4
videos/chunk-000/observation.images.wrist/episode_000000.mp4
```

System features are `timestamp` float32 and `frame_index`, `episode_index`,
`index`, `task_index` int64 (`utils.py:69`). `task` is a special add_frame
input, not a custom Parquet feature; the official reader resolves it to string.

`common/datasets/video_utils.py:245–327` implements `encode_video_frames` with
PyAV, default `libsvtav1`, `yuv420p`, GOP 2, CRF 30. It sorts temporary PNGs by
frame index, uses the first source image dimensions, creates stream at passed
FPS, flushes encoder. D6 keeps this official default. No source JPEG resizing or
re-encoding. Official default decoder resolves to installed `torchcodec
0.13.0+cu130`; full-stream PyAV decoding independently checks actual counts and
constant dimensions. The old `pyav` library backend fails with installed
torchvision `0.29.0.dev20260715+cu130` because `VideoReader` has been removed;
D6 uses the official default torchcodec backend, exactly as OpenPI does.

## Exact OpenPI loader

Paths below are under `/data/projects/vla_ws/openpi`.

- `src/openpi/training/data_loader.py:130–152`: `create_torch_dataset` constructs
  `LeRobotDatasetMetadata(repo_id)` then `LeRobotDataset(repo_id, episodes=...,
  delta_timestamps=...)`. D6 physical action key requires
  `DataConfig(action_sequence_keys=("action",))`; the default is `actions`.
- It forms action horizon at loading time: `[t / dataset_meta.fps for t in
  range(action_horizon)]`. Storage has one physical action per row. LeRobot
  clamps requests within each exported episode; no cross-run action is used.
- `prompt_from_task=True` wraps the dataset in
  `PromptFromLeRobotTask(dataset_meta.tasks)`; `src/openpi/transforms.py:310–324`
  maps `task_index` directly to prompt. This matches LeRobot's resolved task.
- `data_loader.py:173–192`: transform order is repack, robot data transforms,
  Normalize, model transforms. `training/config.py:132–144` for PI05 supplies
  default prompt, image resize to 224, tokenization, then `PadStatesAndActions`.
  `transforms.py:328–339` pads state/actions on the last axis. Native default
  model width is 32; physical storage remains 17.
- `scripts/compute_norm_stats.py:30–63,104–115`: computes stats after repack/data
  transforms, before normalization/model padding, using RunningStats over
  state/actions. D6 does not invoke it. Loader smoke uses
  `skip_norm_stats=True` (identity Normalize), then the actual padding transform;
  no stats are computed/loaded and no tokenizer/checkpoint is instantiated.

## Train/val representation

`LeRobotDataset.load_hf_dataset` lines 616–625 always requests HF Datasets
`split="train"`. Metadata writer line 261 overwrites info.splits to the entire
episode range under train. Exact OpenPI loader has no named split argument;
this checkout does provide `DataConfig.episode_indices` for explicit whole
episode selection. D6 chooses two independent local roots `train/` and `val/`,
each with official canonical metadata. No invented split column. Every derived
run inherits source-manifest split. Empty splits have no LeRobot directory and
are explicitly counted as zero in `export_summary.json`.

## Real source, test policy only

Current Curated `episode_000001`: `dataset_hz=30.0`, offsets
`[0,505,507,901]`, language_instruction `test`; 901 D3 clean transitions.
Therefore contiguous export lengths are 505, 2, 394. Export task is solely the
TEST_THRESHOLD manifest instruction, never Curated's `test`. Source physical
timestamps stay in Curated; export provenance records exact source row ranges.

The production confidence threshold is not frozen. Real D6 output must be
labeled `TEST_THRESHOLD / NOT_PRODUCTION_DATASET` and is not a final training set.
