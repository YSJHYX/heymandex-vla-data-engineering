from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from vla_data.batch.runner import build_curated_dataset
from vla_data.cleaning.build_curated_v1 import build_curated_v1
from vla_data.cleaning.causal_sync import synchronize_episode
from vla_data.export.plan import contiguous_runs
from vla_data.io.curated_episode import CuratedEpisode
from vla_data.io.raw_episode import RawEpisode
from vla_data.quality.evaluator import evaluate_curated_episode


def _raw(tmp_path, *, episode_id="episode_000001"):
    path = tmp_path / f"{episode_id}.npz"
    media = tmp_path / f"{episode_id}_media"
    timestamps = np.arange(1, 9, dtype=np.int64) * 100
    for role in ("head", "right_wrist"):
        directory = media / role / "rgb"
        directory.mkdir(parents=True)
        for index in range(8):
            Image.fromarray(np.full((4, 4, 3), index + 20, dtype=np.uint8)).save(
                directory / f"{index:06d}.jpg"
            )
    state = np.arange(8 * 17, dtype=np.float64).reshape(8, 17) / 100
    action = state + 1.0
    valid = np.array([True] * 7 + [False])
    arm_ts = timestamps + 10
    hand_ts = timestamps + 12
    arrays = {
        "timestamp_ns": timestamps,
        "arm_qpos_rad": state[:, :6],
        "arm_qpos_source_timestamp_ns": timestamps - 10,
        "arm_feedback_valid": np.ones(8, dtype=bool),
        "hand_feedback_sdk_rad": state[:, 6:],
        "hand_feedback_modes": np.full((8, 11), 7),
        "hand_feedback_source_timestamp_ns": timestamps - 9,
        "hand_feedback_valid": np.ones(8, dtype=bool),
        "arm_qcmd_sent_rad": action[:, :6],
        "arm_qcmd_source_timestamp_ns": arm_ts,
        "arm_command_valid": valid,
        "hand_qcmd_effective_canonical_rad": action[:, 6:],
        "hand_qcmd_source_timestamp_ns": hand_ts,
        "hand_command_valid": valid,
        "robot_qcmd_17d_rad": action,
        "robot_qcmd_17d_valid": valid,
        "head_camera_host_timestamp_ns": timestamps - 5,
        "head_camera_valid": np.ones(8, dtype=bool),
        "head_rgb_frame_index": np.arange(8),
        "wrist_camera_host_timestamp_ns": timestamps - 4,
        "wrist_camera_valid": np.ones(8, dtype=bool),
        "wrist_rgb_frame_index": np.arange(8),
        "schema_name": np.asarray("synthetic_raw"),
        "schema_version": np.asarray(3),
        "language_instruction": np.asarray("synthetic task"),
        "dataset_hz": np.asarray(30.0),
    }
    np.savez(path, **arrays)
    return path


def _annotate(path, intervals, flag=None):
    metadata = {
        "operator_intervention": bool(intervals) if flag is None else flag,
        "operator_interventions": [
            {"start_timestamp_ns": start, "end_timestamp_ns": end}
            for start, end in intervals
        ],
    }
    (path.with_name(f"{path.stem}_media") / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )


def _ticks(path):
    return [
        t.source_tick_index
        for t in synchronize_episode(RawEpisode.load(path)).transitions
    ]


def test_legacy_output_is_identical_on_rebuild(tmp_path):
    path = _raw(tmp_path)
    first = build_curated_v1(path, tmp_path / "first")
    second = build_curated_v1(path, tmp_path / "second")
    for name in ("metadata.json", "cleaning_report.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert "clean_runs" not in CuratedEpisode.load(first).metadata
    assert "intervention_stats" not in json.loads(
        (first / "cleaning_report.json").read_text()
    )
    assert _ticks(path) == list(range(7))


def test_explicit_clean_sidecar_keeps_transitions_and_adds_provenance(tmp_path):
    path = _raw(tmp_path)
    before = _ticks(path)
    _annotate(path, [])
    assert RawEpisode.load(path).metadata["operator_intervention"] is False
    assert _ticks(path) == before
    out = build_curated_v1(path, tmp_path / "curated")
    run = CuratedEpisode.load(out).metadata["clean_runs"][0]
    assert run == {
        "source_raw_episode": path.stem,
        "source_segment_index": 0,
        "segment_start_timestamp_ns": 100,
        "segment_end_timestamp_ns": 800,
        "segment_origin": "full_clean",
    }


def test_closed_interval_splits_and_preserves_suffix(tmp_path):
    path = _raw(tmp_path)
    _annotate(path, [(300, 400)])
    assert _ticks(path) == [0, 4, 5, 6]
    out = build_curated_v1(path, tmp_path / "curated")
    episode = CuratedEpisode.load(out)
    assert episode.segment_offsets == (0, 1, 4)
    assert [run["segment_origin"] for run in episode.metadata["clean_runs"]] == [
        "pre_intervention",
        "post_intervention",
    ]
    assert episode.state.shape == (4, 17)
    assert episode.action.shape == (4, 17)
    quality = evaluate_curated_episode(out, tmp_path / "quality")
    assert np.load(quality.mask_path, allow_pickle=False).shape == (4,)
    assert contiguous_runs(episode.segment_offsets, np.ones(4, dtype=bool)) == [
        (0, 0, 0, 1),
        (1, 0, 1, 4),
    ]
    assert (
        json.loads((out / "cleaning_report.json").read_text())["intervention_stats"][
            "contaminated_frames"
        ]
        == 2
    )


def test_open_final_interval_keeps_only_prefix(tmp_path):
    path = _raw(tmp_path)
    _annotate(path, [(300, None)])
    assert _ticks(path) == [0]
    out = build_curated_v1(path, tmp_path / "curated")
    assert [
        r["segment_origin"] for r in CuratedEpisode.load(out).metadata["clean_runs"]
    ] == ["pre_intervention"]


def test_multiple_intervals_make_three_independent_runs(tmp_path):
    path = _raw(tmp_path)
    _annotate(path, [(250, 260), (550, 560)])
    assert _ticks(path) == [0, 2, 3, 5, 6]
    out = build_curated_v1(path, tmp_path / "curated")
    episode = CuratedEpisode.load(out)
    assert episode.segment_offsets == (0, 1, 3, 5)
    assert [r["segment_origin"] for r in episode.metadata["clean_runs"]] == [
        "pre_intervention",
        "between_interventions",
        "post_intervention",
    ]


def test_interval_between_samples_breaks_causal_stitching(tmp_path):
    path = _raw(tmp_path)
    _annotate(path, [(250, 260)])
    sync = synchronize_episode(RawEpisode.load(path))
    assert not sync.intervention.invalid_mask.any()
    assert [t.source_tick_index for t in sync.transitions] == [0, 2, 3, 4, 5, 6]
    assert "INTERVENTION_CONTAMINATED" in sync.rejected[0].reasons


def test_suffix_does_not_borrow_pre_intervention_camera(tmp_path):
    path = _raw(tmp_path)
    _annotate(path, [(250, 260)])
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    arrays["head_camera_host_timestamp_ns"][2] = 240
    np.savez(path, **arrays)
    assert _ticks(path) == [0, 3, 4, 5, 6]


def test_partial_sidecar_metadata_fails_closed(tmp_path):
    path = _raw(tmp_path)
    (path.with_name(f"{path.stem}_media") / "metadata.json").write_text(
        '{"operator_intervention": true}', encoding="utf-8"
    )
    sync = synchronize_episode(RawEpisode.load(path))
    assert not sync.transitions
    assert sync.rejection_counts["INTERVENTION_METADATA_INVALID"] == 7


@pytest.mark.parametrize(
    ("intervals", "flag"),
    [
        ([(250, 260)], False),
        ([], True),
        ([(300, 200)], True),
        ([(300, 300)], True),
        ([(200, 400), (300, 500)], True),
        ([(200, 300), (300, 400)], True),
        ([(400, 500), (200, 300)], True),
        ([(200, None), (500, 600)], True),
    ],
)
def test_invalid_metadata_rejects_entire_episode(tmp_path, intervals, flag):
    path = _raw(tmp_path)
    _annotate(path, intervals, flag=flag)
    sync = synchronize_episode(RawEpisode.load(path))
    assert not sync.transitions
    assert sync.rejection_counts == {"INTERVENTION_METADATA_INVALID": 7}
    with pytest.raises(ValueError, match="INTERVENTION_METADATA_INVALID"):
        build_curated_v1(path, tmp_path / "curated")


def test_batch_stats_failure_isolation_and_annotation_refresh(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    first = _raw(root, episode_id="episode_000001")
    second = _raw(root, episode_id="episode_000002")
    _raw(root, episode_id="episode_000003")
    _annotate(first, [(300, 400)])
    _annotate(second, [(300, 200)])
    result = build_curated_dataset(root, tmp_path / "out")
    assert [(r.episode_id, r.status) for r in result.results] == [
        ("episode_000001", "SUCCESS"),
        ("episode_000002", "FAILED"),
        ("episode_000003", "SUCCESS"),
    ]
    assert "INTERVENTION_METADATA_INVALID" in result.results[1].message
    assert result.summary["intervention_stats"] == {
        "raw_episodes_with_intervention": 2,
        "intervention_intervals": 2,
        "contaminated_frames": 2,
        "clean_runs_recovered": 3,
        "intervention_metadata_invalid_episodes": 1,
        "runs_by_origin": {
            "full_clean": 1,
            "pre_intervention": 1,
            "between_interventions": 0,
            "post_intervention": 1,
        },
    }
    _annotate(first, [(250, 260)])
    refreshed = build_curated_dataset(root, tmp_path / "out")
    assert refreshed.results[0].status == "SUCCESS"
    assert refreshed.results[2].status == "SKIPPED"
    assert CuratedEpisode.load(tmp_path / "out" / first.stem).segment_offsets == (
        0,
        1,
        6,
    )
