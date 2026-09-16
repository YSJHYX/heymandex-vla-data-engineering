import json
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest
from PIL import Image

from vla_data.cleaning.build_curated_v1 import build_curated_v1
from vla_data.cleaning.causal_sync import synchronize_episode
from vla_data.io.curated_episode import CuratedEpisode
from vla_data.io.raw_episode import RawEpisode
from vla_data.validation.curated_v1 import validate_curated_episode


def changed(raw, **changes):
    return replace(raw, arrays=MappingProxyType({**raw.arrays, **changes}))


def test_bad_diagnostics_do_not_reject_required_payload(synthetic_raw_episode):
    raw = RawEpisode.load(synthetic_raw_episode)
    raw = changed(
        raw,
        arm_status=np.full(8, -1),
        arm_controller_error=np.full(8, -1),
        arm_sender_hz=np.zeros(8),
        hardware_execution=np.asarray(False),
        training_ready=np.asarray(False),
        dataset_status=np.asarray("BAD"),
    )
    assert [t.source_tick_index for t in synchronize_episode(raw).transitions] == [3, 4]


@pytest.mark.parametrize("field", ["arm_qpos_rad", "arm_qcmd_sent_rad"])
def test_missing_required_arm_rejects_despite_healthy_diagnostics(
    synthetic_raw_episode, field
):
    raw = RawEpisode.load(synthetic_raw_episode)
    raw = changed(
        raw,
        **{field: np.full_like(raw.arrays[field], np.nan)},
        hardware_execution=np.asarray(True),
        arm_status=np.zeros(8),
        training_ready=np.asarray(True),
        arm_controller_error=np.zeros(8),
    )
    assert not synchronize_episode(raw).transitions
    arrays = dict(raw.arrays)
    del arrays[field]
    with pytest.raises(KeyError, match=field):
        synchronize_episode(replace(raw, arrays=arrays))


def test_wrong_mode_and_stale_effective_command_remain_hard(synthetic_raw_episode):
    raw = RawEpisode.load(synthetic_raw_episode)
    assert not synchronize_episode(
        changed(raw, hand_feedback_modes=np.zeros((8, 11)))
    ).transitions
    flags = raw.arrays["hand_command_valid"].copy()
    flags[3] = False  # finite held command, but sampler acceptance/freshness lost
    result = synchronize_episode(changed(raw, hand_command_valid=flags))
    assert [t.source_tick_index for t in result.transitions] == [4]
    assert result.rejection_counts["ACTION_HAND_COMMAND_FLAG_INVALID"] >= 1


def test_zero_valid_rows_publish_nothing(synthetic_raw_episode, tmp_path):
    raw = RawEpisode.load(synthetic_raw_episode)
    np.savez(
        synthetic_raw_episode, **{**raw.arrays, "arm_qpos_rad": np.full((8, 6), np.nan)}
    )
    with pytest.raises(ValueError, match="no output published"):
        build_curated_v1(synthetic_raw_episode, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_corrupt_reference_rejects_only_affected_transition(
    synthetic_raw_episode, tmp_path
):
    raw = RawEpisode.load(synthetic_raw_episode)
    arrays = {k: a.copy() for k, a in raw.arrays.items()}
    arrays["head_camera_host_timestamp_ns"][3] = 36
    arrays["head_camera_valid"][3] = True
    arrays["head_rgb_frame_index"][3] = 3
    (raw.media_root / "head/rgb/000003.jpg").write_bytes(b"corrupt synthetic JPG")
    np.savez(synthetic_raw_episode, **arrays)
    result = synchronize_episode(RawEpisode.load(synthetic_raw_episode))
    assert [t.source_tick_index for t in result.transitions] == [3]
    assert result.rejection_counts["CAMERA_HEAD_CAMERA_DECODE_INVALID"] == 1
    out = build_curated_v1(synthetic_raw_episode, tmp_path / "out")
    assert CuratedEpisode.load(out).transition_count == 1


def test_readiness_summary_is_warning_but_task_instruction_is_required(
    curated_v1_episode,
):
    path = curated_v1_episode / "metadata.json"
    m = json.loads(path.read_text())
    m.update(hardware_execution=False, training_ready=False, dataset_status="UNKNOWN")
    del m["language_instruction"]
    path.write_text(json.dumps(m))
    report = validate_curated_episode(CuratedEpisode.load(curated_v1_episode))
    assert not report.passed
    assert "language_instruction must be a non-empty string" in report.errors
    assert report.warnings


def test_partial_d3_rgb_failure_preserves_clean_rows(synthetic_raw_episode, tmp_path):
    from vla_data.quality.evaluator import evaluate_episode

    raw = RawEpisode.load(synthetic_raw_episode)
    arrays = {k: a.copy() for k, a in raw.arrays.items()}
    arrays["head_camera_host_timestamp_ns"][3] = 36
    arrays["head_camera_valid"][3] = True
    arrays["head_rgb_frame_index"][3] = 3
    Image.new("RGB", (8, 6), "red").save(raw.media_root / "head/rgb/000003.jpg")
    np.savez(synthetic_raw_episode, **arrays)
    out = build_curated_v1(synthetic_raw_episode, tmp_path / "out")
    (out / "media/head/rgb/000003.jpg").write_bytes(b"synthetic post-D2 corruption")
    quality = evaluate_episode(out, tmp_path / "quality")
    assert quality.status == "ACCEPT_WITH_WARNING"
    assert np.load(quality.mask_path).tolist() == [True, False]
