from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from vla_data.diagnostics import d2_episode
from vla_data.diagnostics.d2_episode import (
    clear_diagnostics_cache,
    diagnose_episode,
)


def _frozen_head(path: Path) -> None:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    rows = arrays["timestamp_ns"].size
    arrays["timestamp_ns"] = (
        2_000_000_000_000 + np.arange(rows, dtype=np.int64) * 1_000_000_000
    )
    arrays["head_camera_host_timestamp_ns"] = np.full(rows, 20, dtype=np.int64)
    arrays["head_camera_valid"] = np.zeros(rows, dtype=bool)
    arrays["head_rgb_frame_index"] = np.full(rows, 2, dtype=np.int64)
    np.savez(path, **arrays)


def test_frozen_head_root_cause_keeps_d2_reason_separate(
    synthetic_raw_episode: Path,
) -> None:
    _frozen_head(synthetic_raw_episode)
    clear_diagnostics_cache()

    result = diagnose_episode(run_name="synthetic", raw_path=synthetic_raw_episode)
    d2 = result["d2_evidence"]
    raw = result["raw_diagnostics"]

    assert d2["aggregate_source"] == "synchronize_episode(read-only)"
    assert d2["rejection_counts"]["CAMERA_HEAD_CAMERA_UNAVAILABLE"] == 2
    assert d2["primary_failure_reason"] == "CAMERA_HEAD_CAMERA_UNAVAILABLE"
    assert "HEAD_CAMERA_FROZEN" not in d2["rejection_counts"]
    assert raw["camera_health"]["head"]["status"] == "FROZEN"
    assert raw["inferred_root_cause"]["code"] == "HEAD_CAMERA_FROZEN"
    assert raw["feedback_health"]["pre_feedback_availability"]["total"] == 2
    assert raw["why_episode_rejected"][-1] == (
        "Therefore no transition could be published."
    )
    assert d2["exclusive_first_failure_funnel"] is None
    assert d2["rejection_groups_may_overlap"] is True
    assert {row["sample_position"] for row in d2["representative_rejections"]} == {
        "first",
        "middle",
        "last",
    }
    camera_samples = d2["representative_rejections_by_reason"][
        "CAMERA_HEAD_CAMERA_UNAVAILABLE"
    ]
    assert camera_samples[0]["raw_context"]["head"]["valid"] is False


def test_existing_cleaning_report_is_aggregate_authority(
    synthetic_raw_episode: Path, tmp_path: Path
) -> None:
    report = tmp_path / "cleaning_report.json"
    report.write_text(
        """{
          "candidate_count": 7,
          "accepted_count": 2,
          "rejected_count": 5,
          "rejection_counts": {"ACTION_NONFINITE": 5},
          "rejection_group_counts": {
            "ACTION": 5, "FEEDBACK": 0, "CAUSALITY": 0, "CAMERA": 0
          }
        }"""
    )
    clear_diagnostics_cache()

    result = diagnose_episode(
        run_name="synthetic",
        raw_path=synthetic_raw_episode,
        cleaning_report_path=report,
        d2_status="SUCCESS",
    )

    assert result["d2_evidence"]["aggregate_source"] == "cleaning_report.json"
    assert result["d2_evidence"]["accepted_count"] == 2
    assert (
        result["d2_evidence"]["representative_source"]
        == "synchronize_episode(read-only)"
    )


def test_cache_uses_raw_mtime_and_size(
    synthetic_raw_episode: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_diagnostics_cache()
    calls = 0
    original = d2_episode.synchronize_episode

    def counted(raw):
        nonlocal calls
        calls += 1
        return original(raw)

    monkeypatch.setattr(d2_episode, "synchronize_episode", counted)
    diagnose_episode(run_name="cache", raw_path=synthetic_raw_episode)
    diagnose_episode(run_name="cache", raw_path=synthetic_raw_episode)
    assert calls == 1

    stat = synthetic_raw_episode.stat()
    os.utime(
        synthetic_raw_episode,
        ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000),
    )
    diagnose_episode(run_name="cache", raw_path=synthetic_raw_episode)
    assert calls == 2


REAL_RAW = Path("/data/vla_runs/biyi_real_test_dataset/raw/episode_000007.npz")


@pytest.mark.skipif(not REAL_RAW.is_file(), reason="real episode_000007 unavailable")
def test_real_episode_000007_acceptance() -> None:
    clear_diagnostics_cache()
    result = diagnose_episode(
        run_name="biyi_real_test_dataset",
        raw_path=REAL_RAW,
        d2_status="FAILED",
    )
    d2 = result["d2_evidence"]
    raw = result["raw_diagnostics"]
    head = raw["camera_health"]["head"]
    wrist = raw["camera_health"]["wrist"]

    assert d2["candidate_count"] == 1667
    assert d2["accepted_count"] == 0
    assert d2["rejection_group_counts"]["ACTION"] == 1048
    assert d2["rejection_group_counts"]["CAMERA"] == 619
    assert d2["rejection_counts"]["CAMERA_HEAD_CAMERA_UNAVAILABLE"] == 619
    assert d2["primary_failure_group"] == "CAMERA"
    assert d2["primary_failure_reason"] == "CAMERA_HEAD_CAMERA_UNAVAILABLE"
    assert d2["most_frequent_rejection_reason"] == "ACTION_NONFINITE"
    assert head["status"] == "FROZEN"
    assert head["total_rows"] == 1668
    assert head["valid_count"] == 0
    assert head["timestamp_unique_count"] == 1
    assert head["frame_index_valid_count"] == 1668
    assert head["master_minus_camera_ns"]["median"] == pytest.approx(1_003_708_408_505)
    assert wrist["status"] == "HEALTHY"
    assert wrist["valid_ratio"] == pytest.approx(1667 / 1668)
    assert wrist["master_minus_camera_ns"]["median"] == pytest.approx(17_017_661.5)
    assert raw["inferred_root_cause"]["code"] == "HEAD_CAMERA_FROZEN"
    assert raw["inferred_root_cause"]["all_action_ready_lost_to_camera"] is True
    assert (
        "D2 cleaning rules should NOT be loosened" in result["diagnostic_report_text"]
    )
