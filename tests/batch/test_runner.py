from __future__ import annotations

import json
import shutil

import numpy as np

from vla_data.batch.runner import (
    build_curated_dataset,
    quality_dataset,
    run_pipeline,
    validate_curated_dataset,
)


def _statuses(result):
    return [(item.episode_id, item.status, item.outcome) for item in result.results]


def test_build_resume_invalid_retry_and_force(raw_dataset_factory, tmp_path) -> None:
    raw = raw_dataset_factory(tmp_path / "raw", (1,))
    curated = tmp_path / "curated"

    first = build_curated_dataset(raw, curated)
    resumed = build_curated_dataset(raw, curated)
    assert _statuses(first) == [("episode_000001", "SUCCESS", None)]
    assert _statuses(resumed) == [("episode_000001", "SKIPPED", None)]

    (curated / "episode_000001" / "trajectory.npz").unlink()
    retried = build_curated_dataset(raw, curated)
    assert _statuses(retried) == [("episode_000001", "SUCCESS", None)]

    forced = build_curated_dataset(raw, curated, force=True)
    assert _statuses(forced) == [("episode_000001", "SUCCESS", None)]


def test_failure_isolation_preserves_following_episode(
    raw_dataset_factory, synthetic_raw_episode, tmp_path
) -> None:
    raw = raw_dataset_factory(tmp_path / "raw", (1, 3))
    shutil.copy2(synthetic_raw_episode, raw / "episode_000002.npz")

    result = build_curated_dataset(raw, tmp_path / "curated", workers=2)

    assert [(item.episode_id, item.status) for item in result.results] == [
        ("episode_000001", "SUCCESS"),
        ("episode_000002", "FAILED"),
        ("episode_000003", "SUCCESS"),
    ]
    assert result.summary["episodes_processed"] == 2
    assert result.summary["episodes_failed"] == 1


def test_workers_are_logically_deterministic_and_episodes_never_concatenate(
    raw_dataset_factory, tmp_path
) -> None:
    raw = raw_dataset_factory(tmp_path / "raw", (1, 2, 10))

    serial = build_curated_dataset(raw, tmp_path / "serial", workers=1)
    parallel = build_curated_dataset(raw, tmp_path / "parallel", workers=3)

    assert _statuses(serial) == _statuses(parallel)
    assert [item.episode_id for item in parallel.results] == [
        "episode_000001",
        "episode_000002",
        "episode_000010",
    ]
    assert (
        serial.summary["episodes_processed"] == parallel.summary["episodes_processed"]
    )
    assert not (tmp_path / "serial" / "trajectory.npz").exists()
    for episode_id in ("episode_000001", "episode_000002", "episode_000010"):
        with np.load(
            tmp_path / "serial" / episode_id / "trajectory.npz",
            allow_pickle=False,
        ) as trajectory:
            assert trajectory["robot_qpos_17d_rad"].shape == (2, 17)


def test_validation_and_quality_resume_with_stable_outcomes(
    raw_dataset_factory, tmp_path
) -> None:
    raw = raw_dataset_factory(tmp_path / "raw", (1, 2))
    curated = tmp_path / "curated"
    validation = tmp_path / "validation"
    quality = tmp_path / "quality"
    build_curated_dataset(raw, curated)

    first_validation = validate_curated_dataset(curated, validation, workers=2)
    resumed_validation = validate_curated_dataset(curated, validation, workers=2)
    assert [item.outcome for item in first_validation.results] == ["PASS", "PASS"]
    assert [item.outcome for item in resumed_validation.results] == ["PASS", "PASS"]
    assert all(item.status == "SKIPPED" for item in resumed_validation.results)

    first_quality = quality_dataset(curated, quality, workers=2)
    resumed_quality = quality_dataset(curated, quality, workers=2)
    assert [item.outcome for item in resumed_quality.results] == [
        item.outcome for item in first_quality.results
    ]
    assert all(item.status == "SKIPPED" for item in resumed_quality.results)
    assert (
        resumed_quality.summary["eligibility_counts"]
        == first_quality.summary["eligibility_counts"]
    )


def test_invalid_previous_validation_is_not_silently_skipped(
    raw_dataset_factory, tmp_path
) -> None:
    raw = raw_dataset_factory(tmp_path / "raw", (1,))
    curated = tmp_path / "curated"
    validation = tmp_path / "validation"
    build_curated_dataset(raw, curated)
    validate_curated_dataset(curated, validation)

    metadata_path = curated / "episode_000001" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["schema_version"] = 999
    metadata_path.write_text(json.dumps(metadata))

    result = validate_curated_dataset(curated, validation)
    assert result.results[0].status == "FAILED"
    assert result.summary["failed_episode_ids"] == ["episode_000001"]


def test_dry_run_writes_nothing(raw_dataset_factory, tmp_path) -> None:
    raw = raw_dataset_factory(tmp_path / "raw", (1, 2))
    output = tmp_path / "curated"

    result = build_curated_dataset(raw, output, dry_run=True)

    assert not output.exists()
    assert all(item.message == "WOULD_PROCESS" for item in result.results)
    assert all(item.status == "WOULD_PROCESS" for item in result.results)
    assert result.summary["episodes_would_process"] == 2


def test_unified_pipeline_dry_run_projects_planned_episodes(
    raw_dataset_factory, tmp_path
) -> None:
    raw = raw_dataset_factory(tmp_path / "raw", (1, 2))
    work = tmp_path / "processed"

    results = run_pipeline(raw, work, workers=2, dry_run=True)

    assert not work.exists()
    for stage in ("curated", "validate", "quality"):
        assert [item.episode_id for item in results[stage].results] == [
            "episode_000001",
            "episode_000002",
        ]
        assert all(item.status == "WOULD_PROCESS" for item in results[stage].results)
        assert results[stage].summary["episodes_would_process"] == 2
        assert results[stage].summary["episodes_processed"] == 0


def test_unified_pipeline_delegates_all_stages_and_preserves_diagnostic_status(
    raw_dataset_factory, tmp_path
) -> None:
    raw = raw_dataset_factory(tmp_path / "raw", (0, 1))
    work = tmp_path / "processed"

    result = run_pipeline(
        raw,
        work,
        workers=2,
        expert_exclude=("episode_000000",),
    )

    assert tuple(result) == ("curated", "validate", "quality")
    outcomes = {item.episode_id: item.outcome for item in result["quality"].results}
    assert outcomes["episode_000000"] == "ACCEPT"
    assert outcomes["episode_000001"] in {"ACCEPT", "ACCEPT_WITH_WARNING"}
    assert (work / "curated" / "dataset_build_summary.json").is_file()
    assert (work / "validation" / "dataset_validation_summary.json").is_file()
    assert (work / "quality" / "dataset_quality_summary.json").is_file()


def test_mini_dataset_covers_valid_warning_diagnostic_and_malformed(
    raw_dataset_factory, synthetic_raw_episode, tmp_path
) -> None:
    raw = raw_dataset_factory(tmp_path / "raw", (0, 1, 2))
    warning_path = raw / "episode_000001.npz"
    with np.load(warning_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    arrays["arm_qcmd_source_timestamp_ns"][4] = arrays["arm_qcmd_source_timestamp_ns"][
        3
    ]
    arrays["hand_qcmd_source_timestamp_ns"][4] = arrays[
        "hand_qcmd_source_timestamp_ns"
    ][3]
    np.savez(warning_path, **arrays)
    shutil.copy2(synthetic_raw_episode, raw / "episode_000003.npz")

    result = run_pipeline(
        raw,
        tmp_path / "processed",
        workers=2,
        expert_exclude=("episode_000002",),
    )

    assert [(item.episode_id, item.status) for item in result["curated"].results] == [
        ("episode_000000", "SUCCESS"),
        ("episode_000001", "SUCCESS"),
        ("episode_000002", "SUCCESS"),
        ("episode_000003", "FAILED"),
    ]
    outcomes = {item.episode_id: item.outcome for item in result["quality"].results}
    assert outcomes == {
        "episode_000000": "ACCEPT",
        "episode_000001": "ACCEPT_WITH_WARNING",
        "episode_000002": "ACCEPT",
    }
