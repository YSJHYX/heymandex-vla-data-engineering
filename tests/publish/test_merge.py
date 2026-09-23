"""Logical episode planning must never use chunks or task strings as identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.publish.conftest import INFO
from vla_data.export.camera_transform import camera_export_transform_contract
from vla_data.publish.merge import UnverifiableProvenanceError, plan_logical_merge


def _dataset(
    root: Path,
    specs: list[tuple[str, str, str, int]],
    *,
    camera_contract: object = "current",
) -> Path:
    """(capture bytes, source episode id, exact task, length)."""

    dataset = root / "dataset"
    (dataset / "meta").mkdir(parents=True)
    info = json.loads(json.dumps(INFO))
    info.update(total_episodes=len(specs), total_frames=sum(row[3] for row in specs))
    (dataset / "meta/info.json").write_text(json.dumps(info))
    tasks = list(dict.fromkeys(task for _, _, task, _ in specs))
    (dataset / "meta/tasks.jsonl").write_text(
        "".join(
            json.dumps({"task_index": i, "task": task}) + "\n"
            for i, task in enumerate(tasks)
        )
    )
    (dataset / "meta/episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": i, "length": length, "tasks": [task]}) + "\n"
            for i, (_, _, task, length) in enumerate(specs)
        )
    )
    provenance = []
    for index, (capture, source_id, task, length) in enumerate(specs):
        raw = root / f"raw_{index}.npz"
        raw.write_bytes(capture.encode())
        curated = root / f"curated_{index}"
        curated.mkdir()
        (curated / "metadata.json").write_text(
            json.dumps({"source_episode_path": str(raw), "language_instruction": task})
        )
        (curated / "cleaning_report.json").write_text(
            json.dumps(
                {"source_provenance": {"dataset_status": "RAW_CAPTURE_QUARANTINED"}}
            )
        )
        row = {
            "lerobot_episode_index": index,
            "curated_path": str(curated),
            "source_episode_id": source_id,
            "source_start_index": 0,
            "source_end_index": length,
            "transition_count": length,
            "task_instruction": task,
            "task_instruction_sha256": hashlib.sha256(task.encode()).hexdigest(),
            "expert_training_status": "REVIEW_REQUIRED",
            "source_dataset_status": "RAW_CAPTURE_QUARANTINED",
        }
        if camera_contract == "current":
            row["camera_transforms"] = camera_export_transform_contract()
        elif camera_contract is not None:
            row["camera_transforms"] = camera_contract
        provenance.append(row)
        parquet = dataset / "data/chunk-000" / f"episode_{index:06d}.parquet"
        parquet.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_bytes(f"parquet-{index}".encode())
        for camera in ("head", "wrist"):
            video = (
                dataset
                / "videos/chunk-000"
                / f"observation.images.{camera}"
                / f"episode_{index:06d}.mp4"
            )
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(f"video-{camera}-{index}".encode())
    (dataset / "meta/source_provenance.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in provenance)
    )
    return dataset


def _plan(
    incoming: Path,
    baseline: Path | None = None,
    *,
    legacy_camera_baseline_revisions: tuple[str, ...] = (),
) -> dict:
    return plan_logical_merge(
        incoming,
        baseline,
        baseline_revision="a" * 40 if baseline else None,
        source_export_fingerprint="b" * 64,
        legacy_camera_baseline_revisions=legacy_camera_baseline_revisions,
    )


def test_empty_baseline_preserves_independent_same_task_episodes(tmp_path) -> None:
    incoming = _dataset(
        tmp_path / "incoming",
        [
            (f"capture-{i}", f"episode_{i:06d}", "grab the ball", length)
            for i, length in enumerate((343, 116, 286, 310), start=3)
        ],
    )
    plan = _plan(incoming)
    assert (
        plan["baseline_episodes"],
        plan["incoming_episodes"],
        plan["to_append"],
    ) == (0, 4, 4)
    assert (plan["merged_episodes"], plan["merged_frames"]) == (4, 1055)
    assert [row["transition_count"] for row in plan["provenance"]] == [
        343,
        116,
        286,
        310,
    ]
    assert [row["lerobot_episode_index"] for row in plan["provenance"]] == [0, 1, 2, 3]
    assert len({row["source_run_key"] for row in plan["provenance"]}) == 4
    assert all(row["upload_batch_id"] == "b" * 64 for row in plan["provenance"])


def test_existing_chunk_zero_and_same_task_append_as_separate_episodes(
    tmp_path,
) -> None:
    baseline = _dataset(
        tmp_path / "baseline", [("old", "episode_000001", "grab the ball", 5)]
    )
    incoming = _dataset(
        tmp_path / "incoming", [("new", "episode_000002", "grab the ball", 7)]
    )
    plan = _plan(incoming, baseline)
    assert plan["append_indices"] == [0]
    assert (plan["merged_episodes"], plan["merged_frames"]) == (2, 12)
    assert [row["lerobot_episode_index"] for row in plan["provenance"]] == [0, 1]
    assert [row["source_episode_id"] for row in plan["provenance"]] == [
        "episode_000001",
        "episode_000002",
    ]
    assert "upload_batch_id" not in plan["provenance"][0]


def test_different_task_adds_episode_without_rewriting_strings(tmp_path) -> None:
    baseline = _dataset(
        tmp_path / "baseline", [("old", "episode_000001", "grab the ball", 5)]
    )
    incoming = _dataset(
        tmp_path / "incoming", [("new", "episode_000002", "place the ball", 7)]
    )
    plan = _plan(incoming, baseline)
    assert [row["task_instruction"] for row in plan["provenance"]] == [
        "grab the ball",
        "place the ball",
    ]


def test_exact_rerun_and_force_policy_have_zero_new_episodes(tmp_path) -> None:
    baseline = _dataset(
        tmp_path / "baseline", [("same", "episode_000001", "grab the ball", 5)]
    )
    incoming = _dataset(
        tmp_path / "incoming", [("same", "episode_000001", "grab the ball", 5)]
    )
    plan = _plan(incoming, baseline)
    assert plan["already_present"] == 1
    assert plan["to_append"] == 0
    assert plan["merged_frames"] == 5
    assert plan["requires_camera_transform_migration"] is False


def test_legacy_baseline_gets_one_full_wrist_migration(tmp_path) -> None:
    baseline = _dataset(
        tmp_path / "baseline",
        [("old", "episode_000001", "grab the ball", 5)],
        camera_contract=None,
    )
    incoming = _dataset(
        tmp_path / "incoming",
        [("same", "episode_000001", "grab the ball", 5)],
    )
    plan = _plan(incoming, baseline, legacy_camera_baseline_revisions=("a" * 40,))

    assert plan["requires_camera_transform_migration"] is True
    assert plan["baseline_wrist_rotation_indices"] == [0]
    assert plan["provenance"][0]["camera_transforms"] == (
        camera_export_transform_contract()
    )
    migration = plan["provenance"][0]["camera_transform_migration"]
    assert migration["wrist_rotation_deg_applied"] == 180
    assert migration["stage"] == "cumulative_lerobot_rebuild"


def test_unregistered_legacy_camera_baseline_fails_closed(tmp_path) -> None:
    baseline = _dataset(
        tmp_path / "baseline",
        [("old", "episode_000001", "grab the ball", 5)],
        camera_contract=None,
    )
    incoming = _dataset(
        tmp_path / "incoming",
        [("new", "episode_000002", "grab the ball", 5)],
    )
    with pytest.raises(
        UnverifiableProvenanceError, match="registered pre-camera-transform"
    ):
        _plan(incoming, baseline)


def test_audited_legacy_revision_registered_by_default_migrates(tmp_path) -> None:
    baseline = _dataset(
        tmp_path / "baseline",
        [(f"old-{i}", f"episode_{i:06d}", "grab the ball", 5) for i in range(1, 5)],
        camera_contract=None,
    )
    incoming = _dataset(
        tmp_path / "incoming",
        [("new", "episode_000005", "grab the ball", 7)],
    )
    plan = plan_logical_merge(
        incoming,
        baseline,
        baseline_revision="5557ba134c6d19dd47b468a38a8396c780aeea5e",
        source_export_fingerprint="b" * 64,
    )

    assert plan["requires_camera_transform_migration"] is True
    assert plan["baseline_wrist_rotation_indices"] == [0, 1, 2, 3]
    assert all(
        row["camera_transforms"] == camera_export_transform_contract()
        for row in plan["provenance"][:4]
    )


def test_current_camera_contract_needs_no_legacy_registration(tmp_path) -> None:
    baseline = _dataset(
        tmp_path / "baseline",
        [("old", "episode_000001", "grab the ball", 5)],
    )
    incoming = _dataset(
        tmp_path / "incoming",
        [("new", "episode_000002", "grab the ball", 7)],
    )
    plan = _plan(incoming, baseline)

    assert plan["requires_camera_transform_migration"] is False
    assert plan["baseline_wrist_rotation_indices"] == []
    assert (plan["to_append"], plan["merged_episodes"], plan["merged_frames"]) == (
        1,
        2,
        12,
    )


def test_invalid_baseline_camera_contract_fails_closed(tmp_path) -> None:
    baseline = _dataset(
        tmp_path / "baseline",
        [("old", "episode_000001", "grab", 5)],
        camera_contract={"wrist": {"rotation_deg": 90}},
    )
    incoming = _dataset(tmp_path / "incoming", [("new", "episode_000002", "grab", 5)])
    with pytest.raises(UnverifiableProvenanceError, match="unsupported camera"):
        _plan(incoming, baseline)


def test_incoming_without_current_camera_contract_fails_closed(tmp_path) -> None:
    incoming = _dataset(
        tmp_path / "incoming",
        [("new", "episode_000002", "grab", 5)],
        camera_contract=None,
    )
    with pytest.raises(UnverifiableProvenanceError, match="rebuild it directly"):
        _plan(incoming)


def test_partial_overlap_only_appends_unseen_runs(tmp_path) -> None:
    baseline = _dataset(
        tmp_path / "baseline",
        [
            ("same-0", "episode_000003", "grab the ball", 3),
            ("same-1", "episode_000004", "grab the ball", 4),
        ],
    )
    incoming = _dataset(
        tmp_path / "incoming",
        [
            ("same-0", "episode_000003", "grab the ball", 3),
            ("same-1", "episode_000004", "grab the ball", 4),
            ("new-2", "episode_000005", "grab the ball", 5),
            ("new-3", "episode_000006", "grab the ball", 6),
        ],
    )
    plan = _plan(incoming, baseline)
    assert plan["duplicate_indices"] == [0, 1]
    assert plan["append_indices"] == [2, 3]
    assert (plan["merged_episodes"], plan["merged_frames"]) == (4, 18)
    assert [row["source_episode_id"] for row in plan["provenance"]] == [
        "episode_000003",
        "episode_000004",
        "episode_000005",
        "episode_000006",
    ]


def test_four_episode_baseline_plus_two_new_runs_is_six(tmp_path) -> None:
    old = [
        (f"old-capture-{i}", f"episode_{i:06d}", "grab the ball", length)
        for i, length in enumerate((343, 116, 286, 310), start=3)
    ]
    baseline = _dataset(tmp_path / "baseline", old)
    incoming = _dataset(
        tmp_path / "incoming",
        [
            ("new-capture-7", "episode_000007", "grab the ball", 12),
            ("new-capture-8", "episode_000008", "grab the ball", 15),
        ],
    )
    plan = _plan(incoming, baseline)
    assert (plan["baseline_episodes"], plan["baseline_frames"]) == (4, 1055)
    assert (plan["incoming_episodes"], plan["already_present"], plan["to_append"]) == (
        2,
        0,
        2,
    )
    assert (plan["merged_episodes"], plan["merged_frames"]) == (6, 1082)
    assert [row["transition_count"] for row in plan["provenance"]] == [
        343,
        116,
        286,
        310,
        12,
        15,
    ]
    assert len({row["source_run_key"] for row in plan["provenance"]}) == 6
    assert all(row["task_instruction"] == "grab the ball" for row in plan["provenance"])


def test_four_episode_baseline_partial_overlap_appends_only_two(tmp_path) -> None:
    old = [
        (f"old-capture-{i}", f"episode_{i:06d}", "grab the ball", length)
        for i, length in enumerate((343, 116, 286, 310), start=3)
    ]
    baseline = _dataset(tmp_path / "baseline", old)
    incoming = _dataset(
        tmp_path / "incoming",
        [
            old[2],
            ("new-capture-7", "episode_000007", "grab the ball", 12),
            ("new-capture-8", "episode_000008", "place the ball", 15),
        ],
    )
    plan = _plan(incoming, baseline)
    assert plan["duplicate_indices"] == [0]
    assert plan["append_indices"] == [1, 2]
    assert (plan["already_present"], plan["to_append"], plan["merged_episodes"]) == (
        1,
        2,
        6,
    )
    assert plan["merged_frames"] == 1082
    assert [row["lerobot_episode_index"] for row in plan["provenance"]] == list(
        range(6)
    )
    assert [row["source_episode_id"] for row in plan["provenance"][:4]] == [
        f"episode_{i:06d}" for i in range(3, 7)
    ]


def test_same_episode_number_from_different_capture_is_not_duplicate(tmp_path) -> None:
    baseline = _dataset(tmp_path / "baseline", [("day-1", "episode_000000", "grab", 3)])
    incoming = _dataset(tmp_path / "incoming", [("day-2", "episode_000000", "grab", 3)])
    assert _plan(incoming, baseline)["to_append"] == 1


def test_missing_raw_identity_fails_closed(tmp_path) -> None:
    incoming = _dataset(
        tmp_path / "incoming", [("missing", "episode_000000", "grab", 3)]
    )
    raw = tmp_path / "incoming/raw_0.npz"
    raw.unlink()
    with pytest.raises(UnverifiableProvenanceError, match="captured RAW identity"):
        _plan(incoming)


def test_synthetic_raw_kind_is_excluded_even_with_diagnostic_real_status(
    tmp_path,
) -> None:
    incoming = _dataset(
        tmp_path / "incoming", [("synthetic", "episode_000002", "grab", 3)]
    )
    report = tmp_path / "incoming/curated_0/cleaning_report.json"
    report.write_text(
        json.dumps({"source_provenance": {"dataset_status": "SYNTHETIC_TEST_ONLY"}})
    )
    with pytest.raises(UnverifiableProvenanceError, match="synthetic"):
        _plan(incoming)


def test_rewritten_legacy_task_is_not_silently_published(tmp_path) -> None:
    incoming = _dataset(
        tmp_path / "incoming", [("capture", "episode_000001", "Place cable", 3)]
    )
    metadata = tmp_path / "incoming/curated_0/metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "source_episode_path": str(tmp_path / "incoming/raw_0.npz"),
                "language_instruction": "test",
            }
        )
    )
    with pytest.raises(UnverifiableProvenanceError, match="differs from original"):
        _plan(incoming)
