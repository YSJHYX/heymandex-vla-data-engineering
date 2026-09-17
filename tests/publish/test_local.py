"""Local dataset-root validation and canonical manifest construction."""

from __future__ import annotations

import json

import pytest

from vla_data.publish import (
    InvalidDatasetRootError,
    InvalidRepoIdError,
    build_canonical_manifest,
    build_readme,
    validate_dataset_root,
    validate_repo_id,
)


@pytest.mark.parametrize(
    "repo_id",
    ["PPPPPilot/VLADexData", "a/b", "user_1/name-2.0"],
)
def test_valid_repo_ids(repo_id: str) -> None:
    assert validate_repo_id(repo_id) == repo_id


@pytest.mark.parametrize(
    "repo_id",
    ["", "no-slash", "/lead", "trail/", "a//b", "sp ace/x", "a/b c", "李/明"],
)
def test_invalid_repo_ids(repo_id: str) -> None:
    with pytest.raises(InvalidRepoIdError):
        validate_repo_id(repo_id)


def test_canonical_layout_accepted(hf_dataset_root) -> None:
    layout = validate_dataset_root(hf_dataset_root)
    assert layout["episodes"] == 1
    assert layout["tasks"] == ["Place the cable on the table."]
    assert layout["fps"] == 30


def test_missing_or_invalid_info_rejected(tmp_path) -> None:
    with pytest.raises(InvalidDatasetRootError, match="not a directory"):
        validate_dataset_root(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(InvalidDatasetRootError, match="info.json"):
        validate_dataset_root(empty)


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"codebase_version": "v3.0"}, "codebase_version"),
        (
            {
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [32]},
                    "action": {"dtype": "float32", "shape": [17]},
                }
            },
            r"float32 \[17\]",
        ),
        (
            {
                "features": {
                    "observation.state": {"dtype": "float64", "shape": [17]},
                    "action": {"dtype": "float32", "shape": [17]},
                }
            },
            r"float32 \[17\]",
        ),
    ],
)
def test_invalid_schema_rejected(hf_dataset_root, patch, message) -> None:
    info_path = hf_dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info.update(patch)
    info_path.write_text(json.dumps(info))
    with pytest.raises(InvalidDatasetRootError, match=message):
        validate_dataset_root(hf_dataset_root)


def test_missing_tasks_rejected(hf_dataset_root) -> None:
    (hf_dataset_root / "meta" / "tasks.jsonl").unlink()
    with pytest.raises(InvalidDatasetRootError, match="tasks.jsonl"):
        validate_dataset_root(hf_dataset_root)


def test_missing_provenance_rejected(hf_dataset_root) -> None:
    path = hf_dataset_root / "meta" / "source_provenance.jsonl"
    path.unlink()
    with pytest.raises(InvalidDatasetRootError, match="source_provenance"):
        validate_dataset_root(hf_dataset_root)


@pytest.mark.parametrize(
    "status", ["REVIEW_REQUIRED", "EXCLUDE_FROM_EXPERT_TRAINING", "ARBITRARY", None]
)
def test_diagnostic_provenance_statuses_do_not_gate_layout(
    hf_dataset_root, status
) -> None:
    path = hf_dataset_root / "meta" / "source_provenance.jsonl"
    row = json.loads(path.read_text())
    row["expert_training_status"] = status
    row["source_dataset_status"] = "RAW_CAPTURE_QUARANTINED"
    path.write_text(json.dumps(row) + "\n")
    assert validate_dataset_root(hf_dataset_root)["episodes"] == 1


@pytest.mark.parametrize("status", ["RAW_CAPTURE_QUARANTINED", "ARBITRARY", None])
def test_source_status_is_diagnostic_only(hf_dataset_root, status) -> None:
    path = hf_dataset_root / "meta" / "source_provenance.jsonl"
    row = json.loads(path.read_text())
    if status is None:
        row.pop("source_dataset_status")
    else:
        row["source_dataset_status"] = status
    path.write_text(json.dumps(row) + "\n")
    assert validate_dataset_root(hf_dataset_root)["episodes"] == 1


def test_missing_camera_feature_rejected(hf_dataset_root) -> None:
    path = hf_dataset_root / "meta" / "info.json"
    info = json.loads(path.read_text())
    del info["features"]["observation.images.wrist"]
    path.write_text(json.dumps(info))
    with pytest.raises(InvalidDatasetRootError, match="wrist"):
        validate_dataset_root(hf_dataset_root)


def test_missing_videos_rejected(hf_dataset_root) -> None:
    (
        hf_dataset_root
        / "videos"
        / "chunk-000"
        / "observation.images.wrist"
        / "episode_000000.mp4"
    ).unlink()
    with pytest.raises(InvalidDatasetRootError, match="MP4"):
        validate_dataset_root(hf_dataset_root)


def test_manifest_is_deterministic(hf_dataset_root) -> None:
    first = build_canonical_manifest(hf_dataset_root)
    second = build_canonical_manifest(hf_dataset_root)
    assert first == second
    assert first["file_count"] == 7  # 4 meta + 1 parquet + 2 MP4
    assert first["file_count"] == len(first["files"])
    assert first["total_bytes"] == sum(f["size_bytes"] for f in first["files"])
    paths = {entry["relative_path"] for entry in first["files"]}
    assert paths == {
        "meta/episodes.jsonl",
        "meta/info.json",
        "meta/tasks.jsonl",
        "meta/source_provenance.jsonl",
        "data/chunk-000/episode_000000.parquet",
        "videos/chunk-000/observation.images.head/episode_000000.mp4",
        "videos/chunk-000/observation.images.wrist/episode_000000.mp4",
    }


def test_manifest_fingerprint_changes_with_content(hf_dataset_root) -> None:
    first = build_canonical_manifest(hf_dataset_root)
    target = hf_dataset_root / "data" / "chunk-000" / "episode_000000.parquet"
    target.write_bytes(target.read_bytes() + b"x")
    second = build_canonical_manifest(hf_dataset_root)
    assert first["fingerprint"] != second["fingerprint"]


def test_readme_carries_private_production_contract(hf_dataset_root) -> None:
    layout = validate_dataset_root(hf_dataset_root)
    readme = build_readme(layout["tasks"], layout["fps"])
    assert "PRIVATE_TRAINING_DATASET" in readme
    assert "measured 17D state" in readme
    assert "exact collection-time task instructions" in readme
