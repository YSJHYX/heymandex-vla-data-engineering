import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from vla_data.manifest import (
    SplitConfig,
    build_training_manifest,
    validate_training_manifest,
)
from vla_data.manifest.split import assign_splits
from vla_data.verification import VerificationPolicy, review_episode, verify_annotations


def build(tree, **kwargs):
    return build_training_manifest(
        tree.curated,
        tree.quality,
        tree.annotation,
        tree.output,
        verification_root=tree.verification,
        **kwargs,
    )


def mutate_json(path, **values):
    data = json.loads(path.read_text())
    data.update(values)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize(
    "status,eligible,reason",
    [
        ("AUTO_LABELED", False, "NEEDS_HUMAN_REVIEW"),
        ("AUTO_ACCEPTED", True, None),
        ("HUMAN_VERIFIED", True, None),
        ("HUMAN_CORRECTED", True, None),
        ("REJECTED", False, "REJECTED"),
    ],
)
def test_review_gates(manifest_tree, status, eligible, reason):
    tree = manifest_tree(status=status)
    result = build(tree)
    assert result.summary["training_eligible"] == int(eligible)
    if reason:
        assert reason in result.records[0]["reason"]
    assert validate_training_manifest(tree.output).passed


def test_legacy_auto_accept_is_not_production_authority(manifest_tree):
    tree = manifest_tree(status="AUTO_ACCEPTED")
    mutate_json(
        tree.annotation / "episode_000000/annotation.json", annotation_policy=None
    )
    verify_annotations(
        tree.annotation, tree.quality, tree.verification, policy=VerificationPolicy(1.0)
    )
    assert "NEEDS_HUMAN_REVIEW" in build(tree).records[0]["reason"]


@pytest.mark.parametrize("outcome", ["REJECT", "EXCLUDE_FROM_EXPERT_TRAINING"])
def test_quality_exclusion_overrides_review(manifest_tree, outcome):
    tree = manifest_tree()
    mutate_json(tree.quality / "episode_000000/quality_report.json", status=outcome)
    result = build(tree)
    assert result.summary["training_eligible"] == 0
    assert any(outcome in reason for reason in result.records[0]["reason"])


def test_explicit_expert_exclusion_overrides_annotation(manifest_tree):
    tree = manifest_tree()
    mutate_json(
        tree.curated / "episode_000000/metadata.json",
        expert_training_status="EXCLUDE_FROM_EXPERT_TRAINING",
    )
    assert "EXCLUDE_FROM_EXPERT_TRAINING" in build(tree).records[0]["reason"]


@pytest.mark.parametrize("clean", [0, 1])
def test_quality_mask_is_exact_training_selection(manifest_tree, clean):
    tree = manifest_tree()
    path = tree.quality / "episode_000000/quality_mask.npy"
    mask = np.load(path)
    mask[:] = False
    mask[:clean] = True
    np.save(path, mask)
    mutate_json(path.with_name("quality_report.json"), clean_transition_count=clean)
    verify_annotations(
        tree.annotation, tree.quality, tree.verification, policy=tree.policy
    )
    result = build(tree)
    assert result.records[0]["transition_count_selected"] == clean
    assert result.summary["train_transitions"] == clean
    assert result.summary["training_eligible"] == int(clean > 0)
    assert validate_training_manifest(tree.output).passed
    assert np.array_equal(
        np.load(result.records[0]["training_transition_mask_path"]), mask
    )


def test_instruction_preserved_exactly(manifest_tree):
    tree = manifest_tree()
    path = tree.annotation / "episode_000000/annotation.json"
    document = json.loads(path.read_text())
    text = "Move  the RED block! 保持大小写。"
    document["model_annotation"]["instruction"] = text
    path.write_text(json.dumps(document))
    verify_annotations(
        tree.annotation, tree.quality, tree.verification, policy=tree.policy
    )
    assert build(tree).records[0]["final_instruction"] == text


@pytest.mark.parametrize(
    "artifact",
    [
        "missing_annotation",
        "corrupt_annotation",
        "missing_quality",
        "wrong_mask_length",
        "bad_counts",
        "missing_curated",
        "corrupt_curated",
        "truncated_curated_zip",
        "invalid_quality_status",
        "annotation_is_directory",
    ],
)
def test_bad_inputs_recorded_without_disappearing(manifest_tree, artifact):
    tree = manifest_tree()
    annotation = tree.annotation / "episode_000000/annotation.json"
    quality = tree.quality / "episode_000000/quality_report.json"
    if artifact == "missing_annotation":
        annotation.unlink()
    elif artifact == "corrupt_annotation":
        annotation.write_text("{broken")
    elif artifact == "missing_quality":
        quality.unlink()
    elif artifact == "wrong_mask_length":
        np.save(quality.with_name("quality_mask.npy"), np.ones(99, dtype=bool))
    elif artifact == "bad_counts":
        mutate_json(quality, clean_transition_count=999)
    elif artifact == "missing_curated":
        (tree.curated / "episode_000000/metadata.json").unlink()
    elif artifact == "invalid_quality_status":
        mutate_json(quality, status=[])
    elif artifact == "annotation_is_directory":
        annotation.unlink()
        annotation.mkdir()
    elif artifact == "truncated_curated_zip":
        path = tree.curated / "episode_000000/trajectory.npz"
        path.write_bytes(path.read_bytes()[:100])
    else:
        (tree.curated / "episode_000000/trajectory.npz").write_bytes(b"corrupt")
    result = build(tree)
    assert result.summary["episodes_discovered"] == 1
    assert result.summary["excluded"] == 1
    assert result.records[0]["reason"]


def test_optional_metadata_and_single_selection(manifest_tree):
    tree = manifest_tree(2)
    mutate_json(
        tree.curated / "episode_000001/metadata.json",
        session_id="synthetic-session",
        collection_group="fixture-group",
    )
    result = build(tree, episode="episode_000001")
    assert len(result.records) == 1
    assert result.records[0]["session_id"] == "synthetic-session"
    assert result.records[0]["collection_group"] == "fixture-group"
    assert result.records[0]["collection_date"] is None
    assert validate_training_manifest(tree.output).passed


def test_empty_dataset_is_valid(tmp_path):
    roots = [tmp_path / name for name in ("c", "q", "a")]
    for root in roots:
        root.mkdir()
    output = tmp_path / "manifest"
    verification = tmp_path / "verification"
    verification.mkdir()
    result = build_training_manifest(*roots, output, verification_root=verification)
    assert result.summary["episodes_discovered"] == 0
    assert validate_training_manifest(output).passed


@pytest.mark.parametrize("key", ["robot_qpos_17d_rad", "robot_qcmd_17d_rad"])
def test_32d_referenced_curated_rejected(manifest_tree, key):
    tree = manifest_tree()
    build(tree)
    path = tree.curated / "episode_000000/trajectory.npz"
    with np.load(path) as data:
        arrays = dict(data)
    arrays[key] = np.zeros((arrays[key].shape[0], 32))
    np.savez(path, **arrays)
    check = validate_training_manifest(tree.output)
    assert not check.passed
    assert "CURATED_DIMENSION_NOT_17" in str(check.errors)
    assert build(tree).summary["training_eligible"] == 0


def test_episode_split_seed_determinism():
    ids = [f"episode_{i:06d}" for i in range(20)]
    config = SplitConfig(validation_fraction=0.2, seed=11)
    first = assign_splits(ids, config)
    assert first == assign_splits(reversed(ids), config)
    assert first != assign_splits(ids, SplitConfig(validation_fraction=0.2, seed=12))
    assert sum(v == "val" for v in first.values()) == 4
    assert not {k for k, v in first.items() if v == "train"} & {
        k for k, v in first.items() if v == "val"
    }


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan"), float("inf")])
def test_bad_split_fraction(fraction):
    with pytest.raises(ValueError):
        SplitConfig(validation_fraction=fraction)


def test_no_group_aware_claim():
    with pytest.raises(ValueError, match="future work"):
        SplitConfig(group_key="session_id")


def test_resume_force_config_and_review_invalidation(manifest_tree):
    tree = manifest_tree(status="AUTO_LABELED")
    assert build(tree).status == "BUILT"
    assert build(tree).status == "SKIPPED"
    review_episode(tree.verification, "episode_000000", status="HUMAN_VERIFIED")
    assert build(tree).summary["training_eligible"] == 1
    assert build(tree).status == "SKIPPED"
    assert build(tree, config=SplitConfig(seed=9)).status == "BUILT"
    assert build(tree, force=True).status == "BUILT"
    review_episode(
        tree.verification,
        "episode_000000",
        status="HUMAN_CORRECTED",
        instruction="Corrected synthetic instruction.",
    )
    result = build(tree)
    assert result.status == "BUILT"
    assert result.records[0]["final_instruction"] == "Corrected synthetic instruction."


def test_quality_change_and_broken_output_rebuild(manifest_tree):
    tree = manifest_tree()
    build(tree)
    mutate_json(
        tree.quality / "episode_000000/quality_report.json",
        status="ACCEPT_WITH_WARNING",
    )
    assert build(tree).status == "BUILT"
    verify_annotations(
        tree.annotation, tree.quality, tree.verification, policy=tree.policy
    )
    assert build(tree).status == "BUILT"
    (tree.output / "train.jsonl").write_text("")
    assert build(tree).status == "BUILT"


def test_builder_is_read_only_and_dry_run_writes_nothing(manifest_tree):
    tree = manifest_tree()
    paths = [
        p
        for root in (tree.curated, tree.quality, tree.annotation, tree.verification)
        for p in root.rglob("*")
        if p.is_file()
    ]

    def snapshot():
        return [
            (p, p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
            for p in paths
        ]

    before = snapshot()
    assert build(tree, dry_run=True).status == "DRY_RUN"
    assert not tree.output.exists()
    build(tree)
    build(tree, force=True)
    assert snapshot() == before


def test_output_cannot_overlap_inputs(manifest_tree):
    tree = manifest_tree()
    with pytest.raises(ValueError, match="separate"):
        build_training_manifest(
            tree.curated,
            tree.quality,
            tree.annotation,
            tree.annotation,
            verification_root=tree.verification,
        )


def test_independent_validator_detects_overlap_and_instruction_tampering(manifest_tree):
    tree = manifest_tree(10)
    build(tree, config=SplitConfig(validation_fraction=0.2, seed=42))
    train = (tree.output / "train.jsonl").read_text().splitlines()
    with (tree.output / "val.jsonl").open("a") as stream:
        stream.write(train[0] + "\n")
    assert "TRAIN_VAL_EPISODE_OVERLAP" in validate_training_manifest(tree.output).errors
    build(tree, force=True)
    rows = [
        json.loads(line)
        for line in (tree.output / "episodes.jsonl").read_text().splitlines()
    ]
    rows[0]["final_instruction"] = "tampered"
    (tree.output / "episodes.jsonl").write_text("\n".join(map(json.dumps, rows)))
    assert not validate_training_manifest(tree.output).passed


def test_ten_reviewed_episode_real_cli_smoke(manifest_tree):
    tree = manifest_tree(10)
    command = [
        sys.executable,
        "-m",
        "vla_data.cli",
        "build-manifest",
        "--curated-root",
        str(tree.curated),
        "--quality-root",
        str(tree.quality),
        "--annotation-root",
        str(tree.annotation),
        "--verification-root",
        str(tree.verification),
        "--output-root",
        str(tree.output),
        "--seed",
        "17",
        "--validation-fraction",
        "0.2",
    ]
    for option in ([], [], ["--force"]):
        result = subprocess.run(
            command + option, capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert "train_episodes: 8" in result.stdout
        assert "validation_episodes: 2" in result.stdout
        assert validate_training_manifest(tree.output).passed
        current = {p.name: p.read_bytes() for p in tree.output.iterdir()}
        if not option and "SKIPPED" not in result.stdout:
            original = current
        assert current == original
    assert not {
        json.loads(line)["episode_id"]
        for line in (tree.output / "train.jsonl").read_text().splitlines()
    } & {
        json.loads(line)["episode_id"]
        for line in (tree.output / "val.jsonl").read_text().splitlines()
    }
