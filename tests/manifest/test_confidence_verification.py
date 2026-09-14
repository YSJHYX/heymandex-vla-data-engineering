"""D5.1 confidence, authority, immutability and no-provider regressions."""

import json
import subprocess
import sys

import pytest

from vla_data.manifest import build_training_manifest, validate_training_manifest
from vla_data.verification import (
    AutoVerificationAuditConfig,
    VerificationPolicy,
    load_verification,
    review_episode,
    verify_annotations,
)
from vla_data.verification.schema import seal

CONFIDENCES = [0.20, 0.50, 0.69, 0.70, 0.79, 0.80, 0.89, 0.90, 0.95, 1.00]


def update_confidence(tree, value, eid="episode_000000"):
    path = tree.annotation / eid / "annotation.json"
    document = json.loads(path.read_text())
    document["model_annotation"]["confidence"] = value
    path.write_text(json.dumps(document))


def verify(tree, threshold, **kwargs):
    return verify_annotations(
        tree.annotation,
        tree.quality,
        tree.verification,
        policy=VerificationPolicy(threshold),
        **kwargs,
    )


def manifest(tree):
    return build_training_manifest(
        tree.curated,
        tree.quality,
        tree.annotation,
        tree.output,
        verification_root=tree.verification,
    )


@pytest.mark.parametrize(
    "confidence,threshold,status",
    [
        (0.9, 0.8, "AUTO_VERIFIED"),
        (0.9, 0.9, "AUTO_VERIFIED"),
        (0.9, 0.91, "NEEDS_HUMAN_REVIEW"),
        (0.0, 0.0, "AUTO_VERIFIED"),
        (1.0, 1.0, "AUTO_VERIFIED"),
        (0.9999999999999999, 1.0, "NEEDS_HUMAN_REVIEW"),
        (0.8999999999999999, 0.9, "NEEDS_HUMAN_REVIEW"),
    ],
)
def test_exact_unrounded_boundary(manifest_tree, confidence, threshold, status):
    tree = manifest_tree(status="AUTO_LABELED")
    update_confidence(tree, confidence)
    result = verify(tree, threshold)
    record = result.records[0]
    assert record["verification_status"] == status
    assert record["verification_source"] == "MODEL_CONFIDENCE_POLICY"
    assert record["final_instruction"] == (
        record["model_instruction"] if status == "AUTO_VERIFIED" else None
    )
    assert manifest(tree).summary["training_eligible"] == int(status == "AUTO_VERIFIED")
    assert validate_training_manifest(tree.output).passed


@pytest.mark.parametrize(
    "value", [-1, 1.01, float("nan"), float("inf"), True, False, "0.9", None]
)
def test_invalid_threshold(value):
    with pytest.raises((TypeError, ValueError)):
        VerificationPolicy(value)


@pytest.mark.parametrize(
    "threshold,auto,needs", [(0.7, 7, 3), (0.8, 5, 5), (0.9, 3, 7), (0.95, 2, 8)]
)
def test_ten_confidences_batch_statistics(manifest_tree, threshold, auto, needs):
    tree = manifest_tree(10, status="AUTO_LABELED")
    for number, confidence in enumerate(CONFIDENCES):
        update_confidence(tree, confidence, f"episode_{number:06d}")
    result = verify(tree, threshold)
    assert result.summary["auto_verified_count"] == auto
    assert result.summary["needs_human_review_count"] == needs
    assert (
        result.summary["annotation_count"]
        == result.summary["quality_eligible_count"]
        == 10
    )
    assert result.summary["confidence_statistics"]["min"] == 0.2
    assert result.summary["confidence_statistics"]["max"] == 1.0
    assert result.summary["confidence_statistics"]["median"] == pytest.approx(0.795)
    assert result.summary["confidence_statistics"]["mean"] == pytest.approx(
        sum(CONFIDENCES) / 10
    )
    assert len(result.summary["confidence_statistics"]) == 9
    assert [r["count"] for r in result.summary["projected_review_workload"]] == [
        3,
        5,
        7,
        8,
    ]
    queue = [
        json.loads(line)
        for line in (tree.verification / "human_review_queue.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(queue) == needs
    assert [r["episode_id"] for r in queue] == sorted(r["episode_id"] for r in queue)
    assert all(r["confidence"] < threshold for r in queue)
    assert all(
        set(r)
        == {
            "episode_id",
            "model_instruction",
            "confidence",
            "annotation_path",
            "keyframes_path",
        }
        for r in queue
    )


@pytest.mark.parametrize("status", ["HUMAN_VERIFIED", "HUMAN_CORRECTED", "REJECTED"])
def test_human_review_authority_queue_and_force(manifest_tree, status):
    tree = manifest_tree(status="AUTO_LABELED")
    before = (tree.annotation / "episode_000000/annotation.json").read_bytes()
    instruction = (
        "  Move the BLUE block. 保留原文  " if status == "HUMAN_CORRECTED" else None
    )
    reviewed = review_episode(
        tree.verification,
        "episode_000000",
        status=status,
        instruction=instruction,
        reviewer="synthetic-reviewer",
    )
    assert (tree.verification / "human_review_queue.jsonl").read_text() == ""
    assert reviewed["verification_source"] == "HUMAN_REVIEW"
    if instruction:
        assert reviewed["final_instruction"] == instruction
    for threshold in [1.0, 0.1]:
        result = verify(tree, threshold, force=True)
        assert result.records[0]["verification_status"] == status
        assert result.records[0]["final_instruction"] == reviewed["final_instruction"]
    assert (tree.annotation / "episode_000000/annotation.json").read_bytes() == before
    result = manifest(tree)
    assert result.summary["training_eligible"] == int(status != "REJECTED")
    assert result.records[0]["final_instruction"] == reviewed["final_instruction"]


def test_threshold_and_policy_change_invalidate_auto_and_manifest(manifest_tree):
    tree = manifest_tree(status="AUTO_LABELED")
    update_confidence(tree, 0.9)
    first = verify(tree, 0.8)
    assert manifest(tree).summary["training_eligible"] == 1
    assert verify(tree, 0.8).skipped == 1
    changed = verify(tree, 0.91)
    assert changed.processed == 1
    assert first.records[0]["fingerprint"] != changed.records[0]["fingerprint"]
    result = manifest(tree)
    assert result.status == "BUILT"
    assert result.summary["training_eligible"] == 0
    version = verify_annotations(
        tree.annotation,
        tree.quality,
        tree.verification,
        policy=VerificationPolicy(0.91, "test_policy_version_2"),
    )
    assert version.processed == 1
    assert manifest(tree).status == "BUILT"


@pytest.mark.parametrize("outcome", ["REJECT", "EXCLUDE_FROM_EXPERT_TRAINING"])
def test_quality_overrides_confidence_one(manifest_tree, outcome):
    tree = manifest_tree(status="AUTO_LABELED")
    update_confidence(tree, 1.0)
    path = tree.quality / "episode_000000/quality_report.json"
    report = json.loads(path.read_text())
    report["status"] = outcome
    path.write_text(json.dumps(report))
    result = verify(tree, 0.0)
    assert result.records[0]["verification_status"] == "INELIGIBLE"
    assert result.records[0]["model_confidence"] is None  # quality before annotation
    assert outcome in result.records[0]["reason"]
    assert manifest(tree).summary["training_eligible"] == 0
    with pytest.raises(ValueError):
        review_episode(tree.verification, "episode_000000", status="HUMAN_VERIFIED")


def test_explicit_curated_exclusion_still_overrides_auto(manifest_tree):
    tree = manifest_tree(status="AUTO_LABELED")
    update_confidence(tree, 1.0)
    verify(tree, 0.0)
    path = tree.curated / "episode_000000/metadata.json"
    metadata = json.loads(path.read_text())
    metadata["expert_training_status"] = "EXCLUDE_FROM_EXPERT_TRAINING"
    path.write_text(json.dumps(metadata))
    assert "EXCLUDE_FROM_EXPERT_TRAINING" in manifest(tree).records[0]["reason"]


def test_dry_run_no_writes_or_source_changes(manifest_tree, tmp_path):
    tree = manifest_tree(status="AUTO_LABELED")
    paths = [
        p
        for root in (tree.annotation, tree.quality)
        for p in root.rglob("*")
        if p.is_file()
    ]
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in paths}
    output = tmp_path / "new_verification"
    result = verify_annotations(
        tree.annotation,
        tree.quality,
        output,
        policy=VerificationPolicy(1.0),
        dry_run=True,
    )
    assert result.summary["needs_human_review_count"] == 1
    assert not output.exists()
    verify_annotations(
        tree.annotation, tree.quality, output, policy=VerificationPolicy(0.5)
    )
    assert {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in paths} == before


def test_stale_annotation_and_bad_verification_fail_closed(manifest_tree):
    tree = manifest_tree(status="AUTO_LABELED")
    verify(tree, 0.5)
    manifest(tree)
    update_confidence(tree, 0.2)
    assert not validate_training_manifest(tree.output).passed
    assert manifest(tree).summary["training_eligible"] == 0
    verify(tree, 0.5)
    assert manifest(tree).records[0]["verification_status"] == "NEEDS_HUMAN_REVIEW"
    path = tree.verification / "episode_000000/verification.json"
    record = load_verification(path)
    record["verification_status"] = "AUTO_VERIFIED"
    record["final_instruction"] = record["model_instruction"]
    path.write_text(json.dumps(seal(record)))
    assert manifest(tree).summary["training_eligible"] == 0
    assert verify(tree, 0.5).processed == 1


def test_single_episode_and_mixed_policy_protection(manifest_tree):
    tree = manifest_tree(2, status="AUTO_LABELED")
    before = (tree.verification / "episode_000001/verification.json").read_bytes()
    assert verify(tree, 1.0, episode="episode_000000").skipped == 1
    assert (
        tree.verification / "episode_000001/verification.json"
    ).read_bytes() == before
    with pytest.raises(ValueError, match="mixed policies"):
        verify(tree, 0.5, episode="episode_000000")
    assert (
        tree.verification / "episode_000001/verification.json"
    ).read_bytes() == before


def test_optional_audit_never_blocks_auto(manifest_tree):
    tree = manifest_tree(10, status="AUTO_LABELED")
    result = verify(
        tree, 0.0, audit=AutoVerificationAuditConfig(random_audit_fraction=0.2, seed=4)
    )
    assert result.summary["auto_verified_count"] == 10
    assert len(result.summary["optional_auto_audit_episode_ids"]) == 2
    assert result.summary["needs_human_review_count"] == 0
    assert (tree.verification / "human_review_queue.jsonl").read_text() == ""


@pytest.mark.parametrize(
    "command,missing",
    [
        ("verify-annotations", "--confidence-threshold"),
        ("build-manifest", "--verification-root"),
    ],
)
def test_required_cli_policy_inputs(command, missing):
    result = subprocess.run(
        [sys.executable, "-m", "vla_data.cli", command],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert missing in result.stderr


def test_fresh_process_no_provider_import_or_network(manifest_tree, tmp_path):
    tree = manifest_tree(status="AUTO_LABELED")
    code = """
import importlib.abc, socket, sys
class NoProvider(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vla_data.annotation.glm_provider":
            raise AssertionError("D5.1 imported GLM provider")
sys.meta_path.insert(0, NoProvider())
def no_network(*args, **kwargs):
    raise AssertionError("D5.1 attempted network")
socket.socket.connect = no_network
from vla_data.cli import main
for threshold in ("0.8", "1.0"):
    assert main(["verify-annotations", "--annotation-root", sys.argv[1], "--quality-root", sys.argv[2], "--output-root", sys.argv[3], "--confidence-threshold", threshold]) == 0
assert main(["review-annotation", "--verification-root", sys.argv[3], "--episode", "episode_000000", "--status", "HUMAN_VERIFIED"]) == 0
assert main(["build-manifest", "--curated-root", sys.argv[4], "--quality-root", sys.argv[2], "--annotation-root", sys.argv[1], "--verification-root", sys.argv[3], "--output-root", sys.argv[5]]) == 0
assert "vla_data.annotation.glm_provider" not in sys.modules
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(tree.annotation),
            str(tree.quality),
            str(tmp_path / "isolated_verify"),
            str(tree.curated),
            str(tmp_path / "isolated_manifest"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_missing_verification_cannot_fall_back_to_d4(manifest_tree):
    tree = manifest_tree(status="AUTO_ACCEPTED")
    (tree.verification / "episode_000000/verification.json").unlink()
    assert "VERIFICATION_INVALID_OR_MISSING" in manifest(tree).records[0]["reason"]
