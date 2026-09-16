import json
import os

import pytest
from PIL import Image

from vla_data.annotation.glm_provider import DeterministicStructuredMockProvider
from vla_data.annotation.hierarchical import annotate_hierarchical_episode
from vla_data.export import export_lerobot
from vla_data.export.plan import make_plan
from vla_data.export.runtime import worker_call
from vla_data.io.curated_episode import CuratedEpisode
from vla_data.manifest import (
    SplitConfig,
    build_training_manifest,
    validate_training_manifest,
)
from vla_data.verification import VerificationPolicy, review_episode, verify_annotations


def payloads(end, middle):
    episode_task = {
        "instruction": "Place the block in the target area.",
        "confidence": 0.95,
        "paraphrases": [],
    }
    segments = [
        {
            "segment_id": "approach",
            "start_curated_index": 0,
            "end_curated_index": middle,
            "instruction": "Approach the block.",
            "confidence": 0.91,
            "paraphrases": [],
        },
        {
            "segment_id": "place",
            "start_curated_index": middle,
            "end_curated_index": end,
            "instruction": "Place the block in the target area.",
            "confidence": 0.89,
            "paraphrases": [],
        },
    ]
    return (
        {
            "episode_task": episode_task,
            "semantic_segments": segments,
            "non_training_intervals": [],
        },
        {"semantic_segments": segments, "non_training_intervals": []},
    )


def prepare(tree, annotation_root):
    for episode_path in sorted(tree.curated.glob("episode_*")):
        episode = CuratedEpisode.load(episode_path)
        coarse, refined = payloads(
            episode.transition_count, max(1, episode.transition_count // 2)
        )
        result = annotate_hierarchical_episode(
            episode_path,
            quality_root=tree.quality,
            output_root=annotation_root,
            provider=DeterministicStructuredMockProvider(coarse, refined),
        )
        assert result.status == "SUCCESS", result.message


def test_segment_verification_review_manifest_split_and_export(manifest_tree, tmp_path):
    tree = manifest_tree(2, status="AUTO_LABELED")
    annotation = tmp_path / "hierarchical"
    verification = tmp_path / "hierarchical-verification"
    manifest = tmp_path / "hierarchical-manifest"
    metadata_path = tree.curated / "episode_000000/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["source_dataset_status"] = "SYNTHETIC_TEST_ONLY"
    metadata_path.write_text(json.dumps(metadata))
    prepare(tree, annotation)
    original = (annotation / "episode_000000/annotation.json").read_bytes()
    verified = verify_annotations(
        annotation,
        tree.quality,
        verification,
        policy=VerificationPolicy(0.9),
    )
    first = verified.records[0]
    assert [s["verification_status"] for s in first["semantic_segments"]] == [
        "AUTO_VERIFIED",
        "NEEDS_HUMAN_REVIEW",
    ]
    assert first["episode_task"]["verification_status"] == "AUTO_VERIFIED"
    review_episode(
        verification,
        "episode_000000",
        status="HUMAN_CORRECTED",
        instruction="Complete the exact synthetic placement task.",
        reviewer="fixture",
    )
    review_episode(
        verification,
        "episode_000000",
        semantic_segment_id="place",
        status="HUMAN_CORRECTED",
        instruction="Set the block in the marked target.",
        reviewer="fixture",
    )
    assert (annotation / "episode_000000/annotation.json").read_bytes() == original
    with pytest.raises(ValueError, match="overlap"):
        review_episode(
            verification,
            "episode_000000",
            semantic_segment_id="approach",
            status="HUMAN_CORRECTED",
            start_curated_index=0,
            end_curated_index=first["semantic_segments"][1]["model_end_curated_index"],
        )
    for episode in ("episode_000001",):
        review_episode(
            verification,
            episode,
            semantic_segment_id="place",
            status="HUMAN_VERIFIED",
        )
    review_episode(
        verification,
        "episode_000001",
        semantic_segment_id="approach",
        status="REJECTED",
        reviewer="fixture",
    )
    result = build_training_manifest(
        tree.curated,
        tree.quality,
        annotation,
        manifest,
        verification_root=verification,
        config=SplitConfig(validation_fraction=0.5, seed=7),
    )
    assert len(result.records) == 4
    assert validate_training_manifest(manifest).passed
    rows = [
        json.loads(line)
        for line in (manifest / "episodes.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 3
    excluded = [
        json.loads(line)
        for line in (manifest / "excluded.jsonl").read_text().splitlines()
    ]
    assert [row["semantic_segment_id"] for row in excluded] == ["approach"]
    for source in {row["source_episode_id"] for row in rows}:
        assert (
            len({row["split"] for row in rows if row["source_episode_id"] == source})
            == 1
        )
    plan = make_plan(manifest)
    assert len(plan["runs"]) == 3
    assert all(run["transition_count"] > 0 for run in plan["runs"])
    assert all(run["semantic_segment_id"] for run in plan["runs"])
    assert all(
        run["training_use_status"] == "NOT_FOR_REAL_MODEL_TRAINING"
        for run in plan["runs"]
        if run["source_episode_id"] == "episode_000000"
    )
    assert all(
        run["episode_task"] == "Complete the exact synthetic placement task."
        for run in plan["runs"]
        if run["source_episode_id"] == "episode_000000"
    )
    corrected = next(
        run
        for run in plan["runs"]
        if run["source_episode_id"] == "episode_000000"
        and run["semantic_segment_id"] == "place"
    )
    assert corrected["final_instruction"] == "Set the block in the marked target."


def test_semantic_lerobot_roundtrip_and_openpi_horizon(manifest_tree, tmp_path):
    python = os.environ.get("VLA_LEROBOT_PYTHON")
    if not python:
        pytest.skip("set VLA_LEROBOT_PYTHON to run semantic boundary integration")
    tree = manifest_tree(1, status="AUTO_LABELED")
    # libsvtav1 does not support the tiny 8x8 unit-test images used elsewhere.
    for path in tree.curated.rglob("*.jpg"):
        with Image.open(path) as image:
            image.resize((128, 96)).save(path)
    annotation = tmp_path / "annotation-v2"
    verification = tmp_path / "verification-v2"
    manifest = tmp_path / "manifest-v2"
    output = tmp_path / "lerobot-v2"
    prepare(tree, annotation)
    verify_annotations(
        annotation,
        tree.quality,
        verification,
        policy=VerificationPolicy(0.0),
    )
    build_training_manifest(
        tree.curated,
        tree.quality,
        annotation,
        manifest,
        verification_root=verification,
    )
    result = export_lerobot(manifest, output, lerobot_python=python)
    assert result["export_runs"] == 2
    plan = make_plan(manifest)
    horizon = worker_call("openpi", {"plan": plan, "output_root": str(output)}, python)
    assert all(run["boundary_safe"] for run in horizon["runs"])
    assert all(run["storage_width"] == 17 for run in horizon["runs"])
    assert all(
        run["normalization"] == "SKIPPED; no stats computed or loaded"
        for run in horizon["runs"]
    )
