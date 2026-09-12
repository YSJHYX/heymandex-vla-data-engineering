"""Episode-level annotation pipeline over the synthetic annotated tree."""

from __future__ import annotations

import hashlib
import json

from vla_data.annotation import (
    DeterministicMockVLMProvider,
    annotate_episode,
)
from vla_data.annotation.pipeline import ANNOTATION_FILENAME, KEYFRAMES_FILENAME


def _jpg_hashes(root):
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.jpg"))
    }


def test_accept_episode_is_annotated_with_auto_labeled_review(annotated_tree) -> None:
    provider = DeterministicMockVLMProvider()
    result = annotate_episode(
        annotated_tree.curated / "episode_000000",
        quality_root=annotated_tree.quality,
        output_root=annotated_tree.output,
        provider=provider,
    )

    assert result.status == "SUCCESS", result.message
    assert provider.call_count == 1
    assert result.review_status == "AUTO_LABELED"
    document = json.loads(
        (annotated_tree.output / "episode_000000" / ANNOTATION_FILENAME).read_text()
    )
    assert document["review"]["status"] == "AUTO_LABELED"
    assert document["final_instruction"] is None
    assert document["quality_outcome"] == "ACCEPT"


def test_accept_with_warning_episode_is_still_eligible(annotated_tree) -> None:
    provider = DeterministicMockVLMProvider()
    result = annotate_episode(
        annotated_tree.curated / "episode_000001",
        quality_root=annotated_tree.quality,
        output_root=annotated_tree.output,
        provider=provider,
    )
    assert result.status == "SUCCESS"
    assert result.quality_outcome == "ACCEPT_WITH_WARNING"
    assert provider.call_count == 1


def test_excluded_episode_never_calls_the_provider(annotated_tree) -> None:
    provider = DeterministicMockVLMProvider()
    result = annotate_episode(
        annotated_tree.curated / "episode_000002",
        quality_root=annotated_tree.quality,
        output_root=annotated_tree.output,
        provider=provider,
    )
    assert result.status == "INELIGIBLE"
    assert result.quality_outcome == "EXCLUDE_FROM_EXPERT_TRAINING"
    assert provider.call_count == 0
    assert not (annotated_tree.output / "episode_000002").exists()


def test_missing_quality_artifacts_fail_without_provider_calls(annotated_tree) -> None:
    provider = DeterministicMockVLMProvider()
    result = annotate_episode(
        annotated_tree.curated / "episode_000000",
        quality_root=annotated_tree.quality / "nowhere",
        output_root=annotated_tree.output,
        provider=provider,
    )
    assert result.status == "FAILED"
    assert provider.call_count == 0


def test_source_jpgs_are_never_modified_and_no_base64_is_written(
    annotated_tree,
) -> None:
    before = _jpg_hashes(annotated_tree.curated)
    provider = DeterministicMockVLMProvider()
    annotate_episode(
        annotated_tree.curated / "episode_000000",
        quality_root=annotated_tree.quality,
        output_root=annotated_tree.output,
        provider=provider,
    )
    assert _jpg_hashes(annotated_tree.curated) == before

    for artifact in annotated_tree.output.rglob("*.json"):
        text = artifact.read_text()
        assert "base64" not in text
        assert len(text) < 20_000  # no embedded image payloads


def test_artifacts_contain_annotation_and_keyframes_provenance(annotated_tree) -> None:
    provider = DeterministicMockVLMProvider()
    annotate_episode(
        annotated_tree.curated / "episode_000000",
        quality_root=annotated_tree.quality,
        output_root=annotated_tree.output,
        provider=provider,
    )
    episode_output = annotated_tree.output / "episode_000000"
    keyframes = json.loads((episode_output / KEYFRAMES_FILENAME).read_text())
    assert keyframes["episode_id"] == "episode_000000"
    assert keyframes["prompt_version"] == "task_instruction_v1"
    for record in keyframes["temporal_points"]:
        assert record["head"]["source_path"].startswith(str(annotated_tree.curated))
        assert record["wrist"]["source_path"].startswith(str(annotated_tree.curated))


def test_curated_trajectory_and_metadata_are_untouched(annotated_tree) -> None:
    trajectory = annotated_tree.curated / "episode_000000" / "trajectory.npz"
    metadata = annotated_tree.curated / "episode_000000" / "metadata.json"
    before = (
        hashlib.sha256(trajectory.read_bytes()).hexdigest(),
        hashlib.sha256(metadata.read_bytes()).hexdigest(),
    )
    provider = DeterministicMockVLMProvider()
    annotate_episode(
        annotated_tree.curated / "episode_000000",
        quality_root=annotated_tree.quality,
        output_root=annotated_tree.output,
        provider=provider,
    )
    after = (
        hashlib.sha256(trajectory.read_bytes()).hexdigest(),
        hashlib.sha256(metadata.read_bytes()).hexdigest(),
    )
    assert before == after
