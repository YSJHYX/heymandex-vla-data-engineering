"""Dataset-level annotation: resume, staleness, force, isolation, concurrency."""

from __future__ import annotations

import json

import pytest

from vla_data.annotation import (
    DeterministicMockVLMProvider,
    annotate_dataset,
)
from vla_data.annotation.batch import STATUS_WOULD_PROCESS
from vla_data.annotation.provider import ProviderError


def _statuses(result):
    return [(item.episode_id, item.status) for item in result.results]


def test_dataset_run_gates_ineligible_episodes_and_annotates_eligible(
    annotated_tree,
) -> None:
    provider = DeterministicMockVLMProvider()
    result = annotate_dataset(
        annotated_tree.curated,
        annotated_tree.quality,
        annotated_tree.output,
        lambda: provider,
    )

    assert _statuses(result) == [
        ("episode_000000", "SUCCESS"),
        ("episode_000001", "SUCCESS"),
        ("episode_000002", "INELIGIBLE"),
    ]
    assert provider.call_count == 2  # the EXCLUDED episode never reaches GLM
    assert result.summary["episodes_discovered"] == 3
    assert result.summary["episodes_eligible"] == 2
    assert result.summary["episodes_ineligible"] == 1
    assert result.summary["review_state_counts"]["AUTO_LABELED"] == 2
    assert result.summary["mean_confidence"] == pytest.approx(0.9)
    assert result.summary_path is not None


def test_resume_skips_valid_matching_annotations(annotated_tree) -> None:
    provider = DeterministicMockVLMProvider()
    first = annotate_dataset(
        annotated_tree.curated,
        annotated_tree.quality,
        annotated_tree.output,
        lambda: provider,
    )
    second = annotate_dataset(
        annotated_tree.curated,
        annotated_tree.quality,
        annotated_tree.output,
        lambda: provider,
    )

    assert first.summary["episodes_processed"] == 2
    assert _statuses(second) == [
        ("episode_000000", "SKIPPED"),
        ("episode_000001", "SKIPPED"),
        ("episode_000002", "INELIGIBLE"),
    ]
    # No additional API spend on resume.
    assert provider.call_count == 2


def test_prompt_version_change_invalidates_resume(annotated_tree, monkeypatch) -> None:
    provider = DeterministicMockVLMProvider()
    annotate_dataset(
        annotated_tree.curated,
        annotated_tree.quality,
        annotated_tree.output,
        lambda: provider,
    )
    import vla_data.annotation.batch as batch_module

    monkeypatch.setattr(batch_module, "PROMPT_VERSION", "task_instruction_v2")
    rerun = annotate_dataset(
        annotated_tree.curated,
        annotated_tree.quality,
        annotated_tree.output,
        lambda: provider,
    )
    by_id = {item.episode_id: item for item in rerun.results}
    assert by_id["episode_000000"].status == "SUCCESS"
    assert by_id["episode_000000"].stale is True
    assert provider.call_count == 4  # both eligible episodes re-annotated


def test_model_change_invalidates_resume(annotated_tree) -> None:
    first_provider = DeterministicMockVLMProvider()
    annotate_dataset(
        annotated_tree.curated,
        annotated_tree.quality,
        annotated_tree.output,
        lambda: first_provider,
    )
    second_provider = DeterministicMockVLMProvider()
    second_provider.model = "mock-vlm-2"
    rerun = annotate_dataset(
        annotated_tree.curated,
        annotated_tree.quality,
        annotated_tree.output,
        lambda: second_provider,
    )
    by_id = {item.episode_id: item for item in rerun.results}
    assert by_id["episode_000000"].status == "SUCCESS"
    assert by_id["episode_000000"].stale is True
    assert second_provider.call_count == 2


def test_corrupt_annotation_is_reprocessed_not_skipped(annotated_tree) -> None:
    provider = DeterministicMockVLMProvider()
    annotate_dataset(
        annotated_tree.curated,
        annotated_tree.quality,
        annotated_tree.output,
        lambda: provider,
    )
    annotation_path = annotated_tree.output / "episode_000000" / "annotation.json"
    annotation_path.write_text("{ not valid json")

    rerun = annotate_dataset(
        annotated_tree.curated,
        annotated_tree.quality,
        annotated_tree.output,
        lambda: provider,
    )
    by_id = {item.episode_id: item for item in rerun.results}
    assert by_id["episode_000000"].status == "SUCCESS"
    assert by_id["episode_000000"].stale is True


def test_force_reprocesses_matching_annotations(annotated_tree) -> None:
    provider = DeterministicMockVLMProvider()
    annotate_dataset(
        annotated_tree.curated,
        annotated_tree.quality,
        annotated_tree.output,
        lambda: provider,
    )
    forced = annotate_dataset(
        annotated_tree.curated,
        annotated_tree.quality,
        annotated_tree.output,
        lambda: provider,
        force=True,
    )
    by_id = {item.episode_id: item for item in forced.results}
    assert by_id["episode_000000"].status == "SUCCESS"
    assert by_id["episode_000000"].stale is True
    assert by_id["episode_000002"].status == "INELIGIBLE"
    assert provider.call_count == 4


def test_provider_failure_is_isolated_per_episode(annotated_tree) -> None:
    calls = {"count": 0}

    class FlakyProvider(DeterministicMockVLMProvider):
        def annotate(self, request):
            calls["count"] += 1
            if request.episode_id == "episode_000001":
                raise ProviderError("INVALID_JSON", "model returned garbage")
            return super().annotate(request)

    provider = FlakyProvider()
    result = annotate_dataset(
        annotated_tree.curated,
        annotated_tree.quality,
        annotated_tree.output,
        lambda: provider,
    )
    assert _statuses(result) == [
        ("episode_000000", "SUCCESS"),
        ("episode_000001", "FAILED"),
        ("episode_000002", "INELIGIBLE"),
    ]
    assert result.failed_count == 1
    assert result.summary["invalid_response_count"] == 1


def test_concurrency_matches_serial_outcomes_and_ordering(annotated_tree, tmp_path):
    serial_provider = DeterministicMockVLMProvider()
    concurrent_provider = DeterministicMockVLMProvider()

    serial = annotate_dataset(
        annotated_tree.curated,
        annotated_tree.quality,
        tmp_path / "serial",
        lambda: serial_provider,
    )
    concurrent = annotate_dataset(
        annotated_tree.curated,
        annotated_tree.quality,
        tmp_path / "concurrent",
        lambda: concurrent_provider,
        concurrency=4,
    )

    assert _statuses(serial) == _statuses(concurrent)
    assert serial.summary == concurrent.summary
    assert concurrent_provider.call_count == 2


def test_dry_run_never_calls_provider_or_writes(annotated_tree) -> None:
    provider = DeterministicMockVLMProvider()
    result = annotate_dataset(
        annotated_tree.curated,
        annotated_tree.quality,
        annotated_tree.output,
        lambda: provider,
        dry_run=True,
    )

    assert _statuses(result) == [
        ("episode_000000", STATUS_WOULD_PROCESS),
        ("episode_000001", STATUS_WOULD_PROCESS),
        ("episode_000002", "INELIGIBLE"),
    ]
    assert provider.call_count == 0
    assert not annotated_tree.output.exists()

    annotate_dataset(
        annotated_tree.curated,
        annotated_tree.quality,
        annotated_tree.output,
        lambda: DeterministicMockVLMProvider(),
    )
    plan = annotate_dataset(
        annotated_tree.curated,
        annotated_tree.quality,
        annotated_tree.output,
        lambda: provider,
        dry_run=True,
    )
    assert _statuses(plan) == [
        ("episode_000000", "WOULD_SKIP"),
        ("episode_000001", "WOULD_SKIP"),
        ("episode_000002", "INELIGIBLE"),
    ]


def test_summary_records_retries_and_confidence_only_compactly(annotated_tree):
    provider = DeterministicMockVLMProvider(confidence=0.75)
    annotate_dataset(
        annotated_tree.curated,
        annotated_tree.quality,
        annotated_tree.output,
        lambda: provider,
    )
    document = json.loads(
        (annotated_tree.output / "dataset_annotation_summary.json").read_text()
    )
    assert document["mean_confidence"] == pytest.approx(0.75)
    assert document["retry_count"] == 0
    # Full model responses are never copied into the dataset summary.
    serialized = json.dumps(document)
    assert "raw_response_text" not in serialized
    assert len(serialized) < 8_000


def test_single_episode_mode_annotates_only_selection(annotated_tree) -> None:
    provider = DeterministicMockVLMProvider()
    result = annotate_dataset(
        annotated_tree.curated,
        annotated_tree.quality,
        annotated_tree.output,
        lambda: provider,
        episode="episode_000001",
    )
    assert _statuses(result) == [("episode_000001", "SUCCESS")]
    assert provider.call_count == 1
