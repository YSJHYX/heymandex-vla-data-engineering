import copy
import json

import pytest

from vla_data.annotation.glm_provider import DeterministicStructuredMockProvider
from vla_data.annotation.hierarchical import (
    annotate_hierarchical_episode,
    clean_domains,
)
from vla_data.annotation.provider import ERROR_INVALID_JSON, ProviderError
from vla_data.annotation.schema import (
    AnnotationSchemaError,
    build_annotation,
    load_annotation,
    validate_hierarchical_annotation,
)
from vla_data.io.curated_episode import CuratedEpisode


def payloads(end=3, middle=1):
    coarse = {
        "episode_task": {
            "instruction": "Place the block in the target area.",
            "confidence": 0.95,
            "paraphrases": ["Move the block into the target area."],
        },
        "semantic_segments": [
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
        ],
        "non_training_intervals": [],
    }
    refined = {
        key: copy.deepcopy(coarse[key])
        for key in ("semantic_segments", "non_training_intervals")
    }
    return coarse, refined


def test_schema_v2_and_legacy_v1_are_both_explicitly_readable(tmp_path):
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(
        json.dumps(
            build_annotation(
                episode_id="episode_000000",
                quality_outcome="ACCEPT",
                model_annotation={
                    "provider": "mock",
                    "model": "mock",
                    "prompt_version": "v1",
                    "instruction": "Place the block.",
                    "confidence": 0.9,
                    "objects": [],
                    "task_type": None,
                    "uncertainty": None,
                },
            )
        )
    )
    legacy = load_annotation(legacy_path)
    assert legacy["schema_version"] == 1
    assert "semantic_segments" not in legacy
    coarse, refined = payloads()
    candidate = {
        "schema_name": "vla_episode_annotation",
        "schema_version": 2,
        "annotation_schema_version": 2,
        "episode_id": "episode_000000",
        "quality_outcome": "ACCEPT",
        "episode_task": coarse["episode_task"],
        **refined,
        "clean_domains": [
            {
                "d2_segment_id": 0,
                "clean_run_id": 0,
                "start_curated_index": 0,
                "end_curated_index": 3,
            }
        ],
        "model_provenance": {
            "provider": "mock",
            "model": "mock",
            "pass_a_prompt_version": "a",
            "pass_b_prompt_version": "b",
        },
    }
    assert validate_hierarchical_annotation(candidate)["schema_version"] == 2
    malformed = copy.deepcopy(candidate)
    malformed["semantic_segments"][1]["start_curated_index"] = 0
    with pytest.raises(AnnotationSchemaError, match="overlap"):
        validate_hierarchical_annotation(malformed)
    malformed = copy.deepcopy(candidate)
    del malformed["episode_task"]["confidence"]
    with pytest.raises(AnnotationSchemaError, match="confidence"):
        validate_hierarchical_annotation(malformed)
    with_non_training = copy.deepcopy(candidate)
    with_non_training["semantic_segments"] = with_non_training["semantic_segments"][:1]
    with_non_training["non_training_intervals"] = [
        {
            "start_curated_index": 1,
            "end_curated_index": 3,
            "reason": "No single visually grounded manipulation subtask.",
        }
    ]
    assert validate_hierarchical_annotation(with_non_training) is with_non_training
    for domains in (
        [
            {
                "d2_segment_id": 0,
                "clean_run_id": 0,
                "start_curated_index": 0,
                "end_curated_index": 1,
            },
            {
                "d2_segment_id": 1,
                "clean_run_id": 0,
                "start_curated_index": 1,
                "end_curated_index": 3,
            },
        ],
        [
            {
                "d2_segment_id": 0,
                "clean_run_id": 0,
                "start_curated_index": 0,
                "end_curated_index": 1,
            },
            {
                "d2_segment_id": 0,
                "clean_run_id": 1,
                "start_curated_index": 2,
                "end_curated_index": 3,
            },
        ],
    ):
        crossing = copy.deepcopy(candidate)
        crossing["clean_domains"] = domains
        crossing["semantic_segments"] = [
            {
                **crossing["semantic_segments"][0],
                "start_curated_index": 0,
                "end_curated_index": 3,
            }
        ]
        crossing["non_training_intervals"] = []
        with pytest.raises(AnnotationSchemaError, match="crosses"):
            validate_hierarchical_annotation(crossing)


def test_two_pass_annotation_uses_only_provided_indices_and_writes_v2(
    annotated_tree, tmp_path
):
    tree = annotated_tree
    end = CuratedEpisode.load(tree.curated / "episode_000000").transition_count
    coarse, refined = payloads(end, max(1, end // 2))
    provider = DeterministicStructuredMockProvider(coarse, refined)
    output = tmp_path / "hierarchical"
    result = annotate_hierarchical_episode(
        tree.curated / "episode_000000",
        quality_root=tree.quality,
        output_root=output,
        provider=provider,
    )
    assert result.status == "SUCCESS" and len(provider.calls) == 2
    annotation = load_annotation(output / "episode_000000/annotation.json")
    assert annotation["annotation_schema_version"] == 2
    assert (
        annotation["source_dataset_status"]
        == CuratedEpisode.load(tree.curated / "episode_000000").metadata[
            "source_dataset_status"
        ]
    )
    assert annotation["model_provenance"]["pass_a_sampling_strategy"].startswith(
        "deterministic_"
    )
    keyframes = json.loads((output / "episode_000000/keyframes.json").read_text())
    assert keyframes["pass_a"]["frames"][0].keys() >= {
        "curated_index",
        "timestamp_ns",
        "relative_time_ns",
        "head",
        "right_wrist",
    }


def test_invented_boundary_and_missing_camera_fail_closed(annotated_tree, tmp_path):
    tree = annotated_tree
    end = CuratedEpisode.load(tree.curated / "episode_000000").transition_count
    coarse, refined = payloads(end, max(1, end // 2))
    coarse["semantic_segments"][0]["end_curated_index"] = 999
    provider = DeterministicStructuredMockProvider(coarse, refined)
    result = annotate_hierarchical_episode(
        tree.curated / "episode_000000",
        quality_root=tree.quality,
        output_root=tmp_path / "bad-boundary",
        provider=provider,
    )
    assert result.status == "FAILED"
    assert "invented" in result.message

    wrist = next((tree.curated / "episode_000000/media/right_wrist/rgb").glob("*.jpg"))
    wrist.unlink()
    result = annotate_hierarchical_episode(
        tree.curated / "episode_000000",
        quality_root=tree.quality,
        output_root=tmp_path / "missing-camera",
        provider=DeterministicStructuredMockProvider(*payloads(end, max(1, end // 2))),
    )
    assert result.status == "FAILED"
    assert "missing" in result.message

    class MalformedProvider:
        provider_name = "malformed"
        model = "malformed"

        def infer_json(self, request):
            raise ProviderError(ERROR_INVALID_JSON, "response contains no JSON object")

    result = annotate_hierarchical_episode(
        tree.curated / "episode_000001",
        quality_root=tree.quality,
        output_root=tmp_path / "malformed",
        provider=MalformedProvider(),
    )
    assert result.status == "FAILED"
    assert "INVALID_JSON" in result.message


def test_clean_domains_do_not_cross_d2_or_d3_holes():
    import numpy as np

    assert clean_domains(
        (0, 3, 6), np.array([True, False, True, True, True, False])
    ) == [
        {
            "d2_segment_id": 0,
            "clean_run_id": 0,
            "start_curated_index": 0,
            "end_curated_index": 1,
        },
        {
            "d2_segment_id": 0,
            "clean_run_id": 1,
            "start_curated_index": 2,
            "end_curated_index": 3,
        },
        {
            "d2_segment_id": 1,
            "clean_run_id": 0,
            "start_curated_index": 3,
            "end_curated_index": 5,
        },
    ]


def test_single_subtask_mock_is_not_oversegmented(annotated_tree, tmp_path):
    episode = CuratedEpisode.load(annotated_tree.curated / "episode_000000")
    segment = {
        "segment_id": "semantic_000",
        "start_curated_index": 0,
        "end_curated_index": episode.transition_count,
        "instruction": "Place the block in the target area.",
        "confidence": 0.95,
        "paraphrases": [],
    }
    provider = DeterministicStructuredMockProvider(
        {
            "episode_task": {
                "instruction": "Place the block in the target area.",
                "confidence": 0.95,
                "paraphrases": [],
            },
            "semantic_segments": [segment],
            "non_training_intervals": [],
        },
        {"semantic_segments": [segment], "non_training_intervals": []},
    )
    result = annotate_hierarchical_episode(
        episode.episode_dir,
        quality_root=annotated_tree.quality,
        output_root=tmp_path / "single",
        provider=provider,
    )
    assert result.status == "SUCCESS"
    assert result.semantic_segments == 1
