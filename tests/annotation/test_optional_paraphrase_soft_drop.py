"""D4.3.1 frozen contract: invalid optional paraphrases soft-drop everywhere.

Canonical episode_task.instruction and semantic_segments[*].instruction stay
hard-gated; a single bad paraphrase must never fail the annotation — for
every prompt family, both production and LIBERO benchmark paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vla_data.annotation.boundary_local import build_boundary_transitions
from vla_data.annotation.glm_provider import DeterministicStructuredMockProvider
from vla_data.annotation.hierarchical import (
    _sanitize_payload_language,
    annotate_hierarchical_episode,
)
from vla_data.annotation.platform_context import DEFAULT_PLATFORM_CONTEXT
from vla_data.annotation.schema import AnnotationSchemaError, load_annotation
from vla_data.benchmark.libero import LIBERO_PLATFORM_CONTEXT


def _payload(task_paraphrases: list[str], segment_paraphrases: list[str]) -> dict:
    return {
        "episode_task": {
            "instruction": "Take the device from the operator",
            "confidence": 0.8,
            "paraphrases": task_paraphrases,
        },
        "semantic_segments": [
            {
                "segment_id": "reach",
                "start_curated_index": 0,
                "end_curated_index": 2,
                "instruction": "Reach toward the device held by the operator",
                "confidence": 0.7,
                "paraphrases": segment_paraphrases,
            },
            {
                "segment_id": "take",
                "start_curated_index": 2,
                "end_curated_index": 3,
                "instruction": "Take the device from the operator",
                "confidence": 0.7,
                "paraphrases": [],
            },
        ],
        "non_training_intervals": [],
    }


# ------------------------------------------------------- sanitizer unit level


def test_invalid_paraphrase_soft_dropped_canonical_preserved() -> None:
    cleaned, rejected = _sanitize_payload_language(
        _payload(["Pick it up"], ["Grab the device"])
    )
    assert cleaned["episode_task"]["instruction"] == "Take the device from the operator"
    assert cleaned["episode_task"]["paraphrases"] == []
    assert cleaned["semantic_segments"][0]["paraphrases"] == ["Grab the device"]
    assert rejected == [
        {
            "field": "episode_task.paraphrases[0]",
            "text": "Pick it up",
            "reason": "UNNAMED_PRONOUN_OBJECT",
        }
    ]


def test_good_paraphrase_kept_bad_removed() -> None:
    cleaned, rejected = _sanitize_payload_language(
        _payload(["Take the device", "Pick it up"], [])
    )
    assert cleaned["episode_task"]["paraphrases"] == ["Take the device"]
    assert len(rejected) == 1
    assert rejected[0]["reason"] == "UNNAMED_PRONOUN_OBJECT"


def test_embodiment_centric_paraphrase_soft_dropped() -> None:
    cleaned, rejected = _sanitize_payload_language(
        _payload(["Move the robot arm toward the device"], [])
    )
    assert cleaned["episode_task"]["paraphrases"] == []
    assert rejected[0]["reason"] == "EMBODIMENT_CENTRIC_LANGUAGE"


def test_invalid_canonical_instruction_still_hard_fails() -> None:
    payload = _payload([], [])
    payload["episode_task"]["instruction"] = "The robot appears to take the device"
    with pytest.raises(AnnotationSchemaError, match="episode_task.instruction"):
        _sanitize_payload_language(payload)


def test_invalid_segment_instruction_still_hard_fails() -> None:
    payload = _payload([], [])
    payload["semantic_segments"][0]["instruction"] = "Move joint 3"
    with pytest.raises(
        AnnotationSchemaError, match=r"semantic_segments\[0\].instruction"
    ):
        _sanitize_payload_language(payload)


# ------------------------------------------------------------ episode level


def _libero_episode_fixture(tmp_path: Path, family: str) -> tuple[Path, Path]:
    episode_dir = tmp_path / "observations" / "episode_000001"
    for role, color in (("head", (200, 40, 40)), ("right_wrist", (40, 40, 200))):
        view_dir = episode_dir / "media" / role / "rgb"
        view_dir.mkdir(parents=True)
        for index in range(3):
            Image.new("RGB", (32, 24), color).save(view_dir / f"{index:06d}.jpg")
    (episode_dir / "benchmark_episode.json").write_text(
        json.dumps(
            {
                "episode_id": "episode_000001",
                "source_frame_count": 3,
                "available_views": ["head", "right_wrist"],
                "timestamps_ns": [0, 10**8, 2 * 10**8],
            }
        )
    )
    quality = tmp_path / "source_index" / "episode_000001"
    quality.mkdir(parents=True)
    np.save(quality / "quality_mask.npy", np.ones(3, dtype=bool))
    (quality / "quality_report.json").write_text(
        json.dumps(
            {
                "schema_name": "vla_quality_report",
                "schema_version": 1,
                "episode_id": "episode_000001",
                "status": "ACCEPT",
            }
        )
    )
    return episode_dir, tmp_path / "annotations"


def _local_response(transition: dict) -> dict:
    return {
        "transition_id": transition["transition_id"],
        "previous_instruction": transition["previous"]["instruction"],
        "next_instruction": transition["next"]["instruction"],
        "boundary_curated_index": transition["coarse_boundary_curated_index"],
        "confidence": 0.7,
        "evidence_status": "SUPPORTED",
        "semantic_correction": None,
    }


@pytest.mark.parametrize(
    "family_context",
    [("v3.2", LIBERO_PLATFORM_CONTEXT), ("v3.2", DEFAULT_PLATFORM_CONTEXT)],
    ids=["libero_benchmark_path", "production_platform_path"],
)
def test_annotation_succeeds_with_invalid_paraphrase_all_paths(
    tmp_path, family_context
) -> None:
    family_name, context = family_context
    episode_dir, output_root = _libero_episode_fixture(tmp_path, family_name)
    payload = _payload(["Pick it up"], [])  # invalid paraphrase on purpose
    domain = [
        {
            "d2_segment_id": 0,
            "clean_run_id": 0,
            "start_curated_index": 0,
            "end_curated_index": 3,
        }
    ]
    transitions = build_boundary_transitions(payload, domain, radius=1)
    provider = DeterministicStructuredMockProvider(
        payload, *[_local_response(t) for t in transitions]
    )
    result = annotate_hierarchical_episode(
        episode_dir,
        quality_root=tmp_path / "source_index",
        output_root=output_root,
        provider=provider,
        prompt_family=family_name,
        platform_context=context,
    )
    assert result.status == "SUCCESS", result.message
    annotation = load_annotation(output_root / "episode_000001/annotation.json")
    assert annotation["episode_task"]["paraphrases"] == []
    assert (
        annotation["episode_task"]["instruction"] == "Take the device from the operator"
    )
    assert len(annotation["semantic_segments"]) == 2
    quarantine = annotation["diagnostics"]["rejected_paraphrases"]
    assert quarantine[0]["reason"] == "UNNAMED_PRONOUN_OBJECT"
    assert (
        "REJECTED_OPTIONAL_PARAPHRASE" in annotation["diagnostics"]["diagnostic_codes"]
    )


def test_no_provider_regeneration_for_bad_paraphrase(tmp_path) -> None:
    episode_dir, output_root = _libero_episode_fixture(tmp_path, "v3.2")
    payload = _payload(["Pick it up"], [])
    domain = [
        {
            "d2_segment_id": 0,
            "clean_run_id": 0,
            "start_curated_index": 0,
            "end_curated_index": 3,
        }
    ]
    transitions = build_boundary_transitions(payload, domain, radius=1)
    provider = DeterministicStructuredMockProvider(
        payload, *[_local_response(t) for t in transitions]
    )
    result = annotate_hierarchical_episode(
        episode_dir,
        quality_root=tmp_path / "source_index",
        output_root=output_root,
        provider=provider,
        prompt_family="v3.2",
        platform_context=LIBERO_PLATFORM_CONTEXT,
    )
    assert result.status == "SUCCESS"
    # Pass A + boundary-local Pass B calls only — no regeneration retry.
    assert result.pass_a_calls == 1
