"""Small physical-contract-valid inputs with explicitly synthetic review decisions."""

from types import SimpleNamespace

import pytest

from vla_data.annotation.schema import apply_review, build_annotation, write_annotation
from vla_data.batch.runner import build_curated_dataset, quality_dataset
from vla_data.verification import VerificationPolicy, review_episode, verify_annotations


@pytest.fixture
def manifest_tree(raw_dataset_factory, tmp_path):
    def factory(count=1, status="HUMAN_VERIFIED"):
        raw = raw_dataset_factory(tmp_path / "raw", tuple(range(count)))
        curated, quality, annotation = (
            tmp_path / name for name in ("curated", "quality", "annotation")
        )
        assert not build_curated_dataset(raw, curated).failed_count
        assert not quality_dataset(curated, quality).failed_count
        for number in range(count):
            eid = f"episode_{number:06d}"
            document = build_annotation(
                episode_id=eid,
                quality_outcome="ACCEPT",
                model_annotation={
                    "provider": "synthetic",
                    "model": "fixture",
                    "prompt_version": "fixture_v1",
                    "instruction": f"Move the test block to fixture {number}.",
                    "confidence": 0.99,
                    "objects": ["test block"],
                    "task_type": "synthetic transport",
                    "uncertainty": None,
                },
            )
            if status == "AUTO_ACCEPTED":
                document = apply_review(
                    document,
                    status=status,
                )
            if status == "AUTO_ACCEPTED":
                document["annotation_policy"] = {
                    "auto_accept_enabled": True,
                    "policy_id": "synthetic_explicit_policy",
                }
            write_annotation(annotation / eid / "annotation.json", document)
        verification = tmp_path / "verification"
        policy = VerificationPolicy(1.0 if status == "AUTO_LABELED" else 0.5)
        verify_annotations(annotation, quality, verification, policy=policy)
        if status in {"HUMAN_VERIFIED", "HUMAN_CORRECTED", "REJECTED"}:
            for number in range(count):
                review_episode(
                    verification,
                    f"episode_{number:06d}",
                    status=status,
                    instruction="Move the RED test block."
                    if status == "HUMAN_CORRECTED"
                    else None,
                )
        return SimpleNamespace(
            curated=curated,
            quality=quality,
            annotation=annotation,
            verification=verification,
            policy=policy,
            output=tmp_path / "manifest",
        )

    return factory
