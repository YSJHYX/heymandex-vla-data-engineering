import hashlib
import json

import numpy as np
import pytest

from vla_data.diagnostics import rgb_review as review


@pytest.fixture
def tree(curated_v1_episode, tmp_path):
    quality = tmp_path / "quality"
    (quality / curated_v1_episode.name).mkdir(parents=True)
    np.save(
        quality / curated_v1_episode.name / "quality_mask.npy",
        np.array([True, False, True]),
    )
    return (
        curated_v1_episode.parent,
        quality,
        tmp_path / "review",
        curated_v1_episode.name,
    )


@pytest.mark.parametrize(
    "selection,indices", [("valid", [0, 2]), ("invalid", [1]), ("all", [0, 1, 2])]
)
def test_selection_and_order(tree, selection, indices):
    curated, quality, _, episode = tree
    plan = review.review_plan(curated / episode / "..", quality, episode, selection)
    assert [r["curated_row"] for r in plan["rows"]] == indices
    assert plan["rows"][0]["images"]["head"].endswith("head/rgb/000000.jpg")
    assert "right_wrist/rgb" in plan["rows"][0]["images"]["right_wrist"]


def test_no_mutation_and_mapping(tree, tmp_path, monkeypatch):
    curated, quality, _, episode = tree
    # Output cannot be inside the (fixture) Curated root.
    output = tmp_path.parent / (tmp_path.name + "-review")
    paths = [
        p
        for root in (curated / episode, quality)
        for p in root.rglob("*")
        if p.is_file()
    ]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    def encode(plan, path):
        frame = review.compose_frame(plan, plan["rows"][0])
        assert frame[25, 1, 0] == 32
        assert frame[25, 11, 0] == 64
        path.write_bytes(b"synthetic encoder stub")

    monkeypatch.setattr(review, "_encode", encode)
    result = review.render_rgb_review(
        curated, quality, output, episode=episode, selection="all"
    )
    assert result["frame_count"] == 3
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_missing_reference_reports_reason(tree):
    curated, quality, _, episode = tree
    (curated / episode / "media/head/rgb/000001.jpg").unlink()
    result = review.review_plan(curated, quality, episode)
    assert [r["curated_row"] for r in result["rows"]] == [0, 1]
    assert result["skipped"][0]["curated_row"] == 2
    assert "MISSING_OR_UNDECODABLE head" in result["skipped"][0]["reasons"][0]


def test_empty_invalid_and_dry_run(tree, tmp_path):
    curated, quality, _, episode = tree
    output = tmp_path.parent / (tmp_path.name + "-review")
    np.save(quality / episode / "quality_mask.npy", np.ones(3, dtype=bool))
    dry = review.render_rgb_review(
        curated, quality, output, episode=episode, selection="invalid", dry_run=True
    )
    assert dry["frame_count"] == 0 and not output.exists()
    result = review.render_rgb_review(
        curated, quality, output, episode=episode, selection="invalid"
    )
    assert result["status"] == "EMPTY" and result["video_path"] is None
    assert (
        json.loads((output / episode / "invalid_head_wrist.json").read_text())["rows"]
        == []
    )


def test_output_cannot_overlap_sources(tree):
    curated, quality, _, episode = tree
    with pytest.raises(ValueError, match="separate"):
        review.render_rgb_review(curated, quality, curated / "review", episode=episode)
