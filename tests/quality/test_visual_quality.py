from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

from vla_data.io.curated_episode import CuratedEpisode
from vla_data.quality.visual import VisualQualityConfig, evaluate_visual


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_missing_jpg_removes_referencing_transitions(curated_v1_episode) -> None:
    path = curated_v1_episode / "media" / "head" / "rgb" / "000001.jpg"
    path.unlink()

    result = evaluate_visual(
        CuratedEpisode.load(curated_v1_episode), VisualQualityConfig()
    )

    assert result.report["head"]["status"] == "EXCLUDE"
    assert result.keep_mask.tolist() == [True, True, False]


def test_corrupt_jpg_detected(curated_v1_episode) -> None:
    path = curated_v1_episode / "media" / "head" / "rgb" / "000001.jpg"
    path.write_bytes(b"not a jpeg")

    result = evaluate_visual(
        CuratedEpisode.load(curated_v1_episode), VisualQualityConfig()
    )

    assert result.report["head"]["decode_failures"] == 1
    assert result.keep_mask.tolist() == [True, True, False]


def test_wrong_resolution_detected(curated_v1_episode) -> None:
    path = curated_v1_episode / "media" / "head" / "rgb" / "000001.jpg"
    Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(path)

    result = evaluate_visual(
        CuratedEpisode.load(curated_v1_episode), VisualQualityConfig()
    )

    assert result.report["head"]["resolution_inconsistencies"] == 1
    assert not np.all(result.keep_mask)


def test_wrong_channel_count_detected(curated_v1_episode) -> None:
    path = curated_v1_episode / "media" / "head" / "rgb" / "000001.jpg"
    Image.fromarray(np.zeros((8, 10), dtype=np.uint8), mode="L").save(path)

    result = evaluate_visual(
        CuratedEpisode.load(curated_v1_episode), VisualQualityConfig()
    )

    assert result.report["head"]["hard_invalid_frame_count"] == 1
    assert not np.all(result.keep_mask)


def test_dark_and_bright_frame_metrics(curated_v1_episode) -> None:
    head = curated_v1_episode / "media" / "head" / "rgb"
    Image.fromarray(np.zeros((8, 10, 3), dtype=np.uint8)).save(head / "000000.jpg")
    Image.fromarray(np.full((8, 10, 3), 255, dtype=np.uint8)).save(head / "000001.jpg")

    result = evaluate_visual(
        CuratedEpisode.load(curated_v1_episode), VisualQualityConfig()
    )

    assert result.report["head"]["dark_frame_count"] == 1
    assert result.report["head"]["bright_frame_count"] == 1


def test_blur_metric_is_deterministic(curated_v1_episode) -> None:
    episode = CuratedEpisode.load(curated_v1_episode)
    first = evaluate_visual(episode, VisualQualityConfig())
    second = evaluate_visual(episode, VisualQualityConfig())

    assert first.report["head"]["blur_score"] == second.report["head"]["blur_score"]


def test_duplicate_reference_is_legal_and_freeze_is_measured(
    curated_v1_episode,
) -> None:
    result = evaluate_visual(
        CuratedEpisode.load(curated_v1_episode), VisualQualityConfig()
    )
    duplicate = result.report["head"]["duplicate_summary"]

    assert np.all(result.keep_mask)
    assert result.report["head"]["status"] == "CLEAN"
    assert duplicate["same_reference_pair_count"] == 1
    assert duplicate["longest_same_reference_run"] == 2
    assert duplicate["longest_near_static_visual_run"] >= 2


def test_rgb_evaluator_never_modifies_jpeg(curated_v1_episode) -> None:
    path = curated_v1_episode / "media" / "head" / "rgb" / "000000.jpg"
    before = _digest(path)

    evaluate_visual(CuratedEpisode.load(curated_v1_episode), VisualQualityConfig())

    assert _digest(path) == before
