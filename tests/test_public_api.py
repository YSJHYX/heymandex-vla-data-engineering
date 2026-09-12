"""Smoke tests for the task-level package API promised to runner callers."""


def test_public_episode_apis_are_importable() -> None:
    from vla_data.cleaning import build_curated_episode, build_curated_v1
    from vla_data.io import CuratedEpisode, RawEpisode
    from vla_data.quality import evaluate_episode
    from vla_data.validation import validate_curated_episode

    assert build_curated_episode is build_curated_v1
    assert callable(validate_curated_episode)
    assert callable(evaluate_episode)
    assert RawEpisode.__name__ == "RawEpisode"
    assert CuratedEpisode.__name__ == "CuratedEpisode"
