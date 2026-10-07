"""Differential equivalence: sorted-index selectors vs the flatnonzero oracle.

The legacy ``_select_feedback`` / ``_select_camera`` functions are frozen as
the semantic oracle; ``_SortedTimestampCandidates`` must reproduce their row
choice, tie-breaking, and rejection classification exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from vla_data.cleaning.causal_sync import (
    _pick_indexed_camera,
    _pick_indexed_feedback,
    _select_camera,
    _select_feedback,
    _SortedTimestampCandidates,
)


def _compare_feedback(timestamps, valid, boundary, before):
    reasons_old: list[str] = []
    reasons_new: list[str] = []
    old = _select_feedback(
        timestamps,
        valid,
        boundary=boundary,
        before=before,
        component="X",
        reasons=reasons_old,
    )
    selector = _SortedTimestampCandidates(timestamps, valid)
    new = _pick_indexed_feedback(
        selector,
        boundary=boundary,
        before=before,
        component="X",
        reasons=reasons_new,
    )
    assert old == new
    assert reasons_old == reasons_new


def _compare_camera(timestamps, valid, frame_indices, boundary, media_dir, media):
    reasons_old: list[str] = []
    reasons_new: list[str] = []
    old = _select_camera(
        timestamps=timestamps,
        valid=valid,
        frame_indices=frame_indices,
        boundary=boundary,
        media_dir=media_dir,
        component="C",
        reasons=reasons_old,
        media_integrity=media,
    )
    selector = _SortedTimestampCandidates(
        timestamps,
        np.asarray(valid, dtype=bool) & (frame_indices >= 0) & (timestamps > 0),
    )
    new = _pick_indexed_camera(
        selector,
        frame_indices=frame_indices,
        boundary=boundary,
        media_dir=media_dir,
        component="C",
        reasons=reasons_new,
        media_integrity=media,
    )
    assert old == new
    assert reasons_old == reasons_new


def test_equivalence_normal_monotonic_timestamps() -> None:
    timestamps = np.arange(100, 900, 7, dtype=np.int64)
    valid = np.ones(timestamps.size, dtype=bool)
    for boundary in (0, 100, 101, 450, 890, 10_000):
        _compare_feedback(timestamps, valid, boundary, before=True)
        _compare_feedback(timestamps, valid, boundary, before=False)


def test_equivalence_duplicate_timestamps_tie_breaking() -> None:
    # Heavy duplicates: ties must resolve to last (before) / first (after)
    # original row exactly like the oracle's argmax/argmin over candidates.
    timestamps = np.array([5, 5, 5, 5, 20, 20, 20, 50, 50, 90, 90, 90], dtype=np.int64)
    valid = np.array(
        [True, False, True, True, False, True, True, False, True, True, False, True]
    )
    for boundary in (5, 6, 20, 21, 50, 51, 90, 91, 200):
        _compare_feedback(timestamps, valid, boundary, before=True)
        _compare_feedback(timestamps, valid, boundary, before=False)


def test_equivalence_duplicate_ties_where_extremes_are_invalid() -> None:
    # The max/min timestamp rows themselves invalid: the pick must move to the
    # next equal timestamp candidate, keeping the oracle's tie rule.
    timestamps = np.array([10, 30, 30, 30, 30, 60], dtype=np.int64)
    valid = np.array([True, False, False, True, False, True])
    for boundary in (30, 31, 60, 61):
        _compare_feedback(timestamps, valid, boundary, before=True)
        _compare_feedback(timestamps, valid, boundary, before=False)


def test_equivalence_nonpositive_and_boundary_edges() -> None:
    # Zero / negative timestamps are never candidates (positive filter), and
    # boundaries equal to a timestamp are strict on both sides.
    timestamps = np.array([-500, 0, 0, 40, 40, 80], dtype=np.int64)
    valid = np.ones(timestamps.size, dtype=bool)
    for boundary in (-600, -1, 0, 1, 40, 41, 80, 81, 10_000):
        _compare_feedback(timestamps, valid, boundary, before=True)
        _compare_feedback(timestamps, valid, boundary, before=False)


def test_equivalence_before_first_and_after_last() -> None:
    timestamps = np.array([1_000, 2_000, 3_000], dtype=np.int64)
    valid = np.array([True, False, True])
    for boundary in (0, 999, 1_000, 1_001, 2_999, 3_000, 3_001):
        _compare_feedback(timestamps, valid, boundary, before=True)
        _compare_feedback(timestamps, valid, boundary, before=False)


def test_equivalence_all_invalid_rows_report_feedback_invalid() -> None:
    timestamps = np.array([100, 200, 300], dtype=np.int64)
    for valid in (
        np.array([False, False, False]),
        np.array([False, True, False]),
    ):
        _compare_feedback(timestamps, valid, 250, before=True)
        _compare_feedback(timestamps, valid, 150, before=False)


def test_equivalence_sparse_valid_rows() -> None:
    rng = np.random.default_rng(7)
    timestamps = rng.integers(1, 5_000, size=800).astype(np.int64) * 1_000_000
    valid = rng.random(800) > 0.7
    for boundary in (0, 1, 500_000_000, 2_500_000_000, 5_000_000_000):
        _compare_feedback(timestamps, valid, boundary, before=True)
        _compare_feedback(timestamps, valid, boundary, before=False)


def test_equivalence_unsorted_timestamps_match_oracle() -> None:
    # The oracle never assumes monotonicity; neither may the sorted index.
    rng = np.random.default_rng(11)
    timestamps = rng.integers(-50, 9_000, size=600).astype(np.int64)
    valid = rng.random(600) > 0.4
    for boundary in range(-100, 9_100, 137):
        _compare_feedback(timestamps, valid, boundary, before=True)
        _compare_feedback(timestamps, valid, boundary, before=False)


def test_equivalence_camera_selection(tmp_path) -> None:
    from PIL import Image

    media = tmp_path / "rgb"
    media.mkdir()
    for index in range(6):
        Image.new("RGB", (8, 6), (index * 30, 60, 90)).save(media / f"{index:06d}.jpg")
    # A corrupt file exercises DECODE_INVALID on both implementations.
    (media / "000009.jpg").write_bytes(b"\xff\xd8\xff garbage")

    timestamps = np.array([10, 10, 40, 40, 70, 70, 90], dtype=np.int64)
    valid = np.array([True, True, True, False, True, True, False])
    frame_indices = np.array([0, 1, 2, 3, 4, 5, 9])
    media_integrity_old: dict = {}
    media_integrity_new: dict = {}
    for boundary in (0, 10, 11, 40, 41, 70, 71, 90, 91, 500):
        _compare_camera(
            timestamps, valid, frame_indices, boundary, media, media_integrity_old
        )
        assert media_integrity_old == media_integrity_new or boundary  # shared state
        _compare_camera(
            timestamps, valid, frame_indices, boundary, media, media_integrity_new
        )


def test_equivalence_camera_negative_frame_indices(tmp_path) -> None:
    media = tmp_path / "rgb"
    media.mkdir()
    from PIL import Image

    Image.new("RGB", (8, 6), (10, 20, 30)).save(media / "000000.jpg")
    timestamps = np.array([10, 20, 30], dtype=np.int64)
    valid = np.array([True, True, True])
    frame_indices = np.array([-1, 0, -1])
    for boundary in (0, 10, 20, 30, 40):
        _compare_camera(timestamps, valid, frame_indices, boundary, media, {})


def test_equivalence_legacy_positive_filter_is_classification_only() -> None:
    # Frozen legacy quirk: when the window contains positive timestamps (so
    # UNAVAILABLE does not fire) but the only eligible rows have nonpositive
    # timestamps, the oracle still SELECTS such a row (the positive filter
    # applies to the UNAVAILABLE classification, not to candidates).
    timestamps = np.array([50, -40, -40, 60], dtype=np.int64)
    valid = np.array([False, True, True, False])
    _compare_feedback(timestamps, valid, 100, before=True)
    _compare_feedback(timestamps, valid, 10, before=False)

    # Same quirk on the after side: negative boundary can admit nonpositive
    # eligible rows that beat every positive candidate for the min pick.
    timestamps = np.array([900, 800, -70], dtype=np.int64)
    valid = np.array([True, True, True])
    _compare_feedback(timestamps, valid, -10, before=False)
    _compare_feedback(timestamps, valid, -100, before=False)


@pytest.mark.parametrize("seed", range(12))
def test_equivalence_randomized_fuzz(seed) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(40):
        size = int(rng.integers(1, 120))
        timestamps = rng.integers(-100, 10_000, size=size).astype(np.int64)
        if rng.random() < 0.5:  # inject heavy duplication
            timestamps = np.repeat(timestamps, 2)[:size]
        valid = rng.random(size) > 0.35
        boundary = int(rng.integers(-150, 10_050))
        _compare_feedback(timestamps, valid, boundary, before=True)
        _compare_feedback(timestamps, valid, boundary, before=False)
