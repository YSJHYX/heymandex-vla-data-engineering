"""Read-only RGB integrity and visual-quality measurement."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, UnidentifiedImageError

from vla_data.io.curated_episode import CuratedEpisode
from vla_data.quality.metrics import summary


@dataclass(frozen=True)
class VisualQualityConfig:
    """Metric definitions plus optional quality anomaly thresholds."""

    dark_pixel_luminance: float = 16.0
    bright_pixel_luminance: float = 240.0
    dark_frame_pixel_ratio: float = 0.98
    bright_frame_pixel_ratio: float = 0.98
    near_static_mean_abs_difference: float = 1.0
    min_blur_score: float | None = None
    max_same_reference_run: int | None = None
    max_near_static_run: int | None = None
    anomaly_policy: Literal["warning", "exclude"] = "warning"

    def __post_init__(self) -> None:
        if not 0 <= self.dark_pixel_luminance <= 255:
            raise ValueError("dark_pixel_luminance must be in [0, 255]")
        if not 0 <= self.bright_pixel_luminance <= 255:
            raise ValueError("bright_pixel_luminance must be in [0, 255]")
        if not 0 <= self.dark_frame_pixel_ratio <= 1:
            raise ValueError("dark_frame_pixel_ratio must be in [0, 1]")
        if not 0 <= self.bright_frame_pixel_ratio <= 1:
            raise ValueError("bright_frame_pixel_ratio must be in [0, 1]")
        if self.near_static_mean_abs_difference < 0:
            raise ValueError("near-static threshold must be >= 0")
        if self.min_blur_score is not None and self.min_blur_score < 0:
            raise ValueError("min_blur_score must be >= 0 or None")
        for name in ("max_same_reference_run", "max_near_static_run"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1 or None")
        if self.anomaly_policy not in {"warning", "exclude"}:
            raise ValueError("anomaly_policy must be 'warning' or 'exclude'")


@dataclass(frozen=True)
class VisualEvaluation:
    report: dict[str, object]
    keep_mask: np.ndarray
    warnings: tuple[str, ...]


def evaluate_visual(
    episode: CuratedEpisode,
    config: VisualQualityConfig,
) -> VisualEvaluation:
    """Evaluate required camera observations without changing any JPEG."""

    role_fields = {
        "head": "head_rgb_frame_index",
        "right_wrist": "right_wrist_rgb_frame_index",
    }
    keep = np.ones(episode.transition_count, dtype=bool)
    warnings: list[str] = []
    role_reports: dict[str, object] = {}

    for role, field in role_fields.items():
        references = np.asarray(episode.trajectory[field], dtype=np.int64)
        role_report, invalid_frames, role_warnings, exclude_for_anomaly = (
            _evaluate_role(episode, role, references, config)
        )
        role_reports[role] = role_report
        warnings.extend(role_warnings)
        if invalid_frames:
            keep &= ~np.isin(references, list(invalid_frames))
        if exclude_for_anomaly:
            keep[:] = False

    keep.setflags(write=False)
    return VisualEvaluation(
        report=role_reports,
        keep_mask=keep,
        warnings=tuple(warnings),
    )


def _evaluate_role(
    episode: CuratedEpisode,
    role: str,
    references: np.ndarray,
    config: VisualQualityConfig,
) -> tuple[dict[str, object], set[int], list[str], bool]:
    unique_indices = sorted({int(value) for value in references.tolist()})
    records: dict[int, dict[str, object]] = {}
    invalid_frames: set[int] = set()
    resolution_counts: Counter[tuple[int, int]] = Counter()

    for index in unique_indices:
        path = episode.media.rgb_path(role, index)
        record = _frame_metrics(path, config)
        records[index] = record
        if not bool(record["integrity_valid"]):
            invalid_frames.add(index)
        elif record["width"] is not None and record["height"] is not None:
            resolution_counts[(int(record["width"]), int(record["height"]))] += 1

    expected_resolution: tuple[int, int] | None = None
    resolution_inconsistencies = 0
    if resolution_counts:
        expected_resolution = min(
            resolution_counts,
            key=lambda item: (-resolution_counts[item], item),
        )
        for index, record in records.items():
            resolution = (record["width"], record["height"])
            if record["integrity_valid"] and resolution != expected_resolution:
                record["integrity_valid"] = False
                reasons = list(record["integrity_errors"])
                reasons.append("UNEXPECTED_RESOLUTION")
                record["integrity_errors"] = reasons
                invalid_frames.add(index)
                resolution_inconsistencies += 1

    decoded = [record for record in records.values() if record["decode_valid"]]
    valid_visual = [record for record in records.values() if record["integrity_valid"]]
    luminance_means = [float(record["mean_luminance"]) for record in valid_visual]
    luminance_stds = [float(record["luminance_std"]) for record in valid_visual]
    dark_ratios = [float(record["dark_pixel_ratio"]) for record in valid_visual]
    bright_ratios = [float(record["bright_pixel_ratio"]) for record in valid_visual]
    blur_scores = [float(record["blur_score"]) for record in valid_visual]

    sequence = _sequence_metrics(episode, role, references, config)
    warnings: list[str] = []
    anomaly = False
    if config.min_blur_score is not None:
        blurry = sum(score < config.min_blur_score for score in blur_scores)
        if blurry:
            warnings.append(f"{role}: {blurry} frames below configured blur score")
            anomaly = True
    if (
        config.max_same_reference_run is not None
        and sequence["longest_same_reference_run"] > config.max_same_reference_run
    ):
        warnings.append(f"{role}: same-reference run exceeds configured maximum")
        anomaly = True
    if (
        config.max_near_static_run is not None
        and sequence["longest_near_static_visual_run"] > config.max_near_static_run
    ):
        warnings.append(f"{role}: near-static run exceeds configured maximum")
        anomaly = True

    if invalid_frames:
        rgb_status = "EXCLUDE"
    elif anomaly:
        rgb_status = "WARNING" if config.anomaly_policy == "warning" else "EXCLUDE"
    else:
        rgb_status = "CLEAN"

    report: dict[str, object] = {
        "status": rgb_status,
        "reference_count": int(references.size),
        "unique_frame_count": len(unique_indices),
        "decoded_frame_count": len(decoded),
        "decode_failures": sum(
            not bool(record["decode_valid"]) for record in records.values()
        ),
        "hard_invalid_frame_count": len(invalid_frames),
        "resolution": list(expected_resolution) if expected_resolution else None,
        "resolution_inconsistencies": resolution_inconsistencies,
        "mean_luminance": summary(luminance_means),
        "luminance_std": summary(luminance_stds),
        "dark_pixel_ratio": summary(dark_ratios),
        "bright_pixel_ratio": summary(bright_ratios),
        "dark_frame_count": sum(
            ratio >= config.dark_frame_pixel_ratio for ratio in dark_ratios
        ),
        "bright_frame_count": sum(
            ratio >= config.bright_frame_pixel_ratio for ratio in bright_ratios
        ),
        "blur_score": summary(blur_scores),
        "duplicate_summary": sequence,
        "frames": {str(index): record for index, record in records.items()},
    }
    exclude = anomaly and config.anomaly_policy == "exclude"
    return report, invalid_frames, warnings, exclude


def _frame_metrics(
    path: Path,
    config: VisualQualityConfig,
) -> dict[str, object]:
    empty = {
        "decode_valid": False,
        "integrity_valid": False,
        "width": None,
        "height": None,
        "channel_count": None,
        "mean_luminance": None,
        "luminance_std": None,
        "dark_pixel_ratio": None,
        "bright_pixel_ratio": None,
        "blur_score": None,
        "integrity_errors": [],
    }
    if not path.is_file():
        empty["integrity_errors"] = ["MISSING_JPG"]
        return empty

    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            channels = len(image.getbands())
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, UnidentifiedImageError, ValueError):
        empty["integrity_errors"] = ["JPEG_DECODE_FAILURE"]
        return empty

    errors: list[str] = []
    if width <= 0 or height <= 0 or rgb.size == 0:
        errors.append("ZERO_SIZE_IMAGE")
    if channels != 3:
        errors.append("WRONG_CHANNEL_COUNT")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        errors.append("RGB_CONVERSION_FAILURE")

    luminance = (
        rgb[..., 0].astype(np.float32) * 0.2126
        + rgb[..., 1].astype(np.float32) * 0.7152
        + rgb[..., 2].astype(np.float32) * 0.0722
    )
    dark_ratio = float(np.mean(luminance <= config.dark_pixel_luminance))
    bright_ratio = float(np.mean(luminance >= config.bright_pixel_luminance))
    return {
        "decode_valid": True,
        "integrity_valid": not errors,
        "width": int(width),
        "height": int(height),
        "channel_count": int(channels),
        "mean_luminance": float(np.mean(luminance)),
        "luminance_std": float(np.std(luminance)),
        "dark_pixel_ratio": dark_ratio,
        "bright_pixel_ratio": bright_ratio,
        "blur_score": _blur_score(luminance),
        "integrity_errors": errors,
    }


def _blur_score(luminance: np.ndarray) -> float:
    horizontal = np.diff(luminance, axis=1)
    vertical = np.diff(luminance, axis=0)
    components: list[float] = []
    if horizontal.size:
        components.append(float(np.mean(np.abs(horizontal))))
    if vertical.size:
        components.append(float(np.mean(np.abs(vertical))))
    return float(np.mean(components)) if components else 0.0


def _sequence_metrics(
    episode: CuratedEpisode,
    role: str,
    references: np.ndarray,
    config: VisualQualityConfig,
) -> dict[str, object]:
    refs = [int(value) for value in references.tolist()]
    longest_same = _longest_equal_run(refs)
    same_pairs = sum(left == right for left, right in pairwise(refs))
    differences: list[float] = []
    exact_pair_flags: list[bool] = []
    last_ref: int | None = None
    last_image: np.ndarray | None = None
    for frame_index in refs:
        if frame_index == last_ref and last_image is not None:
            current = last_image
        else:
            try:
                current = episode.media.read_rgb(role, frame_index)
            except (FileNotFoundError, OSError, ValueError):
                current = None
        if last_image is not None and current is not None:
            if last_image.shape == current.shape:
                exact_pair_flags.append(bool(np.array_equal(current, last_image)))
                difference = float(
                    np.mean(
                        np.abs(
                            current.astype(np.float32) - last_image.astype(np.float32)
                        )
                    )
                )
                differences.append(difference)
            else:
                exact_pair_flags.append(False)
                differences.append(float("nan"))
        last_ref = frame_index
        last_image = current

    near_static_pairs = [
        bool(np.isfinite(value) and value <= config.near_static_mean_abs_difference)
        for value in differences
    ]
    exact_pairs = sum(exact_pair_flags)
    return {
        "same_reference_pair_count": int(same_pairs),
        "longest_same_reference_run": int(longest_same),
        "exact_duplicate_visual_pair_count": int(exact_pairs),
        "longest_exact_duplicate_visual_run": int(
            _longest_true_pair_run(exact_pair_flags)
        ),
        "frame_difference_mean_abs": summary(differences),
        "near_static_pair_count": int(sum(near_static_pairs)),
        "longest_near_static_visual_run": int(
            _longest_true_pair_run(near_static_pairs)
        ),
    }


def _longest_equal_run(values: list[object]) -> int:
    if not values:
        return 0
    longest = 1
    current = 1
    for left, right in pairwise(values):
        current = current + 1 if left is not None and left == right else 1
        longest = max(longest, current)
    return longest


def _longest_true_pair_run(pairs: list[bool]) -> int:
    longest = 1 if pairs else 0
    current = 1
    for value in pairs:
        current = current + 1 if value else 1
        longest = max(longest, current)
    return longest
