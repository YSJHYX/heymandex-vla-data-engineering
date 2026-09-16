"""Annotation-only storyboard rendering for single-image MCP fallback.

A storyboard is an annotation TRANSPORT representation, never training
data: it never enters Curated media, D3 inputs, LeRobot exports, or HF.
Source frames are only read; scaling happens on the storyboard copy.

The renderer understands ONLY typed MultiViewObservation inputs — never
trajectory.npz keys, camera-role aliases, or media-directory conventions
(those mappings happen upstream). Each row is strictly head(i) paired
with right_wrist(i) from the same curated observation, in the given
chronological order.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from vla_data.annotation.provider import (
    CAMERA_ROLE_HEAD,
    CAMERA_ROLE_RIGHT_WRIST,
    ProviderError,
)

MARGIN_WIDTH = 150
ROW_GAP = 6
BACKGROUND = (16, 16, 16)
MARGIN_TEXT = (232, 232, 232)

ANNOTATION_CAMERA_VIEW_MISSING = "ANNOTATION_CAMERA_VIEW_MISSING"
STORYBOARD_BUILD_ERROR = "STORYBOARD_BUILD_ERROR"
STORYBOARD_SIZE_LIMIT = "STORYBOARD_SIZE_LIMIT"

DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 360
DEFAULT_JPEG_QUALITY = 90
JPEG_QUALITY_FLOOR = 60
JPEG_QUALITY_CANDIDATES = (88, 84, 80, 76, 72, 68, 64, JPEG_QUALITY_FLOOR)
RESOLUTION_SCALE_FLOOR = 0.70
RESOLUTION_SCALE_CANDIDATES = (0.90, 0.80, RESOLUTION_SCALE_FLOOR)


@dataclass(frozen=True)
class MultiViewObservation:
    """One curated transition's dual-view observation (typed, pre-validated)."""

    curated_index: int
    timestamp_ns: int
    head_path: Path
    right_wrist_path: Path


@dataclass(frozen=True)
class StoryboardArtifact:
    """A fitted annotation-only JPEG plus auditable transport facts."""

    path: Path
    within_budget: bool
    transport_metadata: dict[str, Any]


def observations_from_request(
    request,  # AnnotationRequest
    *,
    image_items: list | None = None,
) -> list[MultiViewObservation]:
    """Group provider-neutral ImageItems into typed dual-view observations.

    Fails closed with ANNOTATION_CAMERA_VIEW_MISSING when any temporal point
    lacks exactly the canonical head + right_wrist pair — never substitutes,
    duplicates, or reorders views.
    """

    items = (
        image_items
        if image_items is not None
        else [item for item in request.items if hasattr(item, "camera")]
    )
    grouped: dict[int, dict[str, Path]] = {}
    for item in items:
        grouped.setdefault(item.temporal_point, {})[item.camera] = Path(
            item.source_path
        )
    observations: list[MultiViewObservation] = []
    for point in sorted(grouped):
        views = grouped[point]
        missing = [
            role
            for role in (CAMERA_ROLE_HEAD, CAMERA_ROLE_RIGHT_WRIST)
            if role not in views
        ]
        if missing:
            raise ProviderError(
                ANNOTATION_CAMERA_VIEW_MISSING,
                f"curated_index={point} lacks required camera view(s) "
                f"{missing}; present roles: {sorted(views)}; dual-view "
                "annotation contract forbids silent substitution",
            )
        observations.append(
            MultiViewObservation(
                curated_index=point,
                timestamp_ns=0,
                head_path=views[CAMERA_ROLE_HEAD],
                right_wrist_path=views[CAMERA_ROLE_RIGHT_WRIST],
            )
        )
    return observations


def render_storyboard(
    observations: list[MultiViewObservation],
    output_dir: Path,
    *,
    episode_id: str = "episode",
    max_width: int = DEFAULT_FRAME_WIDTH,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    scale: float = 1.0,
    storyboard_pass: str | None = None,
    optimize: bool = False,
) -> Path:
    """Render ordered head|wrist rows with margin metadata to one JPEG."""

    if not observations:
        raise ProviderError(STORYBOARD_BUILD_ERROR, "no observations to render")
    if not 0 < scale <= 1.0:
        raise ProviderError(
            STORYBOARD_BUILD_ERROR, f"storyboard scale must be in (0, 1], got {scale}"
        )
    if not 1 <= jpeg_quality <= 95:
        raise ProviderError(
            STORYBOARD_BUILD_ERROR,
            f"storyboard JPEG quality must be in [1, 95], got {jpeg_quality}",
        )
    frame_width = max(1, round(max_width * scale))
    frame_height = max(1, round(DEFAULT_FRAME_HEIGHT * scale))
    canvas = Image.new(
        "RGB",
        (
            MARGIN_WIDTH + 2 * frame_width + 3 * ROW_GAP,
            len(observations) * (frame_height + ROW_GAP) + ROW_GAP,
        ),
        BACKGROUND,
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, observation in enumerate(observations):
        for label, path in (
            (CAMERA_ROLE_HEAD, observation.head_path),
            (CAMERA_ROLE_RIGHT_WRIST, observation.right_wrist_path),
        ):
            if not Path(path).is_file():
                raise ProviderError(
                    ANNOTATION_CAMERA_VIEW_MISSING,
                    f"curated_index={observation.curated_index} "
                    f"{label} source image is missing: {path}",
                )
        y = ROW_GAP + row_index * (frame_height + ROW_GAP)
        draw.text(
            (8, y + 8),
            f"obs {row_index + 1}/{len(observations)}\n"
            f"curated_index={observation.curated_index}",
            fill=MARGIN_TEXT,
        )
        head = _load_scaled(observation.head_path, frame_height, frame_width)
        wrist = _load_scaled(observation.right_wrist_path, frame_height, frame_width)
        canvas.paste(head, (MARGIN_WIDTH + ROW_GAP, y))
        canvas.paste(wrist, (MARGIN_WIDTH + frame_width + 2 * ROW_GAP, y))

    digest = hashlib.sha1(
        "|".join(
            f"{observation.curated_index}:{observation.head_path}:"
            f"{observation.right_wrist_path}"
            for observation in observations
        ).encode("utf-8")
    ).hexdigest()[:10]
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{storyboard_pass}_" if storyboard_pass else ""
    destination = output_dir / f"{prefix}storyboard_{episode_id}_{digest}.jpg"
    save_options: dict[str, Any] = {
        "format": "JPEG",
        "quality": jpeg_quality,
    }
    if optimize:
        # Explicit only on adaptive attempts. The initial/default path remains
        # byte-for-byte governed by the renderer's historical Pillow defaults.
        save_options.update(optimize=True, subsampling=2)
    canvas.save(destination, **save_options)
    return destination


def fit_storyboard_to_budget(
    observations: list[MultiViewObservation],
    output_dir: Path,
    *,
    target_max_bytes: int,
    episode_id: str = "episode",
    storyboard_pass: str = "storyboard",
    max_width: int = DEFAULT_FRAME_WIDTH,
    default_quality: int = DEFAULT_JPEG_QUALITY,
    quality_candidates: tuple[int, ...] = JPEG_QUALITY_CANDIDATES,
    scale_candidates: tuple[float, ...] = RESOLUTION_SCALE_CANDIDATES,
) -> StoryboardArtifact:
    """Fit one storyboard without changing its temporal evidence.

    The first render uses the existing renderer defaults. Only an over-budget
    artifact enters deterministic adaptive fitting: quality-only at full
    resolution, then the quality floor at bounded resolution fallbacks.
    Source images are opened read-only on every render.
    """

    if target_max_bytes <= 0:
        raise ProviderError(
            STORYBOARD_BUILD_ERROR,
            f"target_max_bytes must be positive, got {target_max_bytes}",
        )
    original_width, original_height = _canvas_dimensions(
        len(observations), max_width=max_width, scale=1.0
    )
    board = render_storyboard(
        observations,
        output_dir,
        episode_id=episode_id,
        max_width=max_width,
        jpeg_quality=default_quality,
        storyboard_pass=storyboard_pass,
    )
    initial_bytes = os.path.getsize(board)
    final_bytes = initial_bytes
    selected_quality = default_quality
    selected_scale = 1.0
    applied = False

    if initial_bytes > target_max_bytes:
        applied = True
        for quality in quality_candidates:
            if quality >= default_quality:
                continue
            board = render_storyboard(
                observations,
                output_dir,
                episode_id=episode_id,
                max_width=max_width,
                jpeg_quality=quality,
                scale=1.0,
                storyboard_pass=storyboard_pass,
                optimize=True,
            )
            final_bytes = os.path.getsize(board)
            selected_quality = quality
            if final_bytes <= target_max_bytes:
                break
        else:
            quality_floor = min(quality_candidates, default=JPEG_QUALITY_FLOOR)
            for fallback_scale in scale_candidates:
                board = render_storyboard(
                    observations,
                    output_dir,
                    episode_id=episode_id,
                    max_width=max_width,
                    jpeg_quality=quality_floor,
                    scale=fallback_scale,
                    storyboard_pass=storyboard_pass,
                    optimize=True,
                )
                final_bytes = os.path.getsize(board)
                selected_quality = quality_floor
                selected_scale = fallback_scale
                if final_bytes <= target_max_bytes:
                    break

    final_width, final_height = _canvas_dimensions(
        len(observations), max_width=max_width, scale=selected_scale
    )
    metadata = {
        "storyboard_pass": storyboard_pass,
        "storyboard_observation_count": len(observations),
        "storyboard_original_width": original_width,
        "storyboard_original_height": original_height,
        "storyboard_initial_bytes": initial_bytes,
        "storyboard_final_bytes": final_bytes,
        "storyboard_target_max_bytes": target_max_bytes,
        "storyboard_jpeg_quality": selected_quality,
        "storyboard_scale": selected_scale,
        "storyboard_final_width": final_width,
        "storyboard_final_height": final_height,
        "storyboard_adaptive_compression_applied": applied,
    }
    return StoryboardArtifact(
        path=board,
        within_budget=final_bytes <= target_max_bytes,
        transport_metadata=metadata,
    )


def _canvas_dimensions(
    observation_count: int, *, max_width: int, scale: float
) -> tuple[int, int]:
    frame_width = max(1, round(max_width * scale))
    frame_height = max(1, round(DEFAULT_FRAME_HEIGHT * scale))
    return (
        MARGIN_WIDTH + 2 * frame_width + 3 * ROW_GAP,
        observation_count * (frame_height + ROW_GAP) + ROW_GAP,
    )


def _load_scaled(path: Path, height: int, width: int) -> Image.Image:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return rgb.resize((width, height), Image.Resampling.LANCZOS)
