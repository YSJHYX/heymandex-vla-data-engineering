"""Episode-level annotation pipeline: eligibility, keyframes, provider, artifacts."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vla_data.annotation.eligibility import (
    EligibilityError,
    evaluate_eligibility,
)
from vla_data.annotation.keyframes import (
    DEFAULT_MAX_TEMPORAL_POINTS,
    KeyframeSelection,
    keyframe_image_items,
    select_keyframes,
)
from vla_data.annotation.prompts import PROMPT_VERSION, build_prompt_items
from vla_data.annotation.provider import (
    AnnotationRequest,
    ProviderError,
    VLMProvider,
)
from vla_data.annotation.schema import (
    build_annotation,
    write_annotation,
)
from vla_data.io.curated_episode import CuratedEpisode

ANNOTATION_FILENAME = "annotation.json"
KEYFRAMES_FILENAME = "keyframes.json"

STATUS_SUCCESS = "SUCCESS"
STATUS_INELIGIBLE = "INELIGIBLE"
STATUS_FAILED = "FAILED"
STATUS_SKIPPED = "SKIPPED"
STATUS_STALE = "STALE"


@dataclass(frozen=True)
class EpisodeAnnotationResult:
    episode_id: str
    status: str
    quality_outcome: str | None
    instruction: str | None
    confidence: float | None
    provider_name: str | None
    model: str | None
    prompt_version: str | None
    attempt_count: int
    image_count: int
    message: str | None = None
    error_type: str | None = None
    http_status: int | None = None
    provider_error_code: str | None = None
    provider_error_message: str | None = None
    provider_request_id: str | None = None
    stale: bool = False
    review_status: str | None = None

    def as_dict(self) -> dict[str, object]:
        values = {
            "episode_id": self.episode_id,
            "status": self.status,
            "quality_outcome": self.quality_outcome,
            "instruction": self.instruction,
            "confidence": self.confidence,
            "provider": self.provider_name,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "attempt_count": self.attempt_count,
            "image_count": self.image_count,
            "stale": self.stale,
            "review_status": self.review_status,
            "message": self.message,
            "error_type": self.error_type,
            "http_status": self.http_status,
            "provider_error_code": self.provider_error_code,
            "provider_error_message": self.provider_error_message,
            "provider_request_id": self.provider_request_id,
        }
        return {key: value for key, value in values.items() if value is not None}


def annotate_episode(
    curated_episode_dir: str | Path,
    *,
    quality_root: str | Path,
    output_root: str | Path,
    provider: VLMProvider,
    max_temporal_points: int = DEFAULT_MAX_TEMPORAL_POINTS,
) -> EpisodeAnnotationResult:
    """Annotate exactly one episode atomically; eligibility gates the provider."""

    episode_dir = Path(curated_episode_dir)
    episode_id = episode_dir.name
    try:
        episode = CuratedEpisode.load(episode_dir)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return _failed(episode_id, quality_outcome=None, error=exc)

    try:
        decision = evaluate_eligibility(
            episode_id,
            transition_count=episode.transition_count,
            quality_report_path=Path(quality_root) / episode_id / "quality_report.json",
            quality_mask_path=Path(quality_root) / episode_id / "quality_mask.npy",
        )
    except EligibilityError as exc:
        return _failed(episode_id, None, exc)

    if not decision.eligible:
        return EpisodeAnnotationResult(
            episode_id=episode_id,
            status=STATUS_INELIGIBLE,
            quality_outcome=decision.quality_outcome,
            instruction=None,
            confidence=None,
            provider_name=None,
            model=None,
            prompt_version=None,
            attempt_count=0,
            image_count=0,
            message=decision.reason,
        )

    try:
        mask = np.load(
            Path(quality_root) / episode_id / "quality_mask.npy", allow_pickle=False
        )
        selection = select_keyframes(
            episode, clean_mask=mask, max_temporal_points=max_temporal_points
        )
        request_items = build_prompt_items(
            keyframe_image_items(selection), prompt_version=PROMPT_VERSION
        )
        request = AnnotationRequest(
            episode_id=episode_id,
            prompt_version=PROMPT_VERSION,
            items=request_items,
        )
        model_annotation = provider.annotate(request)
    except ProviderError as exc:
        return _failed(episode_id, decision.quality_outcome, exc)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return _failed(episode_id, decision.quality_outcome, exc)

    annotation = build_annotation(
        episode_id=episode_id,
        model_annotation=model_annotation.model_block(),
        quality_outcome=str(decision.quality_outcome),
    )
    try:
        _publish_artifacts(Path(output_root), episode_id, annotation, selection)
    except OSError as exc:
        return _failed(episode_id, decision.quality_outcome, exc)

    return EpisodeAnnotationResult(
        episode_id=episode_id,
        status=STATUS_SUCCESS,
        quality_outcome=decision.quality_outcome,
        instruction=model_annotation.instruction,
        confidence=model_annotation.confidence,
        provider_name=model_annotation.provider,
        model=model_annotation.model,
        prompt_version=model_annotation.prompt_version,
        attempt_count=model_annotation.attempt_count,
        image_count=request.image_count,
        review_status="AUTO_LABELED",
    )


def _publish_artifacts(
    output_root: Path,
    episode_id: str,
    annotation: dict[str, object],
    selection: KeyframeSelection,
) -> None:
    """Stage both artifacts, then atomically replace the published directory."""

    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{episode_id}.annotate-", dir=output_root))
    backup: Path | None = None
    try:
        staged_episode = staging / episode_id
        staged_episode.mkdir(parents=True)
        write_annotation(staged_episode / ANNOTATION_FILENAME, annotation)
        with (staged_episode / KEYFRAMES_FILENAME).open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(selection.as_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")
        target = output_root / episode_id
        if target.exists():
            backup = staging / "backup"
            target.rename(backup)
        staged_episode.rename(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _failed(
    episode_id: str, quality_outcome: str | None, error: Exception
) -> EpisodeAnnotationResult:
    from vla_data.annotation.provider import ProviderError

    details = error.details() if isinstance(error, ProviderError) else {}
    return EpisodeAnnotationResult(
        episode_id=episode_id,
        status=STATUS_FAILED,
        quality_outcome=quality_outcome,
        instruction=None,
        confidence=None,
        provider_name=None,
        model=None,
        prompt_version=None,
        attempt_count=0,
        image_count=0,
        message=str(error),
        error_type=type(error).__name__,
        http_status=details.get("http_status"),
        provider_error_code=details.get("provider_error_code"),
        provider_error_message=details.get("provider_error_message"),
        provider_request_id=details.get("request_id"),
    )
