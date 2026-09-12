"""Dataset-level batch annotation with resume, bounded concurrency, summaries."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

from vla_data.annotation.eligibility import EligibilityError, evaluate_eligibility
from vla_data.annotation.keyframes import DEFAULT_MAX_TEMPORAL_POINTS
from vla_data.annotation.pipeline import (
    ANNOTATION_FILENAME,
    KEYFRAMES_FILENAME,
    STATUS_FAILED,
    STATUS_INELIGIBLE,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    EpisodeAnnotationResult,
    annotate_episode,
)
from vla_data.annotation.prompts import PROMPT_VERSION
from vla_data.annotation.provider import VLMProvider
from vla_data.annotation.schema import REVIEW_STATES, load_annotation
from vla_data.batch.discovery import (
    CuratedEpisodeInput,
    DiscoveryError,
    DiscoveryIssue,
    discover_curated_episodes,
)
from vla_data.io.curated_episode import CuratedEpisode

SUMMARY_SCHEMA_NAME = "vla_dataset_annotation_summary"
SUMMARY_VERSION = 1
STATUS_WOULD_PROCESS = "WOULD_PROCESS"
STATUS_WOULD_SKIP = "WOULD_SKIP"
INVALID_RESPONSE_ERRORS = frozenset(
    {"INVALID_JSON", "INVALID_SCHEMA", "EMPTY_RESPONSE"}
)


class AnnotationConfigurationError(ValueError):
    """Invalid global batch-annotation configuration."""


@dataclass(frozen=True)
class DatasetAnnotationResult:
    results: tuple[EpisodeAnnotationResult, ...]
    summary: dict[str, object]
    summary_path: Path | None
    wall_time_s: float

    @property
    def failed_count(self) -> int:
        return sum(result.status == STATUS_FAILED for result in self.results)


def annotate_dataset(
    curated_root: str | Path,
    quality_root: str | Path,
    output_root: str | Path,
    provider_factory: Callable[[], VLMProvider],
    *,
    episode: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    concurrency: int = 1,
    max_temporal_points: int = DEFAULT_MAX_TEMPORAL_POINTS,
) -> DatasetAnnotationResult:
    """Annotate every eligible Curated episode; episodes stay atomic units."""

    started = time.monotonic()
    _validate_concurrency(concurrency)
    curated = Path(curated_root)
    if not curated.is_dir():
        raise AnnotationConfigurationError(
            f"curated root is not a directory: {curated}"
        )
    quality = Path(quality_root)
    if not quality.is_dir():
        raise AnnotationConfigurationError(
            f"quality root is not a directory: {quality}"
        )
    output = Path(output_root).resolve()
    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)
    provider = provider_factory()

    try:
        discovery = discover_curated_episodes(curated, episode=episode)
    except DiscoveryError as exc:
        raise AnnotationConfigurationError(str(exc)) from exc

    items = discovery.episodes
    if dry_run:
        outcomes = tuple(
            _plan_episode(item, quality, output, provider, force) for item in items
        )
    elif concurrency == 1:
        outcomes = tuple(
            _process_episode(
                item, quality, output, provider, force, max_temporal_points
            )
            for item in items
        )
    else:
        outcomes = _run_concurrent(
            items, quality, output, provider, concurrency, force, max_temporal_points
        )

    results = tuple(
        sorted(
            (*outcomes, *_issue_results(discovery.issues)),
            key=lambda result: result.episode_id,
        )
    )
    summary = build_annotation_summary(results)
    summary_path = None
    if not dry_run:
        summary_path = _write_summary(
            output / "dataset_annotation_summary.json", summary
        )
    return DatasetAnnotationResult(
        results, summary, summary_path, time.monotonic() - started
    )


def _process_episode(
    item: CuratedEpisodeInput,
    quality: Path,
    output: Path,
    provider: VLMProvider,
    force: bool,
    max_temporal_points: int,
) -> EpisodeAnnotationResult:
    annotation_path = output / item.episode_id / ANNOTATION_FILENAME
    stale_reason: str | None = None
    if annotation_path.is_file():
        if not force and _valid_annotation_exists(item, output, provider):
            return _skipped_result(item, output)
        stale_reason = (
            "forced re-annotation"
            if force
            else "previous annotation is invalid or does not match "
            "provider/model/prompt_version"
        )
    result = annotate_episode(
        item.episode_path,
        quality_root=quality,
        output_root=output,
        provider=provider,
        max_temporal_points=max_temporal_points,
    )
    if stale_reason and result.status == STATUS_SUCCESS:
        result = replace(result, stale=True, message=stale_reason)
    return result


def _skipped_result(item: CuratedEpisodeInput, output: Path) -> EpisodeAnnotationResult:
    annotation = load_annotation(output / item.episode_id / ANNOTATION_FILENAME)
    model_block = annotation["model_annotation"]
    return EpisodeAnnotationResult(
        episode_id=item.episode_id,
        status=STATUS_SKIPPED,
        quality_outcome=str(annotation.get("quality_outcome")),
        instruction=str(model_block["instruction"]),
        confidence=float(model_block["confidence"]),
        provider_name=str(model_block["provider"]),
        model=str(model_block["model"]),
        prompt_version=str(model_block["prompt_version"]),
        attempt_count=0,
        image_count=0,
        message="valid matching annotation already exists",
        review_status=str(annotation["review"]["status"]),
    )


def _plan_episode(
    item: CuratedEpisodeInput,
    quality: Path,
    output: Path,
    provider: VLMProvider,
    force: bool,
) -> EpisodeAnnotationResult:
    """Dry-run projection: eligibility is real, the provider is never called."""

    try:
        episode = CuratedEpisode.load(item.episode_path)
        decision = evaluate_eligibility(
            item.episode_id,
            transition_count=episode.transition_count,
            quality_report_path=quality / item.episode_id / "quality_report.json",
            quality_mask_path=quality / item.episode_id / "quality_mask.npy",
        )
    except (OSError, ValueError, EligibilityError) as exc:
        return EpisodeAnnotationResult(
            episode_id=item.episode_id,
            status=STATUS_FAILED,
            quality_outcome=None,
            instruction=None,
            confidence=None,
            provider_name=None,
            model=None,
            prompt_version=None,
            attempt_count=0,
            image_count=0,
            message=str(exc),
            error_type=type(exc).__name__,
        )
    if not decision.eligible:
        return EpisodeAnnotationResult(
            episode_id=item.episode_id,
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
    if not force and _valid_annotation_exists(item, output, provider):
        return EpisodeAnnotationResult(
            episode_id=item.episode_id,
            status=STATUS_WOULD_SKIP,
            quality_outcome=decision.quality_outcome,
            instruction=None,
            confidence=None,
            provider_name=provider.provider_name,
            model=provider.model,
            prompt_version=PROMPT_VERSION,
            attempt_count=0,
            image_count=0,
            message="valid matching annotation would be skipped",
        )
    return EpisodeAnnotationResult(
        episode_id=item.episode_id,
        status=STATUS_WOULD_PROCESS,
        quality_outcome=decision.quality_outcome,
        instruction=None,
        confidence=None,
        provider_name=None,
        model=None,
        prompt_version=None,
        attempt_count=0,
        image_count=0,
        message="would call the annotation provider",
    )


def _run_concurrent(
    items: Iterable[CuratedEpisodeInput],
    quality: Path,
    output: Path,
    provider: VLMProvider,
    concurrency: int,
    force: bool,
    max_temporal_points: int,
) -> tuple[EpisodeAnnotationResult, ...]:
    completed: list[EpisodeAnnotationResult] = []
    with ThreadPoolExecutor(
        max_workers=concurrency, thread_name_prefix="vla-annotate"
    ) as pool:
        futures = [
            pool.submit(
                _process_episode,
                item,
                quality,
                output,
                provider,
                force,
                max_temporal_points,
            )
            for item in items
        ]
        for future in as_completed(futures):
            completed.append(future.result())
    return tuple(completed)


def _valid_annotation_exists(
    item: CuratedEpisodeInput, output: Path, provider: VLMProvider
) -> bool:
    episode_output = output / item.episode_id
    annotation_path = episode_output / ANNOTATION_FILENAME
    if not annotation_path.is_file():
        return False
    if not (episode_output / KEYFRAMES_FILENAME).is_file():
        return False
    try:
        annotation = load_annotation(annotation_path)
    except (OSError, ValueError):
        return False
    block = annotation["model_annotation"]
    return (
        block.get("provider") == provider.provider_name
        and block.get("model") == provider.model
        and block.get("prompt_version") == PROMPT_VERSION
    )


def build_annotation_summary(
    results: Iterable[EpisodeAnnotationResult],
) -> dict[str, object]:
    ordered = tuple(sorted(results, key=lambda item: item.episode_id))
    confidences = [
        result.confidence for result in ordered if result.confidence is not None
    ]
    review_counts = Counter(
        result.review_status for result in ordered if result.review_status is not None
    )
    invalid = sum(
        1
        for result in ordered
        if result.error_type in INVALID_RESPONSE_ERRORS
        or _invalid_category(result.message)
    )
    retries = sum(1 for result in ordered if result.attempt_count > 1)
    return {
        "schema_name": SUMMARY_SCHEMA_NAME,
        "schema_version": SUMMARY_VERSION,
        "episodes_discovered": len(ordered),
        "episodes_eligible": sum(
            result.status != STATUS_INELIGIBLE for result in ordered
        ),
        "episodes_ineligible": sum(
            result.status == STATUS_INELIGIBLE for result in ordered
        ),
        "episodes_processed": sum(
            result.status in {STATUS_SUCCESS, STATUS_FAILED} for result in ordered
        ),
        "episodes_skipped": sum(result.status == STATUS_SKIPPED for result in ordered),
        "episodes_failed": sum(result.status == STATUS_FAILED for result in ordered),
        "episodes_would_process": sum(
            result.status == STATUS_WOULD_PROCESS for result in ordered
        ),
        "episodes_would_skip": sum(
            result.status == STATUS_WOULD_SKIP for result in ordered
        ),
        "review_state_counts": {
            status: int(review_counts.get(status, 0)) for status in REVIEW_STATES
        },
        "mean_confidence": (
            round(sum(confidences) / len(confidences), 4) if confidences else None
        ),
        "invalid_response_count": invalid,
        "retry_count": retries,
        "results": [result.as_dict() for result in ordered],
    }


def _invalid_category(message: str | None) -> bool:
    return bool(message) and any(
        prefix in str(message) for prefix in INVALID_RESPONSE_ERRORS
    )


def _issue_results(
    issues: Iterable[DiscoveryIssue],
) -> tuple[EpisodeAnnotationResult, ...]:
    return tuple(
        EpisodeAnnotationResult(
            episode_id=issue.episode_id,
            status=STATUS_FAILED,
            quality_outcome=None,
            instruction=None,
            confidence=None,
            provider_name=None,
            model=None,
            prompt_version=None,
            attempt_count=0,
            image_count=0,
            message=issue.message,
            error_type=issue.error_type,
        )
        for issue in issues
        if issue.episode_id is not None
    )


def _validate_concurrency(concurrency: int) -> None:
    if (
        isinstance(concurrency, bool)
        or int(concurrency) != concurrency
        or concurrency < 1
    ):
        raise AnnotationConfigurationError("concurrency must be an integer >= 1")


def _write_summary(path: Path, summary: dict[str, object]) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
