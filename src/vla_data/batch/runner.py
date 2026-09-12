"""Production dataset runners that delegate every episode to frozen processors."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from vla_data.batch.discovery import (
    CuratedEpisodeInput,
    DiscoveryIssue,
    DiscoveryResult,
    RawEpisodeInput,
    canonical_episode_id,
    discover_curated_episodes,
    discover_raw_episodes,
)
from vla_data.batch.status import DatasetRunResult, EpisodeResult
from vla_data.batch.summary import build_summary, load_summary, write_summary
from vla_data.cleaning.build_curated_v1 import build_curated_v1
from vla_data.contracts.curated_v1 import CURATED_SCHEMA_NAME, CURATED_SCHEMA_VERSION
from vla_data.io.curated_episode import CuratedEpisode
from vla_data.quality.evaluator import evaluate_episode
from vla_data.quality.report import (
    QUALITY_SCHEMA_NAME,
    QUALITY_SCHEMA_VERSION,
    QUALITY_STATUSES,
)
from vla_data.validation.curated_v1 import validate_curated_episode


class BatchConfigurationError(ValueError):
    """Invalid global input or runner configuration."""


def build_curated_dataset(
    input_root: str | Path,
    output_root: str | Path,
    *,
    episode: str | None = None,
    workers: int = 1,
    force: bool = False,
    dry_run: bool = False,
    expert_exclude: Iterable[str] = (),
) -> DatasetRunResult:
    """Discover RAW episodes and invoke the existing D2 builder independently."""

    started = time.monotonic()
    _validate_workers(workers)
    excluded = {canonical_episode_id(value) for value in expert_exclude}
    discovery = discover_raw_episodes(input_root, episode=episode)
    root = Path(output_root).resolve()
    if not dry_run:
        _prepare_output_root(root)

    issue_results = _raw_issue_results(discovery.issues, root)

    def process(item: RawEpisodeInput) -> EpisodeResult:
        target = root / item.episode_id
        if not force and _curated_is_complete(target):
            count = CuratedEpisode.load(target).transition_count
            return EpisodeResult(
                item.episode_id,
                "SKIPPED",
                "build",
                str(item.raw_path),
                str(target),
                "existing Curated v1 output is complete and valid",
                transitions_total=count,
                transitions_valid=count,
            )
        if dry_run:
            return EpisodeResult(
                item.episode_id,
                "WOULD_PROCESS",
                "build",
                str(item.raw_path),
                str(target),
                "WOULD_PROCESS",
            )
        temporary_root = Path(
            tempfile.mkdtemp(prefix=f".{item.episode_id}.build-", dir=root)
        )
        try:
            status = (
                "EXCLUDE_FROM_EXPERT_TRAINING"
                if item.episode_id in excluded
                else "REVIEW_REQUIRED"
            )
            built = build_curated_v1(
                item.raw_path,
                temporary_root,
                media_root=item.media_path,
                expert_training_status=status,
            )
            validation = validate_curated_episode(CuratedEpisode.load(built))
            if not validation.passed:
                raise ValueError("; ".join(validation.errors))
            _publish_directory(built, target)
            return EpisodeResult(
                item.episode_id,
                "SUCCESS",
                "build",
                str(item.raw_path),
                str(target),
                transitions_total=validation.transition_count,
                transitions_valid=validation.transition_count,
            )
        except Exception as exc:  # noqa: BLE001 - isolate one bad episode
            return _failed(item.episode_id, "build", item.raw_path, target, exc)
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)

    processed = _dispatch(discovery.episodes, process, workers)
    results = tuple(
        sorted((*processed, *issue_results), key=lambda item: item.episode_id)
    )
    summary = build_summary("build", results, discovery.issues)
    summary_path = None
    if not dry_run:
        summary_path = write_summary(root / "dataset_build_summary.json", summary)
    return DatasetRunResult(
        "build", results, summary, summary_path, time.monotonic() - started
    )


def validate_curated_dataset(
    input_root: str | Path,
    output_root: str | Path | None = None,
    *,
    episode: str | None = None,
    workers: int = 1,
    force: bool = False,
    dry_run: bool = False,
    planned_episodes: Iterable[str] = (),
) -> DatasetRunResult:
    """Independently validate each Curated episode and write a compact summary."""

    started = time.monotonic()
    _validate_workers(workers)
    discovery = _curated_discovery(
        input_root, episode=episode, dry_run=dry_run, planned_episodes=planned_episodes
    )
    root = Path(output_root if output_root is not None else input_root).resolve()
    summary_file = root / "dataset_validation_summary.json"
    previous = load_summary(summary_file, "vla_dataset_validation_summary")
    previous_results = _previous_results(previous)
    if not dry_run:
        _prepare_output_root(root)

    def process(item: CuratedEpisodeInput) -> EpisodeResult:
        prior = previous_results.get(item.episode_id)
        if (
            not force
            and prior is not None
            and prior.get("status") in {"SUCCESS", "WARNING", "SKIPPED"}
            and _curated_is_complete(item.episode_path)
        ):
            total = int(prior.get("transitions_total", 0))
            valid = int(prior.get("transitions_valid", total))
            return EpisodeResult(
                item.episode_id,
                "SKIPPED",
                "validation",
                str(item.episode_path),
                str(summary_file),
                "previous validation completion is current",
                outcome=str(prior.get("outcome", "PASS")),
                transitions_total=total,
                transitions_valid=valid,
            )
        if dry_run:
            return EpisodeResult(
                item.episode_id,
                "WOULD_PROCESS",
                "validation",
                str(item.episode_path),
                str(summary_file),
                "WOULD_PROCESS",
            )
        try:
            loaded = CuratedEpisode.load(item.episode_path)
            report = validate_curated_episode(loaded)
            if not report.passed:
                return EpisodeResult(
                    item.episode_id,
                    "FAILED",
                    "validation",
                    str(item.episode_path),
                    str(summary_file),
                    "; ".join(report.errors),
                    "CuratedValidationError",
                    outcome=report.quality,
                    transitions_total=report.transition_count,
                )
            status = "WARNING" if report.quality == "WARN" else "SUCCESS"
            return EpisodeResult(
                item.episode_id,
                status,
                "validation",
                str(item.episode_path),
                str(summary_file),
                "; ".join(report.warnings) or None,
                outcome=report.quality,
                transitions_total=report.transition_count,
                transitions_valid=report.transition_count,
            )
        except Exception as exc:  # noqa: BLE001 - isolate one bad episode
            return _failed(
                item.episode_id,
                "validation",
                item.episode_path,
                summary_file,
                exc,
            )

    processed = _dispatch(discovery.episodes, process, workers)
    issue_results = _curated_issue_results(discovery.issues, "validation", summary_file)
    results = tuple(
        sorted((*processed, *issue_results), key=lambda item: item.episode_id)
    )
    summary = build_summary("validation", results, discovery.issues)
    summary_path = None if dry_run else write_summary(summary_file, summary)
    return DatasetRunResult(
        "validation", results, summary, summary_path, time.monotonic() - started
    )


def quality_dataset(
    input_root: str | Path,
    output_root: str | Path,
    *,
    episode: str | None = None,
    workers: int = 1,
    force: bool = False,
    dry_run: bool = False,
    planned_episodes: Iterable[str] = (),
) -> DatasetRunResult:
    """Invoke the existing D3 evaluator once per isolated Curated episode."""

    started = time.monotonic()
    _validate_workers(workers)
    discovery = _curated_discovery(
        input_root, episode=episode, dry_run=dry_run, planned_episodes=planned_episodes
    )
    root = Path(output_root).resolve()
    if not dry_run:
        _prepare_output_root(root)

    def process(item: CuratedEpisodeInput) -> EpisodeResult:
        target = root / item.episode_id
        completed = _quality_completion(target, item.episode_id)
        if (
            not force
            and completed is not None
            and _curated_is_complete(item.episode_path)
        ):
            outcome, count = completed
            return EpisodeResult(
                item.episode_id,
                "SKIPPED",
                "quality",
                str(item.episode_path),
                str(target),
                "existing D3 quality artifacts are complete and valid",
                outcome=outcome,
                transitions_total=count,
                transitions_valid=count,
            )
        if dry_run:
            return EpisodeResult(
                item.episode_id,
                "WOULD_PROCESS",
                "quality",
                str(item.episode_path),
                str(target),
                "WOULD_PROCESS",
            )
        temporary_root = Path(
            tempfile.mkdtemp(prefix=f".{item.episode_id}.quality-", dir=root)
        )
        temporary_output = temporary_root / item.episode_id
        try:
            result = evaluate_episode(item.episode_path, temporary_output)
            _publish_directory(temporary_output, target)
            return EpisodeResult(
                item.episode_id,
                result.status,
                "quality",
                str(item.episode_path),
                str(target),
                outcome=result.status,
                transitions_total=result.transition_count,
                transitions_valid=result.clean_transition_count,
            )
        except Exception as exc:  # noqa: BLE001 - isolate one bad episode
            return _failed(item.episode_id, "quality", item.episode_path, target, exc)
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)

    processed = _dispatch(discovery.episodes, process, workers)
    issue_results = _curated_issue_results(discovery.issues, "quality", root)
    results = tuple(
        sorted((*processed, *issue_results), key=lambda item: item.episode_id)
    )
    summary = build_summary("quality", results, discovery.issues)
    summary_path = None
    if not dry_run:
        summary_path = write_summary(root / "dataset_quality_summary.json", summary)
    return DatasetRunResult(
        "quality", results, summary, summary_path, time.monotonic() - started
    )


def run_pipeline(
    input_root: str | Path,
    work_root: str | Path,
    *,
    stages: Iterable[str] = ("curated", "validate", "quality"),
    episode: str | None = None,
    workers: int = 1,
    force: bool = False,
    dry_run: bool = False,
    expert_exclude: Iterable[str] = (),
) -> dict[str, DatasetRunResult]:
    """Convenience orchestration only; stage runners remain independently usable."""

    selected = tuple(stages)
    allowed = ("curated", "validate", "quality")
    if not selected or len(set(selected)) != len(selected):
        raise BatchConfigurationError("stages must be a non-empty unique sequence")
    if any(stage not in allowed for stage in selected):
        raise BatchConfigurationError(f"stages must be selected from {allowed}")
    if tuple(sorted(selected, key=allowed.index)) != selected:
        raise BatchConfigurationError(
            "stages must follow curated,validate,quality order"
        )
    root = Path(work_root).resolve()
    results: dict[str, DatasetRunResult] = {}
    if "curated" in selected:
        results["curated"] = build_curated_dataset(
            input_root,
            root / "curated",
            episode=episode,
            workers=workers,
            force=force,
            dry_run=dry_run,
            expert_exclude=expert_exclude,
        )
    curated_root = root / "curated"
    # A dry-run build stage materializes nothing, so downstream stages would
    # otherwise fail discovering a Curated root that does not exist yet.  Project
    # the episodes the build stage plans to publish instead.
    planned_episodes: tuple[str, ...] = ()
    if dry_run and not curated_root.is_dir() and "curated" in results:
        planned_episodes = tuple(
            item.episode_id
            for item in results["curated"].results
            if item.status in {"WOULD_PROCESS", "SKIPPED"}
        )
    if "validate" in selected:
        results["validate"] = validate_curated_dataset(
            curated_root,
            root / "validation",
            episode=episode,
            workers=workers,
            force=force,
            dry_run=dry_run,
            planned_episodes=planned_episodes,
        )
    if "quality" in selected:
        results["quality"] = quality_dataset(
            curated_root,
            root / "quality",
            episode=episode,
            workers=workers,
            force=force,
            dry_run=dry_run,
            planned_episodes=planned_episodes,
        )
    return results


def _dispatch[ItemT](
    items: Iterable[ItemT],
    processor: Callable[[ItemT], EpisodeResult],
    workers: int,
) -> tuple[EpisodeResult, ...]:
    values = tuple(items)
    if workers == 1:
        return tuple(processor(item) for item in values)
    completed: list[EpisodeResult] = []
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="vla-episode"
    ) as pool:
        future_items = {pool.submit(processor, item): item for item in values}
        for future in as_completed(future_items):
            completed.append(future.result())
    return tuple(sorted(completed, key=lambda item: item.episode_id))


def _curated_discovery(
    input_root: str | Path,
    *,
    episode: str | None,
    dry_run: bool,
    planned_episodes: Iterable[str],
) -> DiscoveryResult:
    """Discover Curated episodes, or project planned ones during a dry run."""

    if dry_run and planned_episodes and not Path(input_root).is_dir():
        selected = canonical_episode_id(episode) if episode is not None else None
        identifiers = sorted(
            {canonical_episode_id(value) for value in planned_episodes}
        )
        root = Path(input_root)
        return DiscoveryResult(
            tuple(
                CuratedEpisodeInput(
                    episode_id=identifier,
                    numeric_id=int(identifier.removeprefix("episode_")),
                    episode_path=root / identifier,
                )
                for identifier in identifiers
                if selected is None or identifier == selected
            ),
            (),
            (),
        )
    return discover_curated_episodes(input_root, episode=episode)


def _validate_workers(workers: int) -> None:
    if isinstance(workers, bool) or int(workers) != workers or workers < 1:
        raise BatchConfigurationError("workers must be an integer >= 1")


def _prepare_output_root(root: Path) -> None:
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BatchConfigurationError(
            f"cannot create output root {root}: {exc}"
        ) from exc
    if not root.is_dir() or not os.access(root, os.W_OK):
        raise BatchConfigurationError(f"output root is not writable: {root}")


def _publish_directory(source: Path, target: Path) -> None:
    backup: Path | None = None
    if target.exists():
        backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
        os.replace(target, backup)
    try:
        os.replace(source, target)
    except BaseException:
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    else:
        if backup is not None:
            shutil.rmtree(backup)


def _curated_has_current_schema(path: Path) -> bool:
    try:
        episode = CuratedEpisode.load(path)
    except (OSError, ValueError, TypeError, KeyError):
        return False
    return (
        episode.metadata.get("schema_name") == CURATED_SCHEMA_NAME
        and episode.metadata.get("schema_version") == CURATED_SCHEMA_VERSION
        and (path / "cleaning_report.json").is_file()
    )


def _curated_is_complete(path: Path) -> bool:
    if not _curated_has_current_schema(path):
        return False
    try:
        return validate_curated_episode(CuratedEpisode.load(path)).passed
    except (OSError, ValueError, TypeError, KeyError):
        return False


def _quality_completion(path: Path, episode_id: str) -> tuple[str, int] | None:
    report_path = path / "quality_report.json"
    mask_path = path / "quality_mask.npy"
    if not report_path.is_file() or not mask_path.is_file():
        return None
    try:
        with report_path.open(encoding="utf-8") as stream:
            report = json.load(stream)
        mask = np.load(mask_path, allow_pickle=False)
        count = int(report["transition_count"])
    except (OSError, ValueError, TypeError, KeyError):
        return None
    status = str(report.get("status"))
    if (
        report.get("schema_name") != QUALITY_SCHEMA_NAME
        or report.get("schema_version") != QUALITY_SCHEMA_VERSION
        or report.get("episode_id") != episode_id
        or status not in QUALITY_STATUSES
        or mask.dtype != np.dtype(bool)
        or mask.shape != (count,)
    ):
        return None
    return status, count


def _previous_results(
    summary: dict[str, object] | None,
) -> dict[str, dict[str, object]]:
    if summary is None or not isinstance(summary.get("results"), list):
        return {}
    return {
        str(value["episode_id"]): value
        for value in summary["results"]
        if isinstance(value, dict) and "episode_id" in value
    }


def _failed(
    episode_id: str,
    stage: str,
    input_path: Path,
    output_path: Path,
    exc: Exception,
) -> EpisodeResult:
    return EpisodeResult(
        episode_id,
        "FAILED",
        stage,
        str(input_path),
        str(output_path),
        str(exc),
        type(exc).__name__,
        outcome="FAILED",
    )


def _raw_issue_results(
    issues: Iterable[DiscoveryIssue], output_root: Path
) -> tuple[EpisodeResult, ...]:
    return tuple(
        EpisodeResult(
            issue.episode_id,
            "FAILED",
            "build",
            str(issue.path),
            str(output_root / issue.episode_id),
            issue.message,
            issue.error_type,
            outcome="FAILED",
        )
        for issue in issues
        if issue.episode_id is not None
    )


def _curated_issue_results(
    issues: Iterable[DiscoveryIssue], stage: str, output_path: Path
) -> tuple[EpisodeResult, ...]:
    return tuple(
        EpisodeResult(
            issue.episode_id,
            "FAILED",
            stage,
            str(issue.path),
            str(output_path),
            issue.message,
            issue.error_type,
            outcome="FAILED",
        )
        for issue in issues
        if issue.episode_id is not None
    )
