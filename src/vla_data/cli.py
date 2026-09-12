"""Command-line interface for production episode-isolated dataset processing."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from vla_data.annotation.batch import (
    AnnotationConfigurationError,
    annotate_dataset,
)
from vla_data.annotation.glm_provider import (
    GLM46VFlashProvider,
    GLMProviderConfig,
    MissingAPIKeyError,
    resolve_api_key,
)
from vla_data.annotation.keyframes import DEFAULT_MAX_TEMPORAL_POINTS
from vla_data.batch.discovery import DiscoveryError
from vla_data.batch.runner import (
    BatchConfigurationError,
    build_curated_dataset,
    quality_dataset,
    run_pipeline,
    validate_curated_dataset,
)
from vla_data.batch.status import DatasetRunResult


def _common(parser: argparse.ArgumentParser, *, output: bool = True) -> None:
    parser.add_argument("--input-root", type=Path, required=True)
    if output:
        parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episode")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="vla-data",
        description=__doc__,
        epilog=(
            "Exit codes: 0=no episode failures; 1=one or more episode failures; "
            "2=global error."
        ),
    )
    commands = root.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-curated", help="RAW to Curated v1 batch")
    _common(build)
    build.add_argument(
        "--expert-exclude",
        action="append",
        default=[],
        metavar="EPISODE_ID",
        help="persist an explicit expert-training exclusion (repeatable)",
    )

    validate = commands.add_parser(
        "validate-curated", help="independently validate Curated v1 episodes"
    )
    _common(validate, output=False)
    validate.add_argument("--output-root", type=Path)

    quality = commands.add_parser("quality", help="run D3 quality per episode")
    _common(quality)

    run = commands.add_parser("run", help="orchestrate selected dataset stages")
    run.add_argument("--input-root", type=Path, required=True)
    run.add_argument("--work-root", type=Path, required=True)
    run.add_argument("--stages", default="curated,validate,quality")
    run.add_argument("--episode")
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--force", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--verbose", action="store_true")
    run.add_argument(
        "--expert-exclude",
        action="append",
        default=[],
        metavar="EPISODE_ID",
        help="persist an explicit expert-training exclusion (repeatable)",
    )

    annotate = commands.add_parser(
        "annotate", help="GLM model-first task annotation over Curated episodes"
    )
    annotate.add_argument("--curated-root", type=Path, required=True)
    annotate.add_argument("--quality-root", type=Path, required=True)
    annotate.add_argument("--output-root", type=Path, required=True)
    annotate.add_argument("--episode")
    annotate.add_argument("--force", action="store_true")
    annotate.add_argument("--dry-run", action="store_true")
    annotate.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="bounded provider concurrency (API requests in flight)",
    )
    annotate.add_argument(
        "--max-temporal-points",
        type=int,
        default=DEFAULT_MAX_TEMPORAL_POINTS,
    )
    annotate.add_argument("--verbose", action="store_true")
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "build-curated":
            result = build_curated_dataset(
                args.input_root,
                args.output_root,
                episode=args.episode,
                workers=args.workers,
                force=args.force,
                dry_run=args.dry_run,
                expert_exclude=args.expert_exclude,
            )
            _print_result(result, verbose=args.verbose)
            return 1 if result.failed_count else 0
        if args.command == "validate-curated":
            result = validate_curated_dataset(
                args.input_root,
                args.output_root,
                episode=args.episode,
                workers=args.workers,
                force=args.force,
                dry_run=args.dry_run,
            )
            _print_result(result, verbose=args.verbose)
            return 1 if result.failed_count else 0
        if args.command == "quality":
            result = quality_dataset(
                args.input_root,
                args.output_root,
                episode=args.episode,
                workers=args.workers,
                force=args.force,
                dry_run=args.dry_run,
            )
            _print_result(result, verbose=args.verbose)
            return 1 if result.failed_count else 0

        if args.command == "annotate":
            if not args.dry_run:
                try:
                    resolve_api_key()
                except MissingAPIKeyError as exc:
                    print(f"vla-data: {type(exc).__name__}: {exc}", file=sys.stderr)
                    return 2
            annotation_result = annotate_dataset(
                args.curated_root,
                args.quality_root,
                args.output_root,
                lambda: GLM46VFlashProvider(GLMProviderConfig()),
                episode=args.episode,
                force=args.force,
                dry_run=args.dry_run,
                concurrency=args.concurrency,
                max_temporal_points=args.max_temporal_points,
            )
            _print_annotation_result(annotation_result, verbose=args.verbose)
            return 1 if annotation_result.failed_count else 0

        stages = tuple(
            value.strip() for value in args.stages.split(",") if value.strip()
        )
        results = run_pipeline(
            args.input_root,
            args.work_root,
            stages=stages,
            episode=args.episode,
            workers=args.workers,
            force=args.force,
            dry_run=args.dry_run,
            expert_exclude=args.expert_exclude,
        )
        for result in results.values():
            _print_result(result, verbose=args.verbose)
        return 1 if any(result.failed_count for result in results.values()) else 0
    except (
        AnnotationConfigurationError,
        BatchConfigurationError,
        DiscoveryError,
        OSError,
        ValueError,
    ) as exc:
        print(f"vla-data: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def _print_result(result: DatasetRunResult, *, verbose: bool) -> None:
    total = len(result.results)
    for index, item in enumerate(result.results, start=1):
        print(f"[{index}/{total}] {item.episode_id}  {item.status}")
        if verbose and item.message:
            print(f"  {item.message}")
    print(f"Stage      {result.stage}")
    print(f"Total      {total}")
    print(f"Processed  {result.summary['episodes_processed']}")
    print(f"Skipped    {result.summary['episodes_skipped']}")
    print(f"Failed     {result.summary['episodes_failed']}")
    print(f"Wall time  {result.wall_time_s:.3f} s")


def _print_annotation_result(result, *, verbose: bool) -> None:
    total = len(result.results)
    for index, item in enumerate(result.results, start=1):
        detail = item.review_status or item.quality_outcome or ""
        if item.confidence is not None:
            detail = f"{detail} confidence={item.confidence}"
        print(f"[{index}/{total}] {item.episode_id}  {item.status}  {detail}".rstrip())
        if verbose and item.message:
            print(f"  {item.message}")
    print(f"Discovered {result.summary['episodes_discovered']}")
    print(f"Eligible   {result.summary['episodes_eligible']}")
    print(f"Ineligible {result.summary['episodes_ineligible']}")
    print(f"Processed  {result.summary['episodes_processed']}")
    print(f"Skipped    {result.summary['episodes_skipped']}")
    print(f"Failed     {result.summary['episodes_failed']}")
    print(f"Mean conf  {result.summary['mean_confidence']}")
    print(f"Wall time  {result.wall_time_s:.3f} s")


if __name__ == "__main__":
    raise SystemExit(main())
