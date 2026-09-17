"""Command-line interface for production episode-isolated dataset processing."""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from vla_data.annotation.batch import AnnotationConfigurationError
from vla_data.annotation.hierarchical import annotate_hierarchical_dataset
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
from vla_data.export import export_lerobot, validate_lerobot_export
from vla_data.manifest import SplitConfig, build_training_manifest
from vla_data.manifest.qc import review_episode
from vla_data.verification import VerificationPolicy, verify_annotations


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

    rgb = commands.add_parser(
        "render-rgb-review", help="read-only head/wrist QC video from D3 selection"
    )
    for name in ("curated-root", "quality-root", "output-root"):
        rgb.add_argument(f"--{name}", type=Path, required=True)
    rgb.add_argument("--episode", required=True)
    rgb.add_argument("--selection", choices=("valid", "invalid", "all"), default="all")
    rgb.add_argument("--dry-run", action="store_true")

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
        "annotate", help="two-pass GLM hierarchical semantics over D2/D3 clean domains"
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
    annotate.add_argument(
        "--provider",
        choices=("standard-glm", "coding-plan-vision-mcp"),
        default="standard-glm",
        help="annotation VLM backend (default: standard BigModel PaaS GLM)",
    )
    annotate.add_argument(
        "--prompt-version",
        choices=("v1", "v2", "v3", "v3.1", "v3.2", "v3.3"),
        default="v1",
        help=(
            "prompt family (v3.3 = v3.2 recall plus canonical compliance and "
            "coarse-objective separation; v3.2 = stage-specific Pass-A recall "
            "calibration + Pass-B local preservation)"
        ),
    )
    annotate.add_argument(
        "--baseline-annotation-root",
        type=Path,
        help="read-only D4 baseline root for v3 semantic comparison JSON/CSV",
    )
    annotate.add_argument("--verbose", action="store_true")
    review = commands.add_parser(
        "review-annotation", help="review D5.1 verification; preserve D4 annotation"
    )
    review.add_argument("--verification-root", type=Path, required=True)
    review.add_argument("--episode", required=True)
    review.add_argument(
        "--status",
        choices=("HUMAN_VERIFIED", "HUMAN_CORRECTED", "REJECTED"),
        required=True,
    )
    review.add_argument("--instruction")
    review.add_argument("--reviewer")
    review.add_argument("--semantic-segment-id")
    review.add_argument("--start-curated-index", type=int)
    review.add_argument("--end-curated-index", type=int)

    manifest = commands.add_parser(
        "build-manifest", help="resolve approved training references and episode splits"
    )
    for name in (
        "curated-root",
        "quality-root",
        "annotation-root",
        "verification-root",
        "output-root",
    ):
        manifest.add_argument(f"--{name}", type=Path, required=True)
    manifest.add_argument("--episode")
    manifest.add_argument("--force", action="store_true")
    manifest.add_argument("--dry-run", action="store_true")
    manifest.add_argument("--seed", type=int, default=0)
    manifest.add_argument(
        "--validation-fraction",
        type=float,
        default=0.0,
        help="explicit validation fraction; default 0, no production ratio prescribed",
    )
    verify = commands.add_parser(
        "verify-annotations",
        help="quality-first confidence verification without model inference",
    )
    for name in ("annotation-root", "quality-root", "output-root"):
        verify.add_argument(f"--{name}", type=Path, required=True)
    verify.add_argument("--confidence-threshold", type=float, required=True)
    verify.add_argument("--policy-version", default="confidence_gate_v1")
    verify.add_argument("--episode")
    verify.add_argument("--force", action="store_true")
    verify.add_argument("--dry-run", action="store_true")
    export = commands.add_parser(
        "export-lerobot", help="local LeRobot v2.1 export with contiguous-run isolation"
    )
    export.add_argument(
        "--curated-root",
        type=Path,
        help="D2 root for the required direct D2+D3 production path",
    )
    export.add_argument(
        "--quality-root",
        type=Path,
        help="D3 root for the required direct D2+D3 production path",
    )
    export.add_argument(
        "--manifest-root",
        type=Path,
        help="optional legacy D4/D5 semantic-manifest path",
    )
    export.add_argument("--output-root", type=Path, required=True)
    export.add_argument("--episode")
    export.add_argument("--validation-fraction", type=float, default=0.0)
    export.add_argument("--split-seed", type=int, default=0)
    export.add_argument("--dataset-name", default="vla-local")
    export.add_argument(
        "--lerobot-python",
        type=Path,
        help="existing interpreter with audited LeRobot v2.1; never installs packages",
    )
    export.add_argument("--force", action="store_true")
    export.add_argument("--dry-run", action="store_true")

    validate_export = commands.add_parser(
        "validate-lerobot",
        help="reconstruct source plan and validate an existing LeRobot export",
    )
    validate_export.add_argument("--output-root", type=Path, required=True)
    validate_export.add_argument(
        "--lerobot-python",
        type=Path,
        help="existing interpreter with audited LeRobot v2.1; never installs packages",
    )

    probe = commands.add_parser(
        "probe-provider",
        help="zero-inference provider diagnostic (MCP handshake + tool discovery)",
    )
    probe.add_argument(
        "--provider",
        choices=("coding-plan-vision-mcp",),
        default="coding-plan-vision-mcp",
    )

    publish = commands.add_parser(
        "publish-hf", help="publish a validated LeRobot export to Hugging Face"
    )
    publish.add_argument("--dataset-root", type=Path, required=True)
    publish.add_argument("--repo-id", required=True)
    publish.add_argument(
        "--hf-python",
        type=Path,
        help="existing interpreter with huggingface_hub; never installs packages",
    )
    publish.add_argument(
        "--lerobot-python",
        type=Path,
        required=True,
        help="interpreter with audited LeRobot for remote official reload",
    )
    publish.add_argument("--revision")
    publish.add_argument("--cache-dir", type=Path)
    publish.add_argument(
        "--force",
        action="store_true",
        help="conservative retry flag; never duplicates episodes or deletes unknown files",
    )
    publish.add_argument("--dry-run", action="store_true")
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "render-rgb-review":
            import json

            from vla_data.diagnostics.rgb_review import render_rgb_review

            result = render_rgb_review(
                args.curated_root,
                args.quality_root,
                args.output_root,
                episode=args.episode,
                selection=args.selection,
                dry_run=args.dry_run,
            )
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "export-lerobot":
            import json

            result = export_lerobot(
                args.manifest_root,
                args.output_root,
                curated_root=args.curated_root,
                quality_root=args.quality_root,
                episode=args.episode,
                validation_fraction=args.validation_fraction,
                split_seed=args.split_seed,
                dataset_name=args.dataset_name,
                lerobot_python=args.lerobot_python,
                force=args.force,
                dry_run=args.dry_run,
            )
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "validate-lerobot":
            import json

            result = validate_lerobot_export(
                args.output_root,
                lerobot_python=args.lerobot_python,
            )
            print(
                json.dumps(
                    {
                        "passed": result.passed,
                        "errors": result.errors,
                        "evidence": result.evidence,
                    },
                    indent=2,
                )
            )
            return 0 if result.passed else 1
        if args.command == "publish-hf":
            import json

            from vla_data.publish import publish_dataset, validate_remote

            if args.revision is not None:
                if args.dry_run or args.force:
                    raise ValueError(
                        "--revision is verification-only; omit --dry-run/--force"
                    )
                cache = args.cache_dir or Path(
                    tempfile.mkdtemp(prefix="vla_hf_verify_")
                )
                validation = validate_remote(
                    args.dataset_root,
                    args.repo_id,
                    revision=args.revision,
                    hf_python=str(args.hf_python) if args.hf_python else None,
                    lerobot_python=str(args.lerobot_python),
                    cache_dir=cache,
                )
                print(json.dumps(validation, indent=2))
                return 0
            evidence = publish_dataset(
                args.dataset_root,
                args.repo_id,
                hf_python=str(args.hf_python) if args.hf_python else None,
                lerobot_python=str(args.lerobot_python),
                dry_run=args.dry_run,
                force=args.force,
                verification_cache_dir=args.cache_dir,
            )
            print(
                f"Repo        {evidence['repo_id']} "
                f"(account {evidence.get('account', 'not queried')})"
            )
            print(
                f"Local       {evidence['local_file_count']} files, "
                f"{evidence['local_bytes']} bytes, "
                f"fingerprint {evidence['local_fingerprint'][:16]}…"
            )
            if "remote_before" in evidence:
                print(
                    f"Remote      {evidence['remote_before']['file_count']} files, "
                    f"private={evidence['remote_before']['private']}"
                )
            if "merge_plan" in evidence:
                plan = evidence["merge_plan"]
                print(
                    "Merge       "
                    f"BASELINE={plan['baseline_episodes']}/{plan['baseline_frames']} "
                    f"INCOMING={plan['incoming_episodes']}/{plan['incoming_frames']} "
                    f"DUPLICATES={plan['already_present']} "
                    f"TO_APPEND={plan['to_append']} "
                    f"MERGED_TOTAL={plan['merged_episodes']}/{plan['merged_frames']}"
                )
            print(f"Action      {evidence['action']}")
            if evidence.get("would_action"):
                print(f"Would       {evidence['would_action']}")
            if evidence.get("reason"):
                print(f"Reason      {evidence['reason']}")
            if evidence["action"] == "BLOCKED":
                return 2
            if evidence["action"] == "UPLOADED":
                print(f"Commit      {evidence['commit_sha']}")
                print(f"LeRobot tag {evidence['codebase_tag']}")
                print(
                    f"Uploaded    {evidence['uploaded_file_count']} files, "
                    f"{evidence['uploaded_bytes']} bytes"
                )
            if args.dry_run or evidence["action"] != "UPLOADED":
                return 0
            validation = evidence["remote_validation"]
            lerobot = validation["lerobot"]
            print("Remote validation PASS")
            print(f"  revision  {validation['revision']}")
            print(f"  snapshot  {validation['snapshot_path']}")
            print(
                f"  hash      {validation['canonical_file_count']} canonical "
                f"files, {validation['canonical_file_mismatches']} mismatches"
            )
            print(
                f"  LeRobot   episodes={lerobot['episodes']} frames={lerobot['frames']} "
                f"lengths={lerobot['episode_lengths']} fps={lerobot['fps']}"
            )
            print(f"  task      {lerobot['tasks']}")
            print(
                f"  arrays    state={lerobot['state_shape']} "
                f"action={lerobot['action_shape']} "
                f"bitwise_equal={lerobot['state_action_bitwise_equal']}"
            )
            return 0
        if args.command == "verify-annotations":
            result = verify_annotations(
                args.annotation_root,
                args.quality_root,
                args.output_root,
                policy=VerificationPolicy(
                    args.confidence_threshold, args.policy_version
                ),
                episode=args.episode,
                force=args.force,
                dry_run=args.dry_run,
            )
            for record in result.records:
                print(f"{record['episode_id']} {record['verification_status']}")
            prefix = "would_" if args.dry_run else ""
            print(f"{prefix}auto_verify: {result.summary['auto_verified_count']}")
            print(
                f"{prefix}require_human_review: {result.summary['needs_human_review_count']}"
            )
            print(f"ineligible: {result.summary['ineligible_count']}")
            print(f"processed: {result.processed}; skipped: {result.skipped}")
            return 0
        if args.command == "review-annotation":
            updated = review_episode(
                args.verification_root,
                args.episode,
                status=args.status,
                instruction=args.instruction,
                reviewer=args.reviewer,
                semantic_segment_id=args.semantic_segment_id,
                start_curated_index=args.start_curated_index,
                end_curated_index=args.end_curated_index,
            )
            print(f"{args.episode} {updated['verification_status']}")
            return 0
        if args.command == "build-manifest":
            result = build_training_manifest(
                args.curated_root,
                args.quality_root,
                args.annotation_root,
                args.output_root,
                verification_root=args.verification_root,
                episode=args.episode,
                force=args.force,
                dry_run=args.dry_run,
                config=SplitConfig(
                    validation_fraction=args.validation_fraction, seed=args.seed
                ),
            )
            print(result.status)
            for record in result.records:
                print(
                    f"{record['episode_id']} {record['eligibility']} {','.join(record['reason'])}"
                )
            for key in (
                "episodes_discovered",
                "training_eligible",
                "needs_review",
                "excluded",
                "train_episodes",
                "validation_episodes",
            ):
                print(f"{key}: {result.summary[key]}")
            return 0
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

        if args.command == "probe-provider":
            import json

            from vla_data.annotation.mcp_provider import (
                CodingPlanAPIKeyMissingError,
                CodingPlanVisionMCPProvider,
                MCPProviderConfig,
                resolve_z_ai_api_key,
            )

            try:
                resolve_z_ai_api_key()
            except CodingPlanAPIKeyMissingError as exc:
                print(f"vla-data: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 2
            provider = CodingPlanVisionMCPProvider(MCPProviderConfig())
            try:
                print(json.dumps(provider.probe(), indent=2, sort_keys=True))
            finally:
                provider.close()
            return 0

        if args.command == "annotate":
            from vla_data.annotation.glm_provider import (
                GLM46VFlashProvider,
                GLMProviderConfig,
                MissingAPIKeyError,
                resolve_api_key,
            )
            from vla_data.annotation.mcp_provider import (
                CodingPlanAPIKeyMissingError,
                CodingPlanVisionMCPProvider,
                MCPProviderConfig,
                resolve_z_ai_api_key,
            )

            if not args.dry_run:
                if args.provider == "coding-plan-vision-mcp":
                    try:
                        resolve_z_ai_api_key()
                    except CodingPlanAPIKeyMissingError as exc:
                        print(
                            f"vla-data: {type(exc).__name__}: {exc}",
                            file=sys.stderr,
                        )
                        return 2
                else:
                    try:
                        resolve_api_key()
                    except MissingAPIKeyError as exc:
                        print(
                            f"vla-data: {type(exc).__name__}: {exc}",
                            file=sys.stderr,
                        )
                        return 2
            provider = None
            if not args.dry_run:
                if args.provider == "coding-plan-vision-mcp":
                    provider = CodingPlanVisionMCPProvider(
                        MCPProviderConfig(
                            storyboard_root=(
                                args.output_root / "debug_storyboards"
                                if args.prompt_version in {"v3", "v3.1", "v3.2", "v3.3"}
                                else None
                            )
                        )
                    )
                else:
                    provider = GLM46VFlashProvider(GLMProviderConfig())
            try:
                from vla_data.benchmark.episode_source import (
                    is_benchmark_episode_dir,
                )
                from vla_data.benchmark.libero import LIBERO_PLATFORM_CONTEXT

                platform_context = None
                if is_benchmark_episode_dir(
                    args.curated_root / (args.episode or "episode_000000")
                ) or any(
                    is_benchmark_episode_dir(path)
                    for path in args.curated_root.glob("episode_*")
                ):
                    # Benchmark sources describe their own embodiment; the
                    # production RM65B+SG100 context must never leak in.
                    platform_context = LIBERO_PLATFORM_CONTEXT
                annotation_result = annotate_hierarchical_dataset(
                    args.curated_root,
                    args.quality_root,
                    args.output_root,
                    provider,
                    episode=args.episode,
                    force=args.force,
                    dry_run=args.dry_run,
                    concurrency=args.concurrency,
                    max_coarse_points=args.max_temporal_points,
                    prompt_family=args.prompt_version,
                    baseline_annotation_root=args.baseline_annotation_root,
                    platform_context=platform_context,
                )
            finally:
                if provider is not None:
                    close = getattr(provider, "close", None)
                    if close is not None:
                        close()
            _print_hierarchical_result(annotation_result, verbose=args.verbose)
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
        TypeError,
    ) as exc:
        print(f"vla-data: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        # Publication safety failures (auth required, remote not empty,
        # remote validation mismatch) are actionable stops, not tracebacks.
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


def _print_hierarchical_result(result, *, verbose: bool) -> None:
    context = result.summary.get("platform_context")
    if isinstance(context, dict):
        print(f"Platform    {context['robot_arm']} + {context['end_effector']}")
        print(f"Head camera {context['head_camera_role']}")
        print(f"Wrist camera {context['right_wrist_camera_role']}")
    for index, item in enumerate(result.results, start=1):
        print(f"[{index}/{len(result.results)}] {item.episode_id}  {item.status}")
        print(
            f"  domains={item.clean_domains} pass_a_keyframes={item.pass_a_keyframes}"
        )
        if item.pass_a_indices:
            print(f"  pass_a_curated_indices={list(item.pass_a_indices)}")
        print(f"  expected_glm_calls={item.expected_glm_calls}")
        if item.pass_b_mode:
            print(f"  pass_b_mode={item.pass_b_mode}")
        if item.pass_a_calls is not None:
            print(f"  pass_a_calls={item.pass_a_calls}")
        if item.pass_b_calls is not None:
            print(f"  pass_b_calls={item.pass_b_calls}")
        print(f"  prompt_versions={','.join(item.prompt_versions)}")
        print(f"  destination={item.destination}")
        if verbose and item.message:
            print(f"  {item.message}")
    print(f"Discovered {result.summary['episodes_discovered']}")
    print(f"Processed  {result.summary['episodes_processed']}")
    print(f"Skipped    {result.summary['episodes_skipped']}")
    print(f"Ineligible {result.summary['episodes_ineligible']}")
    print(f"Failed     {result.summary['episodes_failed']}")
    print(f"Wall time  {result.wall_time_s:.3f} s")


if __name__ == "__main__":
    raise SystemExit(main())
