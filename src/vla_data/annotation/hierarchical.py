"""Two-pass hierarchical task annotation over D2/D3-authorized domains."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from vla_data.annotation.boundary_local import (
    BOUNDARY_LOCAL_MODE,
    BOUNDARY_LOCAL_SAMPLING_STRATEGY,
    build_boundary_transitions,
    merge_local_boundary_results,
    validate_local_boundary_result,
)
from vla_data.annotation.eligibility import EligibilityError, evaluate_eligibility
from vla_data.annotation.platform_context import (
    DEFAULT_PLATFORM_CONTEXT,
    PlatformContext,
)
from vla_data.annotation.provider import (
    CANONICAL_CAMERA_ROLES,
    ERROR_MODEL_OUTPUT_SCHEMA,
    AnnotationRequest,
    ImageItem,
    ProviderError,
    StructuredModelResponse,
    StructuredVLMProvider,
    TextItem,
)
from vla_data.annotation.provider_raw import (
    load_compatible_provider_response,
    persist_provider_response,
    response_artifact_path,
)
from vla_data.annotation.schema import (
    ANNOTATION_SCHEMA_NAME,
    HIERARCHICAL_ANNOTATION_SCHEMA_VERSION,
    AnnotationSchemaError,
    validate_hierarchical_annotation,
    write_annotation,
)
from vla_data.annotation.semantic_diagnostics import (
    BOUNDARY_EVIDENCE_AMBIGUOUS,
    BOUNDARY_EVIDENCE_INSUFFICIENT,
    REJECTED_OPTIONAL_PARAPHRASE,
    SEMANTIC_TASK_SEQUENCE_MISMATCH,
    VERY_SHORT_NONTRAINING_INTERVAL,
    concrete_task_with_insufficient_domain,
    pass_a_internal_inconsistency,
    task_sequence_mismatch,
    very_short_nontraining_intervals,
)
from vla_data.batch.discovery import discover_curated_episodes
from vla_data.batch.summary import write_summary
from vla_data.io.curated_episode import CuratedEpisode

PASS_A_PROMPT_VERSION = "hierarchical_semantics_coarse_v1"
PASS_B_PROMPT_VERSION = "hierarchical_semantics_boundary_refine_v1"
KEYFRAMES_SCHEMA_NAME = "vla_hierarchical_annotation_keyframes"
DEFAULT_COARSE_POINTS = 12
DEFAULT_BOUNDARY_RADIUS = 6
COARSE_SAMPLING_STRATEGY = "deterministic_even_clean_domain_allocation_v1"
BOUNDARY_SAMPLING_STRATEGY = "deterministic_dense_boundary_neighborhood_v1"
SEMANTIC_RECALL_DIAGNOSTIC_CONFIDENCE_FLOOR = 0.10
SEMANTIC_RECALL_REVIEW = "SEMANTIC_RECALL_REVIEW"
ANNOTATION_STATUS_SUCCESS = "SUCCESS"
ANNOTATION_STATUS_PROVIDER_FAILED = "PROVIDER_FAILED"
ANNOTATION_STATUS_VALIDATION_FAILED = "VALIDATION_FAILED"
ANNOTATION_STATUS_MISSING = "MISSING"
_PROVIDER_FAILURE_CATEGORIES = frozenset(
    {
        "NETWORK_ERROR",
        "HTTP_RATE_LIMIT",
        "HTTP_SERVER_ERROR",
        "HTTP_AUTH",
        "HTTP_QUOTA",
        "HTTP_PROVIDER_OVERLOAD",
        "HTTP_PERMISSION",
        "HTTP_OTHER",
        "MCP_STARTUP_ERROR",
        "MCP_INITIALIZE_ERROR",
        "MCP_TOOL_NOT_FOUND",
        "MCP_TOOL_CALL_ERROR",
        "MCP_TIMEOUT",
        "MCP_PROCESS_EXITED",
    }
)


def _failure_annotation_status(exc: Exception) -> str:
    if isinstance(exc, ProviderError) and exc.category in _PROVIDER_FAILURE_CATEGORIES:
        return ANNOTATION_STATUS_PROVIDER_FAILED
    return ANNOTATION_STATUS_VALIDATION_FAILED


@dataclass(frozen=True)
class PromptFamily:
    """Versioned prompt pair; output shape stays Annotation Schema V2."""

    name: str
    pass_a_version: str
    pass_b_version: str
    build_pass_a: Any
    build_pass_b: Any
    boundary_local: bool = False
    platform_context: PlatformContext | None = None


def get_prompt_family(
    name: str, platform_context: PlatformContext | None = None
) -> PromptFamily:
    from vla_data.annotation import (
        prompts_v2,
        prompts_v3,
        prompts_v31,
        prompts_v32,
        prompts_v33,
    )

    if name == "v1":
        return PromptFamily(
            "v1",
            PASS_A_PROMPT_VERSION,
            PASS_B_PROMPT_VERSION,
            _pass_a_prompt,
            _pass_b_prompt,
        )
    if name == "v2":
        return PromptFamily(
            "v2",
            prompts_v2.PASS_A_PROMPT_VERSION,
            prompts_v2.PASS_B_PROMPT_VERSION,
            prompts_v2.build_pass_a_prompt_v2,
            prompts_v2.build_pass_b_prompt_v2,
        )
    if name == "v3":
        return PromptFamily(
            "v3",
            prompts_v3.PASS_A_PROMPT_VERSION,
            prompts_v3.PASS_B_PROMPT_VERSION,
            prompts_v3.build_pass_a_prompt_v3,
            prompts_v3.build_pass_b_prompt_v3,
            boundary_local=True,
        )
    if name == "v3.1":
        context = platform_context or DEFAULT_PLATFORM_CONTEXT
        return PromptFamily(
            "v3.1",
            prompts_v31.PASS_A_PROMPT_VERSION,
            prompts_v31.PASS_B_PROMPT_VERSION,
            partial(prompts_v31.build_pass_a_prompt_v31, platform_context=context),
            partial(prompts_v31.build_pass_b_prompt_v31, platform_context=context),
            boundary_local=True,
            platform_context=context,
        )
    if name == "v3.2":
        context = platform_context or DEFAULT_PLATFORM_CONTEXT
        return PromptFamily(
            "v3.2",
            prompts_v32.PASS_A_PROMPT_VERSION,
            prompts_v32.PASS_B_PROMPT_VERSION,
            partial(prompts_v32.build_pass_a_prompt_v32, platform_context=context),
            partial(prompts_v32.build_pass_b_prompt_v32, platform_context=context),
            boundary_local=True,
            platform_context=context,
        )
    if name == "v3.3":
        context = platform_context or DEFAULT_PLATFORM_CONTEXT
        return PromptFamily(
            "v3.3",
            prompts_v33.PASS_A_PROMPT_VERSION,
            prompts_v33.PASS_B_PROMPT_VERSION,
            partial(prompts_v33.build_pass_a_prompt_v33, platform_context=context),
            partial(prompts_v33.build_pass_b_prompt_v33, platform_context=context),
            boundary_local=True,
            platform_context=context,
        )
    raise ValueError(
        f"unknown prompt family {name!r} (expected v1, v2, v3, v3.1, v3.2, or v3.3)"
    )


@dataclass(frozen=True)
class HierarchicalResult:
    episode_id: str
    status: str
    quality_outcome: str | None
    clean_domains: int
    pass_a_keyframes: int
    pass_b_keyframes: int
    expected_glm_calls: int
    destination: str
    prompt_versions: tuple[str, str]
    semantic_segments: int | None = None
    message: str | None = None
    pass_a_indices: tuple[int, ...] = ()
    http_status: int | None = None
    provider_error_code: str | None = None
    provider_error_message: str | None = None
    provider_request_id: str | None = None
    planned_provider_calls: int | None = None
    completed_provider_calls: int | None = None
    failed_provider_calls: int | None = None
    minimum_planned_provider_calls: int | None = None
    provider_metadata: dict[str, object] | None = None
    pass_b_mode: str | None = None
    pass_a_calls: int | None = None
    pass_b_calls: int | str | None = None
    actual_planned_provider_calls: int | None = None
    diagnostic_codes: tuple[str, ...] = ()
    annotation_status: str | None = None
    failure_reason: str | None = None
    semantic_evaluated: bool | None = None
    logical_provider_calls: int | None = None
    actual_provider_calls: int | None = None
    replayed_provider_responses: int | None = None
    actual_tools_call_attempts: int | None = None
    retry_count: int | None = None
    platform_context_fingerprint: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "status": self.status,
            "quality_outcome": self.quality_outcome,
            "clean_domains": self.clean_domains,
            "pass_a_keyframes": self.pass_a_keyframes,
            "pass_b_keyframes": self.pass_b_keyframes,
            "expected_glm_calls": self.expected_glm_calls,
            "destination": self.destination,
            "prompt_versions": list(self.prompt_versions),
            "semantic_segments": self.semantic_segments,
            "message": self.message,
            "pass_a_indices": list(self.pass_a_indices),
            "http_status": self.http_status,
            "provider_error_code": self.provider_error_code,
            "provider_error_message": self.provider_error_message,
            "provider_request_id": self.provider_request_id,
            "planned_provider_calls": self.planned_provider_calls,
            "completed_provider_calls": self.completed_provider_calls,
            "failed_provider_calls": self.failed_provider_calls,
            "minimum_planned_provider_calls": self.minimum_planned_provider_calls,
            "provider_metadata": self.provider_metadata,
            "pass_b_mode": self.pass_b_mode,
            "pass_a_calls": self.pass_a_calls,
            "pass_b_calls": self.pass_b_calls,
            "actual_planned_provider_calls": self.actual_planned_provider_calls,
            "diagnostic_codes": list(self.diagnostic_codes),
            "annotation_status": self.annotation_status,
            "failure_reason": self.failure_reason,
            "semantic_evaluated": self.semantic_evaluated,
            "logical_provider_calls": self.logical_provider_calls,
            "actual_provider_calls": self.actual_provider_calls,
            "replayed_provider_responses": self.replayed_provider_responses,
            "actual_tools_call_attempts": self.actual_tools_call_attempts,
            "retry_count": self.retry_count,
            "platform_context_fingerprint": self.platform_context_fingerprint,
        }


@dataclass(frozen=True)
class HierarchicalDatasetResult:
    results: tuple[HierarchicalResult, ...]
    summary: dict[str, object]
    wall_time_s: float

    @property
    def failed_count(self) -> int:
        return sum(result.status == "FAILED" for result in self.results)


def _provider_counter_snapshot(provider: object) -> dict[str, int | None]:
    """Capture cumulative provider counters before one episode."""

    calls = getattr(provider, "calls", None)
    return {
        "planned": getattr(provider, "planned_provider_calls", None),
        "completed": getattr(provider, "completed_provider_calls", None),
        "failed": getattr(provider, "failed_provider_calls", None),
        "tool_attempts": getattr(provider, "mcp_tools_call_attempted", None),
        "retries": getattr(provider, "retry_count", None),
        "calls": len(calls) if isinstance(calls, list) else None,
    }


def _provider_call_deltas(
    provider: object, before: dict[str, int | None], *, episode_failed: bool
) -> dict[str, int | None]:
    """Report episode-local deltas even when one provider serves a dataset."""

    after = _provider_counter_snapshot(provider)

    def delta(name: str) -> int | None:
        left, right = before.get(name), after.get(name)
        if isinstance(left, int) and isinstance(right, int):
            return right - left
        return None

    calls = delta("calls")
    logical = delta("planned")
    completed = delta("completed")
    failed = delta("failed")
    tool_attempts = delta("tool_attempts")
    retries = delta("retries")
    if completed is None and calls is not None:
        completed = max(0, calls - (1 if episode_failed else 0))
    if failed is None and calls is not None:
        failed = 1 if episode_failed and calls else 0
    if logical is None:
        logical = calls
    return {
        "completed_provider_calls": completed,
        "failed_provider_calls": failed,
        "logical_provider_calls": logical,
        "actual_tools_call_attempts": tool_attempts,
        "retry_count": retries,
    }


def _episode_call_accounting(
    provider: object,
    before: dict[str, int | None],
    *,
    episode_failed: bool,
    logical_provider_calls: int,
    actual_provider_calls: int,
    replayed_provider_responses: int,
) -> dict[str, int | None]:
    """Combine provider counters with explicit local-replay accounting."""

    fields = _provider_call_deltas(
        provider,
        before,
        episode_failed=episode_failed,
    )
    fields.update(
        {
            "logical_provider_calls": logical_provider_calls,
            "actual_provider_calls": actual_provider_calls,
            "replayed_provider_responses": replayed_provider_responses,
        }
    )
    return fields


def clean_domains(offsets: tuple[int, ...], mask: np.ndarray) -> list[dict[str, int]]:
    """Intersect D2 physical segments with contiguous D3-clean runs."""

    mask = np.asarray(mask)
    if mask.dtype != np.dtype(bool) or mask.ndim != 1:
        raise AnnotationSchemaError("quality mask must be one-dimensional bool")
    if (
        len(offsets) < 2
        or offsets[0] != 0
        or offsets[-1] != len(mask)
        or any(start >= end for start, end in pairwise(offsets))
    ):
        raise AnnotationSchemaError("invalid D2 segment offsets")
    result: list[dict[str, int]] = []
    for segment_id, (segment_start, segment_end) in enumerate(pairwise(offsets)):
        run_id = 0
        index = segment_start
        while index < segment_end:
            if not mask[index]:
                index += 1
                continue
            start = index
            while index < segment_end and mask[index]:
                index += 1
            result.append(
                {
                    "d2_segment_id": segment_id,
                    "clean_run_id": run_id,
                    "start_curated_index": start,
                    "end_curated_index": index,
                }
            )
            run_id += 1
    return result


def _even_indices(start: int, end: int, count: int) -> list[int]:
    length = end - start
    if length <= count:
        return list(range(start, end))
    return sorted({int(value) for value in np.linspace(start, end - 1, count)})


def _allocate_coarse(domains: list[dict[str, int]], maximum: int) -> list[int]:
    maximum = max(maximum, len(domains))
    lengths = [
        item["end_curated_index"] - item["start_curated_index"] for item in domains
    ]
    allocation = [1] * len(domains)
    remaining = maximum - len(domains)
    while remaining:
        candidate = max(
            range(len(domains)),
            key=lambda index: (lengths[index] / allocation[index], -index),
        )
        allocation[candidate] += 1
        remaining -= 1
    selected: list[int] = []
    for domain, count in zip(domains, allocation, strict=True):
        selected.extend(
            _even_indices(
                domain["start_curated_index"], domain["end_curated_index"], count
            )
        )
    return sorted(set(selected))


def _frame_record(episode: CuratedEpisode, index: int) -> dict[str, object]:
    trajectory = episode.trajectory
    head = int(trajectory["head_rgb_frame_index"][index])
    wrist = int(trajectory["right_wrist_rgb_frame_index"][index])
    head_path = episode.media.rgb_path("head", head)
    wrist_path = episode.media.rgb_path("right_wrist", wrist)
    for path in (head_path, wrist_path):
        if not path.is_file():
            raise AnnotationSchemaError(f"referenced Curated image is missing: {path}")
    return {
        "curated_index": index,
        "timestamp_ns": int(trajectory["action_timestamp_ns"][index]),
        "relative_time_ns": int(
            trajectory["action_timestamp_ns"][index]
            - trajectory["action_timestamp_ns"][0]
        ),
        "head": {"frame_index": head, "source_path": str(head_path)},
        "right_wrist": {"frame_index": wrist, "source_path": str(wrist_path)},
    }


def _items(
    records: list[dict[str, object]], prompt: str
) -> tuple[TextItem | ImageItem, ...]:
    result: list[TextItem | ImageItem] = [TextItem(prompt)]
    for record in records:
        index = int(record["curated_index"])
        result.append(
            TextItem(
                f"curated_index={index}, timestamp_ns={record['timestamp_ns']}; "
                f"relative_time_ns={record['relative_time_ns']}; "
                "head image then wrist image:"
            )
        )
        for camera in CANONICAL_CAMERA_ROLES:
            block = record[camera]
            assert isinstance(block, dict)
            result.append(
                ImageItem(
                    temporal_point=index,
                    camera=camera,
                    label=f"curated_index={index} {camera}",
                    source_path=Path(str(block["source_path"])),
                )
            )
    return tuple(result)


def _pass_a_prompt(domains: list[dict[str, int]], allowed: list[int]) -> str:
    return f"""Label WHAT happens in this complete clean robot trajectory, never joint/motor motion.
Return only JSON with episode_task and a coarse classification:
{{"episode_task":{{"instruction":str,"confidence":number,"task_type":str|null,"objects":[str],"paraphrases":[str]}},
"semantic_segments":[{{"segment_id":str,"start_curated_index":int,"end_curated_index":int,
"instruction":str,"confidence":number,"task_type":str|null,"objects":[str],"paraphrases":[str]}}],
"non_training_intervals":[{{"start_curated_index":int,"end_curated_index":int,"reason":str}}]}}
All boundaries are [start,end), must come from allowed_boundary_indices, must remain within one clean_domain,
and segments plus non-training intervals must exactly classify every clean domain. No minimum length.
Use one short imperative canonical instruction per task/segment. Subtasks should be meaningful intent changes,
prefer verb + object + goal/spatial relation, and must not over-segment motor primitives or collapse clearly distinct intents.
Never create incorrect/distractor language-action pairs. Put ambiguous/idle/unsupported spans in non_training_intervals.
clean_domains={json.dumps(domains, sort_keys=True)}
allowed_boundary_indices={allowed}"""


def _pass_b_prompt(
    domains: list[dict[str, int]], allowed: list[int], coarse: dict[str, Any]
) -> str:
    return f"""Refine only temporal boundaries for the supplied coarse semantic annotation.
Use semantic WHAT instructions; never narrate joints, motors, frames, or timestamps.
Return only JSON with semantic_segments and non_training_intervals in the same shapes as Pass A.
Every [start,end) boundary must be one of allowed_boundary_indices and remain inside one clean_domain.
Classify every clean-domain index exactly once; preserve or improve instruction meaning and confidence.
clean_domains={json.dumps(domains, sort_keys=True)}
allowed_boundary_indices={allowed}
coarse_annotation={json.dumps(coarse, sort_keys=True)}"""


def _boundary_values(payload: dict[str, Any]) -> list[int]:
    values: list[int] = []
    for key in ("semantic_segments", "non_training_intervals"):
        blocks = payload.get(key, [])
        if isinstance(blocks, list):
            for block in blocks:
                if isinstance(block, dict):
                    values.extend(
                        value
                        for value in (
                            block.get("start_curated_index"),
                            block.get("end_curated_index"),
                        )
                        if isinstance(value, int) and not isinstance(value, bool)
                    )
    return values


def _validate_model_stage(
    *,
    episode_id: str,
    quality_outcome: str,
    domains: list[dict[str, int]],
    episode_task: object,
    payload: dict[str, Any],
    allowed_boundaries: set[int],
) -> dict[str, Any]:
    if any(value not in allowed_boundaries for value in _boundary_values(payload)):
        raise AnnotationSchemaError(
            "model invented a boundary outside provided indices"
        )
    candidate = {
        "schema_name": ANNOTATION_SCHEMA_NAME,
        "schema_version": HIERARCHICAL_ANNOTATION_SCHEMA_VERSION,
        "annotation_schema_version": 2,
        "episode_id": episode_id,
        "quality_outcome": quality_outcome,
        "episode_task": episode_task,
        "semantic_segments": payload.get("semantic_segments"),
        "non_training_intervals": payload.get("non_training_intervals"),
        "clean_domains": domains,
        "model_provenance": {
            "provider": "validation-placeholder",
            "model": "validation-placeholder",
            "pass_a_prompt_version": PASS_A_PROMPT_VERSION,
            "pass_b_prompt_version": PASS_B_PROMPT_VERSION,
        },
    }
    return validate_hierarchical_annotation(candidate)


def _provenance(response: StructuredModelResponse) -> dict[str, object]:
    return {
        "prompt_version": response.prompt_version,
        "request_id": response.request_id,
        "attempt_count": response.attempt_count,
        "finish_reason": response.finish_reason,
        "provider_metadata": response.provider_metadata,
        "usage": response.usage,
    }


def _sanitize_payload_language(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Hard-gate canonical text and soft-drop invalid optional paraphrases."""

    from vla_data.annotation.language_validator import validate_instruction

    cleaned = copy.deepcopy(payload)
    blocks: list[tuple[str, Any]] = [("episode_task", cleaned.get("episode_task"))]
    blocks.extend(
        (f"semantic_segments[{index}]", segment)
        for index, segment in enumerate(cleaned.get("semantic_segments", []))
    )
    rejected: list[dict[str, str]] = []
    for field, block in blocks:
        if not isinstance(block, dict):
            continue
        canonical = block.get("instruction")
        violations = validate_instruction(canonical)
        if violations:
            raise AnnotationSchemaError(
                f"{field}.instruction violates controlled language: "
                f"{violations[0].category}"
            )
        paraphrases = block.get("paraphrases", [])
        if not isinstance(paraphrases, list):
            continue  # structural schema validation reports this precisely
        kept = []
        for index, text in enumerate(paraphrases):
            item_violations = validate_instruction(text)
            if item_violations:
                rejected.append(
                    {
                        "field": f"{field}.paraphrases[{index}]",
                        "text": str(text),
                        "reason": item_violations[0].category,
                    }
                )
            else:
                kept.append(text)
        block["paraphrases"] = kept
    return cleaned, rejected


def _annotation_diagnostics(
    annotation: dict[str, Any],
    *,
    boundary_results: list[dict[str, Any]] | None = None,
    rejected_paraphrases: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    task = annotation["episode_task"]
    segments = annotation["semantic_segments"]
    intervals = annotation["non_training_intervals"]
    reasons = {interval["reason"] for interval in intervals}
    recall_review = bool(
        not segments
        and float(task["confidence"]) > SEMANTIC_RECALL_DIAGNOSTIC_CONFIDENCE_FLOOR
        and reasons != {"NO_TASK_RELEVANT_ACTIVITY"}
    )
    text_values = [task.get("instruction", ""), *task.get("paraphrases", [])]
    for segment in segments:
        text_values.extend(
            [segment.get("instruction", ""), *segment.get("paraphrases", [])]
        )
    results = boundary_results or []
    rejected = rejected_paraphrases or []
    short_intervals = very_short_nontraining_intervals(annotation)
    codes = []
    if pass_a_internal_inconsistency(annotation):
        codes.append("PASS_A_INTERNAL_INCONSISTENCY")
    if concrete_task_with_insufficient_domain(annotation) or recall_review:
        codes.append(SEMANTIC_RECALL_REVIEW)
    if task_sequence_mismatch(annotation):
        codes.append(SEMANTIC_TASK_SEQUENCE_MISMATCH)
    if short_intervals:
        codes.append(VERY_SHORT_NONTRAINING_INTERVAL)
    if any(result["evidence_status"] == "AMBIGUOUS" for result in results):
        codes.append(BOUNDARY_EVIDENCE_AMBIGUOUS)
    if any(
        result["evidence_status"] == "INSUFFICIENT_VISUAL_EVIDENCE"
        for result in results
    ):
        codes.append(BOUNDARY_EVIDENCE_INSUFFICIENT)
    if rejected:
        codes.append(REJECTED_OPTIONAL_PARAPHRASE)
    return {
        "diagnostic_codes": codes,
        "review_required": bool(codes),
        "semantic_recall_review": recall_review,
        "semantic_recall_confidence_floor": (
            SEMANTIC_RECALL_DIAGNOSTIC_CONFIDENCE_FLOOR
        ),
        "operator_context_present": any(
            "operator" in str(value).lower() for value in text_values
        ),
        "boundary_uncertainty": [
            {
                "transition_id": result["transition_id"],
                "evidence_status": result["evidence_status"],
                "confidence": result["confidence"],
            }
            for result in results
            if result["evidence_status"]
            in {"AMBIGUOUS", "INSUFFICIENT_VISUAL_EVIDENCE"}
        ],
        "very_short_nontraining_intervals": short_intervals,
        "rejected_paraphrases": rejected,
    }


def _boundary_provenance(
    transition: dict[str, Any],
    result: dict[str, Any],
    response: StructuredModelResponse,
) -> dict[str, object]:
    provider = _provenance(response)
    transport = response.provider_metadata or {}
    return {
        "transition_id": transition["transition_id"],
        "candidate_indices": list(transition["candidate_indices"]),
        "coarse_boundary_curated_index": transition["coarse_boundary_curated_index"],
        "selected_boundary_curated_index": result["boundary_curated_index"],
        "confidence": result["confidence"],
        "evidence_status": result["evidence_status"],
        "semantic_correction": result["semantic_correction"],
        "rejected_semantic_correction": result["rejected_semantic_correction"],
        "storyboard_path": transport.get("storyboard_path"),
        "storyboard_observations": transport.get("storyboard_observation_count"),
        "storyboard_bytes": transport.get("storyboard_final_bytes"),
        "storyboard_jpeg_quality": transport.get("storyboard_jpeg_quality"),
        "storyboard_scale": transport.get("storyboard_scale"),
        "wall_time_s": transport.get("wall_time_s"),
        "provider_result": provider,
    }


def _input_fingerprint(
    episode: CuratedEpisode, quality_episode: Path, records: list[dict[str, object]]
) -> str:
    from vla_data.benchmark.episode_source import (
        BENCHMARK_EPISODE_MARKER,
        is_benchmark_episode_dir,
    )

    if is_benchmark_episode_dir(episode.episode_dir):
        paths = [
            episode.episode_dir / BENCHMARK_EPISODE_MARKER,
            quality_episode / "quality_report.json",
            quality_episode / "quality_mask.npy",
        ]
    else:
        paths = [
            episode.episode_dir / "metadata.json",
            episode.episode_dir / "trajectory.npz",
            quality_episode / "quality_report.json",
            quality_episode / "quality_mask.npy",
        ]
    for record in records:
        for camera in CANONICAL_CAMERA_ROLES:
            block = record[camera]
            assert isinstance(block, dict)
            paths.append(Path(str(block["source_path"])))
    identity = [
        [str(path.resolve()), hashlib.sha256(path.read_bytes()).hexdigest()]
        for path in paths
    ]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def plan_hierarchical_episode(
    curated_episode_dir: str | Path,
    *,
    quality_root: str | Path,
    output_root: str | Path,
    max_coarse_points: int = DEFAULT_COARSE_POINTS,
    prompt_versions: tuple[str, str] | None = None,
    boundary_local: bool = False,
    platform_context: PlatformContext | None = None,
) -> tuple[HierarchicalResult, dict[str, object] | None]:
    episode_dir = Path(curated_episode_dir)
    episode_id = episode_dir.name
    versions = prompt_versions or (PASS_A_PROMPT_VERSION, PASS_B_PROMPT_VERSION)
    destination = Path(output_root).resolve() / episode_id / "annotation.json"
    try:
        from vla_data.benchmark.episode_source import (
            is_benchmark_episode_dir,
            load_benchmark_episode,
        )

        episode = (
            load_benchmark_episode(episode_dir)
            if is_benchmark_episode_dir(episode_dir)
            else CuratedEpisode.load(episode_dir)
        )
        decision = evaluate_eligibility(
            episode_id,
            transition_count=episode.transition_count,
            quality_report_path=Path(quality_root) / episode_id / "quality_report.json",
            quality_mask_path=Path(quality_root) / episode_id / "quality_mask.npy",
        )
        if not decision.eligible:
            return (
                HierarchicalResult(
                    episode_id,
                    "INELIGIBLE",
                    decision.quality_outcome,
                    0,
                    0,
                    0,
                    0,
                    str(destination),
                    versions,
                    message=decision.reason,
                ),
                None,
            )
        mask = np.load(
            Path(quality_root) / episode_id / "quality_mask.npy", allow_pickle=False
        )
        domains = clean_domains(episode.segment_offsets, mask)
        if not domains:
            raise AnnotationSchemaError("eligible episode has no clean domains")
        indices = _allocate_coarse(domains, max_coarse_points)
        records = [_frame_record(episode, index) for index in indices]
        plan = {
            "episode": episode,
            "quality_outcome": str(decision.quality_outcome),
            "clean_domains": domains,
            "pass_a_records": records,
            "pass_a_allowed_boundaries": sorted(
                set(indices)
                | {d["start_curated_index"] for d in domains}
                | {d["end_curated_index"] for d in domains}
            ),
            "input_fingerprint": _input_fingerprint(
                episode, Path(quality_root) / episode_id, records
            ),
            "platform_context": (
                platform_context.as_dict() if platform_context is not None else None
            ),
            "platform_context_fingerprint": (
                platform_context.fingerprint if platform_context is not None else None
            ),
        }
        return (
            HierarchicalResult(
                episode_id,
                "WOULD_PROCESS",
                decision.quality_outcome,
                len(domains),
                len(records),
                0,
                1 if boundary_local else 2,
                str(destination),
                versions,
                message=(
                    "Pass A calls=1; Pass B calls=dynamic_after_pass_a"
                    if boundary_local
                    else "Pass B keyframes are selected around Pass A candidate boundaries"
                ),
                pass_a_indices=tuple(indices),
                minimum_planned_provider_calls=1 if boundary_local else 2,
                pass_b_mode=BOUNDARY_LOCAL_MODE if boundary_local else "consolidated",
                pass_a_calls=1,
                pass_b_calls="dynamic_after_pass_a" if boundary_local else 1,
                annotation_status="WOULD_PROCESS",
                semantic_evaluated=False,
                logical_provider_calls=1,
                actual_provider_calls=0,
                replayed_provider_responses=0,
                platform_context_fingerprint=(
                    platform_context.fingerprint
                    if platform_context is not None
                    else None
                ),
            ),
            plan,
        )
    except (OSError, ValueError, TypeError, KeyError, EligibilityError) as exc:
        return (
            HierarchicalResult(
                episode_id,
                "FAILED",
                None,
                0,
                0,
                0,
                0,
                str(destination),
                versions,
                message=str(exc),
                annotation_status=ANNOTATION_STATUS_VALIDATION_FAILED,
                failure_reason=type(exc).__name__,
                semantic_evaluated=False,
                platform_context_fingerprint=(
                    platform_context.fingerprint
                    if platform_context is not None
                    else None
                ),
            ),
            None,
        )


def annotate_hierarchical_episode(
    curated_episode_dir: str | Path,
    *,
    quality_root: str | Path,
    output_root: str | Path,
    provider: StructuredVLMProvider,
    max_coarse_points: int = DEFAULT_COARSE_POINTS,
    boundary_radius: int = DEFAULT_BOUNDARY_RADIUS,
    prompt_family: str = "v1",
    platform_context: PlatformContext | None = None,
) -> HierarchicalResult:
    curated_path = Path(curated_episode_dir).resolve()
    quality_path = Path(quality_root).resolve()
    output_path = Path(output_root).resolve()
    if any(
        output_path == source
        or output_path in source.parents
        or source in output_path.parents
        for source in (curated_path, quality_path)
    ):
        raise ValueError(
            "annotation output must be separate from Curated and quality inputs"
        )
    family = get_prompt_family(prompt_family, platform_context)
    planned, plan = plan_hierarchical_episode(
        curated_episode_dir,
        quality_root=quality_root,
        output_root=output_root,
        max_coarse_points=max_coarse_points,
        prompt_versions=(family.pass_a_version, family.pass_b_version),
        boundary_local=family.boundary_local,
        platform_context=family.platform_context,
    )
    if plan is None:
        if planned.status == "FAILED":
            _write_failure_record(Path(output_root), planned)
        return planned
    episode = plan["episode"]
    assert isinstance(episode, CuratedEpisode)
    domains = plan["clean_domains"]
    records_a = plan["pass_a_records"]
    allowed_a = plan["pass_a_allowed_boundaries"]
    assert isinstance(domains, list) and isinstance(records_a, list)
    counter_before = _provider_counter_snapshot(provider)
    records_b: list[dict[str, object]] = []
    boundary_transitions: list[dict[str, Any]] = []
    boundary_results: list[dict[str, Any]] = []
    boundary_responses: list[StructuredModelResponse] = []
    boundary_keyframes: list[dict[str, Any]] = []
    rejected_paraphrases: list[dict[str, str]] = []
    actual_planned_calls = 1 if family.boundary_local else 2
    logical_provider_calls = 0
    actual_provider_calls = 0
    replayed_provider_responses = 0
    pass_b_mode = BOUNDARY_LOCAL_MODE if family.boundary_local else "consolidated"

    def obtain_response(
        request: AnnotationRequest,
        *,
        pass_name: str,
        transition_id: str | None = None,
    ) -> StructuredModelResponse:
        """Replay an exact raw response or call and persist before validation."""

        nonlocal logical_provider_calls
        nonlocal actual_provider_calls
        nonlocal replayed_provider_responses
        logical_provider_calls += 1
        artifact = response_artifact_path(
            output_root,
            planned.episode_id,
            pass_name=pass_name,
            transition_id=transition_id,
        )
        replayed = load_compatible_provider_response(
            artifact,
            request,
            pass_name=pass_name,
            transition_id=transition_id,
            input_fingerprint=str(plan["input_fingerprint"]),
            prompt_family=family.name,
            provider_identity=provider.provider_name,
            platform_context_fingerprint=(
                str(plan["platform_context_fingerprint"])
                if plan["platform_context_fingerprint"] is not None
                else None
            ),
            pass_b_mode=pass_b_mode,
        )
        if replayed is not None:
            replayed_provider_responses += 1
            return replayed
        actual_provider_calls += 1
        response = provider.infer_json(request)
        persist_provider_response(
            artifact,
            response,
            request,
            pass_name=pass_name,
            transition_id=transition_id,
            input_fingerprint=str(plan["input_fingerprint"]),
            prompt_family=family.name,
            platform_context_fingerprint=(
                str(plan["platform_context_fingerprint"])
                if plan["platform_context_fingerprint"] is not None
                else None
            ),
            pass_b_mode=pass_b_mode,
            logical_provider_call_index=logical_provider_calls,
        )
        return response

    try:
        request_a = AnnotationRequest(
            episode_id=planned.episode_id,
            prompt_version=family.pass_a_version,
            items=_items(records_a, family.build_pass_a(domains, allowed_a)),
        )
        response_a = obtain_response(request_a, pass_name="A")
        try:
            coarse_payload, rejected_paraphrases = _sanitize_payload_language(
                response_a.payload
            )
        except AnnotationSchemaError as exc:
            if "violates controlled language" not in str(exc):
                raise
            raise ProviderError(ERROR_MODEL_OUTPUT_SCHEMA, str(exc)) from exc
        # D4.3.1 frozen contract: optional paraphrases are metadata only. An
        # invalid paraphrase is soft-dropped with quarantine provenance and
        # must never fail an otherwise valid canonical annotation — for every
        # prompt family. Canonical instructions stay hard-gated inside
        # _sanitize_payload_language.
        episode_task = coarse_payload.get("episode_task")
        _validate_model_stage(
            episode_id=planned.episode_id,
            quality_outcome=str(plan["quality_outcome"]),
            domains=domains,
            episode_task=episode_task,
            payload=coarse_payload,
            allowed_boundaries=set(allowed_a),
        )
        if family.boundary_local:
            boundary_transitions = build_boundary_transitions(
                coarse_payload, domains, radius=boundary_radius
            )
            actual_planned_calls = 1 + len(boundary_transitions)
            for transition in boundary_transitions:
                local_records = [
                    _frame_record(episode, index)
                    for index in transition["candidate_indices"]
                ]
                records_b.extend(local_records)
                request_b = AnnotationRequest(
                    episode_id=planned.episode_id,
                    prompt_version=family.pass_b_version,
                    items=_items(
                        local_records,
                        family.build_pass_b(
                            transition, transition["candidate_indices"]
                        ),
                    ),
                    transport_id=f"pass_b_{transition['transition_id']}",
                )
                response_b = obtain_response(
                    request_b,
                    pass_name="B",
                    transition_id=transition["transition_id"],
                )
                local_result = validate_local_boundary_result(
                    response_b.payload, transition
                )
                boundary_results.append(local_result)
                boundary_responses.append(response_b)
                boundary_keyframes.append(
                    {
                        "transition_id": transition["transition_id"],
                        "coarse_boundary_curated_index": transition[
                            "coarse_boundary_curated_index"
                        ],
                        "allowed_boundary_indices": transition["candidate_indices"],
                        "frames": local_records,
                    }
                )
            refined = merge_local_boundary_results(
                coarse_payload, boundary_transitions, boundary_results
            )
            allowed_b = sorted(
                {
                    value
                    for transition in boundary_transitions
                    for value in transition["candidate_indices"]
                }
                | {
                    value
                    for domain in domains
                    for value in (
                        domain["start_curated_index"],
                        domain["end_curated_index"],
                    )
                }
            )
            annotation = _validate_model_stage(
                episode_id=planned.episode_id,
                quality_outcome=str(plan["quality_outcome"]),
                domains=domains,
                episode_task=episode_task,
                payload=refined,
                allowed_boundaries=set(allowed_b),
            )
        else:
            interior = []
            domain_edges = {
                value
                for domain in domains
                for value in (
                    domain["start_curated_index"],
                    domain["end_curated_index"],
                )
            }
            for value in _boundary_values(response_a.payload):
                if value not in domain_edges:
                    interior.append(value)
            indices_b: set[int] = set()
            for domain in domains:
                start = domain["start_curated_index"]
                end = domain["end_curated_index"]
                candidates = [value for value in interior if start < value < end]
                if not candidates:
                    candidates = _even_indices(start, end, min(5, end - start))
                for center in candidates:
                    indices_b.update(
                        range(
                            max(start, center - boundary_radius),
                            min(end, center + boundary_radius + 1),
                        )
                    )
            records_b = [_frame_record(episode, index) for index in sorted(indices_b)]
            allowed_b = sorted(indices_b | domain_edges)
            request_b = AnnotationRequest(
                episode_id=planned.episode_id,
                prompt_version=family.pass_b_version,
                items=_items(
                    records_b,
                    family.build_pass_b(domains, allowed_b, response_a.payload),
                ),
            )
            response_b = obtain_response(request_b, pass_name="B")
            boundary_responses = [response_b]
            annotation = _validate_model_stage(
                episode_id=planned.episode_id,
                quality_outcome=str(plan["quality_outcome"]),
                domains=domains,
                episode_task=episode_task,
                payload=response_b.payload,
                allowed_boundaries=set(allowed_b),
            )
        pass_b_provenance: object
        if family.boundary_local:
            pass_b_provenance = {
                "mode": BOUNDARY_LOCAL_MODE,
                "boundaries": [
                    _boundary_provenance(transition, result, response)
                    for transition, result, response in zip(
                        boundary_transitions,
                        boundary_results,
                        boundary_responses,
                        strict=True,
                    )
                ],
            }
        else:
            pass_b_provenance = _provenance(boundary_responses[0])
        pass_a_provenance = _provenance(response_a)
        if family.boundary_local:
            # Preserve the global model interpretation before any boundary-local
            # updates so acceptance can distinguish recall from boundary changes.
            pass_a_provenance["coarse_output"] = coarse_payload
        annotation["model_provenance"] = {
            "provider": response_a.provider,
            "model": response_a.model,
            "pass_a_prompt_version": family.pass_a_version,
            "pass_b_prompt_version": family.pass_b_version,
            "prompt_family": family.name,
            "input_fingerprint": plan["input_fingerprint"],
            "pass_a_sampling_strategy": COARSE_SAMPLING_STRATEGY,
            "pass_a_selected_curated_indices": [
                int(record["curated_index"]) for record in records_a
            ],
            "pass_b_mode": (
                BOUNDARY_LOCAL_MODE if family.boundary_local else "consolidated"
            ),
            "pass_b_sampling_strategy": (
                BOUNDARY_LOCAL_SAMPLING_STRATEGY
                if family.boundary_local
                else BOUNDARY_SAMPLING_STRATEGY
            ),
            "pass_b_selected_curated_indices": sorted(
                {int(record["curated_index"]) for record in records_b}
            ),
            "pass_a": pass_a_provenance,
            "pass_b": pass_b_provenance,
            "actual_planned_provider_calls": actual_planned_calls,
        }
        if family.platform_context is not None:
            annotation["model_provenance"]["platform_context"] = (
                family.platform_context.as_dict()
            )
            annotation["model_provenance"]["platform_context_fingerprint"] = (
                family.platform_context.fingerprint
            )
        annotation["source_dataset_status"] = episode.metadata.get(
            "source_dataset_status"
        )
        if family.boundary_local:
            annotation["diagnostics"] = _annotation_diagnostics(
                annotation,
                boundary_results=boundary_results,
                rejected_paraphrases=rejected_paraphrases,
            )
        validate_hierarchical_annotation(annotation)
        keyframes = {
            "schema_name": KEYFRAMES_SCHEMA_NAME,
            "schema_version": 2,
            "episode_id": planned.episode_id,
            "pass_a": {
                "prompt_version": family.pass_a_version,
                "allowed_boundary_indices": allowed_a,
                "frames": records_a,
            },
            "pass_b": (
                {
                    "prompt_version": family.pass_b_version,
                    "mode": BOUNDARY_LOCAL_MODE,
                    "boundaries": boundary_keyframes,
                }
                if family.boundary_local
                else {
                    "prompt_version": family.pass_b_version,
                    "allowed_boundary_indices": allowed_b,
                    "frames": records_b,
                }
            ),
        }
        _publish(Path(output_root), planned.episode_id, annotation, keyframes)
        diagnostics = annotation.get("diagnostics", {})
        return HierarchicalResult(
            planned.episode_id,
            "SUCCESS",
            planned.quality_outcome,
            len(domains),
            len(records_a),
            len(records_b),
            actual_planned_calls,
            planned.destination,
            planned.prompt_versions,
            semantic_segments=len(annotation["semantic_segments"]),
            pass_a_indices=tuple(int(record["curated_index"]) for record in records_a),
            planned_provider_calls=actual_planned_calls,
            minimum_planned_provider_calls=1 if family.boundary_local else 2,
            pass_b_mode=(
                BOUNDARY_LOCAL_MODE if family.boundary_local else "consolidated"
            ),
            pass_a_calls=1,
            pass_b_calls=(len(boundary_transitions) if family.boundary_local else 1),
            actual_planned_provider_calls=actual_planned_calls,
            diagnostic_codes=tuple(diagnostics.get("diagnostic_codes", [])),
            annotation_status=ANNOTATION_STATUS_SUCCESS,
            semantic_evaluated=True,
            platform_context_fingerprint=plan["platform_context_fingerprint"],
            **_episode_call_accounting(
                provider,
                counter_before,
                episode_failed=False,
                logical_provider_calls=logical_provider_calls,
                actual_provider_calls=actual_provider_calls,
                replayed_provider_responses=replayed_provider_responses,
            ),
        )
    except (OSError, ValueError, TypeError, KeyError, ProviderError) as exc:
        provider_fields = exc.details() if isinstance(exc, ProviderError) else {}
        failure_metadata = (
            getattr(provider, "_last_failure_metadata", None)
            if isinstance(exc, ProviderError)
            else None
        )
        failure_status = _failure_annotation_status(exc)
        result = HierarchicalResult(
            planned.episode_id,
            "FAILED",
            planned.quality_outcome,
            len(domains),
            len(records_a),
            0,
            actual_planned_calls,
            planned.destination,
            planned.prompt_versions,
            message=str(exc),
            http_status=provider_fields.get("http_status"),
            provider_error_code=provider_fields.get("provider_error_code"),
            provider_error_message=provider_fields.get("provider_error_message"),
            provider_request_id=provider_fields.get("request_id"),
            planned_provider_calls=actual_planned_calls,
            minimum_planned_provider_calls=1 if family.boundary_local else 2,
            provider_metadata=dict(failure_metadata) if failure_metadata else None,
            pass_b_mode=(
                BOUNDARY_LOCAL_MODE if family.boundary_local else "consolidated"
            ),
            pass_a_calls=1,
            pass_b_calls=(len(boundary_transitions) if family.boundary_local else 1),
            actual_planned_provider_calls=actual_planned_calls,
            annotation_status=failure_status,
            failure_reason=(
                exc.category if isinstance(exc, ProviderError) else type(exc).__name__
            ),
            semantic_evaluated=False,
            pass_a_indices=tuple(int(record["curated_index"]) for record in records_a),
            platform_context_fingerprint=plan["platform_context_fingerprint"],
            **_episode_call_accounting(
                provider,
                counter_before,
                episode_failed=True,
                logical_provider_calls=logical_provider_calls,
                actual_provider_calls=actual_provider_calls,
                replayed_provider_responses=replayed_provider_responses,
            ),
        )
        _write_failure_record(Path(output_root), result)
        return result


def _publish(
    output_root: Path,
    episode_id: str,
    annotation: dict[str, Any],
    keyframes: dict[str, Any],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{episode_id}.hierarchical-", dir=output_root)
    )
    try:
        episode = staging / episode_id
        episode.mkdir()
        # Successful provider responses are persisted before local validation
        # under the live episode directory. Carry them into the atomically
        # published final episode instead of discarding expensive evidence.
        raw_source = output_root / episode_id / "provider_raw"
        if raw_source.is_dir():
            shutil.copytree(raw_source, episode / "provider_raw")
        write_annotation(episode / "annotation.json", annotation)
        (episode / "keyframes.json").write_text(
            json.dumps(keyframes, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        target = output_root / episode_id
        backup = staging / "backup"
        if target.exists():
            target.rename(backup)
        try:
            os.replace(episode, target)
        except OSError:
            if backup.exists() and not target.exists():
                backup.rename(target)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _write_failure_record(output_root: Path, result: HierarchicalResult) -> None:
    """Persist a non-semantic failure marker without fabricating annotation data."""

    episode_root = output_root / result.episode_id
    episode_root.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_name": "vla_annotation_attempt_result",
        "schema_version": 1,
        "episode_id": result.episode_id,
        "annotation_status": result.annotation_status,
        "failure_reason": result.failure_reason,
        "semantic_evaluated": False,
        "result": result.as_dict(),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".annotation_result.", suffix=".tmp", dir=episode_root
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, episode_root / "annotation_result.json")
    finally:
        temporary.unlink(missing_ok=True)


def annotate_hierarchical_dataset(
    curated_root: str | Path,
    quality_root: str | Path,
    output_root: str | Path,
    provider: StructuredVLMProvider | None,
    *,
    episode: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    concurrency: int = 1,
    max_coarse_points: int = DEFAULT_COARSE_POINTS,
    prompt_family: str = "v1",
    baseline_annotation_root: str | Path | None = None,
    platform_context: PlatformContext | None = None,
) -> HierarchicalDatasetResult:
    started = time.monotonic()
    family = get_prompt_family(prompt_family, platform_context)
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    curated_path = Path(curated_root).resolve()
    quality_path = Path(quality_root).resolve()
    output_path = Path(output_root).resolve()
    baseline_path = (
        Path(baseline_annotation_root).resolve()
        if baseline_annotation_root is not None
        else None
    )
    if family.boundary_local and baseline_path is None:
        raise ValueError(
            "boundary-local prompt family requires --baseline-annotation-root "
            "for comparison"
        )
    if baseline_path is not None:
        if (
            baseline_path == output_path
            or baseline_path in output_path.parents
            or output_path in baseline_path.parents
        ):
            raise ValueError(
                "baseline annotation root and output root must be non-overlapping"
            )
        if not baseline_path.is_dir():
            raise ValueError(
                f"baseline annotation root is not a directory: {baseline_path}"
            )
    if any(
        output_path == source
        or output_path in source.parents
        or source in output_path.parents
        for source in (curated_path, quality_path)
    ):
        raise ValueError(
            "annotation output must be separate from Curated and quality inputs"
        )
    discovery = discover_curated_episodes(curated_path, episode=episode)
    if discovery.issues:
        raise ValueError("noncanonical Curated episode directory")
    results: list[HierarchicalResult] = []

    def process(item) -> HierarchicalResult:
        destination = Path(output_root) / item.episode_id
        if not force and (destination / "annotation.json").is_file():
            try:
                from vla_data.annotation.schema import load_annotation

                existing = load_annotation(destination / "annotation.json")
                provenance = existing.get("model_provenance", {})
                planned, current_plan = plan_hierarchical_episode(
                    item.episode_path,
                    quality_root=quality_root,
                    output_root=output_root,
                    max_coarse_points=max_coarse_points,
                    prompt_versions=(family.pass_a_version, family.pass_b_version),
                    boundary_local=family.boundary_local,
                    platform_context=family.platform_context,
                )
                if (
                    existing.get("schema_version") == 2
                    and current_plan is not None
                    and existing["clean_domains"] == current_plan["clean_domains"]
                    and existing["quality_outcome"] == current_plan["quality_outcome"]
                    and provenance.get("input_fingerprint")
                    == current_plan["input_fingerprint"]
                    and provenance.get("pass_a_prompt_version") == family.pass_a_version
                    and provenance.get("pass_b_prompt_version") == family.pass_b_version
                    and provenance.get("prompt_family") == family.name
                    and provenance.get("pass_b_mode")
                    == (
                        BOUNDARY_LOCAL_MODE if family.boundary_local else "consolidated"
                    )
                    and provenance.get("platform_context_fingerprint")
                    == (
                        family.platform_context.fingerprint
                        if family.platform_context is not None
                        else None
                    )
                    and (destination / "keyframes.json").is_file()
                    and (
                        provider is None
                        or provenance.get("provider") == provider.provider_name
                    )
                    and (provider is None or provenance.get("model") == provider.model)
                ):
                    return HierarchicalResult(
                        **{
                            **planned.__dict__,
                            "status": "WOULD_SKIP" if dry_run else "SKIPPED",
                            "expected_glm_calls": 0,
                            "message": "valid matching annotation v2 already exists",
                            "annotation_status": ANNOTATION_STATUS_SUCCESS,
                            "semantic_evaluated": True,
                            "logical_provider_calls": 0,
                            "actual_provider_calls": 0,
                            "replayed_provider_responses": 0,
                        }
                    )
            except (OSError, ValueError, TypeError, KeyError):
                pass
        if dry_run:
            result, _ = plan_hierarchical_episode(
                item.episode_path,
                quality_root=quality_root,
                output_root=output_root,
                max_coarse_points=max_coarse_points,
                prompt_versions=(family.pass_a_version, family.pass_b_version),
                boundary_local=family.boundary_local,
                platform_context=family.platform_context,
            )
        else:
            if provider is None:
                raise ValueError("provider is required outside dry-run")
            result = annotate_hierarchical_episode(
                item.episode_path,
                quality_root=quality_root,
                output_root=output_root,
                provider=provider,
                max_coarse_points=max_coarse_points,
                prompt_family=prompt_family,
                platform_context=family.platform_context,
            )
        return result

    if concurrency == 1 or len(discovery.episodes) < 2:
        results = [process(item) for item in discovery.episodes]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(process, discovery.episodes))
    results.sort(key=lambda result: result.episode_id)
    attempted_or_planned = sum(
        result.status in {"SUCCESS", "FAILED", "WOULD_PROCESS"} for result in results
    )
    summary = {
        "schema_name": "vla_hierarchical_annotation_summary",
        "schema_version": 2,
        "episodes_discovered": len(results),
        "episodes_processed": sum(
            result.status in {"SUCCESS", "FAILED"} for result in results
        ),
        "episodes_skipped": sum(
            result.status in {"SKIPPED", "WOULD_SKIP"} for result in results
        ),
        "episodes_ineligible": sum(result.status == "INELIGIBLE" for result in results),
        "episodes_failed": sum(result.status == "FAILED" for result in results),
        "expected_glm_calls": sum(result.expected_glm_calls for result in results),
        "prompt_family": family.name,
        "prompt_versions": [family.pass_a_version, family.pass_b_version],
        "pass_b_mode": (
            BOUNDARY_LOCAL_MODE if family.boundary_local else "consolidated"
        ),
        "pass_a_calls": attempted_or_planned,
        "pass_b_calls": (
            sum(
                result.pass_b_calls
                for result in results
                if isinstance(result.pass_b_calls, int)
            )
            if not dry_run
            else "dynamic_after_pass_a"
        ),
        "minimum_planned_provider_calls": (1 if family.boundary_local else 2)
        * attempted_or_planned,
        "actual_planned_provider_calls": (
            sum(result.actual_planned_provider_calls or 0 for result in results)
            if not dry_run
            else None
        ),
        "planned_provider_calls": sum(
            result.planned_provider_calls or 0 for result in results
        ),
        "completed_provider_calls": sum(
            result.completed_provider_calls or 0 for result in results
        ),
        "failed_provider_calls": sum(
            result.failed_provider_calls or 0 for result in results
        ),
        "retry_count": sum(result.retry_count or 0 for result in results),
        "logical_provider_calls": sum(
            result.logical_provider_calls or 0 for result in results
        ),
        "actual_provider_calls": sum(
            result.actual_provider_calls or 0 for result in results
        ),
        "replayed_provider_responses": sum(
            result.replayed_provider_responses or 0 for result in results
        ),
        "actual_tools_call_attempts": sum(
            result.actual_tools_call_attempts or 0 for result in results
        ),
        "annotation_status_counts": {
            status: sum(result.annotation_status == status for result in results)
            for status in (
                ANNOTATION_STATUS_SUCCESS,
                ANNOTATION_STATUS_PROVIDER_FAILED,
                ANNOTATION_STATUS_VALIDATION_FAILED,
                ANNOTATION_STATUS_MISSING,
            )
        },
        "platform_context": (
            family.platform_context.as_dict()
            if family.platform_context is not None
            else None
        ),
        "platform_context_fingerprint": (
            family.platform_context.fingerprint
            if family.platform_context is not None
            else None
        ),
        "provider": getattr(provider, "provider_name", None),
        "results": [result.as_dict() for result in results],
    }
    if not dry_run:
        if family.boundary_local and baseline_path is not None:
            from vla_data.annotation.comparison import write_semantic_comparison

            comparison_json, comparison_csv = write_semantic_comparison(
                baseline_path,
                output_path,
                results={result.episode_id: result.as_dict() for result in results},
            )
            summary["semantic_comparison_json"] = str(comparison_json)
            summary["semantic_comparison_csv"] = str(comparison_csv)
        write_summary(output_path / "dataset_annotation_summary.json", summary)
    return HierarchicalDatasetResult(
        tuple(results), summary, time.monotonic() - started
    )
