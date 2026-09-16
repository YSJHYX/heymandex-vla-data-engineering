"""Atomic, secret-minimized persistence and replay of provider responses."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from vla_data.annotation.provider import (
    AnnotationRequest,
    ImageItem,
    StructuredModelResponse,
    TextItem,
)
from vla_data.batch.summary import write_summary

RAW_RESPONSE_SCHEMA_NAME = "vla_provider_raw_response"
RAW_RESPONSE_SCHEMA_VERSION = 1

# Provider metadata is never copied wholesale. This allowlist contains only
# operational provenance emitted by the in-tree providers, never credentials,
# request headers, or environment values.
_SAFE_PROVIDER_METADATA_KEYS = frozenset(
    {
        "mcp_package",
        "mcp_transport",
        "mcp_selected_tool",
        "mcp_tool_selection_reason",
        "available_tool_names",
        "mcp_server_name",
        "mcp_server_version",
        "native_multi_image_or_storyboard",
        "response_model",
        "backend_model_not_exposed_by_mcp",
        "image_count",
        "storyboard_generated",
        "storyboard_observation_count",
        "storyboard_final_bytes",
        "storyboard_target_max_bytes",
        "storyboard_jpeg_quality",
        "storyboard_scale",
        "storyboard_fit_attempts",
        "wall_time_s",
        "logical_calls",
        "actual_tools_call_attempts",
        "retry_count",
        "tools_call_count",
        "mcp_tools_call_attempted",
    }
)


def request_fingerprint(request: AnnotationRequest) -> str:
    """Hash the exact prompt and image evidence supplied to one logical call."""

    items: list[dict[str, object]] = []
    for item in request.items:
        if isinstance(item, TextItem):
            items.append({"type": "text", "text": item.text})
        elif isinstance(item, ImageItem):
            items.append(
                {
                    "type": "image",
                    "temporal_point": item.temporal_point,
                    "camera": item.camera,
                    "label": item.label,
                    "source_path": str(item.source_path.resolve()),
                    "sha256": hashlib.sha256(item.source_path.read_bytes()).hexdigest(),
                }
            )
        else:  # pragma: no cover - closed dataclass union
            raise TypeError(f"unsupported annotation request item: {type(item)!r}")
    document = {
        "episode_id": request.episode_id,
        "prompt_version": request.prompt_version,
        "transport_id": request.transport_id,
        "items": items,
    }
    encoded = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def response_artifact_path(
    output_root: str | Path,
    episode_id: str,
    *,
    pass_name: str,
    transition_id: str | None = None,
) -> Path:
    """Return an unambiguous per-pass artifact path."""

    raw_root = Path(output_root) / episode_id / "provider_raw"
    if pass_name == "A" and transition_id is None:
        return raw_root / "pass_a_response.json"
    if pass_name == "B" and transition_id is not None:
        return raw_root / f"pass_b_transition_{transition_id}.json"
    if pass_name == "B":
        return raw_root / "pass_b_response.json"
    raise ValueError("pass_name must be A or B")


def persist_provider_response(
    path: str | Path,
    response: StructuredModelResponse,
    request: AnnotationRequest,
    *,
    pass_name: str,
    transition_id: str | None,
    input_fingerprint: str,
    prompt_family: str,
    platform_context_fingerprint: str | None,
    pass_b_mode: str,
    logical_provider_call_index: int,
) -> Path:
    """Persist a successful structured response before local validation."""

    safe_metadata = {
        key: value
        for key, value in (response.provider_metadata or {}).items()
        if key in _SAFE_PROVIDER_METADATA_KEYS
    }
    raw_payload = response.payload
    payload_encoded = json.dumps(
        raw_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    actual_attempts = safe_metadata.get("actual_tools_call_attempts")
    if not isinstance(actual_attempts, int):
        actual_attempts = None
    document = {
        "schema_name": RAW_RESPONSE_SCHEMA_NAME,
        "schema_version": RAW_RESPONSE_SCHEMA_VERSION,
        "episode_id": request.episode_id,
        "pass": pass_name,
        "transition_id": transition_id,
        "provider": response.provider,
        "model": response.model,
        "prompt_family": prompt_family,
        "prompt_version": response.prompt_version,
        "request_fingerprint": request_fingerprint(request),
        "input_fingerprint": input_fingerprint,
        "platform_context_fingerprint": platform_context_fingerprint,
        "pass_b_mode": pass_b_mode,
        "persisted_at_utc": datetime.now(UTC).isoformat(),
        "logical_provider_call_index": logical_provider_call_index,
        "actual_tools_call_attempt_count": actual_attempts,
        "provider_provenance": {
            "request_id": response.request_id,
            "attempt_count": response.attempt_count,
            "finish_reason": response.finish_reason,
            "usage": response.usage,
            "safe_provider_metadata": safe_metadata,
        },
        "raw_payload_sha256": hashlib.sha256(payload_encoded).hexdigest(),
        "raw_structured_payload": raw_payload,
    }
    return write_summary(path, document)


def load_compatible_provider_response(
    path: str | Path,
    request: AnnotationRequest,
    *,
    pass_name: str,
    transition_id: str | None,
    input_fingerprint: str,
    prompt_family: str,
    provider_identity: str,
    platform_context_fingerprint: str | None,
    pass_b_mode: str,
) -> StructuredModelResponse | None:
    """Load only an exact-contract replay; malformed/stale artifacts miss."""

    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    expected = {
        "schema_name": RAW_RESPONSE_SCHEMA_NAME,
        "schema_version": RAW_RESPONSE_SCHEMA_VERSION,
        "episode_id": request.episode_id,
        "pass": pass_name,
        "transition_id": transition_id,
        "provider": provider_identity,
        "prompt_family": prompt_family,
        "prompt_version": request.prompt_version,
        "request_fingerprint": request_fingerprint(request),
        "input_fingerprint": input_fingerprint,
        "platform_context_fingerprint": platform_context_fingerprint,
        "pass_b_mode": pass_b_mode,
    }
    if not isinstance(document, dict) or any(
        document.get(key) != value for key, value in expected.items()
    ):
        return None
    payload = document.get("raw_structured_payload")
    if not isinstance(payload, dict):
        return None
    payload_encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    if (
        document.get("raw_payload_sha256")
        != hashlib.sha256(payload_encoded).hexdigest()
    ):
        return None
    provenance = document.get("provider_provenance")
    if not isinstance(provenance, dict):
        return None
    metadata = provenance.get("safe_provider_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = {**metadata, "raw_response_replayed": True}
    attempt_count = provenance.get("attempt_count")
    if isinstance(attempt_count, bool) or not isinstance(attempt_count, int):
        attempt_count = 1
    usage = provenance.get("usage")
    if not isinstance(usage, dict):
        usage = None
    return StructuredModelResponse(
        provider=str(document["provider"]),
        model=str(document.get("model") or "unknown"),
        prompt_version=str(document["prompt_version"]),
        payload=payload,
        request_id=(
            provenance.get("request_id")
            if isinstance(provenance.get("request_id"), str)
            else None
        ),
        attempt_count=attempt_count,
        finish_reason=(
            provenance.get("finish_reason")
            if isinstance(provenance.get("finish_reason"), str)
            else None
        ),
        usage=usage,
        provider_metadata=metadata,
    )
