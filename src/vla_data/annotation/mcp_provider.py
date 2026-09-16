"""Coding Plan Vision MCP provider (@z_ai/mcp-server) over stdio JSON-RPC.

The hierarchical annotation layer only sees the StructuredVLMProvider
protocol; every MCP transport detail (subprocess lifecycle, handshake,
tool-schema discovery, storyboard fallback) lives here.

Credential rule: this provider ONLY reads Z_AI_API_KEY (team Coding Plan
identity). It never falls back to GLM_API_KEY / ZHIPU_API_KEY (personal
PaaS identities).
"""

from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from vla_data.annotation.provider import (
    ERROR_INVALID_JSON,
    ERROR_MODEL_OUTPUT_SCHEMA,
    AnnotationRequest,
    ImageItem,
    ProviderError,
    StructuredModelResponse,
    TextItem,
)
from vla_data.annotation.storyboard import (
    DEFAULT_JPEG_QUALITY,
    JPEG_QUALITY_CANDIDATES,
    RESOLUTION_SCALE_CANDIDATES,
    STORYBOARD_BUILD_ERROR,
    STORYBOARD_SIZE_LIMIT,
    StoryboardArtifact,
    fit_storyboard_to_budget,
    observations_from_request,
)

PROVIDER_NAME = "coding_plan_vision_mcp"
MCP_PACKAGE = "@z_ai/mcp-server"
# Official docs and the published npm package currently disagree on the
# general-purpose vision tool name; runtime tools/list is authoritative.
# Docs names come first only because they are the documented spelling.
GENERAL_IMAGE_TOOL_ALIASES = (
    "image_analysis",  # current official docs spelling
    "analyze_image",  # current @z_ai/mcp-server npm runtime spelling
)
GENERAL_VIDEO_TOOL_ALIASES = (
    "video_analysis",
    "analyze_video",
)
# Not robotics-annotation tools; recorded in diagnostics only.
NON_ANNOTATION_TOOL_HINTS = (
    "ui_to_artifact",
    "extract_text_from_screenshot",
    "diagnose_error_screenshot",
    "understand_technical_diagram",
    "analyze_data_visualization",
    "ui_diff_check",
)
JSONRPC_VERSION = "2.0"
MCP_PROTOCOL_VERSION = "2024-11-05"

MCP_STARTUP_ERROR = "MCP_STARTUP_ERROR"
MCP_INITIALIZE_ERROR = "MCP_INITIALIZE_ERROR"
MCP_TOOL_NOT_FOUND = "MCP_TOOL_NOT_FOUND"
MCP_TOOL_SCHEMA_ERROR = "MCP_TOOL_SCHEMA_ERROR"
MCP_TOOL_CALL_ERROR = "MCP_TOOL_CALL_ERROR"
MCP_TIMEOUT = "MCP_TIMEOUT"
MCP_PROCESS_EXITED = "MCP_PROCESS_EXITED"
MCP_INVALID_RESPONSE = "MCP_INVALID_RESPONSE"
# Backward-compatible public name; canonical validation now happens in the
# pipeline after raw-response persistence, never inside the transport provider.
MODEL_OUTPUT_SCHEMA_ERROR = ERROR_MODEL_OUTPUT_SCHEMA
MCP_IMAGE_SIZE_REJECTED = "MCP_IMAGE_SIZE_REJECTED"
# Local annotation input preparation failed BEFORE any MCP tools/call.
ANNOTATION_INPUT_ERROR = "ANNOTATION_INPUT_ERROR"

# Runtime-reported service limit and our deliberately lower upload target.
# The safety margin covers decimal/binary units, metadata, rounding, and
# provider-side accounting differences.
MAX_MCP_IMAGE_BYTES = 5_000_000
TARGET_MCP_IMAGE_BYTES = 4_500_000


class CodingPlanAPIKeyMissingError(RuntimeError):
    """Z_AI_API_KEY is absent; the personal PaaS key is never substituted."""


def resolve_z_ai_api_key(environment: dict[str, str] | None = None) -> str:
    """Read only Z_AI_API_KEY — strict Coding Plan credential isolation."""

    source = os.environ if environment is None else environment
    key = source.get("Z_AI_API_KEY", "")
    if not key.strip():
        raise CodingPlanAPIKeyMissingError(
            "CODING_PLAN_API_KEY_MISSING: set Z_AI_API_KEY (team Coding Plan "
            "identity); GLM_API_KEY/ZHIPU_API_KEY are never used as fallback"
        )
    return key.strip()


@dataclass(frozen=True)
class MCPProviderConfig:
    command: str = "npx"
    server_args: tuple[str, ...] = ("-y", "@z_ai/mcp-server@latest")
    z_ai_mode: str = "ZHIPU"
    startup_timeout_s: float = 120.0
    call_timeout_s: float = 300.0
    max_call_attempts: int = 2
    handshake_timeout_s: float = 60.0
    storyboard_root: str | Path | None = None
    storyboard_max_width: int = 640
    storyboard_target_max_bytes: int = TARGET_MCP_IMAGE_BYTES
    storyboard_default_jpeg_quality: int = DEFAULT_JPEG_QUALITY
    storyboard_quality_candidates: tuple[int, ...] = JPEG_QUALITY_CANDIDATES
    storyboard_scale_candidates: tuple[float, ...] = RESOLUTION_SCALE_CANDIDATES
    environment: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if not 0 < self.storyboard_target_max_bytes <= MAX_MCP_IMAGE_BYTES:
            raise ValueError(
                "storyboard_target_max_bytes must be positive and no greater "
                f"than MAX_MCP_IMAGE_BYTES={MAX_MCP_IMAGE_BYTES}"
            )
        if not 1 <= self.max_call_attempts <= 2:
            raise ValueError("max_call_attempts must be 1 or 2")


class _MCPClient:
    """Minimal stdio JSON-RPC 2.0 client for one long-lived MCP process."""

    def __init__(self, config: MCPProviderConfig, api_key: str) -> None:
        self._config = config
        self._api_key = api_key
        self._process: subprocess.Popen[str] | None = None
        self._next_id = 0

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        command = shutil.which(self._config.command)
        if command is None:
            raise ProviderError(
                MCP_STARTUP_ERROR,
                f"{self._config.command!r} not found on PATH "
                "(Node.js >= 18 with npx is required)",
            )
        # The key travels ONLY through the subprocess environment — never
        # through argv, logs, or JSON.
        environment = dict(os.environ)
        environment.update(self._config.environment or {})
        environment["Z_AI_API_KEY"] = self._api_key
        environment.setdefault("Z_AI_MODE", self._config.z_ai_mode)
        try:
            self._process = subprocess.Popen(
                [command, *self._config.server_args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise ProviderError(MCP_STARTUP_ERROR, str(exc)) from exc

    def initialize(self) -> dict[str, Any]:
        result = self._request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "vla-data", "version": "1.0"},
            },
            timeout_s=self._config.handshake_timeout_s,
            category=MCP_INITIALIZE_ERROR,
        )
        # Initialized notification (no response expected).
        self._send({"jsonrpc": JSONRPC_VERSION, "method": "notifications/initialized"})
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        result = self._request(
            "tools/list", None, timeout_s=self._config.handshake_timeout_s
        )
        tools = result.get("tools") if isinstance(result, dict) else None
        if not isinstance(tools, list):
            raise ProviderError(
                MCP_INVALID_RESPONSE, "tools/list returned no tool array"
            )
        return [tool for tool in tools if isinstance(tool, dict)]

    def call_tool(
        self, name: str, arguments: dict[str, Any], *, timeout_s: float | None = None
    ) -> dict[str, Any]:
        return self._request(
            "tools/call",
            {"name": name, "arguments": arguments},
            timeout_s=timeout_s or self._config.call_timeout_s,
            category=MCP_TOOL_CALL_ERROR,
        )

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            if process is not None:
                for stream in (process.stdin, process.stdout, process.stderr):
                    try:
                        stream.close()
                    except (OSError, AttributeError):
                        pass
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
        except OSError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for stream in (process.stdout, process.stderr):
            try:
                stream.close()
            except (OSError, AttributeError):
                pass

    # -------------------------------------------------------------- transport

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _send(self, payload: dict[str, Any]) -> None:
        process = self._require_process()
        try:
            assert process.stdin is not None
            process.stdin.write(json.dumps(payload) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise ProviderError(
                MCP_PROCESS_EXITED,
                f"MCP server pipe broken before response: {exc}",
            ) from exc

    def _request(
        self,
        method: str,
        params: dict[str, Any] | None,
        *,
        timeout_s: float,
        category: str = MCP_TOOL_CALL_ERROR,
    ) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        payload: dict[str, Any] = {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params
        self._send(payload)
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProviderError(
                    MCP_TIMEOUT,
                    f"{method} timed out after {timeout_s:.0f}s",
                )
            line = self._read_line(remaining)
            if line is None:
                continue
            message = _parse_jsonrpc_message(line)
            if message is None:
                raise ProviderError(
                    MCP_INVALID_RESPONSE,
                    f"{method}: server sent malformed JSON-RPC: {line[:120]!r}",
                )
            if message.get("id") != request_id:
                continue  # notification or unrelated traffic
            if "error" in message and message["error"] is not None:
                error = message["error"]
                raise ProviderError(
                    category,
                    f"{method} failed: {error.get('message', error)}",
                )
            result = message.get("result")
            if not isinstance(result, dict):
                raise ProviderError(
                    MCP_INVALID_RESPONSE, f"{method} returned no result object"
                )
            return result

    def _read_line(self, timeout_s: float) -> str | None:
        process = self._require_process()
        assert process.stdout is not None
        ready, _, _ = select.select([process.stdout], [], [], max(timeout_s, 0))
        if not ready:
            return None
        line = process.stdout.readline()
        if line == "":
            raise ProviderError(
                MCP_PROCESS_EXITED,
                "MCP server exited unexpectedly while awaiting a response",
            )
        return line

    def _require_process(self) -> subprocess.Popen[str]:
        if not self.alive:
            raise ProviderError(MCP_PROCESS_EXITED, "MCP server is not running")
        return self._process  # type: ignore[return-value]


def _parse_jsonrpc_message(line: str) -> dict[str, Any] | None:
    try:
        message = json.loads(line)
    except ValueError:
        return None
    return message if isinstance(message, dict) else None


def _select_general_image_tool(
    tools: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    """Pick the general-purpose image tool from the RUNTIME tool list.

    The docs spelling wins only when the runtime actually offers it; the npm
    runtime spelling is equally legitimate. Video aliases are deliberately
    not selected (video tools cannot replace precise boundary refinement).
    """

    for alias in GENERAL_IMAGE_TOOL_ALIASES:
        for tool in tools:
            if tool.get("name") == alias:
                reason = "runtime_alias_match" + (
                    " (docs spelling)"
                    if alias == "image_analysis"
                    else " (npm runtime spelling)"
                )
                return tool, reason
    return None, None


def _sanitize_server_info(initialize_result: dict[str, Any]) -> dict[str, str | None]:
    """Read serverInfo name/version when the runtime exposes them; else nulls."""

    info = initialize_result.get("serverInfo")
    if not isinstance(info, dict):
        return {"name": None, "version": None}
    name = info.get("name")
    version = info.get("version")
    return {
        "name": name if isinstance(name, str) and name else None,
        "version": version if isinstance(version, str) and version else None,
    }


def _extract_tool_text(result: dict[str, Any]) -> str:
    if result.get("isError"):
        content = result.get("content")
        blocks = content if isinstance(content, list) else []
        provider_text = " ".join(
            block.get("text", "")
            for block in blocks
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )
        safe_text = " ".join(provider_text.split())[:500]
        raise ProviderError(
            MCP_TOOL_CALL_ERROR,
            f"tool reported isError: {safe_text or 'no text detail'}",
            provider_error_message=safe_text or None,
        )
    content = result.get("content")
    if not isinstance(content, list):
        raise ProviderError(MCP_INVALID_RESPONSE, "tool result has no content list")
    texts = [
        block.get("text")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    texts = [text for text in texts if isinstance(text, str)]
    if not texts:
        raise ProviderError(MCP_INVALID_RESPONSE, "tool result contains no text block")
    return "\n".join(texts)


def _parse_model_json(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        first_newline = candidate.find("\n")
        candidate = candidate[first_newline + 1 :] if first_newline != -1 else ""
        if candidate.rstrip().endswith("```"):
            candidate = candidate.rstrip()[:-3]
    decoder = json.JSONDecoder()
    for offset in range(len(candidate)):
        if candidate[offset] != "{":
            continue
        try:
            value, _ = decoder.raw_decode(candidate, offset)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    raise ProviderError(ERROR_INVALID_JSON, "tool text contains no JSON object")


def _discover_image_argument(tool: dict[str, Any]) -> tuple[str, str, str | None]:
    """Map the runtime-declared image_analysis schema to a calling mode.

    Returns (argument_name, mode, prompt_argument_name_or_None) where mode
    is "paths" (array of local paths), "path" (single path → storyboard
    fallback), or "base64" / "base64_list". Never guesses undocumented
    shapes: an unusable schema raises MCP_TOOL_SCHEMA_ERROR.
    """

    schema = tool.get("inputSchema")
    if not isinstance(schema, dict):
        raise ProviderError(
            MCP_TOOL_SCHEMA_ERROR, "image_analysis has no inputSchema object"
        )
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise ProviderError(
            MCP_TOOL_SCHEMA_ERROR, "image_analysis has no properties object"
        )
    image_argument: tuple[str, str] | None = None
    prompt_argument: str | None = None
    for name, spec in properties.items():
        if not isinstance(spec, dict):
            continue
        lowered = name.lower()
        kind = spec.get("type")
        if (
            prompt_argument is None
            and kind == "string"
            and any(
                word
                for word in ("prompt", "question", "instruction", "query", "text")
                if word in lowered
            )
        ):
            prompt_argument = name
    for name, spec in properties.items():
        if not isinstance(spec, dict):
            continue
        lowered = name.lower()
        kind = spec.get("type")
        if not (
            "image" in lowered
            or "photo" in lowered
            or "picture" in lowered
            or lowered in ("path", "file")
        ):
            continue
        if (
            kind == "array"
            and isinstance(spec.get("items"), dict)
            and spec["items"].get("type") in ("string", None)
        ):
            image_argument = (name, "paths")
            break
        if kind == "string":
            if "base64" in lowered or spec.get("format") == "base64":
                image_argument = (name, "base64")
                break
            image_argument = (name, "path")
            break
    if image_argument is None:
        raise ProviderError(
            MCP_TOOL_SCHEMA_ERROR,
            "image_analysis schema exposes no image input property",
        )
    return image_argument[0], image_argument[1], prompt_argument


class CodingPlanVisionMCPProvider:
    """StructuredVLMProvider backed by the team Coding Plan Vision MCP."""

    def __init__(
        self,
        config: MCPProviderConfig | None = None,
        *,
        client: _MCPClient | None = None,
        api_key: str | None = None,
    ) -> None:
        self.config = config or MCPProviderConfig()
        self.provider_name = PROVIDER_NAME
        # Backend model is managed by the Coding Plan service; only a
        # runtime-reported identifier fills response_model in metadata.
        self.model = "coding-plan-vision-mcp (server-managed backend)"
        self._api_key = api_key
        self._client = client
        self._owns_client = client is None
        self._tool: dict[str, Any] | None = None
        self._tool_selection_reason: str | None = None
        self._runtime_tool_names: list[str] | None = None
        self._server_info: dict[str, str | None] | None = None
        self._session_initialized = False
        self._argument_plan: tuple[str, str, str | None] | None = None
        self._last_board: Path | None = None
        self._last_storyboard_metadata: dict[str, object] | None = None
        self.storyboard_dir: Path | None = None
        self.transport_mode: str | None = None
        # Section-73 observability counters (never "expected = 0" on failure).
        self.planned_provider_calls = 0
        self.completed_provider_calls = 0
        self.failed_provider_calls = 0
        # Incremented only around a real MCP tools/call round trip.
        self.mcp_tools_call_attempted = 0
        self.retry_count = 0
        self._last_failure_metadata: dict[str, object] | None = None

    # ------------------------------------------------------------- protocol

    def infer_json(self, request: AnnotationRequest) -> StructuredModelResponse:
        """Run one logical provider call with at most one transient retry."""

        self.planned_provider_calls += 1
        self._last_failure_metadata = None
        attempts = 0
        tool_attempts_before = self.mcp_tools_call_attempted
        while attempts < self.config.max_call_attempts:
            attempts += 1
            try:
                response = self._infer_json_inner(request)
            except ProviderError as error:
                if attempts < self.config.max_call_attempts and _is_retryable_mcp_error(
                    error
                ):
                    self.retry_count += 1
                    self._reset_owned_session_for_retry()
                    continue
                self.failed_provider_calls += 1
                self._last_failure_metadata = self._failure_metadata(
                    error,
                    actual_tools_call_attempts=(
                        self.mcp_tools_call_attempted - tool_attempts_before
                    ),
                    retry_count=attempts - 1,
                )
                raise
            except Exception as exc:
                self.failed_provider_calls += 1
                wrapped = ProviderError(ANNOTATION_INPUT_ERROR, str(exc))
                self._last_failure_metadata = self._failure_metadata(
                    wrapped,
                    actual_tools_call_attempts=(
                        self.mcp_tools_call_attempted - tool_attempts_before
                    ),
                    retry_count=attempts - 1,
                )
                raise wrapped from exc
            metadata = dict(response.provider_metadata or {})
            metadata.update(
                {
                    "logical_calls": 1,
                    "actual_tools_call_attempts": (
                        self.mcp_tools_call_attempted - tool_attempts_before
                    ),
                    "retry_count": attempts - 1,
                }
            )
            self.completed_provider_calls += 1
            self._last_failure_metadata = None
            return replace(
                response,
                attempt_count=attempts,
                provider_metadata=metadata,
            )
        raise AssertionError("bounded MCP attempt loop exited unexpectedly")

    def _failure_metadata(
        self,
        error: ProviderError,
        *,
        actual_tools_call_attempts: int,
        retry_count: int,
    ) -> dict[str, object]:
        """Sanitized session facts preserved on failure (never secrets)."""

        metadata: dict[str, object] = {
            "provider": PROVIDER_NAME,
            "error_category": error.category,
            "mcp_server_name": (self._server_info or {}).get("name"),
            "mcp_server_version": (self._server_info or {}).get("version"),
            "mcp_selected_tool": (
                str(self._tool.get("name")) if self._tool is not None else None
            ),
            "mcp_tool_selection_reason": self._tool_selection_reason,
            "available_tool_names": list(self._runtime_tool_names or []),
            "image_input_mode": self._image_input_mode,
            "logical_calls": 1,
            "actual_tools_call_attempts": actual_tools_call_attempts,
            "retry_count": retry_count,
            "tools_call_count": actual_tools_call_attempts,
            "mcp_tools_call_attempted": actual_tools_call_attempts,
        }
        if self._last_storyboard_metadata is not None:
            metadata.update(self._last_storyboard_metadata)
        return metadata

    def _reset_owned_session_for_retry(self) -> None:
        """Restart an owned timed-out transport; injected test clients are reused."""

        if not self._owns_client:
            return
        if self._client is not None:
            self._client.close()
        self._client = None
        self._tool = None
        self._tool_selection_reason = None
        self._runtime_tool_names = None
        self._server_info = None
        self._session_initialized = False
        self._argument_plan = None

    @property
    def _image_input_mode(self) -> str | None:
        if self._argument_plan is None:
            return None
        return (
            "native-multi-image"
            if self._argument_plan[1] == "paths"
            else "single-image (storyboard)"
        )

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()

    # ------------------------------------------------------------------ impl

    def _infer_json_inner(self, request: AnnotationRequest) -> StructuredModelResponse:
        started = time.monotonic()
        client = self._ensure_session()
        argument_name, mode, prompt_argument = self._require_plan()
        images = [item for item in request.items if isinstance(item, ImageItem)]
        prompt_text = "\n".join(
            item.text for item in request.items if isinstance(item, TextItem)
        )
        if not images:
            raise ProviderError(
                MCP_TOOL_CALL_ERROR, "annotation request contains no images"
            )

        arguments: dict[str, Any] = {}
        used_storyboard = False
        if mode == "paths":
            arguments[argument_name] = [str(item.source_path) for item in images]
            self.transport_mode = "native_multi_image"
        elif mode == "path":
            artifact = self._render_board(request)
            arguments[argument_name] = str(artifact.path)
            used_storyboard = True
            self.transport_mode = "annotation_storyboard"
        else:  # single base64 image
            artifact = self._render_board(request)
            arguments[argument_name] = _file_base64(artifact.path)
            used_storyboard = True
            self.transport_mode = "annotation_storyboard_base64"
        if prompt_argument is not None:
            arguments[prompt_argument] = prompt_text

        self.mcp_tools_call_attempted += 1
        try:
            result = client.call_tool(self._require_selected_tool_name(), arguments)
            text = _extract_tool_text(result)
        except ProviderError as error:
            if _is_remote_image_size_rejection(error):
                raise ProviderError(
                    MCP_IMAGE_SIZE_REJECTED,
                    "provider rejected a locally preflighted storyboard as too "
                    f"large: {_safe_provider_message(error)}",
                    provider_error_message=_safe_provider_message(error),
                ) from error
            raise
        payload = _parse_model_json(text)
        response_model = _string_field(result, "model")
        return StructuredModelResponse(
            provider=PROVIDER_NAME,
            model=response_model or "coding-plan-vision-mcp (server-managed backend)",
            prompt_version=request.prompt_version,
            payload=payload,
            request_id=_string_field(result, "request_id")
            or _string_field(result, "requestId"),
            attempt_count=1,
            finish_reason="stop",
            usage=None,
            provider_metadata={
                "mcp_package": MCP_PACKAGE,
                "mcp_transport": "stdio",
                "mcp_selected_tool": self._require_selected_tool_name(),
                "mcp_tool_selection_reason": self._tool_selection_reason,
                "available_tool_names": list(self._runtime_tool_names or []),
                "mcp_server_name": (self._server_info or {}).get("name"),
                "mcp_server_version": (self._server_info or {}).get("version"),
                "native_multi_image_or_storyboard": self.transport_mode,
                "response_model": response_model,
                "backend_model_not_exposed_by_mcp": response_model is None,
                "image_count": len(images),
                "storyboard_generated": used_storyboard,
                "storyboard_path": (str(self._last_board) if used_storyboard else None),
                "wall_time_s": round(time.monotonic() - started, 3),
                **(self._last_storyboard_metadata or {}),
            },
        )

    def _ensure_session(self) -> _MCPClient:
        if self._client is not None and self._client.alive and self._tool is not None:
            return self._client
        if self._client is None:
            api_key = self._api_key
            if api_key is None:
                api_key = resolve_z_ai_api_key(self.config.environment)
            self._client = _MCPClient(self.config, api_key)
            self._client.start()
        if not self._session_initialized:
            initialize_result = self._client.initialize()
            self._server_info = _sanitize_server_info(initialize_result)
            self._session_initialized = True
        tools = self._client.list_tools()
        self._runtime_tool_names = [str(tool.get("name")) for tool in tools]
        tool, reason = _select_general_image_tool(tools)
        if tool is None:
            raise ProviderError(
                MCP_TOOL_NOT_FOUND,
                "no compatible general-purpose image tool offered by the MCP "
                f"server; known aliases tried: {list(GENERAL_IMAGE_TOOL_ALIASES)} "
                f"(runtime tools: {self._runtime_tool_names})",
            )
        self._tool_selection_reason = reason
        # A name match alone is not enough: the runtime schema must still
        # express an image source (plus prompt field) or this is SCHEMA_ERROR.
        self._argument_plan = _discover_image_argument(tool)
        self._tool = tool
        return self._client

    def _require_selected_tool_name(self) -> str:
        if self._tool is None:
            self._ensure_session()
        assert self._tool is not None
        return str(self._tool.get("name"))

    def probe(self) -> dict[str, object]:
        """Zero-inference diagnostic: handshake + tools/list + selection only.

        Never issues tools/call, so no billable vision request happens.
        """

        self._ensure_session()
        _, mode, prompt_argument = self._require_plan()
        return {
            "provider": PROVIDER_NAME,
            "mcp_package": MCP_PACKAGE,
            "mcp_server_name": (self._server_info or {}).get("name"),
            "mcp_server_version": (self._server_info or {}).get("version"),
            "mcp_selected_tool": self._require_selected_tool_name(),
            "mcp_tool_selection_reason": self._tool_selection_reason,
            "available_tool_names": list(self._runtime_tool_names or []),
            "image_input_mode": (
                "native-multi-image" if mode == "paths" else "single-image (storyboard)"
            ),
            "prompt_field": prompt_argument,
            "image_field": self._argument_plan[0] if self._argument_plan else None,
            "tools_call_count": 0,
            "ready": True,
        }

    def _require_plan(self) -> tuple[str, str, str | None]:
        if self._argument_plan is None:
            self._ensure_session()
        assert self._argument_plan is not None
        return self._argument_plan

    def _render_board(self, request: AnnotationRequest) -> StoryboardArtifact:
        try:
            observations = observations_from_request(request)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                ANNOTATION_INPUT_ERROR,
                f"failed to build typed dual-view observations: {exc}",
            ) from exc
        if self.storyboard_dir is None:
            base = (
                Path(self.config.storyboard_root)
                if self.config.storyboard_root
                else Path(tempfile.mkdtemp(prefix="vla_storyboards_"))
            )
            self.storyboard_dir = base
            self.storyboard_dir.mkdir(parents=True, exist_ok=True)
        try:
            artifact = fit_storyboard_to_budget(
                observations,
                self.storyboard_dir,
                target_max_bytes=self.config.storyboard_target_max_bytes,
                episode_id=request.episode_id,
                storyboard_pass=(
                    request.transport_id or _storyboard_pass(request.prompt_version)
                ),
                max_width=self.config.storyboard_max_width,
                default_quality=self.config.storyboard_default_jpeg_quality,
                quality_candidates=self.config.storyboard_quality_candidates,
                scale_candidates=self.config.storyboard_scale_candidates,
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                STORYBOARD_BUILD_ERROR, f"storyboard rendering failed: {exc}"
            ) from exc
        self._last_board = artifact.path
        self._last_storyboard_metadata = dict(artifact.transport_metadata)
        if not artifact.within_budget:
            final_bytes = artifact.transport_metadata["storyboard_final_bytes"]
            target = artifact.transport_metadata["storyboard_target_max_bytes"]
            raise ProviderError(
                STORYBOARD_SIZE_LIMIT,
                "storyboard remains over the local upload budget after all "
                f"allowed fitting attempts: final_bytes={final_bytes}, "
                f"target_max_bytes={target}",
            )
        return artifact


def _file_base64(path: Path) -> str:
    import base64

    return base64.b64encode(path.read_bytes()).decode("ascii")


def _storyboard_pass(prompt_version: str) -> str:
    lowered = prompt_version.lower()
    if "boundary" in lowered or "refine" in lowered:
        return "pass_b"
    if "coarse" in lowered:
        return "pass_a"
    return "storyboard"


def _is_remote_image_size_rejection(error: ProviderError) -> bool:
    message = " ".join(
        part
        for part in (error.message, error.provider_error_message)
        if isinstance(part, str)
    ).lower()
    return "image file too large" in message or (
        "image" in message and "maximum allowed" in message
    )


def _is_retryable_mcp_error(error: ProviderError) -> bool:
    """Retry only bounded, plausibly transient transport/service failures."""

    if error.category in {
        MCP_TIMEOUT,
        MCP_PROCESS_EXITED,
        "NETWORK_ERROR",
        "HTTP_RATE_LIMIT",
        "HTTP_SERVER_ERROR",
        "HTTP_PROVIDER_OVERLOAD",
    }:
        return True
    if error.category != MCP_TOOL_CALL_ERROR:
        return False
    message = " ".join(
        part
        for part in (error.message, error.provider_error_message)
        if isinstance(part, str)
    ).lower()
    if any(
        marker in message
        for marker in (
            "api key",
            "unauthorized",
            "forbidden",
            "quota",
            "billing",
            "invalid schema",
            "controlled-language",
        )
    ):
        return False
    return any(
        marker in message
        for marker in (
            "timeout",
            "timed out",
            "temporary",
            "overload",
            "connection reset",
            "network reset",
            "broken pipe",
        )
    )


def _safe_provider_message(error: ProviderError) -> str:
    message = " ".join((error.provider_error_message or error.message).split())
    lowered = message.lower()
    marker = lowered.find("image file too large")
    if marker >= 0:
        message = message[marker:]
    return message[:300]


def _string_field(result: dict[str, Any], key: str) -> str | None:
    value = result.get(key)
    return value if isinstance(value, str) and value else None
