"""Coding Plan Vision MCP provider: discovery, fallback, errors, secrets.

Provider-level tests inject a fake MCP client; transport-level tests drive
a real subprocess speaking (deliberately faulty) JSON-RPC over stdio.
No test contacts the real @z_ai/mcp-server or any network service.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from vla_data.annotation.mcp_provider import (
    MCP_INVALID_RESPONSE,
    MCP_PROCESS_EXITED,
    MCP_STARTUP_ERROR,
    MCP_TIMEOUT,
    MCP_TOOL_CALL_ERROR,
    MCP_TOOL_NOT_FOUND,
    MCP_TOOL_SCHEMA_ERROR,
    CodingPlanAPIKeyMissingError,
    CodingPlanVisionMCPProvider,
    MCPProviderConfig,
    resolve_z_ai_api_key,
)
from vla_data.annotation.provider import (
    AnnotationRequest,
    ImageItem,
    ProviderError,
    TextItem,
)

TEAM_KEY = "zai-team-key-000"
PAYLOAD = json.dumps(
    {
        "episode_task": {
            "instruction": "Place the cable into the tray",
            "confidence": 0.9,
        },
        "semantic_segments": [
            {
                "segment_id": "s0",
                "start_curated_index": 0,
                "end_curated_index": 523,
                "instruction": "Grasp the cable",
                "confidence": 0.88,
            }
        ],
        "non_training_intervals": [],
    }
)


def _tool_schema(properties: dict, name: str = "image_analysis") -> dict:
    return {
        "name": name,
        "inputSchema": {"type": "object", "properties": properties},
    }


class FakeMCPClient:
    def __init__(
        self,
        tools,
        responses,
        *,
        initialize_error: Exception | None = None,
    ) -> None:
        self.tools = tools
        self.responses = list(responses)
        self.initialize_error = initialize_error
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    @property
    def alive(self) -> bool:
        return not self.closed

    def start(self) -> None:
        return None

    server_info: dict | None = None

    def initialize(self) -> dict:
        if self.initialize_error is not None:
            raise self.initialize_error
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "serverInfo": self.server_info,
        }

    def list_tools(self) -> list[dict]:
        return list(self.tools)

    def call_tool(self, name: str, arguments: dict, *, timeout_s=None) -> dict:
        self.calls.append((name, arguments))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self) -> None:
        self.closed = True


def _request(tmp_path: Path) -> AnnotationRequest:
    head = tmp_path / "head.jpg"
    wrist = tmp_path / "wrist.jpg"
    from PIL import Image

    Image.new("RGB", (64, 48), (200, 40, 40)).save(head, format="JPEG")
    Image.new("RGB", (64, 48), (40, 40, 200)).save(wrist, format="JPEG")
    return AnnotationRequest(
        episode_id="episode_000002",
        prompt_version="vla_semantic_coarse_v2",
        items=(
            TextItem("Pass A prompt text"),
            ImageItem(10, "head", "obs1 head", head),
            ImageItem(10, "right_wrist", "obs1 wrist", wrist),
        ),
    )


def _ok_result(text: str = PAYLOAD) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": False}


def _provider(client) -> CodingPlanVisionMCPProvider:
    return CodingPlanVisionMCPProvider(
        MCPProviderConfig(storyboard_root=None), client=client, api_key=TEAM_KEY
    )


# ------------------------------------------------------------ discovery


def test_initialize_and_image_analysis_discovered(tmp_path) -> None:
    client = FakeMCPClient(
        [
            _tool_schema(
                {
                    "images": {"type": "array", "items": {"type": "string"}},
                    "prompt": {"type": "string"},
                }
            )
        ],
        [_ok_result()],
    )
    provider = _provider(client)
    response = provider.infer_json(_request(tmp_path))
    assert response.provider == "coding_plan_vision_mcp"
    assert response.provider_metadata["mcp_selected_tool"] == "image_analysis"
    assert response.provider_metadata["native_multi_image_or_storyboard"] == (
        "native_multi_image"
    )
    assert provider.completed_provider_calls == 1
    assert provider.planned_provider_calls == 1


def test_missing_image_analysis_raises_tool_not_found(tmp_path) -> None:
    client = FakeMCPClient(
        [{"name": "ui_to_artifact", "inputSchema": {"properties": {}}}],
        [],
    )
    provider = _provider(client)
    with pytest.raises(ProviderError, match=MCP_TOOL_NOT_FOUND):
        provider.infer_json(_request(tmp_path))
    assert provider.failed_provider_calls == 1
    assert provider.planned_provider_calls == 1


def test_schema_without_image_property_raises_schema_error(tmp_path) -> None:
    client = FakeMCPClient([_tool_schema({"prompt": {"type": "string"}})], [])
    provider = _provider(client)
    with pytest.raises(ProviderError, match=MCP_TOOL_SCHEMA_ERROR):
        provider.infer_json(_request(tmp_path))


# --------------------------------------------------- transport modes


def test_native_multi_image_passes_ordered_local_paths(tmp_path) -> None:
    client = FakeMCPClient(
        [
            _tool_schema(
                {
                    "images": {"type": "array", "items": {"type": "string"}},
                    "prompt": {"type": "string"},
                }
            )
        ],
        [_ok_result()],
    )
    provider = _provider(client)
    provider.infer_json(_request(tmp_path))
    name, arguments = client.calls[0]
    assert name == "image_analysis"
    assert arguments["images"] == [
        str(tmp_path / "head.jpg"),
        str(tmp_path / "wrist.jpg"),
    ]
    assert arguments["prompt"] == "Pass A prompt text"


def test_single_image_schema_falls_back_to_storyboard(tmp_path) -> None:
    client = FakeMCPClient(
        [
            _tool_schema(
                {"image_path": {"type": "string"}, "prompt": {"type": "string"}}
            )
        ],
        [_ok_result()],
    )
    provider = _provider(client)
    provider.infer_json(_request(tmp_path))
    _name, arguments = client.calls[0]
    board = Path(arguments["image_path"])
    assert board.is_file() and board.suffix == ".jpg"
    # The storyboard carries the ordered observations in one annotated image.
    assert provider.transport_mode == "annotation_storyboard"
    assert provider._last_board == board
    # Source frames untouched.
    assert (tmp_path / "head.jpg").stat().st_size > 0


def test_storyboard_reuses_process_and_is_not_training_data(tmp_path) -> None:
    client = FakeMCPClient(
        [_tool_schema({"image_path": {"type": "string"}})], [_ok_result(), _ok_result()]
    )
    provider = _provider(client)
    provider.infer_json(_request(tmp_path))
    provider.infer_json(_request(tmp_path))
    assert client.initialize_error is None  # session reused, one client
    assert provider.completed_provider_calls == 2
    # Storyboards live under the annotation debug root only (never curated).
    assert provider._last_board.parent != tmp_path


def test_tool_error_result_classified(tmp_path) -> None:
    client = FakeMCPClient(
        [_tool_schema({"images": {"type": "array", "items": {"type": "string"}}})],
        [{"content": [{"type": "text", "text": "boom"}], "isError": True}],
    )
    provider = _provider(client)
    with pytest.raises(ProviderError, match=MCP_TOOL_CALL_ERROR):
        provider.infer_json(_request(tmp_path))


def test_malformed_tool_text_raises_invalid_json(tmp_path) -> None:
    client = FakeMCPClient(
        [_tool_schema({"images": {"type": "array", "items": {"type": "string"}}})],
        [_ok_result("totally not json")],
    )
    provider = _provider(client)
    with pytest.raises(ProviderError, match="INVALID_JSON"):
        provider.infer_json(_request(tmp_path))


def test_transport_returns_structured_payload_before_local_language_gate(
    tmp_path,
) -> None:
    bad = json.dumps(
        {
            "episode_task": {
                "instruction": "Maybe the robot grasps",
                "confidence": 0.5,
            },
            "semantic_segments": [],
            "non_training_intervals": [],
        }
    )
    client = FakeMCPClient(
        [_tool_schema({"images": {"type": "array", "items": {"type": "string"}}})],
        [_ok_result(bad)],
    )
    provider = _provider(client)
    response = provider.infer_json(_request(tmp_path))
    assert response.payload["episode_task"]["instruction"] == "Maybe the robot grasps"
    assert len(client.calls) == 1
    assert provider.retry_count == 0


def test_timeout_retries_once_then_succeeds_with_exact_accounting(tmp_path) -> None:
    client = FakeMCPClient(
        [_tool_schema({"images": {"type": "array", "items": {"type": "string"}}})],
        [ProviderError(MCP_TIMEOUT, "tools/call timed out after 300s"), _ok_result()],
    )
    provider = _provider(client)
    response = provider.infer_json(_request(tmp_path))
    assert response.attempt_count == 2
    assert response.provider_metadata["logical_calls"] == 1
    assert response.provider_metadata["actual_tools_call_attempts"] == 2
    assert response.provider_metadata["retry_count"] == 1
    assert provider.planned_provider_calls == 1
    assert provider.completed_provider_calls == 1
    assert provider.failed_provider_calls == 0
    assert provider.mcp_tools_call_attempted == 2
    assert provider.retry_count == 1


def test_timeout_retry_exhaustion_is_one_failed_logical_call(tmp_path) -> None:
    client = FakeMCPClient(
        [_tool_schema({"images": {"type": "array", "items": {"type": "string"}}})],
        [
            ProviderError(MCP_TIMEOUT, "tools/call timed out after 300s"),
            ProviderError(MCP_TIMEOUT, "tools/call timed out after 300s"),
        ],
    )
    provider = _provider(client)
    with pytest.raises(ProviderError, match=MCP_TIMEOUT):
        provider.infer_json(_request(tmp_path))
    assert provider.planned_provider_calls == 1
    assert provider.completed_provider_calls == 0
    assert provider.failed_provider_calls == 1
    assert provider._last_failure_metadata["logical_calls"] == 1
    assert provider._last_failure_metadata["actual_tools_call_attempts"] == 2
    assert provider._last_failure_metadata["retry_count"] == 1


@pytest.mark.parametrize("category", ["HTTP_AUTH", "HTTP_QUOTA", MCP_TOOL_SCHEMA_ERROR])
def test_nonretryable_provider_errors_are_not_retried(tmp_path, category) -> None:
    client = FakeMCPClient(
        [_tool_schema({"images": {"type": "array", "items": {"type": "string"}}})],
        [ProviderError(category, "nonretryable")],
    )
    provider = _provider(client)
    with pytest.raises(ProviderError, match=category):
        provider.infer_json(_request(tmp_path))
    assert len(client.calls) == 1
    assert provider.retry_count == 0


def test_response_provenance_records_runtime_facts_only(tmp_path) -> None:
    client = FakeMCPClient(
        [_tool_schema({"images": {"type": "array", "items": {"type": "string"}}})],
        [_ok_result()],
    )
    client.server_info = {"name": "@z_ai/mcp-server", "version": "0.1.5"}
    provider = _provider(client)
    response = provider.infer_json(_request(tmp_path))
    metadata = response.provider_metadata
    # No hardcoded backend family: model only when the runtime reports one.
    assert "documented_backend_family" not in metadata
    assert metadata["response_model"] is None
    assert metadata["backend_model_not_exposed_by_mcp"] is True
    assert metadata["mcp_selected_tool"] == "image_analysis"
    assert metadata["mcp_tool_selection_reason"].startswith("runtime_alias_match")
    assert metadata["available_tool_names"] == ["image_analysis"]
    assert metadata["mcp_server_name"] == "@z_ai/mcp-server"
    assert metadata["mcp_server_version"] == "0.1.5"
    assert metadata["mcp_package"] == "@z_ai/mcp-server"
    assert metadata["mcp_transport"] == "stdio"


# ------------------------------------------------------ credential rules


def test_z_ai_api_key_missing_is_explicit(monkeypatch) -> None:
    monkeypatch.delenv("Z_AI_API_KEY", raising=False)
    with pytest.raises(
        CodingPlanAPIKeyMissingError, match="CODING_PLAN_API_KEY_MISSING"
    ):
        resolve_z_ai_api_key()


def test_no_fallback_to_personal_paas_keys(monkeypatch) -> None:
    monkeypatch.delenv("Z_AI_API_KEY", raising=False)
    monkeypatch.setenv("GLM_API_KEY", "personal-key")
    monkeypatch.setenv("ZHIPU_API_KEY", "personal-key")
    with pytest.raises(CodingPlanAPIKeyMissingError):
        resolve_z_ai_api_key()


def test_key_never_appears_in_errors_or_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("Z_AI_API_KEY", raising=False)
    provider = CodingPlanVisionMCPProvider(MCPProviderConfig())
    request = _request(tmp_path)
    with pytest.raises(ProviderError) as excinfo:
        provider.infer_json(request)
    assert TEAM_KEY not in str(excinfo.value)
    client = FakeMCPClient(
        [_tool_schema({"images": {"type": "array", "items": {"type": "string"}}})],
        [_ok_result()],
    )
    provider = _provider(client)
    response = provider.infer_json(request)
    assert TEAM_KEY not in json.dumps(response.provider_metadata)


# ------------------------------------------- transport-level (subprocess)


ECHO_SERVER = r"""
import json, sys, time
for line in sys.stdin:
    try:
        message = json.loads(line)
    except ValueError:
        continue
    method = message.get("method")
    if method == "initialize":
        reply = {"jsonrpc": "2.0", "id": message["id"], "result":
                 {"protocolVersion": "2024-11-05", "capabilities": {}}}
    elif method == "tools/list":
        reply = {"jsonrpc": "2.0", "id": message["id"], "result": {"tools": [
            {"name": "image_analysis", "inputSchema": {"properties": {
                "images": {"type": "array", "items": {"type": "string"}}}}} ]}}
    elif method == "tools/call":
        reply = {"jsonrpc": "2.0", "id": message["id"], "result":
                 {"content": [{"type": "text", "text": __PAYLOAD__}]}}
    else:
        continue
    sys.stdout.write(json.dumps(reply) + "\n")
    sys.stdout.flush()
""".replace("__PAYLOAD__", json.dumps(PAYLOAD))

HANG_SERVER = r"""
import json, sys, time
first = True
for line in sys.stdin:
    message = json.loads(line)
    if message.get("method") == "initialize":
        reply = {"jsonrpc": "2.0", "id": message["id"], "result": {}}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()
    elif message.get("method") == "tools/list":
        reply = {"jsonrpc": "2.0", "id": message["id"], "result": {"tools": []}}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()
    else:
        time.sleep(60)
"""

MALFORMED_SERVER = r"""
import sys
line = sys.stdin.readline()  # initialize
sys.stdout.write("this is not json-rpc\n")
sys.stdout.flush()
time.sleep(30)
"""

EXIT_SERVER = r"""
import sys
sys.stdin.readline()
sys.exit(3)
"""


def _subprocess_client(script: str, **config_overrides) -> object:
    from vla_data.annotation.mcp_provider import MCPProviderConfig, _MCPClient

    config = MCPProviderConfig(
        command=sys.executable,
        server_args=("-c", script),
        handshake_timeout_s=10,
        call_timeout_s=5,
        **config_overrides,
    )
    return _MCPClient(config, TEAM_KEY)


def test_real_subprocess_roundtrip(tmp_path) -> None:
    client = _subprocess_client(ECHO_SERVER)
    client.start()
    client.initialize()
    tools = client.list_tools()
    assert any(tool["name"] == "image_analysis" for tool in tools)
    result = client.call_tool("image_analysis", {"images": ["a.jpg"]})
    assert result["content"][0]["text"] == PAYLOAD
    client.close()


def test_subprocess_timeout_classified() -> None:
    client = _subprocess_client(HANG_SERVER)
    client.start()
    client.initialize()
    with pytest.raises(ProviderError, match=MCP_TIMEOUT):
        client.call_tool("image_analysis", {}, timeout_s=1.0)
    client.close()


def test_subprocess_malformed_jsonrpc_classified() -> None:
    client = _subprocess_client(MALFORMED_SERVER)
    client.start()
    with pytest.raises(ProviderError, match=MCP_INVALID_RESPONSE):
        client.initialize()
    client.close()


def test_subprocess_unexpected_exit_classified() -> None:
    client = _subprocess_client(EXIT_SERVER)
    client.start()
    time.sleep(0.3)
    with pytest.raises(ProviderError, match=MCP_PROCESS_EXITED):
        client.initialize()


def test_startup_failure_when_command_missing() -> None:
    from vla_data.annotation.mcp_provider import MCPProviderConfig, _MCPClient

    client = _MCPClient(
        MCPProviderConfig(command="definitely-not-a-real-binary-xyz"), TEAM_KEY
    )
    with pytest.raises(ProviderError, match=MCP_STARTUP_ERROR):
        client.start()


def test_broken_pipe_classified() -> None:
    client = _subprocess_client(EXIT_SERVER)
    client.start()
    time.sleep(0.3)
    with pytest.raises(ProviderError, match=MCP_PROCESS_EXITED):
        client.call_tool("image_analysis", {})
