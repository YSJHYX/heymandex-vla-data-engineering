"""D4.2.1 runtime tool-alias discovery and schema adaptation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.annotation.test_mcp_provider import (
    FakeMCPClient,
    _ok_result,
    _request,
    _tool_schema,
)
from vla_data.annotation.mcp_provider import (
    MCP_TOOL_NOT_FOUND,
    MCP_TOOL_SCHEMA_ERROR,
    CodingPlanVisionMCPProvider,
    MCPProviderConfig,
    _select_general_image_tool,
)
from vla_data.annotation.provider import ProviderError

CURRENT_NPM_RUNTIME_TOOLS = [
    "ui_to_artifact",
    "extract_text_from_screenshot",
    "diagnose_error_screenshot",
    "understand_technical_diagram",
    "analyze_data_visualization",
    "ui_diff_check",
    "analyze_image",
    "analyze_video",
]


def _tools_named(names: list[str], properties: dict) -> list[dict]:
    return [
        _tool_schema(properties, name=name)
        if name in ("image_analysis", "analyze_image")
        else {"name": name, "inputSchema": {"properties": {}}}
        for name in names
    ]


def _provider_with(tools, responses=None) -> CodingPlanVisionMCPProvider:
    client = FakeMCPClient(tools, responses or [])
    return CodingPlanVisionMCPProvider(
        MCPProviderConfig(), client=client, api_key="team-key"
    )


# ------------------------------------------------------- alias selection (15)


def test_selector_prefers_docs_spelling() -> None:
    tool, reason = _select_general_image_tool(
        _tools_named(["image_analysis"], {"images": {"type": "array"}})
    )
    assert tool is not None and tool["name"] == "image_analysis"
    assert reason == "runtime_alias_match (docs spelling)"


def test_selector_accepts_npm_runtime_spelling() -> None:
    tools = _tools_named(
        CURRENT_NPM_RUNTIME_TOOLS, {"image_source": {"type": "string"}}
    )
    tool, reason = _select_general_image_tool(tools)
    assert tool is not None and tool["name"] == "analyze_image"
    assert reason == "runtime_alias_match (npm runtime spelling)"


def test_selector_is_deterministic_when_both_present() -> None:
    tools = _tools_named(
        ["analyze_image", "image_analysis"], {"images": {"type": "array"}}
    )
    tool, _ = _select_general_image_tool(tools)
    assert tool["name"] == "image_analysis"


def test_selector_rejects_when_neither_alias_present() -> None:
    tool, reason = _select_general_image_tool(_tools_named(["analyze_video"], {}))
    assert tool is None and reason is None


def test_real_runtime_inventory_selects_analyze_image(tmp_path) -> None:
    # The exact runtime tool list observed on the real server.
    provider = _provider_with(
        _tools_named(CURRENT_NPM_RUNTIME_TOOLS, {"image_source": {"type": "string"}}),
        [_ok_result()],
    )
    response = provider.infer_json(_request(tmp_path))
    metadata = response.provider_metadata
    assert metadata["mcp_selected_tool"] == "analyze_image"
    assert metadata["available_tool_names"] == CURRENT_NPM_RUNTIME_TOOLS
    assert provider.transport_mode == "annotation_storyboard"


def test_video_alias_never_selected_as_annotation_tool(tmp_path) -> None:
    provider = _provider_with(
        _tools_named(
            ["analyze_video", "analyze_image"], {"image_source": {"type": "string"}}
        ),
        [_ok_result()],
    )
    provider.infer_json(_request(tmp_path))
    assert provider._require_selected_tool_name() == "analyze_image"


# ------------------------------------------------------ schema variants (16)


def test_runtime_schema_image_source_plus_prompt(tmp_path) -> None:
    provider = _provider_with(
        _tools_named(
            ["analyze_image"],
            {"image_source": {"type": "string"}, "prompt": {"type": "string"}},
        ),
        [_ok_result()],
    )
    provider.infer_json(_request(tmp_path))
    _name, arguments = provider._client.calls[0]
    assert set(arguments) == {"image_source", "prompt"}
    assert Path(arguments["image_source"]).suffix == ".jpg"  # storyboard path


def test_runtime_schema_image_path_plus_instruction(tmp_path) -> None:
    provider = _provider_with(
        _tools_named(
            ["image_analysis"],
            {"image_path": {"type": "string"}, "instruction": {"type": "string"}},
        ),
        [_ok_result()],
    )
    provider.infer_json(_request(tmp_path))
    _name, arguments = provider._client.calls[0]
    assert "image_path" in arguments and "instruction" in arguments


def test_runtime_schema_array_images_is_native_multi_image(tmp_path) -> None:
    provider = _provider_with(
        _tools_named(
            ["analyze_image"],
            {
                "images": {"type": "array", "items": {"type": "string"}},
                "prompt": {"type": "string"},
            },
        ),
        [_ok_result()],
    )
    provider.infer_json(_request(tmp_path))
    _name, arguments = provider._client.calls[0]
    assert isinstance(arguments["images"], list) and len(arguments["images"]) == 2
    assert provider.transport_mode == "native_multi_image"


def test_bare_path_field_is_recognized(tmp_path) -> None:
    provider = _provider_with(
        _tools_named(
            ["analyze_image"],
            {"path": {"type": "string"}, "prompt": {"type": "string"}},
        ),
        [_ok_result()],
    )
    provider.infer_json(_request(tmp_path))
    _name, arguments = provider._client.calls[0]
    assert "path" in arguments


def test_name_match_without_image_schema_is_schema_error(tmp_path) -> None:
    # Section 3: name alone is insufficient — schema must express image+prompt.
    provider = _provider_with(
        _tools_named(["analyze_image"], {"prompt": {"type": "string"}}), []
    )
    with pytest.raises(ProviderError, match=MCP_TOOL_SCHEMA_ERROR):
        provider.infer_json(_request(tmp_path))


def test_no_alias_is_tool_not_found_not_schema_error(tmp_path) -> None:
    provider = _provider_with(_tools_named(["analyze_video"], {}), [])
    with pytest.raises(ProviderError, match=MCP_TOOL_NOT_FOUND):
        provider.infer_json(_request(tmp_path))


# ---------------------------------------------------------- provenance (17)


def test_selected_tool_provenance_is_exact_runtime_name(tmp_path) -> None:
    provider = _provider_with(
        _tools_named(CURRENT_NPM_RUNTIME_TOOLS, {"image_source": {"type": "string"}}),
        [_ok_result()],
    )
    response = provider.infer_json(_request(tmp_path))
    metadata = response.provider_metadata
    assert metadata["mcp_selected_tool"] == "analyze_image"
    assert (
        metadata["mcp_tool_selection_reason"]
        == "runtime_alias_match (npm runtime spelling)"
    )
    assert metadata["response_model"] is None
    assert metadata["backend_model_not_exposed_by_mcp"] is True
    assert "documented_backend_family" not in metadata


def test_server_info_nulls_when_runtime_silent(tmp_path) -> None:
    provider = _provider_with(
        _tools_named(
            ["image_analysis"],
            {"images": {"type": "array", "items": {"type": "string"}}},
        ),
        [_ok_result()],
    )
    response = provider.infer_json(_request(tmp_path))
    metadata = response.provider_metadata
    assert metadata["mcp_server_name"] is None
    assert metadata["mcp_server_version"] is None


def test_probe_reports_selection_without_any_tool_call(tmp_path) -> None:
    provider = _provider_with(
        _tools_named(
            CURRENT_NPM_RUNTIME_TOOLS,
            {"image_source": {"type": "string"}, "prompt": {"type": "string"}},
        ),
    )
    report = provider.probe()
    assert report["provider"] == "coding_plan_vision_mcp"
    assert report["mcp_selected_tool"] == "analyze_image"
    assert report["image_input_mode"] == "single-image (storyboard)"
    assert report["prompt_field"] == "prompt"
    assert report["image_field"] == "image_source"
    assert report["tools_call_count"] == 0
    assert report["ready"] is True
    assert provider._client.calls == []  # zero tools/call during probe
    payload = json.dumps(report)
    assert "team-key" not in payload  # no secrets in probe output
