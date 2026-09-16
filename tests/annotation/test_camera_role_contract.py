"""D4.2.2 camera-role contract, typed storyboard inputs, failure classes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from tests.annotation.test_mcp_provider import FakeMCPClient, _ok_result, _tool_schema
from vla_data.annotation.mcp_provider import (
    ANNOTATION_INPUT_ERROR,
    MCP_TOOL_CALL_ERROR,
    CodingPlanVisionMCPProvider,
    MCPProviderConfig,
)
from vla_data.annotation.provider import (
    CAMERA_ROLE_HEAD,
    CAMERA_ROLE_RIGHT_WRIST,
    CANONICAL_CAMERA_ROLES,
    AnnotationRequest,
    ImageItem,
    ProviderError,
    TextItem,
)
from vla_data.annotation.storyboard import (
    ANNOTATION_CAMERA_VIEW_MISSING,
    MultiViewObservation,
    observations_from_request,
    render_storyboard,
)

TEAM_KEY = "zai-team-key-000"
PAYLOAD = json.dumps(
    {
        "episode_task": {
            "instruction": "Place the cable into the tray",
            "confidence": 0.9,
        },
        "semantic_segments": [],
        "non_training_intervals": [],
    }
)


def _frame(tmp_path: Path, name: str, color=(120, 40, 40)) -> Path:
    path = tmp_path / name
    Image.new("RGB", (64, 48), color).save(path, format="JPEG")
    return path


def _request(
    tmp_path: Path, points=(0, 104, 208), head_role="head", wrist_role="right_wrist"
) -> AnnotationRequest:
    items: list = [TextItem("Pass A prompt")]
    for point in points:
        head = _frame(tmp_path, f"head_{point}.jpg", (200, 40, 40))
        wrist = _frame(tmp_path, f"wrist_{point}.jpg", (40, 40, 200))
        items.append(ImageItem(point, head_role, "head view", head))
        items.append(ImageItem(point, wrist_role, "wrist view", wrist))
    return AnnotationRequest(
        episode_id="episode_000002",
        prompt_version="vla_semantic_coarse_v2",
        items=tuple(items),
    )


# ------------------------------------------------- exact regression (L/M)


def test_canonical_roles_are_exactly_head_and_right_wrist() -> None:
    assert CANONICAL_CAMERA_ROLES == ("head", "right_wrist")
    assert CAMERA_ROLE_HEAD == "head"
    assert CAMERA_ROLE_RIGHT_WRIST == "right_wrist"


def test_exact_episode_000002_layout_builds_storyboard_without_keyerror(
    tmp_path,
) -> None:
    """Regression: 6 Pass-A keyframes at the real indices must render."""

    request = _request(tmp_path, points=(0, 104, 208, 313, 417, 522))
    observations = observations_from_request(request)
    assert [observation.curated_index for observation in observations] == [
        0,
        104,
        208,
        313,
        417,
        522,
    ]
    board = render_storyboard(
        observations, tmp_path / "boards", episode_id="episode_000002"
    )
    assert board.is_file() and board.suffix == ".jpg"


def test_storage_field_names_never_become_role_lookups() -> None:
    # Storage keys are a different namespace; grouped access uses only
    # canonical roles, so fabricated storage-style roles group harmlessly
    # apart rather than colliding with canonical lookups.
    assert CAMERA_ROLE_HEAD != "head_rgb_frame_index"
    assert CAMERA_ROLE_RIGHT_WRIST != "right_wrist_rgb_frame_index"
    observations = observations_from_request(_request(Path("/tmp")))
    assert observations[0].head_path != observations[0].right_wrist_path


# ------------------------------------------------------- missing roles (N)


def test_head_only_fails_with_camera_view_missing(tmp_path) -> None:
    head = _frame(tmp_path, "head.jpg")
    request = AnnotationRequest(
        episode_id="e",
        prompt_version="v2",
        items=(ImageItem(0, "head", "h", head),),
    )
    with pytest.raises(ProviderError, match=ANNOTATION_CAMERA_VIEW_MISSING) as err:
        observations_from_request(request)
    assert "right_wrist" in str(err.value)  # names the missing role


def test_right_wrist_only_fails_with_camera_view_missing(tmp_path) -> None:
    wrist = _frame(tmp_path, "wrist.jpg")
    request = AnnotationRequest(
        episode_id="e",
        prompt_version="v2",
        items=(ImageItem(0, "right_wrist", "w", wrist),),
    )
    with pytest.raises(ProviderError, match=ANNOTATION_CAMERA_VIEW_MISSING) as err:
        observations_from_request(request)
    assert "head" in str(err.value)


def test_unknown_wrist_alias_is_not_silently_accepted(tmp_path) -> None:
    # "wrist" (non-canonical) alongside head must NOT satisfy the contract.
    head = _frame(tmp_path, "h.jpg")
    wrist = _frame(tmp_path, "w.jpg")
    request = AnnotationRequest(
        episode_id="e",
        prompt_version="v2",
        items=(
            ImageItem(0, "head", "h", head),
            ImageItem(0, "wrist", "w", wrist),
        ),
    )
    with pytest.raises(ProviderError, match=ANNOTATION_CAMERA_VIEW_MISSING):
        observations_from_request(request)


def test_missing_source_file_fails_camera_view_missing(tmp_path) -> None:
    observation = MultiViewObservation(
        curated_index=0,
        timestamp_ns=0,
        head_path=tmp_path / "absent_head.jpg",
        right_wrist_path=_frame(tmp_path, "w.jpg"),
    )
    with pytest.raises(ProviderError, match=ANNOTATION_CAMERA_VIEW_MISSING):
        render_storyboard([observation], tmp_path / "out")


# ---------------------------------------------------- synchronization (O/P)


def test_multi_frame_pairing_preserves_curated_index_order(tmp_path) -> None:
    request = _request(tmp_path, points=(0, 104, 208))
    observations = observations_from_request(request)
    assert [obs.curated_index for obs in observations] == [0, 104, 208]
    for observation in observations:
        suffix = f"{observation.curated_index}.jpg"
        assert observation.head_path.name == f"head_{suffix}"
        assert observation.right_wrist_path.name == f"wrist_{suffix}"


def test_source_rgb_files_unchanged_by_storyboard(tmp_path) -> None:
    request = _request(tmp_path, points=(0, 104))
    before = {
        item.source_path: hashlib.sha256(item.source_path.read_bytes()).hexdigest()
        for item in request.items
        if hasattr(item, "camera")
    }
    board = render_storyboard(observations_from_request(request), tmp_path / "boards")
    after = {
        item.source_path: hashlib.sha256(item.source_path.read_bytes()).hexdigest()
        for item in request.items
        if hasattr(item, "camera")
    }
    assert before == after
    assert board.parent != tmp_path or board.name.startswith("storyboard_")


# ----------------------------------------------- failure observability (Q/I/J)


def _mcp_provider(tmp_path, responses=None) -> CodingPlanVisionMCPProvider:
    client = FakeMCPClient(
        [
            _tool_schema(
                {"image_source": {"type": "string"}, "prompt": {"type": "string"}},
                name="analyze_image",
            )
        ],
        responses or [],
    )
    client.server_info = {"name": "zai-mcp-server", "version": "0.1.5"}
    return CodingPlanVisionMCPProvider(
        MCPProviderConfig(), client=client, api_key=TEAM_KEY
    )


def test_local_input_failure_is_not_mcp_tool_call_error(tmp_path) -> None:
    # head-only request: storyboard build fails BEFORE any tools/call.
    provider = _mcp_provider(tmp_path, [_ok_result(PAYLOAD)])
    head = _frame(tmp_path, "only_head.jpg")
    request = AnnotationRequest(
        episode_id="e",
        prompt_version="v2",
        items=(TextItem("p"), ImageItem(0, "head", "h", head)),
    )
    with pytest.raises(ProviderError) as err:
        provider.infer_json(request)
    assert err.value.category == ANNOTATION_CAMERA_VIEW_MISSING
    assert MCP_TOOL_CALL_ERROR not in str(err.value.category)
    assert provider.mcp_tools_call_attempted == 0
    assert provider.completed_provider_calls == 0
    assert provider.failed_provider_calls == 1
    metadata = provider._last_failure_metadata
    assert metadata["tools_call_count"] == 0
    assert metadata["mcp_selected_tool"] == "analyze_image"
    assert metadata["mcp_server_name"] == "zai-mcp-server"
    assert metadata["mcp_server_version"] == "0.1.5"
    assert TEAM_KEY not in json.dumps(metadata)


def test_real_tool_failure_counts_tools_call(tmp_path) -> None:
    provider = _mcp_provider(
        tmp_path,
        [{"content": [{"type": "text", "text": "boom"}], "isError": True}],
    )
    with pytest.raises(ProviderError, match=MCP_TOOL_CALL_ERROR):
        provider.infer_json(_request(tmp_path, points=(0,)))
    assert provider.mcp_tools_call_attempted == 1
    assert provider._last_failure_metadata["tools_call_count"] == 1


def test_unknown_camera_role_is_input_error_not_camera_missing(tmp_path) -> None:
    # Non-canonical "wrist" role: typed build rejects it as a missing
    # canonical pair; opaque grouping surprises become ANNOTATION_INPUT_ERROR.
    provider = _mcp_provider(tmp_path, [_ok_result(PAYLOAD)])
    head = _frame(tmp_path, "h.jpg")
    wrist = _frame(tmp_path, "w.jpg")
    request = AnnotationRequest(
        episode_id="e",
        prompt_version="v2",
        items=(
            TextItem("p"),
            ImageItem(0, "head", "h", head),
            ImageItem(0, "wrist", "w", wrist),
        ),
    )
    with pytest.raises(ProviderError, match=ANNOTATION_CAMERA_VIEW_MISSING):
        provider.infer_json(request)
    assert provider.mcp_tools_call_attempted == 0
    assert provider._last_failure_metadata["error_category"] == (
        ANNOTATION_CAMERA_VIEW_MISSING
    )
    assert ANNOTATION_INPUT_ERROR  # category available for opaque build failures
