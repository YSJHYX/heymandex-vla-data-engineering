"""GLM-4.6V-Flash provider: payload contract, retries, parsing, credentials.

All HTTP interaction goes through an injected fake transport; no test in this
file performs a real network request.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from vla_data.annotation.glm_provider import (
    GLM_ENDPOINT,
    GLM_MODEL_ID,
    GLM46VFlashProvider,
    GLMProviderConfig,
    MissingAPIKeyError,
    resolve_api_key,
)
from vla_data.annotation.provider import (
    AnnotationRequest,
    ImageItem,
    ProviderError,
    TextItem,
)

SECRET = "sk-test-do-not-print-000"
RESPONSE_BODY = {
    "id": "req-123",
    "model": GLM_MODEL_ID,
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "instruction": "Pick up the object and place it into the tray.",
                        "confidence": 0.9,
                        "task_type": "pick_and_place",
                        "objects": ["object", "tray"],
                        "uncertainty": None,
                    }
                ),
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 1200,
        "completion_tokens": 40,
        "total_tokens": 1240,
    },
}


class FakeTransport:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request, timeout_s: float):
        from vla_data.annotation.glm_provider import HTTPResponse

        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        status, body = response
        return HTTPResponse(status, body)


def _image(path: Path, point: int = 1, camera: str = "head") -> ImageItem:
    return ImageItem(
        temporal_point=point, camera=camera, label=f"{camera} view", source_path=path
    )


def _request(tmp_path: Path, image_path: Path) -> AnnotationRequest:
    return AnnotationRequest(
        episode_id="episode_000001",
        prompt_version="task_instruction_v1",
        items=(
            TextItem("Temporal point 1:"),
            _image(image_path, 1, "head"),
            _image(image_path.with_name("wrist.jpg"), 1, "right_wrist"),
            TextItem("Describe the task."),
        ),
    )


@pytest.fixture
def jpg(tmp_path: Path) -> Path:
    from PIL import Image

    head = tmp_path / "head.jpg"
    wrist = tmp_path / "wrist.jpg"
    Image.new("RGB", (6, 6), (255, 0, 0)).save(head, format="JPEG")
    Image.new("RGB", (6, 6), (0, 0, 255)).save(wrist, format="JPEG")
    return head


def _provider(transport, **config_overrides) -> GLM46VFlashProvider:
    config = GLMProviderConfig(
        sleeper=lambda _seconds: None,  # never actually sleep in tests
        **config_overrides,
    )
    return GLM46VFlashProvider(config, transport=transport, api_key=SECRET)


def test_official_http_contract_url_model_headers_thinking(jpg, tmp_path) -> None:
    transport = FakeTransport([(200, json.dumps(RESPONSE_BODY))])
    provider = _provider(transport)
    provider.annotate(_request(tmp_path, jpg))

    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert request.url == GLM_ENDPOINT
    assert GLM_ENDPOINT == "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    assert request.headers["Authorization"] == f"Bearer {SECRET}"
    assert request.headers["Content-Type"] == "application/json"

    payload = json.loads(request.body)
    assert payload["model"] == GLM_MODEL_ID
    assert GLM_MODEL_ID.startswith("glm-4.6v") and GLM_MODEL_ID == GLM_MODEL_ID.strip()
    assert payload["thinking"] == {"type": "enabled"}
    assert "response_format" not in payload
    message = payload["messages"][0]
    assert message["role"] == "user"
    types = [item["type"] for item in message["content"]]
    assert types[:4] == ["text", "image_url", "image_url", "text"]


def test_image_items_carry_base64_of_local_jpg_bytes(jpg, tmp_path) -> None:
    transport = FakeTransport([(200, json.dumps(RESPONSE_BODY))])
    provider = _provider(transport)
    provider.annotate(_request(tmp_path, jpg))

    content = json.loads(transport.requests[0].body)["messages"][0]["content"]
    image_items = [item for item in content if item["type"] == "image_url"]
    assert len(image_items) == 2
    expected_head = base64.b64encode(jpg.read_bytes()).decode("ascii")
    expected_wrist = base64.b64encode(jpg.with_name("wrist.jpg").read_bytes()).decode(
        "ascii"
    )
    assert image_items[0]["image_url"]["url"] == expected_head
    assert image_items[1]["image_url"]["url"] == expected_wrist
    assert any(
        item["text"].startswith("Describe the task.")
        for item in content
        if item["type"] == "text"
    )


def test_max_images_per_request_truncates_and_provider_still_succeeds(
    jpg, tmp_path
) -> None:
    transport = FakeTransport([(200, json.dumps(RESPONSE_BODY))])
    provider = _provider(transport, max_images_per_request=1)
    annotation = provider.annotate(_request(tmp_path, jpg))

    content = json.loads(transport.requests[0].body)["messages"][0]["content"]
    assert sum(item["type"] == "image_url" for item in content) == 1
    assert annotation.instruction.startswith("Pick up")


def test_provenance_request_id_usage_finish_reason(jpg, tmp_path) -> None:
    transport = FakeTransport([(200, json.dumps(RESPONSE_BODY))])
    provider = _provider(transport)
    annotation = provider.annotate(_request(tmp_path, jpg))

    assert annotation.request_id == "req-123"
    assert annotation.finish_reason == "stop"
    assert annotation.usage == {
        "prompt_tokens": 1200,
        "completion_tokens": 40,
        "total_tokens": 1240,
    }
    assert annotation.attempt_count == 1
    assert "Pick up" in annotation.raw_response_text


def test_markdown_fenced_json_is_parsed_safely(jpg, tmp_path) -> None:
    body = json.loads(json.dumps(RESPONSE_BODY))
    body["choices"][0]["message"]["content"] = (
        "```json\n"
        + json.dumps(json.loads(body["choices"][0]["message"]["content"]))
        + "\n```"
    )
    transport = FakeTransport([(200, json.dumps(body))])
    provider = _provider(transport)
    annotation = provider.annotate(_request(tmp_path, jpg))
    assert annotation.instruction.startswith("Pick up")


def test_invalid_json_then_success_retries_within_budget(jpg, tmp_path) -> None:
    body = json.loads(json.dumps(RESPONSE_BODY))
    body["choices"][0]["message"]["content"] = "no json here at all"
    transport = FakeTransport(
        [(200, json.dumps(body)), (200, json.dumps(RESPONSE_BODY))]
    )
    provider = _provider(transport, max_attempts=3)
    annotation = provider.annotate(_request(tmp_path, jpg))

    assert len(transport.requests) == 2
    assert annotation.attempt_count == 2


def test_schema_invalid_retries_then_gives_up_bounded(jpg, tmp_path) -> None:
    body = json.loads(json.dumps(RESPONSE_BODY))
    body["choices"][0]["message"]["content"] = json.dumps(
        {"instruction": "", "confidence": 0.5, "objects": []}
    )
    transport = FakeTransport([(200, json.dumps(body))] * 5)
    provider = _provider(transport, max_attempts=3)

    with pytest.raises(ProviderError, match="INVALID_SCHEMA"):
        provider.annotate(_request(tmp_path, jpg))
    assert len(transport.requests) == 3  # bounded retry


def test_empty_response_content_is_classified(jpg, tmp_path) -> None:
    body = json.loads(json.dumps(RESPONSE_BODY))
    body["choices"][0]["message"]["content"] = "   "
    transport = FakeTransport([(200, json.dumps(body))] * 3)
    provider = _provider(transport, max_attempts=2)
    with pytest.raises(ProviderError, match="EMPTY_RESPONSE"):
        provider.annotate(_request(tmp_path, jpg))
    assert len(transport.requests) == 2


def test_client_error_is_not_retried(jpg, tmp_path) -> None:
    transport = FakeTransport([(401, '{"error": "bad key"}')] * 3)
    provider = _provider(transport, max_attempts=3)
    # 401 is now classified precisely as HTTP_AUTH (still non-retryable).
    with pytest.raises(ProviderError, match="HTTP_AUTH"):
        provider.annotate(_request(tmp_path, jpg))
    assert len(transport.requests) == 1  # 4xx burns no retry budget


def test_rate_limit_and_server_error_are_retried(jpg, tmp_path) -> None:
    transport = FakeTransport(
        [(429, "{}"), (503, "{}"), (200, json.dumps(RESPONSE_BODY))]
    )
    provider = _provider(transport, max_attempts=3)
    annotation = provider.annotate(_request(tmp_path, jpg))
    assert len(transport.requests) == 3
    assert annotation.attempt_count == 3


def test_network_error_is_classified_and_bounded(jpg, tmp_path) -> None:
    import urllib.error

    transport = FakeTransport([urllib.error.URLError("connection reset")] * 3)
    provider = _provider(transport, max_attempts=2)
    with pytest.raises(ProviderError, match="NETWORK_ERROR"):
        provider.annotate(_request(tmp_path, jpg))
    assert len(transport.requests) == 2


def test_error_messages_never_leak_the_api_key(jpg, tmp_path) -> None:
    transport = FakeTransport([(401, "unauthorized")] * 3)
    provider = _provider(transport)
    with pytest.raises(ProviderError) as excinfo:
        provider.annotate(_request(tmp_path, jpg))
    assert SECRET not in str(excinfo.value)
    assert SECRET not in json.dumps(transport.requests[0].body)


def test_api_key_precedence_and_missing(monkeypatch) -> None:
    monkeypatch.delenv("GLM_API_KEY", raising=False)
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        resolve_api_key()

    monkeypatch.setenv("ZHIPU_API_KEY", "zhipu-key")
    assert resolve_api_key() == "zhipu-key"

    monkeypatch.setenv("GLM_API_KEY", "glm-key")
    assert resolve_api_key() == "glm-key"  # GLM_API_KEY wins

    assert resolve_api_key({"ZHIPU_API_KEY": "z", "GLM_API_KEY": "g"}) == "g"
