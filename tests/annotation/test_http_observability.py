"""GLM HTTP business-error observability: classification, details, redaction.

All HTTP interaction goes through an injected fake transport; no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vla_data.annotation.glm_provider import (
    GLM_ENDPOINT,
    GLM_MODEL_ID,
    GLM46VFlashProvider,
    GLMProviderConfig,
    classify_http_error,
)
from vla_data.annotation.pipeline import _failed
from vla_data.annotation.provider import ProviderError

SECRET = "sk-test-never-print-000"


def zhipu_error(code: str | None, message: str) -> str:
    return json.dumps({"error": {"code": code, "message": message}})


class FakeTransport:
    def __init__(self, status: int, body: str, headers: dict | None = None) -> None:
        self.status = status
        self.body = body
        self.headers = headers or {}
        self.calls = 0

    def __call__(self, request, timeout_s: float):
        self.calls += 1
        from vla_data.annotation.glm_provider import HTTPResponse

        return HTTPResponse(self.status, self.body, dict(self.headers))


def _provider(transport) -> GLM46VFlashProvider:
    return GLM46VFlashProvider(
        GLMProviderConfig(sleeper=lambda _s: None),
        transport=transport,
        api_key=SECRET,
    )


def _request(tmp_path: Path):
    from vla_data.annotation.provider import AnnotationRequest, ImageItem, TextItem

    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpeg-bytes")
    return AnnotationRequest(
        episode_id="episode_000002",
        prompt_version="hierarchical_semantics_coarse_v1",
        items=(TextItem("label:"), ImageItem(1, "head", "head view", image)),
    )


# --------------------------------------------------------------- classification


@pytest.mark.parametrize(
    ("status", "code", "expected"),
    [
        (401, "1000", "HTTP_AUTH"),
        (401, None, "HTTP_AUTH"),
        (429, "1113", "HTTP_QUOTA"),
        (429, "1302", "HTTP_RATE_LIMIT"),
        (429, "1305", "HTTP_PROVIDER_OVERLOAD"),
        (429, "1308", "HTTP_QUOTA"),
        (429, "1309", "HTTP_QUOTA"),
        (429, "1311", "HTTP_QUOTA"),
        (429, None, "HTTP_RATE_LIMIT"),
        (403, None, "HTTP_PERMISSION"),
        (500, None, "HTTP_SERVER_ERROR"),
        (400, None, "HTTP_OTHER"),
        (418, "9999", "HTTP_OTHER"),
    ],
)
def test_classification_matrix(status, code, expected) -> None:
    assert classify_http_error(status, code) == expected


# ------------------------------------------------------ business-error scenarios


def test_401_auth_error_not_retried(tmp_path) -> None:
    transport = FakeTransport(401, zhipu_error("1000", "token invalid"))
    with pytest.raises(ProviderError) as excinfo:
        _provider(transport).annotate(_request(tmp_path))
    error = excinfo.value
    assert error.category == "HTTP_AUTH"
    assert error.http_status == 401
    assert error.provider_error_code == "1000"
    assert error.provider_error_message == "token invalid"
    assert transport.calls == 1  # auth burns no retry budget


def test_429_account_balance_is_quota_not_rate_limit(tmp_path) -> None:
    transport = FakeTransport(429, zhipu_error("1113", "余额不足"))
    with pytest.raises(ProviderError) as excinfo:
        _provider(transport).annotate(_request(tmp_path))
    assert excinfo.value.category == "HTTP_QUOTA"
    assert excinfo.value.provider_error_code == "1113"
    assert transport.calls == 1  # billing errors are not retried


def test_429_user_rate_limit_retries_then_keeps_details(tmp_path) -> None:
    transport = FakeTransport(429, zhipu_error("1302", "您当前使用该模型的并发上限"))
    with pytest.raises(ProviderError) as excinfo:
        _provider(transport).annotate(_request(tmp_path))
    error = excinfo.value
    assert error.category == "HTTP_RATE_LIMIT"
    assert error.http_status == 429
    assert error.provider_error_code == "1302"
    assert "1302" in error.message
    assert transport.calls == 3  # rate limits still use the bounded budget


def test_429_provider_overload_retries(tmp_path) -> None:
    transport = FakeTransport(429, zhipu_error("1305", "服务暂时过载"))
    with pytest.raises(ProviderError) as excinfo:
        _provider(transport).annotate(_request(tmp_path))
    assert excinfo.value.category == "HTTP_PROVIDER_OVERLOAD"
    assert transport.calls == 3


def test_429_quota_reset_not_retried(tmp_path) -> None:
    transport = FakeTransport(429, zhipu_error("1308", "调用次数超限"))
    with pytest.raises(ProviderError) as excinfo:
        _provider(transport).annotate(_request(tmp_path))
    assert excinfo.value.category == "HTTP_QUOTA"
    assert transport.calls == 1


def test_429_unknown_body_falls_back_to_rate_limit(tmp_path) -> None:
    transport = FakeTransport(429, "<html>gateway</html>")
    with pytest.raises(ProviderError) as excinfo:
        _provider(transport).annotate(_request(tmp_path))
    error = excinfo.value
    assert error.category == "HTTP_RATE_LIMIT"
    assert error.provider_error_code is None
    assert error.http_status == 429
    assert transport.calls == 3


def test_500_provider_error_body_preserved(tmp_path) -> None:
    transport = FakeTransport(500, zhipu_error("1305", "internal overload"))
    with pytest.raises(ProviderError) as excinfo:
        _provider(transport).annotate(_request(tmp_path))
    error = excinfo.value
    # Zhipu code 1305 means provider overload even over HTTP 500.
    assert error.category == "HTTP_PROVIDER_OVERLOAD"
    assert error.http_status == 500
    assert error.provider_error_code == "1305"


def test_malformed_error_json_keeps_safe_excerpt(tmp_path) -> None:
    transport = FakeTransport(429, "not-json{garbage")
    with pytest.raises(ProviderError) as excinfo:
        _provider(transport).annotate(_request(tmp_path))
    error = excinfo.value
    assert error.provider_error_code is None
    assert error.provider_error_message == "not-json{garbage"
    assert len(error.provider_error_message) <= 200


# --------------------------------------------------------- headers + redaction


def test_rate_limit_headers_whitelisted_and_request_id(tmp_path) -> None:
    transport = FakeTransport(
        429,
        zhipu_error("1302", "rate limited"),
        headers={
            "Retry-After": "17",
            "X-Request-Id": "req-abc-123",
            "Server": "nginx",
            "Set-Cookie": "session=secret-cookie",
        },
    )
    with pytest.raises(ProviderError) as excinfo:
        _provider(transport).annotate(_request(tmp_path))
    error = excinfo.value
    assert error.request_id == "req-abc-123"
    assert error.http_headers == {"retry-after": "17", "x-request-id": "req-abc-123"}
    assert "server" not in error.http_headers
    assert "set-cookie" not in error.http_headers


def test_secrets_never_appear_in_errors_or_details(tmp_path) -> None:
    transport = FakeTransport(
        401,
        zhipu_error("1000", "unauthorized"),
        headers={"Authorization": f"Bearer {SECRET}"},
    )
    with pytest.raises(ProviderError) as excinfo:
        _provider(transport).annotate(_request(tmp_path))
    error = excinfo.value
    rendered = str(error) + json.dumps(error.details())
    assert SECRET not in rendered
    assert "Bearer" not in rendered
    assert not error.http_headers  # Authorization was filtered out entirely


def test_official_endpoint_and_model_unchanged(tmp_path) -> None:
    transport = FakeTransport(429, zhipu_error("1302", "limited"))
    with pytest.raises(ProviderError):
        _provider(transport).annotate(_request(tmp_path))
    # The transport is the only network boundary; contract stays pinned.
    assert GLM_ENDPOINT.endswith("/api/paas/v4/chat/completions")
    assert GLM_MODEL_ID.startswith("glm-4.6v") and GLM_MODEL_ID == GLM_MODEL_ID.strip()


def test_failure_result_carries_provider_fields(tmp_path) -> None:
    from vla_data.annotation.glm_provider import HTTPResponse, _http_error

    error = _http_error(
        HTTPResponse(
            429,
            zhipu_error("1302", "并发上限"),
            {"x-request-id": "req-42"},
        )
    )
    result = _failed("episode_000002", "ACCEPT_WITH_WARNING", error)
    as_dict = result.as_dict()
    assert as_dict["provider_error_code"] == "1302"
    assert as_dict["http_status"] == 429
    assert as_dict["provider_request_id"] == "req-42"
    assert as_dict["error_type"] == "ProviderError"
