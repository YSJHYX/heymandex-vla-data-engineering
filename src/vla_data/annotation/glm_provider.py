"""GLM-4.6V-Flash provider over the official direct HTTP contract.

The core annotation pipeline depends only on the VLMProvider protocol; this
module owns every GLM-specific detail: endpoint, payload shape, base64 image
encoding, retry classification, and response parsing.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import uuid4

from vla_data.annotation.provider import (
    ERROR_EMPTY_RESPONSE,
    ERROR_HTTP_AUTH,
    ERROR_HTTP_OTHER,
    ERROR_HTTP_OVERLOAD,
    ERROR_HTTP_PERMISSION,
    ERROR_HTTP_QUOTA,
    ERROR_INVALID_JSON,
    ERROR_INVALID_SCHEMA,
    ERROR_NETWORK,
    ERROR_RATE_LIMIT,
    ERROR_SERVER,
    NON_RETRYABLE_ERRORS,
    AnnotationRequest,
    ImageItem,
    ModelAnnotation,
    ProviderError,
    StructuredModelResponse,
    TextItem,
)
from vla_data.annotation.schema import AnnotationSchemaError, validate_model_payload

PROVIDER_NAME = "zhipu_bigmodel"
GLM_MODEL_ID = "glm-4.6v-flashx"  # API model id; never pad with whitespace.
GLM_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

# Zhipu business error codes (bigmodel open platform error-code table).
ZHIPU_CODE_MAP: dict[str, str] = {
    "1000": ERROR_HTTP_AUTH,  # authentication / token invalid
    "1001": ERROR_HTTP_AUTH,
    "1113": ERROR_HTTP_QUOTA,  # account balance insufficient (billing)
    "1302": ERROR_RATE_LIMIT,  # user-level rate limit
    "1305": ERROR_HTTP_OVERLOAD,  # provider-side service overload
    "1308": ERROR_HTTP_QUOTA,  # API quota exhausted / resets later
}
# 1309+ subscription/quota/permission band (1309/1310/1311/...).
ZHIPU_QUOTA_BAND = (1309, 1399)

# Response headers safe to keep for rate-limit debugging. Never a full dump.
SAFE_RESPONSE_HEADERS = (
    "retry-after",
    "request-id",
    "x-request-id",
    "rate-limit-limit",
    "rate-limit-remaining",
    "rate-limit-reset",
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
)


class MissingAPIKeyError(RuntimeError):
    """No GLM credential is available in the environment."""


def resolve_api_key(environment: dict[str, str] | None = None) -> str:
    """Resolve the credential with explicit precedence GLM_API_KEY > ZHIPU_API_KEY."""

    source = os.environ if environment is None else environment
    key = source.get("GLM_API_KEY") or source.get("ZHIPU_API_KEY")
    if not key or not key.strip():
        raise MissingAPIKeyError(
            "set GLM_API_KEY (or ZHIPU_API_KEY) before calling the GLM provider"
        )
    return key.strip()


@dataclass(frozen=True)
class GLMProviderConfig:
    model: str = GLM_MODEL_ID
    max_temporal_points: int = 6
    max_images_per_request: int | None = None
    request_timeout_s: float = 120.0
    max_attempts: int = 3
    backoff_base_s: float = 2.0
    environment: dict[str, str] | None = None
    sleeper: Callable[[float], None] = field(
        default=lambda seconds: time.sleep(seconds), repr=False
    )


@dataclass(frozen=True)
class HTTPRequest:
    url: str
    headers: dict[str, str]
    body: str


@dataclass(frozen=True)
class HTTPResponse:
    status: int
    body: str
    headers: dict[str, str] = field(default_factory=dict)


def urllib_transport(request: HTTPRequest, timeout_s: float) -> HTTPResponse:
    """The default stdlib transport; tests inject their own instead."""

    outbound = urllib.request.Request(
        request.url,
        data=request.body.encode("utf-8"),
        headers=request.headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(outbound, timeout=timeout_s) as response:
            return HTTPResponse(
                response.status,
                response.read().decode("utf-8"),
                dict(response.headers.items()),
            )
    except urllib.error.HTTPError as exc:  # 4xx/5xx bodies are useful diagnostics
        return HTTPResponse(
            exc.code,
            exc.read().decode("utf-8", errors="replace"),
            dict(exc.headers.items()) if exc.headers is not None else {},
        )


def classify_http_error(status: int, provider_code: str | None) -> str:
    """Classify a non-2xx response using Zhipu business codes before status.

    A 429 is NOT assumed to be a plain rate limit: Zhipu reports billing,
    quota, and overload conditions over 429 as well.
    """

    if provider_code is not None:
        mapped = ZHIPU_CODE_MAP.get(provider_code)
        if mapped is not None:
            return mapped
        if provider_code.isdigit() and (
            ZHIPU_QUOTA_BAND[0] <= int(provider_code) <= ZHIPU_QUOTA_BAND[1]
        ):
            return ERROR_HTTP_QUOTA  # subscription/quota/permission band
    if status in (401,):
        return ERROR_HTTP_AUTH
    if status in (403,):
        return ERROR_HTTP_PERMISSION
    if status == 429:
        return ERROR_RATE_LIMIT
    if status >= 500:
        return ERROR_SERVER
    return ERROR_HTTP_OTHER


def _parse_error_body(body: str) -> tuple[str | None, str | None]:
    """Extract (provider_error_code, provider_error_message) from the body.

    Tolerates the documented ``{"error": {"code", "message"}}`` shape, an
    ``{"error": "text"}`` shape, flat ``code``/``message`` keys, and malformed
    bodies — never raising on garbage.
    """

    try:
        document = json.loads(body)
    except ValueError:
        return None, (body.strip()[:200] or None)
    if not isinstance(document, dict):
        return None, (body.strip()[:200] or None)
    error = document.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message")
        return (
            str(code) if code is not None else None,
            str(message) if message is not None else None,
        )
    if isinstance(error, str):
        return None, (error[:200] or None)
    code = document.get("code")
    message = document.get("message")
    return (
        str(code) if code is not None else None,
        str(message) if message is not None else None,
    )


def _safe_headers(headers: dict[str, str]) -> dict[str, str]:
    """Whitelist rate-limit/request-id headers; drop everything else."""

    return {
        name.lower(): str(headers[name])
        for name in headers
        if name.lower() in SAFE_RESPONSE_HEADERS
    }


def _http_error(response: HTTPResponse) -> ProviderError:
    """Build a fully-detailed ProviderError for a non-2xx GLM response."""

    code, provider_message = _parse_error_body(response.body)
    category = classify_http_error(response.status, code)
    lowered = {name.lower(): value for name, value in response.headers.items()}
    request_id = lowered.get("request-id") or lowered.get("x-request-id")
    text = f"HTTP {response.status}"
    if code is not None:
        text += f" provider code {code}"
    if provider_message is not None:
        text += f": {provider_message}"
    return ProviderError(
        category,
        text,
        http_status=response.status,
        provider_error_code=code,
        provider_error_message=provider_message,
        request_id=request_id,
        http_headers=_safe_headers(response.headers) or None,
    )


class GLM46VFlashProvider:
    """Annotate episodes via the official zhipu_bigmodel chat-completions API."""

    def __init__(
        self,
        config: GLMProviderConfig | None = None,
        *,
        transport: Callable[[HTTPRequest, float], HTTPResponse] | None = None,
        api_key: str | None = None,
    ) -> None:
        self.config = config or GLMProviderConfig()
        self.provider_name = PROVIDER_NAME
        self.model = self.config.model
        self._transport = transport or urllib_transport
        self._api_key = api_key

    def annotate(self, request: AnnotationRequest) -> ModelAnnotation:
        key = (
            self._api_key
            if self._api_key is not None
            else resolve_api_key(self.config.environment)
        )
        payload = self._build_payload(request)
        attempts = 0
        last_error: ProviderError | None = None
        while attempts < self.config.max_attempts:
            attempts += 1
            try:
                response = self._post(payload, key)
                content, document, choice = self._extract_content(response)
                model_payload = self._parse_payload(content)
                return self._to_annotation(
                    request, model_payload, content, document, choice, attempts
                )
            except ProviderError as error:
                last_error = error
                if error.category in NON_RETRYABLE_ERRORS:
                    break
                if attempts < self.config.max_attempts:
                    self.config.sleeper(self.config.backoff_base_s**attempts)
        assert last_error is not None
        raise ProviderError(
            last_error.category,
            f"gave up after {attempts} attempt(s); last error: {last_error.message}",
            http_status=last_error.http_status,
            provider_error_code=last_error.provider_error_code,
            provider_error_message=last_error.provider_error_message,
            request_id=last_error.request_id,
            http_headers=last_error.http_headers,
        )

    def infer_json(self, request: AnnotationRequest) -> StructuredModelResponse:
        """Run the same bounded GLM transport but retain a generic JSON object."""

        key = (
            self._api_key
            if self._api_key is not None
            else resolve_api_key(self.config.environment)
        )
        payload = self._build_payload(request)
        attempts = 0
        last_error: ProviderError | None = None
        while attempts < self.config.max_attempts:
            attempts += 1
            try:
                response = self._post(payload, key)
                content, document, choice = self._extract_content(response)
                parsed = self._parse_payload(content)
                usage_raw = document.get("usage")
                usage = None
                if isinstance(usage_raw, dict):
                    usage = {
                        name: int(usage_raw[name])
                        for name in (
                            "prompt_tokens",
                            "completion_tokens",
                            "total_tokens",
                        )
                        if isinstance(usage_raw.get(name), int)
                    }
                return StructuredModelResponse(
                    provider=PROVIDER_NAME,
                    model=self.model,
                    prompt_version=request.prompt_version,
                    payload=parsed,
                    request_id=(
                        str(document["id"])
                        if isinstance(document.get("id"), str)
                        else None
                    ),
                    attempt_count=attempts,
                    finish_reason=(
                        str(choice["finish_reason"])
                        if isinstance(choice.get("finish_reason"), str)
                        else None
                    ),
                    usage=usage or None,
                )
            except ProviderError as error:
                last_error = error
                if error.category in NON_RETRYABLE_ERRORS:
                    break
                if attempts < self.config.max_attempts:
                    self.config.sleeper(self.config.backoff_base_s**attempts)
        assert last_error is not None
        raise ProviderError(
            last_error.category,
            f"gave up after {attempts} attempt(s); last error: {last_error.message}",
            http_status=last_error.http_status,
            provider_error_code=last_error.provider_error_code,
            provider_error_message=last_error.provider_error_message,
            request_id=last_error.request_id,
            http_headers=last_error.http_headers,
        )

    # ------------------------------------------------------------------ HTTP

    def _post(self, payload: dict[str, object], api_key: str) -> HTTPResponse:
        request = HTTPRequest(
            url=GLM_ENDPOINT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            body=json.dumps(payload),
        )
        try:
            return self._transport(request, self.config.request_timeout_s)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError(
                ERROR_NETWORK, f"request to GLM endpoint failed: {exc}"
            ) from exc

    def _build_payload(self, request: AnnotationRequest) -> dict[str, object]:
        content: list[dict[str, object]] = []
        images_used = 0
        for item in request.items:
            if isinstance(item, TextItem):
                content.append({"type": "text", "text": item.text})
                continue
            if (
                self.config.max_images_per_request is not None
                and images_used >= self.config.max_images_per_request
            ):
                break
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._encode_image(item)},
                }
            )
            images_used += 1
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "thinking": {"type": "enabled"},
        }

    @staticmethod
    def _encode_image(item: ImageItem) -> str:
        """Read local Curated JPG bytes and encode to base64 in memory only."""

        data = item.source_path.read_bytes()
        return base64.b64encode(data).decode("ascii")

    # -------------------------------------------------------------- response

    def _extract_content(
        self, response: HTTPResponse
    ) -> tuple[str, dict[str, object], dict[str, object]]:
        if not 200 <= response.status < 300:
            raise _http_error(response)
        try:
            document = json.loads(response.body)
        except ValueError as exc:
            raise ProviderError(
                ERROR_INVALID_JSON, f"response body is not JSON: {exc}"
            ) from exc
        if not isinstance(document, dict):
            raise ProviderError(ERROR_INVALID_JSON, "response body is not an object")
        choices = document.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError(ERROR_EMPTY_RESPONSE, "no choices in response")
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ProviderError(ERROR_EMPTY_RESPONSE, "empty message content")
        return content, document, choice

    @staticmethod
    def _parse_payload(content: str) -> dict[str, object]:
        """Safely extract one JSON object from raw or markdown-fenced model text."""

        text = content.strip()
        if text.startswith("```"):
            first_newline = text.find("\n")
            text = text[first_newline + 1 :] if first_newline != -1 else ""
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
        decoder = json.JSONDecoder()
        for offset in range(len(text)):
            if text[offset] != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text, offset)
            except ValueError:
                continue
            if isinstance(value, dict):
                return value
        raise ProviderError(ERROR_INVALID_JSON, "response contains no JSON object")

    def _to_annotation(
        self,
        request: AnnotationRequest,
        model_payload: dict[str, object],
        raw_text: str,
        document: dict[str, object],
        choice: dict[str, object],
        attempts: int,
    ) -> ModelAnnotation:
        try:
            validated = validate_model_payload(model_payload)
        except AnnotationSchemaError as exc:
            raise ProviderError(ERROR_INVALID_SCHEMA, str(exc)) from exc
        usage_raw = document.get("usage")
        usage = None
        if isinstance(usage_raw, dict):
            usage = {
                key: int(usage_raw[key])
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                if isinstance(usage_raw.get(key), int)
            }
        return ModelAnnotation(
            provider=PROVIDER_NAME,
            model=self.model,
            prompt_version=request.prompt_version,
            instruction=str(validated["instruction"]),
            confidence=float(validated["confidence"]),
            task_type=(
                str(validated["task_type"])
                if validated["task_type"] is not None
                else None
            ),
            objects=tuple(str(item) for item in validated["objects"]),
            uncertainty=validated["uncertainty"],
            request_id=(
                str(document["id"]) if isinstance(document.get("id"), str) else None
            ),
            attempt_count=attempts,
            finish_reason=(
                str(choice["finish_reason"])
                if isinstance(choice.get("finish_reason"), str)
                else None
            ),
            usage=usage or None,
            raw_response_text=raw_text[:2000],
        )


class DeterministicMockVLMProvider:
    """Deterministic in-memory provider for tests; never touches the network."""

    provider_name = "deterministic_mock"
    model = "mock-vlm-1"

    def __init__(
        self,
        *,
        instruction: str = "Pick up the object and place it into the tray.",
        confidence: float = 0.9,
        task_type: str | None = "pick_and_place",
        objects: tuple[str, ...] = ("object", "tray"),
        failure: ProviderError | None = None,
        fail_times: int = 0,
    ) -> None:
        self.instruction = instruction
        self.confidence = confidence
        self.task_type = task_type
        self.objects = objects
        self.failure = failure
        self.fail_times = fail_times
        self.call_count = 0
        self.calls: list[AnnotationRequest] = []

    def annotate(self, request: AnnotationRequest) -> ModelAnnotation:
        self.call_count += 1
        self.calls.append(request)
        if self.failure is not None and self.call_count <= self.fail_times:
            raise self.failure
        return ModelAnnotation(
            provider=self.provider_name,
            model=self.model,
            prompt_version=request.prompt_version,
            instruction=self.instruction,
            confidence=self.confidence,
            task_type=self.task_type,
            objects=self.objects,
            uncertainty=None,
            request_id=f"mock-{uuid4().hex[:8]}",
            attempt_count=1,
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            raw_response_text=None,
        )


class DeterministicStructuredMockProvider:
    """Ordered in-memory responses for hierarchical annotation tests."""

    provider_name = "deterministic_mock"
    model = "mock-vlm-structured-1"

    def __init__(self, *payloads: dict[str, object]) -> None:
        self.payloads = list(payloads)
        self.calls: list[AnnotationRequest] = []

    def infer_json(self, request: AnnotationRequest) -> StructuredModelResponse:
        self.calls.append(request)
        if not self.payloads:
            raise ProviderError(ERROR_EMPTY_RESPONSE, "mock response queue exhausted")
        return StructuredModelResponse(
            provider=self.provider_name,
            model=self.model,
            prompt_version=request.prompt_version,
            payload=self.payloads.pop(0),
            request_id=f"mock-{len(self.calls)}",
            attempt_count=1,
            finish_reason="stop",
            usage=None,
        )
