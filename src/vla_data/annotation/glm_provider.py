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
    ERROR_CLIENT_ERROR,
    ERROR_EMPTY_RESPONSE,
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
    TextItem,
)
from vla_data.annotation.schema import AnnotationSchemaError, validate_model_payload

PROVIDER_NAME = "zhipu_bigmodel"
GLM_MODEL_ID = "glm-4.6v-flash"  # API model id; never pad with whitespace.
GLM_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"


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
            return HTTPResponse(response.status, response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # 4xx/5xx bodies are useful diagnostics
        return HTTPResponse(exc.code, exc.read().decode("utf-8", errors="replace"))


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
        if response.status == 429:
            raise ProviderError(ERROR_RATE_LIMIT, "HTTP 429 rate limited")
        if 500 <= response.status < 600:
            raise ProviderError(ERROR_SERVER, f"HTTP {response.status} server error")
        if 400 <= response.status < 600:
            raise ProviderError(
                ERROR_CLIENT_ERROR,
                f"HTTP {response.status} client/config error",
            )
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
