"""Provider-neutral annotation request/response contracts and error classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

ERROR_NETWORK = "NETWORK_ERROR"
ERROR_RATE_LIMIT = "HTTP_RATE_LIMIT"
ERROR_SERVER = "HTTP_SERVER_ERROR"
ERROR_CLIENT_ERROR = "HTTP_CLIENT_ERROR"
ERROR_INVALID_JSON = "INVALID_JSON"
ERROR_INVALID_SCHEMA = "INVALID_SCHEMA"
ERROR_EMPTY_RESPONSE = "EMPTY_RESPONSE"
ERROR_MODEL_OUTPUT_SCHEMA = "MODEL_OUTPUT_SCHEMA_ERROR"

# Fine-grained HTTP business-error classifications (distinct from transport).
ERROR_HTTP_AUTH = "HTTP_AUTH"
ERROR_HTTP_QUOTA = "HTTP_QUOTA"
ERROR_HTTP_OVERLOAD = "HTTP_PROVIDER_OVERLOAD"
ERROR_HTTP_PERMISSION = "HTTP_PERMISSION"
ERROR_HTTP_OTHER = "HTTP_OTHER"


# 4xx configuration/auth failures and hard billing/quota/permission errors
# must not burn the retry budget; rate limits and transient overload may.
NON_RETRYABLE_ERRORS = frozenset(
    {
        ERROR_CLIENT_ERROR,
        ERROR_HTTP_AUTH,
        ERROR_HTTP_QUOTA,
        ERROR_HTTP_PERMISSION,
        ERROR_HTTP_OTHER,
    }
)


class ProviderError(RuntimeError):
    """A provider failure with a stable classification and safe message.

    The rendered message must never contain credentials or request payloads.
    ``http_status`` / ``provider_error_code`` / ``provider_error_message`` /
    ``request_id`` / ``http_headers`` carry redacted, provider-reported
    business-error details for observability.
    """

    def __init__(
        self,
        category: str,
        message: str,
        *,
        http_status: int | None = None,
        provider_error_code: str | None = None,
        provider_error_message: str | None = None,
        request_id: str | None = None,
        http_headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(f"{category}: {message}")
        self.category = category
        self.message = message
        self.http_status = http_status
        self.provider_error_code = provider_error_code
        self.provider_error_message = provider_error_message
        self.request_id = request_id
        self.http_headers = http_headers

    def details(self) -> dict[str, object]:
        """Provider business-error fields for summaries/logs (secret-free)."""

        return {
            key: value
            for key, value in (
                ("http_status", self.http_status),
                ("provider_error_code", self.provider_error_code),
                ("provider_error_message", self.provider_error_message),
                ("request_id", self.request_id),
                ("http_headers", self.http_headers),
            )
            if value is not None
        }


# Canonical annotation camera roles — the single namespace used by the whole
# annotation subsystem. Storage field names (head_rgb_frame_index) and model
# keys (right_wrist_0_rgb) live in separate namespaces and must never be
# derived from these by string concatenation.
CAMERA_ROLE_HEAD = "head"
CAMERA_ROLE_RIGHT_WRIST = "right_wrist"
CANONICAL_CAMERA_ROLES = (CAMERA_ROLE_HEAD, CAMERA_ROLE_RIGHT_WRIST)


@dataclass(frozen=True)
class TextItem:
    """An ordered text segment of the annotation prompt."""

    text: str


@dataclass(frozen=True)
class ImageItem:
    """A local Curated JPG reference; bytes are read only inside the provider."""

    temporal_point: int
    camera: str
    label: str
    source_path: Path


@dataclass(frozen=True)
class AnnotationRequest:
    """One episode-level annotation request in provider-neutral form."""

    episode_id: str
    prompt_version: str
    items: tuple[TextItem | ImageItem, ...]
    transport_id: str | None = None
    image_count: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "image_count",
            sum(isinstance(item, ImageItem) for item in self.items),
        )


@dataclass(frozen=True)
class ModelAnnotation:
    """Validated model output plus text-only provenance (never payloads)."""

    provider: str
    model: str
    prompt_version: str
    instruction: str
    confidence: float
    task_type: str | None
    objects: tuple[str, ...]
    uncertainty: str | None
    request_id: str | None
    attempt_count: int
    finish_reason: str | None
    usage: dict[str, int] | None
    raw_response_text: str | None

    def model_block(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "instruction": self.instruction,
            "confidence": self.confidence,
            "task_type": self.task_type,
            "objects": list(self.objects),
            "uncertainty": self.uncertainty,
            "request_id": self.request_id,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "attempt_count": self.attempt_count,
        }


@dataclass(frozen=True)
class StructuredModelResponse:
    """Provider-neutral JSON response used by multi-pass annotation."""

    provider: str
    model: str
    prompt_version: str
    payload: dict[str, Any]
    request_id: str | None
    attempt_count: int
    finish_reason: str | None
    usage: dict[str, int] | None
    provider_metadata: dict[str, Any] | None = None


@runtime_checkable
class VLMProvider(Protocol):
    """The provider abstraction every annotation pipeline depends on."""

    provider_name: str
    model: str

    def annotate(self, request: AnnotationRequest) -> ModelAnnotation:
        """Annotate one request with bounded retries; raise ProviderError on give-up."""
        ...


@runtime_checkable
class StructuredVLMProvider(Protocol):
    """A VLM that returns parsed JSON without imposing the legacy v1 schema."""

    provider_name: str
    model: str

    def infer_json(self, request: AnnotationRequest) -> StructuredModelResponse:
        """Return one parsed JSON object or raise ProviderError."""
        ...
