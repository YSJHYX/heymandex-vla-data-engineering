"""Provider-neutral annotation request/response contracts and error classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

ERROR_NETWORK = "NETWORK_ERROR"
ERROR_RATE_LIMIT = "HTTP_RATE_LIMIT"
ERROR_SERVER = "HTTP_SERVER_ERROR"
ERROR_CLIENT_ERROR = "HTTP_CLIENT_ERROR"
ERROR_INVALID_JSON = "INVALID_JSON"
ERROR_INVALID_SCHEMA = "INVALID_SCHEMA"
ERROR_EMPTY_RESPONSE = "EMPTY_RESPONSE"


# 4xx configuration/auth failures must not burn the whole retry budget.
NON_RETRYABLE_ERRORS = frozenset({ERROR_CLIENT_ERROR})


class ProviderError(RuntimeError):
    """A provider failure with a stable classification and safe message.

    The rendered message must never contain credentials or request payloads.
    """

    def __init__(self, category: str, message: str) -> None:
        super().__init__(f"{category}: {message}")
        self.category = category
        self.message = message


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


@runtime_checkable
class VLMProvider(Protocol):
    """The provider abstraction every annotation pipeline depends on."""

    provider_name: str
    model: str

    def annotate(self, request: AnnotationRequest) -> ModelAnnotation:
        """Annotate one request with bounded retries; raise ProviderError on give-up."""
        ...
