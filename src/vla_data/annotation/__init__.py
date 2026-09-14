"""Model-first episode annotation layer (GLM-4.6V-Flash) with human QC."""

from vla_data.annotation.batch import (
    AnnotationConfigurationError,
    DatasetAnnotationResult,
    annotate_dataset,
)
from vla_data.annotation.eligibility import (
    EligibilityDecision,
    EligibilityError,
    evaluate_eligibility,
)
from vla_data.annotation.keyframes import (
    DEFAULT_MAX_TEMPORAL_POINTS,
    KeyframeSelection,
    TemporalKeyframe,
    keyframe_image_items,
    select_keyframes,
)
from vla_data.annotation.pipeline import (
    EpisodeAnnotationResult,
    annotate_episode,
)
from vla_data.annotation.prompts import PROMPT_VERSION, build_prompt_items
from vla_data.annotation.provider import (
    AnnotationRequest,
    ImageItem,
    ModelAnnotation,
    ProviderError,
    TextItem,
    VLMProvider,
)
from vla_data.annotation.qc import (
    QCReviewPlan,
    QCSamplingConfig,
    build_review_plan,
    review_annotation,
)
from vla_data.annotation.schema import (
    ANNOTATION_SCHEMA_NAME,
    ANNOTATION_SCHEMA_VERSION,
    REVIEW_AUTO_ACCEPTED,
    REVIEW_AUTO_LABELED,
    REVIEW_HUMAN_CORRECTED,
    REVIEW_HUMAN_VERIFIED,
    REVIEW_REJECTED,
    REVIEW_STATES,
    AnnotationSchemaError,
    apply_review,
    build_annotation,
    load_annotation,
    resolve_final_instruction,
    validate_annotation,
    write_annotation,
)

__all__ = [
    "ANNOTATION_SCHEMA_NAME",
    "ANNOTATION_SCHEMA_VERSION",
    "DEFAULT_MAX_TEMPORAL_POINTS",
    "GLM_ENDPOINT",
    "GLM_MODEL_ID",
    "PROMPT_VERSION",
    "PROVIDER_NAME",
    "REVIEW_AUTO_ACCEPTED",
    "REVIEW_AUTO_LABELED",
    "REVIEW_HUMAN_CORRECTED",
    "REVIEW_HUMAN_VERIFIED",
    "REVIEW_REJECTED",
    "REVIEW_STATES",
    "AnnotationConfigurationError",
    "AnnotationRequest",
    "AnnotationSchemaError",
    "DatasetAnnotationResult",
    "DeterministicMockVLMProvider",
    "EligibilityDecision",
    "EligibilityError",
    "EpisodeAnnotationResult",
    "GLM46VFlashProvider",
    "GLMProviderConfig",
    "ImageItem",
    "KeyframeSelection",
    "MissingAPIKeyError",
    "ModelAnnotation",
    "ProviderError",
    "QCReviewPlan",
    "QCSamplingConfig",
    "TemporalKeyframe",
    "TextItem",
    "VLMProvider",
    "annotate_dataset",
    "annotate_episode",
    "apply_review",
    "build_annotation",
    "build_prompt_items",
    "build_review_plan",
    "evaluate_eligibility",
    "keyframe_image_items",
    "load_annotation",
    "resolve_api_key",
    "resolve_final_instruction",
    "review_annotation",
    "select_keyframes",
    "validate_annotation",
    "write_annotation",
]


def __getattr__(name: str):
    # Preserve the D4 public API without importing a network provider when a
    # downstream metadata stage imports annotation schema/eligibility helpers.
    if name in {
        "GLM_ENDPOINT",
        "GLM_MODEL_ID",
        "PROVIDER_NAME",
        "DeterministicMockVLMProvider",
        "GLM46VFlashProvider",
        "GLMProviderConfig",
        "MissingAPIKeyError",
        "resolve_api_key",
    }:
        from vla_data.annotation import glm_provider

        return getattr(glm_provider, name)
    raise AttributeError(name)
