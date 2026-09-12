"""Versioned annotation prompts; task vocabulary stays domain-neutral."""

from __future__ import annotations

from vla_data.annotation.provider import ImageItem, TextItem

PROMPT_VERSION = "task_instruction_v1"

_TASK_CONTRACT = """\
You are labeling a robot manipulation demonstration for \
Vision-Language-Action training.

The images show one episode in chronological order. Each temporal point \
contains two views:
1. head-camera view
2. wrist-camera view

Infer the overall manipulation task demonstrated across the entire episode.

Return ONLY one JSON object with exactly this shape:
{
  "instruction": "<short imperative task instruction in English>",
  "confidence": <number between 0.0 and 1.0>,
  "task_type": "<short task category> or null",
  "objects": ["<object name>", "..."],
  "uncertainty": "<what is unclear, if anything>" or null
}

Requirements:
- instruction must be short and imperative
- describe the task goal, not robot motion
- identify objects only when visually supported
- do not mention cameras, frames, joints or timestamps
- do not invent unseen details
- if uncertain, lower confidence

Return the JSON object and nothing else."""

_OUTPUT_CONTRACT = (
    "Respond with only the JSON object. No explanations, no markdown fences."
)


def build_prompt_items(
    keyframe_items: tuple[ImageItem, ...],
    *,
    prompt_version: str = PROMPT_VERSION,
) -> tuple[TextItem | ImageItem, ...]:
    """Interleave explicit temporal/camera labels with the image items."""

    items: list[TextItem | ImageItem] = [
        TextItem(
            "The following images present one robot manipulation episode "
            "as ordered temporal points; within every point the first image "
            "is the head-camera view and the second is the wrist-camera view."
        )
    ]
    current_point = -1
    for item in keyframe_items:
        if item.temporal_point != current_point:
            current_point = item.temporal_point
            items.append(TextItem(f"Temporal point {current_point} of the episode:"))
        items.append(TextItem(f"{item.label}:"))
        items.append(item)
    items.append(TextItem(_TASK_CONTRACT))
    items.append(TextItem(_OUTPUT_CONTRACT))
    return tuple(items)
