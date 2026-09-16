"""Review-only diagnostics never invent or delete semantic phases."""

from vla_data.annotation.semantic_diagnostics import (
    task_sequence_mismatch,
    very_short_nontraining_intervals,
)


def _annotation(task: str, segments: list[str], intervals=None) -> dict:
    return {
        "episode_task": {"instruction": task},
        "semantic_segments": [{"instruction": text} for text in segments],
        "non_training_intervals": intervals or [],
    }


def test_obvious_goal_versus_reach_only_sequence_is_flagged() -> None:
    assert task_sequence_mismatch(
        _annotation(
            "Plug the connector into the device",
            ["Reach toward the device", "Reach toward the connector"],
        )
    )
    assert task_sequence_mismatch(
        _annotation("Grasp the object", ["Reach toward the object"])
    )


def test_consistency_diagnostic_does_not_require_a_fixed_plan() -> None:
    assert not task_sequence_mismatch(
        _annotation("Plug the component into the port", ["Align the component"])
    )
    assert not task_sequence_mismatch(
        _annotation("Grasp the object", ["Hold the object"])
    )
    assert not task_sequence_mismatch(
        _annotation("Inspect the workspace", ["Reach toward the object"])
    )


def test_one_transition_nontraining_interval_is_warning_candidate() -> None:
    annotation = _annotation(
        "Reach toward the object",
        ["Reach toward the object"],
        [
            {
                "start_curated_index": 9,
                "end_curated_index": 10,
                "reason": "POST_TASK_IDLE",
            },
            {
                "start_curated_index": 10,
                "end_curated_index": 12,
                "reason": "AMBIGUOUS_VISUAL_EVIDENCE",
            },
        ],
    )
    assert very_short_nontraining_intervals(annotation) == [
        {
            "start_curated_index": 9,
            "end_curated_index": 10,
            "reason": "POST_TASK_IDLE",
            "length": 1,
        }
    ]
