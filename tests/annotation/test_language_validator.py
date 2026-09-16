"""Deterministic controlled-language validation for training instructions."""

from __future__ import annotations

import pytest

from vla_data.annotation.language_validator import (
    CLASS_ACTUATOR,
    CLASS_EMBODIMENT,
    CLASS_EMPTY,
    CLASS_EVALUATIVE,
    CLASS_LENGTH,
    CLASS_MARKDOWN,
    CLASS_NARRATIVE_PREFIX,
    CLASS_PRONOUN,
    CLASS_SPECULATIVE,
    normalize_instruction,
    validate_instruction,
)


def categories(text: str) -> set[str]:
    return {violation.category for violation in validate_instruction(text)}


@pytest.mark.parametrize(
    "bad",
    [
        "The robot appears to be grasping the cable",
        "It seems that the arm moves forward",
        "We can see a cable on the table",
        "In this image the hand closes",
        "According to the image, the object is lifted",
        "I think the robot picks up the cable",
        "Maybe grasp the object",
        "Probably move the cable above the tray",
        "The robotic hand is currently holding the part",
    ],
)
def test_caption_or_speculative_prefixes_rejected(bad: str) -> None:
    assert categories(bad) & {
        CLASS_NARRATIVE_PREFIX,
        CLASS_SPECULATIVE,
        CLASS_EVALUATIVE,
    }


@pytest.mark.parametrize(
    "bad",
    [
        "Move joint 3",
        "Rotate joint 2 by 15 degrees",
        "Increase qpos on the arm",
        "Close the servo",
        "Command finger joint 5",
        "Send gripper position 0.5",
        "Rotate the wrist 20 degrees",
    ],
)
def test_actuator_narration_rejected(bad: str) -> None:
    assert CLASS_ACTUATOR in categories(bad)


@pytest.mark.parametrize(
    "bad",
    [
        "Move the robot arm toward the object",
        "Move the robotic arm toward the connector",
        "Move the robot hand closer",
        "Move the robotic hand closer",
        "Rotate the wrist toward the connector",
        "Move the end effector toward the object",
        "Position the robot arm near the device",
        "Reach the gripper toward the operator's palm",
    ],
)
def test_embodiment_centric_task_language_rejected(bad: str) -> None:
    assert CLASS_EMBODIMENT in categories(bad)


@pytest.mark.parametrize(
    "good",
    [
        "Reach toward the object",
        "Reach toward the connector",
        "Take the device from the operator",
        "Take the object from the operator's hand",
    ],
)
def test_embodiment_neutral_goal_and_operator_context_accepted(good: str) -> None:
    assert CLASS_EMBODIMENT not in categories(good)


@pytest.mark.parametrize(
    "bad",
    [
        "Successfully insert the connector",
        "Carefully place the cable into the tray",
        "Slowly press the foot pad into the slot",
        "Perfectly align the connector with the socket",
    ],
)
def test_evaluative_wording_rejected(bad: str) -> None:
    assert CLASS_EVALUATIVE in categories(bad)


@pytest.mark.parametrize(
    "bad",
    [
        "Pick it up",
        "Move it above the tray",
        "Insert it into the socket",
        "Release this",
    ],
)
def test_unnamed_pronoun_objects_rejected(bad: str) -> None:
    assert CLASS_PRONOUN in categories(bad)


@pytest.mark.parametrize(
    "bad",
    [
        "Grasp the cable `firmly`",
        "**Move the cable**",
        "# Insert the connector",
        'Grasp the {"object": "cable"}',
        "Grasp the [cable]",
    ],
)
def test_markdown_and_json_inside_string_rejected(bad: str) -> None:
    assert categories(bad) & {CLASS_MARKDOWN, CLASS_LENGTH}


def test_long_paragraph_rejected() -> None:
    long_text = " ".join(["Grasp the cable"] * 30)
    assert CLASS_LENGTH in categories(long_text)


def test_empty_rejected() -> None:
    assert CLASS_EMPTY in categories("")
    assert CLASS_EMPTY in categories("   ")


@pytest.mark.parametrize(
    "good",
    [
        "Grasp the cable",
        "Reach toward the cable",
        "Move the cable above the tray",
        "Align the connector with the socket",
        "Insert the connector into the socket",
        "Press the foot pad into the slot",
        "Hold the object in place",
        "Maintain pressure on the part",
        "Release the object",
        "Turn the valve handle",
        "Pull the drawer open",
        "Transfer the component to the tray",
        "Remove the protective cover from the slot",
    ],
)
def test_valid_task_semantic_instructions_pass(good: str) -> None:
    assert validate_instruction(good) == []


def test_no_narrow_verb_whitelist_future_domains_pass() -> None:
    # The validator constrains form, not vocabulary.
    for instruction in (
        "Seat the bearing into the housing",
        "Open the clasp on the box",
        "Wipe the sensor face with the cloth",
    ):
        assert validate_instruction(instruction) == []


def test_normalize_is_deterministic_and_light() -> None:
    assert normalize_instruction("  Grasp   the cable. ") == "Grasp the cable"
    assert normalize_instruction("Hold the object in place.") == (
        "Hold the object in place"
    )


def test_static_semantics_cases_accepted_as_training_language() -> None:
    # Case B: static hold is valid training semantics, not idle.
    assert validate_instruction("Hold the object in place") == []
    assert validate_instruction("Maintain pressure on the part") == []
