"""Canonical RM65 + SG100 physical morphology.

This module defines the physical 17-DoF robot ordering used by the
data-engineering pipeline.

Important:
- This is a physical robot representation.
- It is not a model action-space definition.
- Model-specific padding belongs outside this repository.
"""

from __future__ import annotations

ARM_JOINT_NAMES: tuple[str, ...] = (
    "rm65_joint1",
    "rm65_joint2",
    "rm65_joint3",
    "rm65_joint4",
    "rm65_joint5",
    "rm65_joint6",
)

HAND_JOINT_NAMES: tuple[str, ...] = (
    "thumb_j1",
    "thumb_j2",
    "thumb_j3",
    "index_j1",
    "index_j2",
    "index_j3",
    "middle_j1",
    "middle_j2",
    "little_j1",
    "little_j2",
    "little_j3",
)

ROBOT_JOINT_NAMES: tuple[str, ...] = ARM_JOINT_NAMES + HAND_JOINT_NAMES

ARM_DOF = len(ARM_JOINT_NAMES)
HAND_DOF = len(HAND_JOINT_NAMES)
PHYSICAL_DOF = len(ROBOT_JOINT_NAMES)

ARM_SLICE: tuple[int, int] = (0, ARM_DOF)
HAND_SLICE: tuple[int, int] = (ARM_DOF, PHYSICAL_DOF)

POSITION_UNIT = "rad"


if ARM_DOF != 6:
    raise AssertionError(f"RM65 DoF must be 6, got {ARM_DOF}")

if HAND_DOF != 11:
    raise AssertionError(f"SG100 DoF must be 11, got {HAND_DOF}")

if PHYSICAL_DOF != 17:
    raise AssertionError(f"RM65+SG100 physical DoF must be 17, got {PHYSICAL_DOF}")
