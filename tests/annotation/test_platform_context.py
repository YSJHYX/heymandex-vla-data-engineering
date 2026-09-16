"""D4.3.1 platform interpretation is stable, useful, and label-free."""

from __future__ import annotations

import json

from vla_data.annotation.platform_context import (
    DEFAULT_PLATFORM_CONTEXT,
    PlatformContext,
)


def test_default_context_and_fingerprint_are_canonical() -> None:
    context = DEFAULT_PLATFORM_CONTEXT
    assert context.as_dict() == {
        "robot_arm": "RealMan RM65B",
        "robot_arm_dof": 6,
        "end_effector": "SG100 dexterous hand",
        "end_effector_dof": 11,
        "head_camera_role": "external_global_workstation_camera",
        "head_camera_device": "Intel RealSense D455",
        "right_wrist_camera_role": "robot_mounted_wrist_end_effector_camera",
        "right_wrist_camera_device": "Intel RealSense D405",
        "collection_mode": "real_world_vla_teleoperation",
    }
    clone = PlatformContext(**json.loads(json.dumps(context.as_dict())))
    assert clone.fingerprint == context.fingerprint
    assert len(context.fingerprint) == 64


def test_renderer_explains_platform_without_episode_label_leakage() -> None:
    rendered = DEFAULT_PLATFORM_CONTEXT.render()
    for required in (
        "RealMan RM65B",
        "SG100",
        "external/global workstation camera",
        "wrist-mounted camera",
        "real-world VLA teleoperation",
        "camera ego-motion",
        "Human hands are not part of the robot embodiment",
    ):
        assert required in rendered
    lowered = rendered.lower()
    for episode_specific in ("connector", "device", "grasp", "plug", "insert"):
        assert episode_specific not in lowered
    for secret_name in ("z_ai_api_key", "glm_api_key", "authorization", "token"):
        assert secret_name not in lowered
