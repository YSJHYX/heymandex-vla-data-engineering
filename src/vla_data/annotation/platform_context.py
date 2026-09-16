"""Stable platform interpretation metadata for semantic annotation.

This context explains what the cameras and embodiment are.  It deliberately
contains no episode task, object identity, action sequence, path, or secret.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PlatformContext:
    robot_arm: str
    robot_arm_dof: int
    end_effector: str
    end_effector_dof: int
    head_camera_role: str
    head_camera_device: str | None
    right_wrist_camera_role: str
    right_wrist_camera_device: str | None
    collection_mode: str

    def as_dict(self) -> dict[str, str | int | None]:
        """Return the canonical, secret-free provenance representation."""

        return asdict(self)

    @property
    def fingerprint(self) -> str:
        """Deterministic SHA-256 over canonical context JSON."""

        encoded = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def render(self) -> str:
        """Render observation interpretation context without semantic answers."""

        return f"""PLATFORM CONTEXT

You are annotating a real-world VLA teleoperation demonstration.

Robot embodiment:
- {self.robot_arm}, a {self.robot_arm_dof}-DoF robotic arm.
- {self.end_effector}, a {self.end_effector_dof}-DoF dexterous hand mounted as the end effector.
- Treat the arm and end-effector hand as one robot manipulation embodiment.

Camera roles:
- Storage role "head" is an external/global workstation camera ({self.head_camera_device}); it is not mounted on a humanoid or robot head.
- Storage role "right_wrist" is a wrist-mounted camera ({self.right_wrist_camera_device}) rigidly attached near the robot wrist/end effector and moves with it.
- Large apparent background motion in the right-wrist view may be camera ego-motion caused by {self.robot_arm} motion; it is not by itself object motion.
- Use the global head-camera view to disambiguate wrist-camera ego-motion when both views provide usable evidence.

Human/operator:
- Human hands may appear as the teleoperation operator or task assistant.
- Human hands are not part of the robot embodiment. Do not confuse human-hand motion with robot-hand motion.
- An object may be held, presented, stabilized, or handed over by a human; treat this only as scene context.

Collection mode:
- {self.collection_mode}.

Annotation objective:
- Act as a temporal VLA semantic annotator.
- Infer only observable manipulation intent and temporal task semantics.
- Describe WHAT is demonstrated, not robot-body motion, control commands, joint states, plans, or success claims.
- Platform context explains the observation setup; it does not reveal the episode task, object identity, or future action sequence."""


DEFAULT_PLATFORM_CONTEXT = PlatformContext(
    robot_arm="RealMan RM65B",
    robot_arm_dof=6,
    end_effector="SG100 dexterous hand",
    end_effector_dof=11,
    head_camera_role="external_global_workstation_camera",
    head_camera_device="Intel RealSense D455",
    right_wrist_camera_role="robot_mounted_wrist_end_effector_camera",
    right_wrist_camera_device="Intel RealSense D405",
    collection_mode="real_world_vla_teleoperation",
)
