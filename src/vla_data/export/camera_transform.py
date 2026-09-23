"""Fixed camera mounting corrections for the RM65B + SG100 export profile."""

from __future__ import annotations

import copy

import numpy as np

HEAD_CAMERA_ROLE = "head"
WRIST_CAMERA_ROLE = "wrist"
WRIST_RGB_EXPORT_ROTATION_DEG = 180

CAMERA_FEATURE_TO_EXPORT_ROLE = {
    "observation.images.head": HEAD_CAMERA_ROLE,
    "observation.images.wrist": WRIST_CAMERA_ROLE,
}

CAMERA_EXPORT_TRANSFORMS = {
    HEAD_CAMERA_ROLE: {
        "rotation_deg": 0,
        "stage": "lerobot_export",
        "reason": "identity",
    },
    WRIST_CAMERA_ROLE: {
        "rotation_deg": WRIST_RGB_EXPORT_ROTATION_DEG,
        "stage": "lerobot_export",
        "reason": "fixed_camera_mount_orientation",
    },
}


def camera_export_transform_contract() -> dict[str, dict[str, int | str]]:
    """Return an isolated serializable copy of the fixed export contract."""

    return copy.deepcopy(CAMERA_EXPORT_TRANSFORMS)


def has_current_camera_export_transform(value: object) -> bool:
    """Whether provenance states the complete current camera transform contract."""

    return value == CAMERA_EXPORT_TRANSFORMS


def apply_camera_export_transform(image: np.ndarray, *, camera_role: str) -> np.ndarray:
    """Apply the deterministic mounting correction before LeRobot video encoding."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("camera export image must have shape [H,W,3]")
    if camera_role == HEAD_CAMERA_ROLE:
        return array.copy()
    if camera_role == WRIST_CAMERA_ROLE:
        return np.rot90(array, 2).copy()
    raise ValueError(f"unsupported camera export role: {camera_role}")
