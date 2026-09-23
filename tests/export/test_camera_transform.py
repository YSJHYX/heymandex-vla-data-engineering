from __future__ import annotations

import numpy as np

from vla_data.export.camera_transform import (
    WRIST_RGB_EXPORT_ROTATION_DEG,
    apply_camera_export_transform,
    camera_export_transform_contract,
)


def test_wrist_rotation_is_exact_for_asymmetric_pixels() -> None:
    image = np.asarray(
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
        ],
        dtype=np.uint8,
    )

    transformed = apply_camera_export_transform(image, camera_role="wrist")

    assert WRIST_RGB_EXPORT_ROTATION_DEG == 180
    assert np.array_equal(transformed, np.rot90(image, 2))
    assert transformed.dtype == image.dtype
    assert transformed.shape == image.shape


def test_head_is_pixel_exact_and_input_is_not_aliased() -> None:
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)

    transformed = apply_camera_export_transform(image, camera_role="head")

    assert np.array_equal(transformed, image)
    assert transformed.dtype == image.dtype
    assert transformed.shape == image.shape
    assert transformed is not image


def test_camera_contract_records_fixed_export_stage_and_reason() -> None:
    contract = camera_export_transform_contract()

    assert contract["head"] == {
        "rotation_deg": 0,
        "stage": "lerobot_export",
        "reason": "identity",
    }
    assert contract["wrist"] == {
        "rotation_deg": 180,
        "stage": "lerobot_export",
        "reason": "fixed_camera_mount_orientation",
    }
