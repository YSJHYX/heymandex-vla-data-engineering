from vla_data.morphology.rm65_sg100 import (
    ARM_DOF,
    ARM_JOINT_NAMES,
    ARM_SLICE,
    HAND_DOF,
    HAND_JOINT_NAMES,
    HAND_SLICE,
    PHYSICAL_DOF,
    ROBOT_JOINT_NAMES,
)


def test_rm65_sg100_dimensions() -> None:
    assert ARM_DOF == 6
    assert HAND_DOF == 11
    assert PHYSICAL_DOF == 17


def test_joint_ordering() -> None:
    assert ROBOT_JOINT_NAMES == ARM_JOINT_NAMES + HAND_JOINT_NAMES

    assert ROBOT_JOINT_NAMES[:6] == (
        "rm65_joint1",
        "rm65_joint2",
        "rm65_joint3",
        "rm65_joint4",
        "rm65_joint5",
        "rm65_joint6",
    )

    assert ROBOT_JOINT_NAMES[6:] == (
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


def test_physical_slices() -> None:
    assert ARM_SLICE == (0, 6)
    assert HAND_SLICE == (6, 17)
