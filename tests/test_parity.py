# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot and physics constants match ``task-franka-tabletop``. Needs Isaac Lab and that package."""

from __future__ import annotations

import pytest

pytest.importorskip("isaaclab")
pytest.importorskip("franka_tabletop")


def test_robot_and_physics_constants_match():
    from franka_tabletop.core import robot as original

    from franka_tabletop_rl.scene.robot import ARM_JOINTS, GRIPPER_CLOSED_RAD
    from franka_tabletop_rl.scene.scene_cfg import CONTROL_HZ, EPISODE_LENGTH_S

    assert GRIPPER_CLOSED_RAD == original.GRIPPER_CLOSED_RAD
    assert tuple(ARM_JOINTS) == tuple(original.ARM_JOINTS)
    assert CONTROL_HZ == 15
    assert EPISODE_LENGTH_S == 40.0

