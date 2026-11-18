# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest

from franka_tabletop_rl.scene.cameras import (
    OBSERVATION_CAMERAS,
    requested_cameras,
    selected_cameras,
)


def test_front_camera_is_an_observation():
    assert OBSERVATION_CAMERAS[-1] == "cam_front"
    assert requested_cameras(None) == OBSERVATION_CAMERAS
    assert requested_cameras({"cameras": ["cam_front"]}) == ("cam_front",)


def test_selected_cameras_keeps_observation_order():
    assert selected_cameras(None) == ()
    assert selected_cameras(()) == ()
    assert selected_cameras(["cam_front", "cam_wrist"]) == ("cam_wrist", "cam_front")


def test_selected_cameras_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="not in"):
        selected_cameras(["overhead"])
