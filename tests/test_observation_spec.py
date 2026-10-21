# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera selection keeps observation order and rejects unknown names."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "source" / "franka_tabletop_rl"
sys.path.insert(0, str(PACKAGE))

from franka_tabletop_rl.tabletop.core.observation_spec import (  # noqa: E402
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
