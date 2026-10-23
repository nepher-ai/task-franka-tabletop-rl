# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exterior, wrist, and front cameras. Poses are in the env frame, whose origin is the table."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg, ContactSensorCfg

# Apertures are centimeters. Focal lengths are 2.1 (exterior), 2.8 (wrist), and 24 (front).
_EXTERIOR_LENS = dict(focal_length=2.1, focus_distance=28.0, horizontal_aperture=5.376, vertical_aperture=3.024)
_WRIST_LENS = dict(focal_length=2.8, focus_distance=28.0, horizontal_aperture=5.376, vertical_aperture=3.024)
_FRONT_LENS = dict(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, vertical_aperture=15.29)


def exterior_left_camera() -> CameraCfg:
    """Over-shoulder view from the left."""
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/CamExteriorLeft",
        update_period=0.0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(**_EXTERIOR_LENS),
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.57, 0.66),
            # (x, y, z, w). RoboLab authored (-0.393, -0.195, 0.399, 0.805) as (w, x, y, z).
            rot=(-0.195, 0.399, 0.805, -0.393),
            convention="opengl",
        ),
    )


def exterior_right_camera() -> CameraCfg:
    """Over-shoulder view from the right: the left camera mirrored across the center plane, kept upright."""
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/CamExteriorRight",
        update_period=0.0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(**_EXTERIOR_LENS),
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, -0.57, 0.66),
            # (x, y, z, w). Mirror of the left exterior across the XZ plane, kept upright.
            rot=(0.399, -0.195, -0.393, 0.805),
            convention="opengl",
        ),
    )


def finger_contact(name: str) -> ContactSensorCfg:
    """Contact force [N] on one inner finger. The scene fills one filter per object, in order."""
    return ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*" + name,
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=[],
    )


def front_camera() -> CameraCfg:
    """Front view of the table."""
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/CamFront",
        update_period=0.0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(**_FRONT_LENS),
        offset=CameraCfg.OffsetCfg(
            pos=(1.5, 0.0, 1.0),
            # (x, y, z, w). RoboLab authored (0.653, 0.271, 0.271, 0.653) as (w, x, y, z).
            rot=(0.271, 0.271, 0.653, 0.653),
            convention="opengl",
        ),
    )


def wrist_camera() -> CameraCfg:
    """Camera on the gripper base. The prim name avoids a camera baked into the robot USD."""
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link/CamWrist",
        update_period=0.0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(**_WRIST_LENS),
        offset=CameraCfg.OffsetCfg(
            pos=(0.011, -0.031, -0.074),
            # (x, y, z, w). RoboLab authored (-0.420, 0.570, 0.576, -0.409) as (w, x, y, z).
            rot=(0.570, 0.576, -0.409, -0.420),
            convention="opengl",
        ),
    )
