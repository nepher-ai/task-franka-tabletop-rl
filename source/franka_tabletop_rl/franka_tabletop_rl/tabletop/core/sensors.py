# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exterior, wrist, and front cameras.

The left exterior camera is the over-shoulder view at (0.05, 0.57, 0.66) [m]
in the env frame. The right exterior camera is that view mirrored across the
robot's center plane, at (0.05, -0.57, 0.66) [m], with the same lens. The wrist
camera is parented to the gripper base link. The front camera is RoboLab's
viewport: 1.5 m in front of the robot and 1.0 m up, looking back at the table.
It is recorded and is not sent to the brain. The env origin is the table
surface, so these positions are not shifted up with the table. The exterior and
wrist cameras render at 720x1280. A recording run writes the front, left exterior,
and wrist views as one 720x1280 frame per shard, four by four cells of 180x320,
and renders the front camera at that cell size.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg, ContactSensorCfg

# Apertures are centimeters. Focal lengths are 2.1 (exterior), 2.8 (wrist), and 24 (front).
_EXTERIOR_LENS = dict(focal_length=2.1, focus_distance=28.0, horizontal_aperture=5.376, vertical_aperture=3.024)
_WRIST_LENS = dict(focal_length=2.8, focus_distance=28.0, horizontal_aperture=5.376, vertical_aperture=3.024)
_FRONT_LENS = dict(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, vertical_aperture=15.29)


def exterior_left_camera() -> CameraCfg:
    """Fixed over-shoulder view of the table from the left."""
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
    """Over-shoulder view from the right, the left camera mirrored across the center plane.

    The look direction is the left camera's look direction with Y flipped, and the
    image stays upright. This is the second exterior view FLUX 3 Action was trained on.
    """
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
    """Contact force [N] between one Robotiq inner finger and the tabletop objects.

    ``prim_path`` must resolve to one body. The scene fills ``filter_prim_paths_expr``
    with one prim per placement, in placement order, before the simulation starts.
    """
    return ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*" + name,
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=[],
    )


def front_camera() -> CameraCfg:
    """Front view of the table, matching RoboLab's egocentric mirrored viewport."""
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
    """Camera on the gripper base. The prim name avoids a baked camera in the robot USD."""
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
