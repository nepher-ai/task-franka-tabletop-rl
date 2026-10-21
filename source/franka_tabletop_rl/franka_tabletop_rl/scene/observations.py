# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the Franka joint-position embodiment.

Images are uint8 HWC. Joints are radians. The gripper is 0 when open and 1 when
closed at ``pi/4``. The end-effector pose is the gripper ``base_link`` in the
robot-root frame, then rotated into the tool frame: position [m] plus a
wxyz quaternion.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import convert_quat, quat_mul, subtract_frame_transforms

from .robot import ARM_JOINTS, GRIPPER_CLOSED_RAD

# Tool-frame offset in the simulator order (x, y, z, w). The observation stores wxyz.
# Its matrix is the DROID correction [[0, 0, -1], [-1, 0, 0], [0, 1, 0]] that GR00T's
# eef_9d state expects on the gripper base_link rotation.
_TOOL_OFFSET_XYZW = (0.5, -0.5, -0.5, 0.5)


def camera_rgb(env, sensor_name: str) -> torch.Tensor:
    """RGB image from a scene camera, shape [N, H, W, C]."""
    image = env.scene[sensor_name].data.output["rgb"]
    if image.shape[-1] not in (3, 4) and image.shape[1] in (3, 4):
        image = image.permute(0, 2, 3, 1)
    return image[..., :3].to(dtype=torch.uint8)


def joint_pos(env) -> torch.Tensor:
    """Arm joint positions [rad], shape [N, 7]."""
    robot = env.scene["robot"]
    indexes = _joint_indexes(robot, ARM_JOINTS)
    return robot.data.joint_pos[:, indexes]


def gripper_pos(env) -> torch.Tensor:
    """Finger joint mapped to 0..1, shape [N, 1]. 0 is open and 1 is closed."""
    robot = env.scene["robot"]
    index = _joint_indexes(robot, ["finger_joint"])
    opening = robot.data.joint_pos[:, index]
    return opening / GRIPPER_CLOSED_RAD


def eef_pose(env) -> torch.Tensor:
    """Tool pose in the robot-root frame, shape [N, 7]: xyz [m] and wxyz quaternion."""
    robot = env.scene["robot"]
    index = list(robot.data.body_names).index("base_link")
    position, quaternion = subtract_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        robot.data.body_pos_w.torch[:, index],
        robot.data.body_quat_w.torch[:, index],
    )
    offset = torch.tensor(_TOOL_OFFSET_XYZW, device=quaternion.device, dtype=quaternion.dtype)
    tool = quat_mul(quaternion, offset.expand_as(quaternion))
    return torch.cat((position, convert_quat(tool, to="wxyz")), dim=-1)


def resize_rgb(image: torch.Tensor, size: tuple[int, int] | None) -> torch.Tensor:
    """Antialiased resize of uint8 images [N, H, W, C] to ``size`` as (height, width) [pixels]."""
    if size is None or tuple(image.shape[1:3]) == tuple(size):
        return image
    planes = image.permute(0, 3, 1, 2).to(dtype=torch.float32)
    planes = torch.nn.functional.interpolate(planes, size=size, mode="bilinear", antialias=True, align_corners=False)
    return planes.round().clamp_(0, 255).to(dtype=torch.uint8).permute(0, 2, 3, 1).contiguous()


def _joint_indexes(robot, names: list[str]) -> list[int]:
    available = list(robot.data.joint_names)
    return [available.index(name) for name in names]
