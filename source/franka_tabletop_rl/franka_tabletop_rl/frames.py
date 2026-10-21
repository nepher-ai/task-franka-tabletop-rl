# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tool frame of the 2F-85 gripper, shared by the low-level policy and the planner.

The tracked frame is rigidly attached to ``base_link`` of the gripper. Its origin sits between the
finger pads and its axes are those of ``base_link``: +X is the approach direction, and the fingers
close along Y. With the gripper pointing down, +X is world -Z and the fingers close along world Y.
The robot root has no rotation, so root-frame poses and env-frame poses coincide up to the origin.
"""

from __future__ import annotations

import math

import torch

import isaaclab.utils.math as math_utils

EE_BODY = "base_link"
# Midpoint of the finger pads along the approach axis [m]. The pads span 0.112-0.150 m in the asset.
TCP_OFFSET_B = (0.131, 0.0, 0.0)
# ``base_link`` orientation (xyzw) that points the gripper down with the fingers closing along world Y.
GRIP_DOWN_XYZW = (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))


# Measured with a proportional controller on the IK action (gripper down, any yaw in +-90 deg):
# the tool reaches every target inside this horizontal radius [m] from the robot base, which
# shrinks with height. Closer than MIN_REACH_X_M the arm folds into its joint limits.
REACH_LOW_M = 0.80
REACH_HIGH_M = 0.68
REACH_Z_LOW_M = 0.20
REACH_Z_HIGH_M = 0.35
MIN_REACH_X_M = 0.30


def project_to_reach(pos: torch.Tensor) -> torch.Tensor:
    """Pull tool targets ``[N, 3]`` [m] into the region the gripper-down arm can track."""
    blend = ((pos[:, 2] - REACH_Z_LOW_M) / (REACH_Z_HIGH_M - REACH_Z_LOW_M)).clamp(0.0, 1.0)
    radius_max = REACH_LOW_M + (REACH_HIGH_M - REACH_LOW_M) * blend
    radius = torch.linalg.norm(pos[:, :2], dim=-1).clamp(min=1e-6)
    shrink = (radius_max / radius).clamp(max=1.0)
    projected = pos.clone()
    projected[:, :2] = pos[:, :2] * shrink.unsqueeze(-1)
    projected[:, 0] = projected[:, 0].clamp(min=MIN_REACH_X_M)
    return projected


def grip_down_quat(yaw: torch.Tensor, tilt_x: torch.Tensor | None = None, tilt_y: torch.Tensor | None = None):
    """Orientation (xyzw) of a downward gripper turned by ``yaw`` [rad] about world Z.

    ``tilt_x`` and ``tilt_y`` [rad] optionally lean the approach axis about world X and Y.
    Shape is ``[N, 4]``.
    """
    base = torch.tensor(GRIP_DOWN_XYZW, device=yaw.device, dtype=yaw.dtype).expand(yaw.shape[0], 4)
    zeros = torch.zeros_like(yaw)
    quat = base
    if tilt_x is not None or tilt_y is not None:
        lean_x = math_utils.quat_from_euler_xyz(tilt_x if tilt_x is not None else zeros, zeros, zeros)
        lean_y = math_utils.quat_from_euler_xyz(zeros, tilt_y if tilt_y is not None else zeros, zeros)
        quat = math_utils.quat_mul(math_utils.quat_mul(lean_x, lean_y), base)
    turn = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
    return math_utils.quat_mul(turn, quat)


def tcp_pose_b(robot, body_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Tool pose in the robot root frame: position [m] ``[N, 3]`` and xyzw quaternion ``[N, 4]``."""
    body_pos_w = robot.data.body_pos_w.torch[:, body_index]
    body_quat_w = robot.data.body_quat_w.torch[:, body_index]
    offset = torch.tensor(TCP_OFFSET_B, device=body_pos_w.device, dtype=body_pos_w.dtype)
    tcp_pos_w = body_pos_w + math_utils.quat_apply(body_quat_w, offset.expand_as(body_pos_w))
    return math_utils.subtract_frame_transforms(
        robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, tcp_pos_w, body_quat_w
    )
