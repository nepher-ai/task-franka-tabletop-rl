# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms of the low-level tracker."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from franka_tabletop_rl.frames import tcp_pose_b
from franka_tabletop_rl.scene.robot import GRIPPER_CLOSED_RAD

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def tcp_pose(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Tool pose in the robot root frame: position [m] then xyzw quaternion with ``w >= 0``, ``[N, 7]``."""
    pos, quat = tcp_pose_b(env.scene[asset_cfg.name], asset_cfg.body_ids[0])
    return torch.cat([pos, math_utils.quat_unique(quat)], dim=-1)


def tcp_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Target minus current tool pose: position [m] then rotation vector [rad], ``[N, 6]``."""
    command = env.command_manager.get_command(command_name)
    pos, quat = tcp_pose_b(env.scene[asset_cfg.name], asset_cfg.body_ids[0])
    pos_error, rot_error = math_utils.compute_pose_error(pos, quat, command[:, :3], command[:, 3:7])
    return torch.cat([pos_error, rot_error], dim=-1)


def gripper_closure(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Finger joint angle as a fraction of the closed angle: 0 open, 1 closed, ``[N, 1]``."""
    angle = env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids[0]]
    return (angle / GRIPPER_CLOSED_RAD).clamp(0.0, 1.0).unsqueeze(-1)
