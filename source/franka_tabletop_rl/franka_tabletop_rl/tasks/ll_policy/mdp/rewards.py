# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from franka_tabletop_rl.frames import tcp_pose_b

from .observations import gripper_closure

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _errors(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg):
    command = env.command_manager.get_command(command_name)
    pos, quat = tcp_pose_b(env.scene[asset_cfg.name], asset_cfg.body_ids[0])
    pos_error = torch.linalg.norm(command[:, :3] - pos, dim=-1)
    rot_error = math_utils.quat_error_magnitude(quat, command[:, 3:7])
    return pos_error, rot_error


def position_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Distance [m] between the tool and its target."""
    return _errors(env, command_name, asset_cfg)[0]


def position_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """``1 - tanh(distance / std)`` with ``std`` [m]."""
    return 1.0 - torch.tanh(_errors(env, command_name, asset_cfg)[0] / std)


def orientation_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Angle [rad] between the tool and its target orientation."""
    return _errors(env, command_name, asset_cfg)[1]


def orientation_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """``1 - tanh(angle / std)`` with ``std`` [rad]."""
    return 1.0 - torch.tanh(_errors(env, command_name, asset_cfg)[1] / std)


def gripper_tracking(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, std: float = 0.2
) -> torch.Tensor:
    """``1 - tanh(|closure - target| / std)``; the target is 0 for open and 1 for closed."""
    target = env.command_manager.get_command(command_name)
    return 1.0 - torch.tanh((gripper_closure(env, asset_cfg) - target).abs().squeeze(-1) / std)
