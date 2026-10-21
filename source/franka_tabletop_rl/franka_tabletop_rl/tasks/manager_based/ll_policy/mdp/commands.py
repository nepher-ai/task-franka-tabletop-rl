# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command terms of the low-level tracker: a tool pose target and a gripper target."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from franka_tabletop_rl.frames import grip_down_quat, project_to_reach, tcp_pose_b

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class TcpPoseCommand(CommandTerm):
    """Target tool pose in the robot root frame, ``[N, 7]`` as position [m] and xyzw quaternion.

    Targets point the gripper down with a yaw and a small tilt. A share of the resamples steps
    away from the previous target instead of jumping across the workspace, which trains the short
    precise moves the planner issues.
    """

    cfg: TcpPoseCommandCfg

    def __init__(self, cfg: TcpPoseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 2] = 0.2
        self.pose_command_b[:, 3:] = grip_down_quat(torch.zeros(1, device=self.device))[0]
        self._low = torch.tensor(cfg.ranges.low, device=self.device)
        self._high = torch.tensor(cfg.ranges.high, device=self.device)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_b

    def _update_metrics(self):
        pos, quat = tcp_pose_b(self.robot, self.body_idx)
        self.metrics["position_error"] = torch.linalg.norm(self.pose_command_b[:, :3] - pos, dim=-1)
        self.metrics["orientation_error"] = math_utils.quat_error_magnitude(quat, self.pose_command_b[:, 3:])

    def _resample_command(self, env_ids: Sequence[int] | slice):
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        count = ids.numel()
        uniform = self._low + (self._high - self._low) * torch.rand(count, 3, device=self.device)
        step = (torch.rand(count, 3, device=self.device) * 2.0 - 1.0) * self.cfg.local_step_m
        local = torch.clamp(self.pose_command_b[ids, :3] + step, self._low, self._high)
        use_local = torch.rand(count, 1, device=self.device) < self.cfg.local_prob
        self.pose_command_b[ids, :3] = project_to_reach(torch.where(use_local, local, uniform))
        ranges = self.cfg.ranges
        yaw = (torch.rand(count, device=self.device) * 2.0 - 1.0) * ranges.yaw
        tilt = (torch.rand(count, 2, device=self.device) * 2.0 - 1.0) * ranges.tilt
        self.pose_command_b[ids, 3:] = math_utils.quat_unique(grip_down_quat(yaw, tilt[:, 0], tilt[:, 1]))

    def _update_command(self):
        pass


@configclass
class TcpPoseCommandCfg(CommandTermCfg):
    """Configuration of :class:`TcpPoseCommand`."""

    @configclass
    class Ranges:
        """Sampling box of the target, expressed in the robot root frame."""

        low: tuple[float, float, float] = (0.30, -0.45, 0.03)
        """Lower corner [m]."""
        high: tuple[float, float, float] = (0.80, 0.45, 0.45)
        """Upper corner [m]. Targets are then pulled inside the tracked reach."""
        yaw: float = math.pi / 2
        """Largest yaw [rad] of the gripper about world Z. The gripper is symmetric under a half turn."""
        tilt: float = 0.15
        """Largest lean [rad] of the approach axis about world X and Y."""

    class_type: type = TcpPoseCommand
    asset_name: str = "robot"
    body_name: str = "base_link"
    ranges: Ranges = Ranges()
    local_prob: float = 0.5
    """Share of resamples that stay within ``local_step_m`` of the previous target."""
    local_step_m: float = 0.12
    """Largest step [m] per axis of a local resample."""


class GripperCommand(CommandTerm):
    """Target gripper state per env: 0 open, 1 closed, shape ``[N, 1]``."""

    cfg: GripperCommandCfg

    def __init__(self, cfg: GripperCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._target = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._target

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int] | slice):
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        self._target[ids, 0] = (torch.rand(ids.numel(), device=self.device) < self.cfg.close_prob).float()

    def _update_command(self):
        pass


@configclass
class GripperCommandCfg(CommandTermCfg):
    """Configuration of :class:`GripperCommand`."""

    class_type: type = GripperCommand
    close_prob: float = 0.5
    """Probability that a resample asks for a closed gripper."""
