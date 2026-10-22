# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Planner command terms: the tool pose and grip targets are written by the classical planner.

They keep the names and shapes of the low-level tracker's command terms, so a tracker checkpoint
trained on random targets drives this environment without change.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from franka_tabletop_rl.benchmark.bundle import DEFAULT_ENV_ID, load_benchmark, resolve_bundle
from franka_tabletop_rl.frames import EE_BODY, grip_down_quat, tcp_pose_b
from franka_tabletop_rl.scene.robot import GRIPPER_CLOSED_RAD

from ..planner import ClassicalTabletopPlanner, build_planner
from ..tracker import TaskTracker

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class HLPoseCommand(CommandTerm):
    """Tool pose target ``[N, 7]`` (position [m], xyzw quaternion) chosen by the planner each control step.

    The term also owns the task tracker and the planner, which the reset event and the termination
    terms reach through ``env.command_manager.get_term("ee_pose")``.
    """

    cfg: HLPoseCommandCfg

    def __init__(self, cfg: HLPoseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._env = env
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]
        self.finger_idx = self.robot.find_joints("finger_joint")[0][0]
        benchmark = load_benchmark(resolve_bundle(cfg.env_id))
        layouts = [benchmark.task(task_id) for task_id in cfg.task_ids]
        self.tracker = TaskTracker(
            env,
            layouts,
            env.cfg.stage_plan,
            pose_jitter_m=cfg.pose_jitter_m,
            seed=cfg.layout_seed,
        )
        self.planner: ClassicalTabletopPlanner = build_planner(env, self.tracker, benchmark.stage)
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 2] = 0.26
        self.pose_command_b[:, 3:] = grip_down_quat(torch.zeros(1, device=self.device))[0]
        self.grip_target = torch.zeros(self.num_envs, 1, device=self.device)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["progress"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_b

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        if env_ids is None or isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device)
        self.planner.reset(ids.cpu().numpy())
        self.grip_target[ids] = 0.0
        return extras

    def _update_metrics(self):
        pos, _ = tcp_pose_b(self.robot, self.body_idx)
        self.metrics["position_error"] = torch.linalg.norm(self.pose_command_b[:, :3] - pos, dim=-1)
        self.metrics["progress"] = torch.as_tensor(self.tracker.progress, device=self.device, dtype=torch.float32)

    def _resample_command(self, env_ids: Sequence[int] | slice):
        pass

    def _update_command(self):
        self.tracker.update()
        self.tracker.note_step(self._env.episode_length_buf.detach().cpu().numpy())
        pos, _ = tcp_pose_b(self.robot, self.body_idx)
        angle = self.robot.data.joint_pos.torch[:, self.finger_idx]
        closure = (angle / GRIPPER_CLOSED_RAD).clamp(0.0, 1.0)
        target, yaw, grip = self.planner.step(pos.detach().cpu().numpy(), closure.detach().cpu().numpy())
        self.pose_command_b[:, :3] = torch.as_tensor(target, device=self.device, dtype=torch.float32)
        yaw_t = torch.as_tensor(yaw, device=self.device, dtype=torch.float32)
        self.pose_command_b[:, 3:] = math_utils.quat_unique(grip_down_quat(yaw_t))
        self.grip_target[:, 0] = torch.as_tensor(grip, device=self.device, dtype=torch.float32)


@configclass
class HLPoseCommandCfg(CommandTermCfg):
    """Configuration of :class:`HLPoseCommand`."""

    class_type: type = HLPoseCommand
    asset_name: str = "robot"
    body_name: str = EE_BODY
    env_id: str = DEFAULT_ENV_ID
    """EnvHub id. The benchmark tasks are loaded from its nepher cache."""
    task_ids: tuple[str, ...] = ()
    """Benchmark tasks of the stage. Env ``i`` runs task ``i % T``, and every task runs equally often."""
    pose_jitter_m: float = 0.0
    """Shift [m] of each object in the table plane per episode. Zero reproduces the benchmark layout."""
    layout_seed: int = 1
    """Seed of the first jitter draw."""


class HLGripCommand(CommandTerm):
    """Gripper target ``[N, 1]`` (0 open, 1 closed) written by the planner through the pose term."""

    cfg: HLGripCommandCfg

    def __init__(self, cfg: HLGripCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._env = env

    @property
    def command(self) -> torch.Tensor:
        return self._env.command_manager.get_term(self.cfg.pose_command_name).grip_target

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int] | slice):
        pass

    def _update_command(self):
        pass


@configclass
class HLGripCommandCfg(CommandTermCfg):
    """Configuration of :class:`HLGripCommand`."""

    class_type: type = HLGripCommand
    pose_command_name: str = "ee_pose"
