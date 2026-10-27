# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Planner command terms with the tracker's command names and shapes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from franka_tabletop_rl.benchmark.bundle import DEFAULT_ENV_ID, load_benchmark, resolve_bundle
from franka_tabletop_rl.frames import EE_BODY, grip_down_quat, tcp_pose_b
from franka_tabletop_rl.scene.robot import GRIPPER_CLOSED_RAD

from ..ee_speed import EeSpeedTrace
from ..planner import ClassicalTabletopPlanner, build_planner
from ..tracker import TaskTracker

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class HLPoseCommand(CommandTerm):
    """Tool pose ``[N, 7]`` (position [m], xyzw). Also owns the tracker and planner."""

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
        self._ee_speed = EeSpeedTrace(self.num_envs)
        self._ee_speed_step = -1
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
        self._ee_speed.reset(_env_indices(env_ids, self.num_envs))
        extras = super().reset(env_ids)
        if env_ids is None or isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device)
        self.planner.reset(ids.cpu().numpy())
        self.grip_target[ids] = 0.0
        return extras

    def discard_ee_speed(self) -> None:
        """Drop tool-speed samples on the next reset. The evaluator calls this before ``reset``."""
        self._ee_speed.discard()

    def episode_ee_speeds(self) -> list[np.ndarray]:
        """Tool speed samples [m/s] of the finished episode, or the live one if it has not reset."""
        return self._ee_speed.series()

    def note_ee_speed(self) -> None:
        """Record the tool position once per control step, before a reset can teleport it."""
        step = int(self._env.common_step_counter)
        if step == self._ee_speed_step:
            return
        self._ee_speed_step = step
        pos, _ = tcp_pose_b(self.robot, self.body_idx)
        self._ee_speed.observe(pos.detach().cpu().numpy(), float(self._env.step_dt))

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
        pos_np = pos.detach().cpu().numpy()
        angle = self.robot.data.joint_pos.torch[:, self.finger_idx]
        closure = (angle / GRIPPER_CLOSED_RAD).clamp(0.0, 1.0)
        target, yaw, grip = self.planner.step(pos_np, closure.detach().cpu().numpy())
        self.pose_command_b[:, :3] = torch.as_tensor(target, device=self.device, dtype=torch.float32)
        yaw_t = torch.as_tensor(yaw, device=self.device, dtype=torch.float32)
        self.pose_command_b[:, 3:] = math_utils.quat_unique(grip_down_quat(yaw_t))
        self.grip_target[:, 0] = torch.as_tensor(grip, device=self.device, dtype=torch.float32)


@configclass
class HLPoseCommandCfg(CommandTermCfg):
    class_type: type = HLPoseCommand
    asset_name: str = "robot"
    body_name: str = EE_BODY
    env_id: str = DEFAULT_ENV_ID
    """EnvHub id whose nepher cache holds the tasks."""
    task_ids: tuple[str, ...] = ()
    """Tasks of the stage. Env ``i`` runs task ``i % T``."""
    pose_jitter_m: float = 0.0
    """Per-episode shift [m] of each object in the table plane. Zero keeps the benchmark layout."""
    layout_seed: int = 1


def _env_indices(env_ids: Sequence[int] | slice | None, num_envs: int) -> list[int]:
    if env_ids is None or isinstance(env_ids, slice):
        return list(range(num_envs))
    return [int(index) for index in torch.as_tensor(env_ids).reshape(-1).tolist()]


class HLGripCommand(CommandTerm):
    """Gripper target ``[N, 1]`` (0 open, 1 closed), copied from the pose term."""

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
    class_type: type = HLGripCommand
    pose_command_name: str = "ee_pose"
