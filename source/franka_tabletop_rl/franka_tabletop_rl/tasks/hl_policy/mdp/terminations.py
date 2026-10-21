# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms of the hierarchical environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _tracker(env: ManagerBasedRLEnv, pose_command_name: str):
    tracker = env.command_manager.get_term(pose_command_name).tracker
    tracker.update()
    return tracker


def _multi_phase(env: ManagerBasedRLEnv, pose_command_name: str) -> bool:
    tracker = env.command_manager.get_term(pose_command_name).tracker
    return tracker.plan.phase_count > 1


def task_success(env: ManagerBasedRLEnv, pose_command_name: str = "ee_pose") -> torch.Tensor:
    """True once the benchmark's success expression has held for 0.5 s.

    Several backgrounds share the stage, so a finished env holds instead of resetting on its own.
    """
    if _multi_phase(env, pose_command_name):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return torch.as_tensor(_tracker(env, pose_command_name).success_now, device=env.device)


def object_lost(env: ManagerBasedRLEnv, pose_command_name: str = "ee_pose") -> torch.Tensor:
    """True when an object the task scores is off the table, so the task cannot finish."""
    if _multi_phase(env, pose_command_name):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    tracker = _tracker(env, pose_command_name)
    return torch.as_tensor(tracker.lost & ~tracker.success_now, device=env.device)


def episode_timeout(env: ManagerBasedRLEnv, pose_command_name: str = "ee_pose") -> torch.Tensor:
    """True when an env has used its episode budget and it resets on its own.

    A multi-background stage waits for every env, so this stays false and the barrier resets them
    together. That reset is not a timeout.
    """
    if _multi_phase(env, pose_command_name):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env.episode_length_buf >= env.max_episode_length


def phase_barrier(env: ManagerBasedRLEnv, pose_command_name: str = "ee_pose") -> torch.Tensor:
    """True for every env once the last one of this background has finished."""
    command = env.command_manager.get_term(pose_command_name)
    pending = bool(getattr(command, "barrier", None) is not None and command.barrier.pending)
    if not pending:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
