# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _tracker(env: ManagerBasedRLEnv, pose_command_name: str):
    term = env.command_manager.get_term(pose_command_name)
    term.note_ee_speed()
    tracker = term.tracker
    tracker.update()
    return tracker


def task_success(env: ManagerBasedRLEnv, pose_command_name: str = "ee_pose") -> torch.Tensor:
    """True once the success expression has held on the current state for 0.5 s."""
    return torch.as_tensor(_tracker(env, pose_command_name).success_now, device=env.device)


def object_lost(env: ManagerBasedRLEnv, pose_command_name: str = "ee_pose") -> torch.Tensor:
    """True when a scored object is off the table. It ends the episode as a failure and never counts as success."""
    tracker = _tracker(env, pose_command_name)
    return torch.as_tensor(tracker.lost & ~tracker.success_now, device=env.device)


def episode_timeout(env: ManagerBasedRLEnv, pose_command_name: str = "ee_pose") -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length
