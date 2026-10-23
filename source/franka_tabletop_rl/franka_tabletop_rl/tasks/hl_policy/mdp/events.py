# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_tabletop_layout(env: ManagerBasedRLEnv, env_ids: torch.Tensor, pose_command_name: str = "ee_pose") -> None:
    """Write this scene's poses over the spawn pose. Run after ``reset_scene_to_default``."""
    env.command_manager.get_term(pose_command_name).tracker.begin(env_ids)
