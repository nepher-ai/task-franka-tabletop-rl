# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events of the hierarchical environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_tabletop_layout(env: ManagerBasedRLEnv, env_ids: torch.Tensor, pose_command_name: str = "ee_pose") -> None:
    """Lay the benchmark scene out in ``env_ids`` and clear their scoring state.

    Must run after ``reset_scene_to_default``: that event puts every object back at its spawn pose,
    and this one writes the scene's own poses (with jitter) and zero velocity over it.
    """
    env.command_manager.get_term(pose_command_name).tracker.begin(env_ids)
