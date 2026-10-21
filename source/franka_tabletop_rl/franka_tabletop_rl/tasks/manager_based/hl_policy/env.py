# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hierarchical environment whose one stage holds every benchmark task."""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv

from franka_tabletop_rl.tabletop.bundle_path import resolve_bundle
from franka_tabletop_rl.tabletop.core.clone import planned_worlds
from franka_tabletop_rl.tabletop.tiling import tile_plan

from .task_state import plan_tasks


class HLTabletopEnv(ManagerBasedRLEnv):
    """Manager environment with ``T`` benchmark tasks in one stage.

    The env count is only known when the env is built, because eval-nav sets ``scene.num_envs`` after it
    creates the cfg. So the plan of the ``T`` tasks is tiled here, and env ``i`` runs task ``i % T``.

    Raises:
        ValueError: ``scene.num_envs`` is not a positive multiple of the number of tasks.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        plan = tile_plan(plan_tasks(resolve_bundle(cfg.env_id), cfg.task_ids), cfg.scene.num_envs)
        cfg.mixed_plan = plan
        # Every env owns its object meshes, so the cloner must build one world per env.
        with planned_worlds(plan):
            super().__init__(cfg, render_mode, **kwargs)
