# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hierarchical environment whose one stage holds every selected task."""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv

from franka_tabletop_rl.benchmark.bundle import load_benchmark, resolve_bundle
from franka_tabletop_rl.benchmark.stage_plan import plan_stage
from franka_tabletop_rl.scene.clone import planned_worlds
from franka_tabletop_rl.scene.scene_cfg import mount_stage


class HLTabletopEnv(ManagerBasedRLEnv):
    """Manager environment with every selected task in one stage.

    ``scene.num_envs`` is only final when the env is built, because eval-nav sets it after it
    creates the cfg. The stage plan and the bodies are built here, before the scene is cloned.

    ``env.highlight=on`` records up to 16 tasks for one episode.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        from .highlight import Highlight, HighlightCfg, highlight_envs

        bundle = resolve_bundle(cfg.env_id)
        benchmark = load_benchmark(bundle)
        layouts = [benchmark.task(task_id) for task_id in cfg.task_ids]
        plan = plan_stage(layouts, benchmark.stage, int(cfg.scene.num_envs))
        cfg.scene.env_spacing = plan.env_spacing
        mount_stage(cfg, bundle, benchmark.stage, plan)
        cfg.stage_plan = plan
        self._highlight_envs = ()
        if isinstance(cfg.highlight, HighlightCfg):
            self._highlight_envs = highlight_envs(len(cfg.task_ids), plan.num_envs, int(cfg.seed or 0))
        with planned_worlds(plan):
            super().__init__(cfg, render_mode, **kwargs)
        if self._highlight_envs:
            length = float(cfg.episode_length_s)
            self.video_recorders.append(Highlight(self, self._highlight_envs, length))

    def _init_sim(self):
        if self._highlight_envs:
            from .highlight import enable_rendering

            enable_rendering(self.sim)
        super()._init_sim()
