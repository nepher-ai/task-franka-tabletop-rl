# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hierarchical environment: one stage holds every selected task."""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv

from franka_tabletop_rl.benchmark.bundle import load_benchmark, resolve_bundle
from franka_tabletop_rl.benchmark.stage_plan import plan_stage
from franka_tabletop_rl.scene.clone import planned_worlds
from franka_tabletop_rl.scene.scene_cfg import mount_stage


class HLTabletopEnv(ManagerBasedRLEnv):
    """Builds the stage plan once ``scene.num_envs`` and ``task_id`` are final.

    eval-nav sets that count after it creates the cfg. Play sets ``env.task_id`` after
    ``__post_init__``. ``env.highlight=on`` records up to 16 tasks.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        from .highlight import Highlight, HighlightCfg, highlight_envs

        cfg.sync_task_selection()
        bundle = resolve_bundle(cfg.env_id)
        benchmark = load_benchmark(bundle)
        layouts = [benchmark.task(task_id) for task_id in cfg.task_ids]
        plan = plan_stage(layouts, benchmark.stage, int(cfg.scene.num_envs))
        cfg.scene.env_spacing = plan.env_spacing
        mount_stage(cfg, bundle, benchmark.stage, plan)
        cfg.stage_plan = plan
        # Arm start noise follows the latest reset seed and the count of full-env resets (-1 before the first).
        # An auto-reset keeps the current round.
        self.start_noise_seed = int(cfg.seed or 0)
        self.start_noise_round = -1
        self._highlight_envs = ()
        if isinstance(cfg.highlight, HighlightCfg):
            self._highlight_envs = highlight_envs(
                len(cfg.task_ids), plan.num_envs, int(cfg.seed or 0), tasks=cfg.highlight_tasks
            )
        with planned_worlds(plan):
            super().__init__(cfg, render_mode, **kwargs)
        if self._highlight_envs:
            length = float(cfg.episode_length_s)
            self.video_recorders.append(Highlight(self, self._highlight_envs, length))

    def reset(self, env_ids=slice(None), *, seed: int | None = None, options=None):
        """Reset, and note the seed and round that the arm start noise is drawn from.

        eval-nav reseeds with the same seed on every round, so the round index keeps the starts apart.
        """
        if seed is not None:
            self.start_noise_seed = int(seed)
        if env_ids is None or isinstance(env_ids, slice):
            self.start_noise_round += 1
        return super().reset(env_ids, seed=seed, options=options)

    def _init_sim(self):
        if self._highlight_envs:
            from .highlight import enable_rendering

            enable_rendering(self.sim)
        super()._init_sim()
