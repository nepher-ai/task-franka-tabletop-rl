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

from .highlight_cfg import HighlightCfg
from .task_state import plan_tasks


class HLTabletopEnv(ManagerBasedRLEnv):
    """Manager environment with ``T`` benchmark tasks in one stage.

    The env count is only known when the env is built, because eval-nav sets ``scene.num_envs`` after it
    creates the cfg. So the plan of the ``T`` tasks is tiled here, and env ``i`` runs task ``i % T``.

    ``env.highlight=on`` adds video recording of up to 16 of these tasks. The stage is the same with and
    without it.

    Raises:
        ValueError: ``scene.num_envs`` is not a positive multiple of the number of tasks.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        from .highlight import highlight_envs

        plan = tile_plan(plan_tasks(resolve_bundle(cfg.env_id), cfg.task_ids), cfg.scene.num_envs)
        cfg.mixed_plan = plan
        # An unselected preset is still a preset object, and only a selected one is a highlight cfg.
        self._highlight_cfg = cfg.highlight if isinstance(cfg.highlight, HighlightCfg) else None
        self._highlight_envs: tuple[int, ...] = ()
        self._highlight_cameras = None
        if self._highlight_cfg is not None:
            self._highlight_envs = highlight_envs(len(cfg.task_ids), cfg.scene.num_envs, int(cfg.seed or 0))
        # Every env owns its object meshes, so the cloner must build one world per env.
        with planned_worlds(plan):
            super().__init__(cfg, render_mode, **kwargs)
        if self._highlight_cfg is not None:
            self._start_highlight(cfg)

    def _init_sim(self):
        if self._highlight_cfg is not None:
            from .highlight_cameras import register_highlight_renderer

            register_highlight_renderer(self.sim, self._highlight_cfg)
        super()._init_sim()

    def _start_highlight(self, cfg) -> None:
        """Film the highlighted envs. Only cameras and recorders are added, the stage is not changed."""
        from .highlight import highlight_recorders
        from .highlight_cameras import HighlightCameras
        from .highlight_recorder import HighlightRecorder

        self._highlight_cameras = HighlightCameras(
            self.sim, self._highlight_cfg, self._highlight_envs, cfg.scene.num_envs
        )
        recorders = highlight_recorders(float(cfg.episode_length_s), self.step_dt)
        self.video_recorders = [HighlightRecorder(item, self, self._highlight_cameras) for item in recorders]
        names = ", ".join(cfg.task_ids[index] for index in self._highlight_envs)
        print(f"[INFO] Highlight: {len(self._highlight_envs)} tasks for {cfg.episode_length_s} s")
        print(f"[INFO] Highlight tasks: {names}")

    def close(self):
        cameras, self._highlight_cameras = self._highlight_cameras, None
        if cameras is not None:
            cameras.close()
        super().close()
