# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play env with the film recorder."""

from __future__ import annotations

import torch

from ..tasks.hl_policy.env import HLTabletopEnv
from .latch import PostRollLatch
from .render import enable_film_rendering
from .shots import CONTROL_HZ


class FilmTabletopEnv(HLTabletopEnv):
    """:class:`HLTabletopEnv` that films. ``cfg.film`` must be a :class:`FilmSettings`.

    The success and lost terminations wait for the shot's post-roll through ``film_latch``, so a take has
    footage after the outcome. Scout and shoot use the same post-roll, which keeps their resets in step.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        if cfg.film is None:
            raise ValueError("FilmTabletopEnv needs cfg.film; call apply_film_cfg first")
        self.film = cfg.film
        self.film_latch: PostRollLatch | None = None
        self.film_recorder = None
        super().__init__(cfg, render_mode, **kwargs)
        self.film_recorder.bind()

    def _init_sim(self):
        if self.film.needs_camera:
            enable_film_rendering(self.sim, self.film.antialiasing, self.film.shot.partitioned())
        super()._init_sim()

    def load_managers(self):
        steps = round(self.film.shot.post_roll_s * CONTROL_HZ)
        count = int(self.scene.num_envs)
        outcome = torch.zeros(count, dtype=torch.int64, device=self.device)
        outcome_step = torch.full((count,), -1, dtype=torch.int64, device=self.device)
        self.film_latch = PostRollLatch(outcome, outcome_step, steps)
        super().load_managers()
