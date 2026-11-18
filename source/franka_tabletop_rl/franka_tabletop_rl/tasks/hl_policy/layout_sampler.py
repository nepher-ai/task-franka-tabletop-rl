# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Seeded layouts of the stage: one shared shift per env episode, kept on the work surface.

The env creates the sampler before the scene is built, so the first episode of every env spawns at its
jittered layout and the tracker's first ``begin`` takes the same one. Later episodes draw new layouts at reset.
"""

from __future__ import annotations

from collections.abc import Sequence

from franka_tabletop_rl.benchmark.stage import resolve_surface, table_rect
from franka_tabletop_rl.benchmark.stage_plan import StagePlan
from franka_tabletop_rl.benchmark.task import ShiftLimits, TaskLayout, shift_limits, with_jitter

TABLE_MARGIN_M = 0.03
"""Distance [m] that a shifted object keeps from the edge of the work surface."""


class LayoutSampler:
    """Layouts of the envs of one stage. Draws depend only on the seed, the env, and the episode seed.

    Args:
        layouts: The tasks of the stage, indexed as ``plan.assignment``.
        stage: The ``assets/stage.json`` payload, used for the work-surface box.
        plan: Stage plan with the task of every env.
        seed: Seed of the first layouts. Env ``i`` draws its first layout from ``seed + 1 + i``.
        jitter_x: Half-range [m] of the shared x shift.
        jitter_y: Half-range [m] of the shared y shift.
        margin_m: Distance [m] that objects keep from the surface edge.
    """

    def __init__(
        self,
        layouts: Sequence[TaskLayout],
        stage: dict,
        plan: StagePlan,
        seed: int = 1,
        jitter_x: float = 0.0,
        jitter_y: float = 0.0,
        margin_m: float = TABLE_MARGIN_M,
    ):
        self.jitter_x = float(jitter_x)
        self.jitter_y = float(jitter_y)
        self._layouts = tuple(layouts)
        self._assignment = tuple(int(task) for task in plan.assignment[0])
        self._seed = int(seed)
        self._limits: list[ShiftLimits] = []
        for layout in self._layouts:
            record = resolve_surface(stage, layout.surface)
            center, half = table_rect(record, layout.surface_pose)
            self._limits.append(shift_limits(layout, center, half, margin_m))

    def published(self, env: int) -> TaskLayout:
        """The layout of env ``env`` as the benchmark authored it."""
        return self._layouts[self._assignment[env]]

    def episode(self, env: int, episode_seed: int) -> TaskLayout:
        """The layout of env ``env`` for an episode drawn from ``episode_seed``."""
        task = self._assignment[env]
        return with_jitter(self._layouts[task], episode_seed, self.jitter_x, self.jitter_y, self._limits[task])

    def first(self, env: int) -> TaskLayout:
        """The layout that env ``env`` spawns with and starts its first episode in."""
        return self.episode(env, self._seed + 1 + env)

    def seed_of(self, episode: int) -> int:
        """Seed of the ``episode``-th reset draw, counted over all envs from 1."""
        return self._seed + episode
