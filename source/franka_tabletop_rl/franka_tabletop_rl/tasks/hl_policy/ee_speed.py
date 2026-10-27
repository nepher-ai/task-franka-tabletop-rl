# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-episode tool speed samples for evaluation.

Speeds are the finite difference of the tool position [m/s]. An automatic reset latches the
episode that just ended. An explicit evaluator reset discards that latch so the next episode
starts empty. The first sample after a reset is a position only, so a teleport is not a speed.
"""

from __future__ import annotations

import numpy as np


class EeSpeedTrace:
    """Tool speeds [m/s] for each env, latched when that env resets."""

    def __init__(self, num_envs: int) -> None:
        self.num_envs = int(num_envs)
        self._speeds: list[list[float]] = [[] for _ in range(self.num_envs)]
        self._latched: list[list[float] | None] = [None] * self.num_envs
        self._prev: np.ndarray | None = None
        self._discard = False

    def discard(self) -> None:
        """Drop latched and live samples on the next reset. Used by an explicit evaluator reset."""
        self._discard = True

    def reset(self, env_ids: list[int]) -> None:
        """Latch finished episodes, or drop them when :meth:`discard` was called."""
        if self._discard:
            for index in env_ids:
                self._speeds[index] = []
                self._latched[index] = None
            self._discard = False
        else:
            for index in env_ids:
                if self._speeds[index]:
                    self._latched[index] = self._speeds[index]
                    self._speeds[index] = []
        if self._prev is not None:
            self._prev[env_ids] = np.nan

    def observe(self, position_m: np.ndarray, dt_s: float) -> None:
        """Record one tool position ``[N, 3]`` [m]. ``dt_s`` is the control step [s]."""
        position = np.asarray(position_m, dtype=np.float64).reshape(self.num_envs, 3)
        if self._prev is not None and dt_s > 0:
            delta = np.linalg.norm(position - self._prev, axis=-1)
            for index in range(self.num_envs):
                if np.isfinite(delta[index]):
                    self._speeds[index].append(float(delta[index] / dt_s))
        self._prev = position.copy()

    def series(self) -> list[np.ndarray]:
        """Speed samples [m/s] of the latched episode, or the live one when it has not reset."""
        out: list[np.ndarray] = []
        for index in range(self.num_envs):
            values = self._latched[index] if self._latched[index] is not None else self._speeds[index]
            out.append(np.asarray(values, dtype=np.float64))
        return out
