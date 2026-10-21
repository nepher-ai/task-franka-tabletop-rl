# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation wrapper that exposes the benchmark outcome to eval-nav.

eval-nav reads ``task_completed`` and ``task_failed`` as tensors, one value per env, right after
each step. The env auto-resets a finished episode, so the flags are latched in the tracker and are
cleared only by an explicit ``reset``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class TabletopRLEvalEnv:
    """Pass-through wrapper with the evaluator's outcome interface."""

    def __init__(self, env) -> None:
        self._env = env

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @property
    def unwrapped(self):
        return self._env.unwrapped

    @property
    def _tracker(self):
        return self._env.unwrapped.command_manager.get_term("ee_pose").tracker

    @property
    def task_completed(self) -> torch.Tensor:
        """True where the task's success expression held for 0.5 s since the last reset."""
        return self._tracker.completed

    @property
    def task_failed(self) -> torch.Tensor:
        """True where a scored object was lost since the last reset."""
        return self._tracker.failed

    def task_progress(self) -> np.ndarray:
        """Latched subtask progress in ``[0, 1]``, one value per env."""
        return np.asarray(self._tracker.progress, dtype=np.float64).copy()

    def instructions(self) -> list[str]:
        """Instruction text of each env's task."""
        tracker = self._tracker
        return [tracker.instruction(index) for index in range(tracker.count)]

    def reset(self, *args, **kwargs):
        result = self._env.reset(*args, **kwargs)
        self._tracker.clear_outcomes()
        return result

    def step(self, action):
        return self._env.step(action)

    def close(self):
        return self._env.close()

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)


def wrap_for_eval(env) -> TabletopRLEvalEnv:
    """Wrap a hierarchical tabletop env for eval-nav."""
    return TabletopRLEvalEnv(env)
