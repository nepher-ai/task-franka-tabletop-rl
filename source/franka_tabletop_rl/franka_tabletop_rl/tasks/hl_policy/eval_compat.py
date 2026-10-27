# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""eval-nav outcome flags. They latch until an explicit ``reset``, because the env auto-resets."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class TabletopRLEvalEnv:
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
        """True where success held for 0.5 s since the last reset."""
        return self._tracker.completed

    @property
    def task_failed(self) -> torch.Tensor:
        """True where a scored object was lost since the last reset."""
        return self._tracker.failed

    def task_progress(self) -> np.ndarray:
        """Latched progress in ``[0, 1]``, one value per env."""
        return np.asarray(self._tracker.progress, dtype=np.float64).copy()

    def instructions(self) -> list[str]:
        tracker = self._tracker
        return [tracker.instruction(index) for index in range(tracker.num_envs)]

    def episode_task_ids(self) -> list[str]:
        """Task id of the episode each env just finished."""
        return list(self._tracker.last_task_ids)

    def episode_finished(self):
        """``None``. Each env resets on its own."""
        return self._tracker.episode_finished()

    def episode_steps(self):
        """Steps at finish. ``-1`` means the env has not finished."""
        return self._tracker.episode_steps()

    def episode_ee_speeds(self):
        """Tool speed samples [m/s] per env, or ``None`` when the pose term does not record them."""
        report = getattr(self._pose_command, "episode_ee_speeds", None)
        if report is None:
            return None
        return report()

    def reset(self, *args, **kwargs):
        self._tracker.prepare_reset()
        discard = getattr(self._pose_command, "discard_ee_speed", None)
        if discard is not None:
            discard()
        result = self._env.reset(*args, **kwargs)
        self._tracker.clear_outcomes()
        return result

    @property
    def _pose_command(self):
        return self._env.unwrapped.command_manager.get_term("ee_pose")

    def step(self, action):
        return self._env.step(action)

    def close(self):
        return self._env.close()

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)


def wrap_for_eval(env) -> TabletopRLEvalEnv:
    return TabletopRLEvalEnv(env)
