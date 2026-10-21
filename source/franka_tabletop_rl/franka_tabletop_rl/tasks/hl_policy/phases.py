# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Background phases of one stage.

A finished env holds its pose until every env has finished. The dome then swaps, every env takes
its task for the next background, and the stage resets together. One background does none of this:
each env resets on its own, as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .tracker import TaskTracker


class PhaseBarrier:
    """Watch episode outcomes and request one reset of every env when a phase is over."""

    def __init__(self, tracker: TaskTracker, planner) -> None:
        self.tracker = tracker
        self.planner = planner
        self.pending = False

    @property
    def enabled(self) -> bool:
        return self.tracker.plan.phase_count > 1

    def update(self, episode_length: np.ndarray, max_length: int) -> bool:
        """Mark finished envs and request a phase change once all of them are done.

        Returns:
            True when the stage should reset into the next background.
        """
        if not self.enabled or self.pending:
            return self.pending
        lengths = np.asarray(episode_length).reshape(-1)
        for index in range(self.tracker.num_envs):
            if self.tracker.holding[index]:
                self.planner.hold(index)
                continue
            finished = bool(
                self.tracker.success_now[index] or self.tracker.lost[index] or int(lengths[index]) >= max_length
            )
            if finished:
                self.tracker.mark_held(index)
                self.planner.hold(index)
        if bool(self.tracker.holding.all()):
            nxt = (self.tracker.phase + 1) % self.tracker.plan.phase_count
            self.tracker.queue_phase(nxt)
            self.pending = True
        return self.pending

    def on_reset(self) -> None:
        """The reset into the next phase has started."""
        self.pending = False
