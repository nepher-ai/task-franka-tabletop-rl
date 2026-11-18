# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hold an episode open after its outcome, so a take has footage after the success or the loss.

The latch calls nothing itself. The env feeds it the raw ``success`` and ``object_lost`` flags every
control step, so the tracker keeps updating exactly as it does without film. It reports done only
``post_roll_steps`` control steps after the first outcome, and it keeps that first outcome even if the
raw flags change later. Arrays can be NumPy or Torch, as long as they all live in one namespace.
"""

from __future__ import annotations

NONE, SUCCESS, LOST = 0, 1, 2


class PostRollLatch:
    """Per-env outcome latch.

    Args:
        outcome: Integer array ``[num_envs]``, zero at start. Receives :data:`SUCCESS` or :data:`LOST`.
        outcome_step: Integer array ``[num_envs]``, minus one at start. Receives the step of the outcome.
        post_roll_steps: Control steps to keep going after the outcome. Zero reports the outcome at once.
    """

    def __init__(self, outcome, outcome_step, post_roll_steps: int):
        if post_roll_steps < 0:
            raise ValueError(f"post_roll_steps must not be negative, got {post_roll_steps}")
        self.outcome = outcome
        self.outcome_step = outcome_step
        self.post_roll_steps = int(post_roll_steps)
        self._step = None

    def update(self, step_id: int, length, success, lost) -> None:
        """Take one control step. ``length`` is the episode length [steps] after this step's increment.

        Does nothing when called again for the same ``step_id``, because both termination terms call it.
        """
        if step_id == self._step:
            return
        self._step = step_id
        fresh = length <= 1
        self.outcome[fresh] = NONE
        self.outcome_step[fresh] = -1
        undecided = self.outcome == NONE
        won = undecided & success
        failed = undecided & lost & ~success
        self.outcome[won] = SUCCESS
        self.outcome[failed] = LOST
        decided_now = won | failed
        self.outcome_step[decided_now] = length[decided_now]

    def done(self, kind: int, length):
        """Mask of envs whose ``kind`` outcome has aged past the post-roll."""
        aged = (self.outcome == kind) & (length - self.outcome_step >= self.post_roll_steps)
        return aged
