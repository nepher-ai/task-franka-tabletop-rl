# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode logging and frame timing for film runs. Pure Python, no simulator."""

from __future__ import annotations

import math
from collections.abc import Sequence

Event = tuple[int, str, str]


class EpisodeLog:
    """Per-env log of planner stage changes, and the finished episodes.

    A step number is the control step about to run, counted from 0 at the episode start.
    """

    def __init__(self, num_envs: int):
        self.rows: list[dict] = []
        self._events: list[list[Event]] = [[] for _ in range(num_envs)]
        self._last: list[tuple[str, str] | None] = [None] * num_envs
        self._episode = [-1] * num_envs
        self._open = [False] * num_envs

    def episode(self, env: int) -> int:
        """Number of the env's running episode, or -1 before the first one."""
        return self._episode[env]

    def is_open(self, env: int) -> bool:
        return self._open[env]

    def start(self, env: int) -> int:
        """Begin the next episode of ``env`` and return its number."""
        self._episode[env] += 1
        self._events[env] = []
        self._last[env] = None
        self._open[env] = True
        return self._episode[env]

    def observe(self, env: int, step: int, stage: str, slot: str) -> None:
        """Note the stage and the slot the planner works on. Only changes are logged."""
        if not self._open[env] or self._last[env] == (stage, slot):
            return
        self._last[env] = (stage, slot)
        self._events[env].append((int(step), stage, slot))

    def events(self, env: int) -> list[Event]:
        return list(self._events[env])

    def finish(self, env: int, task_id: str, outcome: str, outcome_step: int | None, length: int) -> dict | None:
        """Close the running episode and return its row, or ``None`` when none was open."""
        if not self._open[env]:
            return None
        self._open[env] = False
        row = {
            "env": env,
            "episode": self._episode[env],
            "task_id": task_id,
            "outcome": outcome,
            "outcome_step": None if outcome_step is None else int(outcome_step),
            "length": int(length),
            "events": [list(event) for event in self._events[env]],
        }
        self.rows.append(row)
        return row


class FrameClock:
    """Frame schedule of a take.

    Take time starts at the first frame of the pre-roll. The episode starts ``pre_roll_s`` later. The
    pre-roll frames show the reset scene; live frames come every ``physics_hz / fps`` physics steps.

    Args:
        fps: Video frame rate [Hz].
        physics_hz: Physics rate [Hz]. It must be a whole multiple of ``fps``.
        pre_roll_s: Footage [s] before the episode starts.

    Raises:
        ValueError: ``physics_hz`` is not a whole multiple of ``fps``.
    """

    def __init__(self, fps: int, physics_hz: float, pre_roll_s: float):
        ratio = physics_hz / fps
        if ratio < 1.0 or abs(ratio - round(ratio)) > 1e-6:
            raise ValueError(f"fps {fps} must divide the physics rate {physics_hz:g} Hz into whole steps")
        self.fps = int(fps)
        self.physics_hz = float(physics_hz)
        self.ratio = int(round(ratio))
        self.pre_roll_s = float(pre_roll_s)

    @property
    def pre_roll_frames(self) -> int:
        """Frames up to and including the one at the episode start."""
        return int(round(self.pre_roll_s * self.fps)) + 1

    def pre_roll_time(self, index: int) -> float:
        """Take time [s] of pre-roll frame ``index``."""
        return index / self.fps

    def is_frame(self, substeps: int) -> bool:
        """Whether ``substeps`` physics steps after the episode start end on a frame."""
        return substeps > 0 and substeps % self.ratio == 0

    def live_time(self, substeps: int) -> float:
        """Take time [s] of the frame ``substeps`` physics steps into the episode."""
        return self.pre_roll_s + substeps / self.physics_hz

    def stop_substeps(self, stop_s: float | None) -> int | None:
        """First substep count, on a frame, at or after episode time ``stop_s`` [s]."""
        if stop_s is None:
            return None
        raw = max(math.ceil(stop_s * self.physics_hz - 1e-9), 1)
        return int(math.ceil(raw / self.ratio)) * self.ratio


def compare_events(expected: Sequence[Event], observed: Sequence[Event], upto_step: int | None = None) -> str | None:
    """Why a shoot's planner timeline differs from the scout's, or ``None`` when they agree.

    Args:
        expected: The scout's ``(step, stage, slot)`` events.
        observed: The shoot's events.
        upto_step: Compare only events before this control step, for takes that stop before the episode ends.
    """

    def clip(events: Sequence[Event]) -> list[tuple[int, str, str]]:
        return [tuple(event) for event in events if upto_step is None or event[0] < upto_step]

    want, got = clip(expected), clip(observed)
    for index, (a, b) in enumerate(zip(want, got)):
        if a != b:
            return f"event {index}: scout {a}, shoot {b}"
    if len(want) != len(got):
        longer = "scout" if len(want) > len(got) else "shoot"
        return f"the {longer} has {abs(len(want) - len(got))} more events"
    return None
