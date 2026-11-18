# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick episodes from a scout log.

A scout run logs one row per finished episode:

* ``env``, ``episode``: the env index and that env's episode number, counted from 0 at the first reset.
* ``task_id`` and ``outcome``: ``success``, ``lost`` (an object left the table), or ``timeout``.
* ``outcome_step``: control steps from the episode start to the outcome, or ``None`` for a timeout.
* ``length``: control steps of the whole episode, including the post-roll after the outcome.
* ``events``: ``[step, stage, slot]`` for every planner stage change.

Times here are seconds [s] after the episode start.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .shots import CONTROL_HZ, Cut, Shot, StageSpec

Row = dict


@dataclass(frozen=True)
class Window:
    """A span [s] after the episode start. The start may be negative, which uses the pre-roll."""

    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def seconds(step: int | float) -> float:
    """Seconds [s] of ``step`` control steps."""
    return float(step) / CONTROL_HZ


def event_step(row: Row, stage: str, after: int = -1) -> int | None:
    """Control step of the first ``stage`` event that is later than step ``after``."""
    for step, name, _slot in row.get("events", ()):
        if name == stage and step > after:
            return int(step)
    return None


def window_of(row: Row, cut: Cut) -> Window | None:
    """Suggested cut window of ``row``, or ``None`` when its anchor event never happened."""
    outcome_step = row.get("outcome_step")
    if cut.anchor == "start":
        base = 0.0
    elif cut.anchor == "first_close":
        step = event_step(row, "CLOSE")
        if step is None:
            return None
        base = seconds(step)
    else:
        if outcome_step is None:
            return None
        base = seconds(outcome_step)
    if cut.anchor == "episode":
        return Window(0.0, base + cut.after_s)
    return Window(base - cut.before_s, base + cut.after_s)


@dataclass(frozen=True)
class TakePlan:
    """One episode to film, and when to stop.

    ``window`` is the span the editor will cut, and ``stop_s`` is the episode time [s] at which the take
    ends, which is the window end plus the post-roll. ``None`` means the episode's own end. The expected
    fields are the scout's record of the episode and are checked against the shoot.
    """

    env: int
    episode: int
    task_id: str
    window: Window
    stop_s: float | None
    outcome: str | None = None
    outcome_step: int | None = None
    length: int | None = None
    events: tuple[tuple[int, str, str], ...] = ()

    def as_dict(self) -> dict:
        return {
            "env": self.env,
            "episode": self.episode,
            "task_id": self.task_id,
            "window_s": [self.window.start_s, self.window.end_s],
            "stop_s": self.stop_s,
            "outcome": self.outcome,
            "outcome_step": self.outcome_step,
            "length": self.length,
        }


def plan_from_row(row: Row, shot: Shot) -> TakePlan:
    """Plan the take of ``shot`` on a scout ``row``.

    Raises:
        ValueError: The row has no window for the shot's cut.
    """
    window = window_of(row, shot.cut)
    if window is None:
        raise ValueError(f"episode {row['env']}/{row['episode']} has no '{shot.cut.anchor}' anchor for {shot.name}")
    stop = None if shot.cut.anchor == "episode" else window.end_s + shot.post_roll_s
    return TakePlan(
        env=int(row["env"]),
        episode=int(row["episode"]),
        task_id=row["task_id"],
        window=window,
        stop_s=stop,
        outcome=row["outcome"],
        outcome_step=row.get("outcome_step"),
        length=int(row["length"]),
        events=tuple((int(step), str(stage), str(slot)) for step, stage, slot in row.get("events", ())),
    )


def plan_first_episode(shot: Shot, task_id: str = "") -> TakePlan:
    """Plan for a shot that films the first episode from its start, such as the parallel grid.

    Raises:
        ValueError: The cut does not anchor at the episode start.
    """
    if shot.cut.anchor != "start":
        raise ValueError(f"{shot.name} needs a scout row because its cut anchors at '{shot.cut.anchor}'")
    window = Window(-shot.cut.before_s, shot.cut.after_s)
    return TakePlan(env=0, episode=0, task_id=task_id, window=window, stop_s=window.end_s + shot.post_roll_s)


def qualifies(row: Row, shot: Shot) -> bool:
    """Whether ``row`` can supply ``shot``: task, outcome, duration, and a window that holds what it needs."""
    if shot.focus and row["task_id"] not in shot.focus:
        return False
    if shot.outcome != "any" and row["outcome"] != shot.outcome:
        return False
    outcome_step = row.get("outcome_step")
    low, high = shot.duration_s
    if low is not None or high is not None:
        if outcome_step is None:
            return False
        duration = seconds(outcome_step)
        if (low is not None and duration < low) or (high is not None and duration > high):
            return False
    window = window_of(row, shot.cut)
    if window is None:
        return False
    if window.start_s < -shot.pre_roll_s or window.end_s > seconds(row["length"]):
        return False
    for stage in shot.cut.require:
        step = event_step(row, stage)
        if step is None or not window.start_s <= seconds(step) <= window.end_s:
            return False
    return True


def _preference(row: Row, shot: Shot) -> float:
    """Smaller is better. Aim for the middle of the allowed duration, else the faster episode."""
    outcome_step = row.get("outcome_step")
    steps = row["length"] if outcome_step is None else outcome_step
    low, high = shot.duration_s
    if low is not None and high is not None:
        return abs(seconds(steps) - 0.5 * (low + high))
    return float(steps)


def ranked(rows: Iterable[Row], shot: Shot) -> list[Row]:
    """Rows that qualify for ``shot``, best first."""
    kept = [row for row in rows if qualifies(row, shot)]
    return sorted(kept, key=lambda row: (_preference(row, shot), row["env"], row["episode"]))


def pick(rows: Iterable[Row], shot: Shot, rank: int = 0) -> Row | None:
    """The ``rank``-th best qualifying row, or ``None``."""
    best = ranked(rows, shot)
    return best[rank] if rank < len(best) else None


def pick_pair(rows: Sequence[Row], failure: Shot, success: Shot) -> tuple[Row, Row] | None:
    """A failure row and a success row of one task, for the same-camera pair.

    Chooses the task with the most balanced supply of both, then the best row of each.
    """
    lost = ranked(rows, failure)
    clean = ranked(rows, success)
    tasks = {row["task_id"] for row in lost} & {row["task_id"] for row in clean}
    if not tasks:
        return None

    def supply(task: str) -> tuple[int, int]:
        a = sum(row["task_id"] == task for row in lost)
        b = sum(row["task_id"] == task for row in clean)
        return min(a, b), a + b

    task = max(sorted(tasks), key=supply)
    return (
        next(row for row in lost if row["task_id"] == task),
        next(row for row in clean if row["task_id"] == task),
    )


def summarize(rows: Iterable[Row]) -> dict[str, Counter]:
    """Outcome counts per task."""
    counts: dict[str, Counter] = {}
    for row in rows:
        counts.setdefault(row["task_id"], Counter())[row["outcome"]] += 1
    return dict(sorted(counts.items()))


def format_summary(rows: Iterable[Row]) -> str:
    """A table of ``success``, ``lost``, and ``timeout`` counts per task."""
    lines = [f"{'task':<36}{'success':>8}{'lost':>6}{'timeout':>9}"]
    for task, counts in summarize(rows).items():
        lines.append(f"{task:<36}{counts['success']:>8}{counts['lost']:>6}{counts['timeout']:>9}")
    return "\n".join(lines)


def scout_path(root: Path, stage: StageSpec) -> Path:
    """Where the scout log of ``stage`` lives under ``root``."""
    return root / "scout" / f"{stage.tag()}.json"


def save_scout(path: Path, stage: StageSpec, task_ids: Sequence[str], rows: Sequence[Row], extra: dict) -> None:
    """Write a scout log. ``extra`` carries run facts such as the checkpoint name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage.as_dict(),
        "task_ids": list(task_ids),
        "control_hz": CONTROL_HZ,
        **extra,
        "rows": list(rows),
    }
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def load_scout(path: Path) -> dict:
    """Read a scout log.

    Raises:
        FileNotFoundError: There is no scout log. Run ``scout`` first.
    """
    if not path.is_file():
        raise FileNotFoundError(f"no scout log at {path}; run the scout mode for this stage first")
    return json.loads(path.read_text(encoding="utf-8"))
