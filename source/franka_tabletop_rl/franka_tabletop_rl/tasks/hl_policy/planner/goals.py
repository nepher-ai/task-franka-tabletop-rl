# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Turn a success expression into the ordered goals the planner will attempt.

``all`` keeps every goal. ``in_order`` keeps that order. ``any`` and ``choose`` keep the goals
whose objects are nearest a home position, decided when the episode layout is known. A task that
asks for ``upright`` is not attempted: the planner has no regrasp.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from franka_tabletop_rl.benchmark.spec import Check, Goal

_HOLD = {"picked", "lifted"}
_PLACE = {"in_container", "on_top", "stacked", "on_side", "outside", "off", "centered", "resting"}


@dataclass(frozen=True)
class PlaceGoal:
    """Pick ``slot`` and release it at a pose computed for ``op``."""

    slot: str
    op: str
    target: str
    side: str = "left"
    margin: float = 0.0


@dataclass(frozen=True)
class HoldGoal:
    """Pick ``slot`` and hold it up. There is no release."""

    slot: str
    op: str
    height_m: float


@dataclass(frozen=True)
class GoalQueue:
    """Goals one env will run, in the order it should prefer them.

    ``ordered`` is true for ``in_order``: the planner must not skip ahead to a nearer object.
    ``supported`` is false when the task needs a skill this planner does not have.
    """

    goals: tuple[PlaceGoal | HoldGoal, ...]
    ordered: bool
    supported: bool


def compile_goals(check: Check | None, positions: dict[str, tuple[float, float]], home_xy=(0.45, 0.0)) -> GoalQueue:
    """Goals for one laid-out task.

    Args:
        check: The task's success expression. ``None`` is unsupported.
        positions: Slot to an xy position [m] on the table, usually the published layout.
        home_xy: Point [m] the nearest-object choice is measured from.

    Returns:
        An empty unsupported queue when the expression is missing or contains ``upright``.
    """
    if check is None or _has_upright(check):
        return GoalQueue((), False, False)
    chosen = _select(check, positions, np.asarray(home_xy, dtype=np.float64))
    return GoalQueue(tuple(chosen), bool(check.order), True)


def _has_upright(check: Check) -> bool:
    if any(goal.op == "upright" for goal in check.goals):
        return True
    return any(_has_upright(child) for child in check.children)


def _select(check: Check, positions: dict[str, tuple[float, float]], home: np.ndarray) -> list[PlaceGoal | HoldGoal]:
    if check.order:
        return [_convert(goal) for goal in check.goals]
    if check.children:
        children = list(check.children)
        if check.logical == "any":
            children = [min(children, key=lambda child: _child_distance(child, positions, home))]
        elif check.logical == "choose":
            ranked = sorted(children, key=lambda child: _child_distance(child, positions, home))
            children = ranked[: max(check.k, 1)]
        goals: list[PlaceGoal | HoldGoal] = []
        for child in children:
            goals.extend(_select(child, positions, home))
        return goals
    goals = [_convert(goal) for goal in check.goals if goal.op in _PLACE or goal.op in _HOLD]
    if check.logical == "all":
        return goals
    ranked = sorted(goals, key=lambda goal: _distance(positions, goal.slot, home))
    keep = 1 if check.logical == "any" else max(check.k, 1)
    return ranked[:keep]


def _convert(goal: Goal) -> PlaceGoal | HoldGoal:
    if goal.op in _HOLD:
        height = goal.margin if goal.op == "picked" and goal.margin > 0.0 else 0.05
        return HoldGoal(goal.slot, goal.op, height)
    if goal.op not in _PLACE:
        raise ValueError(f"planner has no placement for {goal.op}")
    return PlaceGoal(goal.slot, goal.op, goal.target, goal.side, goal.margin)


def _child_distance(check: Check, positions, home) -> float:
    slots = _slots(check)
    if not slots:
        return 0.0
    return min(_distance(positions, slot, home) for slot in slots)


def _slots(check: Check) -> list[str]:
    found = [goal.slot for goal in check.goals]
    for child in check.children:
        found.extend(_slots(child))
    return found


def _distance(positions, slot: str, home: np.ndarray) -> float:
    xy = positions.get(slot)
    if xy is None:
        return 0.0
    return float(np.hypot(xy[0] - home[0], xy[1] - home[1]))
