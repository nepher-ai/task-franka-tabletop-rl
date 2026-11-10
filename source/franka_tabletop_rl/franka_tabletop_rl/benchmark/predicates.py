# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Latched subtask progress. Every stage is judged by the same predicate as the episode success."""

from __future__ import annotations

import numpy as np

from .spec import Check, Goal, TabletopTaskSpec
from .success import Contacts, goal_met, placed_slots

# Stages whose credit is taken back when the object returns to the container or leaves the support.
_REVOCABLE = ("outside", "outside_and_on_surface")
# Stages that score only while they hold. A pickup is over once the object is put down.
_TRANSIENT = ("picked_up",)


def latch_key(goal: Goal) -> str:
    """Latch id of one stage, so two checks on the same object stay distinct."""
    return f"{goal.op}:{goal.slot}:{goal.target}"


def task_progress(
    spec: TabletopTaskSpec,
    poses: dict[str, dict],
    contacts: Contacts,
    latched: dict[str, bool],
    entered: list[str],
    side: str | None = None,
) -> tuple[float, dict[str, bool], list[str]]:
    """Latched subtask progress in ``[0, 1]``, the updated latch, and the ``in_order`` entry order.

    A stage earns credit once its success predicate holds, release and contact included, and keeps
    it. Removal stages lose it again when they stop holding. ``all`` averages, ``any`` takes the max,
    and ``choose`` averages the top ``k``, so more than ``k`` placed objects still score fully.

    Args:
        spec: Task success expression.
        poses: Slot to pose and geometry, as for :func:`~.success.task_satisfied`.
        contacts: Finger and pair contact of the env.
        latched: Latch of the previous step.
        entered: Slots of ``in_order`` in the order they were first placed.
        side: Overrides the default side of ``on_side``.
    """
    if spec.check is None:
        return 0.0, dict(latched), list(entered)
    kept = dict(latched)
    sequence = list(entered)
    score = _score(spec.check, poses, contacts, kept, sequence, side or spec.side)
    return score, kept, sequence


def _score(
    check: Check,
    poses: dict,
    contacts: Contacts,
    latched: dict[str, bool],
    entered: list[str],
    side: str,
    placed: frozenset[str] = frozenset(),
) -> float:
    placed = placed_slots(check, poses, contacts, side, placed)
    if check.order:
        return _order_score(check, poses, contacts, latched, entered, side, placed)
    if check.children:
        values = [_score(child, poses, contacts, latched, entered, side, placed) for child in check.children]
    else:
        values = [_credit(goal, poses, contacts, latched, side, placed) for goal in check.goals]
    return _reduce(check.logical, values, check.k)


def _order_score(
    check: Check,
    poses: dict,
    contacts: Contacts,
    latched: dict[str, bool],
    entered: list[str],
    side: str,
    placed: frozenset[str],
) -> float:
    for goal in check.goals:
        if goal.slot not in entered and goal_met(goal, poses, contacts, side, placed):
            entered.append(goal.slot)
    matched: list[str] = []
    for slot in entered:
        if len(matched) < len(check.order) and slot == check.order[len(matched)]:
            matched.append(slot)
    values = []
    for slot in check.order:
        if slot not in matched:
            values.append(0.0)
            continue
        goal = next(item for item in check.goals if item.slot == slot)
        values.append(_credit(goal, poses, contacts, latched, side, placed))
    return _reduce("all", values, 1)


def _credit(
    goal: Goal, poses: dict, contacts: Contacts, latched: dict[str, bool], side: str, placed: frozenset[str]
) -> float:
    met = goal_met(goal, poses, contacts, side, placed)
    if goal.op in _TRANSIENT:
        return 1.0 if met else 0.0
    key = latch_key(goal)
    if met:
        latched[key] = True
        return 1.0
    if goal.op in _REVOCABLE:
        latched.pop(key, None)
        return 0.0
    return 1.0 if latched.get(key) else 0.0


def _reduce(logical: str, values: list[float], k: int) -> float:
    if not values:
        return 0.0
    if logical == "any":
        return float(max(values))
    if logical == "choose":
        ordered = sorted(values, reverse=True)
        return float(np.mean(ordered[: max(k, 1)]))
    return float(np.mean(values))
