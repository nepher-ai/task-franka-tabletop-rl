# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Instantaneous episode success, following RoboLab's termination terms. Positions are [m].

RoboLab ends an episode when a stateless predicate holds on the current state. A slot in ``grasp``
has to be free of the fingers now, instead of having been grasped and released. The only history is
the first-placement step that ``in_order`` compares. Contact-force checks need object-pair sensors,
so ``stacked`` and ``on_top`` use pose checks only. ``task_progress`` keeps the latched score that
the planner reads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .predicates import _lowest_z, _oriented_corners, _world_to_body, lifted, on_footprint, on_top_of
from .spec import Check, Goal, TabletopTaskSpec

# A local up axis within this angle [rad] of world up counts as upright.
UPRIGHT_TOL_RAD = 0.1
# Half angle [rad] of the cone around a side axis. A cube beside the reference needs to stay within it.
SIDE_CONE_RAD = np.pi / 4.0
# Footprint slack [m] of a stacked block on its support.
STACK_XY_TOL_M = 0.01
# The lowest corner of a stacked block may differ from the support top by this much [m]. It stands in
# for RoboLab's block-to-block contact check.
STACK_GAP_M = 0.01
# Height difference [m] below which an object is level with its reference when it is not grasped.
LEVEL_TOL_M = 0.1


@dataclass(frozen=True)
class _State:
    poses: dict
    touching: dict[str, bool]
    side: str


def task_satisfied(
    spec: TabletopTaskSpec,
    poses: dict[str, dict[str, np.ndarray]],
    touching: dict[str, bool],
    first_placed: dict[str, int],
    side: str | None = None,
) -> bool:
    """Whether the success expression holds on the current state.

    Args:
        spec: Task success expression.
        poses: Slot to ``pos`` [m] ``[1, 3]``, ``quat`` (wxyz) ``[1, 4]``, and ``half`` [m] ``[3]``.
        touching: Slot to whether a finger is in contact now.
        first_placed: Slot to the step it was first placed. Read by ``in_order``.
        side: Overrides the default side of ``on_side``.
    """
    if spec.check is None:
        return False
    state = _State(poses, touching, side or spec.side)
    return _check(spec.check, state, first_placed, spec.check.grasp, spec.check.hold)


def note_placements(
    spec: TabletopTaskSpec,
    poses: dict[str, dict[str, np.ndarray]],
    touching: dict[str, bool],
    first_placed: dict[str, int],
    step: int,
    side: str | None = None,
) -> None:
    """Record the first step at which each ``in_order`` object was placed. Entries are never removed."""
    if spec.check is None:
        return
    state = _State(poses, touching, side or spec.side)
    _note(spec.check, state, first_placed, step, spec.check.grasp, spec.check.hold)


def _note(check: Check, state: _State, first_placed: dict[str, int], step: int, grasp, hold) -> None:
    grasp = check.grasp or grasp
    hold = check.hold or hold
    if check.order:
        for goal in check.goals:
            if goal.slot not in first_placed and _goal(goal, state, grasp, hold):
                first_placed[goal.slot] = step
    for child in check.children:
        _note(child, state, first_placed, step, grasp, hold)


def _check(check: Check, state: _State, first_placed: dict[str, int], grasp, hold) -> bool:
    grasp = check.grasp or grasp
    hold = check.hold or hold
    if check.order:
        return _in_order(check, state, first_placed, grasp, hold)
    if check.children:
        met = [_check(child, state, first_placed, grasp, hold) for child in check.children]
    else:
        met = [_goal(goal, state, grasp, hold) for goal in check.goals]
    return _combine(check.logical, met, check.k)


def _combine(logical: str, met: list[bool], k: int) -> bool:
    if not met:
        return False
    if logical == "any":
        return any(met)
    if logical == "choose":
        return sum(met) == max(k, 1)
    return all(met)


def _in_order(check: Check, state: _State, first_placed: dict[str, int], grasp, hold) -> bool:
    by_slot = {goal.slot: goal for goal in check.goals}
    if not all(_goal(by_slot[slot], state, grasp, hold) for slot in check.order):
        return False
    steps = [first_placed.get(slot, -1) for slot in check.order]
    return all(step >= 0 for step in steps) and all(a <= b for a, b in zip(steps, steps[1:]))


def _goal(goal: Goal, state: _State, grasp, hold) -> bool:
    met = _geometry(goal, state, goal.slot not in grasp and goal.slot not in hold)
    if goal.slot in hold:
        return met and state.touching.get(goal.slot, False)
    if goal.slot in grasp and goal.op != "stacked":
        return met and not state.touching.get(goal.slot, False)
    return met


def _geometry(goal: Goal, state: _State, free: bool) -> bool:
    item = state.poses[goal.slot]
    target = state.poses.get(goal.target)
    if goal.op == "in_container":
        return _in_open_top(item, target)
    if goal.op == "outside":
        return not _in_open_top(item, target)
    if goal.op == "on_side":
        return _on_side(item, target, goal.side or state.side, free)
    if goal.op == "on_top":
        return bool(on_footprint(item["pos"], target["pos"], target["half"] + STACK_XY_TOL_M).reshape(-1)[0])
    if goal.op == "stacked":
        return _stacked(item, target)
    if goal.op == "off":
        # A tilted cube that is still sitting on the support is not unstacked.
        return not bool(np.asarray(on_top_of(item["pos"], target["pos"], 0.04, 0.12)).reshape(-1)[0])
    if goal.op == "upright":
        return _upright(item["quat"])
    if goal.op == "lifted":
        return bool(lifted(item["pos"], 0.0).reshape(-1)[0])
    if goal.op == "picked":
        return _lowest_z(item) >= goal.margin
    if goal.op == "resting":
        height = float(np.asarray(item["pos"])[..., 2].reshape(-1)[0])
        return goal.margin - 0.05 < height < goal.margin + 0.06
    if goal.op == "centered":
        delta = np.abs(np.asarray(item["pos"]) - np.asarray(target["pos"]))[..., :2].reshape(-1)
        limit = goal.margin if goal.margin > 0.0 else 0.05
        return bool(np.all(delta <= limit))
    raise KeyError(f"unknown success predicate {goal.op!r}")


def _in_open_top(item: dict, container: dict) -> bool:
    """Whether the item center is inside the container box, which is open along its local +z."""
    delta = np.asarray(item["pos"], dtype=np.float64) - np.asarray(container["pos"], dtype=np.float64)
    local = _world_to_body(container["quat"], delta).reshape(-1)
    half = np.asarray(container["half"], dtype=np.float64).reshape(-1)[:3]
    return bool(abs(local[0]) <= half[0] and abs(local[1]) <= half[1] and local[2] >= -half[2])


def _on_side(item: dict, reference: dict, side: str, free: bool) -> bool:
    """Whether the item lies in a 45 degree cone toward ``side`` of the reference, in the robot frame.

    The robot sits at the env origin facing +x, so ``front`` is toward the robot (-x).
    """
    delta = np.asarray(item["pos"], dtype=np.float64) - np.asarray(reference["pos"], dtype=np.float64)
    delta = delta.reshape(-1)
    planar = float(np.linalg.norm(delta[:2]))
    if planar <= 1e-6:
        return False
    index, sign = {"left": (1, 1.0), "right": (1, -1.0), "front": (0, -1.0), "behind": (0, 1.0)}[side]
    if sign * delta[index] <= 0.0 or sign * delta[index] / planar < np.cos(SIDE_CONE_RAD):
        return False
    # As in RoboLab, the level check applies only when no grasp requirement does.
    return not free or abs(float(delta[2])) <= LEVEL_TOL_M


def _stacked(item: dict, support: dict) -> bool:
    """Whether the item sits on the support: centered over its footprint, lowest corner at its top."""
    below = _oriented_corners(support)
    low, high = below.min(axis=0), below.max(axis=0)
    center = np.asarray(item["pos"], dtype=np.float64).reshape(-1)
    over = bool(np.all(center[:2] >= low[:2] - STACK_XY_TOL_M) and np.all(center[:2] <= high[:2] + STACK_XY_TOL_M))
    return over and abs(_lowest_z(item) - float(high[2])) <= STACK_GAP_M


def _upright(quat_wxyz: np.ndarray) -> bool:
    """Whether the local z axis is within ``UPRIGHT_TOL_RAD`` of world up."""
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1)[:4]
    return bool(1.0 - 2.0 * (x * x + y * y) >= np.cos(UPRIGHT_TOL_RAD))
