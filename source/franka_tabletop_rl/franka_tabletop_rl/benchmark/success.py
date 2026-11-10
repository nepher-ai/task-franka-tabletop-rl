# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Instantaneous episode success, judged the way the reference benchmark judges it. Lengths are [m], forces [N].

An episode succeeds when a stateless predicate holds on the current state. Containment tests the
hull center of the object against the open-top hull of the container. Support and stacking read the
contact forces between the two bodies. The only history is the first-placement step that
``in_order`` compares. ``task_progress`` keeps the latched score that the planner reads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import (
    SURFACE_SLOT,
    hull_center_in,
    rotation,
    world_box_center,
    world_box_corners,
)
from .spec import Check, Goal, TabletopTaskSpec, goals_of

# A force component above this [N] counts as contact.
CONTACT_FORCE_N = 0.1
# Half angle [rad] of the cone around vertical in which a support force has to point.
SUPPORT_CONE_RAD = math.pi / 4.0
# Half angle [rad] of the cone around a side axis. A cube beside the reference needs to stay within it.
SIDE_CONE_RAD = math.pi / 4.0
# Height difference [m] below which an object is level with its reference. Directions use it unless
# the stage asks for contact or release.
LEVEL_TOL_M = 0.1
# Footprint slack [m] of ``on_top``, ``stacked``, and ``picked_up``.
FOOTPRINT_TOL_M = 0.01
# ``upright`` angle [rad] and ``centered`` distance [m] when the stage names none.
UPRIGHT_TOL_RAD = 0.1
CENTERED_TOL_M = 0.01


@dataclass(frozen=True)
class Contacts:
    """Contact of one env.

    Attributes:
        touching: Slot to whether a finger touches it now.
        touch: ``(a, b)`` to whether any force component on ``a`` from ``b`` exceeds
            :data:`CONTACT_FORCE_N`. A pair is read in either order.
        force: ``(a, b)`` to the net force [N] on ``a`` from ``b``, shape ``[3]``.
    """

    touching: dict[str, bool] = field(default_factory=dict)
    touch: dict[tuple[str, str], bool] = field(default_factory=dict)
    force: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)

    def pair_touch(self, first: str, second: str) -> bool:
        """Whether the two bodies touch. A pair without a sensor does not touch."""
        return bool(self.touch.get((first, second), self.touch.get((second, first), False)))

    def supported(self, item: str, support: str) -> bool:
        """Whether ``support`` pushes ``item`` up within :data:`SUPPORT_CONE_RAD` of vertical."""
        force = self.force.get((item, support))
        if force is None:
            reverse = self.force.get((support, item))
            if reverse is None:
                return False
            force = -np.asarray(reverse, dtype=np.float64)
        force = np.asarray(force, dtype=np.float64).reshape(3)
        magnitude = float(np.linalg.norm(force))
        if magnitude < CONTACT_FORCE_N or force[2] <= 0.0:
            return False
        return bool(force[2] >= magnitude * math.cos(SUPPORT_CONE_RAD))


@dataclass(frozen=True)
class _State:
    poses: dict
    contacts: Contacts
    side: str


def task_satisfied(
    spec: TabletopTaskSpec,
    poses: dict[str, dict],
    contacts: Contacts,
    first_placed: dict[str, int],
    side: str | None = None,
) -> bool:
    """Whether the success expression holds on the current state.

    Args:
        spec: Task success expression.
        poses: Slot to ``pos`` [m] ``[1, 3]``, ``quat`` (wxyz) ``[1, 4]``, ``geometry``
            (:class:`~.geometry.AssetGeometry`), and ``anchor`` (true for a slot without a body).
            The work surface sits under :data:`SURFACE_SLOT`.
        contacts: Finger and pair contact of the env.
        first_placed: Slot to the step it was first placed. Read by ``in_order``.
        side: Overrides the default side of ``on_side``.
    """
    if spec.check is None:
        return False
    return _check(spec.check, _State(poses, contacts, side or spec.side), first_placed)


def note_placements(
    spec: TabletopTaskSpec,
    poses: dict[str, dict],
    contacts: Contacts,
    first_placed: dict[str, int],
    step: int,
    side: str | None = None,
) -> None:
    """Record the first step at which each ``in_order`` object was placed. Entries are never removed."""
    if spec.check is not None:
        _note(spec.check, _State(poses, contacts, side or spec.side), first_placed, step)


def required_pairs(
    spec: TabletopTaskSpec, anchors: set[str] | frozenset[str] = frozenset()
) -> tuple[tuple[str, str], ...]:
    """Body pairs whose contact the expression reads, as ``(item, partner)``.

    A slot in ``anchors`` has no body, so its contact is read against the work surface.
    """
    if spec.check is None:
        return ()
    pairs: list[tuple[str, str]] = []
    for goal in goals_of(spec.check):
        partners = list(goal.contact)
        if goal.op in ("on_top", "stacked"):
            partners.append(goal.target)
        elif goal.op == "outside_and_on_surface":
            partners.append(goal.surface or SURFACE_SLOT)
        for partner in partners:
            pair = (goal.slot, SURFACE_SLOT if partner in anchors else partner)
            if pair not in pairs:
                pairs.append(pair)
    return tuple(pairs)


def goal_met(goal: Goal, poses: dict[str, dict], contacts: Contacts, side: str) -> bool:
    """Whether one stage holds now: its geometry, the contact it asks for, and the release it asks for."""
    return _goal(goal, _State(poses, contacts, side))


def _note(check: Check, state: _State, first_placed: dict[str, int], step: int) -> None:
    if check.order:
        for goal in check.goals:
            if goal.slot not in first_placed and _goal(goal, state):
                first_placed[goal.slot] = step
    for child in check.children:
        _note(child, state, first_placed, step)


def _check(check: Check, state: _State, first_placed: dict[str, int]) -> bool:
    if check.order:
        return _in_order(check, state, first_placed)
    if check.children:
        met = [_check(child, state, first_placed) for child in check.children]
    else:
        met = [_goal(goal, state) for goal in check.goals]
    return _combine(check.logical, met, check.k)


def _combine(logical: str, met: list[bool], k: int) -> bool:
    if not met:
        return False
    if logical == "any":
        return any(met)
    if logical == "choose":
        return sum(met) == max(k, 1)
    return all(met)


def _in_order(check: Check, state: _State, first_placed: dict[str, int]) -> bool:
    by_slot = {goal.slot: goal for goal in check.goals}
    if not all(_goal(by_slot[slot], state) for slot in check.order):
        return False
    steps = [first_placed.get(slot, -1) for slot in check.order]
    return all(step >= 0 for step in steps) and all(a <= b for a, b in zip(steps, steps[1:]))


def _goal(goal: Goal, state: _State) -> bool:
    if not _geometry(goal, state):
        return False
    if any(not state.contacts.pair_touch(goal.slot, _body(state, partner)) for partner in goal.contact):
        return False
    # A slot without a finger reading counts as touched, so a missing sensor cannot pass.
    return not (goal.detach and state.contacts.touching.get(goal.slot, True))


def _body(state: _State, slot: str) -> str:
    """The body that carries ``slot``. An anchor has none, so the work surface stands in."""
    pose = state.poses.get(slot)
    return SURFACE_SLOT if pose is not None and pose.get("anchor") else slot


def _geometry(goal: Goal, state: _State) -> bool:
    item = state.poses[goal.slot]
    target = state.poses.get(goal.target)
    op = goal.op
    if op == "in_container":
        return _contains(item, target, "planes_open_top")
    if op == "enclosed":
        return _contains(item, target, "planes_full")
    if op == "outside":
        return not _contains(item, target, "planes_open_top")
    if op == "outside_and_on_surface":
        surface = goal.surface or SURFACE_SLOT
        return not _contains(item, target, "planes_open_top") and state.contacts.supported(
            goal.slot, _body(state, surface)
        )
    if op == "on_top":
        tolerance = FOOTPRINT_TOL_M if goal.tolerance is None else goal.tolerance
        return state.contacts.supported(goal.slot, _body(state, goal.target)) and _in_footprint(item, target, tolerance)
    if op == "stacked":
        return state.contacts.pair_touch(goal.slot, _body(state, goal.target)) and (
            _above(item, target, True, FOOTPRINT_TOL_M, 0.0) or _above(item, target, False, 0.0, 0.0)
        )
    if op == "on_side":
        return _on_side(item, target, goal.side or state.side, require_level=not goal.contact and not goal.detach)
    if op == "upright":
        return _upright(item["quat"], goal.up_axis, UPRIGHT_TOL_RAD if goal.tolerance is None else goal.tolerance)
    if op == "picked_up":
        surface = state.poses[goal.surface or SURFACE_SLOT]
        grabbed = state.contacts.touching.get(goal.slot, False)
        return grabbed and _above(item, surface, True, FOOTPRINT_TOL_M, goal.distance)
    if op == "centered":
        tolerance = CENTERED_TOL_M if goal.tolerance is None else goal.tolerance
        delta = np.abs(_center(item) - _center(target))[:2]
        return bool(np.all(delta <= tolerance))
    raise KeyError(f"unknown success predicate {op!r}")


def _vec(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)[:3]


def _center(pose: dict) -> np.ndarray:
    return world_box_center(pose["geometry"], _vec(pose["pos"]), pose["quat"])


def _corners(pose: dict) -> np.ndarray:
    return world_box_corners(pose["geometry"], _vec(pose["pos"]), pose["quat"])


def _contains(item: dict, container: dict, planes: str) -> bool:
    """Whether the hull center of ``item`` lies inside the chosen hull planes of ``container``."""
    return hull_center_in(
        item["geometry"],
        _vec(item["pos"]),
        item["quat"],
        _vec(container["pos"]),
        container["quat"],
        getattr(container["geometry"], planes),
    )


def _in_footprint(item: dict, support: dict, tolerance: float) -> bool:
    """Whether the box center of ``item`` lies over the world-aligned box of ``support``. z is free."""
    corners = _corners(support)
    low, high = corners.min(axis=0), corners.max(axis=0)
    center = _center(item)
    return bool(np.all(center[:2] >= low[:2] - tolerance) and np.all(center[:2] <= high[:2] + tolerance))


def _above(item: dict, reference: dict, top: bool, tolerance: float, z_margin: float) -> bool:
    """Whether every box corner of ``item`` is above the top (or bottom) of ``reference`` by ``z_margin``."""
    corners = _corners(reference)
    low, high = corners.min(axis=0), corners.max(axis=0)
    center = _center(item)
    over = bool(np.all(center[:2] >= low[:2] - tolerance) and np.all(center[:2] <= high[:2] + tolerance))
    height = high[2] if top else low[2]
    return over and bool(np.all(_corners(item)[:, 2] >= height + z_margin))


def _on_side(item: dict, reference: dict, side: str, require_level: bool) -> bool:
    """Whether the item lies in a cone toward ``side`` of the reference, in the robot frame.

    The robot sits at the env origin facing +x, so ``front`` is toward the robot (-x). The
    reference benchmark checks the level only when the stage asks for neither contact nor release.
    """
    delta = _vec(item["pos"]) - _vec(reference["pos"])
    planar = float(np.hypot(delta[0], delta[1]))
    if planar <= 1e-6:
        return False
    index, sign = {"left": (1, 1.0), "right": (1, -1.0), "front": (0, -1.0), "behind": (0, 1.0)}[side]
    if sign * delta[index] <= 0.0 or sign * delta[index] / planar < math.cos(SIDE_CONE_RAD):
        return False
    return not require_level or abs(float(_center(item)[2] - _center(reference)[2])) <= LEVEL_TOL_M


def _upright(quat_wxyz, up_axis: str, tolerance: float) -> bool:
    """Whether the root axis ``up_axis`` is within ``tolerance`` [rad] of world up."""
    axis = {"x": 0, "y": 1, "z": 2}[up_axis]
    return bool(rotation(quat_wxyz)[2, axis] >= math.cos(tolerance))
