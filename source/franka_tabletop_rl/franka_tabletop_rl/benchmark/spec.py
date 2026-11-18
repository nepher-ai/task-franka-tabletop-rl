# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success expression of one benchmark task."""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import SURFACE_SLOT

OPS = (
    "in_container",
    "enclosed",
    "outside",
    "outside_and_on_surface",
    "on_top",
    "stacked",
    "on_side",
    "upright",
    "picked_up",
    "centered",
)

# Ops that older bundles used, and what replaced them.
_RETIRED = {
    "picked": "picked_up",
    "lifted": "picked_up",
    "resting": "outside_and_on_surface",
    "off": "outside_and_on_surface",
}

# Ops that name no target to touch, so ``require_contact_with: true`` has no meaning.
_NO_TARGET = ("upright", "picked_up")


@dataclass(frozen=True)
class Goal:
    """One object's geometric stage.

    Attributes:
        slot: Scene slot this stage scores.
        op: Predicate name, one of :data:`OPS`.
        target: Slot of the container, support, or reference. Empty when the stage has none.
        side: Used by ``on_side``. One of left, right, front, behind. Empty takes the side of the task.
        contact: Slots the object has to touch, from ``require_contact_with``.
        detach: The object must be free of the fingers, from ``require_gripper_detached``.
        tolerance: Slack [m] of the footprint of ``on_top`` and of ``centered``, or the angle [rad]
            of ``upright``. ``None`` takes the benchmark default of the op.
        distance: Height [m] of the lowest box corner of ``picked_up`` above the work surface.
        surface: Slot of the support of ``outside_and_on_surface``. Empty means the work surface.
        up_axis: Root axis that ``upright`` keeps vertical.
    """

    slot: str
    op: str
    target: str = ""
    side: str = ""
    contact: tuple[str, ...] = ()
    detach: bool = False
    tolerance: float | None = None
    distance: float = 0.05
    surface: str = ""
    up_axis: str = "z"


@dataclass(frozen=True)
class Check:
    """Success expression. ``all`` takes every stage, ``any`` one, ``choose`` exactly ``k``.

    Progress averages the stage scores of ``all``, takes the max of ``any``, and averages the top ``k``
    of ``choose``.

    Attributes:
        logical: How to combine ``goals`` or ``children``. One of all, any, choose.
        goals: Object stages scored by this node.
        children: Nested expressions. Used instead of ``goals`` when the block groups checks.
        k: How many objects ``choose`` needs.
        order: Slot order for ``in_order``. Empty for every other check.
    """

    logical: str
    goals: tuple[Goal, ...] = ()
    children: tuple[Check, ...] = ()
    k: int = 1
    order: tuple[str, ...] = ()


@dataclass(frozen=True)
class TabletopTaskSpec:
    """One benchmark success expression.

    Attributes:
        task_id: Authored task id.
        roles: Slot name to itself, kept so instruction helpers can name objects.
        success: Expression op.
        failure: Unused. Episode failure is an object leaving the table.
        time_limit_s: Episode budget [s]. The evaluator may override it.
        side: Default side for ``on_side``. One of left, right, front, behind.
        check: Expression the evaluator scores.
    """

    task_id: str
    roles: dict[str, str]
    success: str
    failure: str
    time_limit_s: float
    side: str = "left"
    check: Check | None = None


def spec_from_success(
    task_id: str,
    success: dict,
    time_limit_s: float,
    name_to_slot: dict[str, str],
) -> TabletopTaskSpec:
    """Build a spec from an ``op`` block.

    Raises:
        ValueError: The block is the old short ``kind`` form, or uses a retired op or ``require_grasp``.
    """
    if "kind" in success and "op" not in success and "checks" not in success:
        raise ValueError(f"task {task_id} uses a short-form success kind; write it as an op")
    side = str(success.get("side", "left"))
    check = _parse(task_id, success, name_to_slot, side)
    return TabletopTaskSpec(
        task_id=task_id,
        roles=_role_map(check),
        success=str(success.get("op", "")),
        failure="",
        time_limit_s=float(time_limit_s),
        side=side,
        check=check,
    )


def _parse(task_id: str, block: dict, name_to_slot: dict[str, str], side: str) -> Check:
    for retired in ("require_grasp", "hold"):
        if retired in block:
            raise ValueError(f"task {task_id}: {retired!r} is retired, use require_gripper_detached per op")
    op = str(block.get("op") or "")
    side = str(block.get("side", side))
    if op in ("all", "any", "choose") and "checks" in block:
        children = tuple(_parse(task_id, child, name_to_slot, side) for child in block["checks"])
        return Check(logical=op, children=children, k=int(block.get("k", 1)))
    if op in ("all", "any"):
        raise ValueError(f"success op {op!r} needs a checks list")
    if op in _RETIRED:
        raise ValueError(f"task {task_id}: op {op!r} is retired, use {_RETIRED[op]!r}")
    if op == "in_order":
        slots = _slots(block.get("object", block.get("objects")), name_to_slot)
        target = _target(block, name_to_slot)
        goals = tuple(_goal(task_id, slot, "in_container", target, side, block, name_to_slot) for slot in slots)
        return Check(logical="all", goals=goals, order=slots)
    if op == "choose":
        slots = _slots(block.get("object", block.get("objects")), name_to_slot)
        target = _target(block, name_to_slot)
        goals = tuple(_goal(task_id, slot, "in_container", target, side, block, name_to_slot) for slot in slots)
        return Check(logical="choose", goals=goals, k=int(block.get("k", 1)))
    if op not in OPS:
        raise ValueError(f"task {task_id}: unknown success op {op!r}")
    logical = str(block.get("logical", "all"))
    if logical not in ("all", "any", "choose"):
        raise ValueError(f"logical must be all, any, or choose, got {logical!r}")
    slots = _slots(block.get("object", block.get("objects")), name_to_slot)
    target = _target(block, name_to_slot)
    goals = tuple(_goal(task_id, slot, op, target, side, block, name_to_slot) for slot in slots)
    return Check(logical=logical, goals=goals, k=int(block.get("k", 1)))


def _goal(task_id: str, slot: str, op: str, target: str, side: str, block: dict, name_to_slot: dict[str, str]) -> Goal:
    if op not in _NO_TARGET and not target:
        raise ValueError(f"task {task_id}: op {op!r} needs a container, support, or reference")
    tolerance = block.get("tolerance")
    surface = block.get("surface")
    return Goal(
        slot=slot,
        op=op,
        target=target,
        side=side,
        contact=_contact(task_id, op, target, block.get("require_contact_with", False), name_to_slot),
        detach=bool(block.get("require_gripper_detached", False)),
        tolerance=None if tolerance is None else float(tolerance),
        distance=float(block.get("distance", 0.05)),
        surface=_surface_or_slot(surface, name_to_slot) if surface else "",
        up_axis=str(block.get("up_axis", "z")).lower(),
    )


def _contact(task_id: str, op: str, target: str, value, name_to_slot: dict[str, str]) -> tuple[str, ...]:
    if value is None or value is False or value == []:
        return ()
    if value is True:
        if op in _NO_TARGET or not target:
            raise ValueError(f"task {task_id}: require_contact_with true needs an op with a target, not {op!r}")
        return (target,)
    names = value if isinstance(value, list) else [value]
    return tuple(_surface_or_slot(name, name_to_slot) for name in names)


def _surface_or_slot(name, name_to_slot: dict[str, str]) -> str:
    """Slot of ``name``. A bundle may name the work surface without placing it."""
    if name in name_to_slot:
        return name_to_slot[str(name)]
    if name in ("table", SURFACE_SLOT):
        return SURFACE_SLOT
    raise KeyError(f"success names {name!r}, which the task does not place")


def _target(block: dict, name_to_slot: dict[str, str]) -> str:
    for key in ("container", "reference", "support", "target"):
        if key in block:
            return _slots(block[key], name_to_slot)[0]
    return ""


def _slots(value, name_to_slot: dict[str, str]) -> tuple[str, ...]:
    names = value if isinstance(value, list) else [value]
    slots = []
    for name in names:
        if name not in name_to_slot:
            raise KeyError(f"success names {name!r}, which the task does not place")
        slots.append(name_to_slot[str(name)])
    if not slots:
        raise ValueError("success names no objects")
    return tuple(slots)


def _role_map(check: Check) -> dict[str, str]:
    roles = {}
    for goal in check.goals:
        roles.setdefault(goal.slot, goal.slot)
    for child in check.children:
        for role, slot in _role_map(child).items():
            roles.setdefault(role, slot)
    return roles


def goals_of(check: Check) -> list[Goal]:
    """Every stage under ``check``, depth first."""
    found = list(check.goals)
    for child in check.children:
        found.extend(goals_of(child))
    return found
