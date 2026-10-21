# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success expression of one benchmark task."""

from __future__ import annotations

from dataclasses import dataclass, replace

_MOVED = ("in_container", "on_side", "on_top", "stacked", "lifted", "centered")


@dataclass(frozen=True)
class Goal:
    """One object's geometric stage.

    Attributes:
        slot: Scene slot this stage scores.
        op: Predicate name. One of ``in_container``, ``on_side``, ``on_top``, ``stacked``,
            ``upright``, ``lifted``, ``picked``, ``resting``, ``outside``, ``off``, or ``centered``.
        target: Slot of the container, support, or reference. Empty when the stage has none.
        side: Used by ``on_side``. One of left, right, front, behind.
        margin: Table height [m] for ``resting``, lift height [m] for ``picked``, and distance [m] for ``centered``.
    """

    slot: str
    op: str
    target: str = ""
    side: str = "left"
    margin: float = 0.0


@dataclass(frozen=True)
class Check:
    """Success expression. ``all`` averages object scores, ``any`` takes the max, ``choose`` the top ``k``.

    Attributes:
        logical: How to combine ``goals`` or ``children``. One of all, any, choose.
        goals: Object stages scored by this node.
        children: Nested expressions. Used instead of ``goals`` when the block groups checks.
        grasp: Slots that must be grasped and then released before their stage counts.
        hold: Slots that count only while the gripper is still holding them.
        k: How many objects ``choose`` averages.
        order: Slot order for ``in_order``. Empty for every other check.
    """

    logical: str
    goals: tuple[Goal, ...] = ()
    children: tuple[Check, ...] = ()
    grasp: tuple[str, ...] = ()
    hold: tuple[str, ...] = ()
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
    """Build a spec from a task success block.

    The block is an ``op`` with an object name or a list of names. ``require_grasp`` overrides the
    objects that must be grasped and released.

    Raises:
        ValueError: The block is the old short ``kind`` form, which this project does not load.
    """
    if "kind" in success and "op" not in success and "checks" not in success:
        raise ValueError(f"task {task_id} uses a short-form success kind; write it as an op")
    side = str(success.get("side", "left"))
    check = _parse(success, name_to_slot, side)
    if "require_grasp" in success:
        grasp = _named(success["require_grasp"], name_to_slot)
    else:
        grasp = _default_grasp(check)
    hold = _named(success["hold"], name_to_slot) if "hold" in success else ()
    return TabletopTaskSpec(
        task_id=task_id,
        roles=_role_map(check),
        success=str(success.get("op", "")),
        failure="",
        time_limit_s=float(time_limit_s),
        side=side,
        check=replace(check, grasp=grasp, hold=hold),
    )


def _parse(block: dict, name_to_slot: dict[str, str], side: str) -> Check:
    op = str(block.get("op") or "")
    side = str(block.get("side", side))
    if op in ("all", "any", "choose") and "checks" in block:
        children = tuple(_parse(child, name_to_slot, side) for child in block["checks"])
        return Check(logical=op, children=children, k=int(block.get("k", 1)))
    if op == "in_order":
        slots = _slots(block.get("object", block.get("objects")), name_to_slot)
        target = _target(block, name_to_slot)
        goals = tuple(_goal(slot, "in_container", target, side, block) for slot in slots)
        return Check(logical="all", goals=goals, order=slots)
    if op == "choose":
        slots = _slots(block.get("object", block.get("objects")), name_to_slot)
        target = _target(block, name_to_slot)
        goals = tuple(_goal(slot, "in_container", target, side, block) for slot in slots)
        return Check(logical="choose", goals=goals, k=int(block.get("k", 1)))
    if op in ("all", "any"):
        raise ValueError(f"success op {op!r} needs a checks list")
    logical = str(block.get("logical", "all"))
    if logical not in ("all", "any", "choose"):
        raise ValueError(f"logical must be all, any, or choose, got {logical!r}")
    slots = _slots(block.get("object", block.get("objects")), name_to_slot)
    target = _target(block, name_to_slot)
    goals = tuple(_goal(slot, op, target, side, block) for slot in slots)
    return Check(logical=logical, goals=goals, k=int(block.get("k", 1)))


def _goal(slot: str, op: str, target: str, side: str, block: dict) -> Goal:
    return Goal(slot, op, target, side, float(block.get("margin", 0.0)))


def _named(value, name_to_slot: dict[str, str]) -> tuple[str, ...]:
    """Map object names to slots. An empty list means no slots."""
    if value is None or value == []:
        return ()
    return _slots(value, name_to_slot)


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


def _default_grasp(check: Check) -> tuple[str, ...]:
    found: list[str] = []
    for goal in check.goals:
        if goal.op in _MOVED and goal.slot not in found:
            found.append(goal.slot)
    for child in check.children:
        for slot in _default_grasp(child):
            if slot not in found:
                found.append(slot)
    return tuple(found)


def _role_map(check: Check) -> dict[str, str]:
    roles = {}
    for goal in check.goals:
        roles.setdefault(goal.slot, goal.slot)
    for child in check.children:
        for role, slot in _role_map(child).items():
            roles.setdefault(role, slot)
    return roles
