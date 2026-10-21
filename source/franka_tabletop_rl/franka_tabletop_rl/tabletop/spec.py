# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task specification. A family names object roles and a success predicate."""

from __future__ import annotations

from dataclasses import dataclass, replace

_MOVED = ("in_container", "on_side", "on_top", "stacked", "lifted", "centered")
_SHORT = {
    "item_in_container": ("in_container", ("item",), "container"),
    "item_on_side": ("on_side", ("item",), "reference"),
    "stacked": ("stacked", ("top",), "support"),
    "sorted_one": None,
    "both_in_container": ("in_container", ("first", "second"), "container"),
}


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
    """One manipulation family or one benchmark expression.

    Attributes:
        task_id: Stable family id, or the authored task id.
        roles: Role name to scene slot. Local play families use this mapping.
        success: Short-form predicate name, or the expression op.
        failure: Predicate name. Empty means no failure besides the time limit.
        time_limit_s: Unused by the evaluator. Episode time comes from the eval config.
        side: Used by ``place_relative``. One of left, right, front, behind.
        check: Expression the evaluator scores. Local play families leave this empty.
    """

    task_id: str
    roles: dict[str, str]
    success: str
    failure: str
    time_limit_s: float
    side: str = "left"
    check: Check | None = None


FAMILIES: dict[str, TabletopTaskSpec] = {
    "place_in_container": TabletopTaskSpec(
        task_id="place_in_container",
        roles={"item": "item_a", "container": "container"},
        success="item_in_container",
        failure="",
        time_limit_s=40.0,
    ),
    "place_relative": TabletopTaskSpec(
        task_id="place_relative",
        roles={"item": "item_a", "reference": "reference"},
        success="item_on_side",
        failure="",
        time_limit_s=40.0,
        side="left",
    ),
    "stack_on": TabletopTaskSpec(
        task_id="stack_on",
        roles={"top": "item_a", "support": "item_b"},
        success="stacked",
        failure="",
        time_limit_s=40.0,
    ),
    "sort_by_attribute": TabletopTaskSpec(
        task_id="sort_by_attribute",
        roles={"keep": "item_a", "leave": "item_b", "container": "container"},
        success="sorted_one",
        failure="",
        time_limit_s=50.0,
    ),
    "multi_place": TabletopTaskSpec(
        task_id="multi_place",
        roles={"first": "item_a", "second": "item_b", "container": "container"},
        success="both_in_container",
        failure="",
        time_limit_s=60.0,
    ),
}


def spec_from_success(
    task_id: str,
    success: dict,
    time_limit_s: float,
    name_to_slot: dict[str, str],
) -> TabletopTaskSpec:
    """Build a spec from a scene success block.

    The short form is ``kind`` plus ``roles`` of object names. The general form is an
    ``op`` with an object name or a list of names. ``require_grasp`` overrides the
    objects that must be grasped and released.
    """
    side = str(success.get("side", "left"))
    if "kind" in success and "op" not in success and "checks" not in success:
        check, roles = _short_form(success, name_to_slot, side)
        label = str(success["kind"])
    else:
        check = _parse(success, name_to_slot, side)
        roles = _role_map(check)
        label = str(success.get("op", success.get("kind", "")))
    if "require_grasp" in success:
        grasp = _named(success["require_grasp"], name_to_slot)
    else:
        grasp = _default_grasp(check)
    hold = _named(success["hold"], name_to_slot) if "hold" in success else ()
    return TabletopTaskSpec(
        task_id=task_id,
        roles=roles,
        success=label,
        failure="",
        time_limit_s=float(time_limit_s),
        side=side,
        check=replace(check, grasp=grasp, hold=hold),
    )


def _short_form(success: dict, name_to_slot: dict[str, str], side: str) -> tuple[Check, dict[str, str]]:
    kind = str(success["kind"])
    roles = {}
    for role, name in success["roles"].items():
        if not isinstance(name, str) or name not in name_to_slot:
            raise KeyError(f"success role {role} names {name!r}, which the scene does not place")
        roles[str(role)] = name_to_slot[name]
    if kind == "sorted_one":
        goals = (
            Goal(roles["keep"], "in_container", roles["container"], side),
            Goal(roles["leave"], "outside", roles["container"], side),
        )
        return Check(logical="all", goals=goals), roles
    spec = _SHORT.get(kind)
    if spec is None:
        raise KeyError(f"unknown success predicate {kind!r}")
    op, role_names, target_role = spec
    goals = tuple(Goal(roles[role], op, roles[target_role], side) for role in role_names)
    return Check(logical="all", goals=goals), roles


def _parse(block: dict, name_to_slot: dict[str, str], side: str) -> Check:
    op = str(block.get("op") or block.get("kind") or "")
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
            raise KeyError(f"success names {name!r}, which the scene does not place")
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


def get_family(task_id: str) -> TabletopTaskSpec:
    """Return the family spec."""
    try:
        return FAMILIES[task_id]
    except KeyError as exc:
        known = ", ".join(sorted(FAMILIES))
        raise KeyError(f"unknown task family {task_id!r}. Known: {known}") from exc
