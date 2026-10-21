# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success predicates on object poses. Positions are in meters."""

from __future__ import annotations

import numpy as np

from .spec import Check, Goal, TabletopTaskSpec

_KINDS = ("item_in_container", "item_on_side", "stacked", "sorted_one", "both_in_container")


def inside(
    item_pos: np.ndarray,
    container_pos: np.ndarray,
    half_extents: np.ndarray,
    container_quat_wxyz: np.ndarray | None = None,
) -> np.ndarray:
    """True when the item position is inside the container box.

    ``half_extents`` are the container's local half sizes. The box is open upward: one
    container-height of air above the rim still counts, so an object sitting in a
    shallow bin is inside when its center is above the rim. ``container_quat_wxyz`` is
    the box orientation. Without it the box is axis-aligned in the scene frame.

    Args:
        item_pos: Item positions [m], shape [N, 3].
        container_pos: Container positions [m], shape [N, 3].
        half_extents: Container half sizes [m], shape [3] or [N, 3].
        container_quat_wxyz: Container orientation, scalar-first, shape [4] or [N, 4].
    """
    delta = np.asarray(item_pos, dtype=np.float64) - np.asarray(container_pos, dtype=np.float64)
    if container_quat_wxyz is not None:
        delta = _world_to_body(container_quat_wxyz, delta)
    half = np.asarray(half_extents, dtype=np.float64)
    upper = np.array(half, copy=True)
    upper[..., 2] = half[..., 2] * 3.0
    return np.all(delta <= upper, axis=-1) & np.all(delta >= -half, axis=-1)


def _world_to_body(quat_wxyz: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Rotate scene-frame offsets into the container frame. ``quat_wxyz`` is scalar-first."""
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    vector = np.asarray(delta, dtype=np.float64)
    if quat.ndim == 1:
        quat = np.broadcast_to(quat, vector.shape[:-1] + (4,))
    w, x, y, z = np.moveaxis(quat, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    # Rows of the body-to-scene matrix, so the scene offset is projected onto the body axes.
    axes = (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy)),
        (2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)),
        (2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy)),
    )
    return np.stack(
        [axis[0] * vector[..., 0] + axis[1] * vector[..., 1] + axis[2] * vector[..., 2] for axis in axes],
        axis=-1,
    )


def on_footprint(item_pos: np.ndarray, support_pos: np.ndarray, support_half: np.ndarray) -> np.ndarray:
    """True when the item's horizontal position lies in the support footprint and the item is above it.

    Args:
        item_pos: Item positions [m], shape [N, 3].
        support_pos: Support positions [m], shape [N, 3].
        support_half: Support half sizes [m], shape [3] or [N, 3].
    """
    delta = np.asarray(item_pos) - np.asarray(support_pos)
    half = np.asarray(support_half)
    horizontal = (np.abs(delta[..., 0]) <= half[..., 0]) & (np.abs(delta[..., 1]) <= half[..., 1])
    return horizontal & (delta[..., 2] > 0.0)


def on_top_of(item_pos: np.ndarray, support_pos: np.ndarray, xy_tol_m: float, z_tol_m: float) -> np.ndarray:
    """True when the item sits above the support within a horizontal tolerance."""
    delta = np.asarray(item_pos) - np.asarray(support_pos)
    horizontal = np.linalg.norm(delta[..., :2], axis=-1) <= xy_tol_m
    vertical = (delta[..., 2] > 0.0) & (delta[..., 2] <= z_tol_m)
    return horizontal & vertical


def relative_side(item_pos: np.ndarray, reference_pos: np.ndarray, side: str, gap_m: float = 0.04) -> np.ndarray:
    """True when the item is on the named side of the reference in the table plane.

    Sides are ``left`` (+y), ``right`` (-y), ``front`` (+x), and ``behind`` (-x).
    """
    delta = np.asarray(item_pos) - np.asarray(reference_pos)
    axis = {"left": (1, 1.0), "right": (1, -1.0), "front": (0, 1.0), "behind": (0, -1.0)}[side]
    index, sign = axis
    return delta[..., index] * sign >= gap_m


def lifted(item_pos: np.ndarray, table_z_m: float, height_m: float = 0.05) -> np.ndarray:
    """True when the item is above the table by ``height_m`` [m]."""
    return np.asarray(item_pos)[..., 2] >= table_z_m + height_m


def upright(quat_wxyz: np.ndarray, min_w: float = 0.85) -> np.ndarray:
    """True when a wxyz quaternion is close to identity."""
    quat = np.asarray(quat_wxyz)
    return np.abs(quat[..., 0]) >= min_w


def _body_to_world(quat_wxyz: np.ndarray) -> np.ndarray:
    """Rotation that takes body-frame offsets into the scene frame. ``quat_wxyz`` is scalar-first."""
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1)[:4]
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _oriented_corners(pose: dict) -> np.ndarray:
    """Eight corners [m] of the pose's oriented box, shape [8, 3]."""
    origin = np.asarray(pose["pos"], dtype=np.float64).reshape(-1, 3)[0]
    half = np.asarray(pose["half"], dtype=np.float64).reshape(-1)[:3]
    signs = np.array([[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])
    return origin + (signs * half) @ _body_to_world(pose["quat"]).T


def _lowest_z(pose: dict) -> float:
    """World-frame bottom [m] of the pose's oriented box."""
    return float(_oriented_corners(pose)[:, 2].min())


def _fraction_inside(item: dict, container: dict) -> float:
    """Share of the item's corners that lie in the container's open-top box."""
    corners = _oriented_corners(item)
    mask = inside(corners, container["pos"], container["half"], container.get("quat"))
    return float(np.mean(mask))


def _latch_key(goal: Goal) -> str:
    """Identity of one stage. Two checks on the same object must not share a latch."""
    return f"{goal.op}:{goal.slot}:{goal.target}"


def _let_go(slot: str, held: dict[str, bool], in_contact: dict[str, bool], gripper_open: bool) -> bool:
    """True when the slot was held, the fingers are open, and contact has ended.

    A missing contact entry counts as still touching, so a missing sensor cannot pass.
    """
    return bool(held.get(slot, False) and not in_contact.get(slot, True) and gripper_open)


def all_of(*masks: np.ndarray) -> np.ndarray:
    """Elementwise conjunction."""
    result = np.ones_like(np.asarray(masks[0]), dtype=bool)
    for mask in masks:
        result = result & np.asarray(mask, dtype=bool)
    return result


def any_of(*masks: np.ndarray) -> np.ndarray:
    """Elementwise disjunction."""
    result = np.zeros_like(np.asarray(masks[0]), dtype=bool)
    for mask in masks:
        result = result | np.asarray(mask, dtype=bool)
    return result


def evaluate(spec: TabletopTaskSpec, poses: dict[str, dict[str, np.ndarray]], side: str | None = None) -> np.ndarray:
    """Evaluate ``spec.success`` for a batch of poses.

    Args:
        spec: Family spec or benchmark expression. Roles name the keys of ``poses``.
        poses: ``slot -> {"pos": [N, 3], "quat": [N, 4], "half": [3]}``.
        side: Overrides ``spec.side`` for ``place_relative``.
    """
    if spec.check is not None and spec.success not in _KINDS:
        return np.asarray([_geometry(spec.check, poses, side or spec.side)])
    kind = spec.success
    if kind == "item_in_container":
        item = poses[spec.roles["item"]]
        box = poses[spec.roles["container"]]
        return inside(item["pos"], box["pos"], box["half"], box.get("quat"))
    if kind == "item_on_side":
        item = poses[spec.roles["item"]]
        reference = poses[spec.roles["reference"]]
        return relative_side(item["pos"], reference["pos"], side or spec.side)
    if kind == "stacked":
        top = poses[spec.roles["top"]]
        support = poses[spec.roles["support"]]
        return all_of(on_top_of(top["pos"], support["pos"], 0.04, 0.12), upright(top["quat"]))
    if kind == "sorted_one":
        keep = poses[spec.roles["keep"]]
        leave = poses[spec.roles["leave"]]
        box = poses[spec.roles["container"]]
        return all_of(
            inside(keep["pos"], box["pos"], box["half"], box.get("quat")),
            ~inside(leave["pos"], box["pos"], box["half"], box.get("quat")),
        )
    if kind == "both_in_container":
        first = poses[spec.roles["first"]]
        second = poses[spec.roles["second"]]
        box = poses[spec.roles["container"]]
        return all_of(
            inside(first["pos"], box["pos"], box["half"], box.get("quat")),
            inside(second["pos"], box["pos"], box["half"], box.get("quat")),
        )
    raise KeyError(f"unknown success predicate {kind!r}")


def required_grasp_slots(spec: TabletopTaskSpec) -> tuple[str, ...]:
    """Slots the gripper must have held before the geometric condition counts."""
    if spec.check is not None:
        return spec.check.grasp
    roles = spec.roles
    if spec.success == "both_in_container":
        return (roles["first"], roles["second"])
    if spec.success == "stacked":
        return (roles["top"],)
    if spec.success == "sorted_one":
        return (roles["keep"],)
    if spec.success in ("item_in_container", "item_on_side"):
        return (roles["item"],)
    return ()


def terminal_success(
    spec: TabletopTaskSpec,
    poses: dict[str, dict[str, np.ndarray]],
    held: dict[str, bool],
    in_contact: dict[str, bool],
    side: str | None = None,
    *,
    gripper_open: bool = True,
) -> bool:
    """True when the geometry holds, each required object was grasped, and the gripper has opened.

    ``held`` is sticky. ``in_contact`` is the contact at this step. ``gripper_open`` is the
    finger joint on the open side of the 0.5 snap. A closed gripper does not count as a
    release when the contact report drops for one step.
    """
    if spec.check is not None:
        score, _, _ = task_progress(spec, poses, held, in_contact, {}, [], side=side, gripper_open=gripper_open)
        return score >= 1.0 - 1e-9
    if not bool(np.asarray(evaluate(spec, poses, side=side)).reshape(-1)[0]):
        return False
    return all(_let_go(slot, held, in_contact, gripper_open) for slot in required_grasp_slots(spec))


def task_progress(
    spec: TabletopTaskSpec,
    poses: dict[str, dict[str, np.ndarray]],
    held: dict[str, bool],
    in_contact: dict[str, bool],
    latched: dict[str, bool],
    entered: list[str],
    side: str | None = None,
    *,
    gripper_open: bool = True,
) -> tuple[float, dict[str, bool], list[str]]:
    """Latched subtask progress in ``[0, 1]``.

    A grasp alone scores 0. Placing an object, opening the gripper, and ending contact
    scores 1 for that object. ``all`` averages those scores, ``any`` takes the max, and
    ``choose`` averages the top ``k``. ``latched`` keeps a finished object finished.
    ``entered`` is the order in which named objects first satisfied ``in_container``.

    Returns the score, the updated latch, and the updated entry order.
    """
    chosen = side or spec.side
    if spec.check is None:
        done = bool(np.asarray(evaluate(spec, poses, side=chosen)).reshape(-1)[0])
        released = all(_let_go(slot, held, in_contact, gripper_open) for slot in required_grasp_slots(spec))
        return (1.0 if done and released else 0.0), dict(latched), list(entered)
    kept = dict(latched)
    sequence = list(entered)
    score = _score(
        spec.check,
        poses,
        held,
        in_contact,
        kept,
        sequence,
        chosen,
        spec.check.grasp,
        spec.check.hold,
        gripper_open,
    )
    return score, kept, sequence


def _score(
    check: Check,
    poses: dict,
    held: dict[str, bool],
    in_contact: dict[str, bool],
    latched: dict[str, bool],
    entered: list[str],
    side: str,
    grasp: tuple[str, ...],
    hold: tuple[str, ...],
    gripper_open: bool,
) -> float:
    active = check.grasp or grasp
    active_hold = check.hold or hold
    if check.order:
        return _order_score(
            check, poses, held, in_contact, latched, entered, side, active, active_hold, gripper_open
        )
    if check.children:
        values = [
            _score(child, poses, held, in_contact, latched, entered, side, active, active_hold, gripper_open)
            for child in check.children
        ]
        return _reduce(check.logical, values, check.k)
    values = [
        _credit(goal, active, active_hold, poses, held, in_contact, latched, side, gripper_open)
        for goal in check.goals
    ]
    return _reduce(check.logical, values, check.k)


def _order_score(
    check: Check,
    poses: dict,
    held: dict[str, bool],
    in_contact: dict[str, bool],
    latched: dict[str, bool],
    entered: list[str],
    side: str,
    grasp: tuple[str, ...],
    hold: tuple[str, ...],
    gripper_open: bool,
) -> float:
    for goal in check.goals:
        if goal.slot not in entered and _goal_met(goal, poses, side):
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
        values.append(_credit(goal, grasp, hold, poses, held, in_contact, latched, side, gripper_open))
    return _reduce("all", values, 1)


def _credit(
    goal: Goal,
    grasp: tuple[str, ...],
    hold: tuple[str, ...],
    poses: dict,
    held: dict[str, bool],
    in_contact: dict[str, bool],
    latched: dict[str, bool],
    side: str,
    gripper_open: bool,
) -> float:
    if goal.slot in hold:
        if not _goal_met(goal, poses, side):
            return 0.0
        # A pickup stays successful only while the gripper is still closed on the object.
        return 1.0 if held.get(goal.slot, False) and in_contact.get(goal.slot, False) else 0.0
    key = _latch_key(goal)
    if latched.get(key):
        # Taking an object out has to stay clear, with the gripper still open. Other stages keep credit.
        released = goal.slot not in grasp or _let_go(goal.slot, held, in_contact, gripper_open)
        if goal.op == "outside" and not (_goal_met(goal, poses, side) and released):
            latched.pop(key, None)
            return 0.0
        return 1.0
    if not _goal_met(goal, poses, side):
        return 0.0
    if goal.slot in grasp and not _let_go(goal.slot, held, in_contact, gripper_open):
        return 0.0
    latched[key] = True
    return 1.0


def _goal_met(goal: Goal, poses: dict, side: str) -> bool:
    item = poses[goal.slot]
    target = poses.get(goal.target)
    if goal.op == "in_container":
        return bool(inside(item["pos"], target["pos"], target["half"], target.get("quat")).reshape(-1)[0])
    if goal.op == "outside":
        # Every corner has to clear the container. A long spoon still in a bowl already
        # has part of its box above the rim.
        return _fraction_inside(item, target) == 0.0
    if goal.op == "on_side":
        return bool(relative_side(item["pos"], target["pos"], goal.side or side).reshape(-1)[0])
    if goal.op == "on_top":
        return bool(on_footprint(item["pos"], target["pos"], target["half"]).reshape(-1)[0])
    if goal.op == "stacked":
        stacked = on_top_of(item["pos"], target["pos"], 0.04, 0.12)
        return bool(all_of(stacked, upright(item["quat"])).reshape(-1)[0])
    if goal.op == "off":
        # A tilted cube that is still sitting on the support is not unstacked.
        stacked = on_top_of(item["pos"], target["pos"], 0.04, 0.12)
        return not bool(np.asarray(stacked).reshape(-1)[0])
    if goal.op == "upright":
        return bool(upright(item["quat"]).reshape(-1)[0])
    if goal.op == "lifted":
        return bool(lifted(item["pos"], 0.0).reshape(-1)[0])
    if goal.op == "picked":
        # Use the oriented bottom. Subtracting the local z extent accepts a sideways
        # object whose center has risen while a corner is still on the table.
        return _lowest_z(item) >= goal.margin
    if goal.op == "resting":
        # ``margin`` is the table height [m]. The old band reached 20 cm above it, so a
        # keyboard on a bin rim and a cube still in a tower both counted as on the table.
        height = float(np.asarray(item["pos"])[..., 2].reshape(-1)[0])
        return goal.margin - 0.05 < height < goal.margin + 0.06
    if goal.op == "centered":
        delta = np.asarray(item["pos"]) - np.asarray(target["pos"])
        limit = goal.margin if goal.margin > 0.0 else 0.05
        return bool(np.linalg.norm(delta[..., :2], axis=-1).reshape(-1)[0] <= limit)
    raise KeyError(f"unknown success predicate {goal.op!r}")


def _geometry(check: Check, poses: dict, side: str) -> bool:
    if check.order or not check.children:
        met = [_goal_met(goal, poses, side) for goal in check.goals]
        if check.logical == "any":
            return any(met)
        if check.logical == "choose":
            return sum(met) >= check.k
        return all(met)
    met = [_geometry(child, poses, side) for child in check.children]
    if check.logical == "any":
        return any(met)
    if check.logical == "choose":
        return sum(bool(item) for item in met) >= check.k
    return all(met)


def _reduce(logical: str, values: list[float], k: int) -> float:
    if not values:
        return 0.0
    if logical == "any":
        return float(max(values))
    if logical == "choose":
        ordered = sorted(values, reverse=True)
        return float(np.mean(ordered[: max(k, 1)]))
    return float(np.mean(values))
