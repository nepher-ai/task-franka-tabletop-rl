# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success predicates on object poses. Positions are [m]."""

from __future__ import annotations

import numpy as np

from .spec import Check, Goal, TabletopTaskSpec


def inside(
    item_pos: np.ndarray,
    container_pos: np.ndarray,
    half_extents: np.ndarray,
    container_quat_wxyz: np.ndarray | None = None,
) -> np.ndarray:
    """Whether the item center is inside the container box.

    ``half_extents`` are local half sizes [m]. The box is open upward by one container height.
    Without ``container_quat_wxyz`` (wxyz) the box is axis-aligned. Positions are ``[N, 3]``.
    """
    delta = np.asarray(item_pos, dtype=np.float64) - np.asarray(container_pos, dtype=np.float64)
    if container_quat_wxyz is not None:
        delta = _world_to_body(container_quat_wxyz, delta)
    half = np.asarray(half_extents, dtype=np.float64)
    upper = np.array(half, copy=True)
    upper[..., 2] = half[..., 2] * 3.0
    return np.all(delta <= upper, axis=-1) & np.all(delta >= -half, axis=-1)


def _world_to_body(quat_wxyz: np.ndarray, delta: np.ndarray) -> np.ndarray:
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
    """Whether the item's xy lies in the support footprint and the item is above it. Positions [m]."""
    delta = np.asarray(item_pos) - np.asarray(support_pos)
    half = np.asarray(support_half)
    horizontal = (np.abs(delta[..., 0]) <= half[..., 0]) & (np.abs(delta[..., 1]) <= half[..., 1])
    return horizontal & (delta[..., 2] > 0.0)


def on_top_of(item_pos: np.ndarray, support_pos: np.ndarray, xy_tol_m: float, z_tol_m: float) -> np.ndarray:
    """Whether the item is above the support, within ``xy_tol_m`` [m] and ``z_tol_m`` [m]."""
    delta = np.asarray(item_pos) - np.asarray(support_pos)
    horizontal = np.linalg.norm(delta[..., :2], axis=-1) <= xy_tol_m
    vertical = (delta[..., 2] > 0.0) & (delta[..., 2] <= z_tol_m)
    return horizontal & vertical


def relative_side(item_pos: np.ndarray, reference_pos: np.ndarray, side: str, gap_m: float = 0.04) -> np.ndarray:
    """Whether the item is ``gap_m`` [m] to the ``left`` (+y), ``right`` (-y), ``front`` (-x), or ``behind`` (+x).

    The robot faces +x from the env origin, so ``front`` is toward the robot, as in RoboLab.
    """
    delta = np.asarray(item_pos) - np.asarray(reference_pos)
    axis = {"left": (1, 1.0), "right": (1, -1.0), "front": (0, -1.0), "behind": (0, 1.0)}[side]
    index, sign = axis
    return delta[..., index] * sign >= gap_m


def lifted(item_pos: np.ndarray, table_z_m: float, height_m: float = 0.05) -> np.ndarray:
    """Whether the item is at least ``height_m`` [m] above ``table_z_m``."""
    return np.asarray(item_pos)[..., 2] >= table_z_m + height_m


def upright(quat_wxyz: np.ndarray, min_w: float = 0.85) -> np.ndarray:
    """Whether a wxyz quaternion is within ``min_w`` of identity."""
    quat = np.asarray(quat_wxyz)
    return np.abs(quat[..., 0]) >= min_w


def _body_to_world(quat_wxyz: np.ndarray) -> np.ndarray:
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
    origin = np.asarray(pose["pos"], dtype=np.float64).reshape(-1, 3)[0]
    half = np.asarray(pose["half"], dtype=np.float64).reshape(-1)[:3]
    signs = np.array([[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])
    return origin + (signs * half) @ _body_to_world(pose["quat"]).T


def _lowest_z(pose: dict) -> float:
    return float(_oriented_corners(pose)[:, 2].min())


def _fraction_inside(item: dict, container: dict) -> float:
    corners = _oriented_corners(item)
    mask = inside(corners, container["pos"], container["half"], container.get("quat"))
    return float(np.mean(mask))


def latch_key(goal: Goal) -> str:
    """Latch id of one stage, so two checks on the same object stay distinct."""
    return f"{goal.op}:{goal.slot}:{goal.target}"


def _let_go(slot: str, held: dict[str, bool], in_contact: dict[str, bool], gripper_open: bool) -> bool:
    """Whether the slot was held, the fingers are open, and contact has ended.

    A missing contact entry still counts as touching.
    """
    return bool(held.get(slot, False) and not in_contact.get(slot, True) and gripper_open)


def all_of(*masks: np.ndarray) -> np.ndarray:
    result = np.ones_like(np.asarray(masks[0]), dtype=bool)
    for mask in masks:
        result = result & np.asarray(mask, dtype=bool)
    return result


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
    """Latched subtask progress in ``[0, 1]``, the updated latch, and the ``in_container`` entry order.

    A grasp alone scores 0. Release after the pose scores 1. ``all`` averages, ``any`` takes the max,
    and ``choose`` averages the top ``k``.
    """
    chosen = side or spec.side
    if spec.check is None:
        return 0.0, dict(latched), list(entered)
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
    key = latch_key(goal)
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
