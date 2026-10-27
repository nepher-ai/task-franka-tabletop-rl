# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Object-center positions [m] that make a benchmark predicate true.

``in_container`` is placed on the container grid in the planner. Every other op is handled here.
"""

from __future__ import annotations

import numpy as np

from franka_tabletop_rl.benchmark.predicates import inside, on_footprint, relative_side

from .goals import PlaceGoal

# Extra gap [m] past the predicate's own threshold, so the object is not left intersecting.
_GAP_M = 0.02


def place_center(
    goal: PlaceGoal,
    item_pos: np.ndarray,
    item_half: np.ndarray,
    target_pos: np.ndarray | None,
    target_half: np.ndarray | None,
    target_rot: np.ndarray | None,
    occupied: list[tuple[np.ndarray, np.ndarray]],
    table_center: np.ndarray,
    table_half: np.ndarray,
) -> np.ndarray:
    """Object-center position [m] that satisfies ``goal``. ``occupied`` is ``(center, half)`` to avoid."""
    item_pos = np.asarray(item_pos, dtype=np.float64)
    item_half = np.asarray(item_half, dtype=np.float64)
    if goal.op in ("on_top", "stacked") and target_pos is not None and target_half is not None:
        center = np.array(target_pos, dtype=np.float64)
        center[2] = float(target_pos[2] + target_half[2] + item_half[2] + _GAP_M)
        return center
    if goal.op == "on_side" and target_pos is not None and target_half is not None:
        return _beside(goal.side, target_pos, target_half, item_half)
    if goal.op == "centered":
        anchor = np.asarray(target_pos if target_pos is not None else table_center, dtype=np.float64)
        center = anchor.copy()
        center[2] = float(item_pos[2])
        return center
    if goal.op in ("outside", "off") and target_pos is not None and target_half is not None:
        return _clear_of(item_pos, item_half, target_pos, target_half, target_rot, table_center, table_half)
    if goal.op == "resting":
        height = goal.margin or float(table_center[2])
        return _free_spot(item_pos, item_half, occupied, table_center, table_half, height)
    return _free_spot(item_pos, item_half, occupied, table_center, table_half, float(item_pos[2]))


def placement_satisfies(
    goal: PlaceGoal, center: np.ndarray, item_half: np.ndarray, target_pos, target_half, target_quat
) -> bool:
    """Whether an object centered at ``center`` [m] meets ``goal``."""
    item = {
        "pos": np.asarray(center, dtype=np.float64).reshape(1, 3),
        "quat": np.array([[1.0, 0.0, 0.0, 0.0]]),
        "half": item_half,
    }
    if target_pos is None:
        return True
    target = {
        "pos": np.asarray(target_pos, dtype=np.float64).reshape(1, 3),
        "half": np.asarray(target_half, dtype=np.float64),
        "quat": None if target_quat is None else np.asarray(target_quat, dtype=np.float64).reshape(1, 4),
    }
    if goal.op in ("on_top", "stacked"):
        return bool(on_footprint(item["pos"], target["pos"], target["half"]).reshape(-1)[0])
    if goal.op == "on_side":
        return bool(relative_side(item["pos"], target["pos"], goal.side).reshape(-1)[0])
    if goal.op == "outside":
        return not bool(inside(item["pos"], target["pos"], target["half"], target["quat"]).reshape(-1)[0])
    return True


def _beside(side: str, target_pos, target_half, item_half) -> np.ndarray:
    axis = {"left": (1, 1.0), "right": (1, -1.0), "front": (0, -1.0), "behind": (0, 1.0)}[side]
    index, sign = axis
    gap = float(target_half[index] + item_half[index] + 0.05)
    center = np.array(target_pos, dtype=np.float64)
    center[index] = float(target_pos[index] + sign * gap)
    center[2] = float(target_pos[2] - target_half[2] + item_half[2])
    return center


def _clear_of(item_pos, item_half, target_pos, target_half, target_rot, table_center, table_half) -> np.ndarray:
    direction = np.array([1.0, 0.0, 0.0])
    if target_rot is not None:
        direction = np.asarray(target_rot, dtype=np.float64)[:, 0]
        direction[2] = 0.0
        norm = float(np.linalg.norm(direction))
        direction = direction / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    reach = float(target_half[0] + item_half[0] + 0.05)
    center = np.array(target_pos, dtype=np.float64) + direction * reach
    center[2] = float(table_center[2] + item_half[2])
    return _clamp_xy(center, item_half, table_center, table_half)


def _free_spot(item_pos, item_half, occupied, table_center, table_half, z: float) -> np.ndarray:
    best = None
    best_distance = 1e9
    x_lo = table_center[0] - table_half[0] + item_half[0]
    x_hi = table_center[0] + table_half[0] - item_half[0]
    y_lo = table_center[1] - table_half[1] + item_half[1]
    y_hi = table_center[1] + table_half[1] - item_half[1]
    for x in np.linspace(x_lo, x_hi, 7):
        for y in np.linspace(y_lo, y_hi, 7):
            center = np.array([float(x), float(y), z])
            if any(_overlap(center, item_half, other, half) for other, half in occupied):
                continue
            distance = float(np.hypot(center[0] - item_pos[0], center[1] - item_pos[1]))
            if distance < best_distance:
                best, best_distance = center, distance
    if best is None:
        best = np.array([float(table_center[0]), float(table_center[1]), z])
    return best


def _overlap(center, half, other, other_half) -> bool:
    return bool(
        abs(center[0] - other[0]) < half[0] + other_half[0] and abs(center[1] - other[1]) < half[1] + other_half[1]
    )


def _clamp_xy(center: np.ndarray, item_half, table_center, table_half) -> np.ndarray:
    limited = center.copy()
    for axis in (0, 1):
        lo = float(table_center[axis] - table_half[axis] + item_half[axis])
        hi = float(table_center[axis] + table_half[axis] - item_half[axis])
        limited[axis] = float(np.clip(limited[axis], lo, hi))
    return limited
