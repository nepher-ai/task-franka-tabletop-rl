# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Object root positions [m] that make a benchmark predicate true, on the success geometry.

``in_container`` and ``enclosed`` are placed on the container grid in the planner. Every other op
is handled here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from franka_tabletop_rl.benchmark.geometry import AssetGeometry

from .goals import PlaceGoal

# Extra gap [m] past the predicate's own threshold, so the object is not left intersecting.
_GAP_M = 0.02
# Free space [m] kept between a container and an object placed beside or outside it.
_CLEARANCE_M = 0.05

_SIDE_AXES = {"left": (1, 1.0), "right": (1, -1.0), "front": (0, -1.0), "behind": (0, 1.0)}


@dataclass(frozen=True)
class Body:
    """Root pose [m] and root-frame geometry of one object.

    Attributes:
        pos: Root position [m], shape ``[3]``.
        rot: Body-to-env rotation matrix, shape ``[3, 3]``.
        geometry: Success geometry of the mesh.
    """

    pos: np.ndarray
    rot: np.ndarray
    geometry: AssetGeometry

    @property
    def corners(self) -> np.ndarray:
        """Box corners [m] in the env frame, shape ``[8, 3]``."""
        return self.pos + self.geometry.box_corners @ self.rot.T

    @property
    def box_center(self) -> np.ndarray:
        """Center [m] of the box around the mesh, in the env frame."""
        return self.pos + self.rot @ self.geometry.box_center


def place_center(
    goal: PlaceGoal, item: Body, target: Body, table_center: np.ndarray, table_half: np.ndarray
) -> np.ndarray:
    """Root position [m] at which ``item`` meets ``goal``, keeping its orientation.

    ``table_center`` and ``table_half`` bound the work surface [m] for ops that drop on it.
    """
    to_box = item.box_center - item.pos
    bottom = float(item.corners[:, 2].min() - item.pos[2])
    if goal.op in ("on_top", "stacked"):
        low, high = target.corners.min(axis=0), target.corners.max(axis=0)
        middle = (low + high) / 2.0
        return np.array([middle[0] - to_box[0], middle[1] - to_box[1], high[2] + _GAP_M - bottom])
    if goal.op == "on_side":
        return _beside(goal.side, item, target, bottom)
    if goal.op == "centered":
        middle = target.box_center
        return np.array([middle[0] - to_box[0], middle[1] - to_box[1], item.pos[2]])
    if goal.op in ("outside", "outside_and_on_surface"):
        return _clear_of(item, target, table_center, table_half, bottom)
    raise KeyError(f"planner has no placement for {goal.op}")


def _beside(side: str, item: Body, target: Body, bottom: float) -> np.ndarray:
    index, sign = _SIDE_AXES[side]
    # The side test reads root positions, so the root of the item steps off the root of the target.
    edge = float((sign * (target.corners - target.pos)[:, index]).max())
    back = float((-sign * (item.corners - item.pos)[:, index]).max())
    center = np.array(target.pos, dtype=np.float64)
    center[index] += sign * (edge + back + _CLEARANCE_M)
    center[2] = float(target.corners[:, 2].min() - bottom)
    return center


def _clear_of(item: Body, target: Body, table_center, table_half, bottom: float) -> np.ndarray:
    direction = np.array(target.rot[:, 0], dtype=np.float64)
    direction[2] = 0.0
    norm = float(np.linalg.norm(direction))
    direction = direction / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    edge = float(((target.corners - target.pos) @ direction).max())
    back = float((-(item.corners - item.pos) @ direction).max())
    center = np.array(target.pos, dtype=np.float64) + direction * (edge + back + _CLEARANCE_M)
    center[2] = float(table_center[2] - bottom)
    half = (item.corners.max(axis=0) - item.corners.min(axis=0)) / 2.0
    return _clamp_xy(center, half, table_center, table_half)


def _clamp_xy(center: np.ndarray, item_half, table_center, table_half) -> np.ndarray:
    limited = center.copy()
    for axis in (0, 1):
        lo = float(table_center[axis] - table_half[axis] + item_half[axis])
        hi = float(table_center[axis] + table_half[axis] - item_half[axis])
        limited[axis] = float(np.clip(limited[axis], lo, hi))
    return limited
