# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Top-down grasp search on surface points. The gripper points down and closes on one horizontal axis.

A grasp fits the opening, leaves a descent path for the open fingers, and clears the palm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# 2F-85 pad and opening geometry [m].
PAD_HALF_WIDTH_M = 0.011
PAD_HALF_LENGTH_M = 0.019
OPEN_HALF_M = 0.0425
FINGER_OUTER_M = 0.060
MAX_WIDTH_M = 0.078
PALM_CLEARANCE_M = 0.045
PALM_HALF_M = 0.05
# Lowest tool height [m] that keeps the fingertips off the table.
TOOL_Z_MIN_M = 0.04
_STEP_M = 0.008
_YAWS = np.deg2rad(np.arange(-90.0, 90.0, 15.0))


@dataclass(frozen=True)
class Grasp:
    """One feasible grasp in the object root frame.

    ``center_local`` is the tool position [m], ``axis_local`` the horizontal closing axis,
    ``width_m`` the thickness [m] between the pads, and ``score`` is larger when better.
    """

    center_local: np.ndarray
    axis_local: np.ndarray
    width_m: float
    score: float


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix of a wxyz quaternion."""
    w, x, y, z = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def sample_prim_points(stage, prim_path: str, count: int = 6000, seed: int = 0) -> np.ndarray:
    """Surface points [m] under ``prim_path``, world frame, shape ``[count, 3]``, sampled by triangle area."""
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise ValueError(f"no prim at {prim_path}")
    triangles = []
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if UsdGeom.Imageable(prim).GetPurposeAttr().Get() == UsdGeom.Tokens.guide:
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts or not indices:
            continue
        matrix = np.array(cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        local = np.asarray(points, dtype=np.float64)
        world = local @ matrix[:3, :3] + matrix[3, :3]
        corner = np.asarray(indices, dtype=np.int64)
        sizes = np.asarray(counts, dtype=np.int64)
        starts = np.cumsum(sizes) - sizes
        # Fan-triangulate each polygon around its first vertex.
        for extra in range(1, int(sizes.max()) - 1):
            valid = sizes > extra + 1
            first = corner[starts[valid]]
            second = corner[starts[valid] + extra]
            third = corner[starts[valid] + extra + 1]
            triangles.append(np.stack([world[first], world[second], world[third]], axis=1))
    if not triangles:
        raise ValueError(f"no mesh triangles under {prim_path}")
    tris = np.concatenate(triangles, axis=0)
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=-1)
    if float(area.sum()) <= 0.0:
        raise ValueError(f"degenerate mesh under {prim_path}")
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(tris), size=count, p=area / area.sum())
    bary = rng.random((count, 2))
    flip = bary.sum(axis=1) > 1.0
    bary[flip] = 1.0 - bary[flip]
    a, b, c = tris[pick, 0], tris[pick, 1], tris[pick, 2]
    return a + bary[:, :1] * (b - a) + bary[:, 1:] * (c - a)


def find_grasps(points: np.ndarray, root_z: float, limit: int = 12) -> list[Grasp]:
    """Feasible top-down grasps, best first.

    ``points`` are surface points [m] in the object root frame with world-aligned axes, ``[M, 3]``.
    ``root_z`` is the root height [m] above the table.
    """
    pts = np.asarray(points, dtype=np.float64)
    z_world = pts[:, 2] + root_z
    centroid = pts.mean(axis=0)
    # Grip tall objects above their middle: the fingertips then stay clear of the table and of the base.
    preferred_z = 0.5 * (float(centroid[2]) + float(pts[:, 2].max())) + root_z
    low = max(TOOL_Z_MIN_M, float(z_world.min()) + 0.006)
    high = max(float(z_world.max()) - 0.004, low)
    levels = np.arange(low, high + 1e-6, 0.008)
    if len(levels) > 14:
        levels = np.linspace(low, high, 14)
    found: list[Grasp] = []
    for yaw in _YAWS:
        u_axis = np.array([-math.sin(yaw), math.cos(yaw)])
        v_axis = np.array([math.cos(yaw), math.sin(yaw)])
        pu = pts[:, :2] @ u_axis
        pv = pts[:, :2] @ v_axis
        for cv in np.arange(pv.min(), pv.max() + 1e-6, 0.012):
            strip = np.nonzero(np.abs(pv - cv) < PAD_HALF_WIDTH_M)[0]
            if len(strip) < 8:
                continue
            wide = np.nonzero(np.abs(pv - cv) < PALM_HALF_M)[0]
            wide_u, wide_v, wide_z = pu[wide], pv[wide], z_world[wide]
            for gz in levels:
                band = strip[np.abs(z_world[strip] - gz) < PAD_HALF_LENGTH_M]
                if len(band) < 8:
                    continue
                order = np.argsort(pu[band])
                band_u, band_z = pu[band][order], z_world[band][order]
                for cu in np.arange(band_u[0], band_u[-1] + 1e-9, _STEP_M):
                    window = _window(band_u, cu)
                    if window is None:
                        continue
                    # Recenter the fingers on what they enclose, then require the same set.
                    cu = 0.5 * (band_u[window[0]] + band_u[window[1] - 1])
                    window = _window(band_u, cu)
                    width = float(band_u[window[1] - 1] - band_u[window[0]])
                    if width > MAX_WIDTH_M or abs(cu - 0.5 * (band_u[window[0]] + band_u[window[1] - 1])) > 1e-6:
                        continue
                    if not _feasible(wide_u, wide_v, wide_z, cu, cv, gz):
                        continue
                    xy = cu * u_axis + cv * v_axis
                    pads = band_z[window[0] : window[1]]
                    score = (
                        -3.0 * float(np.hypot(*(xy - centroid[:2])))
                        - 2.0 * abs(gz - preferred_z)
                        + 0.5 * min(float(np.ptp(pads)) / (2 * PAD_HALF_LENGTH_M), 1.0)
                        + 0.1 * min((window[1] - window[0]) / 40.0, 1.0)
                        - 0.05 * abs(float(yaw))
                    )
                    center = np.array([xy[0], xy[1], gz - root_z])
                    found.append(Grasp(center, np.array([u_axis[0], u_axis[1], 0.0]), width, score))
    found.sort(key=lambda grasp: -grasp.score)
    return _distinct(found, limit)


def _window(sorted_u: np.ndarray, center: float) -> tuple[int, int] | None:
    first = int(np.searchsorted(sorted_u, center - OPEN_HALF_M, side="left"))
    last = int(np.searchsorted(sorted_u, center + OPEN_HALF_M, side="right"))
    return (first, last) if last - first >= 4 else None


def _feasible(wide_u, wide_v, wide_z, cu: float, cv: float, gz: float) -> bool:
    offset = np.abs(wide_u - cu)
    in_path = (
        (np.abs(wide_v - cv) < PAD_HALF_WIDTH_M + 0.006)
        & (offset >= OPEN_HALF_M)
        & (offset <= FINGER_OUTER_M)
        & (wide_z > gz - PAD_HALF_LENGTH_M)
    )
    if bool(in_path.any()):
        return False
    return not bool(((offset < PALM_HALF_M) & (wide_z > gz + PALM_CLEARANCE_M)).any())


def _distinct(grasps: list[Grasp], limit: int) -> list[Grasp]:
    """Drop a grasp within 2 cm and 20 degrees of a better one."""
    kept: list[Grasp] = []
    for grasp in grasps:
        if all(
            np.linalg.norm(grasp.center_local - other.center_local) > 0.02
            or abs(float(np.dot(grasp.axis_local, other.axis_local))) < math.cos(math.radians(20.0))
            for other in kept
        ):
            kept.append(grasp)
        if len(kept) >= limit:
            break
    return kept


def fallback_grasp(points: np.ndarray, root_z: float) -> Grasp:
    """Pinch the middle of the narrower horizontal side."""
    pts = np.asarray(points, dtype=np.float64)
    low, high = pts.min(axis=0), pts.max(axis=0)
    center = 0.5 * (low + high)
    axis = np.array([1.0, 0.0, 0.0]) if (high[0] - low[0]) < (high[1] - low[1]) else np.array([0.0, 1.0, 0.0])
    height = max(TOOL_Z_MIN_M, root_z + float(center[2]))
    return Grasp(
        center_local=np.array([center[0], center[1], height - root_z]),
        axis_local=axis,
        width_m=float(min(high[0] - low[0], high[1] - low[1])),
        score=-10.0,
    )


def tool_yaw(axis_world: np.ndarray) -> float:
    """Yaw [rad] about +Z for fingers closing along ``axis_world``, in ``(-pi/2, pi/2]``."""
    yaw = math.atan2(-float(axis_world[0]), float(axis_world[1]))
    if yaw > math.pi / 2 + 1e-9:
        yaw -= math.pi
    elif yaw <= -math.pi / 2 + 1e-9:
        yaw += math.pi
    return yaw
