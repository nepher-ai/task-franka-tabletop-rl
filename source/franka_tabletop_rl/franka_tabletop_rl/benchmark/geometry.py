# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Object geometry for the success checks, built the way the reference benchmark builds it. Lengths are [m].

Each asset keeps, in its root frame, the box around its mesh, the center of that box, the vertex
mean of its convex hull, and the hull face planes ``n . x + d <= 0``. The open-top planes drop the
faces that look up, so a container reaches up without bound.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

# Hull faces with an outward normal z of at least this much form the rim cap of an open-top container.
OPEN_TOP_THRESHOLD = 0.7

SURFACE_SLOT = "surface"
"""Reserved slot of the work surface. A task slot of this name is not allowed."""

_CORNER_SIGNS = np.array([[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])


@dataclass(frozen=True)
class AssetGeometry:
    """Root-frame geometry of one asset.

    Attributes:
        box_corners: Box corners [m], shape ``[8, 3]``.
        box_center: Center of the box [m], shape ``[3]``.
        hull_center: Mean of the convex hull vertices [m], shape ``[3]``. A containment test uses it.
        planes_full: Closed hull planes ``(nx, ny, nz, d)``, shape ``[F, 4]``.
        planes_open_top: Planes of the hull without the upward faces, shape ``[F_kept, 4]``.
        from_mesh: False when the geometry is the ``size_m`` box and not a mesh.
    """

    box_corners: np.ndarray
    box_center: np.ndarray
    hull_center: np.ndarray
    planes_full: np.ndarray
    planes_open_top: np.ndarray
    from_mesh: bool = True


def open_top_planes(planes: np.ndarray, axis: int = 2, threshold: float = OPEN_TOP_THRESHOLD) -> np.ndarray:
    """Planes whose outward normal has an ``axis`` component below ``threshold``."""
    return planes[planes[:, axis] < threshold]


def from_points(points: np.ndarray) -> AssetGeometry:
    """Geometry of mesh points [m] in the root frame.

    Raises:
        scipy.spatial.QhullError: The points do not span a volume.
    """
    from scipy.spatial import ConvexHull

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    low, high = points.min(axis=0), points.max(axis=0)
    hull = ConvexHull(points)
    planes = np.asarray(hull.equations, dtype=np.float64)
    return AssetGeometry(
        box_corners=_box_corners(low, high),
        box_center=(low + high) / 2.0,
        hull_center=points[hull.vertices].mean(axis=0),
        planes_full=planes,
        planes_open_top=open_top_planes(planes),
    )


def from_box(size_m, warn: str | None = None) -> AssetGeometry:
    """Geometry of the ``size_m`` box centered on the root. ``warn`` names the asset that lacks a mesh."""
    if warn is not None:
        warnings.warn(f"{warn} has no mesh; success uses its size_m box", stacklevel=2)
    half = np.asarray(size_m, dtype=np.float64).reshape(3) / 2.0
    eye = np.eye(3)
    planes = np.concatenate(
        [np.concatenate([eye, -half[:, None]], axis=1), np.concatenate([-eye, -half[:, None]], axis=1)]
    )
    return AssetGeometry(
        box_corners=_box_corners(-half, half),
        box_center=np.zeros(3),
        hull_center=np.zeros(3),
        planes_full=planes,
        planes_open_top=open_top_planes(planes),
        from_mesh=False,
    )


def _box_corners(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return (low + high) / 2.0 + _CORNER_SIGNS * (high - low) / 2.0


def read_local_points(stage, prim_path: str) -> np.ndarray:
    """Mesh points [m] of a prim in its own rotated frame, at world scale.

    Default-purpose geometry only, including instance proxies. A mesh gives its points. An implicit
    shape (cube, cylinder, sphere, ...) gives the corners of its extent, so a work surface built from
    cubes keeps its real top. Undoing the prim's rotation and translation but not its scale keeps
    meters, whatever units the asset was authored in.

    Raises:
        ValueError: No prim or no geometry at ``prim_path``.
    """
    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"no prim at {prim_path}")
    time = Usd.TimeCode.Default()
    cache = UsdGeom.XformCache(time)
    to_local = cache.GetLocalToWorldTransform(prim).RemoveScaleShear().GetInverse()
    chunks = []
    for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not child.IsA(UsdGeom.Gprim):
            continue
        if UsdGeom.Gprim(child).GetPurposeAttr().Get() not in (None, UsdGeom.Tokens.default_):
            continue
        if child.IsA(UsdGeom.Mesh):
            points = UsdGeom.Mesh(child).GetPointsAttr().Get()
        else:
            extent = UsdGeom.Boundable.ComputeExtentFromPlugins(UsdGeom.Boundable(child), time)
            points = None if extent is None else [Gf.Vec3f(*corner) for corner in _box_corners(*np.asarray(extent))]
        if points is None or len(points) == 0:
            continue
        to_world = cache.GetLocalToWorldTransform(child)
        chunks.append([list(to_local.Transform(Gf.Vec3d(to_world.Transform(point)))) for point in points])
    if not chunks:
        raise ValueError(f"no geometry under {prim_path}")
    return np.concatenate([np.asarray(chunk, dtype=np.float64) for chunk in chunks])


def rotation(quat_wxyz) -> np.ndarray:
    """Body-to-world rotation matrix of a wxyz quaternion."""
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1)[:4]
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def to_world(pos, quat_wxyz, local: np.ndarray) -> np.ndarray:
    """Root-frame points [m] in the env frame. ``local`` has shape ``[..., 3]``."""
    origin = np.asarray(pos, dtype=np.float64).reshape(3)
    return np.asarray(local, dtype=np.float64) @ rotation(quat_wxyz).T + origin


def world_box_corners(geometry: AssetGeometry, pos, quat_wxyz) -> np.ndarray:
    """Box corners [m] in the env frame, shape ``[8, 3]``."""
    return to_world(pos, quat_wxyz, geometry.box_corners)


def world_box_center(geometry: AssetGeometry, pos, quat_wxyz) -> np.ndarray:
    """Box center [m] in the env frame."""
    return to_world(pos, quat_wxyz, geometry.box_center)


def hull_center_in(item: AssetGeometry, item_pos, item_quat, container_pos, container_quat, planes: np.ndarray) -> bool:
    """Whether the item's hull center is inside the polytope ``planes`` of the container."""
    point = to_world(item_pos, item_quat, item.hull_center)
    local = rotation(container_quat).T @ (point - np.asarray(container_pos, dtype=np.float64).reshape(3))
    return bool(np.max(planes[:, :3] @ local + planes[:, 3]) <= 0.0)
