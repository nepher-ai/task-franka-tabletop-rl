# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One benchmark task: objects on the table and the success expression."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .spec import TabletopTaskSpec, spec_from_success

# Episode budget written into a spec. The env cfg can run a different length.
TASK_TIME_LIMIT_S = 40.0


@dataclass(frozen=True)
class ObjectAsset:
    """One rigid mesh from a task file."""

    name: str
    size_m: tuple[float, float, float]
    kind: str
    usd: str


@dataclass
class Placement:
    slot: str
    asset: ObjectAsset
    position_m: tuple[float, float, float]
    role: str
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale: tuple[float, ...] | None = None
    kinematic: bool = False


@dataclass
class TaskLayout:
    task_id: str
    layout: str
    background: str
    surface: str
    instruction: str
    placements: list[Placement] = field(default_factory=list)
    side: str = "left"
    surface_pose: dict | None = None
    spec: TabletopTaskSpec | None = None

    def by_slot(self) -> dict[str, Placement]:
        return {placement.slot: placement for placement in self.placements}

    def names(self) -> dict[str, str]:
        return {placement.role: placement.asset.name.replace("_", " ") for placement in self.placements}


def asset_from_record(record: dict) -> ObjectAsset:
    try:
        name = str(record["name"])
        usd = str(record["usd"])
        size = tuple(float(value) for value in record["size_m"])
    except KeyError as exc:
        raise KeyError(f"placement is missing {exc.args[0]}") from exc
    if len(size) != 3:
        raise ValueError(f"size_m for {name} must have 3 values")
    return ObjectAsset(name=name, size_m=size, kind=str(record.get("kind", "item")), usd=usd)


def layout_from_payload(payload: dict, *, time_limit_s: float = TASK_TIME_LIMIT_S) -> TaskLayout:
    """Parse a ``tasks/<task_id>.json`` file."""
    task_id = str(payload["task_id"])
    for key in ("layout", "background", "surface", "instruction"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"task {task_id} is missing {key}")
    success = payload.get("success")
    name_to_slot = {str(item["name"]): str(item["slot"]) for item in payload["placements"]}
    spec = spec_from_success(task_id, success, time_limit_s, name_to_slot) if success else None
    placements = [
        Placement(
            slot=str(item["slot"]),
            asset=asset_from_record(item),
            position_m=tuple(float(value) for value in item["position_m"]),
            role=_role(spec, str(item["slot"])),
            quat_wxyz=_quat(item),
            scale=tuple(float(value) for value in item["scale"]) if item.get("scale") else None,
            kinematic=bool(item.get("kinematic")),
        )
        for item in payload["placements"]
    ]
    pose = payload.get("surface_pose")
    return TaskLayout(
        task_id=task_id,
        layout=str(payload["layout"]),
        background=str(payload["background"]),
        surface=str(payload["surface"]),
        instruction=str(payload["instruction"]),
        placements=placements,
        side=str(payload.get("side", spec.side if spec is not None else "left")),
        surface_pose=pose if isinstance(pose, dict) else None,
        spec=spec,
    )


ShiftLimits = tuple[tuple[float, float], tuple[float, float]]
"""Allowed ``(dx, dy)`` as ``((x_low, x_high), (y_low, y_high))`` [m]."""


def footprint_half_extents(size_m: Sequence[float], quat_wxyz: Sequence[float]) -> tuple[float, float]:
    """Half extents [m] along x and y of the world-aligned box around a rotated box of size ``size_m``."""
    w, x, y, z = (float(value) for value in quat_wxyz)
    norm = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        ]
    )
    half = np.abs(rotation) @ (np.asarray(size_m, dtype=np.float64) / 2.0)
    return float(half[0]), float(half[1])


def shift_limits(
    layout: TaskLayout, center_xy: Sequence[float], half_xy: Sequence[float], margin_m: float
) -> ShiftLimits:
    """Shifts [m] that keep the objects on the work surface at ``margin_m`` from its edge.

    ``center_xy`` and ``half_xy`` are the surface center and half sizes [m] in the frame of the placements.
    Only objects whose center is on the surface limit the range. Racks and shelves beside the table move with
    the group. The range always contains 0: an object that already overhangs the margin cannot move further
    out, and does not block a move inward. ``sdg-gen`` applies the same rule to the bundles it derives.
    """
    lows = [-np.inf, -np.inf]
    highs = [np.inf, np.inf]
    for placement in layout.placements:
        if placement.asset.kind == "anchor":
            continue
        position = [float(value) for value in placement.position_m[:2]]
        if any(abs(position[axis] - center_xy[axis]) > half_xy[axis] for axis in range(2)):
            continue
        half = footprint_half_extents(placement.asset.size_m, placement.quat_wxyz)
        for axis in range(2):
            low = center_xy[axis] - half_xy[axis] + margin_m - (position[axis] - half[axis])
            high = center_xy[axis] + half_xy[axis] - margin_m - (position[axis] + half[axis])
            lows[axis] = max(lows[axis], min(low, 0.0))
            highs[axis] = min(highs[axis], max(high, 0.0))
    return (lows[0], highs[0]), (lows[1], highs[1])


def _draw_shift(rng: np.random.Generator, span: float, limit: tuple[float, float] | None) -> float:
    if span <= 0.0:
        return 0.0
    low, high = (-span, span) if limit is None else (max(-span, limit[0]), min(span, limit[1]))
    return float(rng.uniform(low, high))


def with_jitter(
    layout: TaskLayout,
    seed: int,
    jitter_x: float,
    jitter_y: float | None = None,
    limits: ShiftLimits | None = None,
) -> TaskLayout:
    """Add one shared table-plane offset to every object. Meshes, height, and rotation stay put.

    ``jitter_x`` and ``jitter_y`` are half-ranges [m]. One uniform draw on each axis is added to
    every non-anchor placement, so object spacing is unchanged. ``jitter_y`` defaults to ``jitter_x``.
    ``limits`` narrows the draw so that the objects stay on the table, see ``shift_limits``. Anchors and
    ``surface_pose`` stay as authored. The robot and the tables are not placements.
    """
    span_x = float(jitter_x)
    span_y = span_x if jitter_y is None else float(jitter_y)
    if span_x <= 0.0 and span_y <= 0.0:
        return layout
    rng = np.random.default_rng(int(seed))
    dx = _draw_shift(rng, span_x, None if limits is None else limits[0])
    dy = _draw_shift(rng, span_y, None if limits is None else limits[1])
    placements = []
    for placement in layout.placements:
        if placement.asset.kind == "anchor":
            placements.append(placement)
            continue
        x, y, z = placement.position_m
        placements.append(
            Placement(
                slot=placement.slot,
                asset=placement.asset,
                position_m=(x + dx, y + dy, z),
                role=placement.role,
                quat_wxyz=placement.quat_wxyz,
                scale=placement.scale,
                kinematic=placement.kinematic,
            )
        )
    return TaskLayout(
        task_id=layout.task_id,
        layout=layout.layout,
        background=layout.background,
        surface=layout.surface,
        instruction=layout.instruction,
        placements=placements,
        side=layout.side,
        surface_pose=layout.surface_pose,
        spec=layout.spec,
    )


def world_position(
    position_m: tuple[float, float, float], origin: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Task-frame position [m] plus the env origin, in the simulation frame."""
    x, y, z = (float(value) for value in position_m)
    ox, oy, oz = (float(value) for value in origin)
    return (x + ox, y + oy, z + oz)


def _quat(record: dict) -> tuple[float, float, float, float]:
    raw = record.get("quat_wxyz")
    if not raw:
        return (1.0, 0.0, 0.0, 0.0)
    quat = tuple(float(value) for value in raw)
    if len(quat) != 4:
        raise ValueError(f"quat_wxyz for {record.get('name')} must have 4 values")
    return quat


def _role(spec: TabletopTaskSpec | None, slot: str) -> str:
    if spec is None:
        return slot
    for role, named in spec.roles.items():
        if named == slot:
            return role
    return slot
