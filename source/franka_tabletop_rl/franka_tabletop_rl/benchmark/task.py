# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One benchmark task: the objects on the table and the success expression."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .spec import TabletopTaskSpec, spec_from_success

# Episode budget written into a spec. The env cfg can run a different length.
TASK_TIME_LIMIT_S = 40.0


@dataclass(frozen=True)
class ObjectAsset:
    """One rigid mesh named by a task file."""

    name: str
    size_m: tuple[float, float, float]
    kind: str
    usd: str


@dataclass
class Placement:
    """One object on the table."""

    slot: str
    asset: ObjectAsset
    position_m: tuple[float, float, float]
    role: str
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale: tuple[float, ...] | None = None
    kinematic: bool = False


@dataclass
class TaskLayout:
    """Published layout of one task."""

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
        """Role to asset name, for instruction text."""
        return {placement.role: placement.asset.name.replace("_", " ") for placement in self.placements}


def asset_from_record(record: dict) -> ObjectAsset:
    """Build an object from a task placement."""
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
    """Parse a ``tasks/<task_id>.json`` file.

    Raises:
        ValueError: The file omits its layout, background, surface, or instruction.
    """
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


def with_jitter(layout: TaskLayout, seed: int, pose_jitter_m: float) -> TaskLayout:
    """Shift a fixed layout in the table plane. The meshes stay the ones in ``layout``."""
    if pose_jitter_m <= 0.0:
        return layout
    rng = np.random.default_rng(int(seed))
    placements = []
    for placement in layout.placements:
        if placement.asset.kind == "anchor":
            placements.append(placement)
            continue
        x, y, z = placement.position_m
        x += float(rng.uniform(-pose_jitter_m, pose_jitter_m))
        y += float(rng.uniform(-pose_jitter_m, pose_jitter_m))
        placements.append(
            Placement(
                slot=placement.slot,
                asset=placement.asset,
                position_m=(x, y, z),
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
    """Task-frame position plus the environment origin.

    ``write_root_pose_to_sim`` takes the simulation world frame. The origin is not zero once
    several envs share a stage.
    """
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
