# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Seeded scene composition. Placements stay on the table and do not overlap."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..predicates import evaluate
from ..spec import TabletopTaskSpec, get_family, spec_from_success
from .bundle import asset_from_record
from .catalog import ObjectAsset


@dataclass
class Placement:
    """One object on the table."""

    slot: str
    asset: ObjectAsset
    position_m: tuple[float, float, float]
    role: str
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass
class SceneInstance:
    """A fully specified table for one episode seed."""

    archetype: str
    task_id: str
    seed: int
    table_z_m: float
    light: float
    background: tuple[float, float, float]
    placements: list[Placement] = field(default_factory=list)
    side: str = "left"
    instruction: str = ""
    spec: TabletopTaskSpec | None = None

    def names(self) -> dict[str, str]:
        """Role to asset name, for instruction templates."""
        return {placement.role: placement.asset.name.replace("_", " ") for placement in self.placements}

    def by_slot(self) -> dict[str, Placement]:
        return {placement.slot: placement for placement in self.placements}


def load_scene(bundle: str | Path, task_id: str, scene_id: str) -> SceneInstance | None:
    """Load the benchmark layout for one scene. Returns None when the file is absent."""
    path = Path(bundle) / "scenes" / f"{task_id}-{scene_id}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    success = payload.get("success")
    if success:
        name_to_slot = {str(item["name"]): str(item["slot"]) for item in payload["placements"]}
        spec = spec_from_success(task_id, success, 40.0, name_to_slot)
    else:
        spec = get_family(task_id)
    placements = [
        Placement(
            slot=item["slot"],
            asset=asset_from_record(item),
            position_m=tuple(float(value) for value in item["position_m"]),
            role=_role(spec, item["slot"]),
            quat_wxyz=_quat(item),
        )
        for item in payload["placements"]
    ]
    return SceneInstance(
        archetype=str(payload.get("archetype", "")),
        task_id=task_id,
        seed=0,
        table_z_m=0.0,
        light=5000.0,
        background=(1.0, 1.0, 1.0),
        placements=placements,
        side=str(payload.get("side", spec.side)),
        instruction=str(payload.get("instruction", "")),
        spec=spec,
    )


def world_position(
    position_m: tuple[float, float, float], origin: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Scene-frame position plus the environment origin.

    ``write_root_pose_to_sim`` takes the simulation world frame. A shard places
    each table on a centered grid, so the origin is not zero.
    """
    x, y, z = (float(value) for value in position_m)
    ox, oy, oz = (float(value) for value in origin)
    return (x + ox, y + oy, z + oz)


def with_jitter(scene: SceneInstance, seed: int, pose_jitter_m: float) -> SceneInstance:
    """Shift a fixed layout in the table plane. The meshes stay the ones in ``scene``."""
    if pose_jitter_m <= 0.0:
        return scene
    rng = np.random.default_rng(int(seed))
    placements = []
    for placement in scene.placements:
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
            )
        )
    return SceneInstance(
        archetype=scene.archetype,
        task_id=scene.task_id,
        seed=int(seed),
        table_z_m=scene.table_z_m,
        light=scene.light,
        background=scene.background,
        placements=placements,
        side=scene.side,
        instruction=scene.instruction,
        spec=scene.spec,
    )


def compose(
    seed: int,
    task_id: str,
    catalog: dict[str, ObjectAsset],
    *,
    table_m: tuple[float, float, float],
    center_xy: tuple[float, float],
    table_z_m: float = 0.0,
    pose_jitter_m: float = 0.0,
    archetype: str = "",
    light: float = 5000.0,
) -> SceneInstance:
    """Place the family's objects using the benchmark's catalog and table size.

    ``pose_jitter_m`` shifts each object in the table plane [m]. Draws are rejected while the
    geometric success condition is already true, so an episode cannot finish before the robot moves.
    """
    spec = get_family(task_id)
    rng = np.random.default_rng(int(seed))
    scene = _draw(
        rng, spec, archetype, task_id, int(seed), catalog, table_m, center_xy, table_z_m, light, pose_jitter_m
    )
    for _ in range(31):
        if not _geometry_solved(scene, spec):
            return scene
        scene = _draw(
            rng, spec, archetype, task_id, int(seed), catalog, table_m, center_xy, table_z_m, light, pose_jitter_m
        )
    return scene


def _draw(
    rng,
    spec,
    archetype: str,
    task_id: str,
    seed: int,
    catalog: dict[str, ObjectAsset],
    table_m: tuple[float, float, float],
    center_xy: tuple[float, float],
    table_z_m: float,
    light: float,
    pose_jitter_m: float,
) -> SceneInstance:
    side = spec.side
    if task_id == "place_relative":
        side = ("left", "right", "front", "behind")[int(rng.integers(0, 4))]
    slots = list(dict.fromkeys(spec.roles.values()))
    chosen = _choose_assets(rng, slots, spec, catalog)
    table_x, table_y, _table_h = table_m
    positions = _place(rng, chosen, table_x, table_y, center_xy, table_z_m, pose_jitter_m)
    placements = [
        Placement(
            slot=slot,
            asset=chosen[slot],
            position_m=tuple(float(v) for v in positions[slot]),
            role=_role(spec, slot),
        )
        for slot in slots
    ]
    return SceneInstance(
        archetype=archetype,
        task_id=task_id,
        seed=seed,
        table_z_m=float(table_z_m),
        light=float(light),
        background=(1.0, 1.0, 1.0),
        placements=placements,
        side=side,
    )


def _geometry_solved(scene: SceneInstance, spec) -> bool:
    poses = {}
    for placement in scene.placements:
        poses[placement.slot] = {
            "pos": np.asarray([placement.position_m], dtype=np.float64),
            "quat": np.array([[1.0, 0.0, 0.0, 0.0]]),
            "half": np.asarray(placement.asset.size_m, dtype=np.float64) * 0.5,
        }
    return bool(np.asarray(evaluate(spec, poses, side=scene.side)).reshape(-1)[0])


def _quat(record: dict) -> tuple[float, float, float, float]:
    raw = record.get("quat_wxyz")
    if not raw:
        return (1.0, 0.0, 0.0, 0.0)
    quat = tuple(float(value) for value in raw)
    if len(quat) != 4:
        raise ValueError(f"quat_wxyz for {record.get('name')} must have 4 values")
    return quat


def _role(spec, slot: str) -> str:
    for role, named in spec.roles.items():
        if named == slot:
            return role
    return slot


def _choose_assets(
    rng: np.random.Generator, slots: list[str], spec, catalog: dict[str, ObjectAsset]
) -> dict[str, ObjectAsset]:
    items = [asset for asset in catalog.values() if asset.kind == "item"]
    containers = [asset for asset in catalog.values() if asset.kind == "container"]
    references = [asset for asset in catalog.values() if asset.kind == "reference"]
    if not items:
        raise ValueError("benchmark catalog has no objects of kind item")
    rng.shuffle(items)
    chosen = {}
    item_index = 0
    for slot in slots:
        role = _role(spec, slot)
        if slot == "container" or role == "container":
            if not containers:
                raise ValueError("benchmark catalog has no objects of kind container")
            chosen[slot] = containers[int(rng.integers(0, len(containers)))]
        elif slot == "reference" or role == "reference":
            if not references:
                raise ValueError("benchmark catalog has no objects of kind reference")
            chosen[slot] = references[0]
        else:
            chosen[slot] = items[item_index % len(items)]
            item_index += 1
    return chosen


def _place(rng, chosen, table_x, table_y, center_xy, table_z, jitter):
    positions = {}
    occupied: list[tuple[float, float, float]] = []
    margin = 0.08
    center_x, center_y = center_xy
    for slot, asset in chosen.items():
        radius = 0.5 * max(asset.size_m[0], asset.size_m[1])
        # Keep the sample in front of the robot base at the env origin.
        x_lo = max(0.22, center_x - table_x / 2 + 0.12)
        x_hi = center_x + table_x / 2 - 0.12
        y_lo = center_y - table_y / 2 + 0.12
        y_hi = center_y + table_y / 2 - 0.12
        for _attempt in range(80):
            x = float(rng.uniform(x_lo, x_hi))
            y = float(rng.uniform(y_lo, y_hi))
            if jitter:
                x += float(rng.uniform(-jitter, jitter))
                y += float(rng.uniform(-jitter, jitter))
            if all((x - ox) ** 2 + (y - oy) ** 2 >= (radius + orad + margin) ** 2 for ox, oy, orad in occupied):
                z = table_z + 0.5 * asset.size_m[2]
                positions[slot] = (x, y, z)
                occupied.append((x, y, radius))
                break
        else:
            raise RuntimeError(f"could not place {slot} without overlap")
    return positions
