# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plan one stage that holds tasks with different objects.

A packed stage normally holds copies of one scene. A mixed stage holds one rigid body per
object *position*, and each env fills every position with its own task's mesh. Position ``k``
of an env is the ``k``-th placed object of its scene file, anchors excluded. An env with fewer
objects parks a small kinematic stand-in above the robot at its unused positions. The work
surface is a kinematic body that works the same way, so its pose can differ per env.

Nothing here imports Isaac Sim, so the plan can be built and tested without a GPU.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .bundle import resolve_background, resolve_surface, surface_pose

ENTITY_PREFIX = "obj"
"""Scene name of position ``k`` is ``obj<k>``."""

SURFACE_ENTITY = "surface"
"""Scene name of the work-surface body."""

PARK_POSITION_M = (0.0, 0.0, 3.0)
"""Env-frame position [m] of an unused position. It is above the arm's reach and outside every camera view."""


@dataclass(frozen=True)
class ObjectVariant:
    """What decides the mesh an env puts at a position."""

    usd: str
    scale: tuple[float, ...] | None
    kinematic: bool


@dataclass
class MixedPlan:
    """Which mesh each env puts at each position, and where the work surface sits."""

    entities: tuple[str, ...]
    """Scene names of the object positions, in position order."""

    variants: tuple[tuple[ObjectVariant | None, ...], ...]
    """Per position, the distinct meshes. ``None`` is the parked stand-in."""

    choices: tuple[tuple[int, ...], ...]
    """Per position, the index into ``variants`` that each env uses, shape ``[positions][envs]``."""

    slot_entities: tuple[dict[str, str], ...]
    """Per env, the scene slot of each placed object mapped to the position that holds it."""

    start_poses: tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]
    """Per position, the (position [m], wxyz) the body spawns at. The first env that uses it decides."""

    surfaces: tuple[str, ...]
    """Names of the distinct work surfaces."""

    surface_choices: tuple[int, ...]
    """Index into ``surfaces`` for each env."""

    surface_poses: tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]
    """Per env, the work-surface (position [m], wxyz)."""

    @property
    def num_envs(self) -> int:
        return len(self.slot_entities)

    def parked(self, env: int) -> list[str]:
        """Positions that env ``env`` leaves unused."""
        used = set(self.slot_entities[env].values())
        return [name for name in self.entities if name not in used]

    def choice_by_name(self) -> dict[str, tuple[int, ...]]:
        """Variant index per env for every scene asset that has several variants."""
        choices = dict(zip(self.entities, self.choices, strict=True))
        choices[SURFACE_ENTITY] = self.surface_choices
        return choices


def plan_mixed(payloads: Sequence[dict], stage: dict) -> MixedPlan:
    """Plan a stage for one scene file per env.

    Args:
        payloads: Parsed scene files, one per env, in env order.
        stage: The bundle's ``assets/stage.json``, as read by :func:`read_stage`.

    Raises:
        ValueError: There are no scenes, the scenes name different backgrounds, a scene places
            no object, or a scene places one slot twice.
    """
    if not payloads:
        raise ValueError("a mixed stage needs at least one scene")
    backgrounds = {resolve_background(stage, _choice(item, "background"))["name"] for item in payloads}
    if len(backgrounds) != 1:
        raise ValueError(f"scenes of one stage must share a background, got {sorted(backgrounds)}")
    objects = [_objects(item) for item in payloads]
    count = max(len(row) for row in objects)
    entities = tuple(f"{ENTITY_PREFIX}{index}" for index in range(count))
    variants: list[list[ObjectVariant | None]] = []
    choices: list[list[int]] = []
    starts = []
    for index in range(count):
        column = [row[index][1] if index < len(row) else None for row in objects]
        distinct: list[ObjectVariant | None] = []
        for variant in column:
            if variant is not None and variant not in distinct:
                distinct.append(variant)
        if any(variant is None for variant in column):
            distinct.append(None)
        variants.append(distinct)
        choices.append([distinct.index(variant) for variant in column])
        starts.append(_start_pose(objects, index))
    surface_names: list[str] = []
    surface_choices = []
    surface_poses = []
    for item in payloads:
        record = resolve_surface(stage, _choice(item, "surface"))
        name = str(record["name"])
        if name not in surface_names:
            surface_names.append(name)
        surface_choices.append(surface_names.index(name))
        surface_poses.append(surface_pose(record, item))
    return MixedPlan(
        entities=entities,
        variants=tuple(tuple(column) for column in variants),
        choices=tuple(tuple(column) for column in choices),
        slot_entities=tuple({entry[0]: entities[index] for index, entry in enumerate(row)} for row in objects),
        start_poses=tuple(starts),
        surfaces=tuple(surface_names),
        surface_choices=tuple(surface_choices),
        surface_poses=tuple(surface_poses),
    )


def explicit_worlds(
    asset_names: Sequence[str],
    variant_counts: Sequence[int],
    choices: Mapping[str, Sequence[int]],
    num_envs: int,
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
    """One world per env, in the form the cloner's combination planner returns.

    The cloner's own planner would enumerate every combination of variants, which is the
    product of the variant counts of all positions. A mixed stage needs only the ``num_envs``
    combinations its envs use.

    Args:
        asset_names: Names of the per-env spawned scene assets.
        variant_counts: Number of variants of each of those assets.
        choices: Variant index per env for assets that have several. Others use variant 0.
        num_envs: Number of envs.

    Returns:
        World memberships, as indices into the flat variant library, and equal weights.

    Raises:
        ValueError: An asset's choice is outside its variants, or lists the wrong number of envs.
    """
    for name, picks in choices.items():
        if name in asset_names and len(picks) != num_envs:
            raise ValueError(f"{name} lists {len(picks)} choices for {num_envs} envs")
    offsets = np.cumsum([0, *variant_counts])
    worlds = []
    for env in range(num_envs):
        members = []
        for index, name in enumerate(asset_names):
            pick = int(choices[name][env]) if name in choices else 0
            if not 0 <= pick < variant_counts[index]:
                raise ValueError(f"{name} has {variant_counts[index]} variants, env {env} asked for {pick}")
            members.append(int(offsets[index]) + pick)
        worlds.append(tuple(members))
    return tuple(worlds), np.ones(num_envs, dtype=np.float64)


def _choice(payload: dict, key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _objects(payload: dict) -> list[tuple[str, ObjectVariant, dict]]:
    """Placed objects of one scene in file order: slot, mesh, and pose record."""
    placements = payload.get("placements") or []
    row = []
    seen: set[str] = set()
    for item in placements:
        if str(item.get("kind")) == "anchor":
            continue
        slot = str(item["slot"])
        if slot in seen:
            raise ValueError(f"scene {payload.get('task_id')} places {slot} more than once")
        seen.add(slot)
        scale = item.get("scale")
        variant = ObjectVariant(
            usd=str(item["usd"]),
            scale=tuple(float(value) for value in scale) if scale else None,
            kinematic=bool(item.get("kinematic")),
        )
        row.append((slot, variant, item))
    if not row:
        raise ValueError(f"scene {payload.get('task_id')} places no objects")
    return row


def _start_pose(objects: list[list[tuple[str, ObjectVariant, dict]]], index: int):
    for row in objects:
        if index < len(row):
            item = row[index][2]
            position = tuple(float(value) for value in item["position_m"])
            quat = tuple(float(value) for value in item.get("quat_wxyz", (1.0, 0.0, 0.0, 0.0)))
            return position, quat
    return PARK_POSITION_M, (1.0, 0.0, 0.0, 0.0)
