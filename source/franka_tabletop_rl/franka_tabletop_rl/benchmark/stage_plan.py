# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plan one Isaac Lab stage for every task of a benchmark.

One background is one phase. Env ``i`` runs one task of that phase, and across phases it holds the
union of those tasks' objects. A single background keeps today's layout: env ``i`` runs task
``i % T``, unused positions share one park pose, and nothing is put on a shelf.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .stage import resolve_surface, surface_pose
from .task import Placement, TaskLayout

PARK_POSITION_M = (0.0, 0.0, 3.0)
"""Env-frame position [m] of an unused body when the stage has one background."""

SURFACE_ENTITY = "surface"
"""Scene name of the work-surface body when every env needs only one."""

_PARK_Z_M = 2.5
_PARK_CELL_M = 0.45


@dataclass(frozen=True)
class ObjectVariant:
    """What decides the mesh an env puts at a position."""

    usd: str
    scale: tuple[float, ...] | None
    kinematic: bool


@dataclass
class StagePlan:
    """Bodies of one stage and which task each env runs in each background phase."""

    phases: tuple[str, ...]
    """Background name of each phase, in stage order."""

    task_ids: tuple[str, ...]
    """Selected tasks, in benchmark order."""

    assignment: tuple[tuple[int, ...], ...]
    """``assignment[phase][env]`` is an index into ``task_ids``."""

    entities: tuple[str, ...]
    """Scene names of the object positions, in position order."""

    variants: tuple[tuple[ObjectVariant | None, ...], ...]
    """Per position, the distinct meshes. ``None`` is the parked stand-in."""

    choices: tuple[tuple[int, ...], ...]
    """Per position, the index into ``variants`` that each env uses."""

    phase_slots: tuple[tuple[dict[str, str], ...], ...]
    """``phase_slots[phase][env]`` maps a task slot to the body that holds it."""

    start_poses: tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]
    """Per position, the (position [m], wxyz) the body spawns at."""

    surface_entities: tuple[str, ...]
    """Scene names of the work-surface bodies."""

    surface_variants: tuple[tuple[str, ...], ...]
    """Per surface body, the surface names. The last may be ``""`` for an unused body."""

    surface_choices: tuple[tuple[int, ...], ...]
    """Per surface body, the index into ``surface_variants`` that each env uses."""

    surface_pose_by_phase: tuple[tuple[tuple[tuple[float, ...], tuple[float, ...]], ...], ...]
    """``[phase][env]`` pose of that phase's active surface."""

    active_surface: tuple[tuple[int, ...], ...]
    """``[phase][env]`` index into ``surface_entities``."""

    parking: tuple[tuple[tuple[float, float, float], ...], ...]
    """``parking[env][position]`` is where that env leaves the body when the phase does not use it."""

    env_spacing: float
    """Distance [m] between env origins. One background keeps 2.5."""

    @property
    def num_envs(self) -> int:
        return len(self.assignment[0]) if self.assignment else 0

    @property
    def phase_count(self) -> int:
        return len(self.phases)

    def choice_by_name(self) -> dict[str, tuple[int, ...]]:
        """Variant index per env for every scene asset that has several variants."""
        choices = dict(zip(self.entities, self.choices, strict=True))
        for name, column in zip(self.surface_entities, self.surface_choices, strict=True):
            choices[name] = column
        return choices


def plan_stage(layouts: Sequence[TaskLayout], stage: dict, num_envs: int) -> StagePlan:
    """Plan ``num_envs`` envs for ``layouts``, one phase per background.

    Args:
        layouts: Selected tasks, in benchmark order.
        stage: The bundle's ``assets/stage.json``.
        num_envs: Parallel envs. At least the size of the largest background group.

    Raises:
        ValueError: There are no tasks, ``num_envs`` is too small, or a task places no object.
    """
    if not layouts:
        raise ValueError("a stage needs at least one task")
    if num_envs < 1:
        raise ValueError(f"num_envs must be positive, got {num_envs}")
    phases, groups = _phases(layouts, stage)
    largest = max(len(group) for group in groups)
    if num_envs < largest:
        group = phases[groups.index(max(groups, key=len))]
        raise ValueError(f"num_envs must be at least {largest}, the largest background group ({group})")
    sizes = tuple(_object_count(layout) for layout in layouts)
    assignment = _assign(groups, sizes, num_envs)
    multi = len(phases) > 1
    if multi:
        entities, variants, choices, phase_slots, starts = _pack_unions(layouts, assignment)
        parking, spacing = _parking(variants, choices)
    else:
        entities, variants, choices, phase_slots, starts = _pack_single(layouts, assignment)
        parking = tuple(tuple(PARK_POSITION_M for _ in entities) for _ in range(num_envs))
        spacing = 2.5
    surfaces = _surfaces(layouts, stage, assignment, multi)
    return StagePlan(
        phases=phases,
        task_ids=tuple(layout.task_id for layout in layouts),
        assignment=assignment,
        entities=entities,
        variants=variants,
        choices=choices,
        phase_slots=phase_slots,
        start_poses=starts,
        parking=parking,
        env_spacing=spacing,
        **surfaces,
    )


def explicit_worlds(
    asset_names: Sequence[str],
    variant_counts: Sequence[int],
    choices: Mapping[str, Sequence[int]],
    num_envs: int,
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
    """One world per env, in the form the cloner's combination planner returns.

    The cloner's own planner would enumerate every combination of variants. A stage needs only the
    ``num_envs`` combinations its envs use.

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


def _phases(layouts: Sequence[TaskLayout], stage: dict) -> tuple[tuple[str, ...], list[list[int]]]:
    order = [str(record.get("name")) for record in stage.get("backgrounds", [])]
    names: list[str] = []
    for name in order:
        if any(layout.background == name for layout in layouts) and name not in names:
            names.append(name)
    for layout in layouts:
        if layout.background not in names:
            names.append(layout.background)
    groups = [[index for index, layout in enumerate(layouts) if layout.background == name] for name in names]
    return tuple(names), groups


def _assign(groups: list[list[int]], sizes: Sequence[int], num_envs: int) -> tuple[tuple[int, ...], ...]:
    """Phase 0 keeps benchmark order. Later phases fill the lightest envs first."""
    assignment = []
    load = [0] * num_envs
    for phase, group in enumerate(groups):
        if phase == 0:
            row = [group[index % len(group)] for index in range(num_envs)]
        else:
            row = [None] * num_envs
            ranked = sorted(group, key=lambda task: (-sizes[task], task))
            lightest = sorted(range(num_envs), key=lambda env: (load[env], env))
            for task, env in zip(ranked, lightest, strict=False):
                row[env] = task
            copies = ranked or group
            empty = [env for env in lightest if row[env] is None]
            for index, env in enumerate(empty):
                row[env] = copies[index % len(copies)]
        for env, task in enumerate(row):
            load[env] += sizes[int(task)]
        assignment.append(tuple(int(task) for task in row))
    return tuple(assignment)


def _object_count(layout: TaskLayout) -> int:
    return sum(1 for placement in layout.placements if placement.asset.kind != "anchor")


def _objects(layout: TaskLayout) -> list[tuple[str, ObjectVariant, Placement]]:
    row = []
    seen: set[str] = set()
    for placement in layout.placements:
        if placement.asset.kind == "anchor":
            continue
        if placement.slot in seen:
            raise ValueError(f"task {layout.task_id} places {placement.slot} more than once")
        seen.add(placement.slot)
        row.append((placement.slot, _variant(placement), placement))
    if not row:
        raise ValueError(f"task {layout.task_id} places no objects")
    return row


def _variant(placement: Placement) -> ObjectVariant:
    return ObjectVariant(usd=placement.asset.usd, scale=placement.scale, kinematic=placement.kinematic)


def _pack_single(layouts: Sequence[TaskLayout], assignment: tuple[tuple[int, ...], ...]):
    """One column per object index. Env ``i`` contributes task ``assignment[0][i]``."""
    rows = [_objects(layouts[task]) for task in assignment[0]]
    return _columns(rows, single_phase=True)


def _pack_unions(layouts, assignment):
    """One body per distinct (slot, mesh) an env needs across its phases."""
    rows = []
    owners: list[list[list[tuple[int, str]]]] = []
    for env in range(len(assignment[0])):
        row = []
        used: dict[tuple[str, ObjectVariant], int] = {}
        env_owners: list[list[tuple[int, str]]] = []
        for phase in range(len(assignment)):
            for slot, variant, placement in _objects(layouts[assignment[phase][env]]):
                key = (slot, variant)
                if key not in used:
                    used[key] = len(row)
                    row.append((slot, variant, placement))
                    env_owners.append([])
                env_owners[used[key]].append((phase, slot))
        rows.append(row)
        owners.append(env_owners)
    entities, variants, choices, _slots, starts = _columns(rows, single_phase=False)
    phase_slots = _phase_slots_from_owners(owners, entities, len(assignment), len(assignment[0]))
    return entities, variants, choices, phase_slots, starts


def _columns(rows: list[list[tuple[str, ObjectVariant, Placement]]], *, single_phase: bool):
    count = max(len(row) for row in rows)
    entities = tuple(f"obj{index}" for index in range(count))
    variants: list[list[ObjectVariant | None]] = []
    choices: list[list[int]] = []
    starts = []
    slot_rows: list[dict[str, str]] = []
    for row_index, row in enumerate(rows):
        mapping = {}
        for index, (slot, _variant, _placement) in enumerate(row):
            mapping[slot] = entities[index]
        slot_rows.append(mapping)
        del row_index
    for index in range(count):
        column = [row[index][1] if index < len(row) else None for row in rows]
        distinct: list[ObjectVariant | None] = []
        for variant in column:
            if variant is not None and variant not in distinct:
                distinct.append(variant)
        if any(variant is None for variant in column):
            distinct.append(None)
        variants.append(distinct)
        choices.append([distinct.index(variant) for variant in column])
        starts.append(_start_pose(rows, index))
    if single_phase:
        phase_slots = (tuple(slot_rows),)
    else:
        phase_slots = None
    packed_variants = tuple(tuple(column) for column in variants)
    packed_choices = tuple(tuple(column) for column in choices)
    return entities, packed_variants, packed_choices, phase_slots, tuple(starts)


def _phase_slots_from_owners(owners, entities, phases: int, num_envs: int):
    slots: list[list[dict[str, str]]] = [[ {} for _ in range(num_envs)] for _ in range(phases)]
    for env, env_owners in enumerate(owners):
        for index, links in enumerate(env_owners):
            for phase, slot in links:
                slots[phase][env][slot] = entities[index]
    return tuple(tuple(row) for row in slots)


def _start_pose(rows, index: int):
    for row in rows:
        if index < len(row):
            placement = row[index][2]
            return placement.position_m, placement.quat_wxyz
    return PARK_POSITION_M, (1.0, 0.0, 0.0, 0.0)


def _parking(variants, choices):
    """Non-overlapping spots above the table, and the env spacing that keeps shelves apart."""
    num_envs = len(choices[0]) if choices else 0
    count = len(variants)
    cols = max(1, math.ceil(math.sqrt(count))) if count else 1
    span = cols * _PARK_CELL_M
    spacing = max(2.5, span + 0.4)
    origin = -0.5 * (cols - 1) * _PARK_CELL_M
    spots = []
    for index in range(count):
        row, column = divmod(index, cols)
        spots.append((origin + column * _PARK_CELL_M, origin + row * _PARK_CELL_M, _PARK_Z_M))
    parking = tuple(tuple(spots) for _ in range(num_envs))
    return parking, spacing


def _surfaces(layouts: Sequence[TaskLayout], stage: dict, assignment, multi: bool) -> dict:
    num_envs = len(assignment[0])
    phases = len(assignment)
    per_env: list[list[str]] = []
    pose_by_phase = []
    active = []
    for phase in range(phases):
        poses = []
        which = []
        for env in range(num_envs):
            layout = layouts[assignment[phase][env]]
            record = resolve_surface(stage, layout.surface)
            name = str(record["name"])
            if phase == 0:
                per_env.append([])
            if name not in per_env[env]:
                per_env[env].append(name)
            which.append(per_env[env].index(name))
            poses.append(surface_pose(record, {"surface_pose": layout.surface_pose}))
        pose_by_phase.append(tuple(poses))
        active.append(tuple(which))
    width = max(len(row) for row in per_env)
    if not multi:
        names = []
        choices = []
        for env in range(num_envs):
            name = per_env[env][0]
            if name not in names:
                names.append(name)
            choices.append(names.index(name))
        return {
            "surface_entities": (SURFACE_ENTITY,),
            "surface_variants": (tuple(names),),
            "surface_choices": (tuple(choices),),
            "surface_pose_by_phase": tuple(pose_by_phase),
            "active_surface": tuple(active),
        }
    entities = tuple(SURFACE_ENTITY if width == 1 else f"surface{index}" for index in range(width))
    variants = []
    choices = []
    for index in range(width):
        column = [row[index] if index < len(row) else "" for row in per_env]
        distinct = []
        for name in column:
            if name and name not in distinct:
                distinct.append(name)
        if any(not name for name in column):
            distinct.append("")
        variants.append(tuple(distinct))
        choices.append(tuple(distinct.index(name) for name in column))
    return {
        "surface_entities": entities,
        "surface_variants": tuple(variants),
        "surface_choices": tuple(choices),
        "surface_pose_by_phase": tuple(pose_by_phase),
        "active_surface": tuple(active),
    }
