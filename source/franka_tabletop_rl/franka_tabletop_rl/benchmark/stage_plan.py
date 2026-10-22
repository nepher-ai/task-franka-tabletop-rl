# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plan one Isaac Lab stage for every task of a benchmark.

A benchmark has one background, taken from its stage. Another benchmark can name a different one.
Env ``i`` runs task ``i % T``. ``num_envs`` is a multiple of the task count, so every task runs
in the same step and extra envs are whole extra copies. Unused object positions share one park pose.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .stage import resolve_surface, surface_pose
from .task import Placement, TaskLayout

PARK_POSITION_M = (0.0, 0.0, 3.0)
"""Env-frame position [m] of an unused object position."""

SURFACE_ENTITY = "surface"
"""Scene name of the work-surface body."""


@dataclass(frozen=True)
class ObjectVariant:
    """What decides the mesh an env puts at a position."""

    usd: str
    scale: tuple[float, ...] | None
    kinematic: bool


@dataclass
class StagePlan:
    """Bodies of one stage and which task each env runs."""

    phases: tuple[str, ...]
    """The benchmark background, as a one-element tuple."""

    task_ids: tuple[str, ...]
    """Selected tasks, in benchmark order."""

    assignment: tuple[tuple[int, ...], ...]
    """``assignment[0][env]`` is an index into ``task_ids``. Env ``i`` runs task ``i % T``.

    ``num_envs`` is a multiple of the task count, so each task is assigned equally often.
    """

    entities: tuple[str, ...]
    """Scene names of the object positions, in position order."""

    variants: tuple[tuple[ObjectVariant | None, ...], ...]
    """Per position, the distinct meshes. ``None`` is the parked stand-in."""

    choices: tuple[tuple[int, ...], ...]
    """Per position, the index into ``variants`` that each env uses."""

    phase_slots: tuple[tuple[dict[str, str], ...], ...]
    """``phase_slots[0][env]`` maps a task slot to the body that holds it."""

    start_poses: tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]
    """Per position, the (position [m], wxyz) the body spawns at."""

    surface_entities: tuple[str, ...]
    """Scene names of the work-surface bodies."""

    surface_variants: tuple[tuple[str, ...], ...]
    """Per surface body, the surface names. The last may be ``""`` for an unused body."""

    surface_choices: tuple[tuple[int, ...], ...]
    """Per surface body, the index into ``surface_variants`` that each env uses."""

    surface_pose_by_phase: tuple[tuple[tuple[tuple[float, ...], tuple[float, ...]], ...], ...]
    """``[0][env]`` pose of that env's work surface."""

    active_surface: tuple[tuple[int, ...], ...]
    """``[0][env]`` index into ``surface_entities``."""

    parking: tuple[tuple[tuple[float, float, float], ...], ...]
    """``parking[env][position]`` is where an unused object position rests."""

    env_spacing: float
    """Distance [m] between env origins."""

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
    """Plan ``num_envs`` envs for ``layouts``.

    Env ``i`` runs task ``i % T``. The dome is the stage background, the same one for every task.
    A different benchmark names its own background in its stage. ``num_envs`` is ``T``, ``2T``, ``3T``,
    and so on, so the stage holds every selected task at once.

    Args:
        layouts: Selected tasks, in benchmark order.
        stage: The bundle's ``assets/stage.json``.
        num_envs: Parallel envs. A multiple of the number of tasks, and at least that number.

    Raises:
        ValueError: There are no tasks, ``num_envs`` is not a multiple of the task count, or a task
            places no object.
    """
    if not layouts:
        raise ValueError("a stage needs at least one task")
    task_count = len(layouts)
    if num_envs < task_count or num_envs % task_count != 0:
        raise ValueError(
            f"num_envs must be a multiple of the task count ({task_count}) and at least that large, got {num_envs}"
        )
    assignment = (tuple(index % len(layouts) for index in range(num_envs)),)
    entities, variants, choices, phase_slots, starts = _pack_single(layouts, assignment)
    parking = tuple(tuple(PARK_POSITION_M for _ in entities) for _ in range(num_envs))
    return StagePlan(
        phases=(_background_name(stage),),
        task_ids=tuple(layout.task_id for layout in layouts),
        assignment=assignment,
        entities=entities,
        variants=variants,
        choices=choices,
        phase_slots=phase_slots,
        start_poses=starts,
        parking=parking,
        env_spacing=2.5,
        **_surfaces(layouts, stage, assignment),
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


def _background_name(stage: dict) -> str:
    """The one dome this benchmark uses."""
    name = stage.get("background")
    if isinstance(name, str) and name:
        return name
    records = stage.get("backgrounds") or []
    if records and records[0].get("name"):
        return str(records[0]["name"])
    raise ValueError("assets/stage.json names no background")


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
    return _columns(rows)


def _columns(rows: list[list[tuple[str, ObjectVariant, Placement]]]):
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
    packed_variants = tuple(tuple(column) for column in variants)
    packed_choices = tuple(tuple(column) for column in choices)
    return entities, packed_variants, packed_choices, (tuple(slot_rows),), tuple(starts)


def _start_pose(rows, index: int):
    for row in rows:
        if index < len(row):
            placement = row[index][2]
            return placement.position_m, placement.quat_wxyz
    return PARK_POSITION_M, (1.0, 0.0, 0.0, 0.0)


def _surfaces(layouts: Sequence[TaskLayout], stage: dict, assignment) -> dict:
    """One work-surface body. Each env picks the mesh its task names."""
    num_envs = len(assignment[0])
    names: list[str] = []
    choices = []
    poses = []
    for env in range(num_envs):
        layout = layouts[assignment[0][env]]
        record = resolve_surface(stage, layout.surface)
        name = str(record["name"])
        if name not in names:
            names.append(name)
        choices.append(names.index(name))
        poses.append(surface_pose(record, {"surface_pose": layout.surface_pose}))
    return {
        "surface_entities": (SURFACE_ENTITY,),
        "surface_variants": (tuple(names),),
        "surface_choices": (tuple(choices),),
        "surface_pose_by_phase": (tuple(poses),),
        "active_surface": (tuple(0 for _ in range(num_envs)),),
    }
