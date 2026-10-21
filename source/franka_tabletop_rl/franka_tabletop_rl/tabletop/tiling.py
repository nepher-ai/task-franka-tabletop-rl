# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Repeat a mixed-stage plan so that env ``i`` runs scene ``i % T``."""

from __future__ import annotations

from dataclasses import replace

from .scenes.mixed import MixedPlan


def tile_plan(plan: MixedPlan, num_envs: int) -> MixedPlan:
    """Repeat a plan of ``T`` scenes over ``num_envs`` envs.

    Env ``i`` of the result holds the scene of env ``i % T`` of ``plan``.

    Raises:
        ValueError: ``num_envs`` is not a positive multiple of the plan's scene count.
    """
    count = plan.num_envs
    if num_envs <= 0 or num_envs % count:
        raise ValueError(f"num_envs must be a positive multiple of the {count} benchmark tasks, got {num_envs}")
    copies = num_envs // count
    return replace(
        plan,
        choices=tuple(column * copies for column in plan.choices),
        slot_entities=plan.slot_entities * copies,
        surface_choices=plan.surface_choices * copies,
        surface_poses=plan.surface_poses * copies,
    )
