# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage plans for one, three, and five backgrounds."""

from __future__ import annotations

import pytest

from franka_tabletop_rl.benchmark.stage_plan import PARK_POSITION_M, plan_stage
from franka_tabletop_rl.benchmark.task import ObjectAsset, Placement, TaskLayout

_SURFACE = {
    "name": "oak",
    "usd": "assets/oak.usd",
    "pos": [0.5, 0.0, 0.0],
    "rot": [1.0, 0.0, 0.0, 0.0],
}


def _stage(backgrounds: list[str]) -> dict:
    return {
        "backgrounds": [{"name": name} for name in backgrounds],
        "surfaces": [_SURFACE],
        "surface": "oak",
    }


def _layout(task_id: str, background: str, count: int, surface: str = "oak") -> TaskLayout:
    placements = [
        Placement(
            f"item{index}",
            ObjectAsset(f"item{index}", (0.05, 0.05, 0.05), "item", f"assets/{task_id}-{index}.usd"),
            (0.4 + 0.02 * index, 0.0, 0.02),
            "item",
        )
        for index in range(count)
    ]
    return TaskLayout(task_id, "lay", background, surface, f"do {task_id}", placements)


def _loads(plan, layouts) -> list[int]:
    totals = [0] * plan.num_envs
    for row in plan.assignment:
        for env, task in enumerate(row):
            totals[env] += len(layouts[task].placements)
    return totals


def test_one_background_keeps_todays_mapping():
    layouts = [_layout(f"t{index}", "office", 2) for index in range(4)]
    plan = plan_stage(layouts, _stage(["office"]), 8)
    assert plan.phase_count == 1
    assert plan.assignment[0] == tuple(index % 4 for index in range(8))
    assert plan.env_spacing == 2.5
    assert plan.parking[0][0] == PARK_POSITION_M
    assert plan.surface_entities == ("surface",)
    again = plan_stage(layouts, _stage(["office"]), 8)
    assert again.assignment == plan.assignment
    assert again.choices == plan.choices


def test_three_and_five_backgrounds_cover_every_task_and_do_not_overlap_parking():
    for count in (3, 5):
        names = [f"bg{index}" for index in range(count)]
        layouts = []
        for name in names:
            for index in range(3):
                layouts.append(_layout(f"{name}-{index}", name, 2))
        plan = plan_stage(layouts, _stage(names), 4)
        assert plan.phase_count == count
        for phase, name in enumerate(names):
            seen = {layouts[task].task_id for task in plan.assignment[phase]}
            assert seen == {f"{name}-{index}" for index in range(3)}
        spots = plan.parking[0]
        assert len(spots) == len(set(spots))
        loads = _loads(plan, layouts)
        assert max(loads) - min(loads) <= 2
        assert plan_stage(layouts, _stage(names), 4).assignment == plan.assignment


def test_too_few_envs_raises():
    layouts = [_layout(f"t{index}", "office", 1) for index in range(3)]
    with pytest.raises(ValueError, match="at least 3"):
        plan_stage(layouts, _stage(["office"]), 2)
