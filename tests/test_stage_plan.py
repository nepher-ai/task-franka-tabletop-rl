# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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


def test_one_background_keeps_todays_mapping():
    layouts = [_layout(f"t{index}", "office", 2) for index in range(4)]
    plan = plan_stage(layouts, _stage(["office"]), 8)
    assert plan.phases == ("office",)
    assert plan.phase_count == 1
    assert plan.assignment[0] == tuple(index % 4 for index in range(8))
    assert plan.env_spacing == 2.5
    assert plan.parking[0][0] == PARK_POSITION_M
    assert plan.surface_entities == ("surface",)
    again = plan_stage(layouts, _stage(["office"]), 8)
    assert again.assignment == plan.assignment
    assert again.choices == plan.choices


def test_task_backgrounds_do_not_split_the_stage():
    layouts = [
        _layout("a", "office", 2),
        _layout("b", "hall", 1),
        _layout("c", "studio", 3),
    ]
    stage = _stage(["office", "hall", "studio"])
    stage["background"] = "hall"
    plan = plan_stage(layouts, stage, 3)
    assert plan.phases == ("hall",)
    assert plan.assignment[0] == (0, 1, 2)
    assert plan.env_spacing == 2.5
    assert plan.surface_entities == ("surface",)
    wider = plan_stage(layouts, stage, 6)
    assert wider.assignment[0] == (0, 1, 2, 0, 1, 2)


def test_num_envs_must_be_a_multiple_of_the_task_count():
    layouts = [_layout(f"t{index}", "office", 1) for index in range(3)]
    stage = _stage(["office"])
    for count in (0, 2, 4):
        with pytest.raises(ValueError, match="multiple of the task count"):
            plan_stage(layouts, stage, count)
    assert plan_stage(layouts, stage, 6).assignment[0] == (0, 1, 2, 0, 1, 2)
