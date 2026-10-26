# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest

from franka_tabletop_rl.benchmark.stage_plan import (
    IDENTITY_WXYZ,
    PARK_POSITION_M,
    entity_pose,
    plan_stage,
    surface_body_pose,
)
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


def test_entity_pose_is_the_envs_own_layout_not_env_zeros():
    first = _layout("a", "office", 3)
    second = _layout("b", "office", 1)
    second.placements[0].position_m = (0.9, 0.1, 0.02)
    plan = plan_stage([first, second], _stage(["office"]), 2)
    env_layouts = [first, second]
    for column in range(3):
        assert entity_pose(plan, env_layouts, 0, column) == (first.placements[column].position_m, IDENTITY_WXYZ)
    assert entity_pose(plan, env_layouts, 1, 0) == ((0.9, 0.1, 0.02), IDENTITY_WXYZ)
    # Columns the second task leaves empty rest at its parking pose.
    for column in (1, 2):
        assert entity_pose(plan, env_layouts, 1, column) == (PARK_POSITION_M, IDENTITY_WXYZ)


def test_surface_body_pose_is_the_envs_own_surface():
    layouts = [_layout("a", "office", 1), _layout("b", "office", 1)]
    layouts[1].surface_pose = {"pos": [0.6, 0.2, 0.0], "rot": [1.0, 0.0, 0.0, 0.0]}
    plan = plan_stage(layouts, _stage(["office"]), 2)
    assert surface_body_pose(plan, 0, 0) == plan.surface_pose_by_phase[0][0]
    assert surface_body_pose(plan, 1, 0) == plan.surface_pose_by_phase[0][1]
    assert surface_body_pose(plan, 0, 0) != surface_body_pose(plan, 1, 0)
    assert surface_body_pose(plan, 1, 1) == (PARK_POSITION_M, IDENTITY_WXYZ)


def test_num_envs_must_be_a_multiple_of_the_task_count():
    layouts = [_layout(f"t{index}", "office", 1) for index in range(3)]
    stage = _stage(["office"])
    for count in (0, 2, 4):
        with pytest.raises(ValueError, match="multiple of the task count"):
            plan_stage(layouts, stage, count)
    assert plan_stage(layouts, stage, 6).assignment[0] == (0, 1, 2, 0, 1, 2)
