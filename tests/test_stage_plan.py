# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest

from franka_tabletop_rl.benchmark.stage import table_rect
from franka_tabletop_rl.benchmark.stage_plan import (
    IDENTITY_WXYZ,
    PARK_POSITION_M,
    entity_pose,
    plan_stage,
    surface_body_pose,
)
from franka_tabletop_rl.benchmark.task import ObjectAsset, Placement, TaskLayout, shift_limits, with_jitter
from franka_tabletop_rl.tasks.hl_policy.layout_sampler import LayoutSampler

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


def test_jitter_is_one_shift_for_every_object():
    anchor = Placement(
        "table_anchor",
        ObjectAsset("table", (0.7, 1.0, 0.03), "anchor", "assets/fixtures/table_oak.usd"),
        (0.55, 0.0, 0.0),
        "anchor",
    )
    mug = Placement("mug", ObjectAsset("mug", (0.08, 0.08, 0.1), "item", "assets/mug.usd"), (0.4, 0.1, 0.02), "item")
    bowl = Placement(
        "bowl",
        ObjectAsset("bowl", (0.12, 0.12, 0.05), "container", "assets/bowl.usd"),
        (0.62, -0.2, 0.02),
        "item",
    )
    layout = TaskLayout(
        "t", "lay", "office", "oak", "move them", [anchor, mug, bowl], surface_pose={"pos": [0.5, 0.0, 0.0]}
    )
    shifted = with_jitter(layout, 7, 0.05, 0.02)
    dx = shifted.placements[1].position_m[0] - mug.position_m[0]
    dy = shifted.placements[1].position_m[1] - mug.position_m[1]
    assert shifted.placements[0].position_m == anchor.position_m
    assert shifted.placements[2].position_m == (bowl.position_m[0] + dx, bowl.position_m[1] + dy, bowl.position_m[2])
    assert abs(dx) <= 0.05 and abs(dy) <= 0.02
    assert shifted.placements[1].position_m[2] == mug.position_m[2]
    assert shifted.placements[1].quat_wxyz == mug.quat_wxyz
    assert shifted.surface_pose == layout.surface_pose
    assert with_jitter(layout, 7, 0.05, 0.02).placements[1].position_m == shifted.placements[1].position_m


def test_zero_jitter_leaves_the_layout():
    layout = _layout("t", "office", 2)
    assert with_jitter(layout, 1, 0.0, 0.0) is layout


def test_a_zero_axis_stays_put_and_one_range_covers_both_axes():
    layout = _layout("t", "office", 2)
    shifted = with_jitter(layout, 3, 0.0, 0.04)
    dy = shifted.placements[0].position_m[1] - layout.placements[0].position_m[1]
    for before, after in zip(layout.placements, shifted.placements, strict=True):
        assert after.position_m[0] == before.position_m[0]
        assert after.position_m[2] == before.position_m[2]
        assert after.position_m[1] - before.position_m[1] == dy
    assert dy != 0.0
    both = with_jitter(layout, 3, 0.04)
    shared_x = both.placements[1].position_m[0] - layout.placements[1].position_m[0]
    assert both.placements[0].position_m[0] - layout.placements[0].position_m[0] == shared_x
    assert both.placements[0].position_m[1] - layout.placements[0].position_m[1] == (
        both.placements[1].position_m[1] - layout.placements[1].position_m[1]
    )


def _near_edge_layout() -> TaskLayout:
    # Oak surface: center (0.5, 0.0), half sizes (0.35, 0.5). Objects keep a 0.03 m margin.
    cube = Placement("cube", ObjectAsset("cube", (0.05, 0.05, 0.05), "item", "cube.usd"), (0.7, 0.0, 0.02), "item")
    bar = Placement(
        "bar",
        ObjectAsset("bar", (0.2, 0.02, 0.02), "item", "bar.usd"),
        (0.5, 0.3, 0.02),
        "item",
        quat_wxyz=(0.7071068, 0.0, 0.0, 0.7071068),
    )
    return TaskLayout("edge", "lay", "office", "oak", "do edge", [cube, bar])


def test_shift_limits_follow_the_rotated_footprint():
    (x_low, x_high), (y_low, y_high) = shift_limits(_near_edge_layout(), (0.5, 0.0), (0.35, 0.5), 0.03)
    # The cube reaches x = 0.725 and the surface ends at 0.85 - 0.03. The bar is across x (0.49 to 0.51).
    assert x_high == pytest.approx(0.82 - 0.725)
    assert x_low == pytest.approx(0.18 - 0.49)
    # The bar is turned by 90 degrees: its 0.2 m side points along y and reaches y = 0.4.
    assert y_high == pytest.approx(0.47 - 0.4)
    assert y_low == pytest.approx(-0.47 + 0.025)


def test_a_jittered_layout_stays_on_the_surface():
    layout = _near_edge_layout()
    limits = shift_limits(layout, (0.5, 0.0), (0.35, 0.5), 0.03)
    for seed in range(200):
        shifted = with_jitter(layout, seed, 0.5, 0.5, limits)
        dx = shifted.placements[0].position_m[0] - layout.placements[0].position_m[0]
        dy = shifted.placements[0].position_m[1] - layout.placements[0].position_m[1]
        assert limits[0][0] - 1e-9 <= dx <= limits[0][1] + 1e-9
        assert limits[1][0] - 1e-9 <= dy <= limits[1][1] + 1e-9
        assert shifted.placements[1].position_m[0] - layout.placements[1].position_m[0] == pytest.approx(dx)
    unlimited = [with_jitter(layout, seed, 0.5, 0.5).placements[0].position_m[0] for seed in range(200)]
    assert max(unlimited) > 0.7 + limits[0][1], "without limits the draw does leave the surface"


def test_an_object_that_overhangs_the_margin_can_only_move_inward():
    layout = _near_edge_layout()
    layout.placements[0].position_m = (0.84, 0.0, 0.02)  # its center is on the table, its edge is past the margin
    limits = shift_limits(layout, (0.5, 0.0), (0.35, 0.5), 0.03)
    assert limits[0][1] == 0.0 and limits[0][0] < 0.0
    shifts = [with_jitter(layout, seed, 0.1, 0.1, limits).placements[0].position_m for seed in range(30)]
    assert all(x <= 0.84 for x, _, _ in shifts) and any(x < 0.84 for x, _, _ in shifts)


def test_objects_beside_the_table_do_not_limit_the_shift():
    layout = _near_edge_layout()
    rack = Placement("rack", ObjectAsset("rack", (0.4, 1.5, 1.4), "item", "rack.usd"), (0.5, -0.9, 0.02), "item")
    layout.placements.append(rack)
    alone = shift_limits(_near_edge_layout(), (0.5, 0.0), (0.35, 0.5), 0.03)
    assert shift_limits(layout, (0.5, 0.0), (0.35, 0.5), 0.03) == alone
    shifted = with_jitter(layout, 4, 0.05, 0.05, alone)
    assert shifted.placements[2].position_m[0] - rack.position_m[0] == pytest.approx(
        shifted.placements[0].position_m[0] - layout.placements[0].position_m[0]
    )


def test_the_sampler_spawns_the_layout_that_the_first_reset_takes():
    layouts = [_layout(f"t{index}", "office", 2) for index in range(3)]
    stage = _stage(["office"])
    plan = plan_stage(layouts, stage, 6)
    sampler = LayoutSampler(layouts, stage, plan, seed=5, jitter_x=0.05, jitter_y=0.02)
    for env in range(6):
        first = sampler.first(env)
        assert first.task_id == layouts[plan.assignment[0][env]].task_id
        assert [p.position_m for p in first.placements] == [p.position_m for p in sampler.first(env).placements]
        expected = sampler.episode(env, 5 + 1 + env)
        assert [p.position_m for p in first.placements] == [p.position_m for p in expected.placements]
    assert len({sampler.first(env).placements[0].position_m for env in range(6)}) > 1
    assert sampler.first(0).placements[0].position_m != sampler.published(0).placements[0].position_m
    assert sampler.seed_of(7) == 12


def test_a_sampler_without_jitter_returns_the_published_layouts():
    layouts = [_layout("t", "office", 2)]
    stage = _stage(["office"])
    sampler = LayoutSampler(layouts, stage, plan_stage(layouts, stage, 2))
    assert sampler.first(1) is layouts[0]
    assert sampler.episode(0, 9) is layouts[0]


def test_table_rect_uses_the_stage_record_and_a_task_override():
    record = {"table_m": [1.1, 0.7, 0.04], "center_xy": [0.35, 0.1], "pos": [0.0, 0.0, 0.0]}
    assert table_rect(record, None) == ((0.35, 0.1), (0.55, 0.35))
    assert table_rect(record, {"pos": [0.6, -0.2, 0.0]}) == ((0.6, -0.2), (0.55, 0.35))
    assert table_rect({"pos": [0.5, 0.0, 0.0]}, None) == ((0.5, 0.0), (0.35, 0.5))

