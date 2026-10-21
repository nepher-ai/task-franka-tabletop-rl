# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grasp search and the planner's stage logic, without a simulator.

Run with the Isaac Lab python, for example ``isaaclab.bat -p -m pytest tests/test_planner.py``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1] / "source" / "franka_tabletop_rl"
sys.path.insert(0, str(PACKAGE))

import pytest  # noqa: E402

from franka_tabletop_rl.benchmark.spec import spec_from_success  # noqa: E402
from franka_tabletop_rl.benchmark.task import ObjectAsset, Placement, TaskLayout  # noqa: E402
from franka_tabletop_rl.tasks.hl_policy.planner.goals import compile_goals  # noqa: E402
from franka_tabletop_rl.tasks.hl_policy.planner.grasp import (  # noqa: E402
    MAX_WIDTH_M,
    PALM_CLEARANCE_M,
    TOOL_Z_MIN_M,
    find_grasps,
    tool_yaw,
)


def _box(size, center=(0.0, 0.0, 0.0), step=0.006) -> np.ndarray:
    """Surface points of a box with the given size [m]."""
    half = np.asarray(size) / 2
    axes = [np.linspace(-h, h, int(math.ceil(2 * h / step)) + 1) for h in half]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    shell = np.any(np.isclose(np.abs(grid), half, atol=1e-9), axis=1)
    return grid[shell] + np.asarray(center)


def _ring(radius, height, wall=0.004, step=0.005) -> np.ndarray:
    """Open bowl wall: a thin cylinder shell with its base on the root plane."""
    angles = np.arange(0.0, 2 * math.pi, step / radius)
    layers = np.arange(0.0, height + 1e-9, step)
    points = []
    for r in (radius - wall / 2, radius + wall / 2):
        for z in layers:
            points.append(np.stack([r * np.cos(angles), r * np.sin(angles), np.full_like(angles, z)], axis=-1))
    return np.concatenate(points)


def test_cube_is_pinched_at_its_center_across_a_horizontal_axis():
    cube = _box((0.047, 0.047, 0.047), center=(0.0, 0.0, 0.0235))
    grasps = find_grasps(cube, root_z=0.0)
    assert grasps
    best = grasps[0]
    assert abs(best.width_m - 0.047) < 0.01
    assert np.linalg.norm(best.center_local[:2]) < 0.02
    assert best.axis_local[2] == 0.0 and abs(np.linalg.norm(best.axis_local) - 1.0) < 1e-9


def test_tall_object_is_gripped_above_its_middle_and_clear_of_the_table():
    can = _ring(0.034, 0.102)
    best = find_grasps(can, root_z=0.003)[0]
    height = 0.003 + best.center_local[2]
    assert height >= TOOL_Z_MIN_M
    assert height > 0.003 + 0.051


def _machine():
    pytest.importorskip("isaaclab")
    from franka_tabletop_rl.tasks.hl_policy.planner.machine import (
        ClassicalTabletopPlanner,
        Stage,
        drop_offsets,
        reach_limit,
    )

    return ClassicalTabletopPlanner, Stage, drop_offsets, reach_limit


def test_reach_limit_pulls_far_and_near_targets_into_the_arm_s_workspace():
    _, _, _, reach_limit = _machine()
    far = reach_limit(np.array([0.7, 0.5]), 0.24)
    assert np.hypot(*far) <= 0.77
    assert reach_limit(np.array([0.2, 0.0]), 0.24)[0] >= 0.38
    assert np.allclose(reach_limit(np.array([0.5, 0.1]), 0.24), [0.5, 0.1])


def test_block_wider_than_the_opening_is_only_gripped_at_its_edge():
    slab = _box((0.12, 0.12, 0.05), center=(0.0, 0.0, 0.025))
    grasps = find_grasps(slab, root_z=0.0)
    assert all(np.linalg.norm(grasp.center_local[:2]) > 0.04 for grasp in grasps)


def test_bowl_is_gripped_by_its_wall():
    bowl = _ring(radius=0.08, height=0.055)
    grasps = find_grasps(bowl, root_z=0.0)
    assert grasps
    for grasp in grasps:
        assert abs(np.hypot(*grasp.center_local[:2]) - 0.08) < 0.02
        assert grasp.width_m < 0.03


def test_tall_box_is_gripped_near_the_top_so_the_palm_clears_it():
    height = 0.20
    box = _box((0.05, 0.05, height), center=(0.0, 0.0, height / 2))
    grasps = find_grasps(box, root_z=0.0)
    assert grasps
    assert all(grasp.center_local[2] >= height - PALM_CLEARANCE_M - 0.01 for grasp in grasps)
    assert all(grasp.width_m <= MAX_WIDTH_M for grasp in grasps)


def test_tool_yaw_is_a_half_turn_symmetric_angle():
    for angle in np.linspace(-math.pi, math.pi, 25):
        axis = np.array([math.cos(angle), math.sin(angle)])
        yaw = tool_yaw(axis)
        assert -math.pi / 2 < yaw <= math.pi / 2 + 1e-9
        assert abs(math.sin(yaw - tool_yaw(-axis))) < 1e-9
    assert abs(tool_yaw(np.array([0.0, 1.0]))) < 1e-9


def test_drop_offsets_stay_inside_the_footprint_and_do_not_repeat():
    _, _, drop_offsets, _ = _machine()
    assert drop_offsets(1, 0.2, 0.1) == [(0.0, 0.0)]
    offsets = drop_offsets(6, 0.21, 0.14)
    assert len(set(offsets)) == 6
    assert all(abs(x) <= 0.21 and abs(y) <= 0.14 for x, y in offsets)


class _Sim:
    """Perfect tracker and gripper: the tool reaches its target, a closed gripper carries what it holds."""

    def __init__(self, planner, tracker, size):
        self.planner, self.tracker, self.size = planner, tracker, size
        self.tcp = np.array([0.45, 0.0, 0.26])
        self.closure = 0.0
        self.holding: str | None = None
        self.offset = np.zeros(3)

    def step(self) -> None:
        target, _, grip = self.planner.step(self.tcp[None], np.array([self.closure]))
        self.tcp = target[0].copy()
        item = "item"
        pos = self.tracker.roots[item][0][0]
        close = grip[0] > 0.5
        if close and self.holding is None and np.linalg.norm(pos - self.tcp) < 0.03:
            self.holding = item
            self.offset = pos - self.tcp
        self.closure = min(self.closure + 0.3, 0.5 if self.holding else 1.0) if close else max(self.closure - 0.3, 0.0)
        if not close and self.holding:
            self.holding = None
            pos[2] = 0.05
        if self.holding:
            pos[:] = self.tcp + self.offset
        self.tracker.contact[item][0] = bool(self.holding)


def _fixture():
    cube = ObjectAsset("cube", (0.047, 0.047, 0.047), "item", "cube.usd")
    bin_ = ObjectAsset("bin", (0.3, 0.2, 0.1), "container", "bin.usd")
    placements = [
        Placement("item", cube, (0.55, -0.2, 0.0235), "item"),
        Placement("container", bin_, (0.5, 0.25, 0.05), "container"),
    ]
    success = {
        "op": "in_container",
        "objects": ["cube"],
        "logical": "all",
        "container": "bin",
        "require_grasp": ["cube"],
    }
    spec = spec_from_success("t", success, 40.0, {"cube": "item", "bin": "container"})
    scene = TaskLayout("t", "lay", "office", "oak", "put the cube in the bin", placements, spec=spec)
    roots = {
        "item": (np.array([[0.55, -0.2, 0.0235]]), np.array([[0.0, 0.0, 0.0, 1.0]])),
        "container": (np.array([[0.5, 0.25, 0.05]]), np.array([[0.0, 0.0, 0.0, 1.0]])),
    }
    contact = {"item": np.zeros(1, dtype=bool), "container": np.zeros(1, dtype=bool)}
    tracker = SimpleNamespace(
        num_envs=1,
        task_ids=("t",),
        task_index=np.zeros(1, dtype=int),
        specs=(spec,),
        task_layouts=(scene,),
        roots=roots,
        contact=contact,
        latched=[{}],
        is_lost=lambda pos: bool(pos[2] < -0.15),
        root=lambda index, slot: (roots[slot][0][index], roots[slot][1][index]),
        touching=lambda index, slot: bool(contact[slot][index]),
    )
    points = {
        "item": _box((0.047, 0.047, 0.047)) + np.array([0.0, 0.0, 0.0]),
        "container": _box((0.3, 0.2, 0.1)),
    }
    rest = {
        "item": (np.array([0.55, -0.2, 0.0235]), np.array([1.0, 0.0, 0.0, 0.0])),
        "container": (np.array([0.5, 0.25, 0.05]), np.array([1.0, 0.0, 0.0, 0.0])),
    }
    return tracker, [points], [rest]


def test_goal_queue_covers_each_op_and_leaves_upright_unsupported():
    names = {"cube": "item", "bin": "container", "plate": "plate"}
    positions = {"item": (0.5, 0.0), "container": (0.6, 0.2), "plate": (0.9, 0.4)}

    def queue(success):
        spec = spec_from_success("t", success, 40.0, names)
        return compile_goals(spec.check, positions)

    placed = queue({"op": "on_top", "object": "cube", "support": "plate"})
    assert placed.supported and placed.goals[0].op == "on_top" and placed.goals[0].slot == "item"
    held = queue({"op": "picked", "object": "cube"})
    assert held.goals[0].op == "picked"
    ordered = queue({"op": "in_order", "objects": ["plate", "cube"], "container": "bin"})
    assert ordered.ordered and [goal.slot for goal in ordered.goals] == ["plate", "item"]
    either = queue(
        {
            "op": "any",
            "checks": [
                {"op": "in_container", "object": "plate", "container": "bin"},
                {"op": "in_container", "object": "cube", "container": "bin"},
            ],
        }
    )
    assert [goal.slot for goal in either.goals] == ["item"]
    chosen = queue({"op": "choose", "k": 1, "objects": ["plate", "cube"], "container": "bin"})
    assert [goal.slot for goal in chosen.goals] == ["item"]
    upright = queue({"op": "upright", "object": "cube"})
    assert not upright.supported and upright.goals == ()


def test_planner_carries_the_item_over_the_container_and_lets_go():
    ClassicalTabletopPlanner, Stage, _, _ = _machine()
    tracker, points, rest = _fixture()
    planner = ClassicalTabletopPlanner(tracker, points, rest)
    planner.reset([0])
    sim = _Sim(planner, tracker, None)
    stages = set()
    for _ in range(400):
        sim.step()
        stages.add(planner.contexts[0].stage)
        if planner.contexts[0].stage == Stage.DONE:
            break
        if tracker.roots["item"][0][0][2] < 0.07 and Stage.OPEN in stages and sim.holding is None:
            # Once released over the container, the item counts as placed.
            tracker.latched[0]["in_container:item:container"] = True
    item = tracker.roots["item"][0][0]
    container = tracker.roots["container"][0][0]
    assert planner.contexts[0].stage == Stage.DONE
    assert {Stage.PRE_GRASP, Stage.DESCEND, Stage.CLOSE, Stage.LIFT, Stage.CARRY, Stage.LOWER, Stage.OPEN} <= stages
    assert abs(item[0] - container[0]) < 0.12 and abs(item[1] - container[1]) < 0.08
