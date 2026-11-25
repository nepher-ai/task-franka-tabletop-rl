# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open-top containment for a slender object whose hull center sits outside the holder."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from franka_tabletop_rl.benchmark.geometry import centerline_inside_length, rotation
from franka_tabletop_rl.benchmark.spec import spec_from_success
from franka_tabletop_rl.benchmark.success import Contacts, task_satisfied

# Pink spaghetti spoon and the scaled storage box from cooking_pick_pasta_tool.
SPOON = (0.315996, 0.06029, 0.040158)
HOLDER = (0.12836, 0.098834, 0.185406)
CUBE = (0.0625,) * 3
# Open-top bin: x and y in [0.375, 0.625] m, floor at z = 0.
BIN = ((0.5, 0.0, 0.0625), (0.25, 0.25, 0.125))


def _pose(pos, size, quat=(1.0, 0.0, 0.0, 0.0)):
    from franka_tabletop_rl.benchmark.geometry import from_box

    pose = {
        "pos": np.array([pos], dtype=np.float64),
        "quat": np.array([quat], dtype=np.float64),
        "geometry": from_box(size),
    }
    pose["initial_pos"], pose["initial_quat"] = pose["pos"], pose["quat"]
    return pose


def _met(success, poses):
    spec = spec_from_success("t", success, 40.0, {name: name for name in poses})
    return task_satisfied(spec, poses, Contacts(), {})


def _tilt_from_vertical(degrees: float) -> tuple[float, float, float, float]:
    """wxyz rotation about y that sends local +x to ``degrees`` off world up, toward +x."""
    half = 0.5 * np.deg2rad(degrees - 90.0)
    return (float(np.cos(half)), 0.0, float(np.sin(half)), 0.0)


def _spoon_in_holder(degrees: float) -> dict:
    """Spoon whose lower end sits on the holder floor, leaning ``degrees`` off vertical."""
    quat = _tilt_from_vertical(degrees)
    axis = rotation(quat)[:, 0]
    half_z = HOLDER[2] / 2.0
    center = np.array([0.0, 0.0, -half_z]) + axis * (SPOON[0] / 2.0)
    return {
        "pink_spaghetti_spoon": _pose(center, SPOON, quat),
        "storage_box_01": _pose((0.0, 0.0, 0.0), HOLDER),
    }


def _in_holder(poses) -> bool:
    block = {"op": "in_container", "object": "pink_spaghetti_spoon", "container": "storage_box_01"}
    return _met(block, poses)


def test_hull_center_still_decides_a_compact_object():
    block = {"op": "in_container", "object": "cube", "container": "bin"}
    bin_pose = _pose(*BIN)

    def at(x, z=0.09375):
        return {"cube": _pose((x, 0.0, z), CUBE), "bin": bin_pose}

    assert _met(block, at(0.5))
    assert _met(block, at(0.625))
    assert not _met(block, at(0.640625))
    assert _met(block, at(0.5, z=1.0))
    assert not _met(block, at(0.5, z=-0.03125))
    assert not _met({**block, "op": "enclosed"}, at(0.5, z=1.0))


def test_a_seated_rod_counts_when_its_center_is_outside_the_wall():
    # 0.5 m rod, 0.25 m opening. The center is past the wall and 0.2 m of the shaft is inside.
    rod = (0.5, 0.03125, 0.03125)
    poses = {"rod": _pose((0.16, 0.0, 0.0), rod), "bin": _pose((0.0, 0.0, 0.0), (0.25, 0.25, 0.25))}
    block = {"op": "in_container", "object": "rod", "container": "bin"}
    assert _met(block, poses)
    assert not _met({**block, "op": "enclosed"}, poses)
    assert not _met({**block, "op": "outside"}, poses)


def test_a_short_tip_and_a_shaft_above_the_lid_do_not_count():
    rod = (0.5, 0.03125, 0.03125)
    holder = _pose((0.0, 0.0, 0.0), (0.25, 0.25, 0.25))
    block = {"op": "in_container", "object": "rod", "container": "bin"}
    tip = {"rod": _pose((0.34, 0.0, 0.0), rod), "bin": holder}
    above = {"rod": _pose((0.16, 0.0, 0.25), rod), "bin": holder}
    assert not _met(block, tip)
    assert not _met(block, above)


def test_a_slender_center_above_the_rim_does_not_use_the_open_column():
    rod = (0.5, 0.03125, 0.03125)
    quat = _tilt_from_vertical(0.0)
    assert np.allclose(rotation(quat)[:, 0], (0.0, 0.0, 1.0), atol=1e-6)
    poses = {"rod": _pose((0.0, 0.0, 0.4), rod, quat), "bin": _pose((0.0, 0.0, 0.0), (0.25, 0.25, 0.25))}
    block = {"op": "in_container", "object": "rod", "container": "bin"}
    seated = centerline_inside_length(
        poses["rod"]["geometry"],
        poses["rod"]["pos"],
        poses["rod"]["quat"],
        poses["bin"]["pos"],
        poses["bin"]["quat"],
        poses["bin"]["geometry"].planes_full,
    )
    assert seated == 0.0
    assert not _met(block, poses)
    assert not _met({**block, "op": "enclosed"}, poses)


def test_a_slender_center_inside_a_short_cavity_is_not_enough():
    # The center sits in the cavity, but the cavity is shorter than 35% of the rod.
    rod = (1.0, 0.02, 0.02)
    poses = {"rod": _pose((0.0, 0.0, 0.0), rod), "bin": _pose((0.0, 0.0, 0.0), (0.25, 0.25, 0.25))}
    block = {"op": "in_container", "object": "rod", "container": "bin"}
    assert not _met(block, poses)


def test_a_stubby_object_keeps_the_hull_center_test():
    holder = _pose((0.0, 0.0, 0.0), (0.25, 0.25, 0.25))
    block = {"op": "in_container", "object": "item", "container": "bin"}
    # Longest edge is 2.4 times the next, under the 3x slender cut, and past the wall.
    stubby = {"item": _pose((0.2, 0.0, 0.0), (0.3, 0.125, 0.125)), "bin": holder}
    assert not _met(block, stubby)
    assert _met(block, {"item": _pose((0.0, 0.0, 0.0), (0.3, 0.125, 0.125)), "bin": holder})


def test_inside_requires_every_hull_vertex_in_the_closed_hull():
    holder = _pose((0.0, 0.0, 0.0), (0.25, 0.25, 0.25))
    block = {"op": "inside", "object": "carton", "container": "bin"}
    # The carton fills the cavity. Corners on the faces still count.
    flush = {"carton": _pose((0.0, 0.0, 0.0), (0.25, 0.25, 0.25)), "bin": holder}
    assert _met(block, flush)
    # The center is inside, but the top corners are past the lid.
    poking = {"carton": _pose((0.0, 0.0, 0.1), (0.1, 0.1, 0.1)), "bin": holder}
    assert not _met(block, poking)
    assert _met({**block, "op": "enclosed"}, poking)
    assert not _met({**block, "tolerance": 0.5}, poking)
    # Same crate orientation as recycle_cartons_vertical_crate: the mouth faces world -X.
    crate = _pose((0.0, 0.0, 0.0), (0.2, 0.2, 0.2), (0.5, -0.5, -0.5, 0.5))
    seated = {"carton": _pose((-0.02, 0.0, 0.0), (0.04, 0.04, 0.04)), "bin": crate}
    proud = {"carton": _pose((-0.15, 0.0, 0.0), (0.04, 0.04, 0.04)), "bin": crate}
    side = {"op": "inside", "object": "carton", "container": "bin"}
    assert _met(side, seated)
    assert not _met(side, proud)


def test_vertical_crate_scores_inside_and_places_as_a_container():
    import json

    path = (
        Path(__file__).resolve().parents[2]
        / "envhub"
        / "environments"
        / "franka-tabletop-rl-v8"
        / "tasks"
        / "recycle_cartons_vertical_crate.json"
    )
    task = json.loads(path.read_text(encoding="utf-8"))
    names = {"milk_carton": "milk_carton", "orange_juice_carton": "orange_juice_carton", "container_a01": "container_a01"}
    spec = spec_from_success(task["task_id"], task["success"], 90.0, names)
    assert task["success"]["op"] == "inside"
    assert [goal.op for goal in spec.check.goals] == ["inside", "inside"]
    assert all(goal.detach and goal.target == "container_a01" for goal in spec.check.goals)

    from franka_tabletop_rl.tasks.hl_policy.planner.goals import compile_goals

    queue = compile_goals(
        spec.check,
        {"milk_carton": (0.36, 0.31), "orange_juice_carton": (0.5, 0.2), "container_a01": (0.72, 0.0)},
    )
    assert [goal.op for goal in queue.goals] == ["in_container", "in_container"]
    assert [goal.spec.op for goal in queue.goals] == ["inside", "inside"]


def test_pasta_spoon_leaning_in_the_holder():
    """A 30 degree lean seats more than 35% of the spoon. A 45 degree lean does not."""
    shallow = _spoon_in_holder(30.0)
    steep = _spoon_in_holder(45.0)
    assert _in_holder(shallow)
    assert not _in_holder(steep)
    assert not _met(
        {"op": "enclosed", "object": "pink_spaghetti_spoon", "container": "storage_box_01"},
        shallow,
    )
