# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot, physics, and success scoring match ``task-franka-tabletop``.

These checks need Isaac Lab and an installed ``franka_tabletop``. They skip otherwise.
"""

from __future__ import annotations

import pytest

pytest.importorskip("isaaclab")
brain = pytest.importorskip("franka_tabletop")


def test_robot_and_physics_constants_match():
    from franka_tabletop.core import robot as original

    from franka_tabletop_rl.scene.robot import ARM_JOINTS, GRIPPER_CLOSED_RAD
    from franka_tabletop_rl.scene.scene_cfg import CONTROL_HZ, EPISODE_LENGTH_S

    assert GRIPPER_CLOSED_RAD == original.GRIPPER_CLOSED_RAD
    assert tuple(ARM_JOINTS) == tuple(original.ARM_JOINTS)
    assert CONTROL_HZ == 15
    assert EPISODE_LENGTH_S == 40.0


def test_task_progress_matches_the_brain_on_v1(requires_bundle, tmp_path):
    from franka_tabletop_rl.benchmark.bundle import load_benchmark
    from franka_tabletop_rl.benchmark.legacy_layout import legacy_bundle_view
    from franka_tabletop_rl.benchmark.predicates import task_progress

    try:
        from franka_tabletop.predicates import task_progress as brain_progress
    except ImportError:
        pytest.skip("franka_tabletop has no predicates.task_progress")

    root = requires_bundle("franka-tabletop-rl-v1")
    view = legacy_bundle_view(root, tmp_path / "brain")
    benchmark = load_benchmark(root)
    assert (view / "scenes").is_dir()
    for layout in benchmark.layouts:
        if layout.spec is None:
            continue
        poses = {}
        for placement in layout.placements:
            import numpy as np

            poses[placement.slot] = {
                "pos": np.asarray(placement.position_m, dtype=np.float64).reshape(1, 3),
                "quat": np.asarray(placement.quat_wxyz, dtype=np.float64).reshape(1, 4),
                "half": np.asarray(placement.asset.size_m, dtype=np.float64) * 0.5,
            }
        held = {slot: False for slot in poses}
        ours, _, _ = task_progress(layout.spec, poses, held, held, {}, [], side=layout.side, gripper_open=True)
        theirs, _, _ = brain_progress(layout.spec, poses, held, held, {}, [], side=layout.side, gripper_open=True)
        assert ours == pytest.approx(theirs)
