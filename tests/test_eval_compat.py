# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parents[1] / "source" / "franka_tabletop_rl"
sys.path.insert(0, str(PACKAGE))

from franka_tabletop_rl.tasks.hl_policy.eval_compat import wrap_for_eval  # noqa: E402


class _Env:
    def __init__(self, count: int):
        tracker = SimpleNamespace(
            num_envs=count,
            completed=torch.zeros(count, dtype=torch.bool),
            failed=torch.zeros(count, dtype=torch.bool),
            progress=np.zeros(count),
            last_task_ids=[f"task-{index}" for index in range(count)],
            finished_steps=np.full(count, -1),
            instruction=lambda index: f"task {index}",
            prepare_reset=lambda: None,
            episode_finished=lambda: None,
            episode_steps=lambda: tracker.finished_steps.copy(),
        )

        def _clear() -> None:
            tracker.completed.zero_()
            tracker.failed.zero_()

        tracker.clear_outcomes = _clear
        term = SimpleNamespace(tracker=tracker)
        self.unwrapped = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: term), num_envs=count)
        self.tracker = tracker
        self.resets = 0

    def reset(self, *args, **kwargs):
        self.resets += 1
        return {"policy": torch.zeros(self.tracker.num_envs, 1)}, {}

    def step(self, action):
        return {"policy": torch.zeros(self.tracker.num_envs, 1)}, 0.0, False, False, {}


def test_outcomes_are_tensors_that_survive_steps_and_clear_on_reset():
    env = _Env(3)
    wrapped = wrap_for_eval(env)
    env.tracker.completed[1] = True
    wrapped.step(torch.zeros(3, 7))
    assert torch.is_tensor(wrapped.task_completed) and torch.is_tensor(wrapped.task_failed)
    assert wrapped.task_completed.tolist() == [False, True, False]
    wrapped.reset()
    assert env.resets == 1
    assert not bool(wrapped.task_completed.any())


def test_wrapper_forwards_ee_speed_and_discards_it_before_reset():
    env = _Env(1)
    term = env.unwrapped.command_manager.get_term("ee_pose")
    discarded: list[str] = []
    term.episode_ee_speeds = lambda: [np.array([0.2])]
    term.discard_ee_speed = lambda: discarded.append("discard")
    wrapped = wrap_for_eval(env)
    assert wrapped.episode_ee_speeds()[0].tolist() == [0.2]
    wrapped.reset()
    assert discarded == ["discard"]
    assert env.resets == 1


def test_wrapper_reports_the_task_each_env_finished():
    env = _Env(2)
    env.tracker.finished_steps[0] = 12
    wrapped = wrap_for_eval(env)
    assert wrapped.episode_task_ids() == ["task-0", "task-1"]
    assert wrapped.episode_finished() is None
    assert wrapped.episode_steps().tolist() == [12, -1]
    assert wrapped.instructions() == ["task 0", "task 1"]


def test_arm_start_offsets_differ_per_round_and_env_and_repeat_for_the_same_seed():
    from franka_tabletop_rl.tasks.hl_policy.start_noise import arm_start_offsets

    half = 0.05
    envs = list(range(16))
    # eval-nav passes seed 100 to every round, so only the round index separates the starts.
    rounds = [arm_start_offsets(100, round_index, envs, half) for round_index in range(3)]
    starts = np.concatenate(rounds)
    assert starts.shape == (48, 7)
    assert len({tuple(row) for row in starts.tolist()}) == 48
    assert np.abs(starts).max() <= half
    assert np.array_equal(rounds[1], arm_start_offsets(100, 1, envs, half))
    assert np.array_equal(rounds[1][3:5], arm_start_offsets(100, 1, [3, 4], half))
    assert not np.array_equal(rounds[0], arm_start_offsets(101, 0, envs, half))


def test_play_config_import_does_not_load_openusd():
    import pytest

    pytest.importorskip("isaaclab")
    # Play builds this config before Kit starts. Importing OpenUSD first breaks Kit's USD extensions.
    script = (
        "import sys\n"
        "import franka_tabletop_rl.tasks.hl_policy.hl_env_cfg\n"
        "franka_tabletop_rl.tasks.hl_policy.hl_env_cfg.HLEnvCfg_PLAY()\n"
        "sys.exit(1 if 'pxr' in sys.modules else 0)\n"
    )
    completed = subprocess.run([sys.executable, "-c", script], check=False)
    assert completed.returncode == 0


def test_a_finished_pose_is_latched_only_after_it_holds():
    import pytest

    pytest.importorskip("isaaclab")
    from franka_tabletop_rl.benchmark.predicates import task_progress
    from franka_tabletop_rl.benchmark.success import Contacts
    from franka_tabletop_rl.tasks.hl_policy.tracker import SUCCESS_HOLD_S, _commit_hold

    placed = {"in_container:item:container": True}
    stored, entered, hold, _ = {}, [], 0.0, False
    dt = 1.0 / 15.0
    for _ in range(7):
        stored, entered, hold, _ = _commit_hold(stored, entered, placed, ["item"], 1.0, hold, dt)
        assert stored == {}
        assert hold < SUCCESS_HOLD_S
    stored, entered, hold, _ = _commit_hold(stored, entered, placed, ["item"], 1.0, hold, dt)
    assert stored == placed
    assert hold >= SUCCESS_HOLD_S

    stored, entered, hold, _ = {}, [], 0.0, False
    stored, entered, hold, _ = _commit_hold(stored, entered, placed, ["item"], 1.0, hold, dt)
    partial = {"outside:top:bottom": True}
    stored, entered, hold, _ = _commit_hold(stored, entered, partial, [], 0.5, hold, dt)
    assert stored == partial
    assert entered == []
    assert hold == 0.0

    spec = _spec({"op": "in_container", "object": "cube", "container": "bin"}, ["cube", "bin"])
    poses = {"cube": _pose((0.2, 0.0, 0.05), (0.04,) * 3), "bin": _pose((0.8, 0.0, 0.05), (0.2, 0.2, 0.1))}
    assert task_progress(spec, poses, Contacts(), {}, [])[0] == 0.0
    assert task_progress(spec, poses, Contacts(), {"in_container:cube:bin": True}, [])[0] == 1.0


# The fixtures use dyadic lengths [m], so every boundary below is exact in floating point.
def _pose(pos, size, quat=(1.0, 0.0, 0.0, 0.0), anchor=False):
    from franka_tabletop_rl.benchmark.geometry import from_box

    pose = {
        "pos": np.array([pos], dtype=np.float64),
        "quat": np.array([quat], dtype=np.float64),
        "geometry": from_box(size),
    }
    # A pose starts at rest: its reset pose is the pose itself.
    pose["initial_pos"], pose["initial_quat"] = pose["pos"], pose["quat"]
    if anchor:
        pose["anchor"] = True
    return pose


def _spec(success, names):
    from franka_tabletop_rl.benchmark.spec import spec_from_success

    return spec_from_success("t", success, 40.0, {name: name for name in names})


def _met(success, poses, contacts=None, names=None, first_placed=None):
    from franka_tabletop_rl.benchmark.success import Contacts, task_satisfied

    spec = _spec(success, names or list(poses))
    return task_satisfied(spec, poses, contacts or Contacts(), first_placed or {})


def _contacts(**kwargs):
    from franka_tabletop_rl.benchmark.success import Contacts

    return Contacts(**kwargs)


CUBE = (0.0625,) * 3
# Open-top bin: x and y in [0.375, 0.625] m, floor at z = 0.
BIN = _pose((0.5, 0.0, 0.0625), (0.25, 0.25, 0.125))
IN_BIN = {"op": "in_container", "object": "cube", "container": "bin"}


def test_in_container_uses_the_hull_center_and_an_open_top():
    def at(x, z=0.09375):
        return {"cube": _pose((x, 0.0, z), CUBE), "bin": BIN}

    assert _met(IN_BIN, at(0.5))
    # The hull center on the wall plane still counts. A hair beyond it does not.
    assert _met(IN_BIN, at(0.625))
    assert not _met(IN_BIN, at(0.640625))
    # The open top reaches up without bound. Below the floor is outside.
    assert _met(IN_BIN, at(0.5, z=1.0))
    assert not _met(IN_BIN, at(0.5, z=-0.03125))
    # ``enclosed`` keeps the lid, so the same cube far above is not inside.
    assert not _met({**IN_BIN, "op": "enclosed"}, at(0.5, z=1.0))
    assert _met({**IN_BIN, "op": "enclosed"}, at(0.5))


def test_containment_tolerance_is_ignored():
    outside = {"cube": _pose((0.640625, 0.0, 0.09375), CUBE), "bin": BIN}
    assert not _met({**IN_BIN, "tolerance": 0.5}, outside)


def test_containment_tests_the_hull_center_not_the_root():
    from franka_tabletop_rl.benchmark.geometry import from_points

    corner = np.array([[x, y, z] for x in (0.0, 0.25) for y in (0.0, 0.25) for z in (0.0, 0.25)])
    item = {
        "pos": np.array([[0.375, -0.0625, 0.0625]]),
        "quat": np.array([[1.0, 0, 0, 0]]),
        "geometry": from_points(corner),
    }
    # The root sits on the wall, but the hull center of the mesh is 0.125 m inside it.
    assert _met(IN_BIN, {"cube": item, "bin": BIN})
    item = {**item, "pos": np.array([[0.125, -0.0625, 0.0625]])}
    assert not _met(IN_BIN, {"cube": item, "bin": BIN})


def test_release_and_contact_are_asked_per_stage():
    poses = {"cube": _pose((0.5, 0.0, 0.09375), CUBE), "bin": BIN}
    detach = {**IN_BIN, "require_gripper_detached": True}
    assert _met(detach, poses, _contacts(touching={"cube": False}))
    assert not _met(detach, poses, _contacts(touching={"cube": True}))
    # A missing finger reading cannot pass a release check.
    assert not _met(detach, poses, _contacts())
    assert _met(IN_BIN, poses, _contacts(touching={"cube": True}))

    touch = {**IN_BIN, "require_contact_with": True}
    assert not _met(touch, poses, _contacts())
    assert _met(touch, poses, _contacts(touch={("cube", "bin"): True}))
    assert _met(touch, poses, _contacts(touch={("bin", "cube"): True}))


def test_on_top_needs_a_support_force_and_the_box_center_over_the_support():
    plate = _pose((0.5, 0.0, 0.03125), (0.25, 0.25, 0.0625))
    block = {"op": "on_top", "object": "cube", "reference": "plate"}

    def at(x):
        return {"cube": _pose((x, 0.0, 0.09375), CUBE), "plate": plate}

    def push(*force, key=("cube", "plate")):
        return _contacts(force={key: np.array(force)})

    assert _met(block, at(0.5), push(0.0, 0.0, 1.0))
    # Reading the pair the other way round flips the force on the plate into the force on the cube.
    assert _met(block, at(0.5), push(0.0, 0.0, -1.0, key=("plate", "cube")))
    # Below 0.1 N, or outside the 45 degree cone, the plate does not carry the cube.
    assert not _met(block, at(0.5), push(0.0, 0.0, 0.05))
    assert not _met(block, at(0.5), push(1.5, 0.0, 1.0))
    assert _met(block, at(0.5), push(0.9, 0.0, 1.0))
    assert not _met(block, at(0.5), _contacts())
    # The footprint is the plate box grown by 0.01 m.
    assert _met(block, at(0.6328125), push(0.0, 0.0, 1.0))
    assert not _met(block, at(0.640625), push(0.0, 0.0, 1.0))
    assert _met({**block, "tolerance": 0.0625}, at(0.640625), push(0.0, 0.0, 1.0))


def test_stacked_needs_contact_and_a_box_above_the_support():
    red = _pose((0.5, 0.0, 0.0625), (0.125,) * 3)
    block = {"op": "stacked", "object": "yellow", "support": "red"}
    touch = _contacts(touch={("yellow", "red"): True}, touching={"yellow": True})

    def at(x, z):
        return {"yellow": _pose((x, 0.0, z), (0.125,) * 3), "red": red}

    assert _met(block, at(0.5, 0.1875), touch)
    # A block that is still held counts, unless the stage asks for the release.
    assert not _met({**block, "require_gripper_detached": True}, at(0.5, 0.1875), touch)
    assert _met(
        {**block, "require_gripper_detached": True},
        at(0.5, 0.1875),
        _contacts(touch={("yellow", "red"): True}, touching={"yellow": False}),
    )
    assert not _met(block, at(0.5, 0.1875), _contacts(touching={"yellow": True}))
    # Beside the support, the box center is outside its footprint.
    assert not _met(block, at(0.75, 0.0625), touch)


def test_front_is_toward_the_robot_and_the_level_check_can_be_skipped():
    bowl = _pose((0.5, 0.0, 0.0625), (0.125,) * 3)
    block = {"op": "on_side", "object": "cube", "reference": "bowl", "side": "front"}

    def at(x, y, z=0.0625):
        return {"cube": _pose((x, y, z), CUBE), "bowl": bowl}

    assert _met(block, at(0.25, 0.0625))
    assert not _met(block, at(0.75, 0.0))
    # |dy| / |d| = 0.93 is outside the 45 degree cone around -x.
    assert not _met(block, at(0.4, 0.25))
    # Level to within 0.1 m, unless the stage asks for release.
    assert not _met(block, at(0.25, 0.0, z=0.5))
    detached = {**block, "require_gripper_detached": True}
    assert _met(detached, at(0.25, 0.0, z=0.5), _contacts(touching={"cube": False}))
    for side, spot in {"left": (0.5, 0.25), "right": (0.5, -0.25), "behind": (0.75, 0.0)}.items():
        assert _met({**block, "side": side}, at(*spot))
    assert not _met({**block, "side": "left"}, at(0.5, -0.25))


def test_upright_ignores_yaw_and_reads_the_named_root_axis():
    def quat_about(axis, angle):
        half = angle / 2.0
        vector = [0.0, 0.0, 0.0]
        vector[axis] = np.sin(half)
        return (np.cos(half), *vector)

    block = {"op": "upright", "object": "mug"}

    def met(quat, **extra):
        return _met({**block, **extra}, {"mug": _pose((0.0, 0.0, 0.05), (0.08,) * 3, quat)})

    assert met(quat_about(2, np.pi / 2))
    assert met(quat_about(0, 0.05))
    assert not met(quat_about(0, 0.15))
    assert met(quat_about(0, 0.15), tolerance=0.2)
    # Root x pointing up: a quarter turn about -y.
    assert met(quat_about(1, -np.pi / 2), up_axis="x")
    assert not met(quat_about(1, -np.pi / 2))


def test_picked_up_needs_a_finger_and_the_lowest_corner_above_the_work_surface():
    surface = _pose((0.5, 0.0, -0.0625), (1.0, 1.0, 0.125), anchor=True)
    block = {"op": "picked_up", "object": "cube", "distance": 0.25}

    def met(z, grabbed=True):
        poses = {"cube": _pose((0.5, 0.0, z), CUBE), "surface": surface}
        return _met(block, poses, _contacts(touching={"cube": grabbed}), names=["cube"])

    assert met(0.3125)
    assert met(0.28125)
    assert not met(0.25)
    assert not met(0.3125, grabbed=False)


def test_centered_is_per_axis_and_defaults_to_one_centimetre():
    table = _pose((0.5, 0.0, -0.0625), (1.0, 1.0, 0.125), anchor=True)
    block = {"op": "centered", "object": "mug", "reference": "table"}

    def met(x, y, **extra):
        poses = {"mug": _pose((x, y, 0.1), (0.08,) * 3), "table": table}
        return _met({**block, **extra}, poses)

    # Each axis is within 0.01 m, though the distance is 0.011 m.
    assert met(0.5078125, 0.0078125)
    assert not met(0.515625, 0.0)
    assert met(0.515625, 0.0, tolerance=0.02)


def test_outside_and_on_surface_needs_the_work_surface_to_carry_the_object():
    keyboard = {"op": "outside_and_on_surface", "object": "cube", "container": "bin"}

    def poses(x):
        return {"cube": _pose((x, 0.0, 0.03125), CUBE), "bin": BIN}

    carried = _contacts(force={("cube", "surface"): np.array([0.0, 0.0, 1.0])})
    assert _met(keyboard, poses(0.9), carried)
    assert not _met(keyboard, poses(0.9), _contacts())
    assert not _met(keyboard, poses(0.5), carried)
    assert _met({**keyboard, "op": "outside"}, poses(0.9))


def test_choose_needs_exactly_k_objects_in_success_and_at_least_k_in_progress():
    from franka_tabletop_rl.benchmark.predicates import task_progress

    names = ["a", "b", "c", "bin"]
    block = {"op": "in_container", "objects": ["a", "b", "c"], "container": "bin", "logical": "choose", "k": 2}

    def poses(*placed):
        found = {name: _pose((0.5, 0.0, 0.09375) if name in placed else (0.9, 0.5, 0.03125), CUBE) for name in "abc"}
        return {**found, "bin": BIN}

    assert _met(block, poses("a", "c"))
    assert not _met(block, poses("a"))
    assert not _met(block, poses("a", "b", "c"))
    spec = _spec(block, names)
    assert task_progress(spec, poses("a", "b", "c"), _contacts(), {}, [])[0] == 1.0
    assert task_progress(spec, poses("a"), _contacts(), {}, [])[0] == 0.5


def test_in_order_compares_the_first_placement_steps():
    from franka_tabletop_rl.benchmark.success import note_placements, task_satisfied

    spec = _spec({"op": "in_order", "objects": ["a", "b"], "container": "bin"}, ["a", "b", "bin"])
    outside = _pose((0.9, 0.5, 0.03125), CUBE)
    inside = _pose((0.5, 0.0, 0.09375), CUBE)

    def run(first, second):
        placed: dict[str, int] = {}
        for step, poses in enumerate(
            [
                {"a": outside, "b": outside, "bin": BIN},
                {first: inside, "bin": BIN} | {name: outside for name in "ab" if name != first},
                {"a": inside, "b": inside, "bin": BIN},
            ]
        ):
            note_placements(spec, poses, _contacts(), placed, step)
        return task_satisfied(spec, {"a": inside, "b": inside, "bin": BIN}, _contacts(), placed)

    assert run("a", "b")
    assert not run("b", "a")


def test_progress_keeps_credit_but_success_does_not():
    from franka_tabletop_rl.benchmark.predicates import task_progress

    spec = _spec({**IN_BIN, "require_gripper_detached": True}, ["cube", "bin"])
    inside = {"cube": _pose((0.5, 0.0, 0.09375), CUBE), "bin": BIN}
    knocked = {"cube": _pose((0.9, 0.3, 0.03125), CUBE), "bin": BIN}
    free, gripping = _contacts(touching={"cube": False}), _contacts(touching={"cube": True})
    assert task_progress(spec, inside, gripping, {}, [])[0] == 0.0
    score, latched, _ = task_progress(spec, inside, free, {}, [])
    assert score == 1.0
    assert task_progress(spec, knocked, free, latched, [])[0] == 1.0
    assert not _met({**IN_BIN, "require_gripper_detached": True}, knocked, free)


def test_retired_ops_and_flags_are_rejected():
    import pytest

    for block in (
        {"op": "picked", "object": "cube"},
        {"op": "lifted", "object": "cube"},
        {"op": "resting", "object": "cube"},
        {"op": "off", "object": "cube", "support": "bin"},
        {**IN_BIN, "require_grasp": ["cube"]},
        {**IN_BIN, "hold": ["cube"]},
    ):
        with pytest.raises(ValueError):
            _spec(block, ["cube", "bin"])


def _turn(axis, angle):
    quat = [np.cos(angle / 2.0), 0.0, 0.0, 0.0]
    quat[1 + axis] = np.sin(angle / 2.0)
    return tuple(quat)


def _moved(pose, pos=None, quat=None):
    """The pose now at ``pos`` and ``quat``, with the reset pose left where it was."""
    moved = dict(pose)
    if pos is not None:
        moved["pos"] = np.array([pos], dtype=np.float64)
    if quat is not None:
        moved["quat"] = np.array([quat], dtype=np.float64)
    return moved


def _rebased(pose):
    """The pose with its reset pose set to where it is now."""
    return {**pose, "initial_pos": pose["pos"], "initial_quat": pose["quat"]}


def test_a_reference_has_to_stay_near_its_reset_pose():
    shelf = _pose((0.5, 0.0, 0.25), (0.25, 0.25, 0.5))
    block = {"op": "on_top", "object": "cube", "reference": "shelf"}
    carried = _contacts(force={("cube", "shelf"): np.array([0.0, 0.0, 1.0])})
    cube = _pose((0.5, 0.0, 0.53125), CUBE)

    def met(reference, rebase=False):
        return _met(block, {"cube": cube, "shelf": _rebased(reference) if rebase else reference}, carried)

    assert met(shelf)
    # Within 0.05 m and 15 degrees, a settled shelf still counts.
    assert met(_moved(shelf, pos=(0.546875, 0.0, 0.25), quat=_turn(0, np.radians(14.0))))
    # Shoved past 0.05 m, or turned past 15 degrees, it does not. The geometry alone would pass.
    shoved = _moved(shelf, pos=(0.5625, 0.0, 0.25))
    assert not met(shoved) and met(shoved, rebase=True)
    turned = _moved(shelf, quat=_turn(0, np.radians(16.0)))
    assert not met(turned) and met(turned, rebase=True)
    # A tipped shelf is the case in point: the mug rests on the fallen side.
    tipped = _moved(shelf, quat=_turn(1, np.pi / 2.0))
    assert not met(tipped) and met(tipped, rebase=True)


def test_a_reference_without_a_reset_pose_fails_its_stage():
    poses = {"cube": _pose((0.5, 0.0, 0.09375), CUBE), "bin": BIN}
    assert _met(IN_BIN, poses)
    bare = {key: value for key, value in BIN.items() if not key.startswith("initial")}
    assert not _met(IN_BIN, {**poses, "bin": bare})


def test_a_shoved_container_does_not_count_as_taking_an_object_out_of_it():
    block = {"op": "outside_and_on_surface", "object": "cube", "container": "bin"}
    carried = _contacts(force={("cube", "surface"): np.array([0.0, 0.0, 1.0])})
    cube = _pose((0.5, 0.0, 0.03125), CUBE)
    # The container slides away from the object, which sits on the table.
    away = _moved(BIN, pos=(0.5, 0.5, 0.0625))
    assert not _met(block, {"cube": cube, "bin": away}, carried)
    assert _met(block, {"cube": cube, "bin": _rebased(away)}, carried)


def test_objects_the_task_places_may_leave_their_reset_pose_but_the_base_may_not():
    block = {
        "op": "all",
        "checks": [
            {"op": "stacked", "object": "blue", "support": "red"},
            {"op": "stacked", "object": "green", "support": "blue"},
        ],
    }
    edge = (0.125,) * 3
    touch = _contacts(touch={("blue", "red"): True, ("green", "blue"): True})
    red = _pose((0.5, 0.0, 0.0625), edge)
    # Blue and green came from elsewhere. Blue is the support of green and is placed by its own stage.
    blue = _moved(_pose((0.9, 0.5, 0.0625), edge), pos=(0.5, 0.0, 0.1875))
    green = _moved(_pose((0.9, -0.5, 0.0625), edge), pos=(0.5, 0.0, 0.3125))
    assert _met(block, {"red": red, "blue": blue, "green": green}, touch)
    # The base was shoved 0.0625 m, so the tower no longer counts.
    shoved = _moved(red, pos=(0.5, 0.0625, 0.0625))
    assert not _met(block, {"red": shoved, "blue": blue, "green": green}, touch)
    assert _met(block, {"red": _rebased(shoved), "blue": blue, "green": green}, touch)


def test_only_the_passing_branch_of_an_any_keeps_its_support_in_place():
    block = {
        "op": "any",
        "checks": [
            {"op": "stacked", "object": "upper", "support": "lower"},
            {"op": "stacked", "object": "lower", "support": "upper"},
        ],
    }
    edge = (0.125,) * 3
    touch = _contacts(touch={("upper", "lower"): True})
    upper = _moved(_pose((0.9, 0.5, 0.0625), edge), pos=(0.5, 0.0, 0.1875))
    lower = _pose((0.5, 0.0, 0.0625), edge)
    assert _met(block, {"upper": upper, "lower": lower}, touch)
    # The mug that carries the other is not exempt just because the other branch places it.
    assert not _met(block, {"upper": upper, "lower": _moved(lower, pos=(0.5, 0.0625, 0.0625))}, touch)


def test_a_shoved_container_earns_no_progress():
    from franka_tabletop_rl.benchmark.predicates import task_progress

    spec = _spec(IN_BIN, ["cube", "bin"])
    cube = _pose((0.5, 0.5, 0.09375), CUBE)
    shoved = _moved(BIN, pos=(0.5, 0.5, 0.0625))
    score, latched, _ = task_progress(spec, {"cube": cube, "bin": shoved}, _contacts(), {}, [])
    assert score == 0.0 and latched == {}
    assert task_progress(spec, {"cube": cube, "bin": _rebased(shoved)}, _contacts(), {}, [])[0] == 1.0
