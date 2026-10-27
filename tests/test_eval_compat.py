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
    from franka_tabletop_rl.benchmark.spec import spec_from_success
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
    partial = {"off:top:bottom": True}
    stored, entered, hold, _ = _commit_hold(stored, entered, partial, [], 0.5, hold, dt)
    assert stored == partial
    assert entered == []
    assert hold == 0.0

    spec = spec_from_success(
        "t",
        {"op": "in_container", "object": "cube", "container": "bin", "require_grasp": []},
        40.0,
        {"cube": "item", "bin": "container"},
    )
    poses = {
        "item": {
            "pos": np.array([[0.2, 0.0, 0.05]]),
            "quat": np.array([[1.0, 0.0, 0.0, 0.0]]),
            "half": np.array([0.02, 0.02, 0.02]),
        },
        "container": {
            "pos": np.array([[0.8, 0.0, 0.05]]),
            "quat": np.array([[1.0, 0.0, 0.0, 0.0]]),
            "half": np.array([0.1, 0.1, 0.05]),
        },
    }
    open_hand = {"item": False}
    score, _, _ = task_progress(spec, poses, open_hand, open_hand, {}, [], gripper_open=True)
    assert score == 0.0
    score, _, _ = task_progress(spec, poses, open_hand, open_hand, placed, [], gripper_open=True)
    assert score == 1.0


def _pose(pos, half, quat=(1.0, 0.0, 0.0, 0.0)):
    return {
        "pos": np.array([pos], dtype=np.float64),
        "quat": np.array([quat], dtype=np.float64),
        "half": np.array(half, dtype=np.float64),
    }


def _spec(success, names):
    from franka_tabletop_rl.benchmark.spec import spec_from_success

    return spec_from_success("t", success, 40.0, {name: name for name in names})


def test_success_is_judged_on_the_current_state():
    from franka_tabletop_rl.benchmark.predicates import task_progress
    from franka_tabletop_rl.benchmark.success import task_satisfied

    spec = _spec({"op": "in_container", "object": "cube", "container": "bin"}, ["cube", "bin"])
    bin_pose = _pose((0.5, 0.0, 0.05), (0.1, 0.1, 0.05))
    inside = {"cube": _pose((0.5, 0.0, 0.06), (0.02,) * 3), "bin": bin_pose}
    free, gripping = {"cube": False}, {"cube": True}
    assert task_satisfied(spec, inside, free, {})
    # The object still touches the fingers.
    assert not task_satisfied(spec, inside, gripping, {})
    # The latched progress keeps its credit after the cube is knocked out, but success does not.
    knocked = {"cube": _pose((0.9, 0.3, 0.02), (0.02,) * 3), "bin": bin_pose}
    latched = {"in_container:cube:bin": True}
    score, _, _ = task_progress(spec, knocked, free, free, latched, [], gripper_open=True)
    assert score == 1.0
    assert not task_satisfied(spec, knocked, free, {})


def test_a_held_block_still_counts_as_stacked():
    from franka_tabletop_rl.benchmark.success import task_satisfied

    spec = _spec({"op": "stacked", "object": "yellow", "support": "red"}, ["yellow", "red"])
    poses = {
        "red": _pose((0.5, 0.0, 0.0235), (0.0235,) * 3),
        "yellow": _pose((0.5, 0.0, 0.0705), (0.0235,) * 3),
    }
    assert task_satisfied(spec, poses, {"yellow": True}, {})
    hovering = {**poses, "yellow": _pose((0.5, 0.0, 0.1), (0.0235,) * 3)}
    assert not task_satisfied(spec, hovering, {"yellow": True}, {})


def test_upright_ignores_yaw():
    from franka_tabletop_rl.benchmark.success import task_satisfied

    spec = _spec({"op": "upright", "object": "mug"}, ["mug"])
    yawed = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))
    tilted = (np.cos(np.pi / 8), np.sin(np.pi / 8), 0.0, 0.0)
    assert task_satisfied(spec, {"mug": _pose((0, 0, 0.05), (0.04,) * 3, yawed)}, {"mug": False}, {})
    assert not task_satisfied(spec, {"mug": _pose((0, 0, 0.05), (0.04,) * 3, tilted)}, {"mug": False}, {})


def test_front_is_toward_the_robot():
    from franka_tabletop_rl.benchmark.success import task_satisfied

    spec = _spec({"op": "on_side", "object": "cube", "reference": "bowl", "side": "front"}, ["cube", "bowl"])
    bowl = _pose((0.5, 0.0, 0.05), (0.05,) * 3)
    nearer = {"cube": _pose((0.3, 0.02, 0.05), (0.02,) * 3), "bowl": bowl}
    farther = {"cube": _pose((0.7, 0.0, 0.05), (0.02,) * 3), "bowl": bowl}
    assert task_satisfied(spec, nearer, {"cube": False}, {})
    assert not task_satisfied(spec, farther, {"cube": False}, {})


def test_choose_needs_exactly_k_objects():
    from franka_tabletop_rl.benchmark.success import task_satisfied

    names = ["a", "b", "c", "bin"]
    spec = _spec(
        {"op": "in_container", "objects": ["a", "b", "c"], "container": "bin", "logical": "choose", "k": 2}, names
    )
    inside = (0.5, 0.0, 0.06)
    outside = (0.9, 0.5, 0.02)
    free = {name: False for name in names}

    def poses(*placed):
        found = {name: _pose(inside if name in placed else outside, (0.02,) * 3) for name in "abc"}
        return {**found, "bin": _pose((0.5, 0.0, 0.05), (0.1, 0.1, 0.05))}

    assert task_satisfied(spec, poses("a", "c"), free, {})
    assert not task_satisfied(spec, poses("a"), free, {})
    assert not task_satisfied(spec, poses("a", "b", "c"), free, {})


def test_in_order_compares_the_first_placement_steps():
    from franka_tabletop_rl.benchmark.success import note_placements, task_satisfied

    spec = _spec({"op": "in_order", "objects": ["a", "b"], "container": "bin"}, ["a", "b", "bin"])
    free = {"a": False, "b": False}
    bin_pose = _pose((0.5, 0.0, 0.05), (0.1, 0.1, 0.05))
    outside = _pose((0.9, 0.5, 0.02), (0.02,) * 3)
    inside = _pose((0.5, 0.0, 0.06), (0.02,) * 3)

    def run(first, second):
        placed: dict[str, int] = {}
        for step, poses in enumerate(
            [
                {"a": outside, "b": outside, "bin": bin_pose},
                {first: inside, "bin": bin_pose} | {name: outside for name in "ab" if name != first},
                {"a": inside, "b": inside, "bin": bin_pose},
            ]
        ):
            note_placements(spec, poses, free, placed, step)
        return task_satisfied(spec, {"a": inside, "b": inside, "bin": bin_pose}, free, placed)

    assert run("a", "b")
    assert not run("b", "a")
