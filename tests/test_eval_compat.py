# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The evaluator reads outcome flags as tensors, and they must survive the env's auto-reset."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parents[1] / "source" / "franka_tabletop_rl"
sys.path.insert(0, str(PACKAGE))

from franka_tabletop_rl.tasks.manager_based.hl_policy.eval_compat import wrap_for_eval  # noqa: E402
from franka_tabletop_rl.tasks.manager_based.hl_policy.task_state import TaskTracker  # noqa: E402


class _Env:
    def __init__(self, count: int):
        tracker = SimpleNamespace(
            count=count,
            completed=torch.zeros(count, dtype=torch.bool),
            failed=torch.zeros(count, dtype=torch.bool),
            progress=np.zeros(count),
            instruction=lambda index: f"task {index}",
        )
        tracker.clear_outcomes = lambda: TaskTracker.clear_outcomes(tracker)
        term = SimpleNamespace(tracker=tracker)
        self.unwrapped = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: term), num_envs=count)
        self.tracker = tracker
        self.resets = 0

    def reset(self, *args, **kwargs):
        self.resets += 1
        return {"policy": torch.zeros(self.tracker.count, 1)}, {}

    def step(self, action):
        return {"policy": torch.zeros(self.tracker.count, 1)}, 0.0, False, False, {}


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


def test_wrapper_has_no_raw_state_and_reports_progress_and_text():
    wrapped = wrap_for_eval(_Env(2))
    # A numpy raw state would switch eval-nav to a telemetry path this env does not feed.
    assert not hasattr(wrapped, "get_raw_state")
    assert wrapped.task_progress().shape == (2,)
    assert wrapped.instructions() == ["task 0", "task 1"]


def test_tasks_of_scene_names_the_benchmark_tasks_in_env_order():
    from franka_tabletop_rl.tabletop.bundle_path import resolve_bundle
    from franka_tabletop_rl.tasks.manager_based.hl_policy.task_state import (
        ALL_TASKS,
        benchmark_task_ids,
        tasks_of_scene,
    )

    ids = benchmark_task_ids(resolve_bundle())
    assert tasks_of_scene("franka-tabletop-rl-v0", ALL_TASKS) == ids
    assert tasks_of_scene("franka-tabletop-rl-v0", ids[0]) == [ids[0]]
    assert tasks_of_scene("franka-tabletop-rl-v0", "not_a_task") is None
