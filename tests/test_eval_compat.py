# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The evaluator reads outcome flags as tensors and the task each env finished."""

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
    """Parsing the play config must not import OpenUSD.

    ``isaaclab.bat play`` builds this config before Kit starts. Importing the Kit Python
    OpenUSD first makes Kit's own USD extensions fail to load on Windows.
    """
    script = (
        "import sys\n"
        "import franka_tabletop_rl.tasks.hl_policy.hl_env_cfg\n"
        "franka_tabletop_rl.tasks.hl_policy.hl_env_cfg.HLEnvCfg_PLAY()\n"
        "sys.exit(1 if 'pxr' in sys.modules else 0)\n"
    )
    completed = subprocess.run([sys.executable, "-c", script], check=False)
    assert completed.returncode == 0
