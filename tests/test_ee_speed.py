# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tool-speed trace: latch on auto-reset, drop on an explicit evaluator reset."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "source" / "franka_tabletop_rl"
sys.path.insert(0, str(PACKAGE))

from franka_tabletop_rl.tasks.hl_policy.ee_speed import EeSpeedTrace  # noqa: E402


def test_a_reset_latches_the_episode_and_ignores_the_teleport():
    trace = EeSpeedTrace(1)
    trace.observe([[0.0, 0.0, 0.0]], 0.1)
    trace.observe([[0.01, 0.0, 0.0]], 0.1)
    trace.reset([0])
    trace.observe([[1.0, 0.0, 0.0]], 0.1)
    assert trace.series()[0] == pytest.approx([0.1])
    trace.observe([[1.02, 0.0, 0.0]], 0.1)
    assert trace.series()[0] == pytest.approx([0.1])


def test_an_explicit_reset_discards_the_latch():
    trace = EeSpeedTrace(1)
    trace.observe([[0.0, 0.0, 0.0]], 0.1)
    trace.observe([[0.01, 0.0, 0.0]], 0.1)
    trace.reset([0])
    trace.discard()
    trace.reset([0])
    assert trace.series()[0].size == 0


def test_resetting_one_env_keeps_the_other_live():
    trace = EeSpeedTrace(2)
    origin = np.zeros((2, 3))
    trace.observe(origin, 0.1)
    trace.observe(np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]), 0.1)
    trace.reset([0])
    trace.observe(np.array([[5.0, 0.0, 0.0], [0.4, 0.0, 0.0]]), 0.1)
    assert trace.series()[0].tolist() == [1.0]
    assert trace.series()[1].tolist() == [2.0, 2.0]
