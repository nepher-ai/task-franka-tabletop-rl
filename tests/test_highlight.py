# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The play highlight grid is 16 tasks in a 1280 by 720 frame."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "source" / "franka_tabletop_rl"
sys.path.insert(0, str(PACKAGE))

from franka_tabletop_rl.tasks.hl_policy.highlight import (  # noqa: E402
    CELL_HEIGHT,
    CELL_WIDTH,
    HIGHLIGHT_COUNT,
    choose_tasks,
    episode_steps,
    highlight_requested,
    label_cell,
    tile_views,
)


def test_highlight_requested_matches_the_play_flag(monkeypatch: pytest.MonkeyPatch):
    assert highlight_requested(["play", "--highlight"])
    assert not highlight_requested(["play", "--video"])
    assert not highlight_requested(["play"])
    monkeypatch.setenv("FRANKA_TABLETOP_HIGHLIGHT", "1")
    assert highlight_requested()
    monkeypatch.setenv("FRANKA_TABLETOP_HIGHLIGHT", "0")
    assert not highlight_requested(["play"])


def test_choose_tasks_is_a_seeded_permutation_of_sixteen():
    names = [f"task_{index}" for index in range(20)]
    first = choose_tasks(names, HIGHLIGHT_COUNT, seed=1)
    assert first == choose_tasks(names, HIGHLIGHT_COUNT, seed=1)
    assert len(first) == len(set(first)) == HIGHLIGHT_COUNT
    assert set(first) <= set(names)
    assert choose_tasks(names, HIGHLIGHT_COUNT, seed=2) != first


def test_choose_tasks_rejects_a_short_stage():
    with pytest.raises(ValueError, match="16 distinct"):
        choose_tasks(["only_one"], HIGHLIGHT_COUNT, seed=1)


def test_episode_steps_matches_the_control_rate():
    assert episode_steps(40.0, 1.0 / 15.0) == 600


def test_highlight_cameras_render_at_the_cell_size():
    from franka_tabletop_rl.tabletop.core.sensors import at_resolution, front_camera

    full = front_camera()
    cell = at_resolution(full, CELL_HEIGHT, CELL_WIDTH)
    assert (full.height, full.width) == (720, 1280)
    assert (cell.height, cell.width) == (CELL_HEIGHT, CELL_WIDTH)


def test_label_cell_writes_the_task_id_on_the_bottom_strip():
    image = np.full((CELL_HEIGHT, CELL_WIDTH, 3), 180, dtype=np.uint8)
    labeled = label_cell(image, "apple_and_yogurt_in_bowl")
    assert labeled.shape == image.shape
    assert labeled.dtype == np.uint8
    assert np.array_equal(labeled[: CELL_HEIGHT // 2], image[: CELL_HEIGHT // 2])
    bottom = labeled[-28:]
    assert bottom.max() > 200
    assert (bottom == 24).any()


def test_tile_views_places_each_task_in_row_major_order():
    frames = np.zeros((HIGHLIGHT_COUNT, CELL_HEIGHT, CELL_WIDTH, 3), dtype=np.uint8)
    frames[5, 0, 0] = (9, 8, 7)
    canvas = tile_views(frames)
    assert canvas.shape == (720, 1280, 3)
    row, column = divmod(5, 4)
    assert tuple(canvas[row * CELL_HEIGHT, column * CELL_WIDTH]) == (9, 8, 7)
