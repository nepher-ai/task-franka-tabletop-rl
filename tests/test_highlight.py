# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The play highlight grid shows up to 16 tasks of the stage in a 1280 by 720 frame."""

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
    episode_steps,
    highlight_envs,
    tile_views,
)


def test_highlight_stays_off_until_it_is_selected():
    from franka_tabletop_rl.tasks.hl_policy.highlight import HighlightCfg, HighlightPreset
    from franka_tabletop_rl.tasks.hl_policy.hl_env_cfg import HLEnvCfg_PLAY

    cfg = HLEnvCfg_PLAY()
    assert isinstance(cfg.highlight, HighlightPreset)
    cfg.play_mode()
    assert cfg.highlight is None
    assert not hasattr(cfg, "rendering_camera")
    assert cfg.scene.num_envs == len(cfg.task_ids)

    cfg = HLEnvCfg_PLAY()
    cfg.highlight = cfg.highlight.on
    cfg.play_mode()
    assert isinstance(cfg.highlight, HighlightCfg)
    assert (cfg.rendering_camera.height, cfg.rendering_camera.width) == (CELL_HEIGHT, CELL_WIDTH)
    assert cfg.scene.num_envs == len(cfg.task_ids)


def test_a_large_stage_is_sampled_with_the_seed_and_shown_in_benchmark_order():
    first = highlight_envs(40, 80, seed=1)
    assert first == highlight_envs(40, 80, seed=1)
    assert len(first) == len(set(first)) == HIGHLIGHT_COUNT
    assert list(first) == sorted(first)
    assert all(0 <= index < 40 for index in first)
    assert highlight_envs(40, 80, seed=2) != first


def test_highlighted_envs_run_distinct_tasks_without_adding_envs():
    task_count, num_envs = 24, 96
    envs = highlight_envs(task_count, num_envs, seed=3)
    assert all(index < num_envs for index in envs)
    assert len({index % task_count for index in envs}) == len(envs)


@pytest.mark.parametrize("task_count", [1, 5, HIGHLIGHT_COUNT])
def test_a_short_stage_highlights_every_task(task_count: int):
    assert highlight_envs(task_count, task_count * 2, seed=7) == tuple(range(task_count))


def test_highlight_needs_an_env_for_every_task():
    with pytest.raises(ValueError, match="at least one env per task"):
        highlight_envs(20, 10, seed=1)


def test_highlight_tasks_keeps_the_given_indices_in_that_order():
    assert highlight_envs(40, 80, seed=1, tasks=(12, 0, 3)) == (12, 0, 3)
    assert highlight_envs(40, 80, seed=1, tasks=[2]) == (2,)
    assert highlight_envs(40, 80, seed=1, tasks=4) == (4,)
    assert highlight_envs(5, 10, seed=7, tasks=(4, 1)) == (4, 1)
    assert highlight_envs(40, 80, seed=1, tasks="0-15") == tuple(range(16))
    assert highlight_envs(40, 80, seed=1, tasks="12,0,3") == (12, 0, 3)
    assert highlight_envs(40, 80, seed=1, tasks="0-2,5") == (0, 1, 2, 5)
    assert highlight_envs(40, 80, seed=1, tasks="3-1") == (3, 2, 1)


def test_an_empty_highlight_task_list_keeps_the_seeded_sample():
    assert highlight_envs(40, 80, seed=1, tasks=()) == highlight_envs(40, 80, seed=1)
    assert highlight_envs(40, 80, seed=1, tasks=None) == highlight_envs(40, 80, seed=1)
    assert highlight_envs(5, 10, seed=7, tasks=()) == tuple(range(5))


def test_highlight_tasks_must_fit_the_stage_and_the_grid():
    with pytest.raises(ValueError, match="at most"):
        highlight_envs(40, 80, seed=1, tasks=tuple(range(HIGHLIGHT_COUNT + 1)))
    with pytest.raises(ValueError, match="unique"):
        highlight_envs(40, 80, seed=1, tasks=(1, 1))
    with pytest.raises(ValueError, match=r"\[0, 40\)"):
        highlight_envs(40, 80, seed=1, tasks=(40,))
    with pytest.raises(ValueError, match="integers or ranges"):
        highlight_envs(40, 80, seed=1, tasks="0-")
    with pytest.raises(ValueError, match="integers or ranges"):
        highlight_envs(40, 80, seed=1, tasks=(1, "no"))
    with pytest.raises(ValueError, match="at least one env per task"):
        highlight_envs(20, 10, seed=1, tasks=(0,))


def test_episode_steps_matches_the_control_rate():
    assert episode_steps(40.0, 1.0 / 15.0) == 600


def test_tile_views_places_each_task_in_row_major_order():
    frames = np.zeros((HIGHLIGHT_COUNT, CELL_HEIGHT, CELL_WIDTH, 3), dtype=np.uint8)
    frames[5, 0, 0] = (9, 8, 7)
    canvas = tile_views(frames)
    assert canvas.shape == (720, 1280, 3)
    row, column = divmod(5, 4)
    assert tuple(canvas[row * CELL_HEIGHT, column * CELL_WIDTH]) == (9, 8, 7)


def test_tile_views_leaves_unused_cells_black():
    frames = np.full((3, CELL_HEIGHT, CELL_WIDTH, 3), 200, dtype=np.uint8)
    canvas = tile_views(frames)
    assert canvas.shape == (720, 1280, 3)
    assert canvas[:CELL_HEIGHT, : 3 * CELL_WIDTH].min() == 200
    assert canvas[:CELL_HEIGHT, 3 * CELL_WIDTH :].max() == 0
    assert canvas[CELL_HEIGHT:].max() == 0


def test_tile_views_rejects_a_batch_that_does_not_fit_the_grid():
    with pytest.raises(ValueError, match="highlight frames"):
        tile_views(np.zeros((HIGHLIGHT_COUNT + 1, CELL_HEIGHT, CELL_WIDTH, 3), dtype=np.uint8))
