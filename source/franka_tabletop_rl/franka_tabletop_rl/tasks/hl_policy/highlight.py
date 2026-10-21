# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Highlight videos for play.

``env.highlight=on`` on a high-level play task records one mp4 per observation camera. Each frame is a
4 by 4 grid of up to 16 tasks, 1280 by 720 [pixels]. The clip lasts one episode
(``episode_length_s``, the max episode time). The files are
``best_policy/videos/highlight/<camera>.mp4``, separate from the LeRobot dataset.

Highlight only adds recording. The play stage keeps its tasks, env count, and layouts. The
highlighted tasks are a subset of the stage, shown in benchmark order, and only their envs are
rendered.
"""

from __future__ import annotations

import random

import numpy as np

from franka_tabletop_rl.paths import play_video_dir
from franka_tabletop_rl.tabletop.core.observation_spec import OBSERVATION_CAMERAS

# Four columns and four rows. Each cell keeps the camera's 16:9 frame.
HIGHLIGHT_COUNT = 16
GRID = 4
CELL_HEIGHT = 180
CELL_WIDTH = 320


def highlight_envs(task_count: int, num_envs: int, seed: int, count: int = HIGHLIGHT_COUNT) -> tuple[int, ...]:
    """Env indices to film, one per highlighted task, in benchmark order.

    Env ``i`` of the stage runs task ``i % task_count``, so the first env of task ``k`` is env ``k``.
    A stage with at most ``count`` tasks highlights all of them. A larger stage highlights ``count``
    tasks sampled with ``seed``, then sorted back into benchmark order.

    Args:
        task_count: Number of tasks ``T`` in the stage.
        num_envs: Number of envs in the stage.
        seed: Seed of the task sample.
        count: Most tasks to highlight.

    Raises:
        ValueError: The stage has no task or fewer envs than tasks.
    """
    if task_count <= 0 or num_envs < task_count:
        raise ValueError(f"highlight needs at least one env per task, got {num_envs} envs for {task_count} tasks")
    if task_count <= count:
        return tuple(range(task_count))
    return tuple(sorted(random.Random(seed).sample(range(task_count), count)))


def episode_steps(episode_length_s: float, step_dt: float) -> int:
    """Control steps in one episode. ``step_dt`` is one control step [s]."""
    if step_dt <= 0:
        raise ValueError(f"step_dt must be positive, got {step_dt}")
    return max(1, round(episode_length_s / step_dt))


def tile_views(frames: np.ndarray) -> np.ndarray:
    """Pack up to ``HIGHLIGHT_COUNT`` cell images into one 1280 by 720 [pixels] frame.

    ``frames`` is ``[n, height, width, 3]`` uint8 with ``n <= 16``. Cells are placed row by row. Unused
    cells stay black.

    Raises:
        ValueError: The batch is empty, larger than the grid, or not of the cell size.
    """
    count = frames.shape[0] if frames.ndim == 4 else 0
    if frames.ndim != 4 or not 0 < count <= HIGHLIGHT_COUNT or frames.shape[1:] != (CELL_HEIGHT, CELL_WIDTH, 3):
        raise ValueError(
            f"highlight frames must be [1..{HIGHLIGHT_COUNT}, {CELL_HEIGHT}, {CELL_WIDTH}, 3], got {list(frames.shape)}"
        )
    cells = np.zeros((HIGHLIGHT_COUNT, CELL_HEIGHT, CELL_WIDTH, 3), dtype=np.uint8)
    cells[:count] = frames
    rows = cells.reshape(GRID, GRID, CELL_HEIGHT, CELL_WIDTH, 3).transpose(0, 2, 1, 3, 4)
    return np.ascontiguousarray(rows.reshape(GRID * CELL_HEIGHT, GRID * CELL_WIDTH, 3))


def highlight_recorders(episode_length_s: float, step_dt: float) -> list:
    """One recorder cfg per observation camera, lasting one episode."""
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    steps = episode_steps(episode_length_s, step_dt)
    directory = str(play_video_dir())
    fps = max(1, round(1.0 / step_dt))
    return [
        VideoRecorderCfg(
            source=f"sensor:{name}",
            output_dir=directory,
            output_filename_prefix=name,
            video_length=steps,
            video_interval=0,
            fps=fps,
        )
        for name in OBSERVATION_CAMERAS
    ]
