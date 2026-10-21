# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Highlight videos for play.

``--highlight`` on a high-level play task records one mp4 per observation camera. Each frame is a
4 by 4 grid of 16 tasks, 1280 by 720 [pixels]. The clip lasts one episode
(``episode_length_s``, the max episode time). The files are
``best_policy/videos/highlight/<camera>.mp4``, separate from the LeRobot dataset.
"""

from __future__ import annotations

import os
import random
import sys
from collections.abc import Sequence

import numpy as np

from franka_tabletop_rl.paths import play_video_dir
from franka_tabletop_rl.tabletop.core.observation_spec import OBSERVATION_CAMERAS

# Set by ``isaaclab play --highlight`` before Kit starts, because that flag is then removed from argv.
HIGHLIGHT_ENV = "FRANKA_TABLETOP_HIGHLIGHT"

# Four columns and four rows. Each cell keeps the camera's 16:9 frame.
HIGHLIGHT_COUNT = 16
GRID = 4
CELL_HEIGHT = 180
CELL_WIDTH = 320


def highlight_requested(argv: Sequence[str] | None = None) -> bool:
    """True when this process was started with ``--highlight``.

    ``argv`` overrides the process command line. The play entrypoint also sets
    :data:`HIGHLIGHT_ENV`, because it removes the flag from ``sys.argv`` before the env is built.
    """
    if argv is None and os.environ.get(HIGHLIGHT_ENV) == "1":
        return True
    return "--highlight" in (sys.argv if argv is None else argv)


def choose_tasks(task_ids: Sequence[str], count: int, seed: int) -> tuple[str, ...]:
    """``count`` distinct tasks in a random order fixed by ``seed``.

    Raises:
        ValueError: The stage has fewer than ``count`` distinct tasks.
    """
    unique = list(dict.fromkeys(task_ids))
    if len(unique) < count:
        raise ValueError(
            f"--highlight needs {count} distinct tasks for the highlight grid, the stage has {len(unique)}"
        )
    return tuple(random.Random(seed).sample(unique, count))


def episode_steps(episode_length_s: float, step_dt: float) -> int:
    """Control steps in one episode. ``step_dt`` is one control step [s]."""
    if step_dt <= 0:
        raise ValueError(f"step_dt must be positive, got {step_dt}")
    return max(1, round(episode_length_s / step_dt))


def as_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    """RGB uint8 view of one image. Float images in ``[0, 1]`` are scaled to ``[0, 255]``."""
    image = np.ascontiguousarray(frame[..., :3])
    if np.issubdtype(image.dtype, np.floating):
        if image.size and float(image.max()) <= 1.0:
            image = image * 255.0
        image = np.rint(image)
    return image.astype(np.uint8, copy=False)


def label_cell(image: np.ndarray, task_id: str) -> np.ndarray:
    """Copy of one cell with ``task_id`` in white on a dark strip along the bottom."""
    import cv2

    canvas = np.array(image, copy=True)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    margin = 4
    scale = 0.38
    (text_w, text_h), baseline = cv2.getTextSize(task_id, font, scale, thickness)
    room = canvas.shape[1] - 2 * margin
    if text_w > room and text_w > 0:
        scale *= room / text_w
        (text_w, text_h), baseline = cv2.getTextSize(task_id, font, scale, thickness)
    bar = text_h + baseline + 2 * margin
    canvas[-bar:, :] = (24, 24, 24)
    origin = (margin, canvas.shape[0] - baseline - margin)
    cv2.putText(canvas, task_id, origin, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return canvas


def tile_views(frames: np.ndarray) -> np.ndarray:
    """Pack ``HIGHLIGHT_COUNT`` cell images into one 1280 by 720 [pixels] frame.

    ``frames`` is ``[16, height, width, 3]``. Cells are placed row by row.

    Raises:
        ValueError: The batch is not 16 images of the cell size.
    """
    if frames.shape != (HIGHLIGHT_COUNT, CELL_HEIGHT, CELL_WIDTH, 3):
        raise ValueError(
            f"highlight frames must be {[HIGHLIGHT_COUNT, CELL_HEIGHT, CELL_WIDTH, 3]}, got {list(frames.shape)}"
        )
    canvas = np.zeros((GRID * CELL_HEIGHT, GRID * CELL_WIDTH, 3), dtype=np.uint8)
    for index in range(HIGHLIGHT_COUNT):
        row, column = divmod(index, GRID)
        top, left = row * CELL_HEIGHT, column * CELL_WIDTH
        canvas[top : top + CELL_HEIGHT, left : left + CELL_WIDTH] = frames[index]
    return canvas


def apply_highlight(cfg) -> None:
    """Select 16 tasks, spawn every observation camera, and record one grid video each.

    ``cfg`` is a high-level env cfg. The stage must already name at least 16 tasks. The clip
    lasts ``episode_length_s`` [s], the max episode time.
    """
    cfg.task_ids = choose_tasks(cfg.task_ids, HIGHLIGHT_COUNT, int(cfg.seed or 0))
    cfg.scene.num_envs = HIGHLIGHT_COUNT
    cfg.commands.ee_pose.task_ids = cfg.task_ids
    _enable_cameras(cfg)
    step_dt = float(cfg.sim.dt) * int(cfg.decimation)
    cfg.video_recorders = highlight_recorders(float(cfg.episode_length_s), step_dt)
    names = ", ".join(cfg.task_ids)
    print(
        f"[INFO] Highlight: {HIGHLIGHT_COUNT} tasks for {cfg.episode_length_s} s, "
        f"videos in {play_video_dir()}\n[INFO] Highlight tasks: {names}"
    )


def _enable_cameras(cfg) -> None:
    """Put every observation camera back on the scene. Play drops them unless recording."""
    from franka_tabletop_rl.tabletop.core import sensors

    makers = {
        "cam_exterior_left": sensors.exterior_left_camera,
        "cam_exterior_right": sensors.exterior_right_camera,
        "cam_wrist": sensors.wrist_camera,
        "cam_front": sensors.front_camera,
    }
    # The saved frame is the 4 by 4 grid. Rendering the full 1280 by 720 per env is discarded.
    for name in OBSERVATION_CAMERAS:
        camera = sensors.at_resolution(makers[name](), CELL_HEIGHT, CELL_WIDTH)
        setattr(cfg.scene, name, camera)
    cfg.cameras = OBSERVATION_CAMERAS


def highlight_recorders(episode_length_s: float, step_dt: float) -> list:
    """One recorder per observation camera, lasting one episode."""
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
