# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wireframe previews of a shot, drawn from the benchmark layout with the same camera math as the film.

This is a framing aid that needs no simulator: it draws the work surface, the task's objects as boxes, and a
schematic arm. It shows where the lens looks and how much of the table fits, not what the render looks like.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import rig
from .points import Canvas, palette
from .shots import PointResolver, Shot

TABLE_COLOR = (90, 110, 140)
ARM_COLOR = (200, 200, 210)
GRID_SPACING_M = 2.5
GRID_SIDE = 4
_EDGES = (
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)  # fmt: skip


@dataclass(frozen=True)
class Box:
    """An oriented box: center [m], half sizes [m], and a rotation matrix ``[3, 3]``."""

    center: np.ndarray
    half: np.ndarray
    rotation: np.ndarray
    color: tuple[int, int, int]

    def corners(self) -> np.ndarray:
        signs = np.array([[x, y, z] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)], dtype=np.float64)
        return self.center + (signs * self.half) @ self.rotation.T


def _rotation_from_wxyz(quat: Sequence[float]) -> np.ndarray:
    w, x, y, z = quat
    return rig.quat_xyzw_to_matrix((x, y, z, w))


def layout_boxes(layout, stage: dict) -> list[Box]:
    """Boxes of the work surface and of every placed object of a benchmark task layout."""
    boxes = []
    record = next((s for s in stage.get("surfaces", ()) if s.get("name") == layout.surface), None)
    if record:
        half = np.asarray(record.get("table_m") or (0.7, 1.0, 0.04), dtype=np.float64) / 2.0
        cx, cy = record.get("center_xy", (0.547, 0.0))
        top = float(record.get("table_z_m", 0.0))
        boxes.append(Box(np.array([cx, cy, top - half[2]]), half, np.eye(3), TABLE_COLOR))
    items = [p for p in layout.placements if p.asset.kind != "anchor"]
    for color, placement in zip(palette(len(items)), items, strict=True):
        half = np.asarray(placement.asset.size_m, dtype=np.float64) / 2.0
        center = np.asarray(placement.position_m, dtype=np.float64)
        boxes.append(Box(center, half, _rotation_from_wxyz(placement.quat_wxyz), color))
    return boxes


def arm_schematic(tool: np.ndarray) -> np.ndarray:
    """A rough Franka silhouette from the base to ``tool``, for scale only."""
    tool = np.asarray(tool, dtype=np.float64)
    elbow = np.array([0.5 * tool[0] + 0.05, 0.5 * tool[1], max(tool[2] + 0.45, 0.62)])
    return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.333], elbow, tool + (0.0, 0.0, 0.13), tool])


def _draw_box(canvas: Canvas, box: Box, eye, rotation, matrix) -> None:
    pixels, _ = rig.project(box.corners(), eye, rotation, matrix)
    for a, b in _EDGES:
        canvas.polyline(pixels[[a, b]], box.color, 0)


def render_view(
    canvas: Canvas, boxes: Sequence[Box], arm: np.ndarray, eye, target, hfov_deg: float, offsets: Sequence[np.ndarray]
) -> np.ndarray:
    """Draw the scene once per env offset [m] from a camera and return the image."""
    canvas.clear()
    rotation = rig.look_at_matrix(eye, target)
    matrix = rig.intrinsics(canvas.width, canvas.height, hfov_deg)
    for offset in offsets:
        for box in boxes:
            _draw_box(canvas, Box(box.center + offset, box.half, box.rotation, box.color), eye, rotation, matrix)
        pixels, _ = rig.project(arm + offset, eye, rotation, matrix)
        canvas.polyline(pixels, ARM_COLOR, 1)
    # horizon-free guides: the image center and the rule-of-thirds lines
    for fraction in (1 / 3, 2 / 3):
        canvas.polyline(
            np.array([[canvas.width * fraction, 0], [canvas.width * fraction, canvas.height]]), (40, 46, 56), 0
        )
        canvas.polyline(
            np.array([[0, canvas.height * fraction], [canvas.width, canvas.height * fraction]]), (40, 46, 56), 0
        )
    return canvas.image.copy()


def grid_offsets(count: int) -> list[np.ndarray]:
    """Env origins [m] of ``count`` envs on a grid, with the centroid at the origin."""
    side = GRID_SIDE if count > 1 else 1
    cells = [np.array([i % side * GRID_SPACING_M, i // side * GRID_SPACING_M, 0.0]) for i in range(count)]
    centre = np.mean(cells, axis=0)
    centre[2] = 0.0
    return [cell - centre for cell in cells]


def shot_frames(
    shot: Shot,
    boxes: Sequence[Box],
    tool: np.ndarray,
    times: Sequence[float],
    size: tuple[int, int] = (960, 540),
    env_count: int = 16,
) -> list[np.ndarray]:
    """Wireframe images of ``shot`` at take times ``times`` [s].

    ``tool`` is where the gripper is for shots that follow it. Shots that show the whole grid draw
    ``env_count`` copies of the scene.
    """
    canvas = Canvas(*size)
    resolver = PointResolver(np.zeros(3), tool=lambda: np.asarray(tool, dtype=np.float64))
    offsets = grid_offsets(env_count) if shot.all_envs else [np.zeros(3)]
    arm = arm_schematic(tool)
    frames = []
    for t in times:
        eye, target = rig.evaluate_path(shot.keys, t, resolver, shot.easing)
        frames.append(render_view(canvas, boxes, arm, eye, target, shot.hfov_deg, offsets))
    return frames


def contact_sheet(frames: Sequence[np.ndarray], columns: int = 2) -> np.ndarray:
    """Tile equally sized frames into one image, row by row."""
    height, width, _ = frames[0].shape
    rows = -(-len(frames) // columns)
    sheet = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        sheet[row * height : (row + 1) * height, column * width : (column + 1) * width] = frame
    return sheet
