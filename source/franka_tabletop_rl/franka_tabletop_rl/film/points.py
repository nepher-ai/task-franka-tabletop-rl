# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The point-cloud pass: the planner's view of the table, drawn on black from the film camera.

The planner works on per-object surface points, grasp candidates, and place goals. This pass draws exactly
that, at the live object poses, with the same camera pose and intrinsics as the RGB pass so the two can be
cross-faded. Nothing here renders through Kit: points are projected and splatted with NumPy.
"""

from __future__ import annotations

import colorsys
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from . import rig

BACKGROUND = (6, 8, 12)
TABLE_COLOR = (70, 86, 110)
ARM_COLOR = (150, 160, 175)
TOOL_COLOR = (80, 220, 255)
TARGET_COLOR = (255, 255, 255)
GRASP_COLOR = (255, 90, 70)
RELEASE_COLOR = (90, 255, 140)
GOAL_COLOR = (255, 215, 60)
NEAR_SHADE = 1.0
FAR_SHADE = 0.35
STAGES_GRASPING = ("PRE_GRASP", "DESCEND", "CLOSE")
STAGES_PLACING = ("CARRY", "LOWER", "OPEN")


def palette(count: int) -> list[tuple[int, int, int]]:
    """``count`` well-separated bright colors, stable for a given count."""
    colors = []
    for index in range(count):
        hue = (index * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 1.0)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


class Canvas:
    """An RGB ``uint8`` image that points, lines, and marks are drawn into."""

    def __init__(self, width: int, height: int, background: Sequence[int] = BACKGROUND):
        self.width = int(width)
        self.height = int(height)
        self.background = np.asarray(background, dtype=np.uint8)
        self.image = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self.clear()

    def clear(self) -> None:
        self.image[:] = self.background

    def splat(self, pixels: np.ndarray, depth: np.ndarray, colors: np.ndarray | Sequence[int], radius: int = 1) -> None:
        """Draw discs of ``radius`` pixels; the nearest point wins where discs overlap.

        Args:
            pixels: Pixel coordinates ``[N, 2]`` (u right, v down). Non-finite rows are skipped.
            depth: Depth [m] ``[N]`` that orders the discs.
            colors: One color, or ``[N, 3]`` colors, ``uint8``.
            radius: Disc radius [pixels]. Zero draws single pixels.
        """
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        depth = np.asarray(depth, dtype=np.float64).reshape(-1)
        colors = np.broadcast_to(np.asarray(colors, dtype=np.uint8), (len(pixels), 3))
        keep = np.isfinite(pixels).all(axis=1) & np.isfinite(depth)
        if not keep.any():
            return
        pixels, depth, colors = pixels[keep], depth[keep], colors[keep]
        offsets = [
            (dx, dy)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
            if dx * dx + dy * dy <= radius * radius + radius
        ]
        base_u = np.round(pixels[:, 0]).astype(np.int64)
        base_v = np.round(pixels[:, 1]).astype(np.int64)
        u = np.concatenate([base_u + dx for dx, _ in offsets])
        v = np.concatenate([base_v + dy for _, dy in offsets])
        z = np.tile(depth, len(offsets))
        rgb = np.tile(colors, (len(offsets), 1))
        inside = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        u, v, z, rgb = u[inside], v[inside], z[inside], rgb[inside]
        order = np.argsort(-z, kind="stable")
        self.image[v[order], u[order]] = rgb[order]

    def polyline(self, pixels: np.ndarray, color: Sequence[int], radius: int = 1, closed: bool = False) -> None:
        """Draw straight segments through ``pixels`` ``[N, 2]``. Segments with a missing end are skipped."""
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        if closed:
            pixels = np.vstack([pixels, pixels[:1]])
        start, end = pixels[:-1], pixels[1:]
        ok = np.isfinite(start).all(axis=1) & np.isfinite(end).all(axis=1)
        start, end = start[ok], end[ok]
        if len(start) == 0:
            return
        length = np.hypot(*(end - start).T)
        count = int(min(max(length.max(), 1.0), 4000.0)) + 1
        t = np.linspace(0.0, 1.0, count)[None, :, None]
        samples = (start[:, None, :] * (1.0 - t) + end[:, None, :] * t).reshape(-1, 2)
        self.splat(samples, np.zeros(len(samples)), color, radius)

    def circle(self, center: np.ndarray, radius_px: float, color: Sequence[int], thickness: int = 1) -> None:
        """Draw a circle outline around ``center`` (u, v). A missing center draws nothing."""
        if not np.isfinite(center).all():
            return
        angle = np.linspace(0.0, 2.0 * math.pi, 49)
        ring = np.stack([center[0] + radius_px * np.cos(angle), center[1] + radius_px * np.sin(angle)], axis=1)
        self.polyline(ring, color, thickness)

    def cross(self, center: np.ndarray, half_px: float, color: Sequence[int], thickness: int = 1) -> None:
        """Draw a plus sign around ``center`` (u, v)."""
        if not np.isfinite(center).all():
            return
        u, v = center
        self.polyline(np.array([[u - half_px, v], [u + half_px, v]]), color, thickness)
        self.polyline(np.array([[u, v - half_px], [u, v + half_px]]), color, thickness)


@dataclass
class Cloud:
    """One object's surface points [m] in the world frame."""

    points: np.ndarray
    color: tuple[int, int, int]
    root: np.ndarray
    is_current: bool = False


@dataclass
class PointFrame:
    """Everything the pass draws at one instant. All positions are world positions [m]."""

    clouds: list[Cloud] = field(default_factory=list)
    table: np.ndarray | None = None
    arm: np.ndarray | None = None
    tool: np.ndarray | None = None
    target: np.ndarray | None = None
    grasp: np.ndarray | None = None
    release: np.ndarray | None = None
    goals: list[np.ndarray] = field(default_factory=list)


def _shade(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    """Brightness in ``[FAR_SHADE, NEAR_SHADE]``, brighter when closer."""
    span = max(far - near, 1e-6)
    ratio = np.clip((far - depth) / span, 0.0, 1.0)
    return FAR_SHADE + (NEAR_SHADE - FAR_SHADE) * ratio


def draw_frame(
    canvas: Canvas,
    frame: PointFrame,
    eye: Sequence[float],
    rotation: np.ndarray,
    matrix: np.ndarray,
    point_radius: int = 1,
) -> np.ndarray:
    """Draw ``frame`` from a camera into ``canvas`` and return it.

    Args:
        canvas: Target image. It is cleared first.
        frame: What to draw.
        eye: Camera position [m].
        rotation: Camera to world matrix ``[3, 3]`` from :func:`rig.look_at_matrix`.
        matrix: Pinhole matrix ``[3, 3]`` [pixels] from :func:`rig.intrinsics`.
        point_radius: Disc radius [pixels] of surface points.
    """
    canvas.clear()
    fx = float(matrix[0, 0])

    def pix(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return rig.project(np.asarray(points, dtype=np.float64).reshape(-1, 3), eye, rotation, matrix)

    if frame.table is not None:
        canvas.polyline(pix(frame.table)[0], TABLE_COLOR, 1, closed=True)
    if frame.arm is not None:
        arm_px, _ = pix(frame.arm)
        canvas.polyline(arm_px, ARM_COLOR, 2)
        canvas.splat(arm_px, np.zeros(len(arm_px)), ARM_COLOR, 3)

    if frame.clouds:
        pixels, depths, colors = [], [], []
        for cloud in frame.clouds:
            px, depth = pix(cloud.points)
            base = np.asarray(cloud.color, dtype=np.float64)
            if cloud.is_current:
                base = np.minimum(base + 60.0, 255.0)
            pixels.append(px)
            depths.append(depth)
            colors.append(np.broadcast_to(base, (len(px), 3)))
        pixels, depths, colors = np.vstack(pixels), np.concatenate(depths), np.vstack(colors)
        seen = np.isfinite(depths) & (depths > 0)
        if seen.any():
            shade = _shade(depths, float(depths[seen].min()), float(depths[seen].max()))
            colors = np.clip(colors * shade[:, None], 0, 255).astype(np.uint8)
            canvas.splat(pixels, depths, colors, point_radius)

    def ring(position: np.ndarray, radius_m: float, color: Sequence[int], thickness: int = 2) -> None:
        center, depth = pix(position)
        if np.isfinite(depth[0]) and depth[0] > 0:
            canvas.circle(center[0], fx * radius_m / depth[0], color, thickness)

    def cross(position: np.ndarray, half_m: float, color: Sequence[int], thickness: int = 2) -> None:
        center, depth = pix(position)
        if np.isfinite(depth[0]) and depth[0] > 0:
            canvas.cross(center[0], fx * half_m / depth[0], color, thickness)

    for cloud in frame.clouds:
        ring(cloud.root, 0.025, cloud.color, 2 if cloud.is_current else 1)
    for position in frame.goals:
        ring(position, 0.05, GOAL_COLOR, 2)
    if frame.grasp is not None:
        cross(frame.grasp, 0.03, GRASP_COLOR)
    if frame.release is not None:
        cross(frame.release, 0.03, RELEASE_COLOR)
    if frame.target is not None:
        ring(frame.target, 0.02, TARGET_COLOR, 1)
    if frame.tool is not None:
        ring(frame.tool, 0.015, TOOL_COLOR, 2)
    return canvas.image


def _to_numpy(values) -> np.ndarray:
    values = values.torch if hasattr(values, "torch") else values
    return values.detach().cpu().numpy()


class PlannerPointPass:
    """Builds a :class:`PointFrame` from one env of a running film env.

    Args:
        env: The unwrapped env. It needs the ``ee_pose`` command term, which owns the tracker and planner.
        env_index: The env to draw.
        stride: Keep every ``stride``-th surface point of the planner cloud.
    """

    def __init__(self, env, env_index: int = 0, stride: int = 3):
        from franka_tabletop_rl.frames import EE_BODY, TCP_OFFSET_B

        self.env = env
        self.index = int(env_index)
        self.stride = max(int(stride), 1)
        self.term = env.command_manager.get_term("ee_pose")
        self.tracker = self.term.tracker
        self.planner = self.term.planner
        self.robot = env.scene["robot"]
        self.tool_body = self.robot.body_names.index(EE_BODY)
        self.tool_offset = np.asarray(TCP_OFFSET_B, dtype=np.float64)
        links = [
            (int(m.group(1)), i)
            for i, n in enumerate(self.robot.body_names)
            if (m := re.fullmatch(r"panda_link(\d)", n))
        ]
        self.arm_bodies = [i for _, i in sorted(links)] + [self.tool_body]
        self._colors: dict[str, tuple[int, int, int]] = {}

    def _color(self, slot: str, slots: Sequence[str]) -> tuple[int, int, int]:
        if slot not in self._colors:
            for name, color in zip(slots, palette(len(slots)), strict=True):
                self._colors[name] = color
        return self._colors[slot]

    def tool_position(self) -> np.ndarray:
        """World position [m] of the tool point of the filmed env."""
        k = self.index
        pos = _to_numpy(self.robot.data.body_pos_w)[k, self.tool_body].astype(np.float64)
        quat = _to_numpy(self.robot.data.body_quat_w)[k, self.tool_body]
        return pos + rig.quat_xyzw_to_matrix(quat) @ self.tool_offset

    def build(self) -> PointFrame:
        """Snapshot the live scene of the filmed env."""
        k = self.index
        tracker, planner = self.tracker, self.planner
        task = int(tracker.task_index[k])
        origin = np.asarray(tracker.origin(k), dtype=np.float64)
        layout = tracker.env_layouts[k]
        slots = [p.slot for p in layout.placements if p.asset.kind != "anchor"]
        context = planner.contexts[k]
        stage = context.stage.name
        current = None
        if 0 <= context.current < len(context.goals):
            current = context.goals[context.current]

        frame = PointFrame()
        roots: dict[str, np.ndarray] = {}
        for slot in slots:
            cloud = planner.points[task].get(slot)
            if cloud is None:
                continue
            entity = self.env.scene[tracker.entity(k, slot)]
            pos = _to_numpy(entity.data.root_pos_w)[k].astype(np.float64)
            quat = _to_numpy(entity.data.root_quat_w)[k]
            roots[slot] = pos
            world = np.asarray(cloud[:: self.stride], dtype=np.float64) @ rig.quat_xyzw_to_matrix(quat).T + pos
            frame.clouds.append(
                Cloud(world, self._color(slot, slots), pos, is_current=current is not None and slot == current.slot)
            )

        tables = planner.tables
        if tables:
            center, half = tables[task]
            sx, sy = float(half[0]), float(half[1])
            corners = np.array([[-sx, -sy], [sx, -sy], [sx, sy], [-sx, sy]], dtype=np.float64)
            frame.table = np.column_stack([corners + center[:2], np.full(4, float(center[2]))]) + origin

        body_pos = _to_numpy(self.robot.data.body_pos_w)[k].astype(np.float64)
        frame.arm = body_pos[self.arm_bodies]
        frame.tool = self.tool_position()
        frame.target = _to_numpy(self.term.pose_command_b)[k, :3].astype(np.float64) + origin
        if stage in STAGES_GRASPING:
            frame.grasp = context.grasp_center + origin
        if stage in STAGES_PLACING:
            frame.release = context.release + origin
        goal_slot = getattr(current, "target", "") if current is not None else ""
        if goal_slot and goal_slot in roots:
            frame.goals.append(roots[goal_slot])
        return frame
