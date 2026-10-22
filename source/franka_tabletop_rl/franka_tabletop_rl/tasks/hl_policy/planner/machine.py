# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Classical pick-and-place planner: the high level of the hierarchical policy.

Per env it picks each object a task asks to move, carries it to the pose that satisfies that
goal, lets go, and retreats, then moves to the next object. A task that must be left upright is
left alone. It emits a tool position, a tool yaw and a gripper state at the control rate.

The planner reads object and tool poses from the env and uses no privileged action: every output
is a target that the tracker has to reach on its own.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from franka_tabletop_rl.benchmark.predicates import latch_key
from franka_tabletop_rl.benchmark.spec import Goal
from franka_tabletop_rl.benchmark.stage import resolve_surface
from franka_tabletop_rl.frames import MIN_REACH_X_M, REACH_HIGH_M, REACH_LOW_M, REACH_Z_HIGH_M, REACH_Z_LOW_M

from ..tracker import TaskTracker
from .goals import HoldGoal, PlaceGoal, compile_goals
from .grasp import TOOL_Z_MIN_M, Grasp, fallback_grasp, find_grasps, quat_wxyz_to_matrix, sample_prim_points, tool_yaw
from .targets import place_center

# Carrot speed limits: the tracker was trained on steps of up to 12 cm, and slower is more precise.
STEP_M = 0.05
STEP_YAW_RAD = 0.3
POS_TOL_M = 0.012
Z_TOL_M = 0.008
TRAVEL_Z_M = 0.24
SAFE_Z_M = 0.30
HOME = np.array([0.45, 0.0, 0.26])
CARRY_CLEARANCE_M = 0.09
RELEASE_CLEARANCE_M = 0.03
LIFT_MIN_M = 0.04
MISSED_CLOSURE = 0.92
MAX_ATTEMPTS = 6
TILT_RESEARCH_RAD = 0.3
# Oak box used when a stage does not name a work surface [m].
_OAK_CENTER = np.array([0.547, 0.0, 0.0])
_OAK_HALF = np.array([0.35, 0.5, 0.02])


def reach_limit(xy: np.ndarray, z: float) -> np.ndarray:
    """Pull a horizontal tool position [m] inside the tracked reach at height `z` [m].

    The radius and the minimum x match :func:`franka_tabletop_rl.frames.project_to_reach`.
    """
    blend = float(np.clip((z - REACH_Z_LOW_M) / (REACH_Z_HIGH_M - REACH_Z_LOW_M), 0.0, 1.0))
    radius_max = REACH_LOW_M + (REACH_HIGH_M - REACH_LOW_M) * blend
    limited = np.asarray(xy, dtype=np.float64).copy()
    limited *= min(radius_max / max(float(np.hypot(*limited)), 1e-6), 1.0)
    limited[0] = max(limited[0], MIN_REACH_X_M)
    return limited


def table_box(stage: dict | None, layout) -> tuple[np.ndarray, np.ndarray]:
    """Work-surface center [m] and half sizes [m] for one task.

    A task ``surface_pose`` overrides the stage center. Without a stage, the oak box is used.
    """
    if not stage:
        return _OAK_CENTER.copy(), _OAK_HALF.copy()
    record = resolve_surface(stage, layout.surface)
    half = np.asarray(record.get("table_m") or (0.7, 1.0, 0.04), dtype=np.float64) / 2.0
    table_z = float(record.get("table_z_m", 0.0))
    pose = layout.surface_pose if isinstance(getattr(layout, "surface_pose", None), dict) else None
    if pose and pose.get("pos"):
        pos = pose["pos"]
        center = np.array([float(pos[0]), float(pos[1]), table_z])
    else:
        raw = record.get("center_xy") or tuple(record.get("pos", (0.547, 0.0)))[:2]
        center = np.array([float(raw[0]), float(raw[1]), table_z])
    return center, half


def _without_redundant_resting(goals: list) -> list:
    """Drop ``resting`` when that slot is also taken outside a container or off a support.

    The one placement sets the object on the table, which is what ``resting`` scores.
    """
    moved = {goal.slot for goal in goals if isinstance(goal, PlaceGoal) and goal.op in ("outside", "off")}
    return [
        goal
        for goal in goals
        if not (isinstance(goal, PlaceGoal) and goal.op == "resting" and goal.slot in moved)
    ]


class Stage(IntEnum):
    INIT = 0
    PRE_GRASP = 1
    DESCEND = 2
    CLOSE = 3
    LIFT = 4
    CARRY = 5
    LOWER = 6
    OPEN = 7
    RETRACT = 8
    RECOVER = 9
    DONE = 10


@dataclass
class _Context:
    stage: Stage = Stage.INIT
    age: int = 0
    timer: int = 0
    pos: np.ndarray = field(default_factory=lambda: HOME.copy())
    yaw: float = 0.0
    grip: float = 0.0
    target: np.ndarray = field(default_factory=lambda: HOME.copy())
    target_yaw: float = 0.0
    todo: list[int] = field(default_factory=list)
    current: int = -1
    attempts: dict[int, int] = field(default_factory=dict)
    candidates: list[Grasp] = field(default_factory=list)
    candidate_index: int = 0
    grasp_center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    grasp_yaw: float = 0.0
    carry_z: float = TRAVEL_Z_M
    release: np.ndarray = field(default_factory=lambda: np.zeros(3))
    below: float = 0.05
    previous_closure: float = 0.0
    rest_z: float = 0.0
    goals: list = field(default_factory=list)
    offsets: dict = field(default_factory=dict)
    ordered: bool = False
    supported: bool = True
    keep_grip: bool = False


def drop_offsets(count: int, half_x: float, half_y: float, margin: float = 0.05) -> list[tuple[float, float]]:
    """Cell centers [m] of a grid inside a container's footprint, in the container frame."""
    inner_x, inner_y = max(half_x - margin, 0.0), max(half_y - margin, 0.0)
    if count <= 1:
        return [(0.0, 0.0)]
    columns = max(1, min(count, round(math.sqrt(count * max(inner_x, 1e-3) / max(inner_y, 1e-3)))))
    rows = math.ceil(count / columns)
    offsets = []
    for index in range(count):
        row, column = divmod(index, columns)
        x = 0.0 if columns == 1 else -inner_x + 2.0 * inner_x * column / (columns - 1)
        y = 0.0 if rows == 1 else -inner_y + 2.0 * inner_y * row / (rows - 1)
        offsets.append((x, y))
    return offsets


class ClassicalTabletopPlanner:
    """Vectorized over envs by a per-env state machine.

    Args:
        tracker: Scene layout and scoring of the envs. Env ``i`` plans ``tracker.task_index[i]``.
        points: Per task, the surface points [m] of each slot in its root frame, ``[{slot: [M, 3]}]``.
        rest: Per task, the root pose of each slot at the published layout,
            ``[{slot: (position, wxyz quaternion)}]``.
    """

    def __init__(
        self,
        tracker: TaskTracker,
        points: Sequence[dict[str, np.ndarray]],
        rest: Sequence[dict[str, tuple]],
    ):
        self.tracker = tracker
        self.points = points
        self.rest = rest
        self.num_envs = tracker.num_envs
        self.count = self.num_envs
        self.contexts = [_Context() for _ in range(self.num_envs)]
        self._grasps: dict[tuple[int, str], list[Grasp]] = {}
        self._warned: set[str] = set()
        self.tables: list[tuple[np.ndarray, np.ndarray]] | None = None

    # ------------------------------------------------------------------ setup

    def _task(self, index: int) -> int:
        return int(self.tracker.task_index[index])

    def _table(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if not self.tables:
            return _OAK_CENTER.copy(), _OAK_HALF.copy()
        return self.tables[self._task(index)]

    def _slot(self, context: _Context) -> str:
        return context.goals[context.current].slot

    def _target(self, context: _Context) -> str:
        return getattr(context.goals[context.current], "target", "")

    def _queue(self, index: int):
        task = self._task(index)
        spec = self.tracker.specs[task]
        layout = self.tracker.env_layouts[index]
        positions = {item.slot: item.position_m[:2] for item in layout.placements}
        check = None if spec is None else spec.check
        return compile_goals(check, positions)

    def _container_offsets(self, task: int, goals: list) -> dict[int, tuple[float, float]]:
        """Drop offset [m] of every ``in_container`` goal, in its container frame."""
        by_container: dict[str, list[int]] = {}
        for index, goal in enumerate(goals):
            if isinstance(goal, PlaceGoal) and goal.op == "in_container":
                by_container.setdefault(goal.target, []).append(index)
        offsets: dict[int, tuple[float, float]] = {}
        sizes = {item.slot: item.asset.size_m for item in self.tracker.task_layouts[task].placements}
        for container, members in by_container.items():
            size = sizes[container]
            for index, offset in zip(members, drop_offsets(len(members), size[0] / 2, size[1] / 2), strict=True):
                offsets[index] = offset
        return offsets

    def _warn(self, task_id: str) -> None:
        if task_id not in self._warned:
            self._warned.add(task_id)
            print(f"[WARN] task {task_id} is not a pick-and-place the planner can attempt; it will score 0")

    def reset(self, env_ids) -> None:
        for index in np.asarray(env_ids).reshape(-1).tolist():
            index = int(index)
            context = _Context()
            queue = self._queue(index)
            context.goals = _without_redundant_resting(list(queue.goals))
            context.ordered = queue.ordered
            context.supported = queue.supported and bool(context.goals)
            if context.supported:
                context.todo = list(range(len(context.goals)))
                context.offsets = self._container_offsets(self._task(index), context.goals)
            else:
                context.stage = Stage.DONE
                self._warn(self.tracker.task_ids[self._task(index)])
            self.contexts[index] = context

    # ------------------------------------------------------------------ geometry

    def _anchor(self, slot: str, index: int):
        """A task anchor is a published pose, not a spawned body."""
        placement = self.tracker.env_layouts[index].by_slot().get(slot)
        if placement is None or placement.asset.kind != "anchor":
            return None
        return placement

    def _pose(self, slot: str, index: int) -> tuple[np.ndarray, np.ndarray]:
        anchor = self._anchor(slot, index)
        if anchor is not None:
            pos = np.asarray(anchor.position_m, dtype=np.float64)
            quat = np.asarray(anchor.quat_wxyz, dtype=np.float64)
            return pos, quat_wxyz_to_matrix(quat)
        root_pos, root_quat = self.tracker.root(index, slot)
        x, y, z, w = root_quat.astype(np.float64)
        return root_pos.astype(np.float64), quat_wxyz_to_matrix(np.array([w, x, y, z]))

    def _top_z(self, slot: str, index: int) -> float:
        anchor = self._anchor(slot, index)
        if anchor is not None:
            return float(anchor.position_m[2] + anchor.asset.size_m[2] / 2)
        pos, rot = self._pose(slot, index)
        return float((self.points[self._task(index)][slot][::3] @ rot.T)[:, 2].max() + pos[2])

    def _bottom_z(self, slot: str, index: int) -> float:
        anchor = self._anchor(slot, index)
        if anchor is not None:
            return float(anchor.position_m[2] - anchor.asset.size_m[2] / 2)
        pos, rot = self._pose(slot, index)
        return float((self.points[self._task(index)][slot][::3] @ rot.T)[:, 2].min() + pos[2])

    def _rest_grasps(self, task: int, slot: str) -> list[Grasp]:
        if (task, slot) not in self._grasps:
            pos0, quat0 = self.rest[task][slot]
            rot0 = quat_wxyz_to_matrix(np.asarray(quat0))
            found = find_grasps(self.points[task][slot] @ rot0.T, float(pos0[2]))
            # Store in the object frame so a rotated object keeps the same grasp.
            self._grasps[(task, slot)] = [
                Grasp(rot0.T @ g.center_local, rot0.T @ g.axis_local, g.width_m, g.score) for g in found
            ]
        return self._grasps[(task, slot)]

    def _candidates(self, slot: str, index: int) -> list[Grasp]:
        """Grasps in the object frame for the object's current pose."""
        task = self._task(index)
        points = self.points[task][slot]
        pos, rot = self._pose(slot, index)
        pos0, quat0 = self.rest[task][slot]
        rel = rot @ quat_wxyz_to_matrix(np.asarray(quat0)).T
        tilt = math.acos(float(np.clip(rel[2, 2], -1.0, 1.0)))
        if tilt > TILT_RESEARCH_RAD:
            found = find_grasps(points @ rot.T, float(pos[2]))
            grasps = [Grasp(rot.T @ g.center_local, rot.T @ g.axis_local, g.width_m, g.score) for g in found]
        else:
            grasps = list(self._rest_grasps(task, slot))
        if not grasps:
            found = fallback_grasp(points @ rot.T, float(pos[2]))
            grasps = [Grasp(rot.T @ found.center_local, rot.T @ found.axis_local, found.width_m, found.score)]
        return grasps

    @staticmethod
    def _reachable(point: np.ndarray) -> bool:
        limited = reach_limit(point[:2], float(point[2]))
        return bool(np.linalg.norm(limited - np.asarray(point[:2], dtype=np.float64)) < 1e-4)

    def _begin_grasp(self, context: _Context, index: int) -> None:
        slot = self._slot(context)
        if not context.candidates:
            context.candidates = self._candidates(slot, index)
            # Reachable grasps first, keeping the search order inside each group.
            pos, rot = self._pose(slot, index)
            flags = [self._reachable(pos + rot @ g.center_local) for g in context.candidates]
            ordered = [g for g, ok in zip(context.candidates, flags) if ok]
            context.candidates = ordered + [g for g, ok in zip(context.candidates, flags) if not ok]
        grasp = context.candidates[context.candidate_index % len(context.candidates)]
        pos, rot = self._pose(slot, index)
        context.grasp_center = pos + rot @ grasp.center_local
        # An object that settled lower than where it was published must not push the fingers into the table.
        context.grasp_center[2] = max(context.grasp_center[2], TOOL_Z_MIN_M)
        context.grasp_yaw = tool_yaw(rot @ grasp.axis_local)
        context.rest_z = float(pos[2])
        context.stage = Stage.PRE_GRASP
        context.timer = 0

    def _pre_z(self, grasp_z: float) -> float:
        return float(np.clip(grasp_z + 0.10, 0.16, 0.30))

    # ------------------------------------------------------------------ state machine

    def step(self, tcp: np.ndarray, closure: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Targets for every env: position [m] ``[N, 3]``, yaw [rad] ``[N]``, grip ``[N]`` (0 open, 1 close)."""
        positions = np.zeros((self.num_envs, 3))
        yaws = np.zeros(self.num_envs)
        grips = np.zeros(self.num_envs)
        for index, context in enumerate(self.contexts):
            self._advance(context, index, tcp[index], float(closure[index]))
            positions[index] = context.pos
            yaws[index] = context.yaw
            grips[index] = context.grip
        return positions, yaws, grips

    def _advance(self, context: _Context, index: int, tcp: np.ndarray, closure: float) -> None:
        context.age += 1
        if context.age <= 2:
            context.pos = tcp.copy()
            context.target = tcp.copy()
            return
        context.timer += 1
        handler = {
            Stage.INIT: self._init,
            Stage.PRE_GRASP: self._pre_grasp,
            Stage.DESCEND: self._descend,
            Stage.CLOSE: self._close,
            Stage.LIFT: self._lift,
            Stage.CARRY: self._carry,
            Stage.LOWER: self._lower,
            Stage.OPEN: self._open,
            Stage.RETRACT: self._retract,
            Stage.RECOVER: self._recover,
            Stage.DONE: self._done,
        }[context.stage]
        handler(context, index, tcp, closure)
        self._slew(context)
        context.previous_closure = closure

    @staticmethod
    def _slew(context: _Context) -> None:
        delta = context.target - context.pos
        norm = float(np.linalg.norm(delta))
        context.pos = context.target.copy() if norm <= STEP_M else context.pos + delta * (STEP_M / norm)
        dyaw = context.target_yaw - context.yaw
        context.yaw += float(np.clip(dyaw, -STEP_YAW_RAD, STEP_YAW_RAD))

    @staticmethod
    def _arrived(context: _Context, tcp: np.ndarray, tol: float = POS_TOL_M) -> bool:
        reached = np.linalg.norm(context.target - tcp) < tol
        settled = np.allclose(context.pos, context.target) and abs(context.yaw - context.target_yaw) < 0.05
        return bool(reached and settled)

    def _switch(self, context: _Context, stage: Stage) -> None:
        context.stage = stage
        context.timer = 0

    def _init(self, context, index, tcp, closure) -> None:
        if not context.supported:
            context.target = np.array([HOME[0], HOME[1], SAFE_Z_M])
            context.grip = 0.0
            self._switch(context, Stage.DONE)
            return
        remaining = [k for k in context.todo if context.attempts.get(k, 0) < MAX_ATTEMPTS]
        remaining = [k for k in remaining if not self._placed(index, k) and not self._gone(index, k)]
        if not remaining:
            context.target = np.array([HOME[0], HOME[1], SAFE_Z_M])
            context.grip = 0.0
            self._switch(context, Stage.DONE)
            return

        def distance(k: int) -> float:
            pos = self.tracker.root(index, context.goals[k].slot)[0]
            return float(np.hypot(pos[0] - tcp[0], pos[1] - tcp[1]))

        context.current = remaining[0] if context.ordered else min(remaining, key=distance)
        context.candidates = []
        context.candidate_index = context.attempts.get(context.current, 0)
        context.attempts[context.current] = context.attempts.get(context.current, 0) + 1
        context.grip = 0.0
        self._begin_grasp(context, index)

    def _gone(self, index: int, pair: int) -> bool:
        return bool(self.tracker.is_lost(self.tracker.root(index, self.contexts[index].goals[pair].slot)[0]))

    def _placed(self, index: int, pair: int) -> bool:
        goal = self.contexts[index].goals[pair]
        spec_goal = Goal(goal.slot, goal.op, getattr(goal, "target", ""))
        return bool(self.tracker.latched[index].get(latch_key(spec_goal), False))

    def _pre_grasp(self, context, index, tcp, closure) -> None:
        context.grip = 0.0
        center = context.grasp_center
        pre_z = self._pre_z(center[2])
        near = float(np.hypot(*(tcp[:2] - center[:2]))) < 0.03
        # Travel high, then drop to the pre-grasp height once above the object.
        z = pre_z if near else max(pre_z, min(TRAVEL_Z_M, max(tcp[2], pre_z)))
        context.target = np.array([center[0], center[1], z])
        context.target_yaw = context.grasp_yaw
        if near and self._arrived(context, tcp, 0.015):
            self._switch(context, Stage.DESCEND)
        elif context.timer > 120:
            self._fail(context)

    def _descend(self, context, index, tcp, closure) -> None:
        context.target = context.grasp_center.copy()
        context.target_yaw = context.grasp_yaw
        aligned = float(np.hypot(*(tcp[:2] - context.grasp_center[:2]))) < 0.008
        if abs(tcp[2] - context.grasp_center[2]) < Z_TOL_M and context.timer >= 3 and aligned or context.timer > 40:
            self._switch(context, Stage.CLOSE)

    def _close(self, context, index, tcp, closure) -> None:
        slot = self._slot(context)
        context.grip = 1.0
        context.target = context.grasp_center.copy()
        context.target_yaw = context.grasp_yaw
        if context.timer < 5:
            return
        settled = abs(closure - context.previous_closure) < 0.01
        gripped = self.tracker.touching(index, slot)
        if closure > MISSED_CLOSURE and not gripped:
            self._fail(context)
        elif settled and gripped:
            self._plan_carry(context, index)
            self._switch(context, Stage.LIFT)
        elif context.timer > 20:
            self._fail(context)

    def _plan_carry(self, context: _Context, index: int) -> None:
        goal = context.goals[context.current]
        slot = goal.slot
        context.below = float(self._tool_to_bottom(slot, index, context))
        if isinstance(goal, HoldGoal):
            context.keep_grip = True
            context.carry_z = float(np.clip(goal.height_m + context.below + CARRY_CLEARANCE_M, TRAVEL_Z_M, 0.34))
            context.release = np.array([context.grasp_center[0], context.grasp_center[1], context.carry_z])
            return
        if goal.op == "in_container":
            top = self._top_z(goal.target, index)
            context.carry_z = float(np.clip(top + context.below + CARRY_CLEARANCE_M, TRAVEL_Z_M, 0.34))
            c_pos, c_rot = self._pose(goal.target, index)
            dx, dy = context.offsets[context.current]
            target_root = c_pos + c_rot @ np.array([dx, dy, 0.0])
            o_pos, _ = self._pose(slot, index)
            shift = context.grasp_center[:2] - o_pos[:2]
            context.release = np.array(
                [target_root[0] + shift[0], target_root[1] + shift[1], top + context.below + RELEASE_CLEARANCE_M]
            )
            return
        context.carry_z = TRAVEL_Z_M
        context.release = self._other_release(context, index, goal)

    def _other_release(self, context: _Context, index: int, goal: PlaceGoal) -> np.ndarray:
        """Tool release position [m] for a goal other than ``in_container``."""
        o_pos, _ = self._pose(goal.slot, index)
        item_half = self._half(index, goal.slot)
        target_pos = target_half = target_rot = None
        if goal.target:
            target_pos, target_rot = self._pose(goal.target, index)
            target_half = self._half(index, goal.target)
        occupied = []
        for placement in self.tracker.env_layouts[index].placements:
            if placement.slot == goal.slot or placement.asset.kind == "anchor":
                continue
            center = np.asarray(placement.position_m, dtype=np.float64)
            occupied.append((center, np.asarray(placement.asset.size_m) / 2))
        table_center, table_half = self._table(index)
        center = place_center(
            goal, o_pos, item_half, target_pos, target_half, target_rot, occupied, table_center, table_half
        )
        shift = context.grasp_center[:2] - o_pos[:2]
        height = float(center[2] + context.below + RELEASE_CLEARANCE_M)
        return np.array([center[0] + shift[0], center[1] + shift[1], height])

    def _half(self, index: int, slot: str) -> np.ndarray:
        layout = self.tracker.task_layouts[self._task(index)]
        return np.asarray(layout.by_slot()[slot].asset.size_m, dtype=np.float64) / 2

    def _tool_to_bottom(self, slot: str, index: int, context: _Context) -> float:
        return max(context.grasp_center[2] - self._bottom_z(slot, index), 0.0)

    def _held_object(self, context: _Context, index: int, tcp: np.ndarray) -> bool:
        slot = self._slot(context)
        pos, _ = self._pose(slot, index)
        near = float(np.linalg.norm(pos[:2] - tcp[:2])) < 0.12 and abs(pos[2] - tcp[2]) < 0.15
        return bool(near and self.tracker.touching(index, slot))

    def _lift(self, context, index, tcp, closure) -> None:
        context.grip = 1.0
        context.target = np.array([context.grasp_center[0], context.grasp_center[1], context.carry_z])
        context.target_yaw = context.grasp_yaw
        slot = self._slot(context)
        if context.timer > 4 and not self._held_object(context, index, tcp):
            self._fail(context)
        elif abs(tcp[2] - context.carry_z) < 0.02 and context.timer >= 3:
            pos, _ = self._pose(slot, index)
            if pos[2] - context.rest_z < LIFT_MIN_M:
                self._fail(context)
            elif context.keep_grip:
                self._switch(context, Stage.DONE)
            else:
                self._switch(context, Stage.CARRY)

    def _carry(self, context, index, tcp, closure) -> None:
        context.grip = 1.0
        # The release point depends on where the object hangs, so refresh it while the container may shift.
        self._plan_release_xy(context, index, tcp)
        carry_xy = reach_limit(context.release[:2], context.carry_z)
        context.target = np.array([carry_xy[0], carry_xy[1], context.carry_z])
        context.target_yaw = context.grasp_yaw
        if not self._held_object(context, index, tcp):
            self._fail(context)
        elif float(np.hypot(*(tcp[:2] - carry_xy))) < 0.02 and context.timer >= 2:
            self._switch(context, Stage.LOWER)

    def _plan_release_xy(self, context: _Context, index: int, tcp: np.ndarray) -> None:
        goal = context.goals[context.current]
        if not isinstance(goal, PlaceGoal) or goal.op != "in_container":
            return
        c_pos, c_rot = self._pose(goal.target, index)
        dx, dy = context.offsets[context.current]
        target_root = c_pos + c_rot @ np.array([dx, dy, 0.0])
        o_pos, _ = self._pose(goal.slot, index)
        context.release[:2] = target_root[:2] + (tcp[:2] - o_pos[:2])

    def _lower(self, context, index, tcp, closure) -> None:
        context.grip = 1.0
        lower_xy = reach_limit(context.release[:2], context.release[2])
        context.target = np.array([lower_xy[0], lower_xy[1], context.release[2]])
        context.target_yaw = context.grasp_yaw
        if not self._held_object(context, index, tcp) and context.timer > 3:
            # The object left the fingers over the container: let go.
            self._switch(context, Stage.OPEN)
        elif abs(tcp[2] - context.release[2]) < 0.012 and context.timer >= 2 or context.timer > 60:
            self._switch(context, Stage.OPEN)

    def _open(self, context, index, tcp, closure) -> None:
        context.grip = 0.0
        context.target = tcp.copy() if context.timer == 1 else context.target
        if closure < 0.15 and context.timer >= 5 or context.timer > 30:
            self._switch(context, Stage.RETRACT)

    def _retract(self, context, index, tcp, closure) -> None:
        context.grip = 0.0
        context.target = np.array([tcp[0], tcp[1], max(context.carry_z, TRAVEL_Z_M)])
        if abs(tcp[2] - context.target[2]) < 0.03 and context.timer >= 2:
            self._switch(context, Stage.INIT)

    def _recover(self, context, index, tcp, closure) -> None:
        context.grip = 0.0
        context.target = np.array([tcp[0], tcp[1], TRAVEL_Z_M])
        if closure < 0.15 and abs(tcp[2] - TRAVEL_Z_M) < 0.04 or context.timer > 40:
            self._switch(context, Stage.INIT)

    def _done(self, context, index, tcp, closure) -> None:
        context.grip = 1.0 if context.keep_grip else 0.0
        if context.keep_grip:
            context.target = np.array([context.grasp_center[0], context.grasp_center[1], context.carry_z])
        else:
            context.target = np.array([HOME[0], HOME[1], SAFE_Z_M])

    def _fail(self, context: _Context) -> None:
        """Let go, back off upward, and try the next grasp of the same object."""
        context.grip = 0.0
        self._switch(context, Stage.RECOVER)


def _stage(env):
    import omni.usd

    return omni.usd.get_context().get_stage()


def build_planner(
    env, tracker: TaskTracker, stage: dict | None = None, points_per_object: int = 6000
) -> ClassicalTabletopPlanner:
    """Sample every task's object surfaces from the spawned prims and build the planner.

    Each USD variant (path and scale) is sampled once, from an env that holds it. A point's
    root-frame position is ``R^T (p - root)`` with the rotation ``R`` of that body's spawn pose.
    ``stage`` is the benchmark stage; it supplies each task's work-surface box.
    """
    usd = _stage(env)
    plan = tracker.plan
    points: list[dict[str, np.ndarray]] = []
    rest: list[dict[str, tuple]] = []
    # One surface sample per USD variant. Tasks that share a mesh reuse it.
    sampled: dict[tuple, np.ndarray] = {}
    for task, published in enumerate(tracker.task_layouts):
        phase, sample = tracker.sample_env(task)
        origin = tracker.origin(sample)
        task_points: dict[str, np.ndarray] = {}
        task_rest: dict[str, tuple] = {}
        for placement in published.placements:
            if placement.asset.kind == "anchor":
                continue
            variant = (placement.asset.usd, tuple(placement.scale or ()))
            if variant not in sampled:
                entity = plan.phase_slots[phase][sample][placement.slot]
                spawn_pos, spawn_quat = plan.start_poses[plan.entities.index(entity)]
                world = sample_prim_points(usd, f"/World/envs/env_{sample}/{entity}", points_per_object)
                spawn_rot = quat_wxyz_to_matrix(np.asarray(spawn_quat, dtype=np.float64))
                sampled[variant] = (world - (origin + np.asarray(spawn_pos, dtype=np.float64))) @ spawn_rot
            task_points[placement.slot] = sampled[variant]
            task_rest[placement.slot] = (
                np.asarray(placement.position_m, dtype=np.float64),
                np.asarray(placement.quat_wxyz, dtype=np.float64),
            )
        points.append(task_points)
        rest.append(task_rest)
    planner = ClassicalTabletopPlanner(tracker, points, rest)
    planner.tables = [table_box(stage, layout) for layout in tracker.task_layouts]
    return planner
