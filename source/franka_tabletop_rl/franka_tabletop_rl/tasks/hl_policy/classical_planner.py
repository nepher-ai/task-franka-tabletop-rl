# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Classical pick-and-place planner: the high level of the hierarchical policy.

Per env it picks each object of an ``in_container`` task, carries it over its container, lets go,
and retreats, then moves to the next object. It emits a tool position, a tool yaw and a gripper
state at the control rate. The learned tracker turns those targets into joint motion.

The planner reads object and tool poses from the env and uses no privileged action: every output
is a target that the tracker has to reach on its own.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from franka_tabletop_rl.frames import MIN_REACH_X_M, REACH_HIGH_M, REACH_LOW_M, REACH_Z_HIGH_M, REACH_Z_LOW_M

from .grasp import TOOL_Z_MIN_M, Grasp, fallback_grasp, find_grasps, quat_wxyz_to_matrix, sample_prim_points, tool_yaw
from .task_state import TaskTracker

# Carrot speed limits: the tracker was trained on steps of up to 12 cm, and slower is more precise.
STEP_M = 0.05
STEP_YAW_RAD = 0.3
POS_TOL_M = 0.012
HIGH_MIN_X_M = 0.38
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
# Horizontal reach limits of the tracked workspace [m].
_REACH_MAX_M = 0.78


def reach_limit(xy: np.ndarray, z: float) -> np.ndarray:
    """Pull a horizontal tool position [m] inside the tracked reach at height `z` [m]."""
    blend = float(np.clip((z - REACH_Z_LOW_M) / (REACH_Z_HIGH_M - REACH_Z_LOW_M), 0.0, 1.0))
    radius_max = REACH_LOW_M + (REACH_HIGH_M - REACH_LOW_M) * blend
    limited = np.asarray(xy, dtype=np.float64).copy()
    limited *= min(radius_max / max(float(np.hypot(*limited)), 1e-6), 1.0)
    # Close to the base the arm cannot hold the grip-down pose high above the table.
    limited[0] = max(limited[0], MIN_REACH_X_M if z < 0.12 else HIGH_MIN_X_M)
    return limited


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
        tracker: Scene layout and scoring of the envs. Env ``i`` plans task ``tracker.task_index[i]``.
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
        self.count = tracker.count
        self.pairs = [self._pairs(task) for task in range(len(tracker.task_ids))]
        self.contexts = [_Context() for _ in range(self.count)]
        self._grasps: dict[tuple[int, str], list[Grasp]] = {}
        self._offsets = [self._container_offsets(task) for task in range(len(tracker.task_ids))]

    # ------------------------------------------------------------------ setup

    def _task(self, index: int) -> int:
        return int(self.tracker.task_index[index])

    def _pair(self, context: _Context, index: int) -> tuple[str, str]:
        """(item, container) that env ``index`` is working on."""
        return self.pairs[self._task(index)][context.current]

    def _pairs(self, task: int) -> list[tuple[str, str]]:
        """(item, container) pairs of task ``task`` in the order the task lists them."""
        pairs: list[tuple[str, str]] = []

        def walk(check) -> None:
            for goal in check.goals:
                if goal.op == "in_container" and (goal.slot, goal.target) not in pairs:
                    pairs.append((goal.slot, goal.target))
            for child in check.children:
                walk(child)

        check = self.tracker.specs[task].check
        task_id = self.tracker.task_ids[task]
        if check is None:
            raise ValueError(f"task {task_id} has no success expression")
        walk(check)
        if not pairs:
            raise NotImplementedError(f"task {task_id} has no in_container goal")
        return pairs

    def _container_offsets(self, task: int) -> dict[int, tuple[float, float]]:
        """Drop offset [m] of every pair of task ``task`` in its container frame."""
        by_container: dict[str, list[int]] = {}
        for index, (_, container) in enumerate(self.pairs[task]):
            by_container.setdefault(container, []).append(index)
        offsets: dict[int, tuple[float, float]] = {}
        sizes = {item.slot: item.asset.size_m for item in self.tracker.published[task].placements}
        for container, members in by_container.items():
            size = sizes[container]
            for index, offset in zip(members, drop_offsets(len(members), size[0] / 2, size[1] / 2)):
                offsets[index] = offset
        return offsets

    def reset(self, env_ids) -> None:
        for index in np.asarray(env_ids).reshape(-1).tolist():
            context = _Context()
            context.todo = list(range(len(self.pairs[self._task(int(index))])))
            self.contexts[int(index)] = context

    # ------------------------------------------------------------------ geometry

    def _pose(self, slot: str, index: int) -> tuple[np.ndarray, np.ndarray]:
        root_pos, root_quat = self.tracker.root(index, slot)
        x, y, z, w = root_quat.astype(np.float64)
        return root_pos.astype(np.float64), quat_wxyz_to_matrix(np.array([w, x, y, z]))

    def _top_z(self, slot: str, index: int) -> float:
        pos, rot = self._pose(slot, index)
        return float((self.points[self._task(index)][slot][::3] @ rot.T)[:, 2].max() + pos[2])

    def _bottom_z(self, slot: str, index: int) -> float:
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
        return bool(np.hypot(point[0], point[1]) <= _REACH_MAX_M and point[0] >= MIN_REACH_X_M)

    def _begin_grasp(self, context: _Context, index: int) -> None:
        slot = self._pair(context, index)[0]
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
        positions = np.zeros((self.count, 3))
        yaws = np.zeros(self.count)
        grips = np.zeros(self.count)
        for index, context in enumerate(self.contexts):
            self._advance(context, index, tcp[index], float(closure[index]))
            positions[index] = context.pos
            yaws[index] = context.yaw
            grips[index] = context.grip
        return positions, yaws, grips

    def _advance(self, context: _Context, index: int, tcp: np.ndarray, closure: float) -> None:
        context.age += 1
        if context.age <= 2:
            context.pos = tcp.copy() if context.age == 2 else context.pos
            context.target = HOME.copy()
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
        pairs = self.pairs[self._task(index)]
        remaining = [k for k in context.todo if context.attempts.get(k, 0) < MAX_ATTEMPTS]
        remaining = [k for k in remaining if not self._placed(index, k) and not self._gone(index, k)]
        if not remaining:
            context.target = np.array([HOME[0], HOME[1], SAFE_Z_M])
            context.grip = 0.0
            self._switch(context, Stage.DONE)
            return

        def distance(k: int) -> float:
            pos = self.tracker.root(index, pairs[k][0])[0]
            return float(np.hypot(pos[0] - tcp[0], pos[1] - tcp[1]))

        context.current = min(remaining, key=distance)
        context.candidates = []
        context.candidate_index = context.attempts.get(context.current, 0)
        context.attempts[context.current] = context.attempts.get(context.current, 0) + 1
        context.grip = 0.0
        self._begin_grasp(context, index)

    def _gone(self, index: int, pair: int) -> bool:
        return bool(self.tracker.is_lost(self.tracker.root(index, self.pairs[self._task(index)][pair][0])[0]))

    def _placed(self, index: int, pair: int) -> bool:
        slot, container = self.pairs[self._task(index)][pair]
        return bool(self.tracker.latched[index].get(f"in_container:{slot}:{container}", False))

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
        slot = self._pair(context, index)[0]
        context.grip = 1.0
        context.target = context.grasp_center.copy()
        context.target_yaw = context.grasp_yaw
        if context.timer < 5:
            return
        settled = abs(closure - context.previous_closure) < 0.01
        gripped = self.tracker.touching(index, slot)
        if closure > MISSED_CLOSURE:
            self._fail(context)
        elif settled and gripped:
            self._plan_carry(context, index)
            self._switch(context, Stage.LIFT)
        elif context.timer > 20:
            self._fail(context)

    def _plan_carry(self, context: _Context, index: int) -> None:
        slot, container = self._pair(context, index)
        context.below = float(self._tool_to_bottom(slot, index, context))
        top = self._top_z(container, index)
        context.carry_z = float(np.clip(top + context.below + CARRY_CLEARANCE_M, TRAVEL_Z_M, 0.34))
        c_pos, c_rot = self._pose(container, index)
        dx, dy = self._offsets[self._task(index)][context.current]
        target_root = c_pos + c_rot @ np.array([dx, dy, 0.0])
        o_pos, _ = self._pose(slot, index)
        # Keep the object under the tool wherever it hangs in the fingers.
        shift = context.grasp_center[:2] - o_pos[:2]
        context.release = np.array(
            [target_root[0] + shift[0], target_root[1] + shift[1], top + context.below + RELEASE_CLEARANCE_M]
        )

    def _tool_to_bottom(self, slot: str, index: int, context: _Context) -> float:
        return max(context.grasp_center[2] - self._bottom_z(slot, index), 0.0)

    def _held_object(self, context: _Context, index: int, tcp: np.ndarray) -> bool:
        slot = self._pair(context, index)[0]
        pos, _ = self._pose(slot, index)
        near = float(np.linalg.norm(pos[:2] - tcp[:2])) < 0.12 and abs(pos[2] - tcp[2]) < 0.15
        return bool(near and self.tracker.touching(index, slot))

    def _lift(self, context, index, tcp, closure) -> None:
        context.grip = 1.0
        context.target = np.array([context.grasp_center[0], context.grasp_center[1], context.carry_z])
        context.target_yaw = context.grasp_yaw
        slot = self._pair(context, index)[0]
        if context.timer > 4 and not self._held_object(context, index, tcp):
            self._fail(context)
        elif abs(tcp[2] - context.carry_z) < 0.02 and context.timer >= 3:
            pos, _ = self._pose(slot, index)
            if pos[2] - context.rest_z < LIFT_MIN_M:
                self._fail(context)
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
        slot, container = self._pair(context, index)
        c_pos, c_rot = self._pose(container, index)
        dx, dy = self._offsets[self._task(index)][context.current]
        target_root = c_pos + c_rot @ np.array([dx, dy, 0.0])
        o_pos, _ = self._pose(slot, index)
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
        context.grip = 0.0
        context.target = np.array([HOME[0], HOME[1], SAFE_Z_M])

    def _fail(self, context: _Context) -> None:
        """Let go, back off upward, and try the next grasp of the same object."""
        context.grip = 0.0
        self._switch(context, Stage.RECOVER)


def _stage(env):
    import omni.usd

    return omni.usd.get_context().get_stage()


def build_planner(env, tracker: TaskTracker, points_per_object: int = 6000) -> ClassicalTabletopPlanner:
    """Sample every task's object surfaces from the spawned prims and build the planner.

    Env ``t`` runs task ``t``. Each body spawns at ``plan.start_poses[k]`` in every env, so a point's
    root-frame position is ``R^T (p - root)`` with the rotation ``R`` of that spawn pose.
    """
    stage = _stage(env)
    plan = tracker.plan
    points: list[dict[str, np.ndarray]] = []
    rest: list[dict[str, tuple]] = []
    for task, published in enumerate(tracker.published):
        origin = tracker.origin(task)
        task_points: dict[str, np.ndarray] = {}
        task_rest: dict[str, tuple] = {}
        for placement in published.placements:
            if placement.asset.kind == "anchor":
                continue
            entity = tracker.entity(task, placement.slot)
            spawn_pos, spawn_quat = plan.start_poses[plan.entities.index(entity)]
            world = sample_prim_points(stage, f"/World/envs/env_{task}/{entity}", points_per_object)
            spawn_rot = quat_wxyz_to_matrix(np.asarray(spawn_quat, dtype=np.float64))
            task_points[placement.slot] = (world - (origin + np.asarray(spawn_pos, dtype=np.float64))) @ spawn_rot
            task_rest[placement.slot] = (
                np.asarray(placement.position_m, dtype=np.float64),
                np.asarray(placement.quat_wxyz, dtype=np.float64),
            )
        points.append(task_points)
        rest.append(task_rest)
    return ClassicalTabletopPlanner(tracker, points, rest)
