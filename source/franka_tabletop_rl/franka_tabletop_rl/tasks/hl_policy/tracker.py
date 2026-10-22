# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-env task state of the hierarchical environment.

One stage holds every selected task. Env ``i`` runs task ``i % T`` and resets on its own. The dome
is the benchmark's one background. Scoring uses the benchmark predicates, so a task counts as
solved here exactly when the brain benchmark counts it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from franka_tabletop_rl.benchmark.stage_plan import StagePlan
from franka_tabletop_rl.benchmark.task import TaskLayout, with_jitter, world_position
from franka_tabletop_rl.benchmark.predicates import task_progress
from franka_tabletop_rl.scene.contact import slots_in_contact
from franka_tabletop_rl.scene.robot import GRIPPER_CLOSED_RAD

# A finished pose has to hold this long before the episode ends. It matches the brain benchmark.
SUCCESS_HOLD_S = 0.5
# An object this far below the table or this far from the env origin is gone.
LOST_Z_M = -0.15
LOST_RADIUS_M = 1.5


def _host(values) -> np.ndarray:
    values = values.torch if hasattr(values, "torch") else values
    return values.detach().cpu().numpy()


class TaskTracker:
    """Scene layout, benchmark scoring, and latched outcome flags.

    Args:
        env: The environment. Its scene holds the object bodies of ``plan``.
        layouts: One layout per selected task, in ``plan.task_ids`` order.
        plan: Stage plan already sized to ``env.num_envs``.
        pose_jitter_m: Per-episode shift [m] of each object in the table plane.
        seed: Seed of the first jitter draw.
    """

    def __init__(self, env, layouts: Sequence[TaskLayout], plan: StagePlan, pose_jitter_m: float = 0.0, seed: int = 1):
        self.env = env
        self.plan = plan
        self.task_ids = tuple(plan.task_ids)
        self.task_layouts = tuple(layouts)
        self.num_envs = env.num_envs
        if plan.num_envs != self.num_envs:
            raise ValueError(f"plan covers {plan.num_envs} envs, the env has {self.num_envs}")
        if len(self.task_layouts) != len(self.task_ids):
            raise ValueError("layouts and plan.task_ids must name the same tasks")
        self.specs = tuple(layout.spec for layout in self.task_layouts)
        self.phase = 0
        self.task_index = np.array(plan.assignment[0], dtype=int)
        self.entities = plan.entities
        self._entity_slots = [
            [{entity: slot for slot, entity in names.items()} for names in phase] for phase in plan.phase_slots
        ]
        self.pose_jitter_m = float(pose_jitter_m)
        self.device = env.device
        self._seed = int(seed)
        self._episodes = 0
        self._active = False
        self.required = tuple(self._required_slots(index) for index in range(len(self.task_ids)))
        self.env_layouts: list[TaskLayout] = [self.task_layouts[int(task)] for task in self.task_index]
        self.held = {entity: np.zeros(self.num_envs, dtype=bool) for entity in self.entities}
        self.latched: list[dict[str, bool]] = [{} for _ in range(self.num_envs)]
        self.entered: list[list[str]] = [[] for _ in range(self.num_envs)]
        self.hold_s = np.zeros(self.num_envs, dtype=np.float64)
        self.progress = np.zeros(self.num_envs, dtype=np.float64)
        self.lost = np.zeros(self.num_envs, dtype=bool)
        self.success_now = np.zeros(self.num_envs, dtype=bool)
        self.last_success = np.zeros(self.num_envs, dtype=bool)
        self.last_progress = np.zeros(self.num_envs, dtype=np.float64)
        self.last_task_ids = [self.task_id_of(index) for index in range(self.num_envs)]
        self.finished_steps = np.full(self.num_envs, -1, dtype=int)
        self._steps = np.zeros(self.num_envs, dtype=int)
        self.completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.roots: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.contact: dict[str, np.ndarray] = {}
        self.gripper_open = np.ones(self.num_envs, dtype=bool)
        self._origins = _host(env.scene.env_origins).reshape(-1, 3)
        self._updated_at = -1
        self._finger = int(env.scene["robot"].find_joints("finger_joint")[0][0])

    def _required_slots(self, task: int) -> tuple[str, ...]:
        """Slots whose stage the success expression of task ``task`` scores."""
        names: list[str] = []

        def walk(check) -> None:
            for goal in check.goals:
                if goal.slot not in names:
                    names.append(goal.slot)
            for child in check.children:
                walk(child)

        spec = self.specs[task]
        if spec is not None and spec.check is not None:
            walk(spec.check)
        placed = {item.slot for item in self.task_layouts[task].placements if item.asset.kind != "anchor"}
        return tuple(slot for slot in names if slot in placed)

    def task_id_of(self, index: int) -> str:
        """Benchmark task id that env ``index`` is running."""
        return self.task_ids[int(self.task_index[index])]

    def instruction(self, index: int) -> str:
        return self.env_layouts[index].instruction

    def entity(self, index: int, slot: str) -> str:
        """Scene body that holds ``slot`` in env ``index``."""
        return self.plan.phase_slots[self.phase][index][slot]

    def sample_env(self, task: int) -> tuple[int, int]:
        """``(0, env)`` whose body carries task ``task``, for mesh sampling."""
        for env, assigned in enumerate(self.plan.assignment[0]):
            if assigned == task:
                return 0, env
        raise RuntimeError(f"no env runs task {self.task_ids[task]}")

    def root(self, index: int, slot: str) -> tuple[np.ndarray, np.ndarray]:
        """Root position [m] relative to the env origin and xyzw quaternion of a slot."""
        entity = self.entity(index, slot)
        return self.roots[entity][0][index], self.roots[entity][1][index]

    def touching(self, index: int, slot: str) -> bool:
        """True while a finger presses the slot's body in env ``index``."""
        return bool(self.contact[self.entity(index, slot)][index])

    def note_step(self, lengths) -> None:
        """Remember the command manager's episode length [steps]. Finish counts come from the env."""
        self._steps = np.asarray(lengths, dtype=int).reshape(-1)

    def prepare_reset(self) -> None:
        """Forget the previous episode's finish marks. An explicit evaluator reset calls this."""
        self.finished_steps[:] = -1
        self._active = False

    def begin(self, env_ids: torch.Tensor | slice | None, seed: int | None = None) -> None:
        """Lay the current tasks out in ``env_ids`` and clear their scoring state."""
        if env_ids is None or isinstance(env_ids, slice):
            ids = list(range(self.num_envs))
        else:
            ids = [int(i) for i in torch.as_tensor(env_ids).reshape(-1).tolist()]
        if self._active:
            for index in ids:
                length = self.env.episode_length_buf[index]
                self.finished_steps[index] = int(length.item() if hasattr(length, "item") else length)
                self.last_task_ids[index] = self.task_id_of(index)
                self.last_success[index] = self.success_now[index]
                self.last_progress[index] = self.progress[index]
        self._active = True
        for index in ids:
            self._episodes += 1
            episode_seed = int(seed if seed is not None else self._seed + self._episodes)
            published = self.task_layouts[int(self.task_index[index])]
            self.env_layouts[index] = with_jitter(published, episode_seed, self.pose_jitter_m)
            self.latched[index] = {}
            self.entered[index] = []
            for held in self.held.values():
                held[index] = False
            self.hold_s[index] = 0.0
            self.progress[index] = 0.0
            self.lost[index] = False
            self.success_now[index] = False
        self._write_layout(ids)
        self._updated_at = -1

    def episode_finished(self):
        """``None``. Each env resets on its own, so there is no shared finish mask."""
        return

    def episode_steps(self) -> np.ndarray:
        """Steps each env had run when it finished. ``-1`` means it has not finished."""
        return self.finished_steps.copy()

    def _write_layout(self, ids: list[int]) -> None:
        """Pose every body: this env's objects, parked spares, and the work surface."""
        env_ids = torch.tensor(ids, device=self.device)
        inverse = self._entity_slots[self.phase]
        for entity_index, entity in enumerate(self.entities):
            poses = []
            for index in ids:
                slot = inverse[index].get(entity)
                if slot is None:
                    position = self.plan.parking[index][entity_index]
                    quat = (1.0, 0.0, 0.0, 0.0)
                else:
                    placement = self.env_layouts[index].by_slot()[slot]
                    position, quat = placement.position_m, placement.quat_wxyz
                poses.append(self._world_pose(index, position, quat))
            self._write(self.env.scene[entity], poses, env_ids)
        for surface_index, name in enumerate(self.plan.surface_entities):
            poses = []
            for index in ids:
                if surface_index == int(self.plan.active_surface[self.phase][index]):
                    position, quat = self.plan.surface_pose_by_phase[self.phase][index]
                else:
                    position, quat = self.plan.parking[index][0], (1.0, 0.0, 0.0, 0.0)
                poses.append(self._world_pose(index, position, quat))
            self._write(self.env.scene[name], poses, env_ids)

    def _world_pose(self, index: int, position, quat_wxyz) -> list[float]:
        origin = tuple(float(value) for value in self._origins[index])
        w, x, y, z = quat_wxyz
        return [*world_position(position, origin), x, y, z, w]

    def _write(self, entity, poses: list[list[float]], env_ids: torch.Tensor) -> None:
        entity.write_root_pose_to_sim(torch.tensor(poses, device=self.device, dtype=torch.float32), env_ids=env_ids)
        entity.write_root_velocity_to_sim(
            torch.zeros((len(poses), 6), device=self.device, dtype=torch.float32), env_ids=env_ids
        )

    def read_roots(self) -> None:
        """Copy every body's root pose [m] to the host, positions relative to the env origin."""
        self.roots = {}
        for entity in self.entities:
            data = self.env.scene[entity].data
            pos = _host(data.root_pos_w).reshape(-1, 3) - self._origins
            self.roots[entity] = (pos, _host(data.root_quat_w).reshape(-1, 4))

    def update(self) -> None:
        """Score progress once per control step."""
        step = int(self.env.common_step_counter)
        if step == self._updated_at:
            return
        self._updated_at = step
        self.read_roots()
        self.contact = slots_in_contact(self.env.scene, self.num_envs, self.entities)
        for entity in self.entities:
            self.held[entity] |= self.contact[entity]
        finger = _host(self.env.scene["robot"].data.joint_pos)[:, self._finger]
        self.gripper_open = (finger / GRIPPER_CLOSED_RAD) < 0.5
        dt = float(self.env.step_dt)
        for index, layout in enumerate(self.env_layouts):
            task = int(self.task_index[index])
            names = self.plan.phase_slots[self.phase][index]
            held = {slot: bool(self.held[entity][index]) for slot, entity in names.items()}
            touching = {slot: bool(self.contact[entity][index]) for slot, entity in names.items()}
            spec = self.specs[task]
            if spec is None:
                continue
            score, latched, entered = task_progress(
                spec,
                self._poses(index, layout),
                held,
                touching,
                self.latched[index],
                self.entered[index],
                side=layout.side,
                gripper_open=bool(self.gripper_open[index]),
            )
            self.progress[index] = score
            self.latched[index] = latched
            self.entered[index] = entered
            if score >= 1.0 - 1e-9:
                self.hold_s[index] += dt
            else:
                self.hold_s[index] = 0.0
            self.success_now[index] = self.hold_s[index] >= SUCCESS_HOLD_S
            self.lost[index] = any(self.is_lost(self.root(index, slot)[0]) for slot in self.required[task])
        done = torch.as_tensor(self.success_now, device=self.device)
        self.completed |= done
        self.failed |= torch.as_tensor(self.lost, device=self.device) & ~done

    @staticmethod
    def is_lost(pos: np.ndarray) -> bool:
        return bool(pos[2] < LOST_Z_M or np.linalg.norm(pos[:2]) > LOST_RADIUS_M)

    def _poses(self, index: int, layout: TaskLayout) -> dict:
        poses = {}
        for placement in layout.placements:
            if placement.asset.kind == "anchor":
                pos = np.asarray(placement.position_m, dtype=np.float64)
                quat = np.asarray(placement.quat_wxyz, dtype=np.float64)
            else:
                root_pos, root_quat = self.root(index, placement.slot)
                pos = root_pos.astype(np.float64)
                x, y, z, w = root_quat.astype(np.float64)
                quat = np.array([w, x, y, z])
            poses[placement.slot] = {
                "pos": pos.reshape(1, 3),
                "quat": quat.reshape(1, 4),
                "half": np.array(placement.asset.size_m, dtype=np.float64) * 0.5,
            }
        return poses

    def origin(self, index: int) -> np.ndarray:
        """Env origin [m] of env ``index`` in the simulation frame."""
        return self._origins[index]

    def clear_outcomes(self) -> None:
        """Forget latched success and failure. Called on an explicit evaluator reset."""
        self.completed.zero_()
        self.failed.zero_()
