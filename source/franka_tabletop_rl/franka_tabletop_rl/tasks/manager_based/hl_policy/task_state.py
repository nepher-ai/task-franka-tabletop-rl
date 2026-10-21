# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-env task state of the hierarchical environment.

One stage holds every benchmark task: env ``i`` runs task ``i % T`` of ``T``. The tracker lays each
env's scene out in the shared object bodies, scores progress with the benchmark's own predicates,
and holds the latched success and failure flags that eval-nav reads. It mirrors the scoring of
``TabletopEvalEnv`` so a task counts as solved here exactly when the brain benchmark counts it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import yaml

from franka_tabletop_rl.tabletop.core.contact import slots_in_contact
from franka_tabletop_rl.tabletop.core.robot import GRIPPER_CLOSED_RAD
from franka_tabletop_rl.tabletop.instructions import phrase
from franka_tabletop_rl.tabletop.predicates import task_progress
from franka_tabletop_rl.tabletop.scene_cfg import read_scene_payload
from franka_tabletop_rl.tabletop.scenes.bundle import read_stage
from franka_tabletop_rl.tabletop.scenes.composer import SceneInstance, load_scene, with_jitter, world_position
from franka_tabletop_rl.tabletop.scenes.mixed import PARK_POSITION_M, SURFACE_ENTITY, MixedPlan, plan_mixed

# A finished pose has to hold this long before the episode ends. It matches the brain benchmark.
SUCCESS_HOLD_S = 0.5
# An object this far below the table or this far from the env origin is gone.
LOST_Z_M = -0.15
LOST_RADIUS_M = 1.5

ALL_TASKS = "all"
"""``scene_id`` that selects every task of the benchmark."""


def benchmark_task_ids(bundle: Path) -> list[str]:
    """Task ids of the bundle's ``benchmark.yaml``, in file order."""
    benchmark = yaml.safe_load((bundle / "benchmark.yaml").read_text(encoding="utf-8"))
    return [str(task["task_id"]) for task in benchmark["tasks"]]


def tasks_of_scene(env_id: str, scene: str) -> list[str] | None:
    """Task ids that ``scene`` runs, in env order.

    eval-nav asks the task module this instead of matching ``scene`` against the bundle's scene list.
    ``None`` means ``scene`` is not one of this benchmark's scene ids, so the caller keeps its own check.

    Raises:
        FileNotFoundError: The bundle ``env_id`` is not in the nepher cache.
    """
    from franka_tabletop_rl.tabletop.bundle_path import resolve_bundle

    bundle = resolve_bundle(env_id)
    try:
        return resolve_tasks(bundle, scene)
    except FileNotFoundError:
        return None


def resolve_tasks(bundle: Path, scene_id: str) -> list[str]:
    """Task ids a ``scene_id`` stands for: every benchmark task for ``"all"``, else that one task.

    Raises:
        FileNotFoundError: The bundle has no scene of a single named task.
    """
    if scene_id == ALL_TASKS:
        return benchmark_task_ids(bundle)
    scene_for_task(bundle, scene_id)
    return [scene_id]


def scene_for_task(bundle: Path, task_id: str) -> tuple[str, SceneInstance]:
    """Scene id and layout of the benchmark task ``task_id``.

    Raises:
        FileNotFoundError: The bundle has no scene of that task.
    """
    matches = sorted((bundle / "scenes").glob(f"{task_id}-*.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one scene for task {task_id} under {bundle}, found {len(matches)}")
    scene_id = matches[0].stem[len(task_id) + 1 :]
    scene = load_scene(bundle, task_id, scene_id)
    if scene is None:
        raise FileNotFoundError(f"cannot load scene {task_id}-{scene_id}")
    return scene_id, scene


def task_payloads(bundle: Path, task_ids: Sequence[str]) -> list[dict]:
    """Raw scene file of each task, in task order."""
    return [read_scene_payload(bundle, task_id, scene_for_task(bundle, task_id)[0]) for task_id in task_ids]


def plan_tasks(bundle: Path, task_ids: Sequence[str]) -> MixedPlan:
    """Mixed-stage plan with one env per task."""
    return plan_mixed(task_payloads(bundle, task_ids), read_stage(bundle))


def _host(values) -> np.ndarray:
    values = values.torch if hasattr(values, "torch") else values
    return values.detach().cpu().numpy()


class TaskTracker:
    """Scene layout, benchmark scoring, and latched outcome flags for ``env.num_envs`` envs.

    Args:
        env: The environment. Its scene holds the object bodies of ``plan``.
        bundle: Directory of the benchmark bundle.
        task_ids: The ``T`` tasks of the stage. Env ``i`` runs ``task_ids[i % T]``.
        plan: Mixed-stage plan already tiled to ``env.num_envs`` envs.
        pose_jitter_m: Per-episode shift [m] of each object in the table plane.
        seed: Seed of the first jitter draw.
    """

    def __init__(
        self,
        env,
        bundle: Path,
        task_ids: Sequence[str],
        plan: MixedPlan,
        pose_jitter_m: float = 0.0,
        seed: int = 1,
    ):
        self.env = env
        self.task_ids = tuple(task_ids)
        self.plan = plan
        self.count = env.num_envs
        if plan.num_envs != self.count:
            raise ValueError(f"plan covers {plan.num_envs} envs, the env has {self.count}")
        loaded = [scene_for_task(bundle, task_id) for task_id in self.task_ids]
        self.scene_ids = tuple(scene_id for scene_id, _ in loaded)
        self.published = tuple(scene for _, scene in loaded)
        self.specs = tuple(scene.spec for scene in self.published)
        self.task_index = np.arange(self.count) % len(self.task_ids)
        self.entities = plan.entities
        self.slot_entities = plan.slot_entities
        # Slot held by each body of each env, the inverse of ``slot_entities``.
        self._entity_slots = [{entity: slot for slot, entity in names.items()} for names in self.slot_entities]
        self.pose_jitter_m = float(pose_jitter_m)
        self.device = env.device
        self._seed = int(seed)
        self._episodes = 0
        self.required = tuple(self._required_slots(index) for index in range(len(self.task_ids)))
        self.scenes: list[SceneInstance] = [self.published[int(task)] for task in self.task_index]
        self.held = {entity: np.zeros(self.count, dtype=bool) for entity in self.entities}
        self.latched: list[dict[str, bool]] = [{} for _ in range(self.count)]
        self.entered: list[list[str]] = [[] for _ in range(self.count)]
        self.hold_s = np.zeros(self.count, dtype=np.float64)
        self.progress = np.zeros(self.count, dtype=np.float64)
        self.lost = np.zeros(self.count, dtype=bool)
        self.success_now = np.zeros(self.count, dtype=bool)
        # Outcome of the episode that ended last in each env, for recorders that read it after the auto-reset.
        self.last_success = np.zeros(self.count, dtype=bool)
        self.last_progress = np.zeros(self.count, dtype=np.float64)
        # Latched until ``clear_outcomes``: the evaluator reads them after the env has auto-reset.
        self.completed = torch.zeros(self.count, dtype=torch.bool, device=self.device)
        self.failed = torch.zeros(self.count, dtype=torch.bool, device=self.device)
        self.roots: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.contact: dict[str, np.ndarray] = {}
        self.gripper_open = np.ones(self.count, dtype=bool)
        self._origins = _host(env.scene.env_origins).reshape(-1, 3)
        self._updated_at = -1

    # ------------------------------------------------------------------ layout

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
        if spec.check is not None:
            walk(spec.check)
        placed = {item.slot for item in self.published[task].placements if item.asset.kind != "anchor"}
        return tuple(slot for slot in names if slot in placed)

    def task_id_of(self, index: int) -> str:
        """Benchmark task id that env ``index`` runs."""
        return self.task_ids[int(self.task_index[index])]

    def instruction(self, index: int) -> str:
        scene = self.scenes[index]
        try:
            return scene.instruction or phrase(self.task_id_of(index), 0, scene.names(), side=scene.side)
        except KeyError:
            return ""

    def entity(self, index: int, slot: str) -> str:
        """Scene body that holds ``slot`` in env ``index``."""
        return self.slot_entities[index][slot]

    def root(self, index: int, slot: str) -> tuple[np.ndarray, np.ndarray]:
        """Root position [m] relative to the env origin and xyzw quaternion of a slot in env ``index``."""
        entity = self.entity(index, slot)
        return self.roots[entity][0][index], self.roots[entity][1][index]

    def touching(self, index: int, slot: str) -> bool:
        """True while a finger presses the slot's body in env ``index``."""
        return bool(self.contact[self.entity(index, slot)][index])

    def begin(self, env_ids: torch.Tensor | slice | None, seed: int | None = None) -> None:
        """Lay the scene out in ``env_ids`` and clear their scoring state."""
        if env_ids is None or isinstance(env_ids, slice):
            ids = list(range(self.count))
        else:
            ids = [int(i) for i in torch.as_tensor(env_ids).reshape(-1).tolist()]
        for index in ids:
            self.last_success[index] = self.success_now[index]
            self.last_progress[index] = self.progress[index]
            self._episodes += 1
            episode_seed = int(seed if seed is not None else self._seed + self._episodes)
            published = self.published[int(self.task_index[index])]
            self.scenes[index] = with_jitter(published, episode_seed, self.pose_jitter_m)
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

    def _write_layout(self, ids: list[int]) -> None:
        """Pose every body of the stage in ``ids``: this env's objects, parked spares, and the work surface."""
        env_ids = torch.tensor(ids, device=self.device)
        for entity in self.entities:
            poses = []
            for index in ids:
                slot = self._entity_slots[index].get(entity)
                if slot is None:
                    position, quat = PARK_POSITION_M, (1.0, 0.0, 0.0, 0.0)
                else:
                    placement = self.scenes[index].by_slot()[slot]
                    position, quat = placement.position_m, placement.quat_wxyz
                poses.append(self._world_pose(index, position, quat))
            self._write(self.env.scene[entity], poses, env_ids)
        surface = [self._world_pose(index, *self.plan.surface_poses[index]) for index in ids]
        self._write(self.env.scene[SURFACE_ENTITY], surface, env_ids)

    def _world_pose(self, index: int, position, quat_wxyz) -> list[float]:
        origin = tuple(float(value) for value in self._origins[index])
        w, x, y, z = quat_wxyz
        return [*world_position(position, origin), x, y, z, w]

    def _write(self, entity, poses: list[list[float]], env_ids: torch.Tensor) -> None:
        entity.write_root_pose_to_sim(torch.tensor(poses, device=self.device, dtype=torch.float32), env_ids=env_ids)
        entity.write_root_velocity_to_sim(
            torch.zeros((len(poses), 6), device=self.device, dtype=torch.float32), env_ids=env_ids
        )

    # ------------------------------------------------------------------ scoring

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
        self.contact = slots_in_contact(self.env.scene, self.count, self.entities)
        for entity in self.entities:
            self.held[entity] |= self.contact[entity]
        finger = _host(self.env.scene["robot"].data.joint_pos)[:, self._finger_index()]
        self.gripper_open = (finger / GRIPPER_CLOSED_RAD) < 0.5
        dt = float(self.env.step_dt)
        for index, scene in enumerate(self.scenes):
            task = int(self.task_index[index])
            names = self.slot_entities[index]
            held = {slot: bool(self.held[entity][index]) for slot, entity in names.items()}
            touching = {slot: bool(self.contact[entity][index]) for slot, entity in names.items()}
            score, latched, entered = task_progress(
                self.specs[task],
                self._poses(index, scene),
                held,
                touching,
                self.latched[index],
                self.entered[index],
                side=scene.side,
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

    def _finger_index(self) -> int:
        robot = self.env.scene["robot"]
        return int(robot.find_joints("finger_joint")[0][0])

    @staticmethod
    def is_lost(pos: np.ndarray) -> bool:
        return bool(pos[2] < LOST_Z_M or np.linalg.norm(pos[:2]) > LOST_RADIUS_M)

    def _poses(self, index: int, scene: SceneInstance) -> dict:
        poses = {}
        for placement in scene.placements:
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
        """Env origin [m] of env `index` in the simulation frame."""
        return self._origins[index]

    def clear_outcomes(self) -> None:
        """Forget latched success and failure. Called on an explicit evaluator reset."""
        self.completed.zero_()
        self.failed.zero_()
