# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils

from franka_tabletop_rl.benchmark.bundle import load_benchmark, resolve_bundle
from franka_tabletop_rl.benchmark.stage import xyzw_from_wxyz
from franka_tabletop_rl.benchmark.stage_plan import entity_pose, surface_body_pose
from franka_tabletop_rl.scene.robot import ARM_JOINTS

from ..start_noise import arm_start_offsets

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_tabletop_layout(env: ManagerBasedRLEnv, env_ids: torch.Tensor, pose_command_name: str = "ee_pose") -> None:
    """Write this scene's poses over the spawn pose. Run after ``reset_scene_to_default``."""
    env.command_manager.get_term(pose_command_name).tracker.begin(env_ids)


def reset_arm_start_noise(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice) -> None:
    """Offset the arm from the ready pose by seeded noise. Run after ``reset_scene_to_default``.

    The half range is ``env.cfg.arm_start_noise_rad`` [rad]. The offsets follow the seed, the reset round and the
    env index, so every full reset of an evaluation starts from a different but reproducible pose.
    """
    half_range = float(env.cfg.arm_start_noise_rad)
    if half_range <= 0.0:
        return
    robot = env.scene["robot"]
    if isinstance(env_ids, slice):
        ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    else:
        ids = torch.as_tensor(env_ids, device=env.device).long()
    joint_ids = robot.find_joints(ARM_JOINTS)[0]
    offsets = arm_start_offsets(env.start_noise_seed, env.start_noise_round, ids.tolist(), half_range, len(joint_ids))
    rows = ids[:, None]
    position = robot.data.default_joint_pos.torch[rows, joint_ids].clone()
    position += torch.as_tensor(offsets, device=env.device, dtype=position.dtype)
    limits = robot.data.soft_joint_pos_limits.torch[rows, joint_ids]
    position = position.clamp(limits[..., 0], limits[..., 1])
    robot.write_joint_position_to_sim_index(position=position, joint_ids=joint_ids, env_ids=env_ids)
    robot.set_joint_position_target_index(target=position, joint_ids=joint_ids, env_ids=env_ids)


def place_planned_layout(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> None:
    """Spawn every env's bodies at that env's own layout, before physics starts.

    A body is created once for all envs, at the pose of the first task that uses it. Without this the
    warm-up step and the kinematic setup see the wrong layout, and the first reset then teleports the
    kinematic table and shelves under the objects of the other envs.
    """
    plan = env.cfg.stage_plan
    benchmark = load_benchmark(resolve_bundle(env.cfg.env_id))
    env_layouts = [benchmark.task(plan.task_ids[task]) for task in plan.assignment[0]]
    stage = env.sim.stage
    for index in range(plan.num_envs):
        root = f"{env.scene.env_ns}/env_{index}"
        for entity_index, name in enumerate(plan.entities):
            _place(stage, f"{root}/{name}", *entity_pose(plan, env_layouts, index, entity_index))
        for surface_index, name in enumerate(plan.surface_entities):
            leaf = env.scene[name].cfg.prim_path.rsplit("/", 1)[-1]
            _place(stage, f"{root}/{leaf}", *surface_body_pose(plan, index, surface_index))


def _place(stage, path: str, position, quat_wxyz) -> None:
    sim_utils.standardize_xform_ops(
        stage.GetPrimAtPath(path),
        translation=tuple(float(value) for value in position),
        orientation=xyzw_from_wxyz(tuple(float(value) for value in quat_wxyz)),
    )
