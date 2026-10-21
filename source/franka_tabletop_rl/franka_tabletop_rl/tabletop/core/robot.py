# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka arm with a 2F-85 gripper and an absolute joint-position action."""

from __future__ import annotations

import math

from isaaclab_physx.sim.schemas import PhysxArticulationCfg

import isaaclab.envs.mdp as mdp

from isaaclab_assets.robots.franka import FRANKA_ROBOTIQ_GRIPPER_CFG

from ..paths import packaged_asset

# Ready pose [rad]. The gripper is open at 0 and closed at pi/4.
READY_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -math.pi / 5,
    "panda_joint3": 0.0,
    "panda_joint4": -4 * math.pi / 5,
    "panda_joint5": 0.0,
    "panda_joint6": 3 * math.pi / 5,
    "panda_joint7": 0.0,
    "finger_joint": 0.0,
    ".*_inner_finger_joint": 0.0,
    ".*_inner_finger_knuckle_joint": 0.0,
    ".*_outer_.*_joint": 0.0,
}
GRIPPER_CLOSED_RAD = math.pi / 4

ARM_JOINTS = [f"panda_joint{index}" for index in range(1, 8)]
ACTION_JOINTS = ARM_JOINTS + ["finger_joint"]


def franka_2f85(prim_path: str = "{ENV_REGEX_NS}/Robot"):
    """Articulation at the env origin. The USD already includes the 2F-85 gripper."""
    robot = FRANKA_ROBOTIQ_GRIPPER_CFG.replace(prim_path=prim_path)
    robot.spawn = robot.spawn.replace(
        usd_path=packaged_asset("robots", "franka_robotiq_2f85.usd"),
        variants=None,
        activate_contact_sensors=True,
        articulation_props=_solver_props(robot.spawn.articulation_props),
    )
    robot.init_state = robot.init_state.replace(
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos=READY_JOINT_POS,
    )
    robot.actuators = _arm_drives(robot.actuators)
    return robot


def joint_position_action() -> mdp.JointPositionActionCfg:
    """8-D absolute joint targets [rad] plus the finger joint. Scale is 1 and there is no default offset."""
    return mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ACTION_JOINTS,
        scale=1.0,
        use_default_offset=False,
    )


def with_position_iterations(robot, iterations: int):
    """Copy of the robot cfg with ``iterations`` solver position iterations."""
    if iterations < 1:
        raise ValueError(f"solver position iterations must be positive, got {iterations}")
    return robot.replace(
        spawn=robot.spawn.replace(articulation_props=_solver_props(robot.spawn.articulation_props, iterations))
    )


def _solver_props(props: list, position_iterations: int = 64) -> list:
    """Self-collision off, ``position_iterations`` position iterations, 0 velocity iterations."""
    updated = []
    for prop in props:
        if isinstance(prop, PhysxArticulationCfg):
            updated.append(
                prop.replace(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=position_iterations,
                    solver_velocity_iteration_count=0,
                )
            )
        elif hasattr(prop, "self_collision_enabled"):
            updated.append(prop.replace(self_collision_enabled=False))
        else:
            updated.append(prop)
    return updated


def _arm_drives(actuators: dict) -> dict:
    """Shoulder effort 87 N·m and forearm 12 N·m, both at stiffness 400 and damping 80.

    The finger uses the drive authored on the asset. Its velocity limit is 5 rad/s.
    """
    finger = actuators["gripper_drive"].replace(
        joint_names_expr=["finger_joint"],
        stiffness=None,
        damping=None,
        joint_effort_limit=None,
        joint_velocity_limit=5.0,
    )
    return {
        "panda_shoulder": actuators["panda_shoulder"].replace(
            joint_effort_limit=87.0,
            joint_velocity_limit=2.175,
            stiffness=400.0,
            damping=80.0,
        ),
        "panda_forearm": actuators["panda_forearm"].replace(
            joint_effort_limit=12.0,
            joint_velocity_limit=2.61,
            stiffness=400.0,
            damping=80.0,
        ),
        "gripper": finger,
    }
