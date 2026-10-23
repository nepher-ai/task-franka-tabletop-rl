# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Low-level tracker. The policy sees 43 values and returns a 7-D action the IK term integrates.

Observation: joint position and velocity (7 + 7), gripper (1), tool pose (7), pose error (6),
pose target (7), grip target (1), last action (7).
"""

from __future__ import annotations

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from franka_tabletop_rl.benchmark.bundle import resolve_bundle
from franka_tabletop_rl.frames import EE_BODY, TCP_OFFSET_B
from franka_tabletop_rl.scene.robot import ARM_JOINTS
from franka_tabletop_rl.scene.scene_cfg import TabletopSceneCfg, apply_stage_assets, apply_tabletop_physics

from . import mdp

LL_EPISODE_LENGTH_S = 8.0
# Action scale: metres and radians of tool motion per unit action. The gripper entry is a sign test.
ACTION_SCALE = (0.06, 0.06, 0.06, 0.2, 0.2, 0.2, 1.0)

_EE = SceneEntityCfg("robot", body_names=EE_BODY)
_ARM = SceneEntityCfg("robot", joint_names=ARM_JOINTS)
_FINGER = SceneEntityCfg("robot", joint_names=["finger_joint"])


@configclass
class LLSceneCfg(TabletopSceneCfg):
    """Tabletop scene without cameras, contact sensors, or objects."""

    cam_exterior_left = None
    cam_exterior_right = None
    cam_wrist = None
    cam_front = None
    gripper_left = None
    gripper_right = None


@configclass
class CommandsCfg:
    """Tool pose and gripper targets, resampled every 2 to 4 s."""

    ee_pose = mdp.TcpPoseCommandCfg(asset_name="robot", body_name=EE_BODY, resampling_time_range=(2.0, 4.0))
    grip_cmd = mdp.GripperCommandCfg(resampling_time_range=(2.0, 4.0), close_prob=0.5)


@configclass
class ActionsCfg:
    """Relative tool pose and gripper in, absolute joint targets out."""

    arm_action = mdp.IkJointCommandActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINTS,
        body_name=EE_BODY,
        body_offset=mdp.IkJointCommandActionCfg.OffsetCfg(pos=TCP_OFFSET_B),
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=ACTION_SCALE,
    )


@configclass
class ObservationsCfg:
    """43-D policy observation."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _ARM}, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _ARM}, noise=Unoise(n_min=-0.01, n_max=0.01))
        gripper = ObsTerm(func=mdp.gripper_closure, params={"asset_cfg": _FINGER})
        tcp_pose = ObsTerm(func=mdp.tcp_pose, params={"asset_cfg": _EE})
        tcp_error = ObsTerm(func=mdp.tcp_error, params={"command_name": "ee_pose", "asset_cfg": _EE})
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        grip_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "grip_cmd"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Ready pose plus a random arm offset."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_arm = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.35, 0.35), "velocity_range": (0.0, 0.0), "asset_cfg": _ARM},
    )


@configclass
class RewardsCfg:
    """Pose tracking, gripper tracking, and smoothness."""

    pos_coarse = RewTerm(func=mdp.position_error, weight=-2.0, params={"command_name": "ee_pose", "asset_cfg": _EE})
    pos_fine = RewTerm(
        func=mdp.position_error_tanh,
        weight=2.0,
        params={"std": 0.05, "command_name": "ee_pose", "asset_cfg": _EE},
    )
    pos_ultrafine = RewTerm(
        func=mdp.position_error_tanh,
        weight=1.5,
        params={"std": 0.02, "command_name": "ee_pose", "asset_cfg": _EE},
    )
    pos_millimetre = RewTerm(
        func=mdp.position_error_tanh,
        weight=3.0,
        params={"std": 0.008, "command_name": "ee_pose", "asset_cfg": _EE},
    )
    ori_coarse = RewTerm(func=mdp.orientation_error, weight=-1.0, params={"command_name": "ee_pose", "asset_cfg": _EE})
    ori_fine = RewTerm(
        func=mdp.orientation_error_tanh,
        weight=2.0,
        params={"std": 0.15, "command_name": "ee_pose", "asset_cfg": _EE},
    )
    ori_ultrafine = RewTerm(
        func=mdp.orientation_error_tanh,
        weight=2.0,
        params={"std": 0.04, "command_name": "ee_pose", "asset_cfg": _EE},
    )
    grip = RewTerm(
        func=mdp.gripper_tracking,
        weight=0.5,
        params={"command_name": "grip_cmd", "asset_cfg": _FINGER},
    )
    arm_deviation = RewTerm(func=mdp.joint_deviation_l1, weight=-0.03, params={"asset_cfg": _ARM})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001, params={"asset_cfg": _ARM})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Raise the smoothness penalties once tracking works."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.03, "num_steps": 10_000}
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.004, "num_steps": 10_000}
    )


@configclass
class LLEnvCfg(ManagerBasedRLEnvCfg):
    scene: LLSceneCfg = LLSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        apply_tabletop_physics(self)
        apply_stage_assets(self, resolve_bundle())
        # Training needs neither the dome light nor a light per env.
        self.scene.background = None
        self.scene.light = None
        self.episode_length_s = LL_EPISODE_LENGTH_S


@configclass
class LLEnvCfg_PLAY(LLEnvCfg):
    """Fewer envs and no observation noise."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
