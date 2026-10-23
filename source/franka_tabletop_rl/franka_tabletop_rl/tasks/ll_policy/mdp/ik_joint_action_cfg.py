# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tool-frame IK action config. The class is a string so parsing this module does not import OpenUSD."""

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass


@configclass
class IkJointCommandActionCfg(DifferentialInverseKinematicsActionCfg):
    """``scale`` is three position deltas [m], three rotation deltas [rad], and the gripper."""

    class_type: type | str = "{DIR}.ik_joint_action:IkJointCommandAction"
    finger_joint_name: str = "finger_joint"
