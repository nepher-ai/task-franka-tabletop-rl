# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the tool-frame IK action.

The action class lives in :mod:`.ik_joint_action` and is named here as a string. Importing that
module pulls in OpenUSD, which has to be loaded by Kit. Parsing this config must not do that.
"""

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass


@configclass
class IkJointCommandActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration of the relative tool-pose action.

    ``scale`` has seven entries: three position deltas [m], three rotation deltas [rad], and the gripper.
    """

    class_type: type | str = "{DIR}.ik_joint_action:IkJointCommandAction"
    finger_joint_name: str = "finger_joint"
