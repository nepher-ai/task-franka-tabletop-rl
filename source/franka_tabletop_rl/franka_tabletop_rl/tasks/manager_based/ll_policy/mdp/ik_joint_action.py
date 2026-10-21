# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Relative tool-pose action that is turned into absolute joint targets by damped least-squares IK.

The 7-D action is ``[dx, dy, dz, drx, dry, drz, grip]``. The first six entries move the tool frame
(see :mod:`franka_tabletop_rl.frames`) relative to its current pose in the robot root frame. The
last entry closes the gripper when it is positive. Each step writes absolute targets for the seven
arm joints and the finger joint, which is the action space of the brain benchmark.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import configclass, index_fill_

from franka_tabletop_rl.tabletop.core.robot import GRIPPER_CLOSED_RAD

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class IkJointCommandAction(DifferentialInverseKinematicsAction):
    """Relative tool-pose IK plus a binary gripper, producing absolute joint targets."""

    cfg: IkJointCommandActionCfg

    def __init__(self, cfg: IkJointCommandActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._finger_ids, _ = self._asset.find_joints(cfg.finger_joint_name)
        self._closed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._joint_command = torch.zeros(self.num_envs, self._num_joints + 1, device=self.device)

    @property
    def action_dim(self) -> int:
        # Six tool-pose deltas and one gripper entry.
        return self._ik_controller.action_dim + 1

    @property
    def joint_command(self) -> torch.Tensor:
        """Absolute targets of the last step [rad]: seven arm joints then the finger joint, shape ``[N, 8]``."""
        return self._joint_command

    @property
    def gripper_closed(self) -> torch.Tensor:
        """True where the last action closed the gripper, shape ``[N]``."""
        return self._closed

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = self._raw_actions * self._scale
        self._closed = actions[:, -1] > 0.0
        ee_pos, ee_quat = self._compute_frame_pose()
        self._ik_controller.set_command(self._processed_actions[:, :-1], ee_pos, ee_quat)

    def apply_actions(self):
        ee_pos, ee_quat = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos.torch[:, self._joint_ids]
        jacobian = self._compute_frame_jacobian()
        target = self._ik_controller.compute(ee_pos, ee_quat, jacobian, joint_pos)
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        target = torch.clamp(target, limits[..., 0], limits[..., 1])
        finger = torch.where(
            self._closed, torch.full_like(target[:, 0], GRIPPER_CLOSED_RAD), torch.zeros_like(target[:, 0])
        )
        self._asset.set_joint_position_target_index(target=target, joint_ids=self._joint_ids)
        self._asset.set_joint_position_target_index(target=finger[:, None], joint_ids=self._finger_ids)
        self._joint_command[:, :-1] = target
        self._joint_command[:, -1] = finger

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        index_fill_(self._closed, env_ids, False)


@configclass
class IkJointCommandActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration of :class:`IkJointCommandAction`.

    ``scale`` has seven entries: three position deltas [m], three rotation deltas [rad], and the gripper.
    """

    class_type: type = IkJointCommandAction
    finger_joint_name: str = "finger_joint"
