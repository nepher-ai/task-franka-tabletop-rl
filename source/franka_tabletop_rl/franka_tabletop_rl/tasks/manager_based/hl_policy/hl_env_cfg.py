# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hierarchical environment: a classical planner sets tool targets, the learned tracker follows them.

The env is the benchmark tabletop (robot, physics, scenes, scoring) with the low-level tracker's
observation and action. The planner is a command term, so the tracker checkpoint is the only policy
that the evaluator loads. One stage holds every benchmark task: ``scene_id`` is ``"all"`` for all of
them, or one task id, as eval-nav passes it. Env ``i`` runs task ``i % T``, so ``num_envs`` must be a
multiple of the number of tasks ``T``.
"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from franka_tabletop_rl.frames import EE_BODY
from franka_tabletop_rl.tabletop.bundle_path import DEFAULT_ENV_ID, resolve_bundle
from franka_tabletop_rl.tabletop.scene_cfg import (
    TabletopSceneCfg,
    apply_stage_assets,
    apply_tabletop_physics,
    mount_mixed,
)

from ..ll_policy.ll_env_cfg import ActionsCfg, ObservationsCfg
from . import mdp
from .task_state import ALL_TASKS, resolve_tasks, task_payloads

# Cameras the brain benchmark sends, kept only when recording demonstrations.
_RECORD_CAMERAS = ("cam_exterior_left", "cam_exterior_right", "cam_wrist")
_ALL_CAMERAS = (*_RECORD_CAMERAS, "cam_front")


@configclass
class CommandsCfg:
    """Planner targets under the names the tracker was trained on."""

    ee_pose = mdp.HLPoseCommandCfg(asset_name="robot", body_name=EE_BODY, resampling_time_range=(1.0e6, 1.0e6))
    grip_cmd = mdp.HLGripCommandCfg(resampling_time_range=(1.0e6, 1.0e6))


@configclass
class EventCfg:
    """Default robot pose, then the scene's own object layout."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_layout = EventTerm(func=mdp.reset_tabletop_layout, mode="reset")


@configclass
class TerminationsCfg:
    """The benchmark's success after a 0.5 s hold, an unrecoverable object, and the episode limit."""

    success = DoneTerm(func=mdp.task_success)
    object_lost = DoneTerm(func=mdp.object_lost)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class RewardsCfg:
    """No reward: success is scored by the benchmark predicates."""


@configclass
class HLEnvCfg(ManagerBasedRLEnvCfg):
    """Tabletop benchmark tasks run by planner plus tracker, all in one stage.

    Args:
        env_id: EnvHub id. Assets are loaded from that id's nepher cache.
        scene_id: ``"all"`` for every task of the benchmark, or one task id (for example
            ``cubes_and_blocks_in_bin``). Env ``i`` runs task ``i % T``, and ``scene.num_envs`` must be a
            multiple of the number of tasks ``T``. It defaults to one env per task.
        pose_jitter_m: Per-episode shift [m] of each object in the table plane. Zero is the benchmark layout.
        record_cameras: Keep the brain's cameras and the dome light, for demonstrations.
    """

    scene: TabletopSceneCfg = TabletopSceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    env_id: str = DEFAULT_ENV_ID
    scene_id: str = ALL_TASKS
    pose_jitter_m: float = 0.0
    record_cameras: bool = False
    layout_seed: int = 1
    task_ids: tuple[str, ...] = ()
    """Tasks of the stage, filled from ``scene_id``."""
    mixed_plan: object | None = None
    """:class:`~franka_tabletop_rl.tabletop.scenes.mixed.MixedPlan` tiled to ``scene.num_envs`` envs.
    :class:`HLTabletopEnv` fills it when it is built. It cannot be set here, because ``configclass`` rewrites
    nested fields after ``__post_init__`` and the plan's records are frozen."""

    def __post_init__(self):
        apply_tabletop_physics(self)
        self.observations.policy.enable_corruption = False
        bundle = resolve_bundle(self.env_id)
        self.task_ids = tuple(resolve_tasks(bundle, str(self.scene_id)))
        payloads = task_payloads(bundle, self.task_ids)
        stage = apply_stage_assets(self, bundle, payloads[0])
        mount_mixed(self, bundle, stage, payloads)
        # One env per task unless the caller asks for more copies.
        self.scene.num_envs = len(self.task_ids)
        keep = _RECORD_CAMERAS if self.record_cameras else ()
        for name in _ALL_CAMERAS:
            if name not in keep:
                setattr(self.scene, name, None)
        if not self.record_cameras:
            # Without cameras neither the dome light nor the light per env is needed.
            self.scene.background = None
            self.scene.light = None
        command = self.commands.ee_pose
        command.env_id = self.env_id
        command.task_ids = self.task_ids
        command.pose_jitter_m = self.pose_jitter_m
        command.layout_seed = self.layout_seed


@configclass
class HLEnvCfg_PLAY(HLEnvCfg):
    """Evaluation layout of the benchmark: one env per task, or more copies through ``scene.num_envs``."""
