# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hierarchical environment: a classical planner sets tool targets, the learned tracker follows them.

The env is the benchmark tabletop (robot, physics, tasks, scoring) with the low-level tracker's
observation and action. The planner is a command term, so the tracker checkpoint is the only policy
that the evaluator loads. One stage holds every selected task: ``task_id`` is ``"all"`` or one task
id, as eval-nav passes it. The stage names one background. Another benchmark can name a different one.
"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.visualizers.visualizer_cfg import VisualizerCfg

from franka_tabletop_rl.benchmark.bundle import ALL_TASKS, DEFAULT_ENV_ID, load_benchmark, resolve_bundle
from franka_tabletop_rl.frames import EE_BODY
from franka_tabletop_rl.scene.cameras import OBSERVATION_CAMERAS, selected_cameras
from franka_tabletop_rl.scene.scene_cfg import TabletopSceneCfg, apply_stage_assets, apply_tabletop_physics

from ..ll_policy.ll_env_cfg import ActionsCfg, ObservationsCfg
from . import mdp
from .highlight import HighlightCfg, HighlightPreset, rendering_camera


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
    time_out = DoneTerm(func=mdp.episode_timeout, time_out=True)


@configclass
class RewardsCfg:
    """No reward: success is scored by the benchmark predicates."""


@configclass
class HLEnvCfg(ManagerBasedRLEnvCfg):
    """Tabletop benchmark tasks run by planner plus tracker, all in one stage.

    Args:
        env_id: EnvHub id. Assets are loaded from that id's nepher cache.
        task_id: ``"all"`` for every task of the benchmark, or one task id (for example
            ``cubes_and_blocks_in_bin``). ``scene.num_envs`` defaults to one env per selected task.
            A larger count has to be a multiple of that number, so every task runs together.
        pose_jitter_m: Per-episode shift [m] of each object in the table plane. Zero is the benchmark layout.
        cameras: Observation cameras to spawn. Empty spawns none. The office dome and the table light
            stay either way, so a Kit window shows the benchmark stage.
        highlight: ``on`` records highlight videos of up to 16 tasks and changes nothing else. Pass it as
            ``env.highlight=on``. ``highlight_tasks`` films those task indices in that order, for example
            ``env.highlight_tasks=0-15`` or ``env.highlight_tasks=0,5,12``. An empty selection keeps the
            seeded sample.
    """

    scene: TabletopSceneCfg = TabletopSceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    env_id: str = DEFAULT_ENV_ID
    task_id: str = ALL_TASKS
    pose_jitter_m: float = 0.0
    cameras: tuple[str, ...] = ()
    """Observation cameras to spawn. Empty spawns none."""
    layout_seed: int = 1
    highlight: HighlightPreset = HighlightPreset()
    highlight_tasks: tuple[int, ...] = ()
    """Task indices to film when highlight is on, as indices or an inclusive range such as ``0-15``.

    Empty samples up to 16 tasks with the seed.
    """
    task_ids: tuple[str, ...] = ()
    """Tasks of the stage, filled from ``task_id``."""
    stage_plan: object | None = None
    """:class:`~franka_tabletop_rl.benchmark.stage_plan.StagePlan` for ``scene.num_envs`` envs.
    :class:`HLTabletopEnv` fills it when the env is built, after eval-nav has set ``scene.num_envs``."""

    def __post_init__(self):
        apply_tabletop_physics(self)
        # Kit otherwise paints a solid sky over the office dome. None keeps the dome texture.
        self.sim.default_visualizer_cfg = VisualizerCfg(background_color=None)
        self.observations.policy.enable_corruption = False
        bundle = resolve_bundle(self.env_id)
        benchmark = load_benchmark(bundle)
        self.task_ids = benchmark.select(str(self.task_id))
        # Bodies are mounted once num_envs is final. The dome is the stage background.
        apply_stage_assets(self, bundle, benchmark.stage)
        self.scene.num_envs = len(self.task_ids)
        keep = selected_cameras(self.cameras)
        self.cameras = keep
        for name in OBSERVATION_CAMERAS:
            if name not in keep:
                setattr(self.scene, name, None)
        command = self.commands.ee_pose
        command.env_id = self.env_id
        command.task_ids = self.task_ids
        command.body_name = EE_BODY
        command.pose_jitter_m = self.pose_jitter_m
        command.layout_seed = self.layout_seed

    def play_mode(self):
        """Play defaults, plus the rendering camera when highlight videos were selected.

        The camera is not on the scene, so nothing spawns it. Its presence tells Kit to load the
        rendering app. A plain play drops the preset, so the rendering app stays off.

        The base play mode caps the scene at 50 envs. This benchmark needs one env per task, so
        that count is restored. ``--num_envs`` is applied after this method.
        """
        task_count = len(self.task_ids)
        super().play_mode()
        self.scene.num_envs = task_count
        if isinstance(self.highlight, HighlightCfg):
            self.rendering_camera = rendering_camera()
        else:
            self.highlight = None


@configclass
class HLEnvCfg_PLAY(HLEnvCfg):
    """Evaluation layout: one env per task, or a whole number of extra copies through ``scene.num_envs``."""
