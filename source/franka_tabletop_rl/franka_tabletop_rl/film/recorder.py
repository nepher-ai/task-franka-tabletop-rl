# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The film recorder: logs every episode, and renders one take from physics substeps.

The env calls a recorder term after every physics step, which is the only place that sees 60 or 120 frames
per second of a 15 Hz policy. In ``scout`` mode the term only logs. In ``shoot`` and ``preview`` it renders
the planned episode: the pre-roll first, then a frame on every ``physics_hz / fps`` physics steps, until the
planned stop or the end of the episode.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from isaaclab.managers import DatasetExportMode, RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from . import rig
from .latch import LOST, SUCCESS
from .points import Canvas, PlannerPointPass, draw_frame
from .render import FilmCamera, VideoSink, save_png
from .shots import CONTROL_HZ, PointResolver
from .timeline import EpisodeLog, FrameClock, compare_events

if TYPE_CHECKING:
    from .settings import FilmSettings

WARMUP_RENDERS = 4
"""Renders at the first pose that are not written, so the denoiser settles after the camera cut."""


class FilmRecorderTerm(RecorderTerm):
    """Episode log for every env, plus the take of the planned episode.

    The env keeps its film settings in ``env.film`` and the recorder in ``env.film_recorder``. ``rows`` holds
    one record per finished episode. ``finished`` turns true when the take is closed.
    """

    def __init__(self, cfg: RecorderTermCfg, env):
        super().__init__(cfg, env)
        self.settings: FilmSettings = env.film
        env.film_recorder = self
        self.rows: list[dict] = []
        self.finished = False
        self.take: dict | None = None
        self._log = EpisodeLog(env.num_envs)
        self._bound = False
        self._filming = False
        self._env_index = 0
        self._base = 0
        self._substep = 0
        self._stop = None

    # ------------------------------------------------------------------ setup

    def bind(self) -> None:
        """Find the planner and, when filming, create the camera and the video. Safe to call twice."""
        if self._bound:
            return
        env = self._env
        command = env.command_manager.get_term("ee_pose")
        self._planner = command.planner
        self._tracker = command.tracker
        self._bound = True
        self._decimation = int(env.cfg.decimation)
        if abs(1.0 / env.step_dt - CONTROL_HZ) > 1e-6:
            raise RuntimeError(
                f"film timing assumes a {CONTROL_HZ} Hz policy, the env runs at {1.0 / env.step_dt:g} Hz"
            )
        if not self.settings.renders:
            return
        shot, plan = self.settings.shot, self.settings.plan
        width, height = shot.resolution
        self._env_index = int(plan.env)
        self._clock = FrameClock(shot.fps, 1.0 / env.physics_dt, shot.pre_roll_s)
        owner = None if shot.all_envs else self._env_index
        self._camera = None
        if self.settings.needs_camera:
            self._camera = FilmCamera(env, width, height, shot.hfov_deg, owner, self.settings.antialiasing)
        self._points = PlannerPointPass(env, self._env_index)
        self._canvas = Canvas(width, height) if self.settings.pass_name == "points" else None
        self._matrix = rig.intrinsics(width, height, shot.hfov_deg)
        name = f"{shot.name}_{self.settings.pass_name}"
        self._stem = self.settings.out_dir / name
        self._video: VideoSink | None = None
        self._origin = self._frame_origin(shot.frame)
        self._stop = self._clock.stop_substeps(plan.stop_s)
        self._tool: np.ndarray | None = None
        self._stills = sorted(self.settings.stills_s)
        self._tool_dt = 1.0 / shot.fps
        self._frames = 0

    def _frame_origin(self, frame: str) -> np.ndarray:
        origins = self._env.scene.env_origins
        origins = (origins.torch if hasattr(origins, "torch") else origins).detach().cpu().numpy()
        if frame == "world":
            return np.zeros(3)
        if frame == "grid":
            centre = origins.mean(axis=0)
            return np.array([centre[0], centre[1], 0.0])
        return origins[self._env_index].astype(np.float64)

    # --------------------------------------------------------------- recorder

    def record_pre_step(self):
        self.bind()
        lengths = self._env.episode_length_buf.detach().cpu().numpy()
        planner = self._planner
        for env_index in range(len(lengths)):
            context = planner.contexts[env_index]
            slot = context.goals[context.current].slot if 0 <= context.current < len(context.goals) else ""
            self._log.observe(env_index, int(lengths[env_index]), context.stage.name, slot)
        if self._filming:
            self._base = int(lengths[self._env_index]) * self._decimation
            self._substep = 0
        return None, None

    def record_post_physics_decimation_step(self):
        if not self._filming:
            return None, None
        self._substep += 1
        count = self._base + self._substep
        if self._clock.is_frame(count):
            self._emit(self._clock.live_time(count))
        if self._stop is not None and count >= self._stop:
            self._close_take("planned stop")
        return None, None

    def record_pre_reset(self, env_ids):
        self.bind()
        env = self._env
        latch = env.film_latch
        lengths = env.episode_length_buf.detach().cpu().numpy()
        for env_index in (int(i) for i in env_ids):
            kind = int(latch.outcome[env_index])
            outcome = {SUCCESS: "success", LOST: "lost"}.get(kind, "timeout")
            step = int(latch.outcome_step[env_index]) if kind else None
            row = self._log.finish(
                env_index, self._tracker.task_id_of(env_index), outcome, step, int(lengths[env_index])
            )
            if row is None:
                continue
            self.rows.append(row)
            if self._filming and env_index == self._env_index:
                self._close_take("episode end", row)
        return None, None

    def record_post_reset(self, env_ids):
        self.bind()
        for env_index in (int(i) for i in env_ids):
            episode = self._log.start(env_index)
            if not self.settings.renders or self.finished or self._filming:
                continue
            plan = self.settings.plan
            if env_index == self._env_index and episode == plan.episode:
                self._begin_take()
        return None, None

    # ------------------------------------------------------------------- take

    def _begin_take(self) -> None:
        env, shot = self._env, self.settings.shot
        env.scene.write_data_to_sim()
        env.sim.forward()
        self._video = VideoSink(self._stem.with_suffix(".mp4"), *shot.resolution, shot.fps)
        self._filming = True
        self._substep = 0
        self._base = 0
        self._tool = None
        if self._camera is not None:
            for _ in range(WARMUP_RENDERS):
                self._place_camera(0.0)
                env.sim.render()
                self._camera.capture()
        for index in range(self._clock.pre_roll_frames):
            if index and self._camera is not None:
                env.sim.render()
            self._emit(self._clock.pre_roll_time(index))

    def _tool_world(self) -> np.ndarray:
        raw = self._points.tool_position()
        shot = self.settings.shot
        self._tool = rig.follow(self._tool, raw, self._tool_dt, shot.follow_tau_s)
        return self._tool

    def _place_camera(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        shot = self.settings.shot
        resolver = PointResolver(self._origin, tool=self._tool_world)
        eye, target = rig.evaluate_path(shot.keys, t, resolver, shot.easing)
        if self._camera is not None:
            self._camera.set_pose(eye, target)
        return eye, target

    def _emit(self, t: float) -> None:
        eye, target = self._place_camera(t)
        if self.settings.pass_name == "points":
            rotation = rig.look_at_matrix(eye, target)
            radius = max(1, round(self._canvas.height / 540))
            frame = draw_frame(self._canvas, self._points.build(), eye, rotation, self._matrix, radius)
        else:
            frame = self._camera.capture()
        if self._frames == 0 and self._camera is not None and float(frame.mean()) < 1.0:
            print("[WARN] Film: the first frame is black. The rendering app may not be active.", flush=True)
        self._video.write(frame)
        self._frames += 1
        while self._stills and t >= self._stills[0] - 0.5 / self.settings.shot.fps:
            at = self._stills.pop(0)
            save_png(self._stem.parent / f"{self._stem.name}_t{at:05.2f}.png", frame)

    def _close_take(self, reason: str, row: dict | None = None) -> None:
        if not self._filming:
            return
        self._filming = False
        self.finished = True
        self._video.close()
        if self._camera is not None:
            self._camera.close()
        plan, shot = self.settings.plan, self.settings.shot
        # A take that stops inside the episode can only be compared up to the step it reached.
        upto = None if row is not None else int(self._env.episode_length_buf[self._env_index]) + 1
        events = self._log.events(self._env_index)
        problem = compare_events(plan.events, events, upto) if plan.events else None
        if row is not None and plan.length is not None:
            for name in ("outcome", "outcome_step", "length"):
                if getattr(plan, name) != row[name]:
                    problem = problem or f"{name}: scout {getattr(plan, name)}, shoot {row[name]}"
        pre = shot.pre_roll_s
        self.take = {
            "shot": shot.name,
            "number": shot.number,
            "pass": self.settings.pass_name,
            "video": str(self._video.path),
            "frames": self._frames,
            "fps": shot.fps,
            "resolution": list(shot.resolution),
            "duration_s": self._frames / shot.fps,
            "ended_by": reason,
            "plan": plan.as_dict(),
            "pre_roll_s": pre,
            "cut_in_s": max(plan.window.start_s + pre, 0.0),
            "cut_out_s": plan.window.end_s + pre,
            "outcome_at_s": None if plan.outcome_step is None else pre + plan.outcome_step / CONTROL_HZ,
            "events": [[pre + step / CONTROL_HZ, stage, slot] for step, stage, slot in events],
            "replay_matches_scout": problem is None if plan.events else None,
            "replay_problem": problem,
            "camera": {"hfov_deg": shot.hfov_deg, "easing": shot.easing, "frame": shot.frame},
            "notes": shot.notes,
            **self.settings.meta,
        }
        path = self._stem.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.take, indent=1), encoding="utf-8")
        status = "matches the scout" if plan.events else "has no scout to compare"
        if problem is not None:
            status = f"DIFFERS FROM THE SCOUT: {problem}"
        print(f"[INFO] Film take: {self._video.path} ({self._frames} frames, {reason}); replay {status}", flush=True)

    def close(self, file_path: str):
        """Close an unfinished take so its file is playable."""
        self._close_take("closed early")


@configclass
class FilmRecorderTermCfg(RecorderTermCfg):
    """Config of :class:`FilmRecorderTerm`."""

    class_type: type[RecorderTerm] = FilmRecorderTerm


@configclass
class FilmRecordersCfg(RecorderManagerBaseCfg):
    """One film recorder term. It writes no dataset."""

    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset: bool = False
    film: FilmRecorderTermCfg = FilmRecorderTermCfg()
