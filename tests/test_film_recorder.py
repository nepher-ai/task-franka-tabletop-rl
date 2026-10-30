# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The film recorder's take lifecycle, run against a fake env. No Isaac Sim, no GPU, no rendering.

The fake env mimics what the recorder sees: the hook order of ``ManagerBasedRLEnv.step``, eight physics
steps per control step, and a latch that holds an episode open after its outcome. The camera and the
video writer are fakes that count what they are asked to do.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
from franka_tabletop_rl.film import latch, points, selection, shots
from franka_tabletop_rl.film.settings import FilmSettings

DECIMATION = 8


class _Tensor:
    """The few tensor methods the recorder calls."""

    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values

    def __getitem__(self, index):
        return self.values[index]


class _FakeCamera:
    instances: list = []

    def __init__(self, env, width, height, hfov_deg, env_index, antialiasing=None):
        self.size = (height, width)
        self.poses: list = []
        self.captures = 0
        self.closed = False
        _FakeCamera.instances.append(self)

    def set_pose(self, eye, target):
        self.poses.append((np.array(eye), np.array(target)))

    def capture(self):
        self.captures += 1
        return np.full((*self.size, 3), 90, dtype=np.uint8)

    def close(self):
        self.closed = True


class _FakeVideo:
    instances: list = []

    def __init__(self, path, width, height, fps):
        self.path, self.fps, self.frames, self.closed = path, fps, 0, False
        _FakeVideo.instances.append(self)

    def write(self, frame):
        self.frames += 1

    def close(self):
        self.closed = True


class _FakePoints:
    def __init__(self, env, env_index):
        pass

    def tool_position(self):
        return np.array([0.5, 0.0, 0.2])

    def build(self):
        return points.PointFrame()


class _FakeEnv:
    """One env that follows a fixed planner timeline and succeeds at ``success_at`` control steps."""

    def __init__(self, settings, success_at=None, stage_steps=((0, "INIT"), (3, "PRE_GRASP"), (10, "CLOSE"))):
        self.film = settings
        self.num_envs = 1
        self.cfg = SimpleNamespace(decimation=DECIMATION)
        self.step_dt = DECIMATION / 120.0
        self.physics_dt = 1.0 / 120.0
        self.lengths = np.zeros(1, dtype=np.int64)
        self.max_length = 600
        self.success_at = success_at
        self.stage_steps = stage_steps
        self.film_latch = latch.PostRollLatch(
            np.zeros(1, dtype=np.int64), np.full(1, -1, dtype=np.int64), round(settings.shot.post_roll_s * 15)
        )
        self.context = SimpleNamespace(
            stage=SimpleNamespace(name="INIT"), current=0, goals=[SimpleNamespace(slot="cup")]
        )
        self.command_manager = SimpleNamespace(
            get_term=lambda name: SimpleNamespace(
                planner=SimpleNamespace(contexts=[self.context]),
                tracker=SimpleNamespace(task_id_of=lambda index: "task_a"),
            )
        )
        self.scene = SimpleNamespace(env_origins=_Tensor([[2.0, 0.0, 0.0]]), write_data_to_sim=lambda: None)
        self.sim = SimpleNamespace(render=lambda: None, forward=lambda: None, get_physics_step_count=lambda: 0)
        self.step_id = 0
        self.resets = 0

    @property
    def episode_length_buf(self):
        return _Tensor(self.lengths)

    def reset(self, term):
        self.lengths[:] = 0
        term.record_post_reset([0])

    def step(self, term):
        """One control step in the order ``ManagerBasedRLEnv.step`` calls the recorder."""
        length = int(self.lengths[0])
        for step, name in self.stage_steps:
            if length >= step:
                self.context.stage = SimpleNamespace(name=name)
        term.record_pre_step()
        for _ in range(DECIMATION):
            term.record_post_physics_decimation_step()
        self.lengths[0] += 1
        self.step_id += 1
        success = np.array([self.success_at is not None and self.lengths[0] >= self.success_at])
        self.film_latch.update(self.step_id, self.lengths, success, np.array([False]))
        done = self.film_latch.done(latch.SUCCESS, self.lengths) | (self.lengths >= self.max_length)
        if done[0]:
            term.record_pre_reset([0])
            self.lengths[:] = 0
            term.record_post_reset([0])


@pytest.fixture
def recorder_module(monkeypatch):
    """``film.recorder`` imported against a stub of the few Isaac Lab names it uses, and fake render parts."""
    managers = types.ModuleType("isaaclab.managers")

    class RecorderTerm:
        def __init__(self, cfg, env):
            self.cfg, self._env = cfg, env

    managers.RecorderTerm = RecorderTerm
    managers.RecorderTermCfg = type("RecorderTermCfg", (), {})
    managers.RecorderManagerBaseCfg = type("RecorderManagerBaseCfg", (), {})
    managers.DatasetExportMode = SimpleNamespace(EXPORT_NONE=0)
    utils = types.ModuleType("isaaclab.utils")
    utils.configclass = lambda cls: cls
    package = types.ModuleType("isaaclab")
    package.__path__ = []
    for name, module in (("isaaclab", package), ("isaaclab.managers", managers), ("isaaclab.utils", utils)):
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules.pop("franka_tabletop_rl.film.recorder", None)
    module = importlib.import_module("franka_tabletop_rl.film.recorder")
    monkeypatch.setattr(module, "FilmCamera", _FakeCamera)
    monkeypatch.setattr(module, "VideoSink", _FakeVideo)
    monkeypatch.setattr(module, "PlannerPointPass", _FakePoints)
    _FakeCamera.instances.clear()
    _FakeVideo.instances.clear()
    yield module
    sys.modules.pop("franka_tabletop_rl.film.recorder", None)


def _term(module, env):
    term = module.FilmRecorderTerm(SimpleNamespace(), env)
    return term


def _scout_rows(module, tmp_path, shot, success_at, episodes=1):
    settings = FilmSettings(shot, "scout", tmp_path)
    env = _FakeEnv(settings, success_at)
    term = _term(module, env)
    env.reset(term)
    while len(term.rows) < episodes:
        env.step(term)
    return term.rows


def test_scout_logs_the_planner_timeline_and_outcome_of_each_episode(recorder_module, tmp_path):
    shot = shots.SHOTS["long_run"]
    (row,) = _scout_rows(recorder_module, tmp_path, shot, success_at=30)
    assert row["outcome"] == "success" and row["outcome_step"] == 30
    assert row["length"] == 30 + 45  # the 3 s post-roll keeps the episode open
    assert [(e[0], e[1]) for e in row["events"]] == [(0, "INIT"), (3, "PRE_GRASP"), (10, "CLOSE")]
    assert not _FakeVideo.instances and not _FakeCamera.instances, "a scout renders nothing"
    assert selection.window_of(row, shots.Cut("first_close", 1.0, 1.0)).start_s == pytest.approx(10 / 15 - 1.0)


def test_shoot_writes_pre_roll_then_a_frame_per_frame_period_until_the_episode_ends(recorder_module, tmp_path):
    shot = shots.SHOTS["long_run"].with_overrides(resolution=(64, 36))
    (row,) = _scout_rows(recorder_module, tmp_path, shot, success_at=30)
    plan = selection.plan_from_row(row, shot)
    assert plan.stop_s is None
    settings = FilmSettings(shot, "shoot", tmp_path / "out", "rgb", plan)
    env = _FakeEnv(settings, success_at=30)
    term = _term(recorder_module, env)
    env.reset(term)
    assert _FakeVideo.instances[0].frames == 121, "pre-roll is 2 s at 60 fps plus the episode-start frame"
    while not term.finished:
        env.step(term)
    video = _FakeVideo.instances[0]
    assert video.frames == 121 + (30 + 45) * DECIMATION // 2
    assert video.closed and _FakeCamera.instances[0].closed
    take = term.take
    assert take["ended_by"] == "episode end" and take["replay_matches_scout"] is True
    assert take["duration_s"] == pytest.approx(video.frames / 60)
    assert take["outcome_at_s"] == pytest.approx(2.0 + 30 / 15)
    assert take["cut_in_s"] == pytest.approx(2.0) and take["cut_out_s"] == pytest.approx(2.0 + 30 / 15 + 1.0)
    on_disk = json.loads((tmp_path / "out" / "long_run_rgb.json").read_text())
    assert on_disk["events"][2] == [pytest.approx(2.0 + 10 / 15), "CLOSE", "cup"]


def test_camera_follows_the_path_in_take_time_with_the_env_origin(recorder_module, tmp_path):
    shot = shots.SHOTS["busy_table"].with_overrides(resolution=(64, 36))
    plan = selection.plan_first_episode(shot)
    env = _FakeEnv(FilmSettings(shot, "shoot", tmp_path, "rgb", plan))
    term = _term(recorder_module, env)
    env.reset(term)
    camera = _FakeCamera.instances[0]
    first, second_to_last = camera.poses[recorder_module.WARMUP_RENDERS], camera.poses[-1]
    start, end = shot.keys[0], shot.keys[-1]
    assert np.allclose(first[0], np.array(start.eye) + (2.0, 0.0, 0.0)), "env frame is offset by the env origin"
    # the last pre-roll frame is at 2 s of a 10 s push-in, so the eye has moved but not arrived
    assert not np.allclose(second_to_last[0], first[0])
    assert not np.allclose(second_to_last[0], np.array(end.eye) + (2.0, 0.0, 0.0))
    assert camera.captures == recorder_module.WARMUP_RENDERS + 121


def test_planned_stop_ends_the_take_early_and_checks_the_events_so_far(recorder_module, tmp_path):
    shot = shots.SHOTS["busy_table"].with_overrides(resolution=(64, 36))
    plan = selection.plan_first_episode(shot)
    settings = FilmSettings(shot, "shoot", tmp_path, "rgb", plan)
    env = _FakeEnv(settings)
    term = _term(recorder_module, env)
    env.reset(term)
    while not term.finished:
        env.step(term)
    take = term.take
    assert take["ended_by"] == "planned stop"
    assert take["frames"] == 121 + int(5.0 * 60 + 3.0 * 60)  # 5 s window plus 3 s post-roll
    assert env.lengths[0] == 8 * 15, "the sim stopped at 8 s of episode time"
    assert take["replay_matches_scout"] is None  # no scout events to compare against


def test_a_replay_that_differs_from_the_scout_is_reported(recorder_module, tmp_path):
    shot = shots.SHOTS["long_run"].with_overrides(resolution=(64, 36))
    (row,) = _scout_rows(recorder_module, tmp_path, shot, success_at=30)
    plan = selection.plan_from_row(row, shot)
    settings = FilmSettings(shot, "shoot", tmp_path / "out", "rgb", plan)
    env = _FakeEnv(settings, success_at=31)  # the replay is one step slower
    term = _term(recorder_module, env)
    env.reset(term)
    while not term.finished:
        env.step(term)
    assert term.take["replay_matches_scout"] is False
    assert "outcome_step" in term.take["replay_problem"]


def test_points_pass_writes_the_same_timeline_without_a_camera(recorder_module, tmp_path):
    shot = shots.SHOTS["busy_table"].with_overrides(resolution=(64, 36))
    plan = selection.plan_first_episode(shot)
    settings = FilmSettings(shot, "shoot", tmp_path, "points", plan)
    env = _FakeEnv(settings)
    term = _term(recorder_module, env)
    env.reset(term)
    assert not _FakeCamera.instances, "the points pass needs no RTX camera"
    while not term.finished:
        env.step(term)
    assert term.take["frames"] == 121 + 480
    assert (tmp_path / "busy_table_points.json").is_file()


def test_the_take_starts_at_the_planned_episode_not_before(recorder_module, tmp_path):
    shot = shots.SHOTS["long_run"].with_overrides(resolution=(64, 36))
    rows = _scout_rows(recorder_module, tmp_path, shot, success_at=30, episodes=2)
    plan = selection.plan_from_row(rows[1], shot)
    assert plan.episode == 1
    env = _FakeEnv(FilmSettings(shot, "shoot", tmp_path / "out", "rgb", plan), success_at=30)
    term = _term(recorder_module, env)
    env.reset(term)
    assert not _FakeVideo.instances, "episode 0 is not the planned one"
    while not term.finished:
        env.step(term)
    assert term.take["replay_matches_scout"] is True
    assert term.take["plan"]["episode"] == 1
