# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Film tooling that needs only NumPy: camera rig, shot catalog, selection, latch, and rasterizer."""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "source" / "franka_tabletop_rl"
sys.path.insert(0, str(PACKAGE))

from franka_tabletop_rl.film import latch, points, rig, selection, settings, shots, timeline  # noqa: E402


def _identity(point):
    return np.asarray(point, dtype=np.float64)


def test_path_holds_the_end_keys_and_eases_between_them():
    keys = [rig.Key(0.0, (0, 0, 0), (1, 0, 0)), rig.Key(2.0, (2, 0, 0), (1, 0, 0))]
    assert np.allclose(rig.evaluate_path(keys, -1.0, _identity)[0], (0, 0, 0))
    assert np.allclose(rig.evaluate_path(keys, 5.0, _identity)[0], (2, 0, 0))
    assert np.allclose(rig.evaluate_path(keys, 1.0, _identity, "linear")[0], (1, 0, 0))
    # smoothstep is symmetric about the midpoint and slower than linear near the ends
    assert np.allclose(rig.evaluate_path(keys, 1.0, _identity, "smooth")[0], (1, 0, 0))
    assert (
        rig.evaluate_path(keys, 0.2, _identity, "smooth")[0][0]
        < rig.evaluate_path(keys, 0.2, _identity, "linear")[0][0]
    )


def test_spline_passes_through_every_key_with_continuous_speed():
    keys = [
        rig.Key(0.0, (0, 0, 0), (0, 0, 0)),
        rig.Key(1.0, (1, 1, 0), (0, 0, 0)),
        rig.Key(2.0, (2, 0, 0), (0, 0, 0)),
    ]
    for key in keys:
        assert np.allclose(rig.evaluate_path(keys, key.t, _identity, "spline")[0], key.eye)
    before = rig.evaluate_path(keys, 1.0 - 1e-4, _identity, "spline")[0]
    after = rig.evaluate_path(keys, 1.0 + 1e-4, _identity, "spline")[0]
    middle = rig.evaluate_path(keys, 1.0, _identity, "spline")[0]
    assert np.allclose((middle - before) / 1e-4, (after - middle) / 1e-4, atol=1e-2)


def test_path_rejects_unsorted_keys_and_unknown_easing():
    keys = [rig.Key(1.0, (0, 0, 0), (1, 0, 0)), rig.Key(1.0, (1, 0, 0), (1, 0, 0))]
    with pytest.raises(ValueError, match="strictly increase"):
        rig.evaluate_path(keys, 0.5, _identity)
    with pytest.raises(ValueError, match="easing"):
        rig.evaluate_path(keys[:1], 0.5, _identity, "bounce")


def test_look_at_points_the_negative_z_axis_at_the_target_and_keeps_the_horizon_level():
    eye, target = np.array([1.0, 2.0, 0.5]), np.array([-0.4, 0.3, 0.1])
    matrix = rig.look_at_matrix(eye, target)
    forward = (target - eye) / np.linalg.norm(target - eye)
    assert np.allclose(-matrix[:, 2], forward)
    assert np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
    assert abs(matrix[2, 0]) < 1e-12  # the camera's right axis stays horizontal
    assert matrix[2, 1] > 0.0  # the camera's up axis leans toward world up


def test_look_down_does_not_degenerate():
    matrix = rig.look_at_matrix((0.0, 0.0, 2.0), (0.0, 0.0, 0.0))
    assert np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
    assert np.allclose(-matrix[:, 2], (0, 0, -1))


def test_quaternion_rotates_the_camera_axes_like_the_matrix():
    for eye, target in [((1, 0, 1), (0, 0, 0)), ((0.3, -2, 0.1), (0.5, 0.2, 0.2)), ((0, 0, 3), (0.1, 0, 0))]:
        matrix = rig.look_at_matrix(eye, target)
        x, y, z, w = rig.look_at_quat(eye, target)
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        assert np.allclose(rotation, matrix, atol=1e-9)
        assert w >= 0.0


def test_projection_matches_a_hand_computed_pinhole():
    width, height = 1920, 1080
    matrix = rig.intrinsics(width, height, 90.0)
    assert matrix[0, 0] == pytest.approx(width / 2.0)  # tan(45 deg) = 1
    rotation = rig.look_at_matrix((0, 0, 0), (1, 0, 0))
    pixels, depth = rig.project(
        np.array([[2.0, 0.0, 0.0], [2.0, -0.5, 0.25], [-1.0, 0.0, 0.0]]), (0, 0, 0), rotation, matrix
    )
    assert np.allclose(pixels[0], (width / 2, height / 2))
    # Looking along +X, world -Y is to the right and +Z is up. Offset / depth = 0.25 and 0.125.
    assert np.allclose(pixels[1], (width / 2 + width / 2 * 0.25, height / 2 - width / 2 * 0.125))
    assert np.allclose(depth[:2], (2.0, 2.0))
    assert depth[2] < 0.0 and np.isnan(pixels[2]).all()


def test_lens_helpers_round_trip_the_field_of_view():
    focal = rig.focal_length(60.0)
    assert 2 * math.degrees(math.atan(rig.HORIZONTAL_APERTURE / (2 * focal))) == pytest.approx(60.0)
    assert rig.vertical_aperture(1920, 1080) == pytest.approx(rig.HORIZONTAL_APERTURE * 9 / 16)
    with pytest.raises(ValueError):
        rig.focal_length(0.0)


def test_follow_blends_toward_the_target_and_snaps_without_a_time_constant():
    start, goal = np.zeros(3), np.array([1.0, 0.0, 0.0])
    assert np.allclose(rig.follow(None, goal, 0.01, 0.2), goal)
    assert np.allclose(rig.follow(start, goal, 0.01, 0.0), goal)
    step = rig.follow(start, goal, 0.2, 0.2)
    assert step[0] == pytest.approx(1.0 - math.exp(-1.0))


def test_frame_count_rounds_to_whole_frames():
    assert rig.frame_count(2.0, 60) == 120
    assert rig.frame_count(0.0, 60) == 0
    assert rig.frame_count(1.0 / 15.0, 60) == 4


# --------------------------------------------------------------------------- shots and selection

BENCHMARK = Path(__file__).resolve().parents[2] / "envhub" / "environments" / "franka-tabletop-rl-v0" / "benchmark.yaml"


def test_catalog_shots_are_consistent_with_the_shot_list():
    assert {name for _, name, _ in shots.SHOT_LIST} == set(shots.SHOTS)
    assert sorted({number for number, _, _ in shots.SHOT_LIST}) == [1, 2, 3, 4, 5, 6, 7]
    for number, name, pass_name in shots.SHOT_LIST:
        shot = shots.get_shot(name)
        assert pass_name in shot.passes
        assert shot.number == number or (name == "busy_table" and number == 3)
        times = [key.t for key in shot.keys]
        assert times == sorted(set(times)), f"{name} key times must strictly increase"
        width, height = shot.resolution
        assert width % 2 == 0 and height % 2 == 0, "mp4 needs an even size"
        assert 120 % shot.fps == 0, "frames must land on physics steps"
        assert shot.resolution[0] >= 1920 and shot.fps >= 60, "the request asks for 1080p and 60 fps"
    with pytest.raises(KeyError):
        shots.get_shot("nope")


def test_catalog_focus_tasks_exist_in_the_benchmark():
    if not BENCHMARK.is_file():
        pytest.skip("benchmark.yaml is not in this checkout")
    known = set(re.findall(r"task_id:\s*(\S+)", BENCHMARK.read_text(encoding="utf-8")))
    assert known
    for shot in shots.SHOTS.values():
        assert set(shot.focus) <= known, f"{shot.name} names a task that is not in the benchmark"
    # The grid shot shows 16 envs, one per task.
    assert len(known) == 16


def test_shot_rejects_bad_fields():
    base = shots.SHOTS["hero"]
    with pytest.raises(ValueError):
        base.with_overrides(outcome="maybe")
    with pytest.raises(ValueError):
        base.with_overrides(frame="moon")
    with pytest.raises(ValueError):
        base.with_overrides(keys=())
    with pytest.raises(ValueError):
        base.with_overrides(passes=("points",))
    with pytest.raises(ValueError):
        shots.Cut("sometime", 0.0, 1.0)
    assert base.with_overrides(fps=120).fps == 120 and base.fps == 60


def test_point_resolver_offsets_plain_points_and_follows_anchors():
    resolver = shots.PointResolver(
        (10.0, 0.0, 0.0), statics={"table": np.array([1.0, 2.0, 3.0])}, tool=lambda: np.array([5.0, 5.0, 5.0])
    )
    assert np.allclose(resolver((1.0, 2.0, 3.0)), (11.0, 2.0, 3.0))
    assert np.allclose(resolver(shots.Anchor("table", (0.0, 0.0, 1.0))), (1.0, 2.0, 4.0))
    assert np.allclose(resolver(shots.Anchor("tool", (0.1, 0.0, 0.0))), (5.1, 5.0, 5.0))
    with pytest.raises(KeyError):
        resolver(shots.Anchor("missing"))
    with pytest.raises(ValueError):
        shots.PointResolver((0, 0, 0))(shots.Anchor("tool"))


def _row(env, episode, task, outcome, outcome_step, length, events):
    return {
        "env": env,
        "episode": episode,
        "task_id": task,
        "outcome": outcome,
        "outcome_step": outcome_step,
        "length": length,
        "events": events,
    }


GRASP = [[0, "INIT", ""], [3, "PRE_GRASP", "a"], [30, "CLOSE", "a"], [50, "LIFT", "a"], [70, "CARRY", "a"]]


def test_windows_follow_the_anchor_and_reject_missing_events():
    row = _row(0, 0, "bagel_on_plate", "success", 120, 165, GRASP)
    hero = shots.SHOTS["hero"]
    window = selection.window_of(row, hero.cut)
    assert (window.start_s, window.end_s) == pytest.approx((2.0 - 3.0, 2.0 + 4.5))
    assert selection.window_of(row, shots.Cut("outcome", 1.0, 2.0)).end_s == pytest.approx(8.0 + 2.0)
    assert selection.window_of(row, shots.Cut("episode", 0.0, 1.0)) == selection.Window(0.0, 9.0)
    never_closed = _row(0, 0, "bagel_on_plate", "timeout", None, 600, [[0, "INIT", ""]])
    assert selection.window_of(never_closed, hero.cut) is None
    assert selection.window_of(never_closed, shots.Cut("outcome", 1.0, 1.0)) is None
    assert selection.window_of(never_closed, shots.Cut("start", 0.0, 5.0)) == selection.Window(0.0, 5.0)


def test_qualification_checks_task_outcome_duration_stages_and_room():
    hero = shots.SHOTS["hero"]
    good = _row(0, 0, "stack_yellow_on_red", "success", 120, 165, GRASP)
    assert selection.qualifies(good, hero)
    assert not selection.qualifies({**good, "task_id": "stack_white_mugs"}, hero)
    assert not selection.qualifies({**good, "outcome": "lost"}, hero)
    # LIFT and CARRY must begin inside the window, or the hero grasp does not carry.
    assert not selection.qualifies({**good, "events": GRASP[:3]}, hero)
    # the window must fit inside the episode
    assert not selection.qualifies({**good, "length": 50}, hero)
    run = shots.SHOTS["long_run"]
    fast = _row(0, 0, "stack_white_mugs", "success", 15 * 12, 15 * 12 + 45, GRASP)
    slow = _row(0, 1, "stack_white_mugs", "success", 15 * 30, 15 * 30 + 45, GRASP)
    assert not selection.qualifies(fast, run)
    assert selection.qualifies(slow, run)
    # a late CLOSE leaves too little pre-roll when the window starts before the episode
    early = _row(
        0,
        0,
        "stack_yellow_on_red",
        "success",
        120,
        165,
        [[0, "INIT", ""], [10, "CLOSE", "a"], [20, "LIFT", "a"], [30, "CARRY", "a"]],
    )
    assert not selection.qualifies(early, hero)


def test_ranking_prefers_the_middle_of_the_allowed_duration():
    run = shots.SHOTS["long_run"]
    rows = [
        _row(0, i, "stack_white_mugs", "success", 15 * seconds, 15 * seconds + 45, GRASP)
        for i, seconds in enumerate((21, 29, 38))
    ]
    assert selection.pick(rows, run)["episode"] == 1
    assert selection.pick(rows, run, rank=2)["episode"] in (0, 2)
    assert selection.pick(rows, run, rank=3) is None


def test_pair_uses_one_task_with_both_outcomes():
    failure, success = shots.SHOTS["failure"], shots.SHOTS["success"]
    long_events = [[0, "INIT", ""], [3, "PRE_GRASP", "a"], [30, "CLOSE", "a"]]
    rows = [
        _row(0, 0, "stack_white_mugs", "lost", 200, 245, long_events),
        _row(1, 0, "stack_white_mugs", "success", 240, 285, long_events),
        _row(2, 0, "grab_a_bagel", "lost", 200, 245, long_events),
    ]
    lost, clean = selection.pick_pair(rows, failure, success)
    assert lost["task_id"] == clean["task_id"] == "stack_white_mugs"
    assert lost["outcome"] == "lost" and clean["outcome"] == "success"
    assert selection.pick_pair(rows[:1], failure, success) is None


def test_take_plans_stop_after_the_window_plus_the_post_roll():
    hero = shots.SHOTS["hero"]
    row = _row(3, 2, "bagel_on_plate", "success", 120, 165, GRASP)
    plan = selection.plan_from_row(row, hero)
    assert (plan.env, plan.episode) == (3, 2)
    assert plan.stop_s == pytest.approx(plan.window.end_s + hero.post_roll_s)
    assert plan.events[1] == (3, "PRE_GRASP", "a")
    assert selection.plan_from_row(row, shots.SHOTS["long_run"]).stop_s is None
    with pytest.raises(ValueError):
        selection.plan_from_row(_row(0, 0, "x", "timeout", None, 600, []), hero)
    grid = selection.plan_first_episode(shots.SHOTS["parallel"])
    assert (grid.env, grid.episode, grid.stop_s) == (0, 0, 5.0 + 3.0)
    with pytest.raises(ValueError):
        selection.plan_first_episode(hero)


def test_scout_log_round_trips(tmp_path):
    stage = shots.StageSpec(num_envs=64, jitter_m=0.02)
    path = selection.scout_path(tmp_path, stage)
    rows = [_row(0, 0, "stack_white_mugs", "lost", 200, 245, GRASP)]
    selection.save_scout(path, stage, ["stack_white_mugs"], rows, {"post_roll_s": 3.0})
    loaded = selection.load_scout(path)
    assert loaded["rows"] == rows and loaded["post_roll_s"] == 3.0
    assert loaded["stage"]["num_envs"] == 64
    with pytest.raises(FileNotFoundError):
        selection.load_scout(tmp_path / "none.json")
    assert "stack_white_mugs" in selection.format_summary(rows)


# ------------------------------------------------------------------------ timeline, latch, settings


def test_episode_log_records_only_changes_and_closes_rows():
    log = timeline.EpisodeLog(2)
    log.observe(0, 0, "INIT", "")  # before the first episode: ignored
    assert log.start(0) == 0
    log.observe(0, 0, "INIT", "")
    log.observe(0, 1, "INIT", "")
    log.observe(0, 4, "PRE_GRASP", "cup")
    log.observe(0, 9, "PRE_GRASP", "cup")
    log.observe(0, 20, "PRE_GRASP", "plate")
    assert log.events(0) == [(0, "INIT", ""), (4, "PRE_GRASP", "cup"), (20, "PRE_GRASP", "plate")]
    row = log.finish(0, "task", "timeout", None, 600)
    assert row["episode"] == 0 and row["outcome_step"] is None and row["length"] == 600
    assert log.finish(0, "task", "timeout", None, 600) is None
    assert log.start(0) == 1 and log.events(0) == []
    assert log.rows == [row]
    assert log.episode(1) == -1


def test_frame_clock_places_frames_on_physics_steps():
    clock = timeline.FrameClock(60, 120.0, 2.0)
    assert clock.ratio == 2 and clock.pre_roll_frames == 121
    assert clock.pre_roll_time(120) == pytest.approx(2.0)
    assert [clock.is_frame(n) for n in range(0, 5)] == [False, False, True, False, True]
    # the first live frame is one frame after the episode-start frame, so there is no gap or repeat
    assert clock.live_time(2) - clock.pre_roll_time(120) == pytest.approx(1 / 60)
    assert clock.stop_substeps(1.0) == 120 and clock.stop_substeps(1.001) == 122 and clock.stop_substeps(None) is None
    assert timeline.FrameClock(120, 120.0, 0.0).ratio == 1
    with pytest.raises(ValueError):
        timeline.FrameClock(50, 120.0, 1.0)
    with pytest.raises(ValueError):
        timeline.FrameClock(240, 120.0, 1.0)


def test_compare_events_reports_the_first_difference():
    base = [(0, "INIT", ""), (3, "PRE_GRASP", "a"), (30, "CLOSE", "a")]
    assert timeline.compare_events(base, [list(event) for event in base]) is None
    assert "event 1" in timeline.compare_events(base, [base[0], (4, "PRE_GRASP", "a"), base[2]])
    assert "more events" in timeline.compare_events(base, base[:2])
    # a take that stops at step 10 only owes the events before it
    assert timeline.compare_events(base, base[:2], upto_step=10) is None


def _latch(post_roll, count=2):
    return latch.PostRollLatch(np.zeros(count, dtype=np.int64), np.full(count, -1, dtype=np.int64), post_roll)


def test_latch_holds_the_episode_open_for_the_post_roll():
    gate = _latch(3)
    no, yes = np.array([False, False]), np.array([True, False])
    for step, length in enumerate((10, 11, 12, 13, 14), start=1):
        gate.update(step, np.array([length, length]), yes if length >= 11 else no, no)
        done = gate.done(latch.SUCCESS, np.array([length, length]))
        assert done[0] == (length >= 14), f"length {length}"
        assert not done[1]
    assert gate.outcome_step[0] == 11


def test_latch_keeps_the_first_outcome_and_ignores_repeat_updates():
    gate = _latch(0, count=1)
    ones, zeros = np.array([True]), np.array([False])
    gate.update(1, np.array([5]), zeros, ones)  # lost first
    gate.update(2, np.array([6]), ones, zeros)  # a late success does not change it
    assert gate.outcome[0] == latch.LOST and gate.outcome_step[0] == 5
    assert gate.done(latch.LOST, np.array([5]))[0] and not gate.done(latch.SUCCESS, np.array([6]))[0]
    gate.update(2, np.array([99]), zeros, zeros)  # same step id: ignored
    assert gate.outcome_step[0] == 5
    with pytest.raises(ValueError):
        _latch(-1)


def test_latch_clears_when_an_env_starts_a_new_episode():
    gate = _latch(0, count=1)
    gate.update(1, np.array([8]), np.array([True]), np.array([False]))
    assert gate.outcome[0] == latch.SUCCESS
    gate.update(2, np.array([1]), np.array([False]), np.array([False]))
    assert gate.outcome[0] == latch.NONE and gate.outcome_step[0] == -1


def test_latch_with_no_post_roll_matches_the_plain_terminations():
    gate = _latch(0, count=1)
    gate.update(1, np.array([4]), np.array([True]), np.array([False]))
    assert gate.done(latch.SUCCESS, np.array([4]))[0] and not gate.done(latch.LOST, np.array([4]))[0]


def test_settings_validate_mode_pass_and_plan(tmp_path):
    shot = shots.SHOTS["busy_table"]
    plan = selection.plan_first_episode(shot)
    ok = settings.FilmSettings(shot, "shoot", tmp_path, "points", plan)
    assert ok.renders
    assert not settings.FilmSettings(shot, "scout", tmp_path).renders
    with pytest.raises(ValueError):
        settings.FilmSettings(shot, "shoot", tmp_path)
    with pytest.raises(ValueError):
        settings.FilmSettings(shot, "film", tmp_path, plan=plan)
    with pytest.raises(ValueError):
        settings.FilmSettings(shots.SHOTS["hero"], "shoot", tmp_path, "points", plan)


# ------------------------------------------------------------------------------------- rasterizer


def test_splat_draws_the_nearest_point_on_top():
    canvas = points.Canvas(16, 16, background=(0, 0, 0))
    here = np.array([[8.0, 8.0], [8.0, 8.0]])
    red_green = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    canvas.splat(here, np.array([1.0, 2.0]), red_green, 0)
    assert tuple(canvas.image[8, 8]) == (255, 0, 0)
    canvas.splat(here, np.array([3.0, 2.0]), red_green[::-1], 0)
    assert tuple(canvas.image[8, 8]) == (255, 0, 0)  # depth 3 loses to depth 2 in the same call, then redraws
    canvas.clear()
    canvas.splat(here, np.array([3.0, 2.0]), red_green, 0)
    assert tuple(canvas.image[8, 8]) == (0, 255, 0)
    canvas.clear()
    canvas.splat(np.array([[np.nan, 3.0], [-5.0, 3.0], [100.0, 3.0]]), np.ones(3), (9, 9, 9), 1)
    assert canvas.image.sum() == 0  # off-screen and NaN points draw nothing


def test_polyline_connects_its_ends_and_skips_missing_points():
    canvas = points.Canvas(32, 32, background=(0, 0, 0))
    canvas.polyline(np.array([[4.0, 4.0], [20.0, 4.0]]), (200, 200, 200), 0)
    assert (canvas.image[4, 4:21] == 200).all()
    canvas.clear()
    canvas.polyline(np.array([[4.0, 4.0], [np.nan, 4.0], [20.0, 20.0]]), (200, 200, 200), 0)
    assert canvas.image.sum() == 0
    canvas.circle(np.array([16.0, 16.0]), 6.0, (255, 255, 255), 0)
    assert canvas.image[16, 22].min() == 255 and canvas.image[16, 16].max() == 0


def test_point_frame_lands_where_the_projection_says():
    width, height = 320, 180
    matrix = rig.intrinsics(width, height, 60.0)
    eye, target = (0.0, -1.0, 0.0), (0.0, 0.0, 0.0)
    rotation = rig.look_at_matrix(eye, target)
    cloud = points.Cloud(np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]), (255, 255, 255), np.array([0.0, 0.0, 0.0]))
    frame = points.PointFrame(clouds=[cloud])
    canvas = points.Canvas(width, height)
    image = points.draw_frame(canvas, frame, eye, rotation, matrix, point_radius=0)
    pixels, _ = rig.project(cloud.points, eye, rotation, matrix)
    for u, v in np.round(pixels).astype(int):
        assert image[v, u].max() > image[0, 0].max()
    # something behind the camera is dropped instead of drawn mirrored
    spot = np.array([0.0, -3.0, 0.0])
    behind = points.PointFrame(clouds=[points.Cloud(spot[None], (255, 255, 255), spot)])
    assert (points.draw_frame(canvas, behind, eye, rotation, matrix, 0) == canvas.background).all()


def test_palette_colors_are_distinct():
    colors = points.palette(12)
    assert len(set(colors)) == 12
    assert all(max(color) >= 250 for color in colors)
