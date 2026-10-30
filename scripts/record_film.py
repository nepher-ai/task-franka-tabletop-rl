# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record the shots of the Franka video from the trained policy, offscreen, with no UI.

Modes:

* ``list``: print the shot catalog.
* ``scout``: run the stage without rendering and log every episode, then say which episodes can supply the shot.
* ``shoot``: replay the stage, render the chosen episode through the shot's camera, and write
  ``<shot>_<pass>.mp4`` and ``<shot>_<pass>.json`` with the cut points.
* ``preview``: the same as ``shoot`` at 960 by 540 and 15 fps, with PNG stills at the camera keys and cut points.

A shoot replays its scout, so both must use the same checkpoint, stage, and seed. The take file says whether
the replay matched.
"""

# Launch Isaac Sim first. Modules that import pxr must load after the app starts.

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record the shots of the Franka video.")
parser.add_argument("mode", choices=("list", "scout", "shoot", "preview"))
parser.add_argument("shot", nargs="?", default=None, help="Shot name. See the list mode.")
parser.add_argument("--pass", dest="pass_name", choices=("rgb", "points"), default="rgb", help="Image pass to render.")
parser.add_argument("--output", type=str, default=None, help="Output folder. Defaults to <project>/film.")
parser.add_argument("--checkpoint", type=str, default=None, help="Tracker checkpoint file.")
parser.add_argument("--resolution", type=str, default=None, help="Override the shot resolution, for example 1920x1080.")
parser.add_argument(
    "--fps", type=int, default=None, help="Override the frame rate [Hz]. It must divide the physics rate."
)
parser.add_argument("--rank", type=int, default=0, help="Use the n-th best episode of the scout (0 is the best).")
parser.add_argument("--env", type=int, default=None, help="Film this env, together with --episode, instead of ranking.")
parser.add_argument("--episode", type=int, default=None, help="Episode number of --env to film.")
parser.add_argument("--episodes_per_env", type=int, default=3, help="Scout: stop once every env finished this many.")
parser.add_argument(
    "--keep_scouting", action="store_true", default=False, help="Scout: do not stop at the first match."
)
parser.add_argument("--max_steps", type=int, default=None, help="Give up after this many control steps.")
parser.add_argument("--antialiasing", type=str, default=None, choices=("Off", "FXAA", "DLSS", "TAA", "DLAA"))
parser.add_argument(
    "--force", action="store_true", default=False, help="Shoot even if the scout used another checkpoint."
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.mode in ("shoot", "preview") and args_cli.pass_name == "rgb":
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports that need the simulation app follow.

import hashlib  # noqa: E402
from pathlib import Path  # noqa: E402

import franka_tabletop_rl.tasks  # noqa: E402, F401
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from franka_tabletop_rl.film import selection, shots  # noqa: E402
from franka_tabletop_rl.film.env_cfg import FilmEnvCfg, apply_film_cfg  # noqa: E402
from franka_tabletop_rl.film.settings import FilmSettings, preview_stills  # noqa: E402
from franka_tabletop_rl.paths import project_root, sync_best_policy  # noqa: E402
from franka_tabletop_rl.scene.scene_cfg import EPISODE_LENGTH_S  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

PLAY_ID = "Nepher-FrankaTabletopRL-HL-Play-v0"
FILM_ID = "Nepher-FrankaTabletopRL-HL-Film-v0"
PREVIEW_RESOLUTION = (960, 540)
PREVIEW_FPS = 15
PAIR = ("failure", "success")


def _parse_resolution(text: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in text.lower().split("x"))
    except ValueError:
        raise SystemExit(f"--resolution must look like 1920x1080, got {text!r}") from None
    return width, height


def _effective_shot(name: str, preview: bool) -> shots.Shot:
    shot = shots.get_shot(name)
    changes = {}
    if args_cli.resolution:
        changes["resolution"] = _parse_resolution(args_cli.resolution)
    if args_cli.fps:
        changes["fps"] = args_cli.fps
    if preview:
        changes.setdefault("resolution", PREVIEW_RESOLUTION)
        changes.setdefault("fps", PREVIEW_FPS)
    return shot.with_overrides(**changes) if changes else shot


def _sha(path: str) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


def _print_catalog() -> None:
    print(f"{'#':<3}{'shot':<12}{'passes':<14}{'resolution':<12}{'fps':<5}{'stage':<10}title")
    for number, name, pass_name in shots.SHOT_LIST:
        shot = shots.get_shot(name)
        stage = "all envs" if shot.all_envs else f"{shot.stage.num_envs or 'auto'} envs"
        print(
            f"{number:<3}{name:<12}{pass_name:<14}{'x'.join(map(str, shot.resolution)):<12}{shot.fps:<5}{stage:<10}"
            f"{shot.title}"
        )


def _build_env(shot: shots.Shot, settings: FilmSettings):
    stage = shot.stage
    cfg = FilmEnvCfg(task_id=stage.task_id, pose_jitter_m=stage.jitter_m, cameras=(), layout_seed=stage.seed)
    cfg.seed = stage.seed
    if stage.num_envs is not None:
        cfg.scene.num_envs = stage.num_envs
    apply_film_cfg(cfg, settings)
    return gym.make(FILM_ID, cfg=cfg)


def _load_policy(env_factory):
    installed_version = check_rsl_rl_version()
    agent_cfg = load_cfg_from_registry(PLAY_ID, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    checkpoint = sync_best_policy(
        agent_cfg.experiment_name,
        agent_cfg.load_run,
        agent_cfg.load_checkpoint,
        explicit_checkpoint=args_cli.checkpoint,
    )
    gym_env = env_factory()
    unwrapped = gym_env.unwrapped
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    runner = create_rsl_rl_runner(env, agent_cfg, log_dir=None)
    runner.load(checkpoint)
    return env, unwrapped, runner.get_inference_policy(device=unwrapped.device), checkpoint


def _run(env, unwrapped, policy, max_steps: int, stop) -> int:
    """Step the policy until ``stop()`` is true or ``max_steps``. Returns the steps taken."""
    obs = env.get_observations()
    steps = 0
    while steps < max_steps and simulation_app.is_running() and not stop():
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        steps += 1
    return steps


def _found(rows, shot: shots.Shot):
    if shot.name in PAIR:
        return selection.pick_pair(rows, shots.get_shot(PAIR[0]), shots.get_shot(PAIR[1]))
    return selection.pick(rows, shot)


def _chosen_row(rows, shot: shots.Shot):
    if args_cli.env is not None or args_cli.episode is not None:
        if args_cli.env is None or args_cli.episode is None:
            raise SystemExit("--env and --episode go together")
        match = [r for r in rows if r["env"] == args_cli.env and r["episode"] == args_cli.episode]
        if not match:
            raise SystemExit(f"the scout has no episode {args_cli.episode} of env {args_cli.env}")
        return match[0]
    if shot.name in PAIR:
        pair = selection.pick_pair(rows, shots.get_shot(PAIR[0]), shots.get_shot(PAIR[1]))
        if pair is None:
            raise SystemExit("no failure and success of one task in the scout; scout longer or on another seed")
        return pair[PAIR.index(shot.name)]
    row = selection.pick(rows, shot, args_cli.rank)
    if row is None:
        raise SystemExit(f"the scout has no episode {args_cli.rank} that suits {shot.name}; scout longer")
    return row


def _scout(shot: shots.Shot, out_dir: Path) -> None:
    if shot.all_envs:
        print(f"[INFO] {shot.name} films the first episode of every env; no scout is needed.")
        return
    settings = FilmSettings(shot, "scout", out_dir)
    env, unwrapped, policy, checkpoint = _load_policy(lambda: _build_env(shot, settings))
    recorder = unwrapped.film_recorder
    limit = args_cli.max_steps or args_cli.episodes_per_env * round(EPISODE_LENGTH_S * 15 + shot.post_roll_s * 15)

    def done() -> bool:
        finished = [0] * unwrapped.num_envs
        for row in recorder.rows:
            finished[row["env"]] += 1
        if min(finished) >= args_cli.episodes_per_env:
            return True
        return not args_cli.keep_scouting and _found(recorder.rows, shot) is not None

    steps = _run(env, unwrapped, policy, limit, done)
    rows = recorder.rows
    print(f"[INFO] Scout: {len(rows)} episodes in {steps} control steps")
    print(selection.format_summary(rows))
    path = selection.scout_path(out_dir, shot.stage)
    if path.is_file():
        previous = selection.load_scout(path)
        if previous.get("steps", 0) > steps and previous.get("checkpoint_sha") == _sha(checkpoint):
            print(f"[INFO] Keeping the longer scout already at {path}")
            rows = previous["rows"]
            steps = previous["steps"]
    meta = {"checkpoint_sha": _sha(checkpoint), "post_roll_s": shot.post_roll_s, "steps": steps}
    selection.save_scout(path, shot.stage, unwrapped.cfg.task_ids, rows, meta)
    best = _found(rows, shot)
    if best is None:
        print(
            f"[WARN] No episode of the scout suits {shot.name} yet. Scout longer (--episodes_per_env, --keep_scouting)."
        )
    else:
        pair = best if shot.name in PAIR else (best,)
        for row in pair:
            window = selection.window_of(row, shot.cut)
            print(
                f"[INFO] {shot.name}: env {row['env']} episode {row['episode']} {row['task_id']} {row['outcome']}, "
                f"window {window.start_s:.1f} to {window.end_s:.1f} s"
            )
    print(f"[RESULT] scout log: {path}")
    env.close()


def _shoot(shot: shots.Shot, out_dir: Path, preview: bool) -> None:
    base = shots.get_shot(shot.name)
    if shot.all_envs:
        plan = selection.plan_first_episode(shot)
        scout_meta = {}
    else:
        log = selection.load_scout(selection.scout_path(out_dir, shot.stage))
        if abs(log.get("post_roll_s", -1.0) - shot.post_roll_s) > 1e-9:
            raise SystemExit("the scout used another post-roll, so the resets would not line up; scout again")
        plan = selection.plan_from_row(_chosen_row(log["rows"], base), shot)
        scout_meta = {"scout_checkpoint_sha": log.get("checkpoint_sha")}
    stills = preview_stills(shot, plan) if preview else ()
    mode = "preview" if preview else "shoot"
    target = out_dir / "preview" if preview else out_dir
    settings = FilmSettings(
        shot,
        mode,
        target,
        args_cli.pass_name,
        plan,
        stills_s=stills,
        antialiasing=args_cli.antialiasing,
        meta={**scout_meta, "stage": shot.stage.as_dict()},
    )
    env, unwrapped, policy, checkpoint = _load_policy(lambda: _build_env(shot, settings))
    sha = _sha(checkpoint)
    if scout_meta and scout_meta["scout_checkpoint_sha"] not in (None, sha):
        message = f"the scout used checkpoint {scout_meta['scout_checkpoint_sha']}, this run uses {sha}"
        if not args_cli.force:
            raise SystemExit(f"{message}. Scout again, or pass --force.")
        print(f"[WARN] {message}")
    recorder = unwrapped.film_recorder
    unwrapped.film_recorder.settings.meta["checkpoint_sha"] = sha
    steps_per_episode = round(EPISODE_LENGTH_S * 15 + shot.post_roll_s * 15)
    limit = args_cli.max_steps or (plan.episode + 2) * steps_per_episode
    _run(env, unwrapped, policy, limit, lambda: recorder.finished)
    if not recorder.finished:
        env.close()
        raise SystemExit(f"[ERROR] the take was not completed within {limit} control steps")
    take = recorder.take
    cut = f"cut {take['cut_in_s']:.2f} to {take['cut_out_s']:.2f} s of {take['duration_s']:.2f} s"
    print(f"[RESULT] {take['video']}: {cut}")
    env.close()


def main() -> None:
    if args_cli.mode == "list":
        _print_catalog()
        return
    if args_cli.shot is None:
        raise SystemExit("name a shot; run the list mode to see them")
    out_dir = Path(args_cli.output) if args_cli.output else project_root() / "film"
    preview = args_cli.mode == "preview"
    gym.register(id=FILM_ID, entry_point="franka_tabletop_rl.film.env:FilmTabletopEnv", disable_env_checker=True)
    shot = _effective_shot(args_cli.shot, preview)
    if args_cli.pass_name not in shot.passes:
        raise SystemExit(f"{shot.name} has no {args_cli.pass_name} pass; it has {shot.passes}")
    if args_cli.mode == "scout":
        _scout(_effective_shot(args_cli.shot, False), out_dir)
    else:
        _shoot(shot, out_dir, preview)


if __name__ == "__main__":
    main()
    simulation_app.close()
    sys.exit(0)
