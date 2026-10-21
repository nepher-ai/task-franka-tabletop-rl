# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a tracker checkpoint, publish it to ``best_policy/``, and report its tracking error.

The chosen checkpoint is copied to ``best_policy/best_policy.pt``, which the high-level environment
and eval-nav load. The policy is also exported to ``best_policy/exported/ll_policy.{pt,onnx}``.

Examples:
    python scripts/rsl_rl/play.py --task Nepher-FrankaTabletopRL-LL-Play-v0
    python scripts/rsl_rl/play.py --task Nepher-FrankaTabletopRL-LL-Play-v0 --checkpoint path/model_1500.pt
    python scripts/rsl_rl/play.py --task Nepher-FrankaTabletopRL-HL-Play-v0 --num_envs 64 --num_episodes 2
    python scripts/rsl_rl/play.py --task Nepher-FrankaTabletopRL-HL-Play-v0 --scene_id throw_away_apple
"""

"""Launch Isaac Sim first: modules that import ``pxr`` must load after the app starts."""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play and publish a tracker checkpoint.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--task", type=str, default="Nepher-FrankaTabletopRL-LL-Play-v0", help="Registered gym id.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point key.")
parser.add_argument(
    "--scene_id", type=str, default=None, help="Benchmark task id of a high-level task. Every task when omitted."
)
parser.add_argument("--steps", type=int, default=600, help="Control steps to run.")
parser.add_argument(
    "--num_episodes",
    type=int,
    default=None,
    help="High-level tasks: episodes each env runs, instead of --steps. Reports success per task.",
)
parser.add_argument("--real_time", action="store_true", default=False, help="Run at the control rate.")
parser.add_argument("--no_export", action="store_true", default=False, help="Skip the JIT and ONNX export.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os  # noqa: E402
import time  # noqa: E402

import franka_tabletop_rl.tasks  # noqa: E402, F401
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from policy_paths import BEST_POLICY_EXPORT_DIR, sync_best_policy  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils.hydra import resolve_task_config  # noqa: E402


def _print_task_table(tracker, outcomes: list[tuple[int, bool, float]]) -> None:
    """Print episodes, successes and mean progress for every task of the stage."""
    if not outcomes:
        return
    rows: dict[str, list[tuple[bool, float]]] = {task_id: [] for task_id in tracker.task_ids}
    for index, success, progress in outcomes:
        rows[tracker.task_id_of(index)].append((success, progress))
    width = max(len(task_id) for task_id in rows)
    print(f"[RESULT] {'task'.ljust(width)}  episodes  success  progress")
    for task_id, episodes in rows.items():
        wins = sum(success for success, _ in episodes)
        mean = sum(progress for _, progress in episodes) / max(len(episodes), 1)
        print(f"[RESULT] {task_id.ljust(width)}  {len(episodes):8d}  {wins:7d}  {mean:8.2f}")
    total = sum(success for _, success, _ in outcomes)
    print(f"[RESULT] all tasks: {total}/{len(outcomes)} episodes succeeded")


def main() -> None:
    installed_version = check_rsl_rl_version()
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, overrides=hydra_args)
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    if args_cli.scene_id is not None:
        # The high-level cfg lays its task out while it is built, so rebuild it for the chosen task.
        env_cfg = type(env_cfg)(scene_id=args_cli.scene_id)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    checkpoint = sync_best_policy(
        agent_cfg.experiment_name,
        agent_cfg.load_run,
        agent_cfg.load_checkpoint,
        explicit_checkpoint=args_cli.checkpoint,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = create_rsl_rl_runner(env, agent_cfg, log_dir=None)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=unwrapped.device)

    if not args_cli.no_export:
        os.makedirs(BEST_POLICY_EXPORT_DIR, exist_ok=True)
        runner.export_policy_to_jit(BEST_POLICY_EXPORT_DIR, filename="ll_policy.pt")
        runner.export_policy_to_onnx(BEST_POLICY_EXPORT_DIR, filename="ll_policy.onnx")
        print(f"[INFO] exported policy to {BEST_POLICY_EXPORT_DIR}")

    command = unwrapped.command_manager.get_term("ee_pose")
    dt = unwrapped.step_dt
    obs = env.get_observations()
    position_error: list[torch.Tensor] = []
    orientation_error: list[torch.Tensor] = []
    tracker = getattr(command, "tracker", None)
    outcomes: list[tuple[int, bool, float]] = []
    finished = torch.zeros(unwrapped.num_envs, dtype=torch.long, device=unwrapped.device)
    for step in range(args_cli.steps if args_cli.num_episodes is None else 10**9):
        start = time.time()
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        if tracker is not None:
            # The tracker keeps the outcome of the episode that just ended until the env is reset again.
            for index in torch.nonzero(dones).flatten().tolist():
                if args_cli.num_episodes is None or finished[index] < args_cli.num_episodes:
                    outcomes.append((index, bool(tracker.last_success[index]), float(tracker.last_progress[index])))
                finished[index] += 1
            # An episode ends on success too, so envs finish at different steps. Fast ones keep running.
            if args_cli.num_episodes is not None and bool((finished >= args_cli.num_episodes).all()):
                break
        if "orientation_error" in command.metrics:
            position_error.append(command.metrics["position_error"].clone())
            orientation_error.append(command.metrics["orientation_error"].clone())
        if args_cli.real_time:
            time.sleep(max(0.0, dt - (time.time() - start)))
    if orientation_error:
        # Errors right after a new target is drawn are large by design. The median over steps and
        # envs is the tracking quality while a target is held.
        pos = torch.stack(position_error).flatten()
        ori = torch.stack(orientation_error).flatten()
        print(
            f"[RESULT] tracking position error: median {pos.median() * 100:.2f} cm, "
            f"mean {pos.mean() * 100:.2f} cm; orientation error: median {ori.median():.3f} rad, "
            f"mean {ori.mean():.3f} rad"
        )
    if tracker is not None:
        print(f"[RESULT] completed {int(tracker.completed.sum())}/{tracker.count} envs")
        _print_task_table(tracker, outcomes)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
