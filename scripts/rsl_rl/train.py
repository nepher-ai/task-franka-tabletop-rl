# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the low-level tool-pose tracker with RSL-RL PPO.

Examples:
    python scripts/rsl_rl/train.py --task Nepher-FrankaTabletopRL-LL-v0
    python scripts/rsl_rl/train.py --task Nepher-FrankaTabletopRL-LL-v0 --num_envs 2048
    python scripts/rsl_rl/train.py --task Nepher-FrankaTabletopRL-LL-v0 --resume
"""

"""Launch Isaac Sim first: modules that import ``pxr`` must load after the app starts."""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train the low-level tracker with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of each recorded video [steps].")
parser.add_argument("--video_interval", type=int, default=2000, help="Steps between recordings.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--task", type=str, default="Nepher-FrankaTabletopRL-LL-v0", help="Registered gym id.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point key.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override the iteration count.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os  # noqa: E402
from datetime import datetime  # noqa: E402

import franka_tabletop_rl.tasks  # noqa: E402, F401
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.utils.io import dump_yaml  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import resolve_task_config  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main() -> None:
    installed_version = check_rsl_rl_version()
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, overrides=hydra_args)
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment to: {log_root_path}")
    # Resolve the resume path before the new run folder exists, so it is not picked as "latest".
    resume_path = None
    if agent_cfg.resume:
        explicit = args_cli.checkpoint
        if explicit and os.path.isfile(os.path.expanduser(explicit)):
            resume_path = os.path.abspath(os.path.expanduser(explicit))
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Resuming from: {resume_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "train"),
            step_trigger=lambda step: step % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = create_rsl_rl_runner(env, agent_cfg, log_dir=log_dir)
    runner.add_git_repo_to_log(__file__)
    if resume_path is not None:
        runner.load(resume_path)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=agent_cfg.init_at_random_ep_len
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
