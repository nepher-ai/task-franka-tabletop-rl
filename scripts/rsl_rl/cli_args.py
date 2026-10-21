# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CLI argument helpers for the RSL-RL scripts."""

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser) -> None:
    """Add the RSL-RL arguments shared by training and playback."""
    group = parser.add_argument_group("rsl_rl", description="RSL-RL agent arguments.")
    group.add_argument("--experiment_name", type=str, default=None, help="Experiment folder name for logs.")
    group.add_argument("--run_name", type=str, default=None, help="Run name suffix appended to the log directory.")
    group.add_argument("--resume", action="store_true", default=False, help="Resume from the latest checkpoint.")
    group.add_argument("--load_run", type=str, default=None, help="Run folder to resume from.")
    group.add_argument("--checkpoint", type=str, default=None, help="Exact checkpoint file to load.")
    group.add_argument("--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"})
    group.add_argument("--log_project_name", type=str, default=None, help="Project name for wandb or neptune.")


def update_rsl_rl_cfg(agent_cfg: RslRlBaseRunnerCfg, args_cli: argparse.Namespace) -> RslRlBaseRunnerCfg:
    """Override runner fields with the parsed command-line values."""
    if getattr(args_cli, "seed", None) is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10_000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume:
        agent_cfg.resume = True
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name
    return agent_cfg
