# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Low-level tool-pose tracker.

Registered environments:
  Nepher-FrankaTabletopRL-LL-v0       training (4096 envs, observation noise on)
  Nepher-FrankaTabletopRL-LL-Play-v0  evaluation (32 envs, noise off)
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Nepher-FrankaTabletopRL-LL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ll_env_cfg:LLEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:LLPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Nepher-FrankaTabletopRL-LL-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ll_env_cfg:LLEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:LLPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
