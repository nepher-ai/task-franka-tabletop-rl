# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Planner plus tracker.

Registered environments:
  Nepher-FrankaTabletopRL-HL-Play-v0        every task in one stage (``task_id="all"``)
  Nepher-FrankaTabletopRL-HL-EnvhubPlay-v0  eval-nav entry

Both load the tracker checkpoint and the nepher cache of the cfg's ``env_id``.
"""

import gymnasium as gym

from ..ll_policy import agents

_RUNNER = f"{agents.__name__}.ppo_cfg:LLPPORunnerCfg"
_PUBLISHED = f"{agents.__name__}.ppo_cfg:LLPPORunnerCfg_ENVHUB_PLAY"

gym.register(
    id="Nepher-FrankaTabletopRL-HL-Play-v0",
    entry_point=f"{__name__}.env:HLTabletopEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hl_env_cfg:HLEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _RUNNER,
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Nepher-FrankaTabletopRL-HL-EnvhubPlay-v0",
    entry_point=f"{__name__}.env:HLTabletopEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hl_env_cfg:HLEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _PUBLISHED,
        "default_agent": "rsl_rl",
    },
)
