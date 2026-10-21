# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hierarchical environment: classical planner plus the learned tracker.

Registered environments:
  Nepher-FrankaTabletopRL-HL-Play-v0        every benchmark task in one stage (``scene_id="all"``)
  Nepher-FrankaTabletopRL-HL-EnvhubPlay-v0  the eval-nav entry, ``cfg_class(env_id=..., scene_id="all")``

Both load the tracker checkpoint of the low-level task as their policy, and the benchmark
assets from the nepher cache of ``franka-tabletop-rl-v0``.
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
        "env_cfg_entry_point": f"{__name__}.hl_env_cfg_envhub:HLEnvCfg_Envhub_PLAY",
        "rsl_rl_cfg_entry_point": _PUBLISHED,
        "default_agent": "rsl_rl",
    },
)
