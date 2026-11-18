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

from ..ll_policy import agents


def _register() -> None:
    """Register the high-level tasks. Skipped when gymnasium is not installed."""
    try:
        import gymnasium as gym
    except ImportError:
        return

    runner = f"{agents.__name__}.ppo_cfg:LLPPORunnerCfg"
    published = f"{agents.__name__}.ppo_cfg:LLPPORunnerCfg_ENVHUB_PLAY"

    gym.register(
        id="Nepher-FrankaTabletopRL-HL-Play-v0",
        entry_point=f"{__name__}.env:HLTabletopEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.hl_env_cfg:HLEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": runner,
            "default_agent": "rsl_rl",
        },
    )

    gym.register(
        id="Nepher-FrankaTabletopRL-HL-EnvhubPlay-v0",
        entry_point=f"{__name__}.env:HLTabletopEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.hl_env_cfg:HLEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": published,
            "default_agent": "rsl_rl",
        },
    )


_register()
