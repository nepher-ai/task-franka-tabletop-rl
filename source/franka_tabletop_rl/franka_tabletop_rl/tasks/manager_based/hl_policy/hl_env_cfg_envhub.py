# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""EnvHub entry of the hierarchical environment.

eval-nav builds the cfg as ``cfg_class(env_id=<benchmark env>, scene_id=<env_scene>)`` and then sets
``scene.num_envs``. The env_scene of ``task-franka-tabletop-rl.yaml`` is ``"all"``, so one stage holds
every task of the benchmark and env ``i`` runs task ``i % T``. A single task id selects just that task.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .hl_env_cfg import HLEnvCfg


@configclass
class HLEnvCfg_Envhub(HLEnvCfg):
    """``HLEnvCfg`` addressed by an EnvHub env id and a scene id (``"all"`` or a task id)."""


@configclass
class HLEnvCfg_Envhub_PLAY(HLEnvCfg_Envhub):
    """Local play of the benchmark: one env per task, or more copies through ``scene.num_envs``."""
