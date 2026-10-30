# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env config for film runs: the play stage plus a recorder, a post-roll, and hidden parked objects."""

from __future__ import annotations

import torch

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ..tasks.hl_policy import mdp
from ..tasks.hl_policy.hl_env_cfg import HLEnvCfg_PLAY
from .latch import LOST, SUCCESS
from .recorder import FilmRecordersCfg
from .settings import FilmSettings


@configclass
class FilmEnvCfg(HLEnvCfg_PLAY):
    """Play stage for film. ``film`` is filled by :func:`apply_film_cfg`."""

    film: object | None = None
    """:class:`FilmSettings` of the run."""


def _advance_latch(env, pose_command_name: str) -> None:
    success = mdp.task_success(env, pose_command_name)
    lost = mdp.object_lost(env, pose_command_name)
    env.film_latch.update(int(env.common_step_counter), env.episode_length_buf, success, lost)


def film_task_success(env, pose_command_name: str = "ee_pose") -> torch.Tensor:
    """True once the success has held for the post-roll [s] after it first held."""
    _advance_latch(env, pose_command_name)
    return env.film_latch.done(SUCCESS, env.episode_length_buf)


def film_object_lost(env, pose_command_name: str = "ee_pose") -> torch.Tensor:
    """True once the post-roll [s] after an object left the table has passed."""
    _advance_latch(env, pose_command_name)
    return env.film_latch.done(LOST, env.episode_length_buf)


def hide_parked_objects(env, env_ids) -> None:
    """Hide every body that its env does not use, so a parked object never shows in a wide shot.

    Visibility is the only thing that changes. Physics and the layout stay as they are.
    """
    from pxr import UsdGeom

    plan = env.cfg.stage_plan
    stage = env.sim.stage
    hidden = []
    for index in range(plan.num_envs):
        root = f"{env.scene.env_ns}/env_{index}"
        used = set(plan.phase_slots[0][index].values())
        hidden += [f"{root}/{name}" for name in plan.entities if name not in used]
        active = int(plan.active_surface[0][index])
        for surface_index, name in enumerate(plan.surface_entities):
            if surface_index != active:
                hidden.append(f"{root}/{env.scene[name].cfg.prim_path.rsplit('/', 1)[-1]}")
    for path in hidden:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()


def apply_film_cfg(cfg: FilmEnvCfg, settings: FilmSettings) -> FilmEnvCfg:
    """Wire the recorder, the post-roll terminations, and the hidden parked objects into ``cfg``.

    Everything that changes how an episode unfolds is the same for scout and shoot, so a shoot replays its
    scout. Only the post-roll delays a reset, and it comes from the shot.
    """
    cfg.film = settings
    cfg.recorders = FilmRecordersCfg()
    cfg.terminations.success = DoneTerm(func=film_task_success)
    cfg.terminations.object_lost = DoneTerm(func=film_object_lost)
    cfg.events.film_hide_parked = EventTerm(func=hide_parked_objects, mode="prestartup")
    cfg.commands.ee_pose.debug_vis = False
    cfg.commands.grip_cmd.debug_vis = False
    return cfg
