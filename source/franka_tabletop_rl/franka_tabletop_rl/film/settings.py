# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""What a film run was asked to do. Plain data, so the env, the recorder, and the script agree."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .selection import TakePlan
from .shots import Shot

MODES = ("scout", "shoot", "preview")


@dataclass(frozen=True)
class FilmSettings:
    """One film run.

    Args:
        shot: The shot, with any resolution and fps override already applied.
        mode: ``scout`` logs episodes and renders nothing, ``shoot`` films ``plan``, and ``preview`` films
            ``plan`` small and also saves stills.
        out_dir: Folder of the take files.
        pass_name: ``rgb`` or ``points``.
        plan: The episode to film. Required unless ``mode`` is ``scout``.
        stills_s: Take times [s] that ``preview`` saves as PNG files.
        antialiasing: RTX antialiasing mode name, or ``None`` to keep the launch setting.
        meta: Facts the take file should carry, such as the checkpoint name.
    """

    shot: Shot
    mode: str
    out_dir: Path
    pass_name: str = "rgb"
    plan: TakePlan | None = None
    stills_s: tuple[float, ...] = ()
    antialiasing: str | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.pass_name not in self.shot.passes:
            raise ValueError(f"{self.shot.name} has no '{self.pass_name}' pass, only {self.shot.passes}")
        if self.mode != "scout" and self.plan is None:
            raise ValueError(f"mode {self.mode!r} needs a take plan")

    @property
    def renders(self) -> bool:
        """Whether this run writes a video."""
        return self.mode != "scout"

    @property
    def needs_camera(self) -> bool:
        """Whether Kit renders the scene. The points pass is drawn from poses and needs no RTX."""
        return self.renders and self.pass_name == "rgb"


def preview_stills(shot: Shot, plan: TakePlan) -> tuple[float, ...]:
    """Take times [s] for preview stills: the episode start, both cut points, and every camera key."""
    pre = shot.pre_roll_s
    end = pre + (plan.stop_s if plan.stop_s is not None else plan.window.end_s)
    times = {pre, plan.window.start_s + pre, plan.window.end_s + pre, 0.0}
    times.update(key.t for key in shot.keys)
    middle = 0.5 * (plan.window.start_s + plan.window.end_s) + pre
    times.add(middle)
    return tuple(sorted(round(t, 2) for t in times if 0.0 <= t <= end))
