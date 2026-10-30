# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The shot catalog: what to film, with which camera, and how to pick the episode.

Camera times are seconds after the first rendered frame, which is the start of the pre-roll. The
episode starts ``pre_roll_s`` later. Points are in the shot's ``frame``:

* ``env``: offsets [m] from the filmed env's origin, which is the robot base.
* ``grid``: offsets [m] in x and y from the centroid of all env origins, and an absolute z [m].
* ``world``: absolute simulation coordinates [m].

A point can also be an :class:`Anchor`, which follows a live or stored position.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import numpy as np

from .rig import Key, Vec3

CONTROL_HZ = 15
"""Policy rate [Hz]. A control step is the unit of the episode timeline."""

OUTCOMES = ("success", "lost", "any")
FRAMES = ("env", "grid", "world")
CUT_ANCHORS = ("start", "episode", "first_close", "outcome")
PASSES = ("rgb", "points")
DEFAULT_ENV_ID = "franka-tabletop-rl-v0"


@dataclass(frozen=True)
class Anchor:
    """A named position plus an offset [m] in world axes.

    ``tool`` follows the gripper tool point of the filmed env. ``slot:<name>`` is the start position of
    that task slot. ``table`` is the center of the work surface top.
    """

    name: str
    offset: Vec3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class StageSpec:
    """Everything that decides how a run unfolds. A scout and its shoot must share it exactly."""

    env_id: str = DEFAULT_ENV_ID
    task_id: str = "all"
    num_envs: int | None = None
    jitter_m: float = 0.0
    seed: int = 1

    def tag(self) -> str:
        """File-name friendly identity, for example ``franka-tabletop-rl-v0_all_auto_j0.0_s1``."""
        envs = "auto" if self.num_envs is None else str(self.num_envs)
        return f"{self.env_id}_{self.task_id}_{envs}_j{self.jitter_m:g}_s{self.seed}"

    def as_dict(self) -> dict:
        return {
            "env_id": self.env_id,
            "task_id": self.task_id,
            "num_envs": self.num_envs,
            "jitter_m": self.jitter_m,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class Cut:
    """Suggested in and out points for the editor, from the planner timeline.

    The window runs from ``before_s`` [s] before the anchor to ``after_s`` [s] after it. ``start`` is the
    episode start, ``first_close`` the first gripper close, and ``outcome`` the success or loss. ``episode``
    starts at the episode start and stretches to the outcome, so ``after_s`` is a tail after the outcome.
    ``require`` names planner stages that must begin inside the window for an episode to qualify.
    """

    anchor: str
    before_s: float
    after_s: float
    require: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.anchor not in CUT_ANCHORS:
            raise ValueError(f"cut anchor must be one of {CUT_ANCHORS}, got {self.anchor!r}")


@dataclass(frozen=True)
class Shot:
    """One shot of the shot list."""

    name: str
    number: int
    title: str
    stage: StageSpec
    keys: tuple[Key, ...]
    hfov_deg: float
    cut: Cut
    focus: tuple[str, ...] = ()
    outcome: str = "success"
    frame: str = "env"
    easing: str = "smooth"
    resolution: tuple[int, int] = (1920, 1080)
    fps: int = 60
    passes: tuple[str, ...] = ("rgb",)
    duration_s: tuple[float | None, float | None] = (None, None)
    """Allowed time [s] from the episode start to the outcome. ``None`` leaves that side open."""
    pre_roll_s: float = 2.0
    post_roll_s: float = 3.0
    follow_tau_s: float = 0.0
    """Smoothing time [s] of the ``tool`` anchor. Zero follows the gripper exactly."""
    all_envs: bool = False
    """Film every env in one take, from the first reset, instead of one episode of one env."""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}, got {self.outcome!r}")
        if self.frame not in FRAMES:
            raise ValueError(f"frame must be one of {FRAMES}, got {self.frame!r}")
        unknown = set(self.passes) - set(PASSES)
        if unknown or "rgb" not in self.passes:
            raise ValueError(f"passes must include rgb and only use {PASSES}, got {self.passes}")
        if not self.keys:
            raise ValueError(f"shot {self.name} has no camera keys")
        if self.pre_roll_s < 0.0 or self.post_roll_s < 0.0:
            raise ValueError("pre_roll_s and post_roll_s must not be negative")

    def partitioned(self) -> bool:
        """Whether RTX should draw only the filmed env. Shots that show several envs draw them all."""
        return not self.all_envs

    def with_overrides(self, **changes) -> Shot:
        """Copy with fields replaced, for example the resolution or the frame rate."""
        return replace(self, **changes)


class PointResolver:
    """Turns a shot point into a world position [m].

    ``origin`` is the offset [m] of the shot frame. ``statics`` maps anchor names to world positions, and
    ``tool`` returns the live tool position in world coordinates.
    """

    def __init__(
        self,
        origin: np.ndarray,
        statics: Mapping[str, np.ndarray] | None = None,
        tool: Callable[[], np.ndarray] | None = None,
    ):
        self._origin = np.asarray(origin, dtype=np.float64)
        self._statics = dict(statics or {})
        self._tool = tool

    def __call__(self, point: object) -> np.ndarray:
        if isinstance(point, Anchor):
            return self._anchor(point.name) + np.asarray(point.offset, dtype=np.float64)
        return self._origin + np.asarray(point, dtype=np.float64)

    def _anchor(self, name: str) -> np.ndarray:
        if name == "tool":
            if self._tool is None:
                raise ValueError("the tool anchor needs a filmed env")
            return np.asarray(self._tool(), dtype=np.float64)
        if name not in self._statics:
            raise KeyError(f"unknown anchor {name!r}; known: {sorted(self._statics)}")
        return np.asarray(self._statics[name], dtype=np.float64)


_DEFAULT = StageSpec()
# One stage for the whole shot list: every task of v0, one env each. A scout of it serves shots 1, 2, 4, 6, 7.
_LONG_RUN_TASKS = (
    "food_packing_by_color",
    "block_stacking_specified_order",
    "block_stacking_order_agnostic",
    "stack_white_mugs",
    "stack_yellow_on_red",
    "bananas_out_of_bin",
)
# Tasks the tracker usually finishes, so the grasp shots get a clean lift and carry.
_GRASP_TASKS = ("stack_yellow_on_red", "white_mug_in_center_of_table", "bananas_out_of_bin", "unstack_rubiks_cube")
# The failure and success pair needs many tries, so it scouts a bigger, jittered stage.
_SEARCH = StageSpec(num_envs=64, jitter_m=0.02)
_PAIR_VIEW = (Key(0.0, (1.60, -1.00, 0.50), (0.62, 0.04, 0.14)),)

SHOTS: dict[str, Shot] = {
    shot.name: shot
    for shot in (
        Shot(
            name="hero",
            number=1,
            title="Hero grasp",
            stage=_DEFAULT,
            focus=("stack_yellow_on_red",),
            # Low and side-on, on the side away from the tall house of this layout, so the carry comes toward
            # the lens. The drift is slow.
            keys=(
                Key(0.0, (1.05, -0.55, 0.17), (0.48, 0.06, 0.10)),
                Key(15.0, (0.95, -0.42, 0.19), (0.48, 0.06, 0.10)),
            ),
            hfov_deg=50.0,
            cut=Cut("first_close", before_s=3.0, after_s=4.5, require=("LIFT", "CARRY")),
            notes="6-8 s. Arm reaches, closes, lifts, and carries toward the camera.",
        ),
        Shot(
            name="busy_table",
            number=2,
            title="Busy table, push-in",
            stage=_DEFAULT,
            focus=("grab_a_bagel",),
            outcome="any",
            keys=(
                Key(0.0, (1.95, 0.60, 1.20), (0.40, 0.0, 0.10)),
                Key(10.0, (1.40, 0.30, 0.90), (0.40, 0.0, 0.12)),
            ),
            hfov_deg=55.0,
            cut=Cut("start", before_s=0.0, after_s=5.0),
            passes=("rgb", "points"),
            notes="5 s. The points pass is shot 3: same take, same camera, planner point cloud and markers.",
        ),
        Shot(
            name="long_run",
            number=4,
            title="One long run",
            stage=_DEFAULT,
            focus=_LONG_RUN_TASKS,
            keys=(Key(0.0, (1.60, -0.62, 0.62), (0.55, 0.0, 0.08)),),
            hfov_deg=52.0,
            cut=Cut("episode", before_s=0.0, after_s=1.0),
            duration_s=(15.0, 40.0),
            notes="15-40 s unbroken pack or stack, first grasp to everything placed. The tracker rarely needs 20 s.",
        ),
        Shot(
            name="failure",
            number=5,
            title="Failure: an object falls off the table",
            stage=_SEARCH,
            outcome="lost",
            keys=_PAIR_VIEW,
            hfov_deg=64.0,
            cut=Cut("outcome", before_s=3.5, after_s=1.5),
            notes="5 s. Natural failures only, found by scouting. Same task and camera as the success take.",
        ),
        Shot(
            name="success",
            number=5,
            title="Success: a clean run",
            stage=_SEARCH,
            outcome="success",
            keys=_PAIR_VIEW,
            hfov_deg=64.0,
            cut=Cut("outcome", before_s=4.0, after_s=1.0),
            notes="5 s. Same task and camera as the failure take.",
        ),
        Shot(
            name="closeup",
            number=6,
            title="Gripper close-up",
            stage=_DEFAULT,
            focus=_GRASP_TASKS,
            keys=(
                Key(0.0, Anchor("tool", (0.25, -0.36, 0.14)), Anchor("tool", (0.0, 0.0, -0.02))),
                Key(12.0, Anchor("tool", (0.25, -0.36, 0.14)), Anchor("tool", (0.0, 0.0, -0.02))),
            ),
            hfov_deg=45.0,
            cut=Cut("first_close", before_s=1.2, after_s=1.3),
            resolution=(3840, 2160),
            follow_tau_s=0.15,
            notes="2.5 s at 60 fps. Use --fps 120 --resolution 1920x1080 for extra slow motion.",
        ),
        Shot(
            name="parallel",
            number=7,
            title="Parallel environments",
            stage=_DEFAULT,
            outcome="any",
            frame="grid",
            keys=(
                Key(0.0, (0.5, -10.2, 11.2), (0.5, 0.0, 0.0)),
                Key(9.0, (0.5, -9.2, 10.2), (0.5, 0.0, 0.0)),
            ),
            hfov_deg=58.0,
            cut=Cut("start", before_s=0.0, after_s=5.0),
            all_envs=True,
            notes="5 s. All envs in one grid, moving from the first step.",
        ),
    )
}

# The shot list as numbered by the request, mapped to (shot, pass).
SHOT_LIST: tuple[tuple[int, str, str], ...] = (
    (1, "hero", "rgb"),
    (2, "busy_table", "rgb"),
    (3, "busy_table", "points"),
    (4, "long_run", "rgb"),
    (5, "failure", "rgb"),
    (5, "success", "rgb"),
    (6, "closeup", "rgb"),
    (7, "parallel", "rgb"),
)


def get_shot(name: str) -> Shot:
    """The named shot.

    Raises:
        KeyError: ``name`` is not in the catalog.
    """
    try:
        return SHOTS[name]
    except KeyError:
        raise KeyError(f"unknown shot {name!r}; choose from {sorted(SHOTS)}") from None


def episode_seconds(steps: int | float) -> float:
    """Seconds [s] of ``steps`` control steps."""
    return float(steps) / CONTROL_HZ


__all__ = [
    "Anchor",
    "Cut",
    "PointResolver",
    "SHOTS",
    "SHOT_LIST",
    "Shot",
    "StageSpec",
    "episode_seconds",
    "get_shot",
]
