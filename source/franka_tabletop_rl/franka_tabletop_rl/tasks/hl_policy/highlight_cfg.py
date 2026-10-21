# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config switch of the highlight videos.

``env.highlight=on`` on a play command selects :class:`HighlightCfg`. The selected cfg holds the four
highlight cameras, which tells Isaac Lab to start Kit with camera rendering before the env is built. No
other part of the cfg changes. Without the selector the field is ``None``.
"""

from __future__ import annotations

from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg

from franka_tabletop_rl.tabletop.core import sensors

from .highlight import CELL_HEIGHT, CELL_WIDTH

# Highlight cameras sit next to the observation cameras, so they get their own prim names.
_SUFFIX = "Highlight"


def cell_camera(maker) -> CameraCfg:
    """An observation camera at the 320 by 180 [pixels] highlight cell size, under a highlight prim name."""
    cfg = maker()
    prefix, _, leaf = cfg.prim_path.rpartition("/")
    return cfg.replace(prim_path=f"{prefix}/{leaf}{_SUFFIX}", height=CELL_HEIGHT, width=CELL_WIDTH)


@configclass
class HighlightCfg:
    """The observation cameras of the highlight videos, one field per camera name.

    The cameras are rendered only in the highlighted envs. They are not scene sensors.
    """

    cam_exterior_left: CameraCfg = cell_camera(sensors.exterior_left_camera)
    cam_exterior_right: CameraCfg = cell_camera(sensors.exterior_right_camera)
    cam_wrist: CameraCfg = cell_camera(sensors.wrist_camera)
    cam_front: CameraCfg = cell_camera(sensors.front_camera)


@configclass
class HighlightPreset(PresetCfg):
    """``env.highlight=on`` records highlight videos. The default records none."""

    default: HighlightCfg | None = None
    on: HighlightCfg = HighlightCfg()
