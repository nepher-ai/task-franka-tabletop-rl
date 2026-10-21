# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation cameras that render only in the highlighted envs.

A scene ``Camera`` sensor owns one prim per env, so it renders every env. Highlight needs a few envs
of a possibly large stage. These cameras are plain USD camera prims under the selected envs, rendered
through the same renderer and render context as the sensors, one tiled render product per camera. The
scene is untouched, and the GPU renders only the selected envs, at the 320 by 180 [pixels] cell size.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from isaaclab import sim as sim_utils
from isaaclab.renderers import CameraRenderSpec
from isaaclab.sensors.camera.camera_data import CameraData
from isaaclab.utils.math import convert_camera_frame_orientation_convention

from franka_tabletop_rl.tabletop.core.observation_spec import OBSERVATION_CAMERAS

from .highlight import CELL_HEIGHT, CELL_WIDTH, tile_views
from .highlight_cfg import HighlightCfg

_ENV_NS = "{ENV_REGEX_NS}/"


def register_highlight_renderer(sim, cfg: HighlightCfg) -> None:
    """Register the camera renderer before the scene is built, as a ``Camera`` sensor does.

    The renderer must exist before cloning and ``sim.reset()``, and the simulation must keep
    rendering every step so the clips get a fresh frame each step.
    """
    sim.require_visual_shapes()
    sim.get_or_create_backend(cfg.cam_front.renderer_cfg)
    sim.set_setting("/isaaclab/render/rtx_sensors", True)
    sim.set_setting("/physics/fabricUpdateTransformations", True)


class HighlightCameras:
    """The four observation cameras in each highlighted env, one tiled render product per camera.

    Args:
        sim: The simulation context of the play env.
        cfg: The selected highlight cfg, one camera cfg per observation camera.
        env_indices: Envs to film, in tile order.
        num_envs: Number of envs in the stage.

    Raises:
        RuntimeError: A highlighted env does not hold the camera's parent prim.
    """

    def __init__(self, sim, cfg: HighlightCfg, env_indices: Sequence[int], num_envs: int):
        self._sim = sim
        self._env_indices = tuple(env_indices)
        self._requests: list = []
        self._outputs: dict[str, torch.Tensor] = {}
        self._rendered_step = -1
        self._renderer = None
        self._render_data: list = []

        cfgs = {name: getattr(cfg, name) for name in OBSERVATION_CAMERAS}
        stage = sim.stage
        with sim_utils.use_stage(stage):
            paths = {name: self._author(stage, camera) for name, camera in cfgs.items()}
        self._renderer = sim.get_or_create_backend(cfg.cam_front.renderer_cfg)
        # Scene partitions are authored once, so every camera prim must exist before this call.
        sim.render_context.ensure_prepare_stage(stage, num_envs)
        count = len(self._env_indices)
        specs = self._renderer.supported_output_types()
        for name, camera in cfgs.items():
            spec = CameraRenderSpec(
                cfg=camera,
                device=str(sim.device),
                num_instances=count,
                camera_prim_paths=paths[name],
                view_count=count,
            )
            self._renderer.prepare_cameras(stage, spec)
            render_data = self._renderer.create_render_data(spec)
            data = CameraData.allocate(["rgb"], CELL_HEIGHT, CELL_WIDTH, count, str(sim.device), specs)
            self._renderer.set_outputs(render_data, data.output)
            self._render_data.append(render_data)
            self._requests.append((self._renderer, render_data, data))
            self._outputs[name] = data.output["rgb"]

    def _author(self, stage, camera) -> tuple[str, ...]:
        """Author one camera prim under each highlighted env and return their paths."""
        rotation = torch.tensor(camera.offset.rot, dtype=torch.float32).unsqueeze(0)
        orientation = convert_camera_frame_orientation_convention(
            rotation, origin=camera.offset.convention, target="opengl"
        )
        orientation = orientation.squeeze(0).numpy()
        relative = camera.prim_path.removeprefix(_ENV_NS)
        paths = []
        for index in self._env_indices:
            path = f"/World/envs/env_{index}/{relative}"
            parent = path.rsplit("/", 1)[0]
            if not stage.GetPrimAtPath(parent).IsValid():
                raise RuntimeError(f"highlight camera parent '{parent}' is not in the stage")
            camera.spawn.func(path, camera.spawn, translation=camera.offset.pos, orientation=orientation)
            paths.append(path)
        return tuple(paths)

    def frame(self, name: str) -> np.ndarray:
        """One 1280 by 720 [pixels] RGB uint8 frame of the camera, cells in highlight order.

        All cameras render together once per physics step. Cells that are not filled stay black.
        """
        self._render()
        views = self._outputs[name].torch
        return tile_views(views.cpu().numpy())

    def _render(self) -> None:
        step = self._sim.get_physics_step_count()
        if step == self._rendered_step:
            return
        self._sim.render_context.render_into_cameras(self._requests, step)
        self._rendered_step = step

    def close(self) -> None:
        """Release the render products."""
        for render_data in self._render_data:
            self._renderer.cleanup(render_data)
        self._render_data.clear()
        self._requests.clear()
        self._outputs.clear()
