# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Highlight videos for play.

``env.highlight=on`` records one mp4 per observation camera. Each frame is a 4 by 4 grid of up to 16
tasks, 1280 by 720 [pixels], lasting one episode (``episode_length_s``). The files are
``best_policy/videos/highlight/<camera>.mp4``. Play exits once those files are written.

The play stage is unchanged. A scene camera renders every env, so this renders only the highlighted
envs, at the 320 by 180 [pixels] cell size. ``play_mode`` puts one camera config on the cfg so Kit
starts its rendering app, and that camera is never spawned.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np

from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg

from franka_tabletop_rl.paths import play_video_dir
from franka_tabletop_rl.scene import sensors
from franka_tabletop_rl.scene.cameras import OBSERVATION_CAMERAS

HIGHLIGHT_COUNT = 16
GRID = 4
CELL_HEIGHT = 180
CELL_WIDTH = 320

_CAMERAS = {
    "cam_exterior_left": sensors.exterior_left_camera,
    "cam_exterior_right": sensors.exterior_right_camera,
    "cam_wrist": sensors.wrist_camera,
    "cam_front": sensors.front_camera,
}


@configclass
class HighlightCfg:
    """Selected by ``env.highlight=on``. The stage itself is not part of this cfg."""


@configclass
class HighlightPreset(PresetCfg):
    """``env.highlight=on`` records highlight videos. The default records none."""

    default: HighlightCfg | None = None
    on: HighlightCfg = HighlightCfg()


def rendering_camera() -> CameraCfg:
    """One camera config that makes Kit load its rendering app. The scene does not spawn it."""
    camera = sensors.front_camera()
    return camera.replace(prim_path="{ENV_REGEX_NS}/Highlight", height=CELL_HEIGHT, width=CELL_WIDTH)


def enable_rendering(sim) -> None:
    """Register the camera renderer before the scene is built, as a scene camera would."""
    sim.require_visual_shapes()
    sim.get_or_create_backend(rendering_camera().renderer_cfg)
    sim.set_setting("/isaaclab/render/rtx_sensors", True)
    sim.set_setting("/physics/fabricUpdateTransformations", True)


def highlight_envs(task_count: int, num_envs: int, seed: int, count: int = HIGHLIGHT_COUNT) -> tuple[int, ...]:
    """Env indices to film, one per highlighted task, in benchmark order.

    Env ``i`` runs task ``i % task_count``, so the first env of task ``k`` is env ``k``. A stage with
    at most ``count`` tasks highlights all of them. A larger stage highlights ``count`` tasks sampled
    with ``seed``, then sorted back into benchmark order.

    Raises:
        ValueError: The stage has no task or fewer envs than tasks.
    """
    if task_count <= 0 or num_envs < task_count:
        raise ValueError(f"highlight needs at least one env per task, got {num_envs} envs for {task_count} tasks")
    if task_count <= count:
        return tuple(range(task_count))
    return tuple(sorted(random.Random(seed).sample(range(task_count), count)))


def episode_steps(episode_length_s: float, step_dt: float) -> int:
    """Control steps in one episode. ``step_dt`` is one control step [s]."""
    if step_dt <= 0:
        raise ValueError(f"step_dt must be positive, got {step_dt}")
    return max(1, round(episode_length_s / step_dt))


def tile_views(frames: np.ndarray) -> np.ndarray:
    """Pack up to 16 cell images into one 1280 by 720 [pixels] frame. Unused cells stay black.

    Raises:
        ValueError: The batch is empty, larger than the grid, or not of the cell size.
    """
    count = frames.shape[0] if frames.ndim == 4 else 0
    if frames.ndim != 4 or not 0 < count <= HIGHLIGHT_COUNT or frames.shape[1:] != (CELL_HEIGHT, CELL_WIDTH, 3):
        raise ValueError(
            f"highlight frames must be [1..{HIGHLIGHT_COUNT}, {CELL_HEIGHT}, {CELL_WIDTH}, 3], got {list(frames.shape)}"
        )
    cells = np.zeros((HIGHLIGHT_COUNT, CELL_HEIGHT, CELL_WIDTH, 3), dtype=np.uint8)
    cells[:count] = frames
    rows = cells.reshape(GRID, GRID, CELL_HEIGHT, CELL_WIDTH, 3).transpose(0, 2, 1, 3, 4)
    return np.ascontiguousarray(rows.reshape(GRID * CELL_HEIGHT, GRID * CELL_WIDTH, 3))


class Highlight:
    """Renders the four cameras in the highlighted envs and writes one mp4 per camera.

    Frames go straight to the encoder. ``step`` and ``close`` are what the env calls on its video
    recorders, so this needs no recorder cfg.
    """

    def __init__(self, env, env_indices: Sequence[int], episode_length_s: float):
        from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

        from isaaclab import sim as sim_utils
        from isaaclab.renderers import CameraRenderSpec
        from isaaclab.sensors.camera.camera_data import CameraData

        self._steps = episode_steps(episode_length_s, env.step_dt)
        self._fps = max(1, round(1.0 / env.step_dt))
        self._sim = env.sim
        self._indices = tuple(env_indices)
        self._count = 0
        self._closed = False
        self._checked = False
        self._writer_cls = FFMPEG_VideoWriter
        self._writers: dict = {}
        self._outputs: dict = {}
        self._render_data: list = []

        cfgs = {name: _cell_camera(_CAMERAS[name]()) for name in OBSERVATION_CAMERAS}
        stage = env.sim.stage
        with sim_utils.use_stage(stage):
            paths = {name: _author(stage, cfg, self._indices) for name, cfg in cfgs.items()}
        renderer = env.sim.get_or_create_backend(rendering_camera().renderer_cfg)
        env.sim.render_context.ensure_prepare_stage(stage, env.num_envs)
        specs = renderer.supported_output_types()
        requests = []
        for name, cfg in cfgs.items():
            spec = CameraRenderSpec(
                cfg=cfg,
                device=str(env.device),
                num_instances=len(self._indices),
                camera_prim_paths=paths[name],
                view_count=len(self._indices),
            )
            renderer.prepare_cameras(stage, spec)
            render_data = renderer.create_render_data(spec)
            data = CameraData.allocate(["rgb"], CELL_HEIGHT, CELL_WIDTH, len(self._indices), str(env.device), specs)
            renderer.set_outputs(render_data, data.output)
            self._render_data.append(render_data)
            requests.append((renderer, render_data, data))
            self._outputs[name] = data.output["rgb"]
        self._renderer = renderer
        self._requests = requests
        names = ", ".join(env.cfg.task_ids[index] for index in self._indices)
        print(f"[INFO] Highlight: {len(self._indices)} tasks for {episode_length_s} s")
        print(f"[INFO] Highlight tasks: {names}")

    def step(self) -> None:
        """Write one frame of every camera."""
        if self._closed or self._count >= self._steps:
            return
        self._count += 1
        self._sim.render_context.render_into_cameras(self._requests, self._sim.get_physics_step_count())
        for name, output in self._outputs.items():
            frame = tile_views(output.torch.cpu().numpy())
            if not self._checked:
                self._checked = True
                if float(frame.mean()) < 2.0:
                    print(f"[WARN] Highlight: {name} looks black. The rendering app may not be active.")
            self._writer(name).write_frame(frame)
        if self._count == self._steps:
            self.close()

    def close(self) -> None:
        """Finish the clips and release the render products.

        A finished episode ends the process. The play loop otherwise keeps stepping after the
        files are written.
        """
        if self._closed:
            return
        finished = self._count >= self._steps
        self._closed = True
        for name, writer in self._writers.items():
            writer.close()
            print(f"[INFO] Highlight video: {play_video_dir() / name}.mp4")
        for render_data in self._render_data:
            self._renderer.cleanup(render_data)
        self._render_data.clear()
        if finished:
            raise SystemExit(0)

    def _writer(self, name: str):
        writer = self._writers.get(name)
        if writer is None:
            directory = play_video_dir()
            directory.mkdir(parents=True, exist_ok=True)
            size = (GRID * CELL_WIDTH, GRID * CELL_HEIGHT)
            writer = self._writer_cls(str(directory / f"{name}.mp4"), size, self._fps)
            self._writers[name] = writer
        return writer


def _cell_camera(camera: CameraCfg) -> CameraCfg:
    """The observation camera at the highlight cell size, under its own prim name."""
    prefix, _, leaf = camera.prim_path.rpartition("/")
    return camera.replace(prim_path=f"{prefix}/{leaf}Highlight", height=CELL_HEIGHT, width=CELL_WIDTH)


def _author(stage, camera: CameraCfg, indices: Sequence[int]) -> tuple[str, ...]:
    """Author one camera prim under each highlighted env and return the paths."""
    import torch

    from isaaclab.utils.math import convert_camera_frame_orientation_convention

    rotation = torch.tensor(camera.offset.rot, dtype=torch.float32).unsqueeze(0)
    orientation = convert_camera_frame_orientation_convention(
        rotation, origin=camera.offset.convention, target="opengl"
    )
    orientation = orientation.squeeze(0).numpy()
    relative = camera.prim_path.removeprefix("{ENV_REGEX_NS}/")
    paths = []
    for index in indices:
        path = f"/World/envs/env_{index}/{relative}"
        parent = path.rsplit("/", 1)[0]
        if not stage.GetPrimAtPath(parent).IsValid():
            raise RuntimeError(f"highlight camera parent '{parent}' is not in the stage")
        camera.spawn.func(path, camera.spawn, translation=camera.offset.pos, orientation=orientation)
        paths.append(path)
    return tuple(paths)
