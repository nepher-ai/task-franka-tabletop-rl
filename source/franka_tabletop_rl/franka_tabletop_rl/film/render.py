# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offscreen film camera: one free RTX camera, posed every frame, plus the mp4 and PNG writers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import rig

CAMERA_NAME = "FilmCamera"
CLIPPING_RANGE_M = (0.05, 100.0)


def film_camera_cfg(width: int, height: int, hfov_deg: float, prim_path: str, antialiasing: str | None, partitioned):
    """Camera config of the film camera.

    Args:
        width: Image width [pixels].
        height: Image height [pixels].
        hfov_deg: Horizontal field of view [deg].
        prim_path: Camera prim path.
        antialiasing: RTX antialiasing mode name, or ``None`` to keep the launch setting.
        partitioned: Whether RTX draws only the env that owns the camera. Shots of the whole grid turn it off.
    """
    from isaaclab import sim as sim_utils

    from franka_tabletop_rl.tasks.hl_policy.highlight import rendering_camera

    base = rendering_camera()
    renderer = base.renderer_cfg.replace(enable_scene_partitioning=bool(partitioned))
    if antialiasing is not None:
        settings = renderer.global_settings.replace(antialiasing_mode=antialiasing)
        renderer = renderer.replace(global_settings=settings)
    lens = sim_utils.PinholeCameraCfg(
        focal_length=rig.focal_length(hfov_deg),
        horizontal_aperture=rig.HORIZONTAL_APERTURE,
        vertical_aperture=rig.vertical_aperture(width, height),
        focus_distance=400.0,
        clipping_range=CLIPPING_RANGE_M,
    )
    return base.replace(prim_path=prim_path, height=height, width=width, spawn=lens, renderer_cfg=renderer)


def film_renderer_cfg(antialiasing: str | None, partitioned: bool):
    """The renderer config the film camera uses, for registering the backend before the scene is built."""
    return film_camera_cfg(16, 16, 60.0, "/World/FilmProbe", antialiasing, partitioned).renderer_cfg


def enable_film_rendering(sim, antialiasing: str | None, partitioned: bool) -> None:
    """Register the film renderer before the scene is built, as a scene camera would."""
    sim.require_visual_shapes()
    sim.get_or_create_backend(film_renderer_cfg(antialiasing, partitioned))
    sim.set_setting("/isaaclab/render/rtx_sensors", True)
    sim.set_setting("/physics/fabricUpdateTransformations", True)


class FilmCamera:
    """A free camera whose pose is written every frame and whose image is read back as ``uint8`` RGB.

    Args:
        env: The unwrapped env.
        width: Image width [pixels].
        height: Image height [pixels].
        hfov_deg: Horizontal field of view [deg].
        env_index: Env that owns the camera. ``None`` puts the camera outside every env, so it sees all of them.
        antialiasing: RTX antialiasing mode name, or ``None``.
    """

    def __init__(
        self,
        env,
        width: int,
        height: int,
        hfov_deg: float,
        env_index: int | None,
        antialiasing: str | None = None,
    ):
        import torch
        import warp as wp

        from isaaclab import sim as sim_utils
        from isaaclab.renderers import CameraRenderSpec
        from isaaclab.sensors.camera.camera_data import CameraData
        from isaaclab.sim.views import FrameView

        self._torch, self._wp = torch, wp
        self.env = env
        self.width, self.height = int(width), int(height)
        self.hfov_deg = float(hfov_deg)
        self.device = str(env.device)
        stage = env.sim.stage
        parent = "/World" if env_index is None else f"{env.scene.env_ns}/env_{int(env_index)}"
        if not stage.GetPrimAtPath(parent).IsValid():
            raise RuntimeError(f"film camera parent '{parent}' is not in the stage")
        self.path = f"{parent}/{CAMERA_NAME}"
        cfg = film_camera_cfg(self.width, self.height, hfov_deg, self.path, antialiasing, env_index is not None)
        with sim_utils.use_stage(stage):
            cfg.spawn.func(self.path, cfg.spawn, translation=(0.0, 0.0, 1.0), orientation=(0.0, 0.0, 0.0, 1.0))
        renderer = env.sim.get_or_create_backend(cfg.renderer_cfg)
        env.sim.render_context.ensure_prepare_stage(stage, env.num_envs)
        spec = CameraRenderSpec(
            cfg=cfg,
            device=self.device,
            num_instances=1,
            camera_prim_paths=(self.path,),
            view_count=1,
        )
        renderer.prepare_cameras(stage, spec)
        self._render_data = renderer.create_render_data(spec)
        data = CameraData.allocate(["rgb"], self.height, self.width, 1, self.device, renderer.supported_output_types())
        renderer.set_outputs(self._render_data, data.output)
        self._renderer = renderer
        self._data = data
        self._requests = [(renderer, self._render_data, data)]
        self._view = FrameView(self.path, device=self.device, stage=stage)
        self._index = wp.array(np.zeros(1, dtype=np.int32), device=self.device)

    def set_pose(self, eye: np.ndarray, target: np.ndarray) -> None:
        """Place the camera at ``eye`` looking at ``target``, both world positions [m]."""
        torch, wp = self._torch, self._wp
        position = torch.tensor([list(map(float, eye))], dtype=torch.float32, device=self.device)
        orientation = torch.tensor([rig.look_at_quat(eye, target).tolist()], dtype=torch.float32, device=self.device)
        with self._view.xform_world_space_writer() as writer:
            writer.set_poses(
                wp.from_torch(position.contiguous(), dtype=wp.vec3f),
                wp.from_torch(orientation.contiguous(), dtype=wp.vec4f),
                self._index,
            )

    def capture(self) -> np.ndarray:
        """Render the scene as it is now and return RGB ``[height, width, 3]``, ``uint8``."""
        sim = self.env.sim
        sim.render_context.render_into_cameras(self._requests, sim.get_physics_step_count())
        frame = self._data.output["rgb"].torch[0, ..., :3]
        return frame.cpu().numpy()

    def close(self) -> None:
        self._renderer.cleanup(self._render_data)


class VideoSink:
    """H.264 mp4 writer. Frames are RGB ``uint8`` ``[height, width, 3]``.

    Args:
        path: Output file. Its folder is created.
        width: Image width [pixels]. It must be even.
        height: Image height [pixels]. It must be even.
        fps: Frame rate [Hz].
        crf: x264 quality, lower is better. 12 is visually lossless.
    """

    def __init__(self, path: Path, width: int, height: int, fps: int, crf: int = 12):
        from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

        if width % 2 or height % 2:
            raise ValueError(f"mp4 needs an even size, got {width} by {height}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.width, self.height = int(width), int(height)
        self.frames = 0
        self._writer = FFMPEG_VideoWriter(
            str(path), (self.width, self.height), fps, codec="libx264", preset="slow", ffmpeg_params=["-crf", str(crf)]
        )

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"frame is {frame.shape}, the video is {(self.height, self.width, 3)}")
        self._writer.write_frame(np.ascontiguousarray(frame, dtype=np.uint8))
        self.frames += 1

    def close(self) -> None:
        self._writer.close()


def save_png(path: Path, frame: np.ndarray) -> None:
    """Write one RGB frame as a PNG."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(frame, dtype=np.uint8)).save(path)
