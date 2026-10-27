# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play highlight videos: one mp4 per camera, a 4 by 4 grid of up to 16 tasks.

``env.highlight=on`` writes ``best_policy/videos/highlight/<camera>.mp4`` for one episode, then
exits. ``env.highlight_tasks`` is indices or a range such as ``0-15``. Cells are 320 by 180
[pixels]. The rendering camera is never spawned; it only makes Kit load its rendering app.
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
    """``env.highlight=on``. Does not change the stage."""


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


def highlight_envs(
    task_count: int,
    num_envs: int,
    seed: int,
    count: int = HIGHLIGHT_COUNT,
    tasks: Sequence[int] | int | str | None = None,
) -> tuple[int, ...]:
    """Env index of each highlighted task. Env ``k`` is the first env of task ``k``.

    ``tasks`` is indices and inclusive ranges (``"0,5"``, ``"0-15"``), in that order. Empty selects
    every task, or ``count`` tasks sampled with ``seed`` when the stage is larger.

    Raises:
        ValueError: Fewer envs than tasks, or ``tasks`` is not at most 16 unique in-range indices.
    """
    if task_count <= 0 or num_envs < task_count:
        raise ValueError(f"highlight needs at least one env per task, got {num_envs} envs for {task_count} tasks")
    selected = _selected_tasks(tasks)
    if selected is not None:
        if len(selected) > HIGHLIGHT_COUNT:
            raise ValueError(f"highlight shows at most {HIGHLIGHT_COUNT} tasks, got {len(selected)}")
        if len(selected) != len(set(selected)):
            raise ValueError(f"highlight task indices must be unique, got {selected}")
        outside = [index for index in selected if index < 0 or index >= task_count]
        if outside:
            raise ValueError(f"highlight task indices must be in [0, {task_count}), got {outside}")
        return selected
    if task_count <= count:
        return tuple(range(task_count))
    return tuple(sorted(random.Random(seed).sample(range(task_count), count)))


def _selected_tasks(tasks: Sequence[int] | int | str | None) -> tuple[int, ...] | None:
    """Parsed task indices, or None when the selection is empty."""
    if tasks is None:
        return None
    if isinstance(tasks, str):
        chosen = _parse_task_spec(tasks)
    elif isinstance(tasks, int) and not isinstance(tasks, bool):
        chosen = [tasks]
    else:
        chosen = []
        for value in tasks:
            if isinstance(value, str):
                chosen.extend(_parse_task_spec(value))
            elif isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"highlight task indices must be integers or ranges like 0-15, got {value!r}")
            else:
                chosen.append(value)
    if not chosen:
        return None
    return tuple(chosen)


def _parse_task_spec(spec: str) -> list[int]:
    chosen = []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        bounds = token.split("-")
        if len(bounds) == 1:
            chosen.append(_task_index(bounds[0]))
            continue
        if len(bounds) == 2 and bounds[0].strip() and bounds[1].strip():
            start, end = _task_index(bounds[0]), _task_index(bounds[1])
            step = 1 if end >= start else -1
            chosen.extend(range(start, end + step, step))
            continue
        raise ValueError(f"highlight task indices must be integers or ranges like 0-15, got {token!r}")
    return chosen


def _task_index(token: str) -> int:
    token = token.strip()
    if not token.isdigit():
        raise ValueError(f"highlight task indices must be integers or ranges like 0-15, got {token!r}")
    return int(token)


def episode_steps(episode_length_s: float, step_dt: float) -> int:
    """Control steps in one episode. ``step_dt`` is one control step [s]."""
    if step_dt <= 0:
        raise ValueError(f"step_dt must be positive, got {step_dt}")
    return max(1, round(episode_length_s / step_dt))


def tile_views(frames: np.ndarray) -> np.ndarray:
    """Pack up to 16 cells into one 1280 by 720 [pixels] frame. Unused cells stay black."""
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
    """Renders the highlighted envs and writes one mp4 per camera. The env calls ``step`` and ``close``."""

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
        """Finish the clips. A completed episode raises ``SystemExit`` so play does not keep stepping."""
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
    prefix, _, leaf = camera.prim_path.rpartition("/")
    return camera.replace(prim_path=f"{prefix}/{leaf}Highlight", height=CELL_HEIGHT, width=CELL_WIDTH)


def _author(stage, camera: CameraCfg, indices: Sequence[int]) -> tuple[str, ...]:
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
