# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Video recorder that writes the 16-task highlight grid."""

from __future__ import annotations

import os

import numpy as np
import torch

from isaaclab.envs.utils.video_recorder import VideoRecorder, _parse_source

from .highlight import CELL_HEIGHT, CELL_WIDTH, HIGHLIGHT_COUNT, as_uint8_rgb, label_cell, tile_views


class HighlightRecorder(VideoRecorder):
    """One camera's highlight clip: env 0 through 15 tiled into a 1280 by 720 frame.

    The file is ``<output_dir>/<camera>.mp4``.
    """

    def _clip_path(self, index: int) -> str:
        del index
        camera = _parse_source(self.cfg.source)[1]
        return os.path.join(self._effective_output_dir(), f"{camera}.mp4")

    def _close_clip(self) -> None:
        if self._frames:
            print(f"[INFO] Highlight video: {self._clip_path(self._clip_index)}")
        super()._close_clip()

    def _frame_from_sensor(self, name: str, gt_type: str = "rgb") -> np.ndarray | None:
        del gt_type  # The highlight clip is the RGB grid.
        scene = getattr(self._env, "scene", None)
        sensor = getattr(scene, "sensors", {}).get(name)
        if sensor is None:
            raise RuntimeError(f"[HighlightRecorder] Sensor '{name}' is not in the scene.")
        raw = sensor.data.output["rgb"]
        batch = raw.torch if hasattr(raw, "torch") else raw
        count = batch.shape[0]
        if count < HIGHLIGHT_COUNT:
            raise RuntimeError(
                f"[HighlightRecorder] {name} has {count} views, the highlight grid needs {HIGHLIGHT_COUNT}."
            )
        labels = _task_ids(self._env)
        cells = [
            label_cell(_fit_cell(_cell(batch, index)), labels[index]) for index in range(HIGHLIGHT_COUNT)
        ]
        return tile_views(np.stack(cells))


def _task_ids(env: object) -> tuple[str, ...]:
    """Task id of env 0 through 15. Highlight play puts one chosen task on each env."""
    names = tuple(getattr(getattr(env, "cfg", None), "task_ids", ()) or ())
    if len(names) < HIGHLIGHT_COUNT:
        raise RuntimeError(
            f"[HighlightRecorder] the stage has {len(names)} tasks, the grid needs {HIGHLIGHT_COUNT}."
        )
    return names[:HIGHLIGHT_COUNT]


def _cell(batch: torch.Tensor | np.ndarray, index: int) -> np.ndarray:
    view = batch[index]
    array = view.detach().cpu().numpy() if isinstance(view, torch.Tensor) else np.asarray(view)
    if array.shape[0] in (3, 4) and array.ndim == 3:
        array = np.moveaxis(array, 0, -1)
    return as_uint8_rgb(array)


def _fit_cell(image: np.ndarray) -> np.ndarray:
    """Resize one RGB view to the highlight cell, or return it when it already matches."""
    if image.shape[0] == CELL_HEIGHT and image.shape[1] == CELL_WIDTH:
        return image
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0).float()
    tensor = torch.nn.functional.interpolate(
        tensor, size=(CELL_HEIGHT, CELL_WIDTH), mode="bilinear", align_corners=False
    )
    return tensor.round().clamp_(0, 255).to(dtype=torch.uint8).squeeze(0).permute(1, 2, 0).contiguous().numpy()
