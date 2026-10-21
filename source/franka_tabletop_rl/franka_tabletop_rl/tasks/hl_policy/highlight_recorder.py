# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Video recorder that writes the highlight grid."""

from __future__ import annotations

import logging
import os

import numpy as np
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

from isaaclab.envs.utils.video_recorder import VideoRecorder, _parse_source

from .highlight import CELL_HEIGHT, CELL_WIDTH, GRID
from .highlight_cameras import HighlightCameras

logger = logging.getLogger(__name__)


class HighlightRecorder(VideoRecorder):
    """One camera's highlight clip: the highlighted envs tiled into a 1280 by 720 frame.

    The file is ``<output_dir>/<camera>.mp4``. Frames stream to the encoder, so a clip does not
    stay in memory.

    Args:
        cfg: Recorder cfg whose source is ``sensor:<camera>``.
        env: The play env.
        cameras: The highlight cameras that render the frames.
    """

    def __init__(self, cfg, env, cameras: HighlightCameras):
        super().__init__(cfg, env)
        self._cameras = cameras
        self._camera = _parse_source(cfg.source)[1]
        self._writer: FFMPEG_VideoWriter | None = None

    def _clip_path(self, index: int) -> str:
        del index
        return os.path.join(self._effective_output_dir(), f"{self._camera}.mp4")

    def _get_frame(self) -> np.ndarray | None:
        """Write this step's frame to the encoder. Nothing is kept for ``step`` to buffer."""
        if self._frame_error_logged:
            return None
        try:
            frame = self._cameras.frame(self._camera)
            if self._writer is None:
                os.makedirs(self._effective_output_dir(), exist_ok=True)
                size = (GRID * CELL_WIDTH, GRID * CELL_HEIGHT)
                self._writer = FFMPEG_VideoWriter(self._clip_path(self._clip_index), size, self.cfg.fps)
            self._writer.write_frame(frame)
        except Exception:
            logger.exception("[HighlightRecorder] %s: frame capture failed, the clip stops.", self._camera)
            self._frame_error_logged = True
        return None

    def _close_clip(self) -> None:
        writer, self._writer = self._writer, None
        self._recording = False
        if writer is not None:
            writer.close()
            print(f"[INFO] Highlight video: {self._clip_path(self._clip_index)}")

    def close(self) -> None:
        """Finish the clip if it is still open."""
        self._close_clip()
