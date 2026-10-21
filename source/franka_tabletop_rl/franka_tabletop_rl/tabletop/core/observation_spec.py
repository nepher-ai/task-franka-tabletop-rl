# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Which cameras a brain reads, and at what image size.

A brain declares this in ``agent.yaml`` and the server repeats it in ``hello``.
The environment then renders and copies only those cameras. Leaving the block
out keeps every camera at the rendered size.
"""

from __future__ import annotations

# Observation names, in the order the observation dict lists them.
OBSERVATION_CAMERAS = ("cam_exterior_left", "cam_exterior_right", "cam_wrist")


def requested_cameras(spec: dict | None) -> tuple[str, ...]:
    """Cameras to send. All of them when the spec names none.

    Raises:
        ValueError: The spec names a camera that is not an observation.
    """
    names = (spec or {}).get("cameras")
    if not names:
        return OBSERVATION_CAMERAS
    unknown = sorted(set(names) - set(OBSERVATION_CAMERAS))
    if unknown:
        raise ValueError(f"observation cameras {unknown} are not in {OBSERVATION_CAMERAS}")
    return tuple(name for name in OBSERVATION_CAMERAS if name in names)


def requested_image_hw(spec: dict | None) -> tuple[int, int] | None:
    """Image size [pixels] to send as ``(height, width)``, or ``None`` for the rendered size."""
    size = (spec or {}).get("image_hw")
    if not size:
        return None
    height, width = (int(value) for value in size)
    if height < 1 or width < 1:
        raise ValueError(f"observation image_hw must be positive, got {size}")
    return (height, width)


def unused_cameras(spec: dict | None) -> tuple[str, ...]:
    """Observation cameras the brain does not read."""
    wanted = requested_cameras(spec)
    return tuple(name for name in OBSERVATION_CAMERAS if name not in wanted)
