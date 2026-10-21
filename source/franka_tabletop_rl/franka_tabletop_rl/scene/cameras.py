# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation cameras of the tabletop stage."""

from __future__ import annotations

# Observation names, in the order the observation dict lists them.
OBSERVATION_CAMERAS = ("cam_exterior_left", "cam_exterior_right", "cam_wrist", "cam_front")


def requested_cameras(spec: dict | None) -> tuple[str, ...]:
    """Cameras named by ``spec``, or every observation camera when it names none.

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


def selected_cameras(names: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Observation cameras to keep, in observation order.

    ``None`` and an empty sequence keep none.

    Raises:
        ValueError: A name is not an observation camera.
    """
    if not names:
        return ()
    return requested_cameras({"cameras": names})
