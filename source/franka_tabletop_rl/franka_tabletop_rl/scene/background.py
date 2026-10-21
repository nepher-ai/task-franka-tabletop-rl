# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Swap the stage dome when a background phase changes.

The dome is one light for the whole stage, so every env sees the same background. Phases take
turns: the texture changes only when every env has finished its episode.
"""

from __future__ import annotations


def apply_dome(texture: str, intensity: float, texture_format: str, *, visible: bool = True) -> None:
    """Point ``/World/background`` at one latlong texture.

    Args:
        texture: Absolute path of the dome image.
        intensity: Dome intensity.
        texture_format: ``latlong`` or ``angular``.
        visible: Whether the dome shows in camera rays.
    """
    import omni.usd

    prim = omni.usd.get_context().get_stage().GetPrimAtPath("/World/background")
    _set(prim, "inputs:texture:file", texture)
    _set(prim, "inputs:intensity", float(intensity))
    _set(prim, "inputs:texture:format", str(texture_format))
    _set(prim, "inputs:visibleInPrimaryRay", bool(visible))


def _set(prim, name: str, value) -> None:
    attribute = prim.GetAttribute(name)
    if attribute:
        attribute.Set(value)
