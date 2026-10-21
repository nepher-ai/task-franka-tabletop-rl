# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Path to the robot USD shipped next to this package."""

from pathlib import Path

_ASSETS = Path(__file__).resolve().parent / "assets"


def packaged_asset(*parts: str) -> str:
    """Absolute path of a file under ``franka_tabletop/assets``."""
    return str(_ASSETS.joinpath(*parts))
