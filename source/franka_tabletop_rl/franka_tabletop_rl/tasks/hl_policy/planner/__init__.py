# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

__all__ = ["ClassicalTabletopPlanner", "Stage", "build_planner", "drop_offsets", "reach_limit"]


def __getattr__(name: str) -> Any:
    """Load the planner on first use so grasp helpers can be imported on their own."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .machine import ClassicalTabletopPlanner, Stage, build_planner, drop_offsets, reach_limit

    exports = {
        "ClassicalTabletopPlanner": ClassicalTabletopPlanner,
        "Stage": Stage,
        "build_planner": build_planner,
        "drop_offsets": drop_offsets,
        "reach_limit": reach_limit,
    }
    globals().update(exports)
    return exports[name]
