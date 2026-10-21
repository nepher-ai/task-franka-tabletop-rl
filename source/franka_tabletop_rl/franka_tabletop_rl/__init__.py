# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hierarchical Franka tabletop pick-and-place."""

from __future__ import annotations

from typing import Any

# Gym registration. No Isaac Lab import happens at package load.
from . import tasks  # noqa: F401

__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    """Lazy export used by eval-nav through ``task_module``."""
    if name == "wrap_for_eval":
        from .tasks.hl_policy.eval_compat import wrap_for_eval

        return wrap_for_eval
    if name == "tasks_of_scene":
        from .tasks.hl_policy.task_state import tasks_of_scene

        return tasks_of_scene
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["tasks", "tasks_of_scene", "wrap_for_eval"]
