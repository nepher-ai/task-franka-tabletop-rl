# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hierarchical Franka tabletop tasks."""

from __future__ import annotations

from typing import Any

# Gym registration. Importing the package loads the task entry points.
from . import tasks  # noqa: F401

__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    """Lazy exports for eval-nav. ``task_ids_of`` does not import Isaac Lab. ``wrap_for_eval`` does."""
    if name == "wrap_for_eval":
        from .tasks.hl_policy.eval_compat import wrap_for_eval

        return wrap_for_eval
    if name == "task_ids_of":
        from .benchmark.bundle import task_ids_of

        return task_ids_of
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["task_ids_of", "tasks", "wrap_for_eval"]
