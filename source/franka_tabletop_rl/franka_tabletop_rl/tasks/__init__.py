# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register the tabletop RL tasks. Imported through the isaaclab.tasks entry point."""

from __future__ import annotations

from typing import Any

from . import hl_policy, ll_policy  # noqa: F401


def __getattr__(name: str) -> Any:
    """Load ``wrap_for_eval`` on demand. It needs torch, and the task package does not."""
    if name == "wrap_for_eval":
        from .hl_policy.eval_compat import wrap_for_eval

        globals()["wrap_for_eval"] = wrap_for_eval
        return wrap_for_eval
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["hl_policy", "ll_policy", "wrap_for_eval"]
