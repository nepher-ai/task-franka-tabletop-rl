# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register the tabletop RL tasks. Imported through the isaaclab.tasks entry point."""

from franka_tabletop_rl.tasks.checkpoint import prefer_checkpoint_file

from . import hl_policy, ll_policy  # noqa: F401
from .hl_policy.eval_compat import wrap_for_eval  # noqa: F401

prefer_checkpoint_file()
