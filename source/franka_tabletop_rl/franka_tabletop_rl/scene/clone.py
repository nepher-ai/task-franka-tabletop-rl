# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Give the scene cloner one world per env."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from franka_tabletop_rl.benchmark.stage_plan import StagePlan, explicit_worlds


@contextmanager
def planned_worlds(plan: StagePlan) -> Iterator[None]:
    """Build the ``num_envs`` worlds in ``plan`` instead of every variant combination."""
    from isaaclab import cloner

    original = cloner.make_valid_clone_combinations
    choices = plan.choice_by_name()

    def make_worlds(asset_names, variant_counts, clone_combinations=None, *, all_asset_names=None):
        del clone_combinations, all_asset_names
        return explicit_worlds(asset_names, variant_counts, choices, plan.num_envs)

    cloner.make_valid_clone_combinations = make_worlds
    try:
        yield
    finally:
        cloner.make_valid_clone_combinations = original
