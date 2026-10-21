# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Give the scene cloner one world per env of a mixed stage."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from franka_tabletop_rl.benchmark.stage_plan import StagePlan, explicit_worlds


@contextmanager
def planned_worlds(plan: StagePlan) -> Iterator[None]:
    """Make the scene build the worlds of ``plan`` while its envs are constructed.

    The cloner would otherwise enumerate every combination of variants. This swaps in the
    ``num_envs`` worlds the plan names, and restores the cloner afterwards.
    """
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
