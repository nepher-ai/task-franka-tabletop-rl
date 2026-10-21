# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Give the scene cloner one world per env of a mixed stage."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from ..scenes.mixed import MixedPlan, explicit_worlds


@contextmanager
def planned_worlds(plan: MixedPlan) -> Iterator[None]:
    """Make the scene build the worlds of ``plan`` while its envs are constructed.

    :class:`~isaaclab.scene.InteractiveScene` asks the cloner for the legal combinations
    of scene assets. For positions with a different mesh in every env, that is the product
    of all variant counts. This swaps in :func:`explicit_worlds` for the duration of the
    ``with`` block and restores the cloner afterwards, also when construction fails.
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
