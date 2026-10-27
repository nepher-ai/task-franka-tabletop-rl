# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Seeded arm start-pose noise.

eval-nav reseeds the env with the same seed before every round of episodes, so the global generator
repeats. Offsets come from the seed, the round and the env index instead.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def arm_start_offsets(
    seed: int,
    round_index: int,
    env_indices: Sequence[int],
    half_range_rad: float,
    joint_count: int = 7,
) -> np.ndarray:
    """Joint offsets [rad] for the start of an episode, shape ``[len(env_indices), joint_count]``.

    The same ``seed``, ``round_index`` and env index always give the same row.

    Args:
        seed: Evaluation seed passed to ``reset``.
        round_index: Number of full-env resets before this one.
        env_indices: Env index of each row.
        half_range_rad: Each offset is uniform in ``[-half_range_rad, half_range_rad]`` [rad].
        joint_count: Arm joints per env.
    """
    rows = []
    for env_index in env_indices:
        entropy = [int(seed) % 2**32, int(round_index), int(env_index)]
        rng = np.random.default_rng(np.random.SeedSequence(entropy))
        rows.append(rng.uniform(-half_range_rad, half_range_rad, joint_count))
    return np.asarray(rows, dtype=np.float64).reshape(len(rows), joint_count)
