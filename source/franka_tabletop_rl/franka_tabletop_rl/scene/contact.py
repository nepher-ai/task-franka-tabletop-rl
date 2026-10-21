# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gripper contact against the tabletop objects.

A grasp is a finger force above ``GRASP_FORCE_N``. The same threshold says the
gripper has released the object. Filter order is the slot order the scene mounted.
"""

from __future__ import annotations

import numpy as np

FINGER_SENSORS = ("gripper_left", "gripper_right")
GRASP_FORCE_N = 0.1


def slots_in_contact(scene, num_envs: int, slots: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Per-slot contact with either finger. Each value is a bool array of length ``num_envs``.

    Args:
        scene: Interactive scene that owns the finger contact sensors.
        num_envs: Number of vectorized environments.
        slots: Slot names in the same order as the sensor filters.
    """
    held = {slot: np.zeros(num_envs, dtype=bool) for slot in slots}
    if not slots:
        return held
    for name in FINGER_SENSORS:
        sensor = scene[name]
        matrix = sensor.data.normal_force_matrix_w
        if matrix is None:
            continue
        magnitude = _filter_magnitudes(matrix)
        if magnitude.shape[-1] != len(slots):
            raise ValueError(f"contact sensor {name} has {magnitude.shape[-1]} filters, expected {len(slots)}")
        for index, slot in enumerate(slots):
            held[slot] |= magnitude[:, index] > GRASP_FORCE_N
    return held


def _filter_magnitudes(matrix) -> np.ndarray:
    """Force magnitude [N] per environment and filter body."""
    values = matrix.torch if hasattr(matrix, "torch") else matrix
    array = values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)
    magnitude = np.linalg.norm(array, axis=-1)
    while magnitude.ndim > 2:
        magnitude = magnitude.max(axis=1)
    return magnitude
