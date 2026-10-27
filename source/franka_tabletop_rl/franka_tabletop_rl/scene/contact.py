# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gripper contact. A grasp is a finger force above ``GRASP_FORCE_N``. Filter order is slot order."""

from __future__ import annotations

import numpy as np

FINGER_SENSORS = ("gripper_left", "gripper_right")
GRASP_FORCE_N = 0.1


def slots_in_contact(scene, num_envs: int, slots: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Per-slot contact with either finger, as bool arrays of length ``num_envs``.

    ``slots`` is in sensor-filter order.
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
    values = matrix.torch if hasattr(matrix, "torch") else matrix
    array = values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)
    magnitude = np.linalg.norm(array, axis=-1)
    while magnitude.ndim > 2:
        magnitude = magnitude.max(axis=1)
    return magnitude
