# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera paths, look-at orientation, pinhole intrinsics, and point projection.

Everything here is a pure NumPy function, so it runs without Isaac Sim. The camera follows the USD
convention: it looks along its local -Z axis, +Y is up, and +X is right. Pixels follow the image
convention: ``u`` grows to the right and ``v`` grows downward.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

EASINGS = ("linear", "smooth", "spline")
"""``smooth`` eases in and out of every segment. ``spline`` keeps the speed continuous through keys."""

# Horizontal aperture [cm] of the scene cameras. Only the focal length to aperture ratio matters.
HORIZONTAL_APERTURE = 20.955

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class Key:
    """One camera key: ``t`` [s] after the first rendered frame, with eye and target points.

    The points can be anything the ``resolve`` function of :func:`evaluate_path` understands.
    """

    t: float
    eye: object
    target: object


def frame_count(seconds: float, fps: float) -> int:
    """Whole frames in ``seconds`` [s] at ``fps`` [Hz]."""
    return max(0, round(float(seconds) * float(fps)))


def smoothstep(u: float) -> float:
    """Cubic ease in and out on ``[0, 1]``."""
    u = min(max(float(u), 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _hermite(p0: np.ndarray, p1: np.ndarray, m0: np.ndarray, m1: np.ndarray, u: float, width: float) -> np.ndarray:
    u2, u3 = u * u, u * u * u
    return (
        (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * width * m0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * width * m1
    )


def evaluate_path(
    keys: Sequence[Key],
    t: float,
    resolve: Callable[[object], np.ndarray],
    easing: str = "smooth",
) -> tuple[np.ndarray, np.ndarray]:
    """Eye and target [m] at time ``t`` [s]. Times outside the keys hold the first or last key.

    Raises:
        ValueError: There are no keys, the keys are not strictly increasing in time, or ``easing`` is unknown.
    """
    if easing not in EASINGS:
        raise ValueError(f"easing must be one of {EASINGS}, got {easing!r}")
    if not keys:
        raise ValueError("a camera path needs at least one key")
    times = [float(key.t) for key in keys]
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        raise ValueError(f"camera key times must strictly increase, got {times}")
    eyes = [np.asarray(resolve(key.eye), dtype=np.float64) for key in keys]
    targets = [np.asarray(resolve(key.target), dtype=np.float64) for key in keys]
    if t <= times[0]:
        return eyes[0].copy(), targets[0].copy()
    if t >= times[-1]:
        return eyes[-1].copy(), targets[-1].copy()
    index = max(i for i, start in enumerate(times) if start <= t and i < len(times) - 1)
    width = times[index + 1] - times[index]
    u = (t - times[index]) / width
    if easing == "spline":
        return (
            _spline(eyes, times, index, u),
            _spline(targets, times, index, u),
        )
    if easing == "smooth":
        u = smoothstep(u)
    eye = eyes[index] + (eyes[index + 1] - eyes[index]) * u
    target = targets[index] + (targets[index + 1] - targets[index]) * u
    return eye, target


def _spline(points: list[np.ndarray], times: list[float], index: int, u: float) -> np.ndarray:
    def tangent(i: int) -> np.ndarray:
        if i <= 0 or i >= len(points) - 1:
            return np.zeros(3)
        return (points[i + 1] - points[i - 1]) / (times[i + 1] - times[i - 1])

    width = times[index + 1] - times[index]
    return _hermite(points[index], points[index + 1], tangent(index), tangent(index + 1), u, width)


def follow(previous: np.ndarray | None, target: np.ndarray, dt: float, time_constant: float) -> np.ndarray:
    """Exponential smoothing of a followed point. ``time_constant`` [s] of zero follows exactly."""
    target = np.asarray(target, dtype=np.float64)
    if previous is None or time_constant <= 0.0:
        return target.copy()
    blend = 1.0 - math.exp(-float(dt) / float(time_constant))
    return previous + (target - previous) * blend


def look_at_matrix(eye: Sequence[float], target: Sequence[float], up: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera to world rotation ``[3, 3]``. Its columns are right, up, and back (the camera's X, Y, Z).

    Raises:
        ValueError: The eye and the target coincide.
    """
    eye_v = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye_v
    length = float(np.linalg.norm(forward))
    if length < 1e-9:
        raise ValueError("camera eye and target coincide")
    forward /= length
    right = np.cross(forward, np.asarray(up, dtype=np.float64))
    if float(np.linalg.norm(right)) < 1e-6:
        # Looking straight along the up hint: fall back to world X as the side axis.
        right = np.cross(forward, np.array([1.0, 0.0, 0.0]) if abs(forward[0]) < 0.9 else np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    return np.stack([right, true_up, -forward], axis=1)


def matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    """Unit quaternion ``(x, y, z, w)`` of a rotation matrix, with ``w >= 0``."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        quat = np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        quat = np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        quat = np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        quat = np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s])
    quat /= np.linalg.norm(quat)
    return -quat if quat[3] < 0.0 else quat


def look_at_quat(eye: Sequence[float], target: Sequence[float], up: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """World orientation ``(x, y, z, w)`` of a USD camera at ``eye`` that looks at ``target``."""
    return matrix_to_quat_xyzw(look_at_matrix(eye, target, up))


def quat_xyzw_to_matrix(quat: Sequence[float]) -> np.ndarray:
    """Rotation matrix ``[3, 3]`` of an ``(x, y, z, w)`` quaternion. It is normalized first."""
    x, y, z, w = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def focal_length(hfov_deg: float, horizontal_aperture: float = HORIZONTAL_APERTURE) -> float:
    """Focal length for a horizontal field of view [deg], in the unit of ``horizontal_aperture``."""
    if not 0.0 < hfov_deg < 180.0:
        raise ValueError(f"hfov_deg must be in (0, 180), got {hfov_deg}")
    return horizontal_aperture / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))


def vertical_aperture(width: int, height: int, horizontal_aperture: float = HORIZONTAL_APERTURE) -> float:
    """Vertical aperture that gives square pixels at ``width`` by ``height``."""
    return horizontal_aperture * height / width


def intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Pinhole matrix ``[3, 3]`` [pixels] that matches the scene cameras: ``fx = fy = W f / aperture``."""
    f = focal_length(hfov_deg)
    fx = width * f / HORIZONTAL_APERTURE
    return np.array([[fx, 0.0, width * 0.5], [0.0, fx, height * 0.5], [0.0, 0.0, 1.0]])


def project(
    points: np.ndarray, eye: Sequence[float], rotation: np.ndarray, matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pixels ``[N, 2]`` (u, v) and depths ``[N]`` [m] of world ``points`` ``[N, 3]``.

    ``rotation`` is the camera to world matrix from :func:`look_at_matrix`. A depth at or below zero is behind
    the camera and has no valid pixel.
    """
    cam = (np.asarray(points, dtype=np.float64) - np.asarray(eye, dtype=np.float64)) @ rotation
    depth = -cam[:, 2]
    safe = np.where(depth > 1e-9, depth, np.nan)
    u = matrix[0, 2] + matrix[0, 0] * cam[:, 0] / safe
    v = matrix[1, 2] - matrix[1, 1] * cam[:, 1] / safe
    return np.stack([u, v], axis=1), depth
