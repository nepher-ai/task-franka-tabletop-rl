# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A rigid-object record. Which meshes exist is declared by the benchmark, not by this package."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectAsset:
    """One rigid mesh."""

    name: str
    size_m: tuple[float, float, float]
    kind: str
    usd: str

    def bundle_usd(self) -> str:
        """Path of the mesh relative to a benchmark bundle."""
        if self.usd.startswith("assets/"):
            return self.usd
        return f"assets/objects/{self.usd}"


# Seeds for local play and demo recording. Benchmark episodes use the bundle's own scenes.
PUBLIC_SEEDS = (11, 23, 37, 41, 59)
