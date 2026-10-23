# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "source" / "franka_tabletop_rl"
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

# Importing the package registers Gym tasks and needs Isaac Lab. Pure tests load submodules only.
try:
    import isaaclab_tasks  # noqa: F401
except ImportError:
    import types

    def _namespace(name: str, folder: Path) -> None:
        if name in sys.modules:
            return
        stub = types.ModuleType(name)
        stub.__path__ = [str(folder)]
        stub.__package__ = name
        sys.modules[name] = stub

    root = PACKAGE / "franka_tabletop_rl"
    _namespace("franka_tabletop_rl", root)
    _namespace("franka_tabletop_rl.tasks", root / "tasks")
    _namespace("franka_tabletop_rl.tasks.hl_policy", root / "tasks" / "hl_policy")
    _namespace("franka_tabletop_rl.tasks.hl_policy.planner", root / "tasks" / "hl_policy" / "planner")
    _namespace("franka_tabletop_rl.tasks.hl_policy.mdp", root / "tasks" / "hl_policy" / "mdp")


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_bundle: the EnvHub bundle is not in the nepher cache")
    config.addinivalue_line("markers", "requires_isaaclab: Isaac Lab is not importable")


@pytest.fixture
def requires_bundle():
    """Skip when ``env_id`` is not cached. Pass the id, or use the default v1 bundle."""

    def _need(env_id: str = "franka-tabletop-rl-v1"):
        from franka_tabletop_rl.benchmark.bundle import resolve_bundle

        try:
            return resolve_bundle(env_id)
        except FileNotFoundError as exc:
            pytest.skip(str(exc))

    return _need


@pytest.fixture
def requires_isaaclab():
    """Skip when this interpreter cannot import Isaac Lab."""
    return pytest.importorskip("isaaclab")
