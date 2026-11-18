# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import site
from pathlib import Path

import pytest

# The checkout imports without an editable install. pytest's pythonpath puts this directory first.
site.addsitedir(str(Path(__file__).resolve().parents[1] / "source" / "franka_tabletop_rl"))


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_isaaclab: Isaac Lab is not importable")


@pytest.fixture
def requires_isaaclab():
    """Skip when this interpreter cannot import Isaac Lab."""
    return pytest.importorskip("isaaclab")
