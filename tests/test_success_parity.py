# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The success code is duplicated in ``task-franka-tabletop``. The two copies have to stay the same."""

from __future__ import annotations

from pathlib import Path

import pytest

RL = Path(__file__).resolve().parents[1] / "source" / "franka_tabletop_rl" / "franka_tabletop_rl" / "benchmark"
ORIGINAL = (
    Path(__file__).resolve().parents[2] / "task-franka-tabletop" / "source" / "franka_tabletop" / "franka_tabletop"
)

pytestmark = pytest.mark.skipif(not ORIGINAL.is_dir(), reason="task-franka-tabletop is not checked out")


@pytest.mark.parametrize("name", ["geometry", "success", "predicates"])
def test_shared_modules_are_identical(name: str):
    assert (RL / f"{name}.py").read_text(encoding="utf-8") == (ORIGINAL / f"{name}.py").read_text(encoding="utf-8")


def _between(text: str, start: str, end: str | None) -> str:
    begin = text.index(start)
    return text[begin : text.index(end, begin)] if end else text[begin:]


def test_spec_schema_and_parser_are_identical():
    rl = (RL / "spec.py").read_text(encoding="utf-8")
    original = (ORIGINAL / "spec.py").read_text(encoding="utf-8")
    # The local families sit between the dataclasses and the parser, and the spec docstring differs.
    assert _between(rl, "OPS = (", "@dataclass(frozen=True)\nclass TabletopTaskSpec") == _between(
        original, "OPS = (", "@dataclass(frozen=True)\nclass TabletopTaskSpec"
    )
    assert _between(rl, "def spec_from_success(", None) == _between(original, "def spec_from_success(", None)
