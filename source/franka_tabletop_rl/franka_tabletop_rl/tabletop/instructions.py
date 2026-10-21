# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Instruction templates. Hidden phase pools are files in the benchmark bundle."""

from __future__ import annotations

import json
from pathlib import Path

VERBS = ("put", "place", "move", "set")
PUBLIC_SALT = "public"

_TEMPLATES = {
    "place_in_container": "{verb} the {item} in the {container}",
    "place_relative": "{verb} the {item} to the {side} of the {reference}",
    "stack_on": "{verb} the {top} on the {support}",
    "sort_by_attribute": "{verb} the {keep} in the {container} and leave the {leave} out",
    "multi_place": "{verb} the {first} and the {second} in the {container}",
}


def phrase(task_id: str, index: int, names: dict[str, str], salt: str = PUBLIC_SALT, side: str = "left") -> str:
    """Render one instruction. ``salt`` and ``index`` pick the verb."""
    verb = VERBS[(index + _mix(salt)) % len(VERBS)]
    return _TEMPLATES[task_id].format(verb=verb, side=side, **names)


def load_pool(path: Path, task_id: str) -> list[str] | None:
    """Load a hidden pool file. Returns None when the file is absent."""
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data.get(task_id)
    return list(values) if values else None


def _mix(salt: str) -> int:
    return sum(ord(char) for char in salt)
