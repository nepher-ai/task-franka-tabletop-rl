# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read ``assets/stage.json``: backgrounds, work surfaces, and the mount under the arm."""

from __future__ import annotations

import json
from pathlib import Path


def read_stage(bundle: str | Path) -> dict:
    """Backgrounds, work surfaces, mount, light, and ground.

    Plural keys are lists. A single object under the singular key is a one-entry list.
    """
    path = Path(bundle) / "assets" / "stage.json"
    if not path.is_file():
        raise FileNotFoundError(f"benchmark must declare assets/stage.json under {bundle}")
    payload = dict(json.loads(path.read_text(encoding="utf-8")))
    backgrounds, background = _records(payload, "backgrounds", "background")
    surfaces, surface = _records(payload, "surfaces", "surface")
    for record in backgrounds:
        if "texture" not in record:
            raise ValueError(f"background {record.get('name', '')!r} is missing texture")
    for record in surfaces:
        if "usd" not in record:
            raise ValueError(f"surface {record.get('name', '')!r} is missing usd")
    fixture = payload.get("fixture")
    if not isinstance(fixture, dict) or "usd" not in fixture:
        raise ValueError("assets/stage.json fixture is missing usd")
    payload["backgrounds"] = backgrounds
    payload["background"] = background
    payload["surfaces"] = surfaces
    payload["surface"] = surface
    return payload


def resolve_background(stage: dict, name: str | None = None) -> dict:
    """One background record. ``name`` overrides the stage default."""
    return _select(stage["backgrounds"], name if name is not None else stage.get("background"), "background")


def xyzw_from_wxyz(quat: tuple[float, ...]) -> tuple[float, float, float, float]:
    """Isaac Lab xyzw rotation from a benchmark ``(w, x, y, z)`` quaternion."""
    w, x, y, z = (float(value) for value in quat)
    return (x, y, z, w)


def surface_pose(stage_surface: dict, scene: dict) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Table pose as position [m] and wxyz. A scene ``surface_pose`` overrides the stage."""
    override = scene.get("surface_pose")
    source = override if isinstance(override, dict) else stage_surface
    pos = tuple(float(value) for value in source.get("pos", stage_surface.get("pos", (0.0, 0.0, 0.0))))
    rot = tuple(float(value) for value in source.get("rot", stage_surface.get("rot", (1.0, 0.0, 0.0, 0.0))))
    return pos, rot


def resolve_surface(stage: dict, name: str | None = None) -> dict:
    """One work-surface record. ``name`` overrides the stage default."""
    return _select(stage["surfaces"], name if name is not None else stage.get("surface"), "surface")


def _records(payload: dict, plural: str, singular: str) -> tuple[list[dict], str | None]:
    default = payload.get(singular)
    if plural in payload:
        records = payload[plural]
        if not isinstance(records, list) or not records:
            raise ValueError(f"assets/stage.json {plural} must be a non-empty list")
        if isinstance(default, dict):
            raise ValueError(f"assets/stage.json {singular} must name one entry of {plural}")
        name = default if isinstance(default, str) else None
        return list(records), name
    if isinstance(default, dict):
        return [default], default.get("name")
    raise ValueError(f"assets/stage.json is missing {plural}")


def _select(records: list[dict], name: str | None, kind: str) -> dict:
    if name is None:
        if len(records) == 1:
            return records[0]
        raise ValueError(f"assets/stage.json lists {len(records)} {kind}s and names no default")
    for record in records:
        if record.get("name") == name:
            return record
    known = ", ".join(str(record.get("name")) for record in records)
    raise KeyError(f"benchmark has no {kind} {name!r}. Known: {known}")


def choice(payload: dict, key: str) -> str | None:
    """String override on a task file, or None for the stage default."""
    value = payload.get(key)
    return value if isinstance(value, str) else None
