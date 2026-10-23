# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from franka_tabletop_rl.benchmark.bundle import ALL_TASKS, load_benchmark, task_ids_of

_BUILDER = Path(__file__).resolve().parents[1] / "scripts" / "build_bundle.py"
_spec = importlib.util.spec_from_file_location("build_bundle", _BUILDER)
build_bundle_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_bundle_mod)
build_bundle = build_bundle_mod.build_bundle


def _source(root: Path, tasks: list[tuple[str, str, str]]) -> None:
    (root / "scenes").mkdir(parents=True)
    (root / "assets").mkdir()
    entries = []
    for task_id, layout, background in tasks:
        payload = {
            "task_id": task_id,
            "scene_id": layout,
            "background": background,
            "surface": "oak",
            "instruction": f"do {task_id}",
            "placements": [
                {
                    "slot": "item",
                    "name": "cube",
                    "kind": "item",
                    "usd": "assets/objects/cube.usd",
                    "size_m": [0.05, 0.05, 0.05],
                    "position_m": [0.5, 0.0, 0.02],
                }
            ],
            "success": {"op": "picked", "object": "cube"},
        }
        (root / "scenes" / f"{task_id}-{layout}.json").write_text(json.dumps(payload), encoding="utf-8")
        entries.append({"task_id": task_id, "scene": layout, "background": background})
    (root / "benchmark.yaml").write_text(yaml.safe_dump({"version": "src", "tasks": entries}), encoding="utf-8")
    (root / "assets" / "objects").mkdir()
    (root / "assets" / "objects" / "cube.usd").write_text("usd", encoding="utf-8")
    (root / "assets" / "surfaces").mkdir()
    (root / "assets" / "surfaces" / "oak.usd").write_text("usd", encoding="utf-8")
    (root / "assets" / "backgrounds").mkdir()
    (root / "assets" / "backgrounds" / "office.png").write_text("png", encoding="utf-8")
    (root / "assets" / "fixtures").mkdir()
    (root / "assets" / "fixtures" / "table.usd").write_text("usd", encoding="utf-8")
    stage = {
        "background": "home_office",
        "backgrounds": [{"name": "home_office", "texture": "assets/backgrounds/office.png"}],
        "surface": "oak",
        "surfaces": [{"name": "oak", "usd": "assets/surfaces/oak.usd", "pos": [0, 0, 0], "rot": [1, 0, 0, 0]}],
        "fixture": {"usd": "assets/fixtures/table.usd"},
        "physics": {"gpu_max_rigid_patch_count": 1048576},
    }
    (root / "assets" / "stage.json").write_text(json.dumps(stage), encoding="utf-8")


def test_a_built_bundle_round_trips(tmp_path: Path):
    source = tmp_path / "src"
    _source(source, [("alpha", "lay", "hall"), ("beta", "lay", "studio")])
    out = tmp_path / "out"
    assert build_bundle(source, out, "demo", "0.1.0", None) == 2
    benchmark = load_benchmark(out)
    assert benchmark.select(ALL_TASKS) == ("alpha", "beta")
    assert benchmark.select("alpha") == ("alpha",)
    with pytest.raises(FileNotFoundError):
        benchmark.select("missing")
    written = []
    for task in benchmark.task_ids:
        written.append(json.loads((out / "tasks" / f"{task}.json").read_text(encoding="utf-8"))["task_id"])
    assert benchmark.task_ids == tuple(written)
    for task_id in benchmark.task_ids:
        payload = json.loads((out / "tasks" / f"{task_id}.json").read_text(encoding="utf-8"))
        assert payload["layout"] and payload["instruction"]
        assert payload["background"] == "home_office"
        assert "scene_id" not in payload
    written_stage = json.loads((out / "assets" / "stage.json").read_text(encoding="utf-8"))
    assert written_stage["background"] == "home_office"
    assert [record["name"] for record in written_stage["backgrounds"]] == ["home_office"]
    manifest = yaml.safe_load((out / "manifest.yaml").read_text(encoding="utf-8"))
    assert "preset_scenes" not in manifest
    assert manifest["version"] == "0.1.0"


def test_cached_v1_index_matches_its_files(requires_bundle):
    root = requires_bundle("franka-tabletop-rl-v1")
    benchmark = load_benchmark(root)
    files = {path.stem for path in (root / "tasks").glob("*.json")}
    assert set(benchmark.task_ids) == files
    assert len(benchmark.task_ids) == 120
    for task_id in benchmark.task_ids:
        layout = benchmark.task(task_id)
        assert layout.layout and layout.instruction and layout.background
    assert task_ids_of("franka-tabletop-rl-v1", ALL_TASKS) == list(benchmark.task_ids)
    assert task_ids_of("franka-tabletop-rl-v1", benchmark.task_ids[0]) == [benchmark.task_ids[0]]
    assert task_ids_of("franka-tabletop-rl-v1", "not_a_task") is None


def test_background_argument_selects_the_dome(tmp_path: Path):
    source = tmp_path / "src"
    _source(source, [("alpha", "lay", "home_office"), ("beta", "lay", "home_office")])
    stage_path = source / "assets" / "stage.json"
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    stage["backgrounds"].append({"name": "hall", "texture": "assets/backgrounds/hall.png"})
    stage_path.write_text(json.dumps(stage), encoding="utf-8")
    (source / "assets" / "backgrounds" / "hall.png").write_text("png", encoding="utf-8")
    out = tmp_path / "out"
    assert build_bundle(source, out, "demo", "1.0.0", None, background="hall") == 2
    for task_id in ("alpha", "beta"):
        payload = json.loads((out / "tasks" / f"{task_id}.json").read_text(encoding="utf-8"))
        assert payload["background"] == "hall"
    written = json.loads((out / "assets" / "stage.json").read_text(encoding="utf-8"))
    assert written["background"] == "hall"
    assert [record["name"] for record in written["backgrounds"]] == ["hall"]
    assert (out / "assets" / "backgrounds" / "hall.png").is_file()
    assert not (out / "assets" / "backgrounds" / "office.png").exists()
