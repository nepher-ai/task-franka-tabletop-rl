# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build an RL EnvHub bundle from a brain tabletop bundle.

Writes ``tasks/<task_id>.json`` (``scene_id`` renamed to ``layout``), an index-only ``benchmark.yaml``,
and the assets those tasks use. ``manifest.yaml`` has no ``preset_scenes``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import yaml

# One background and 16 tasks fit the published v0 buffer. A larger stage needs more pairs.
_LARGE_PATCH_COUNT = 2**23


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive an RL tabletop bundle from a brain bundle.")
    parser.add_argument("--source", type=Path, required=True, help="Brain bundle directory.")
    parser.add_argument("--out", type=Path, required=True, help="RL bundle directory to write.")
    parser.add_argument("--id", required=True, help="EnvHub id, also the benchmark version name.")
    parser.add_argument("--version", default="1.0.0", help="Manifest version.")
    parser.add_argument("--tasks", nargs="*", default=None, help="Task ids to keep, in this order. Default: all.")
    parser.add_argument("--name", default=None, help="Manifest name. Default: a name built from the id.")
    parser.add_argument(
        "--background",
        default=None,
        help="Dome for every task. Default: the source stage's background.",
    )
    args = parser.parse_args()
    written = build_bundle(
        args.source, args.out, args.id, args.version, args.tasks, args.name, args.background
    )
    print(f"wrote {written} tasks to {args.out}")


def build_bundle(
    source: Path,
    out: Path,
    env_id: str,
    version: str,
    task_ids: list[str] | None,
    name: str | None = None,
    background: str | None = None,
) -> int:
    """Write the RL bundle and return how many tasks it contains.

    ``background`` overrides the source stage. Every task uses that one dome.
    """
    source = Path(source)
    out = Path(out)
    index = yaml.safe_load((source / "benchmark.yaml").read_text(encoding="utf-8"))
    entries = list(index["tasks"])
    by_id = {str(entry["task_id"]): entry for entry in entries}
    chosen = list(task_ids) if task_ids else [str(entry["task_id"]) for entry in entries]
    unknown = [task_id for task_id in chosen if task_id not in by_id]
    if unknown:
        raise ValueError(f"source benchmark has no tasks: {unknown}")
    payloads = []
    for task_id in chosen:
        entry = by_id[task_id]
        path = _scene_file(source, task_id, str(entry["scene"]))
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "scene_id" in payload:
            payload["layout"] = payload.pop("scene_id")
        else:
            payload["layout"] = str(entry["scene"])
        payload["task_id"] = task_id
        payloads.append(payload)

    stage = json.loads((source / "assets" / "stage.json").read_text(encoding="utf-8"))
    records = [record for record in stage.get("backgrounds") or [] if record.get("name")]
    by_name = {str(record["name"]): record for record in records}
    chosen_bg = background or stage.get("background")
    if not isinstance(chosen_bg, str) or not chosen_bg:
        raise ValueError("assets/stage.json names no background, and --background was not set")
    if chosen_bg not in by_name:
        raise ValueError(f"stage has no background {chosen_bg!r}")
    for payload in payloads:
        payload["background"] = chosen_bg

    if out.exists():
        for name_ in ("tasks", "scenes"):
            folder = out / name_
            if folder.exists():
                shutil.rmtree(folder)
    (out / "tasks").mkdir(parents=True, exist_ok=True)
    used_usd: set[str] = set()
    surfaces: set[str] = set()
    for payload in payloads:
        (out / "tasks" / f"{payload['task_id']}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        surfaces.add(str(payload["surface"]))
        for item in payload["placements"]:
            used_usd.add(str(item["usd"]))
    stage["backgrounds"] = [by_name[chosen_bg]]
    stage["background"] = chosen_bg
    stage["surfaces"] = [record for record in stage["surfaces"] if record["name"] in surfaces]
    stage["surface"] = stage["surfaces"][0]["name"]
    if len(chosen) > 16:
        physics = dict(stage.get("physics") or {})
        physics["gpu_max_rigid_patch_count"] = max(int(physics.get("gpu_max_rigid_patch_count", 0)), _LARGE_PATCH_COUNT)
        stage["physics"] = physics
    for record in stage["backgrounds"]:
        used_usd.add(str(record["texture"]))
    for record in stage["surfaces"]:
        used_usd.add(str(record["usd"]))
    used_usd.add(str(stage["fixture"]["usd"]))

    assets = out / "assets"
    if assets.exists():
        shutil.rmtree(assets)
    for relative in sorted(used_usd):
        src = source / relative
        dest = out / relative
        if not src.is_file():
            raise FileNotFoundError(f"source bundle is missing {relative}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        _copy_sidecar(src, dest)
    (out / "assets").mkdir(exist_ok=True)
    (out / "assets" / "stage.json").write_text(json.dumps(stage, indent=2) + "\n", encoding="utf-8")
    _write_catalog(source, out, used_usd)
    header = (
        f"version: {version}\n"
        "# Task index. instruction and success live only in tasks/<task_id>.json.\n"
        "# layout is the table arrangement. background is the benchmark dome.\n"
    )
    body = yaml.safe_dump(
        {
            "tasks": [
                {"task_id": payload["task_id"], "layout": payload["layout"], "background": payload["background"]}
                for payload in payloads
            ]
        },
        sort_keys=False,
    )
    (out / "benchmark.yaml").write_text(header + body, encoding="utf-8")
    manifest = {
        "id": env_id,
        "name": name or env_id,
        "description": f"{len(payloads)} tabletop tasks in one stage with one background.",
        "category": "manipulation",
        "version": version,
        "benchmark": True,
    }
    (out / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return len(payloads)


def _scene_file(source: Path, task_id: str, layout: str) -> Path:
    direct = source / "scenes" / f"{task_id}-{layout}.json"
    if direct.is_file():
        return direct
    matches = sorted((source / "scenes").glob(f"{task_id}-*.json"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"source has no scene file for {task_id}")


def _write_catalog(source: Path, out: Path, used_usd: set[str]) -> None:
    path = source / "assets" / "catalog.json"
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("objects") if isinstance(payload, dict) else payload
    kept = [record for record in records if str(record.get("usd")) in used_usd]
    text = json.dumps({"objects": kept} if isinstance(payload, dict) else kept, indent=2) + "\n"
    (out / "assets" / "catalog.json").write_text(text, encoding="utf-8")


def _copy_sidecar(src: Path, dest: Path) -> None:
    """Copy a same-named folder and a sibling ``textures`` directory so the mesh keeps its materials."""
    folder = src.with_suffix("")
    if folder.is_dir():
        shutil.copytree(folder, dest.with_suffix(""), dirs_exist_ok=True)
    textures = src.parent / "textures"
    if textures.is_dir():
        shutil.copytree(textures, dest.parent / "textures", dirs_exist_ok=True)
    _copy_local_layers(src, dest)


def _copy_local_layers(src: Path, dest: Path) -> None:
    """Copy ``@./file.usd@`` layers referenced by a text USDA."""
    try:
        text = src.read_text(encoding="utf-8")
    except UnicodeError:
        return
    if not text.startswith("#usda"):
        return
    for ref in re.findall(r"@(\./[^@\s]+)@", text):
        child_src = (src.parent / ref).resolve()
        child_dest = (dest.parent / ref).resolve()
        if not child_src.is_file() or child_dest.exists():
            continue
        child_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(child_src, child_dest)
        _copy_sidecar(child_src, child_dest)


if __name__ == "__main__":
    main()
