# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A brain-format view of an RL bundle, for parity tests and replay.

``task-franka-tabletop`` still reads ``scenes/<task_id>-<layout>.json`` and a ``scene`` key in
``benchmark.yaml``. This writes that view without copying the asset tree.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml

from .bundle import load_benchmark


def legacy_bundle_view(bundle: str | Path, root: str | Path) -> Path:
    """Write a brain-format bundle under ``root`` and return that directory.

    ``assets/`` is a junction to the RL bundle. Task files are copied as scene files with the
    ``layout`` key renamed back to ``scene_id``.

    Raises:
        OSError: The platform cannot create a directory junction or symlink.
    """
    bundle = Path(bundle)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    assets = root / "assets"
    if not assets.exists():
        _link_dir(bundle / "assets", assets)
    benchmark = load_benchmark(bundle)
    scenes = root / "scenes"
    scenes.mkdir(exist_ok=True)
    index = []
    for task_id, record in zip(benchmark.task_ids, benchmark.records, strict=True):
        payload = dict(record)
        layout = str(payload.pop("layout"))
        payload["scene_id"] = layout
        (scenes / f"{task_id}-{layout}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        index.append(
            {
                "task_id": task_id,
                "scene": layout,
                "instruction": payload.get("instruction", ""),
                "success": payload.get("success"),
            }
        )
    text = yaml.safe_dump({"version": benchmark.version, "tasks": index}, sort_keys=False)
    (root / "benchmark.yaml").write_text(text, encoding="utf-8")
    return root


def _link_dir(source: Path, dest: Path) -> None:
    """Point ``dest`` at ``source``. A junction is used when a symlink is not permitted."""
    try:
        os.symlink(source, dest, target_is_directory=True)
    except OSError:
        subprocess.check_call(["cmd", "/c", "mklink", "/J", str(dest), str(source)])
