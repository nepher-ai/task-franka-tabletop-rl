# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Brain-format view of an RL bundle for parity tests and replay.

``task-franka-tabletop`` reads ``scenes/<task_id>-<layout>.json``. Assets are copied into the view.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from .bundle import load_benchmark


def legacy_bundle_view(bundle: str | Path, root: str | Path) -> Path:
    """Write a brain-format bundle under ``root``. ``assets/`` is a copy of the RL bundle.

    Task files are copied with ``layout`` renamed to ``scene_id``.
    """
    bundle = Path(bundle)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    assets = root / "assets"
    if not assets.exists():
        shutil.copytree(bundle / "assets", assets)
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
