# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoints and logs beside this task project, outside the Python package."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def packaged_asset(*parts: str) -> str:
    """Path of a file under ``franka_tabletop_rl/assets``."""
    return str(Path(__file__).resolve().parent.joinpath("assets", *parts))


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def log_dir() -> Path:
    """Tracker training runs. An absolute path replaces Isaac Lab's ``logs/rsl_rl`` prefix."""
    return project_root() / "logs" / "rsl_rl" / "franka_tabletop_ll_ee_tracking"


def best_policy_path() -> Path:
    """Published tracker checkpoint."""
    return project_root() / "best_policy" / "best_policy.pt"


def play_video_dir() -> Path:
    """Directory for highlight play videos."""
    return best_policy_path().parent / "videos" / "highlight"


def sync_best_policy(
    experiment_name: str,
    load_run: str,
    load_checkpoint: str,
    *,
    explicit_checkpoint: str | None = None,
) -> str:
    """Copy the chosen checkpoint to ``best_policy/best_policy.pt`` and return that path.

    Uses ``explicit_checkpoint``, else an existing absolute ``load_checkpoint``, else the latest
    log match, else the file already published.
    """
    destination = best_policy_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = _resolve_source(experiment_name, load_run, load_checkpoint, explicit_checkpoint)
    if source is not None:
        if os.path.abspath(source) != os.path.abspath(destination):
            shutil.copy2(source, destination)
        print(f"[INFO] low-level policy: {source} -> {destination}")
        return str(destination)
    if destination.is_file():
        return str(destination)
    raise FileNotFoundError(
        f"no low-level checkpoint: train first or place one at {destination} "
        f"(logs are read from {log_dir()})"
    )


def _resolve_source(
    experiment_name: str, load_run: str, load_checkpoint: str, explicit_checkpoint: str | None
) -> str | None:
    if explicit_checkpoint:
        source = os.path.abspath(explicit_checkpoint)
        if not os.path.isfile(source):
            raise FileNotFoundError(f"checkpoint not found: {source}")
        return source
    if os.path.isabs(load_checkpoint) and os.path.isfile(load_checkpoint):
        return os.path.abspath(load_checkpoint)
    return _latest_in_logs(experiment_name, load_run, load_checkpoint)


def _latest_in_logs(experiment_name: str, load_run: str, load_checkpoint: str) -> str | None:
    from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path

    root = experiment_name if os.path.isabs(experiment_name) else str(log_dir())
    if not os.path.isdir(root):
        return None
    try:
        return get_checkpoint_path(root, load_run, load_checkpoint)
    except ValueError:
        return None
