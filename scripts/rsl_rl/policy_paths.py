# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve low-level checkpoints through ``best_policy/``.

Playback copies the chosen RSL-RL checkpoint to ``best_policy/best_policy.pt``. The high-level
environment and eval-nav always load that fixed path, never a timestamped run folder.
"""

from __future__ import annotations

import os
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
BEST_POLICY_DIR = os.path.join(PROJECT_ROOT, "best_policy")
BEST_POLICY_CHECKPOINT = os.path.join(BEST_POLICY_DIR, "best_policy.pt")
BEST_POLICY_EXPORT_DIR = os.path.join(BEST_POLICY_DIR, "exported")


def log_root_path(experiment_name: str) -> str:
    """Directory that holds the training runs of ``experiment_name``."""
    return os.path.join(PROJECT_ROOT, "logs", "rsl_rl", experiment_name)


def sync_best_policy(
    experiment_name: str, load_run: str, load_checkpoint: str, *, explicit_checkpoint: str | None = None
) -> str:
    """Copy the chosen checkpoint to ``best_policy/best_policy.pt`` and return that path.

    The source is ``explicit_checkpoint``, else the latest match under ``logs/``, else the file
    that is already in ``best_policy/``.

    Raises:
        FileNotFoundError: No checkpoint can be found.
    """
    os.makedirs(BEST_POLICY_DIR, exist_ok=True)
    source = None
    if explicit_checkpoint:
        source = os.path.abspath(explicit_checkpoint)
    else:
        source = _latest_in_logs(experiment_name, load_run, load_checkpoint)
    if source is not None:
        if not os.path.isfile(source):
            raise FileNotFoundError(f"checkpoint not found: {source}")
        if os.path.abspath(source) != os.path.abspath(BEST_POLICY_CHECKPOINT):
            shutil.copy2(source, BEST_POLICY_CHECKPOINT)
        print(f"[INFO] low-level policy: {source} -> {BEST_POLICY_CHECKPOINT}")
        return BEST_POLICY_CHECKPOINT
    if os.path.isfile(BEST_POLICY_CHECKPOINT):
        return BEST_POLICY_CHECKPOINT
    raise FileNotFoundError(
        f"no low-level checkpoint: train first or place one at {BEST_POLICY_CHECKPOINT} "
        f"(logs are read from {log_root_path(experiment_name)})"
    )


def _latest_in_logs(experiment_name: str, load_run: str, load_checkpoint: str) -> str | None:
    from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path

    root = log_root_path(experiment_name)
    if not os.path.isdir(root):
        return None
    try:
        return get_checkpoint_path(root, load_run, load_checkpoint)
    except ValueError:
        return None
