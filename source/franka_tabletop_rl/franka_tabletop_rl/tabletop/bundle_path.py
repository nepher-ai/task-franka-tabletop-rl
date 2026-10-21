# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Where the benchmark bundle of this project lives."""

from __future__ import annotations

from pathlib import Path

DEFAULT_ENV_ID = "franka-tabletop-rl-v0"


def resolve_bundle(env_id: str | None = None) -> Path:
    """Directory of the EnvHub bundle ``env_id`` in the nepher cache.

    ``Nepher-FrankaTabletopRL-HL-Play-v0`` and ``Nepher-FrankaTabletopRL-HL-EnvhubPlay-v0``
    load the table, the background, and the objects from this directory.

    Raises:
        FileNotFoundError: ``nepher`` is not installed, or the bundle is not cached.
    """
    env_id = env_id or DEFAULT_ENV_ID
    try:
        from nepher.storage.cache import get_cache_manager
    except ImportError as error:
        raise FileNotFoundError(
            f"EnvHub bundle {env_id} is loaded from the nepher cache. "
            f"Install nepher and run `nepher download {env_id}`."
        ) from error
    cache = get_cache_manager(category="manipulation")
    if not cache.is_cached(env_id):
        raise FileNotFoundError(
            f"EnvHub bundle {env_id} is not in the nepher cache ({cache.cache_dir}). Run `nepher download {env_id}`."
        )
    return cache.get_env_cache_path(env_id)
