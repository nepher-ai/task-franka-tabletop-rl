# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recorder bases from Isaac Lab, or stand-ins when Isaac Lab is not installed.

The film recorder subclasses these at import. Unit tests load it with no simulator, so the
stand-ins cover only the constructor and the config decorator those tests execute.
"""

try:
    from isaaclab.managers import DatasetExportMode, RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
    from isaaclab.utils import configclass
except ImportError:

    class RecorderTerm:
        """Stand-in that stores the config and the env."""

        def __init__(self, cfg, env):
            self.cfg = cfg
            self._env = env

    class RecorderTermCfg:
        """Stand-in config base."""

    class RecorderManagerBaseCfg:
        """Stand-in recorder manager config."""

    class DatasetExportMode:
        """Stand-in export mode. Only ``EXPORT_NONE`` is read."""

        EXPORT_NONE = 0

    def configclass(cls):
        """Return the class unchanged when Isaac Lab's decorator is unavailable."""
        return cls
