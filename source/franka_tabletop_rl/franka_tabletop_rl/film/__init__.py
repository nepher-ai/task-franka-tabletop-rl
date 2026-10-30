# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Film tooling: scripted camera shots rendered between control steps, without UI.

Modules that need Isaac Sim (``recorder``, ``render``, ``env``, ``env_cfg``) are imported on demand.
``rig``, ``shots``, ``selection``, ``settings``, ``timeline``, ``latch``, and ``points`` need only NumPy.
"""
