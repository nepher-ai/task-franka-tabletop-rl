# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'franka_tabletop_rl' python package."""

import os

import tomllib
from setuptools import find_packages, setup

EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(EXTENSION_PATH, "config", "extension.toml"), "rb") as _ext:
    EXTENSION_TOML_DATA = tomllib.load(_ext)

setup(
    name="franka_tabletop_rl",
    packages=find_packages(),
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    license="BSD-3-Clause",
    include_package_data=True,
    python_requires=">=3.11",
    install_requires=["numpy>=1.20", "psutil"],
    extras_require={"record": ["pyarrow>=14", "av>=12"]},
    entry_points={"isaaclab.tasks": ["franka_tabletop_rl = franka_tabletop_rl.tasks"]},
    zip_safe=False,
)
