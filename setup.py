# Copyright (C) Mitsubishi Electric Research Labs (MERL) 2023
#
# SPDX-License-Identifier: AGPL-3.0-or-later
from pathlib import Path

from setuptools import find_packages, setup

requirements = Path("requirements.txt").read_text().splitlines()

setup(
    name="src",
    version="0.1.0",
    python_requires=">=3.7",
    install_requires=requirements,
    packages=find_packages(),
    scripts=["bin/slurmrun", "bin/slurmrunyes"],
)
