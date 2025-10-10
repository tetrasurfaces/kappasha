# Copyright 2025 Beau Ayres, xAI
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Proprietary Software - All Rights Reserved
#
# This software is proprietary and confidential. Unauthorized copying,
# distribution, modification, or use is strictly prohibited without
# express written permission from Beau Ayres.
#
# AGPL-3.0-or-later licensed
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

#!/usr/bin/env python3
# setup.py - Build script for Cython-optimized KappashaOS components.

from setuptools import setup, Extension
from Cython.Build import cythonize
import os

ext_modules = [
    Extension(
        "kappasha_os_cython",
        ["kappasha_os_cython.pyx"],
        extra_compile_args=["-O3"],
        extra_link_args=[]
    ),
    Extension(
        "tetrasurfaces.tetra.arc_control",
        ["tetrasurfaces/tetra/arc_control.py"],
        extra_compile_args=["-O3"]
    ),
    Extension(
        "tetrasurfaces.tetra.arc_listen",
        ["tetrasurfaces/tetra/arc_listen.py"],
        extra_compile_args=["-O3"]
    ),
    Extension(
        "tetrasurfaces.tetra.plasma",
        ["tetrasurfaces/tetra/plasma.py"],
        extra_compile_args=["-O3"]
    ),
    Extension(
        "tetrasurfaces.tetra.mig",
        ["tetrasurfaces/tetra/mig.py"],
        extra_compile_args=["-O3"]
    ),
    Extension(
        "tetrasurfaces.tetra.tribola_sim",
        ["tetrasurfaces/tetra/tribola_sim.py"],
        extra_compile_args=["-O3"]
    )
]

setup(
    name="kappasha_os",
    packages=["tetrasurfaces", "tetrasurfaces.tetra"],
    ext_modules=cythonize(ext_modules),
    zip_safe=False,
)
