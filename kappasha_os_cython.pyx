# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
# kappasha_os_cython.pyx - Cython-optimized functions for KappashaOS.
# Dual License:
# - For core software: AGPL-3.0-or-later licensed. -- OliviaLynnArchive fork, 2025
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# - For hardware/embodiment interfaces (if any): Licensed under the Apache License, Version 2.0
#   with xAI amendments for safety (prohibits misuse in hashing; revocable for unethical use).
#   See http://www.apache.org/licenses/LICENSE-2.0 for details.
#
# Copyright 2025 xAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
cimport numpy as np
from libc.math cimport sin, cos

cdef extern from "math.h":
    double fabs(double x)

def project_third_angle(np.ndarray[np.float64_t, ndim=3] grid, double kappa):
    """Cython-optimized third-angle projection with kappa tilt."""
    cdef int i, j, k
    cdef int size = grid.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] front = np.zeros((size, size), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] right = np.zeros((size, size), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] top = np.zeros((size, size), dtype=np.float64)
    cdef double[:, :, ::1] grid_view = grid
    cdef double[:, ::1] front_view = front
    cdef double[:, ::1] right_view = right
    cdef double[:, ::1] top_view = top
    cdef double tilt_x, tilt_y

    for i in range(size):
        for j in range(size):
            front_view[i, j] = grid_view[i, j, 0]
            right_view[i, j] = grid_view[size - 1, j, i] * (1 - kappa * sin(i / 4.0))
            top_view[i, j] = grid_view[i, size - 1, j] * (1 - kappa * cos(j / 4.0))

    return front, right, top

def thought_arb_cython(object curve, list history, str intent):
    """Cython-optimized thought arbitrage check."""
    cdef int i
    cdef double mental_kappa, real_kappa
    cdef str real_hash
    cdef bint tangent

    if len(history) < 2:
        return "hold"
    mental_kappa = history[-2][1]  # Use kappa as intent proxy
    tangent, _ = curve.spiral_tangent(mental_kappa, intent)
    real_hash = "stable"
    for i in range(len(history)):
        if "kappa=" in str(history[i]) and "hash=" in str(history[i]):
            real_hash = str(history[i]).split("hash=")[1].split()[0]
            break
    if tangent and real_hash != "stable" and fabs(mental_kappa - float(real_hash.split("=")[1])) > 0.1:
        return "unwind"
    return "hold"
