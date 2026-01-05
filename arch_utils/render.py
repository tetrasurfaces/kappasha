#!/usr/bin/env python3
# arch_utils/render.py - Rhombus voxel grid rendering to STL with kappa tilt.
# Part of Kappasha OS, pure civilian engineering.
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
from tetra.tetras import fractal_tetra
import struct

def render(grid, kappa, surface_id="grid"):
    """Render rhombus voxel grid as STL with dynamic kappa tilt, mark decision clashes."""
    triangles = []
    for i in range(grid.shape[0] - 1):
        for j in range(grid.shape[1] - 1):
            for k in range(grid.shape[2] - 1):
                if grid[i, j, k] > 0:  # Active voxel
                    p0 = np.array([i, j, k])
                    p1 = np.array([i + 1, j, k])
                    p2 = np.array([i, j + 1, k])
                    p3 = np.array([i, j, k + 1])
                    # Dynamic kappa tilt with decision clash shading
                    tilt_mat = np.array([[1, 0, -kappa * (1 + np.sin(i/4))], [0, 1, -kappa * (1 + np.cos(j/4))], [0, 0, 1]])
                    p0 = tilt_mat @ p0
                    p1 = tilt_mat @ p1
                    p2 = tilt_mat @ p2
                    p3 = tilt_mat @ p3
                    # Mock decision clash (e.g., if kappa > 0.3, shade red)
                    if kappa > 0.3:
                        p0[2] += 0.5  # Lift for visual clash
                    triangles.append([p0, p1, p2])
                    triangles.append([p0, p2, p3])
    # Add fractal tetra for depth
    tetra_mesh = fractal_tetra(surface_id, kappa)
    triangles.extend(tetra_mesh)
    # Dynamic STL export
    filename = f"surface_{surface_id}_{int(kappa*100)}.stl"
    with open(filename, 'wb') as f:
        f.write(f"ID: {surface_id}_kappa{kappa:.2f}".ljust(80, ' ').encode('utf-8'))
        f.write(struct.pack('<I', len(triangles)))
        for tri in triangles:
            v1 = np.array(tri[1]) - np.array(tri[0])
            v2 = np.array(tri[2]) - np.array(tri[0])
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            normal = normal / norm_len if norm_len > 0 else np.array([0.0, 0.0, 1.0])
            f.write(struct.pack('<3f', *normal))
            for p in tri:
                f.write(struct.pack('<3f', *p))
            f.write(struct.pack('<H', 0))
    print(f"arch_utils: Rendered dynamic rhombus grid to {filename} with decision clashes")
    return filename
