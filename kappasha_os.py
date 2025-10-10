#!/usr/bin/env python3
# kappasha_os.py - Kappa-tilted OS with rhombus voxel navigation, factory sim integration.
# CLI-driven, no GUI, DOS Navigator soul in 3D. Pure civilian engineering.
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

import simpy
import numpy as np
from nav3d import RhombusNav
from factory_sim import FactorySim
from ghost_hand import GhostHand
from thought_curve import ThoughtCurve
from tetras import fractal_tetra  # For rhombus grid rendering
import struct

class arch_utils:
    @staticmethod
    def render(grid, kappa, surface_id="grid"):
        """Render rhombus voxel grid as STL with kappa tilt."""
        triangles = []
        for i in range(grid.shape[0] - 1):
            for j in range(grid.shape[1] - 1):
                for k in range(grid.shape[2] - 1):
                    if grid[i, j, k] > 0:  # Active voxel
                        p0 = np.array([i, j, k])
                        p1 = np.array([i + 1, j, k])
                        p2 = np.array([i, j + 1, k])
                        p3 = np.array([i, j, k + 1])
                        # Kappa tilt
                        tilt_mat = np.array([[1, 0, -kappa], [0, 1, -kappa], [0, 0, 1]])
                        p0 = tilt_mat @ p0
                        p1 = tilt_mat @ p1
                        p2 = tilt_mat @ p2
                        p3 = tilt_mat @ p3
                        triangles.append([p0, p1, p2])
                        triangles.append([p0, p2, p3])
        # Add fractal tetra for depth
        tetra_mesh = fractal_tetra(surface_id, kappa)
        triangles.extend(tetra_mesh)
        # Mock STL export
        filename = f"surface_{surface_id}.stl"
        with open(filename, 'wb') as f:
            f.write(f"ID: {surface_id}".ljust(80, ' ').encode('utf-8'))
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
        print(f"arch_utils: Rendered rhombus grid to {filename}")
        return filename

class dev_utils:
    @staticmethod
    def lockout(factory, target):
        """Trigger lockout on target with kappa awareness."""
        factory.trigger_emergency(target)
        print(f"dev_utils: Locked out {target}")

    @staticmethod
    def hedge(curve, path):
        """Hedge path with ThoughtCurve tangent check."""
        if len(path) < 2:
            return "hold"
        tangent, _ = curve.spiral_tangent(path[-2], path[-1])
        return "unwind" if tangent else "hold"

class KappashaOS:
    def __init__(self):
        self.env = simpy.Environment()
        self.nav = RhombusNav(kappa=0.2)
        self.factory = FactorySim(self.env)
        self.hand = GhostHand(kappa=0.2)
        self.curve = ThoughtCurve()
        self.commands = []
        print("Kappasha OS booted - kappa-tilted rhombus grid ready.")

    def run_command(self, cmd):
        """Execute CLI commands with kappa awareness."""
        self.commands.append(cmd)
        if cmd == "kappa ls":
            front, right, top = self.nav.project_third_angle()
            print("FRONT:\n", front[:3, :3])
            print("RIGHT:\n", right[:3, :3])
            print("TOP:\n", top[:3, :3])
        elif cmd.startswith("kappa tilt"):
            try:
                dk = float(cmd.split()[2])
                self.nav.kappa += dk
                self.factory.kappa += dk
                self.hand.kappa += dk
                self.hand.pulse(2)
                print(f"Kappa now {self.nav.kappa:.3f}")
            except:
                print("usage: kappa tilt 0.05")
        elif cmd.startswith("kappa cd"):
            try:
                path = cmd.split()[2]
                self.nav.path.append(path)
                hedge_action = dev_utils.hedge(self.curve, self.nav.path)
                if hedge_action == "unwind":
                    self.hand.pulse(3)
                    print("Path hedge: unwind")
                print(f"Curved to /{path}")
            except:
                print("usage: kappa cd logs")
        elif cmd.startswith("kappa unlock"):
            try:
                coord = tuple(map(int, cmd.split()[2].strip("()").split(",")))
                if self.nav.unlock_edge(coord):
                    self.factory.register_kappa("edge_unlock")
            except:
                print("usage: kappa unlock (7,0,0)")
        elif cmd == "arch_utils render":
            filename = arch_utils.render(self.nav.grid, self.nav.kappa)
            print(f"arch_utils: Rendered to {filename}")
        elif cmd.startswith("dev_utils lockout"):
            try:
                target = cmd.split()[2]
                dev_utils.lockout(self.factory, target)
            except:
                print("usage: dev_utils lockout gas_line")
        else:
            print("kappa: ls | tilt 0.05 | cd logs | unlock (7,0,0) | arch_utils render | dev_utils lockout gas_line")

    def run_day(self):
        """Simulate a factory day with kappa navigation."""
        print(f"Day start - Situational Kappa = {self.factory.get_situational_kappa():.3f}")
        yield self.env.timeout(20)
        self.factory.trigger_emergency("gas_rupture")
        self.factory.register_kappa("gas_rupture")
        self.run_command("kappa cd weld")
        self.run_command("kappa unlock (7,0,0)")
        yield self.env.process(self.factory.auto_rig("gas_line"))
        self.run_command("kappa ls")
        self.run_command("arch_utils render")
        print(f"Day end - Situational Kappa = {self.factory.get_situational_kappa():.3f}")

if __name__ == "__main__":
    os = KappashaOS()
    os.env.process(os.run_day())
    os.env.run(until=60)
