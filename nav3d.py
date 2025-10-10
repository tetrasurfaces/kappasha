#!/usr/bin/env python3
# nav3d.py - Rhombus voxel navigator, kappa-tilted, DOS Navigator soul in 3D grid.
# Third-angle projection with kappa for file edge unlock. CLI only, no GUI.
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
from ghost_hand import GhostHand  # Haptic feedback for kappa tilt
from thought_curve import ThoughtCurve  # Path tangents and hedging

class RhombusNav:
    def __init__(self, kappa=0.1):
        self.kappa = kappa  # Kappa for grid tilt
        self.grid = np.zeros((8, 8, 8))  # 8x8x8 rhombus voxel cube
        self.curve = ThoughtCurve()  # For path hedging
        self.hand = GhostHand(kappa=self.kappa)  # Haptic interface
        self.path = []  # Kappa trail for navigation
        print("RhombusNav initialized - kappa-tilted 3D grid ready.")

    def project_third_angle(self):
        """Third-angle projection: front, right, top faces tilted by kappa."""
        front = self.grid[:, :, 0]  # z=0 face
        right = self.grid[7, :, :]  # x=7 face
        top = self.grid[:, 7, :]    # y=7 face
        # Apply kappa tilt to right and top faces
        tilt_mat = np.array([[1, 0, -self.kappa],
                             [0, 1, -self.kappa],
                             [0, 0, 1]])
        right = (tilt_mat @ right.reshape(-1, 3)).reshape(right.shape)
        top = (tilt_mat @ top.reshape(-1, 3)).reshape(top.shape)
        return front, right, top

    def unlock_edge(self, coord):
        """Check voxel hash for drift; unlock edge if kappa spikes."""
        drift = np.random.rand()  # Mock hash drift
        if drift < self.kappa + 0.1:  # Threshold for edge unlock
            self.hand.pulse(2)  # Haptic alert for drift
            print(f"Edge unlocked: {coord} - kappa tilt {self.kappa:.3f}")
            return True
        else:
            self.hand.pulse(1)  # Stable signal
            print(f"Stable edge: {coord} - no drift")
            return False

    def nav(self, cmd):
        """CLI navigator with kappa-tilted verbs."""
        if cmd == "ls":
            front, right, top = self.project_third_angle()
            print("FRONT:\n", front[:3, :3])
            print("RIGHT:\n", right[:3, :3])
            print("TOP:\n", top[:3, :3])
        elif cmd.startswith("tilt"):
            try:
                dk = float(cmd.split()[1])
                self.kappa += dk
                self.hand.pulse(2)  # Haptic feedback
                print(f"Kappa now {self.kappa:.3f}")
            except:
                print("usage: tilt 0.05")
        elif cmd.startswith("cd"):
            try:
                path = cmd.split()[1]
                self.path.append(path)
                if len(self.path) > 1:
                    tangent, _ = self.curve.spiral_tangent(self.path[-2], self.path[-1])
                    if tangent:
                        self.hand.pulse(3)  # Hedge alert
                        print("Path hedge: unwind")
                print(f"Curved to /{path}")
            except:
                print("usage: cd logs")
        elif cmd.startswith("unlock"):
            try:
                coord = tuple(map(int, cmd.split()[1].strip("()").split(",")))
                self.unlock_edge(coord)
            except:
                print("usage: unlock (7,0,0)")
        else:
            print("nav: ls | tilt 0.05 | cd logs | unlock (7,0,0)")

# Test stub
if __name__ == "__main__":
    nav = RhombusNav(kappa=0.2)
    commands = ["cd gate", "cd weld", "tilt 0.1", "ls", "unlock (7,0,0)"]
    for c in commands:
        print(f"\n> {c}")
        nav.nav(c)
