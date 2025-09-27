# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Anonymous
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from matplotlib.widgets import Slider
import hashlib
from decimal import Decimal, getcontext
import warnings
from matplotlib import MatplotlibDeprecationWarning
import struct
import base64

def compute_kappa_grid(grid_size=100, num_angles=369, spiral_factor=0.1, compound_levels=3, decay_scale=None, spiral_params=(0.001, np.log((1 + np.sqrt(5))/2) / (np.pi / 2))):
    """
    Pre-computes a 3D grid of kappa (curvature) values based on compound spiral intersections with angle layers.
   
    This updated function integrates compound logarithmic spiral patterns with nested levels for even decay over space,
    enhancing parametric modeling and enabling speed-over-curve driving in CMM/CAD/CAM. It combines the original spiral
    parameters with compound sets (angles on angles) for multi-scale effects.
   
    Args:
        grid_size (int): Size of the square grid (grid_size x grid_size).
        num_angles (int): Number of angular slices (default 369 for slight overlap beyond 360 degrees).
        spiral_factor (float): Controls the tightness of the compound spirals (b in log spiral r = exp(b * theta)).
        compound_levels (int): Number of compound sets (nested spirals) for multi-scale decay.
        decay_scale (float, optional): Scale for exponential decay (default grid_size / 2).
        spiral_params (tuple): (A, B) for the base logarithmic spiral r = A * exp(B * theta) (original params preserved).
   
    Returns:
        np.ndarray: 3D array of kappa values at grid points (grid_size, grid_size, num_angles).
    """
    if decay_scale is None:
        decay_scale = grid_size / 2.0
   
    A, B_base = spiral_params # Preserve original spiral params
    kappa_grid_3d = np.zeros((grid_size, grid_size, num_angles))
    cx, cy = grid_size / 2.0, grid_size / 2.0
   
    for k in range(num_angles):
        base_theta_k = k * 2 * np.pi / num_angles # Angle in radians, modulo 2π
       
        for i in range(grid_size):
            for j in range(grid_size):
                dx = j - cx
                dy = i - cy
                r = np.sqrt(dx**2 + dy**2) + 1e-10
                if r < 1e-6:
                    kappa_grid_3d[i, j, k] = 1.0 # Max kappa at center
                    continue
               
                theta = np.arctan2(dy, dx) # Base angle at position
                kappa = 0.0
               
                # Compound sets: nested angles on angles with integrated original spiral modulation
                current_theta = theta
                for level in range(1, compound_levels + 1):
                    scale = 1.0 / level # Smaller scales for higher levels
                    B_level = B_base * scale # Scale base B for compound levels
                    # Spiral modulation: compound by adding log(r)-based twist, incorporating original (B^2 + 1) / (r * (1 + B^2))
                    spiral_theta = current_theta + np.log(1 + r * scale) / spiral_factor
                    # Diff modulo 2π, centered around 0
                    diff = (spiral_theta - base_theta_k) % (2 * np.pi) - np.pi
                    # Gaussian around spiral arm + even exponential decay over r + original kappa formula
                    level_kappa = np.exp(-diff**2 / (0.5 * scale)) * np.exp(-r / decay_scale) * ((B_level**2 + 1) / (r * (1 + B_level**2)))
                    kappa += level_kappa
                   
                    # Next level: angle on angle (compound)
                    current_theta = spiral_theta % (2 * np.pi)
               
                kappa_grid_3d[i, j, k] = kappa / compound_levels # Normalize by levels
   
    return kappa_grid_3d
# Example usage (for testing)
if __name__ == "__main__":
    grid = compute_kappa_grid()
    print("3D Kappa Grid Shape:", grid.shape)
    print("Sample Kappa Value:", grid[50, 50, 0])
