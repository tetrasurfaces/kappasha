# kappawise.py
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
def compute_kappa_grid(grid_size=100, spiral_params=(0.001, np.log((1 + np.sqrt(5))/2) / (np.pi / 2))):
    """
    Pre-computes a grid of kappa (curvature) values based on spiral curvature field.
   
    Args:
        grid_size (int): Size of the square grid (grid_size x grid_size).
        spiral_params (tuple): (A, B) for the logarithmic spiral r = A * exp(B * theta).
   
    Returns:
        np.ndarray: 2D array of kappa values at grid points.
    """
    A, B = spiral_params
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2) + 1e-10  # Avoid division by zero
    kappa_grid = (B**2 + 1) / (r * (1 + B**2))
    return kappa_grid
# Example usage (for testing)
if __name__ == "__main__":
    grid = compute_kappa_grid()
    print("Kappa Grid Shape:", grid.shape)
    print("Sample Kappa Values:", grid[50, 50])
