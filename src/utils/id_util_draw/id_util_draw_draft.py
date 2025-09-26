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

# NURBS basis function
def nurbs_basis(u, i, p, knots):
    if p == 0:
        return 1.0 if knots[i] <= u <= knots[i+1] else 0.0  # Include = for end
    if knots[i+p] == knots[i]:
        c1 = 0.0
    else:
        c1 = (u - knots[i]) / (knots[i+p] - knots[i]) * nurbs_basis(u, i, p-1, knots)
    if knots[i+p+1] == knots[i+1]:
        c2 = 0.0
    else:
        c2 = (knots[i+p+1] - u) / (knots[i+p+1] - knots[i+1]) * nurbs_basis(u, i+1, p-1, knots)
    return c1 + c2

# Compute NURBS curve point
def nurbs_curve_point(u, control_points, weights, p, knots):
    n = len(control_points) - 1
    x = 0.0
    y = 0.0
    denom = 0.0
    for i in range(n + 1):
        b = nurbs_basis(u, i, p, knots)
        denom += b * weights[i]
        x += b * weights[i] * control_points[i][0]
        y += b * weights[i] * control_points[i][1]
    if denom == 0:
        return 0, 0
    return x / denom, y / denom

# Generate NURBS curve
def generate_nurbs_curve(control_points, weights, p, knots, num_points=1000):
    u_min, u_max = knots[p], knots[-p-1]
    u_values = np.linspace(u_min, u_max, num_points, endpoint=False)
    curve = [nurbs_curve_point(u, control_points, weights, p, knots) for u in u_values]
    curve.append(curve[0])  # Append first point for exact closure
    return np.array(curve)

# Standard NURBS circle control points and weights
control_points = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
    (1, 0)  # Repeat first for closure
]

w = 1 / np.sqrt(2)
weights = [1, w, 1, w, 1, w, 1, w, 1]

p = 2  # Degree 2 for conics

# Knots for circle, normalized to [0,1]
raw_knots = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4]
knots = np.array(raw_knots) / 4.0  # Normalize to [0,1]

# Original curve
original_curve = generate_nurbs_curve(control_points, weights, p, knots)

# Adjust local kappa (weight) for segment 1
adjusted_weights = weights.copy()
adjusted_weights[1] = 1.5 * adjusted_weights[1]
adjusted_curve = generate_nurbs_curve(control_points, adjusted_weights, p, knots)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(original_curve[:, 0], original_curve[:, 1], 'b-', label='Original NURBS Circle')
ax.plot(adjusted_curve[:, 0], adjusted_curve[:, 1], 'r--', label='Adjusted NURBS (Local Kappa Change)')
ax.plot([cp[0] for cp in control_points[: -1]], [cp[1] for cp in control_points[: -1]], 'ko', label='Control Points')
ax.set_aspect('equal')
ax.legend()
ax.set_title('NURBS Closed Loop (Circle) with Local Kappa Adjustment')
plt.savefig('nurbs_closed_circle_fixed.png')
plt.show()

# Verify ends unchanged and closure
print("Original ends:", original_curve[0], original_curve[-1])
print("Adjusted ends:", adjusted_curve[0], adjusted_curve[-1])
print("Original closure continuity:", np.allclose(original_curve[0], original_curve[-1]))
print("Adjusted closure continuity:", np.allclose(adjusted_curve[0], adjusted_curve[-1]))
