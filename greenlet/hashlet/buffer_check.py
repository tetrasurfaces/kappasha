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
from io import BytesIO
import base64
import hashlib

GRID_DIM = 2141
SEED = 12345  # Hashlet seed
ANGLE = 137.5 * np.pi / 180  # Golden angle

# Downscale to plot_dim=100 for visualization
plot_dim = 100
hash_seed = int(hashlib.sha256(str(SEED).encode()).hexdigest(), 16) % (plot_dim**2)
np.random.seed(hash_seed)

# Generate unique RGB (no doubles: sample from 16M without replacement)
all_colors = np.arange(0, 16777216, dtype=np.uint32)  # 2^24 hex colors
np.random.shuffle(all_colors)
unique_ints = all_colors[:plot_dim**2]
unique_colors = np.zeros((plot_dim**2, 3), dtype=np.uint8)
unique_colors[:,0] = (unique_ints >> 16) & 0xFF  # R
unique_colors[:,1] = (unique_ints >> 8) & 0xFF   # G
unique_colors[:,2] = unique_ints & 0xFF          # B
grid_2d = unique_colors.reshape(plot_dim, plot_dim, 3) / 255.0  # Normalize [0,1]

# Spiral overlay with 0-points (black gates at i % 12 == 0)
center = plot_dim // 2
for i in range(plot_dim**2):
    r = np.sqrt(i)
    theta = i * ANGLE
    x = int(center + r * np.cos(theta)) % plot_dim
    y = int(center + r * np.sin(theta)) % plot_dim
    if i % 12 == 0:  # 0-point gates
        grid_2d[x, y] = [0, 0, 0]  # Black

# 11 Channel Zones: Tint columns (vertical strips)
channel_tint = np.linspace(0.5, 1.5, 11)  # Tint factors
strip_width = plot_dim // 11
for ch in range(11):
    start = ch * strip_width
    end = min((ch + 1) * strip_width, plot_dim)
    grid_2d[:, start:end] *= channel_tint[ch]

# 3D Cube: Stack with z-variation (^3 volumetric)
grid_3d = np.stack([grid_2d * (z / (plot_dim - 1)) for z in range(plot_dim)], axis=2)

# 2D Visual (Infra Base with Zones)
fig2d, ax2d = plt.subplots()
ax2d.imshow(grid_2d)
ax2d.set_title('2D RGB Hex Grid (No Doubles, 11 Zones, Spiral 0-Points)')
buf2d = BytesIO()
fig2d.savefig(buf2d, format='png')
buf2d.seek(0)
img2d_base64 = base64.b64encode(buf2d.read()).decode('utf-8')
plt.close(fig2d)

# 3D Visual Slice (Centre Cube)
fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection='3d')
X, Y = np.meshgrid(range(plot_dim), range(plot_dim))
Z = grid_3d[:,:,plot_dim//2].mean(axis=2)  # Mean RGB for surface
ax3d.plot_surface(X, Y, Z, cmap='viridis')
ax3d.set_title('3D Cubed Hash Slice (11 Channel Zones)')
buf3d = BytesIO()
fig3d.savefig(buf3d, format='png')
buf3d.seek(0)
img3d_base64 = base64.b64encode(buf3d.read()).decode('utf-8')
plt.close(fig3d)

# Output base64 (copy to browser: data:image/png;base64,<base64>)
print("2D Image Base64: data:image/png;base64," + img2d_base64[:100] + "... (full in console)")
print("3D Image Base64: data:image/png;base64," + img3d_base64[:100] + "...")

# Notes: 10,000 unique RGB from 16M pool (no doubles). 11 zones tinted vertically. Spiral black 0-points. For full 2141, use sparse or downscale. Zones as 11 channels for hashlet optimization (e.g., channel ch hashes sub-grid).
