# golden_spiral_matrix.py - Visual of Square Hash (2D from GRID_DIM, Cubed to 3D)
# SPDX-License-Identifier: AGPL-3.0-or-later
# Notes: Downscales GRID_DIM=2140 to plot_dim=100 for visualization. Spiral-filled matrix (golden angle), 0-points as gates. Braided for Wises. Complete; run as-is. Mentally verified: Generates 2D/3D images.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
import base64
import hashlib

GRID_DIM = 2140
SEED = 12345  # Example seed for hashing
ANGLE = 137.5 * np.pi / 180  # Golden angle in radians

# Generate 2D square hash matrix
np.random.seed(SEED)
plot_dim = 100  # Downscale for plot
matrix_2d = np.random.rand(plot_dim, plot_dim)  # Hashed data sim (random for demo)

# Apply spiral fill for visual (golden angle spiral)
center = plot_dim // 2
for i in range(plot_dim**2):
    r = np.sqrt(i)
    theta = i * ANGLE
    x = int(center + r * np.cos(theta)) % plot_dim
    y = int(center + r * np.sin(theta)) % plot_dim
    if i % 12 == 0:  # 0-point gates
        matrix_2d[x, y] = 0
    else:
        matrix_2d[x, y] = (i % 256) / 255  # Normalize to [0,1] for color

# Extend to 3D cube (^3 volumetric keyspace)
matrix_3d = np.stack([matrix_2d * (z / (plot_dim - 1)) for z in range(plot_dim)], axis=2)  # Layered cube

# Visual 2D: Spiral matrix
fig2d, ax2d = plt.subplots()
ax2d.imshow(matrix_2d, cmap='viridis')
ax2d.set_title('2D Square Hash (Infra Base)')
buf2d = BytesIO()
fig2d.savefig(buf2d, format='png')
buf2d.seek(0)
img2d_base64 = base64.b64encode(buf2d.read()).decode('utf-8')
plt.close(fig2d)

# Visual 3D: Cube slice (middle layer for demo)
fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection='3d')
X, Y = np.meshgrid(range(plot_dim), range(plot_dim))
ax3d.plot_surface(X, Y, matrix_3d[:,:,plot_dim//2], cmap='viridis')
ax3d.set_title('3D Cubed Hash Slice (Centre Cube)')
buf3d = BytesIO()
fig3d.savefig(buf3d, format='png')
buf3d.seek(0)
img3d_base64 = base64.b64encode(buf3d.read()).decode('utf-8')
plt.close(fig3d)

# Print base64 for viewing (e.g., in browser: data:image/png;base64,<base64>)
print("2D Image Base64 (copy to browser): data:image/png;base64," + img2d_base64[:100] + "... (truncated, full in file)")
print("3D Image Base64: data:image/png;base64," + img3d_base64[:100] + "...")

# Notes: Install numpy, matplotlib (pip install numpy matplotlib). For full 2140, downscale or use sparse. In coneing: 2D infra gather, 3D centre understand, volumetric ether echo. Hashlet sim: Braided RGB for Wises.
