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
from scipy.interpolate import CubicSpline
from scipy.spatial import Delaunay
import hashlib

# Constants
PHI = (1 + np.sqrt(5)) / 2
WIDTH, HEIGHT = 420 / 110, 1.0
PURPLE_LINES = [1/3, 2/3]
A_SPIRAL, B_SPIRAL_BASE, K_BASE = 0.1, 0.1, 0.1
REDPRINT = {'bg': '#2C001E', 'curve': '#FF5E00', 'points': '#FF9500', 'ghost': '#00FF7F', 'tool': '#A60000'}

# Mersenne Exponents (52 total)
MERSENNE_EXP = [
    2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281,
    3217, 4253, 4423, 9689, 9941, 11213, 19937, 21701, 23209, 44497, 86243,
    110503, 132049, 216091, 756839, 859433, 1257787, 1398269, 2976221, 3021377,
    6972593, 13466917, 20996011, 24036583, 25964951, 30402457, 32582657,
    37156667, 42643801, 43112609, 57885161, 74207281, 77232917, 82589933, 136279841
]
EXP_RANGE_PER_X = (1_100_000_000 - 2) / 1.0

# Model Selection at Startup
model_hash = input("Enter model hash (default: tetrahedron, 'ipod' for iPod shape): ").strip() or "tetrahedron"
hash_obj = hashlib.sha256(model_hash.encode())
hash_val = int(hash_obj.hexdigest(), 16) % 1000 / 1000.0
b_spiral_factor = float(input("Enter b_spiral factor [1.0]: ") or 1.0)
k_factor = float(input("Enter k factor [1.0]: ") or 1.0)
B_SPIRAL, K = B_SPIRAL_BASE * b_spiral_factor, K_BASE * k_factor

# Helper Functions
def compute_green_segment(t, scale=1.0):
    y_base = 0.1 * scale
    nodes = [(1/3, 0), (1/3 + 1/9, y_base), (1/3 + 2/9, y_base), (2/3, 0)]
    x, y = [], []
    pts_per_seg = len(t) // 3
    for i in range(3):
        x1, y1 = nodes[i]
        x2, y2 = nodes[i + 1]
        chord = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        x_m, y_m = (x1 + x2) / 2, (y1 + y2) / 2
        if abs(y2 - y1) < 1e-10:
            x_c, y_c = x_m, y_m + np.sqrt((chord/2)**2)
        else:
            slope = -(x2 - x1) / (y2 - y1)
            a, b = 1 + slope**2, -2 * x_m + 2 * slope * (y_m - slope * x_m)
            c = x_m**2 + (y_m - slope * x_m)**2 - (chord/2)**2
            x_c = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
            y_c = y_m + slope * (x_c - x_m)
        R = np.sqrt((x1 - x_c)**2 + (y1 - y_c)**2)
        theta1, theta2 = np.arctan2(y1 - y_c, x1 - x_c), np.arctan2(y2 - y_c, x2 - x_c)
        if theta2 < theta1: theta2 += 2 * np.pi
        theta = np.linspace(theta1, theta2, pts_per_seg)
        x.extend(x_c + R * np.cos(theta))
        y.extend(y_c + R * np.sin(theta))
    return np.array(x), np.array(y), nodes

def compute_arc_length(x, y):
    return np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))

def tetrahedron_model(n_points=52):
    t = np.linspace(0, 1, n_points)
    theta = 2 * np.pi * t
    r = A_SPIRAL * np.exp(B_SPIRAL * theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = K * np.sin(4 * theta) * (1 + hash_val)
    return x, y, z

def ipod_model(n_points=52):
    t = np.linspace(0, 1, n_points)
    theta = 2 * np.pi * t
    r = 0.5 + 0.2 * np.sin(6 * theta)  # Rounded rectangle profile
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = K * np.sin(12 * theta) * (1 + hash_val)  # Periodic z with curvature control
    return x, y, z

# Core Data
t_green = np.linspace(0, 1, 200)
x_green, y_green, green_nodes = compute_green_segment(t_green)
arc_length_green = compute_arc_length(x_green, y_green)
x_positions = [(exp - 2) / EXP_RANGE_PER_X * WIDTH for exp in MERSENNE_EXP]

# Hybrid Curve
curves = []
total_arc_length = 0
for x_pos in x_positions:
    scale = x_pos / (PURPLE_LINES[1] - PURPLE_LINES[0])
    x_new, y_new, _ = compute_green_segment(t_green, scale)
    x_new -= x_new[0]
    arc_length = compute_arc_length(x_new, y_new)
    total_arc_length += arc_length
    curves.append((x_new, y_new, arc_length))

hybrid_x, hybrid_y = [], []
cumulative_arcs = [0]
current_length = 0
for x_new, y_new, arc_length in curves:
    x_scaled = x_new * (arc_length / arc_length_green) + current_length
    hybrid_x.extend(x_scaled)
    hybrid_y.extend(y_new)
    current_length += arc_length
    cumulative_arcs.append(current_length)

# 3D Model Selection
model_func = ipod_model if model_hash.lower() == 'ipod' else tetrahedron_model
x_base, y_base, z_base = model_func()
cs_x = CubicSpline(np.linspace(0, 1, 52), x_base, bc_type='periodic')
cs_y = CubicSpline(np.linspace(0, 1, 52), y_base, bc_type='periodic')
cs_z = CubicSpline(np.linspace(0, 1, 52), z_base, bc_type='periodic')
t_fine = np.linspace(0, 1, 200)
x_3d, y_3d, z_3d = cs_x(t_fine), cs_y(t_fine), cs_z(t_fine)

# Mersenne Points and Steps
indices = np.linspace(0, 199, 52, dtype=int)  # Sample 52 points from 0 to 199
mersenne_3d = [(x_3d[i], y_3d[i], z_3d[i]) for i in indices]
steps = [(mersenne_3d[:i+1], [(p[0], p[1]) for p in mersenne_3d[:i+1]], 
          Delaunay([(p[0], p[1]) for p in mersenne_3d[:i+1]]) if i >= 2 else None) 
         for i in range(52)]

# Plot Setup
fig = plt.figure(figsize=(20, 10), facecolor=REDPRINT['bg'])
ax1 = fig.add_subplot(121, projection='3d', facecolor=REDPRINT['bg'])
ax2 = fig.add_subplot(122, facecolor=REDPRINT['bg'])

# State
current_step, mode, selected_points, kappa_shift = 0, None, [], 0.0
divider_dist, last_divider = None, None
new_points_3d = []

def update_plot():
    ax1.clear()
    ax2.clear()
    points_3d, points_2d, tri = steps[current_step]
    
    # Adjusted 3D Curve with Kappa Tilt
    z_adj = z_3d + kappa_shift * np.sin(t_fine * 2 * np.pi)
    ax1.plot(x_3d, y_3d, z_adj, color=REDPRINT['curve'], lw=2)
    ax1.scatter([p[0] for p in points_3d], [p[1] for p in points_3d], 
                [p[2] + kappa_shift * np.sin(i * 0.1) for i, p in enumerate(points_3d)], 
                c=REDPRINT['points'], s=50, picker=5)
    if new_points_3d:
        ax1.scatter([p[0] for p in new_points_3d], [p[1] for p in new_points_3d], 
                    [p[2] for p in new_points_3d], c='blue', marker='o', s=50)
    
    # 2D Projection
    ax2.plot([0, WIDTH, WIDTH, 0, 0], [0, 0, HEIGHT, HEIGHT, 0], 'k-')
    for x in PURPLE_LINES:
        ax2.plot([x, x], [0, HEIGHT], 'm-')
    ax2.plot(hybrid_x, hybrid_y, color=REDPRINT['curve'])
    ax2.scatter([p[0] for p in points_2d], [p[1] for p in points_2d], c=REDPRINT['points'], s=50, picker=5)
    if tri:
        for simplex in tri.simplices:
            for i in range(3):
                p1, p2 = points_2d[simplex[i]], points_2d[simplex[(i + 1) % 3]]
                ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', alpha=0.3)
    if new_points_3d:
        ax2.scatter([p[0] for p in new_points_3d], [p[1] for p in new_points_3d], c='blue', marker='o', s=50)
    
    ax2.set_xlim(0, WIDTH)
    ax2.set_ylim(-0.2, HEIGHT + 0.2)
    ax2.set_aspect('equal')
    ax1.set_title(f"Step {current_step + 1}: M{MERSENNE_EXP[current_step]}", color='white')
    ax2.set_title("2D Projection", color='white')
    
    # Toolface HUD
    hud_text = f"Mode: {mode or 'None'}\nPoints: {len(selected_points)}\nKappa: {kappa_shift:.2f}"
    if divider_dist:
        hud_text += f"\nDivider: {divider_dist:.4f}"
    fig.text(0.01, 0.99, hud_text, color='white', fontsize=10, va='top', ha='left', 
             bbox=dict(facecolor=REDPRINT['tool'], alpha=0.8))
    
    fig.canvas.draw()

def on_key(event):
    global current_step, mode, selected_points, kappa_shift, divider_dist, last_divider
    if event.key == 'n' and current_step < 51:
        current_step += 1
        update_plot()
    elif event.key == 'b' and current_step > 0:
        current_step -= 1
        update_plot()
    elif event.key == 'm':
        mode = 'measure' if mode != 'measure' else None
        selected_points = []
        divider_dist = None
        last_divider = None
        print(f"Measure mode: {'On' if mode else 'Off'}")
        update_plot()
    elif event.key == 'escape':
        mode = None
        selected_points = []
        kappa_shift = 0.0
        divider_dist = None
        update_plot()

def on_pick(event):
    global selected_points, kappa_shift, divider_dist, last_divider, new_points_3d
    if not mode or not event.mouseevent.inaxes:
        return
    artist = event.artist
    ind = event.ind[0]
    
    if event.mouseevent.inaxes == ax1 and isinstance(artist, plt.collections.PathCollection):
        x, y, z = artist._offsets3d
        point = (x[ind], y[ind], z[ind])
        selected_points.append(point)
        print(f"Selected 3D: {point}")
        
        if mode == 'measure':
            if len(selected_points) == 2:
                dist = np.linalg.norm(np.array(selected_points[1]) - np.array(selected_points[0]))
                print(f"Distance: {dist:.4f}")
                kappa_shift = dist * 0.1  # Tilt model based on ruler movement
                ax1.plot([p[0] for p in selected_points], [p[1] for p in selected_points], 
                         [p[2] for p in selected_points], c=REDPRINT['tool'], lw=2)
                update_plot()
            elif len(selected_points) == 3:
                v1 = np.array(selected_points[0]) - np.array(selected_points[1])
                v2 = np.array(selected_points[2]) - np.array(selected_points[1])
                angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10), -1, 1)))
                print(f"Angle: {angle:.2f}°")
                # Ghost Curves
                for offset in [-0.05, 0.05]:
                    gx = [p[0] + offset for p in selected_points]
                    gy = [p[1] for p in selected_points]
                    gz = [p[2] + K * np.sin(angle * np.pi / 180) for p in selected_points]
                    ax1.plot(gx, gy, gz, c=REDPRINT['ghost'], lw=1, alpha=0.5)
                selected_points = []
                update_plot()
    
    elif event.mouseevent.inaxes == ax2 and event.mouseevent.button == 3:
        x, y = artist.get_offsets()[ind]
        point = next((p for p in mersenne_3d + new_points_3d if np.isclose(p[0], x) and np.isclose(p[1], y)), (x, y, 0.0))
        selected_points.append(point)
        if len(selected_points) == 2:
            divider_dist = np.sqrt((selected_points[1][0] - selected_points[0][0])**2 + 
                                   (selected_points[1][1] - selected_points[0][1])**2)
            last_divider = selected_points[1]
            print(f"Divider set: {divider_dist:.4f}")
            selected_points = []

def on_click(event):
    global last_divider, new_points_3d
    if event.inaxes == ax2 and mode == 'measure' and divider_dist and event.button == 1:
        x, y = event.xdata, event.ydata
        if last_divider:
            dx, dy = x - last_divider[0], y - last_divider[1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0:
                dx, dy = dx / dist * divider_dist, dy / dist * divider_dist
                new_point = (last_divider[0] + dx, last_divider[1] + dy, 0.0)
                new_points_3d.append(new_point)
                last_divider = new_point
                print(f"New point: {new_point}")
                update_plot()

# Initial Plot and Events
update_plot()
fig.canvas.mpl_connect('key_press_event', on_key)
fig.canvas.mpl_connect('pick_event', on_pick)
fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()

# Parametric Output for Machine Interface
print("Parametric Output:")
for i, (x, y, z) in enumerate(mersenne_3d):
    speed = int(hashlib.sha256(f"{x}{y}{z}".encode()).hexdigest()[-4:], 16) % 1000 / 1000.0
    print(f"Point {i}: ({x:.4f}, {y:.4f}, {z:.4f}), Speed: {speed:.4f}")

