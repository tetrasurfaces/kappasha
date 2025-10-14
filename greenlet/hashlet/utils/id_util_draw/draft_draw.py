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
import hashlib

# Try to import mpld3; if it fails, set a flag to skip HTML export
try:
    import mpld3
    MPLD3_AVAILABLE = True
except ImportError:
    print("mpld3 not installed. HTML export will be skipped. Install mpld3 with 'pip install mpld3' to enable.")
    MPLD3_AVAILABLE = False

# Define the A4 and A3 page dimensions
width = 420 / 110  # A3 long side = 420mm, A4 short side = 110mm
height = 1.0  # A3 short side = 297mm
purple_lines = [1/3, 2/3]  # Divide the first A4 short side into thirds
unit_per_mm = 1.0 / 110
scale_label = f"Scale: 1mm = {unit_per_mm:.5f} units (A4 short side = 110mm)"

# Define the golden spiral
phi = (1 + np.sqrt(5)) / 2
b = np.log(phi) / (np.pi / 2)
a = 0.01

# Define κθπ for the green segment
kappa = 1 / phi
theta_max = kappa * np.pi**2 / phi
print(f"κθπ (at θ_max): {theta_max:.2f} radians")

# Compute the full spiral
theta_full = np.linspace(0, 10 * np.pi, 1000)
r_full = a * np.exp(b * theta_full)
x_full = r_full * np.cos(theta_full)
y_full = r_full * np.sin(theta_full)

# Compute the green segment (θ from π to 2π)
theta_green = np.linspace(np.pi, 2 * np.pi, 200)
r_green = a * np.exp(b * theta_green)
x_green = r_green * np.cos(theta_green)
y_green = r_green * np.sin(theta_green)

# Compute the chord and shift
x1, y1 = x_green[0], y_green[0]
x2, y2 = x_green[-1], y_green[-1]
chord_length = np.abs(x2 - x1)
print(f"Original chord length: {chord_length:.4f}")

# Shift so the segment starts at x=0
x_green_shifted = x_green - x1
x_green_final = x_green_shifted
# Scale to match the target chord length (between purple lines)
target_chord = purple_lines[1] - purple_lines[0]
scale_factor = target_chord / chord_length
x_green_scaled = x_green_final * scale_factor
y_green_scaled = y_green * scale_factor
# Shift to start at the first purple line
x_green_final = x_green_scaled + purple_lines[0]

# Compute κ at 2πR for the green segment
r_max = a * np.exp(b * theta_max)
two_pi_r = 2 * np.pi * r_max
kappa_at_2piR = two_pi_r / phi
print(f"2πR: {two_pi_r:.4f}, κ at 2πR: {kappa_at_2piR:.4f}")

# Define the 52 Mersenne prime exponents
mersenne_exponents = [
    2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281,
    3217, 4253, 4423, 9689, 9941, 11213, 19937, 21701, 23209, 44497, 86243,
    110503, 132049, 216091, 756839, 859433, 1257787, 1398269, 2976221, 3021377,
    6972593, 13466917, 20996011, 24036583, 25964951, 30402457, 32582657,
    37156667, 42643801, 43112609, 57885161, 74207281, 77232917, 82589933
]
mersenne_exponents.append(1_000_000_007)

# Map exponents to x-positions (0 to width)
min_exponent = 2
max_exponent_at_x1 = 1_100_000_000
max_x = width
exponent_range_per_x = (max_exponent_at_x1 - min_exponent) / 1.0
max_exponent = min_exponent + exponent_range_per_x * max_x
print(f"Exponent at x={max_x:.3f}: {max_exponent:.0f}")

x_positions = [(exponent - min_exponent) / exponent_range_per_x * max_x for exponent in mersenne_exponents]

# Create the 52 curves
curves = []
curve_lines = []
for i, (exponent, x_pos) in enumerate(zip(mersenne_exponents, x_positions)):
    scale = x_pos / chord_length if chord_length != 0 else 1.0
    x_new = x_green * scale
    y_new = y_green * scale
    x_new_shifted = x_new - x_new[0]
    curves.append((x_new_shifted, y_new, f"M{exponent}"))
    curve_lines.append(None)

# A4 short edge divisions (110 parts)
division_step = 1.0 / 110
division_positions = np.arange(0, 1.0 + division_step, division_step)

# Scale key for the title
scale_key_positions = division_positions[::10]
scale_key_exponents = [int(min_exponent + (max_exponent_at_x1 - min_exponent) * (x / 1.0)) for x in scale_key_positions]
scale_key_text = "Scale (x=0 to 1): " + ", ".join([f"{x:.2f}: {exp:,}" for x, exp in zip(scale_key_positions, scale_key_exponents)])

# Flags for Mersenne primes
flag_length = 0.5
start_y = -0.1
wedge_angles = np.linspace(90, 360, len(curves))
flag_positions = []
annotation_objects = []
harmonic_frequencies = []
circle_markers = []
min_exp = min(mersenne_exponents)
max_exp = max(mersenne_exponents)
log_min = np.log(min_exp)
log_max = np.log(max_exp)
min_freq_exp = -4.459
max_freq_exp = 5.506
exponent_range = max_freq_exp - min_freq_exp
log_range = log_max - log_min
for i, (x_new, y_new, label) in enumerate(curves):
    x_end = x_new[-1]
    y_end = y_new[-1]
    x_start = x_end
    y_start = start_y
    angle = np.deg2rad(wedge_angles[i])
    x_flag = x_start + flag_length * np.cos(angle)
    y_flag = y_start + flag_length * np.sin(angle)
    exponent = mersenne_exponents[i]
    scaled_exponent = min_freq_exp + (np.log(exponent) - log_min) / log_range * exponent_range
    freq = 440 * 2**scaled_exponent
    harmonic_frequencies.append(freq)
    flag_positions.append((x_end, y_end, x_start, y_start, x_flag, y_flag, label, freq))
    angle_deg = wedge_angles[i]
    if (angle_deg - 90) % 5 == 0:
        angle_rad = np.deg2rad(angle_deg)
        x_marker = x_start + (flag_length * 0.5) * np.cos(angle_rad)
        y_marker = y_start + (flag_length * 0.5) * np.sin(angle_rad)
        circle_markers.append((x_marker, y_marker))

# Plot setup
fig, ax = plt.subplots(figsize=(12, 8 * width))
# A3 page
ax.plot([0, width, width, 0, 0], [0, 0, height, height, 0], 'k-', label='A3 Page')
# First A4 page
ax.plot([0, 1.0, 1.0, 0, 0], [0, 0, height, height, 0], 'k--', label='A4 Page 1')
# Second A4 page
ax.plot([1.0, 2.0, 2.0, 1.0, 1.0], [0, 0, height, height, 0], 'k--', label='A4 Page 2')
# Purple lines (in first A4)
for x in purple_lines:
    ax.plot([x, x], [0, height], 'm-')
# Red datum line
ax.plot([0, max_x], [0, 0], 'r-')
# A4 short edge divisions (first A4 only)
for x in division_positions:
    ax.plot([x, x], [0, 0.02], 'k-', alpha=0.3)
# Plot circle division markers
for x_marker, y_marker in circle_markers:
    ax.plot(x_marker, y_marker, 'k.', markersize=3)
# Full spiral
ax.plot(x_full, y_full, 'k-')
# Green segment
ax.plot(x_green_final, y_green_scaled, 'g-')
# 52 Mersenne prime curves
colors = plt.cm.viridis(np.linspace(0, 1, len(curves)))
for i, (x_new, y_new, label) in enumerate(curves):
    line, = ax.plot(x_new, y_new, color=colors[i])
    curve_lines[i] = line
# Flags and staggered labels
label_y_offset = 0.05
show_harmonics = False
harmonic_texts = []
for i, (x_end, y_end, x_start, y_start, x_flag, y_flag, label, freq) in enumerate(flag_positions):
    ax.plot([x_end, x_start], [y_end, y_start], 'k--', alpha=0.3)
    ax.plot([x_start, x_flag], [y_start, y_flag], 'k-', alpha=0.5)
    y_label = y_flag - (i % 5) * label_y_offset
    text = ax.text(x_flag, y_label, label, ha='left', va='top', fontsize=6, rotation=45, picker=5)
    harmonic_text = ax.text(x_flag, y_label - 0.1, f"{freq:.1f} Hz", ha='left', va='top', fontsize=6, rotation=45, visible=False)
    annotation_objects.append((text, i))
    harmonic_texts.append(harmonic_text)

# Golden window 1 (vertical at x = 1/3)
idx_crossings_x = np.where(np.diff(np.sign(x_full - purple_lines[0])))[0]
if len(idx_crossings_x) >= 2:
    y1 = y_full[idx_crossings_x[0]]
    y2 = y_full[idx_crossings_x[1]]
    golden_window_1 = np.abs(y2 - y1)
    print(f"Golden Window 1 at x={purple_lines[0]}: {golden_window_1:.4f}")
    ax.fill_between([purple_lines[0] - 0.05, purple_lines[0] + 0.05], min(y1, y2), max(y1, y2), color='yellow', alpha=0.5)

# Golden window 2 (horizontal at y = 1/3)
idx_crossings_y = np.where(np.diff(np.sign(y_full - 1/3)))[0]
if len(idx_crossings_y) >= 2:
    x1 = x_full[idx_crossings_y[0]]
    x2 = x_full[idx_crossings_y[1]]
    golden_window_2 = np.abs(x2 - x1)
    print(f"Golden Window 2 at y=1/3: {golden_window_2:.4f}")
    ax.fill_betweenx([1/3 - 0.05, 1/3 + 0.05], min(x1, x2), max(x1, x2), color='orange', alpha=0.5)

# Scale label
ax.text(max_x, 1.10337, scale_label, ha='right', va='bottom', fontsize=8)

# Set plot properties
ax.set_xlim(-0.1, max_x + 0.1)
ax.set_ylim(-1.5, height + 0.1)
ax.set_xlabel('x (Exponents: 2 to 11B)')
ax.set_ylabel('y')
ax.set_title('Golden Spiral with 52 Mersenne Prime Curves on A3 Page\n' + scale_key_text, fontsize=10, pad=20)
ax.grid(True)
ax.set_aspect('equal')

# Highlighting functionality
highlighted = [None, None]

def on_pick(event):
    global highlighted
    artist = event.artist
    for text, idx in annotation_objects:
        if artist == text:
            if highlighted[0] is not None:
                highlighted[0].set_color('black')
                highlighted[0].set_weight('normal')
                curve_lines[highlighted[1]].set_linewidth(1.0)
                curve_lines[highlighted[1]].set_color(colors[highlighted[1]])
            text.set_color('red')
            text.set_weight('bold')
            curve_lines[idx].set_linewidth(2.0)
            curve_lines[idx].set_color('red')
            highlighted = [text, idx]
            fig.canvas.draw()
            break

def on_click_deselect(event):
    global highlighted
    if event.inaxes != ax:
        return
    clicked_on_annotation = False
    for text, idx in annotation_objects:
        if text.contains(event)[0]:
            clicked_on_annotation = True
            break
    if not clicked_on_annotation and highlighted[0] is not None:
        highlighted[0].set_color('black')
        highlighted[0].set_weight('normal')
        curve_lines[highlighted[1]].set_linewidth(1.0)
        curve_lines[highlighted[1]].set_color(colors[highlighted[1]])
        highlighted = [None, None]
        fig.canvas.draw()

# Curve cache for hashing
curve_cache = {}

def compute_curve_points(theta_start, theta_end, num_points, scale_factor, rotation_angle=0):
    # Create a hash key based on parameters
    key = f"{theta_start:.2f}:{theta_end:.2f}:{num_points}:{scale_factor:.4f}:{rotation_angle:.2f}"
    key_hash = hashlib.md5(key.encode()).hexdigest()
    if key_hash in curve_cache:
        return curve_cache[key_hash]
    theta = np.linspace(theta_start, theta_end, num_points)
    r = scale_factor * a * np.exp(b * theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    # Apply rotation
    if rotation_angle != 0:
        angle_rad = np.deg2rad(rotation_angle)
        x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
        x, y = x_rot, y_rot
    curve_cache[key_hash] = (x, y)
    return x, y

# Dynamic LOD
def get_num_points_for_curve():
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    view_width = xlim[1] - xlim[0]
    view_height = ylim[1] - ylim[0]
    # Base number of points when fully zoomed out
    base_points = 20
    max_points = 200
    # Zoom factor: smaller view range means more zoom
    full_range = max_x  # Full x-range when zoomed out
    zoom_factor = full_range / view_width
    num_points = int(base_points + (max_points - base_points) * min(zoom_factor / 10, 1))
    return max(base_points, min(max_points, num_points))

# Cursor, spiral, and circumference setup
cursor, = ax.plot([], [], 'ro', markersize=8, label='κ Spiral Cursor', visible=False)
cursor_spiral, = ax.plot([], [], 'g-', alpha=0.5, visible=False)
cursor_circumference = plt.Circle((0, 0), 0, color='b', fill=False, linestyle='--', alpha=0.5, visible=False)
ax.add_patch(cursor_circumference)
cursor_text = ax.text(max_x / 2, 1.15, '', ha='center', va='bottom', fontsize=8, visible=False)
baseline_spiral, = ax.plot([], [], 'g-', alpha=0.5, label='Baseline Spiral', visible=False)
baseline_spiral_2, = ax.plot([], [], 'g-', alpha=0.5, label='Baseline Spiral 2', visible=False)
# Crosslines
vertical_line, = ax.plot([], [], 'k--', alpha=0.5, visible=False)
horizontal_line, = ax.plot([], [], 'k--', alpha=0.5, visible=False)
vertical_label = ax.text(target_chord, height + 0.05, f'Chord: {target_chord:.4f}', ha='center', va='bottom', fontsize=8, visible=False)
# Protractor elements
protractor_line, = ax.plot([], [], 'b-', alpha=0.8, visible=False)
protractor_text = ax.text(0, 0, '', ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.8), visible=False)
protractor_arc, = ax.plot([], [], 'b-', alpha=0.5, visible=False)
protractor_spiral_2, = ax.plot([], [], 'g-', alpha=0.5, visible=False)
# Baseline angle (grey ghost line)
baseline_angle_line, = ax.plot([0, max_x], [0, 0], 'grey', alpha=0.3, linestyle='--', visible=False)
# Swinging ghost curves
ghost_curves = []
for _ in range(4):  # ±5°, ±10° (4 curves total)
    line, = ax.plot([], [], 'grey', alpha=0.2, visible=False)
    ghost_curves.append(line)
# Ruler elements
ruler_active = False
ruler_points = []
ruler_line, = ax.plot([], [], 'k-', linewidth=2, visible=False)
ruler_divisions = []
for _ in range(10):  # Up to 10 division markers
    marker, = ax.plot([], [], 'k|', markersize=10, markeredgewidth=2, visible=False)
    ruler_divisions.append(marker)
ruler_vanishing_line, = ax.plot([], [], 'k--', alpha=0.5, visible=False)

# Variables to track state
protractor_active = False

def toggle_protractor(event):
    global protractor_active
    if event.key == 'p':
        protractor_active = not protractor_active
        cursor.set_visible(protractor_active)
        cursor_spiral.set_visible(protractor_active)
        cursor_circumference.set_visible(protractor_active)
        cursor_text.set_visible(protractor_active)
        baseline_spiral.set_visible(protractor_active)
        baseline_spiral_2.set_visible(protractor_active)
        vertical_line.set_visible(protractor_active)
        horizontal_line.set_visible(protractor_active)
        vertical_label.set_visible(protractor_active)
        protractor_line.set_visible(protractor_active)
        protractor_text.set_visible(protractor_active)
        protractor_arc.set_visible(protractor_active)
        protractor_spiral_2.set_visible(protractor_active)
        baseline_angle_line.set_visible(protractor_active)
        for curve in ghost_curves:
            curve.set_visible(protractor_active)
        print(f"Protractor tool {'enabled' if protractor_active else 'disabled'}")
        fig.canvas.draw()

def toggle_ruler(event):
    global ruler_active, ruler_points
    if event.key == 'r':
        ruler_active = not ruler_active
        ruler_line.set_visible(ruler_active)
        ruler_vanishing_line.set_visible(ruler_active)
        for marker in ruler_divisions:
            marker.set_visible(ruler_active)
        if not ruler_active:
            ruler_points = []
            ruler_line.set_data([], [])
            ruler_vanishing_line.set_data([], [])
            for marker in ruler_divisions:
                marker.set_data([], [])
        print(f"Ruler tool {'enabled' if ruler_active else 'disabled'}")
        fig.canvas.draw()

def on_click_ruler(event):
    global ruler_points
    if not ruler_active or event.inaxes != ax:
        return
    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return
    ruler_points.append((x, y))
    if len(ruler_points) == 1:
        print("Ruler start point set:", ruler_points[0])
    elif len(ruler_points) == 2:
        print("Ruler depth point set:", ruler_points[1])
        # Draw the ruler line
        x1, y1 = ruler_points[0]
        x2, y2 = ruler_points[1]
        # Extend the ruler line to a reasonable length
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length == 0:
            length = 1e-10
        dx_norm = dx / length
        dy_norm = dy / length
        ruler_length = max_x  # Extend to the edge of the plot
        x_end = x1 + ruler_length * dx_norm
        y_end = y1 + ruler_length * dy_norm
        ruler_line.set_data([x1, x_end], [y1, y_end])
        # Draw the vanishing line (from x2, y2 to a far point)
        vanishing_x = x2 + 10 * (x2 - x1)
        vanishing_y = y2 + 10 * (y2 - y1)
        ruler_vanishing_line.set_data([x2, vanishing_x], [y2, vanishing_y])
        # Compute perspective divisions (halves, thirds, quarters)
        divisions = {
            'halves': [0.5],
            'thirds': [1/3, 2/3],
            'quarters': [0.25, 0.5, 0.75]
        }
        # Perspective projection
        for i, (name, t_values) in enumerate(divisions.items()):
            for j, t in enumerate(t_values):
                if i * len(t_values) + j >= len(ruler_divisions):
                    break
                # Perspective parameter: adjust t based on distance to vanishing point
                d1 = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                d2 = np.sqrt((x_end - x2)**2 + (y_end - y2)**2)
                if d2 == 0:
                    d2 = 1e-10
                perspective_t = t * (d1 + d2) / (d1 + t * (d2 - d1))
                x_div = x1 + perspective_t * (x_end - x1)
                y_div = y1 + perspective_t * (y_end - y1)
                ruler_divisions[i * len(t_values) + j].set_data([x_div], [y_div])
        # Check for Mersenne prime alignment
        for i, x_pos in enumerate(x_positions):
            # Find the closest point on the ruler line to x_pos
            t = ((x_pos - x1) * (x_end - x1) + (0 - y1) * (y_end - y1)) / (ruler_length**2)
            t = max(0, min(1, t))
            x_proj = x1 + t * (x_end - x1)
            y_proj = y1 + t * (y_end - y1)
            distance = np.sqrt((x_pos - x_proj)**2 + (0 - y_proj)**2)
            if distance < 0.05:  # Threshold for "alignment"
                print(f"Mersenne prime M{mersenne_exponents[i]} aligns with ruler at x={x_pos:.4f}")
        ruler_points = []  # Reset for the next ruler
        fig.canvas.draw()

def on_motion(event):
    if not protractor_active or event.inaxes != ax:
        return
    x, y = event.xdata, event.ydata
    if x is None or y is None:
        return
    # Update cursor position
    cursor.set_data([x], [y])
    # Update circumference
    radius = np.sqrt(x**2 + y**2)
    cursor_circumference.set_center((x, y))
    cursor_circumference.set_radius(radius)
    # Dynamic LOD: Adjust number of points based on zoom
    num_points = get_num_points_for_curve()
    # Update cursor spiral
    x_spiral, y_spiral = compute_curve_points(np.pi, 2 * np.pi, num_points, 1.0)
    cursor_spiral.set_data(x + x_spiral, y + y_spiral)
    # Update baseline spiral (indexed at (0,0))
    x_base = 0.0
    scale_factor = (event.xdata / max_x) if event.xdata > 0 else 0.01
    scaled_a = a * scale_factor
    height_factor = (event.ydata / height) if event.ydata > 0 else 0.01
    x_base_spiral, y_base_spiral = compute_curve_points(2 * np.pi, np.pi, num_points, scale_factor)
    x_base_spiral = x_base + x_base_spiral * np.abs(np.cos(np.linspace(2 * np.pi, np.pi, num_points)))
    y_base_spiral = y_base_spiral * height_factor
    baseline_spiral.set_data(x_base_spiral, y_base_spiral)
    # Compute the chord length of the baseline spiral
    x_start = x_base_spiral[0]
    y_start = y_base_spiral[0]
    x_end = x_base_spiral[-1]
    y_end = y_base_spiral[-1]
    baseline_chord = np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
    # Update second baseline spiral (indexed at (1.0, 0))
    x_base_2 = 1.0
    x_base_spiral_2, y_base_spiral_2 = compute_curve_points(2 * np.pi, np.pi, num_points, scale_factor)
    x_base_spiral_2 = x_base_2 + x_base_spiral_2 * np.abs(np.cos(np.linspace(2 * np.pi, np.pi, num_points)))
    y_base_spiral_2 = y_base_spiral_2 * height_factor
    baseline_spiral_2.set_data(x_base_spiral_2, y_base_spiral_2)
    # Compute the chord length of the second baseline spiral
    x_start_2 = x_base_spiral_2[0]
    y_start_2 = y_base_spiral_2[0]
    x_end_2 = x_base_spiral_2[-1]
    y_end_2 = y_base_spiral_2[-1]
    baseline_chord_2 = np.sqrt((x_end_2 - x_start_2)**2 + (y_end_2 - y_start_2)**2)
    # Update crosslines
    vertical_line.set_data([target_chord, target_chord], [0, height])
    vertical_label.set_position((target_chord, height + 0.05))
    if y > 0:
        horizontal_line.set_data([0, max_x], [y, y])
    else:
        horizontal_line.set_data([], [])
    # Update protractor line (from (0,0) to mouse position)
    anchor_x, anchor_y = 0.0, 0.0
    protractor_line.set_data([anchor_x, x], [anchor_y, y])
    # Compute the angle relative to the baseline (y=0)
    dx = x - anchor_x
    dy = y - anchor_y
    angle = np.arctan2(dy, dx) * 180 / np.pi
    # Update protractor arc
    mid_x = (anchor_x + x) / 2
    mid_y = (anchor_y + y) / 2
    radius_arc = np.sqrt(dx**2 + dy**2) / 4
    start_angle = 0
    end_angle = angle
    theta_arc = np.linspace(np.deg2rad(start_angle), np.deg2rad(end_angle), num_points)
    x_arc = mid_x + radius_arc * np.cos(theta_arc)
    y_arc = mid_y + radius_arc * np.sin(theta_arc)
    protractor_arc.set_data(x_arc, y_arc)
    # Update swinging ghost curves
    offsets = [-10, -5, 5, 10]  # Degrees
    for i, offset in enumerate(offsets):
        angle_offset = angle + offset
        x_ghost, y_ghost = compute_curve_points(np.pi, 2 * np.pi, num_points // 2, 1.0, angle_offset)
        ghost_curves[i].set_data(anchor_x + x_ghost, anchor_y + y_ghost)
    # Update protractor spiral at the mouse position
    line_vec = np.array([x - anchor_x, y - anchor_y])
    line_len = np.sqrt(dx**2 + dy**2)
    if line_len == 0:
        line_len = 1e-10
    normal_vec = np.array([-(y - anchor_y), x - anchor_x]) / line_len
    x_spiral, y_spiral = compute_curve_points(np.pi, 2 * np.pi, num_points, 1.0)
    x_mirrored = []
    y_mirrored = []
    for xs, ys in zip(x_spiral, y_spiral):
        point = np.array([xs, ys])
        v = point - np.array([anchor_x, anchor_y])
        projection = np.dot(v, normal_vec) * normal_vec
        mirrored_point = point - 2 * projection
        x_mirrored.append(mirrored_point[0])
        y_mirrored.append(mirrored_point[1])
    protractor_spiral_2.set_data(x + x_mirrored, y + y_mirrored)
    # Update protractor text
    protractor_text.set_position((mid_x, mid_y))
    protractor_text.set_text(f'Angle: {angle:.2f}°\nκ at 2πR: {kappa_at_2piR:.4f}')
    # Calculate chord length from cursor to the start of the green segment
    x_start_green, y_start_green = x_green_final[0], y_green_scaled[0]
    chord_to_green = np.sqrt((x - x_start_green)**2 + (y - y_start_green)**2)
    # Update cursor text
    text_str = (f'κ: {scale_factor:.4f}\n'
                f'Height Factor: {height_factor:.4f}\n'
                f'Cursor: ({x:.4f}, {y:.4f})\n'
                f'Chord to Green: {chord_to_green:.4f}\n'
                f'Baseline Chord (x=0): {baseline_chord:.4f}\n'
                f'Baseline Chord (x=1): {baseline_chord_2:.4f}')
    cursor_text.set_text(text_str)
    fig.canvas.draw()

def toggle_harmonics(event):
    global show_harmonics
    if event.key == 'h':
        show_harmonics = not show_harmonics
        for text in harmonic_texts:
            text.set_visible(show_harmonics)
        print(f"Harmonic frequencies {'shown' if show_harmonics else 'hidden'}")
        fig.canvas.draw()

def save_plot(event):
    if event.key == 'w':
        plt.savefig("nu_curve.png", dpi=300, bbox_inches='tight')
        print("Plot saved as nu_curve.png")
        if MPLD3_AVAILABLE:
            mpld3.save_html(fig, "nu_curve.html")
            print("Interactive plot saved as nu_curve.html")
        else:
            print("Skipping HTML export because mpld3 is not installed.")

fig.canvas.mpl_connect('pick_event', on_pick)
fig.canvas.mpl_connect('button_press_event', on_click_deselect)
fig.canvas.mpl_connect('key_press_event', toggle_protractor)
fig.canvas.mpl_connect('key_press_event', toggle_ruler)
fig.canvas.mpl_connect('button_press_event', on_click_ruler)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('key_press_event', toggle_harmonics)
fig.canvas.mpl_connect('key_press_event', save_plot)

plt.show()
