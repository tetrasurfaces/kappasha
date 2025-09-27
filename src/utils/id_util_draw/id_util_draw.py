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

# Set precision for Decimal
getcontext().prec = 28
# Suppress warnings
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
# A3 landscape dimensions (normalized: width long side, height short side)
WIDTH = 420 / 297 # A3 landscape: 420mm width, 297mm height, normalized height=1.0
HEIGHT = 1.0
PURPLE_LINES = [1/3, 2/3] # Dividers on the width
unit_per_mm = 1.0 / 297 # Normalize to A3 short side
scale_label = f"Scale: 1mm = {unit_per_mm:.5f} units (A3 short side = 297mm)"
# Dreyfuss human factors: Optimal eye distance ~20 inches (508mm)
EYE_DISTANCE = 500 * unit_per_mm # Normalized eye distance to viewport
HORIZON_HEIGHT = HEIGHT * 0.5 # Default horizon line at half height
EYE_LINE = HORIZON_HEIGHT # Eye line coincides with horizon
# Golden spiral parameters
PHI = (1 + np.sqrt(5)) / 2
kappa = 1 / PHI
A_SPIRAL = 0.001 # Scaled down slightly from 0.01 to fit better
B_SPIRAL = np.log(PHI) / (np.pi / 2)
# Global variables for interactive modes
protractor_active = False
ruler_active = False
draw_mode = False
dimension_active = False
selected_curve = None
hidden_elements = []
protractor_points = []
protractor_line = None
protractor_text = None
ruler_points = []
ruler_line = None
ruler_text = None
dimension_labels = []
drawing_points = [] # Kappa nodes (first endpoint of each greenchord)
kappas = [] # Kappa values at each node
green_curve_line = None # Single plot object for the interoperated greencurve
CLOSE_THRESHOLD = 0.05 # Distance to first point to consider closing
vanishing_points = [] # Vanishing points for each triangulation
previous_kappa = 1.0 # Initial kappa for decay
curvature = 1.0 # Initial curvature (kappa)
current_vertices = None
current_faces = None
# Compute golden spiral
def compute_golden_spiral():
    theta = np.linspace(0, 10 * np.pi, 1000)
    r = A_SPIRAL * np.exp(B_SPIRAL * theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    # Scale down to fit within page bounds
    scale_factor = min(WIDTH, HEIGHT) / (2 * np.max(np.abs([x, y]))) * 0.8 # 80% of max to fit comfortably
    x *= scale_factor
    y *= scale_factor
    return x, y
# Custom interoperations for greencurve (custom kappa NURBS with endpoint kappa and theta decay, upgraded to degree 5 for G4 approx G5)
def custom_interoperations_green_curve(points, kappas):
    """
    Custom kappa NURBS-like curve through points with endpoint kappa and theta decay for curvature continuity.
   
    Args:
        points (list): List of (x, y) points (kappa nodes).
        kappas (list): Kappa values at each node.
   
    Returns:
        tuple: (x, y) arrays for the interoperated curve.
    """
    if len(points) < 2:
        return np.array([]), np.array([])
   
    x_points = [p[0] for p in points]
    y_points = [p[1] for p in points]
    t = np.cumsum([0] + [np.sqrt((x_points[i+1] - x_points[i])**2 + (y_points[i+1] - y_points[i])**2) for i in range(len(points)-1)])
    t_fine = np.linspace(0, t[-1], 1000) if t[-1] > 0 else np.linspace(0, 1, 1000)
   
    # Upgrade to degree 5 for higher continuity (G4, approximating G5)
    degree = 5
   
    # Custom NURBS basis functions (recursive for higher degree)
    def nurbs_basis(u, i, p, knots):
        if p == 0:
            return 1.0 if knots[i] <= u < knots[i+1] else 0.0
        if knots[i+p] == knots[i]:
            c1 = 0.0
        else:
            c1 = (u - knots[i]) / (knots[i+p] - knots[i]) * nurbs_basis(u, i, p-1, knots)
        if knots[i+p+1] == knots[i+1]:
            c2 = 0.0
        else:
            c2 = (knots[i+p+1] - u) / (knots[i+p+1] - knots[i+1]) * nurbs_basis(u, i+1, p-1, knots)
        return c1 + c2
   
    # Generate knots based on theta (distance), non-uniform for decay, adjusted for higher degree
    knots = [0] * (degree + 1) + list(np.cumsum([kappas[i] for i in range(len(points))])) + [t[-1]] * (degree + 1) # Clamped knots for endpoint interpolation
   
    x_fine = []
    y_fine = []
    for u in t_fine:
        x_val = 0.0
        y_val = 0.0
        n = len(points) - 1
        for i in range(n + 1):
            b = nurbs_basis(u, i, degree, knots) # Higher degree basis
            weight = kappas[i] if i < len(kappas) else kappas[-1] # Weight by kappa
            x_val += b * x_points[i] * weight
            y_val += b * y_points[i] * weight
        # Theta decay adjustment
        decay = np.exp(-u / t[-1] / 20.0) if t[-1] > 0 else 1.0
        x_val *= decay
        y_val *= decay
        x_fine.append(x_val)
        y_fine.append(y_val)
   
    return np.array(x_fine), np.array(y_fine)
# Compute kappa for a segment, second endpoint influences next kappa
def compute_segment_kappa(p1, p2, base_kappa=1.0, prev_kappa=1.0):
    """
    Computes kappa for a segment with decay based on theta (distance).
  
    Args:
        p1 (tuple): Starting point (x1, y1, first endpoint, kappa node).
        p2 (tuple): Ending point (x2, y2, theta).
        base_kappa (float): Base kappa value from slider.
        prev_kappa (float): Previous kappa for decay calculation.
  
    Returns:
        float: Current kappa value (for second endpoint).
    """
    x1, y1 = p1
    x2, y2 = p2
    theta = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) # Theta is distance
    if theta < 1e-10:
        return prev_kappa
    decay_factor = np.exp(-theta / WIDTH / 20.0) # Further reduced decay rate
    return prev_kappa * decay_factor * base_kappa
# Golden window calculation
def compute_golden_window(x_spiral, y_spiral):
    idx_crossings = np.where(np.diff(np.sign(x_spiral - PURPLE_LINES[0] * WIDTH)))[0]
    if len(idx_crossings) >= 2:
        y1 = y_spiral[idx_crossings[0]]
        y2 = y_spiral[idx_crossings[1]]
        return np.abs(y2 - y1), min(y1, y2), max(y1, y2)
    return 0, 0, 0
# Compute vanishing point for a triangulation
def compute_vanishing_point(tri_points, eye_distance=EYE_DISTANCE):
    mid_x = np.mean([p[0] for p in tri_points])
    mid_y = np.mean([p[1] for p in tri_points])
    vx = mid_x
    vy = HORIZON_HEIGHT + eye_distance * (mid_y - EYE_LINE) / WIDTH
    return vx, vy
# Redraw green curve
def redraw_green_curve():
    global green_curve_line
    if green_curve_line:
        green_curve_line.remove()
        green_curve_line = None
    if len(drawing_points) >= 2:
        x_green, y_green = custom_interoperations_green_curve(drawing_points, kappas)
        green_curve_line, = ax_2d.plot(x_green, y_green, 'g-', label='Green Curve' if green_curve_line is None else None)
    fig_2d.canvas.draw()
# Setup figures
fig_2d = plt.figure(figsize=(14, 8))
ax_2d = fig_2d.add_subplot(111)
fig_3d = plt.figure(figsize=(10, 6))
ax_3d = fig_3d.add_subplot(111, projection='3d')
fig_controls = plt.figure(figsize=(4, 6))
ax_curvature = fig_controls.add_axes([0.2, 0.8, 0.6, 0.03])
curvature_slider = Slider(ax_curvature, 'Curvature (kappa)', 0.1, 2.0, valinit=curvature)
# Plot A3 page
ax_2d.plot([0, WIDTH, WIDTH, 0, 0], [0, 0, HEIGHT, HEIGHT, 0], 'k-', label='A3 Landscape Page')
for x in PURPLE_LINES:
    ax_2d.plot([x * WIDTH, x * WIDTH], [0, HEIGHT], 'm-', label='Purple Dividers' if x == PURPLE_LINES[0] else None)
# Horizon line
horizon_line, = ax_2d.plot([0, WIDTH], [HORIZON_HEIGHT, HORIZON_HEIGHT], 'b:', label='Horizon/Eye Line')
# Golden spiral
x_spiral, y_spiral = compute_golden_spiral()
golden_spiral, = ax_2d.plot(x_spiral + WIDTH/2, y_spiral + HEIGHT/2, 'gold', label='Golden Spiral')
# Golden window
golden_window, y_min, y_max = compute_golden_window(x_spiral + WIDTH/2, y_spiral + HEIGHT/2)
ax_2d.fill_between([PURPLE_LINES[0] * WIDTH - 0.05, PURPLE_LINES[0] * WIDTH + 0.05], y_min, y_max, color='yellow', alpha=0.5, label='Golden Window')
# Ghost curve init
ghost_curve, = ax_2d.plot([], [], 'g--', label='Ghost Curve Preview')
# Control indicators in legend
ax_2d.plot([], [], ' ', label='R: Toggle draw mode')
ax_2d.plot([], [], 'b--', label='A: Toggle protractor')
ax_2d.plot([], [], 'c-', label='M: Toggle measure (ruler)')
ax_2d.plot([], [], ' ', label='D: Toggle dimension')
ax_2d.plot([], [], 'r-', label='C: Close polyhedron (manual)')
ax_2d.plot([], [], ' ', label='Click near first point to close')
ax_2d.plot([], [], ' ', label='Click to select curve')
ax_2d.plot([], [], ' ', label='G: To construction geom')
ax_2d.plot([], [], ' ', label='H: Hide/show')
ax_2d.plot([], [], ' ', label='E: Reset canvas')
ax_2d.plot([], [], ' ', label='S: Export STL')
ax_2d.plot([], [], 'k-', label='Curvature Slider (Controls window)')
# Update curvature
def update_curvature(val):
    global curvature
    curvature = val
    if len(drawing_points) >= 1:
        kappas[-1] = curvature
        redraw_green_curve()
    fig_2d.canvas.draw()
curvature_slider.on_changed(update_curvature)
# Toggle draw mode
def toggle_draw(event):
    global draw_mode
    if event.key == 'r':
        draw_mode = not draw_mode
        print(f"Draw mode {'enabled' if draw_mode else 'disabled'}")
        fig_2d.canvas.draw()
# Toggle protractor
def toggle_protractor(event):
    global protractor_active
    if event.key == 'a':
        protractor_active = not protractor_active
        print(f"Protractor tool {'enabled' if protractor_active else 'disabled'}")
        if not protractor_active:
            if protractor_line:
                protractor_line.remove()
                protractor_line = None
            if protractor_text:
                protractor_text.remove()
                protractor_text = None
            protractor_points.clear()
        fig_2d.canvas.draw()
# On click for protractor
def on_click_protractor(event):
    global protractor_line, protractor_text
    if protractor_active and event.inaxes == ax_2d and event.button == 1:
        protractor_points.append((event.xdata, event.ydata))
        if len(protractor_points) == 2:
            x1, y1 = protractor_points[0]
            x2, y2 = protractor_points[1]
            if protractor_line:
                protractor_line.remove()
            protractor_line, = ax_2d.plot([x1, x2], [y1, y2], 'b--')
            dx = x2 - x1
            dy = y2 - y1
            angle = np.arctan2(dy, dx) * 180 / np.pi
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            if protractor_text:
                protractor_text.remove()
            protractor_text = ax_2d.text(mid_x, mid_y, f'Angle: {angle:.2f}°', ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
            protractor_points.clear()
            fig_2d.canvas.draw()
# Toggle ruler (measure)
def toggle_ruler(event):
    global ruler_active
    if event.key == 'm':
        ruler_active = not ruler_active
        print(f"Measure (ruler) tool {'enabled' if ruler_active else 'disabled'}")
        if not ruler_active:
            if ruler_line:
                ruler_line.remove()
                ruler_line = None
            if ruler_text:
                ruler_text.remove()
                ruler_text = None
            ruler_points.clear()
        fig_2d.canvas.draw()
# On click for ruler
def on_click_ruler(event):
    global ruler_line, ruler_text
    if ruler_active and event.inaxes == ax_2d and event.button == 1:
        ruler_points.append((event.xdata, event.ydata))
        if len(ruler_points) == 2:
            x1, y1 = ruler_points[0]
            x2, y2 = ruler_points[1]
            if ruler_line:
                ruler_line.remove()
            ruler_line, = ax_2d.plot([x1, x2], [y1, y2], 'c-')
            dx = x2 - x1
            dy = y2 - y1
            dist = np.sqrt(dx**2 + dy**2)
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            if ruler_text:
                ruler_text.remove()
            ruler_text = ax_2d.text(mid_x, mid_y, f'Dist: {dist:.4f}', ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
            ruler_points.clear()
            fig_2d.canvas.draw()
# Toggle dimension
def toggle_dimension(event):
    global dimension_active
    if event.key == 'd':
        dimension_active = not dimension_active
        print(f"Dimension tool {'enabled' if dimension_active else 'disabled'}")
        fig_2d.canvas.draw()
# On click for dimension
def on_click_dimension(event):
    if dimension_active and event.inaxes == ax_2d and event.button == 1:
        if ruler_active and ruler_text:
            dimension_labels.append(ruler_text)
        elif selected_curve:
            x, y = selected_curve.get_data()
            length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
            mid_x = np.mean(x)
            mid_y = np.mean(y)
            dim_text = ax_2d.text(mid_x, mid_y + 0.05, f'Len: {length:.4f}', ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
            dimension_labels.append(dim_text)
        fig_2d.canvas.draw()
# Drawing mode: Add kappa nodes and update continuous greencurve
def on_click_draw(event):
    global green_curve_line, selected_curve, previous_kappa, vanishing_points, current_vertices, current_faces
    if event.inaxes == ax_2d and event.button == 1:
        x, y = event.xdata, event.ydata
        if draw_mode and not (protractor_active or ruler_active or dimension_active):
            # Check if near first point to close
            if len(drawing_points) > 2:
                dx_first = x - drawing_points[0][0]
                dy_first = y - drawing_points[0][1]
                dist_first = np.sqrt(dx_first**2 + dy_first**2)
                if dist_first < CLOSE_THRESHOLD:
                    # Adjust kappa1 based on last theta and kappa
                    last_theta = np.sqrt((drawing_points[-1][0] - drawing_points[0][0])**2 + (drawing_points[-1][1] - drawing_points[0][1])**2)
                    decay_factor = np.exp(-last_theta / WIDTH / 20.0)
                    kappas[0] = kappas[-1] * decay_factor * curvature # Affect kappa1 with last kappa and decay
                    drawing_points.append(drawing_points[0])
                    kappas.append(curvature)
                    redraw_green_curve()
                    # Get closed curve
                    x_curve, y_curve = green_curve_line.get_data()
                    if np.hypot(x_curve[-1] - x_curve[0], y_curve[-1] - y_curve[0]) > 1e-5:
                        x_curve = np.append(x_curve, x_curve[0])
                        y_curve = np.append(y_curve, y_curve[0])
                    ax_3d.cla()
                    current_vertices, current_faces = build_mesh(x_curve, y_curve, num_points=50)
                    verts = [[current_vertices[i] for i in f] for f in current_faces]
                    ax_3d.add_collection3d(Poly3DCollection(verts, alpha=0.5, facecolors=cm.viridis(np.linspace(0, 1, len(verts)))))
                    ax_3d.set_xlabel('X')
                    ax_3d.set_ylabel('Y')
                    ax_3d.set_zlabel('Z')
                    ax_3d.set_title('3D User Model (Compound Curvature with End Caps)')
                    fig_3d.canvas.draw()
                    print("Polyhedron closed and 3D model generated")
                    fig_2d.canvas.draw()
                    return
            # Add new kappa node (first endpoint)
            drawing_points.append((x, y))
            ax_2d.scatter(x, y, color='r', s=50, label='Kappa Node' if len(drawing_points) == 1 else None)
            kappas.append(curvature)
            if len(drawing_points) > 1:
                previous_kappa = compute_segment_kappa(drawing_points[-2], drawing_points[-1], curvature, previous_kappa)
                redraw_green_curve()
                if len(drawing_points) >= 2:
                    t = np.linspace(0, 1, 100)
                    x_green, y_green = green_curve_line.get_data()
                    curv = compute_curvature(x_green, y_green, t)
                    print(f"Green curve curvature: Max={curv.max():.4f}, Min={curv.min():.4f}")
            if len(drawing_points) >= 3:
                print("Third point added: Introducing depth and triangulation")
                tri_points = drawing_points[-3:]
                vp = compute_vanishing_point(tri_points)
                vanishing_points.append(vp)
                ax_2d.scatter(vp[0], vp[1], color='purple', s=30, label='Vanishing Point' if len(vanishing_points) == 1 else None)
            fig_2d.canvas.draw()
        elif not draw_mode and not (protractor_active or ruler_active or dimension_active):
            min_dist = float('inf')
            selected_curve = None
            if green_curve_line:
                x_curve, y_curve = green_curve_line.get_data()
                dist = np.min(np.sqrt((x_curve - x)**2 + (y_curve - y)**2))
                if dist < min_dist and dist < 0.05:
                    min_dist = dist
                    selected_curve = green_curve_line
            if selected_curve:
                selected_curve.set_linewidth(3.0)
                print("Green curve selected")
                fig_2d.canvas.draw()
# Ghost curve preview on motion (cursor at theta)
def on_motion(event):
    global previous_kappa
    if draw_mode and len(drawing_points) > 0 and event.inaxes == ax_2d and not (protractor_active or ruler_active or dimension_active):
        x, y = event.xdata, event.ydata
        preview_points = drawing_points + [(x, y)]
        preview_kappas = kappas + [curvature]
        if len(preview_points) > 2:
            dx_first = x - drawing_points[0][0]
            dy_first = y - drawing_points[0][1]
            dist_first = np.sqrt(dx_first**2 + dy_first**2)
            if dist_first < CLOSE_THRESHOLD:
                preview_points[-1] = drawing_points[0]
                preview_kappas[-1] = curvature
                # Preview kappa1 adjustment for closure
                last_theta = np.sqrt((drawing_points[-1][0] - preview_points[-1][0])**2 + (drawing_points[-1][1] - preview_points[-1][1])**2)
                decay_factor = np.exp(-last_theta / WIDTH / 20.0)
                preview_kappas[0] = preview_kappas[-1] * decay_factor * curvature
        # Compute kappa for preview segment (cursor at theta)
        if len(preview_points) > 1:
            preview_kappa = compute_segment_kappa(preview_points[-2], preview_points[-1], curvature, previous_kappa)
            preview_kappas[-1] = preview_kappa
        x_ghost, y_ghost = custom_interoperations_green_curve(preview_points, preview_kappas)
        ghost_curve.set_data(x_ghost, y_ghost)
        fig_2d.canvas.draw()
# Close polyhedron (manual trigger)
def close_polyhedron(event):
    if event.key == 'c':
        print("Close via clicking near first point when ghosted")
# Change to construction geometry
def to_construction(event):
    global selected_curve
    if event.key == 'g' and selected_curve:
        selected_curve.set_linestyle('--')
        selected_curve.set_color('gray')
        print("Green curve changed to construction geometry")
        selected_curve = None
        fig_2d.canvas.draw()
# Hide/show
def hide_show(event):
    global hidden_elements, selected_curve
    if event.key == 'h':
        if selected_curve:
            if selected_curve.get_visible():
                selected_curve.set_visible(False)
                hidden_elements.append(selected_curve)
                print("Green curve hidden")
                selected_curve = None
            else:
                selected_curve.set_visible(True)
                if selected_curve in hidden_elements:
                    hidden_elements.remove(selected_curve)
                print("Green curve shown")
                selected_curve = None
        else:
            for elem in hidden_elements:
                elem.set_visible(True)
            hidden_elements.clear()
            print("All hidden elements shown")
        fig_2d.canvas.draw()
# Reset canvas
def reset_canvas(event):
    global drawing_points, kappas, previous_kappa, green_curve_line, vanishing_points, selected_curve, current_vertices, current_faces
    if event.key == 'e':
        drawing_points = []
        kappas = []
        previous_kappa = 1.0
        if green_curve_line:
            green_curve_line.remove()
            green_curve_line = None
        vanishing_points = []
        selected_curve = None
        ax_3d.cla()
        current_vertices = None
        current_faces = None
        display_ipod_surface()  # Reset to default
        print("Canvas reset")
        fig_2d.canvas.draw()
# Compute curvature for continuity check
def compute_curvature(x, y, t):
    dt = t[1] - t[0]
    dx_dt = np.gradient(x, dt)
    dy_dt = np.gradient(y, dt)
    d2x_dt2 = np.gradient(dx_dt, dt)
    d2y_dt2 = np.gradient(dy_dt, dt)
    numerator = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2)
    denominator = (dx_dt**2 + dy_dt**2)**1.5
    denominator = np.where(denominator == 0, 1e-10, denominator)
    return numerator / denominator
# Generate base iPod curve (closed for boundary surface)
def generate_ipod_curve_closed(num_points=100):
    t = np.linspace(0, 2 * np.pi, num_points) # Full closed loop
    r_x = 1.0 # Width
    r_y = 1.5 # Height (taller than wide)
    x = r_x * np.cos(t)
    y = r_y * np.sin(t)
    # Add rounding for iPod-like bezel (simple modulation)
    x += 0.1 * np.sin(4 * t)
    y += 0.1 * np.cos(4 * t)
    return x, y
# Build mesh for 3D model
def build_mesh(x_curve, y_curve, height=0.5, num_layers=20, num_points=None):
    if num_points is not None:
        indices = np.linspace(0, len(x_curve) - 1, num_points, dtype=int)
        x_curve = x_curve[indices]
        y_curve = y_curve[indices]
    vertices = []
    faces = []
    n = len(x_curve)
    # Loft layers
    for k in range(num_layers):
        z = height * k / (num_layers - 1)
        for i in range(n):
            vertices.append([x_curve[i], y_curve[i], z])
    # Side faces
    for k in range(num_layers - 1):
        for i in range(n):
            next_i = (i + 1) % n
            base = k * n
            next_base = (k + 1) * n
            faces.append([base + i, base + next_i, next_base + next_i])
            faces.append([base + i, next_base + next_i, next_base + i])
    # Bottom cap (fan triangulation from center)
    center_bottom = len(vertices)
    center_x = np.mean(x_curve)
    center_y = np.mean(y_curve)
    vertices.append([center_x, center_y, 0])
    for i in range(n):
        next_i = (i + 1) % n
        faces.append([center_bottom, i, next_i])
    # Top cap (fan triangulation from center)
    center_top = len(vertices)
    vertices.append([center_x, center_y, height])
    top_base = (num_layers - 1) * n
    for i in range(n):
        next_i = (i + 1) % n
        faces.append([center_top, top_base + next_i, top_base + i])  # Reversed for outward normal
    # Convert to numpy
    vertices = np.array(vertices)
    # Add compound curvature modulation
    vertices[:, 2] += 0.1 * np.sin(4 * np.pi * vertices[:, 0]) * np.cos(2 * np.pi * vertices[:, 1])
    vertices[:, 0] += 0.05 * np.sin(2 * np.pi * vertices[:, 2])
    vertices[:, 1] += 0.05 * np.cos(2 * np.pi * vertices[:, 2])
    return vertices, faces
# Function to compute normals
def compute_normal(v1, v2, v3):
    vec1 = v2 - v1
    vec2 = v3 - v1
    normal = np.cross(vec1, vec2)
    norm = np.linalg.norm(normal)
    return normal / norm if norm != 0 else normal
# Export current model to STL
def export_stl():
    global current_vertices, current_faces
    if current_vertices is None or current_faces is None:
        print("No model to export")
        return
    stl_data = b'\x00' * 80  # Header
    stl_data += struct.pack('<I', len(current_faces))  # Number of triangles
    for face in current_faces:
        v1 = current_vertices[face[0]]
        v2 = current_vertices[face[1]]
        v3 = current_vertices[face[2]]
        normal = compute_normal(v1, v2, v3)
        stl_data += struct.pack('<3f', *normal)
        stl_data += struct.pack('<3f', *v1)
        stl_data += struct.pack('<3f', *v2)
        stl_data += struct.pack('<3f', *v3)
        stl_data += b'\x00\x00'  # Attribute byte count
    filename = 'model.stl'
    with open(filename, 'wb') as f:
        f.write(stl_data)
    print(f"Saved to {filename}")
    stl_base64 = base64.b64encode(stl_data).decode('utf-8')
    print("Base64 STL:")
    print(stl_base64)
# Save STL on key press
def save_stl(event):
    if event.key == 's':
        export_stl()
# Display iPod surface by default in 3D with curvature continuous end caps
def display_ipod_surface():
    global current_vertices, current_faces
    x_curve, y_curve = generate_ipod_curve_closed(50)
    current_vertices, current_faces = build_mesh(x_curve, y_curve)
    verts = [[current_vertices[i] for i in f] for f in current_faces]
    ax_3d.add_collection3d(Poly3DCollection(verts, alpha=0.5, facecolors=cm.viridis(np.linspace(0, 1, len(verts)))))
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_title('3D iPod Projected Surface (Compound Curvature with End Caps)')
    fig_3d.canvas.draw()
# Draw default iPod ellipse as green curve on 2D canvas
def draw_default_ipod(ax, color='g-'):
    num_control = 8
    # Generate closed curve with one extra point
    x, y = generate_ipod_curve_closed(num_points=num_control + 1)
    # Slice to remove the last point, making it open
    x_control = x[:-1]
    y_control = y[:-1]
    scale = 0.2  # Scale to fit page
    x_control *= scale
    y_control *= scale
    x_control += WIDTH / 2
    y_control += HEIGHT / 2
    points = list(zip(x_control, y_control))
    kappas_ipod = [1.0] * len(points)
    x_interp, y_interp = custom_interoperations_green_curve(points, kappas_ipod)
    ax.plot(x_interp, y_interp, color, linewidth=3)
# Connect events
fig_2d.canvas.mpl_connect('key_press_event', toggle_draw)
fig_2d.canvas.mpl_connect('key_press_event', toggle_protractor)
fig_2d.canvas.mpl_connect('key_press_event', toggle_ruler)
fig_2d.canvas.mpl_connect('key_press_event', toggle_dimension)
fig_2d.canvas.mpl_connect('key_press_event', to_construction)
fig_2d.canvas.mpl_connect('key_press_event', hide_show)
fig_2d.canvas.mpl_connect('key_press_event', reset_canvas)
fig_2d.canvas.mpl_connect('key_press_event', save_stl)
fig_2d.canvas.mpl_connect('button_press_event', on_click_protractor)
fig_2d.canvas.mpl_connect('button_press_event', on_click_ruler)
fig_2d.canvas.mpl_connect('button_press_event', on_click_dimension)
fig_2d.canvas.mpl_connect('button_press_event', on_click_draw)
fig_2d.canvas.mpl_connect('motion_notify_event', on_motion)
fig_2d.canvas.mpl_connect('key_press_event', close_polyhedron)
# Plot properties
ax_2d.set_xlim(0, WIDTH)
ax_2d.set_ylim(0, HEIGHT)
ax_2d.set_aspect('equal')
ax_2d.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize='small')
ax_2d.grid(True)
ax_2d.set_title('2D Drawing Tool on A3 Landscape with Continuous Green Curve')
display_ipod_surface() # Show iPod surface on load
draw_default_ipod(ax_2d) # Draw default iPod ellipse on load with G5 interoperations
plt.show()
