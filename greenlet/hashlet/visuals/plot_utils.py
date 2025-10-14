import numpy as np
from matplotlib.colors import LightSource
from src.config import *

def compute_green_segment(t, scale=1.0):
    """Compute a green segment for visualization with scaled curvature."""
    try:
        y_base = 0.1 * scale
        nodes = [(1/3, 0), (1/3 + 1/9, y_base), (1/3 + 2/9, y_base), (2/3, 0)]
        x, y = [], []
        total_points = len(t)
        points_per_segment = total_points // 3
        extra_points = total_points % 3
        for i in range(3):
            x1, y1 = nodes[i]
            x2, y2 = nodes[i + 1]
            chord = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            x_m, y_m = (x1 + x2) / 2, (y1 + y2) / 2
            if abs(y2 - y1) < 1e-10:
                x_c, y_c = x_m, y_m + np.sqrt((chord/2)**2 + 1e-12)
            else:
                slope = -(x2 - x1) / (y2 - y1)
                a = 1 + slope**2
                b = -2 * x_m + 2 * slope * (y_m - slope * x_m)
                c = x_m**2 + (y_m - slope * x_m)**2 - (chord/2)**2
                disc = b**2 - 4 * a * c
                if disc < 0: disc = 0
                x_c = (-b + np.sqrt(disc)) / (2 * a)
                y_c = y_m + slope * (x_c - x_m)
            R = np.sqrt((x1 - x_c)**2 + (y1 - y_c)**2 + 1e-12)
            theta1 = np.arctan2(y1 - y_c, x1 - x_c)
            theta2 = np.arctan2(y2 - y_c, x2 - x_c)
            if theta2 < theta1: theta2 += 2 * np.pi
            points = points_per_segment + (1 if i < extra_points else 0)
            theta = np.linspace(theta1, theta2, points)
            x.extend(x_c + R * np.cos(theta))
            y.extend(y_c + R * np.sin(theta))
        x, y = np.array(x[:total_points]), np.array(y[:total_points])  # Trim to exact length
        return x, y, nodes
    except Exception as e:
        logger.error(f"Compute green segment error: {e}")
        return np.zeros_like(t), np.zeros_like(t), []

def compute_curvature(x, y, t):
    """Compute curvature for a given curve defined by x, y, and parameter t."""
    try:
        if len(x) != len(y) or len(x) != len(t):
            logger.error(f"Dimension mismatch: len(x)={len(x)}, len(y)={len(y)}, len(t)={len(t)}")
            return np.array([0.02500125] * len(t)), np.zeros_like(t), np.zeros_like(t)
        dx_dt = np.gradient(x, np.diff(t).mean())
        dy_dt = np.gradient(y, np.diff(t).mean())
        d2x_dt2 = np.gradient(dx_dt, np.diff(t).mean())
        d2y_dt2 = np.gradient(dy_dt, np.diff(t).mean())
        numerator = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2)
        denominator = (dx_dt**2 + dy_dt**2)**1.5 + 1e-12
        kappa = numerator / denominator
        return kappa, dx_dt, dy_dt
    except Exception as e:
        logger.error(f"Compute curvature error: {e}")
        return np.array([0.02500125] * len(t)), np.zeros_like(t), np.zeros_like(t)

def create_blob_surface(x, y, z, radius_base, num_sides, kappa, curve_mode='k_curves'):
    """Create a 3D blob surface for visualization with Boas rendering."""
    try:
        if len(x) != len(y) or len(y) != len(z) or len(z) != len(kappa):
            logger.error(f"Input length mismatch in create_blob_surface: len(x)={len(x)}, len(y)={len(y)}, len(z)={len(z)}, len(kappa)={len(kappa)}")
            return np.zeros((num_sides, len(x))), np.zeros((num_sides, len(x))), np.zeros((num_sides, len(x))), \
                   np.zeros((num_sides, len(x))), np.zeros((num_sides, len(x))), np.zeros((num_sides, len(x)))
        t = np.linspace(0, 1, len(x))
        theta = np.linspace(0, 2 * np.pi, num_sides)
        T, Theta = np.meshgrid(t, theta)
        dx_dt = np.gradient(x, t)
        dy_dt = np.gradient(y, t)
        dz_dt = np.gradient(z, t)
        tangent = np.array([dx_dt, dy_dt, dz_dt]).T
        norm = np.linalg.norm(tangent, axis=1)[:, np.newaxis] + 1e-12
        tangent /= norm
        arbitrary = np.zeros((len(t), 3))
        condition = np.abs(tangent[:, 2]) < 0.9
        arbitrary[condition] = [0, 0, 1]
        arbitrary[~condition] = [1, 0, 0]
        perp1 = np.cross(tangent, arbitrary)
        perp1 /= np.linalg.norm(perp1, axis=1)[:, np.newaxis] + 1e-12
        perp2 = np.cross(tangent, perp1)
        blue_factor = 4  # Slight, 4-stride
        gold_factor = 6  # Relief, 6-stride
        radius = radius_base * (1 + 0.5 * np.sin(2 * np.pi * t) + 0.3 * np.cos(4 * np.pi * t))
        if curve_mode == 'k_curves':
            radius *= (1 + gaussian_filter1d(kappa, sigma=2, mode='wrap'))
        elif curve_mode == 'arcs':
            radius *= (1 + gaussian_filter1d(kappa, sigma=1, mode='wrap'))
        elif curve_mode == 'kappa_vectors':
            radius *= (1 + np.tan(np.cumsum(kappa) / len(kappa) * 2 * np.pi))
        radial_x_blue = radius[np.newaxis, :] * (np.cos(Theta) * perp1[:, 0][np.newaxis, :] + np.sin(Theta) * perp2[:, 0][np.newaxis, :]) + x[np.newaxis, :]
        radial_y_blue = radius[np.newaxis, :] * (np.cos(Theta) * perp1[:, 1][np.newaxis, :] + np.sin(Theta) * perp2[:, 1][np.newaxis, :]) + y[np.newaxis, :]
        radial_z_blue = radius[np.newaxis, :] * (np.cos(Theta) * perp1[:, 2][np.newaxis, :] + np.sin(Theta) * perp2[:, 2][np.newaxis, :]) + z[np.newaxis, :]
        radial_x_gold = radius[np.newaxis, :] * (np.cos(Theta) * perp1[:, 0][np.newaxis, :] + np.sin(Theta) * perp2[:, 0][np.newaxis, :]) + x[np.newaxis, :]
        radial_y_gold = radius[np.newaxis, :] * (np.cos(Theta) * perp1[:, 1][np.newaxis, :] + np.sin(Theta) * perp2[:, 1][np.newaxis, :]) + y[np.newaxis, :]
        radial_z_gold = radius[np.newaxis, :] * (np.cos(Theta) * perp1[:, 2][np.newaxis, :] + np.sin(Theta) * perp2[:, 2][np.newaxis, :]) + z[np.newaxis, :]
    except Exception as e:
        logger.error(f"Create blob surface error: {e}")
        return np.zeros((num_sides, len(x))), np.zeros((num_sides, len(x))), np.zeros((num_sides, len(x))), \
               np.zeros((num_sides, len(x))), np.zeros((num_sides, len(x))), np.zeros((num_sides, len(x)))

def add_light_slicks(ax, center=[0,0,0], num_rays=6, length=1.5, color='yellow', alpha=0.5):
    """Add light slicks to a 3D plot for visualization."""
    try:
        for i in range(num_rays):
            angle = i * (2 * np.pi / num_rays)
            end = [center[0] + length * np.cos(angle), center[1] + length * np.sin(angle), center[2]]
            ax.plot([center[0], end[0]], [center[1], end[1]], [center[2], end[2]], color=color, alpha=alpha, lw=2)
    except Exception as e:
        logger.error(f"Add light slicks error: {e}")
