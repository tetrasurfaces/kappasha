import math
import numpy as np
import hashlib
from src.config import *
from typing import Dict

def get_grid_label(index: int) -> str:
    """Generate a grid label from mock words or symbols."""
    return mock_words[index % len(mock_words)] if index < len(mock_words) else alphanum_symbols[index % len(alphanum_symbols)]

def get_kappa_coordinates(radius: int, curvature: float, height: int) -> Dict:
    """Calculate kappa-based coordinates for grid positioning."""
    try:
        n = int(hashlib.sha256(str(radius).encode()).hexdigest(), 16) % 100
        abs_n = abs(n - 12) / 12
        num = PHI_FLOAT ** abs_n - PHI_FLOAT ** (-abs_n)
        denom = abs(PHI_FLOAT ** (10/3) - PHI_FLOAT ** (-10/3)) * abs(PHI_FLOAT ** (-5/6) - PHI_FLOAT ** (5/6))
        spiral_index_float = (1 + KAPPA_BASE * num / denom) * (2 / 1.5) - 0.333
        spiral_index = math.floor(spiral_index_float * GRID_DIM) % GRID_DIM
        x = math.floor((radius * math.cos(curvature) + spiral_index) % GRID_DIM)
        y = math.floor(height % GRID_DIM)
        z = radius % BUFFER_BLOCK_LIMIT
        return {'x': x, 'y': y, 'z': z}
    except Exception as e:
        logger.error(f"Kappa coordinates error: {e}")
        return {'x': 0, 'y': 0, 'z': 0}

def reverse_tuple_parse_grid(x: int, y: int) -> Dict:
    """Reverse parse grid coordinates."""
    try:
        return {'x': (GRID_DIM - x) % GRID_DIM, 'y': (GRID_DIM - y) % GRID_DIM}
    except Exception as e:
        logger.error(f"Reverse tuple parse error: {e}")
        return {'x': x, 'y': y}

def get_dynamic_kappa_base(block_height: int) -> float:
    """Calculate dynamic kappa base based on block height."""
    prime_index = block_height % 52
    fluctuation = 0.0027 * (prime_index / 51)
    return KAPPA_BASE + fluctuation

def kappa_calc(n: int, block_height: int = 0) -> float:
    """Calculate kappa value for a given n and block height."""
    try:
        kappa_base = get_dynamic_kappa_base(block_height)
        abs_n = abs(n - 12) / 12
        num = PHI_FLOAT ** abs_n - PHI_FLOAT ** (-abs_n)
        denom = abs(PHI_FLOAT ** (10/3) - PHI_FLOAT ** (-10/3)) * abs(PHI_FLOAT ** (-5/6) - PHI_FLOAT ** (5/6))
        result = (1 + kappa_base * num / denom) * (2 / 1.5) - 0.333 if 2 < n < 52 else max(0, 1.5 * math.exp(-((n - 60) ** 2) / 400) * math.cos(0.5 * (n - 316)))
        return result
    except Exception as e:
        logger.error(f"Kappa calc error: {e}")
        return 0.0

def compute_fibonacci_spiral_segment(chord_length, l_intersect, h_intersect, theta_start, theta_end, num_points=1000):
    """Compute Fibonacci spiral segment for curve mapping."""
    phi = (1 + np.sqrt(5)) / 2
    b = np.log(phi) / (np.pi / 2)
    a = 1.0
    theta = np.linspace(theta_start, theta_end, num_points)
    r = a * np.exp(b * theta)
    l = r * np.cos(theta)
    h = r * np.sin(theta)
    target_chord = chord_length
    best_i, best_j = None, None
    min_error = float('inf')
    for i in range(len(theta)):
        for j in range(i + 1, len(theta)):
            chord = np.sqrt((l[j] - l[i])**2 + (h[j] - h[i])**2)
            error = abs(chord - target_chord)
            if error < min_error:
                min_error = error
                best_i, best_j = i, j
    best_theta1, best_theta2 = theta[best_i], theta[best_j]
    if best_theta1 > best_theta2:
        best_theta1, best_theta2 = best_theta2, best_theta1
    theta_segment = np.linspace(best_theta1, best_theta2, 200)
    r_segment = a * np.exp(b * theta_segment)
    l_segment = r_segment * np.cos(theta_segment)
    h_segment = r_segment * np.sin(theta_segment)
    l_shifted = l_segment - l_segment[0]
    h_shifted = h_segment - h_segment[0]
    end_point = np.array([l_shifted[-1], h_shifted[-1]])
    angle = np.arctan2(end_point[1], end_point[0])
    rotation_matrix = np.array([[np.cos(-angle), -np.sin(-angle)],
                               [np.sin(-angle), np.cos(-angle)]])
    points = np.vstack((l_shifted, h_shifted))
    rotated_points = rotation_matrix @ points
    l_rotated = rotated_points[0, :]
    h_rotated = rotated_points[1, :]
    l_scaled = l_rotated * (chord_length / l_rotated[-1])
    h_scaled = h_rotated * (chord_length / l_rotated[-1])
    h_flipped = -h_scaled
    l_flipped = chord_length - l_scaled
    l_final = chord_length - l_flipped
    idx_intersect = np.argmin(np.abs(l_final - l_intersect))
    l_at_intersect = l_final[idx_intersect]
    h_max = np.max(np.abs(h_flipped))
    h_normalized = h_flipped / h_max if h_max != 0 else h_flipped
    h_intersect_normalized = h_normalized[idx_intersect]
    h_scale = h_intersect / h_intersect_normalized if h_intersect_normalized != 0 else 1.0
    h_final = h_normalized * h_scale
    h_final = h_final - min(h_final)
    dl = np.diff(l_final)
    d2l = np.diff(dl)
    dh = np.diff(h_final)
    d2h = np.diff(dh)
    kappa = np.abs(dl[:-1] * d2h - dh[:-1] * d2l) / (dl[:-1]**2 + dh[:-1]**2)**1.5
    return l_final, h_final, theta_segment, (l_final[0], h_final[0]), l_at_intersect, kappa

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
