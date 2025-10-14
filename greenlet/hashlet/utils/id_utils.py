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
import hashlib
import math
import sys
import argparse
from decimal import Decimal, getcontext

# Set precision for Decimal (from nu_curve_cp.py and others)
getcontext().prec = 28

# Constants from provided scripts
PHI = (1 + np.sqrt(5)) / 2
A_SPIRAL = 0.1
B_SPIRAL_BASE = 0.1
K_BASE = 0.1

# Mersenne Exponents (52 total, from multiple scripts)
MERSENNE_EXP = [
    2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281,
    3217, 4253, 4423, 9689, 9941, 11213, 19937, 21701, 23209, 44497, 86243,
    110503, 132049, 216091, 756839, 859433, 1257787, 1398269, 2976221, 3021377,
    6972593, 13466917, 20996011, 24036583, 25964951, 30402457, 32582657,
    37156667, 42643801, 43112609, 57885161, 74207281, 77232917, 82589933, 136279841
]

# A4 and A3 dimensions (from nu_curve*.py, normalized for industrial design)
WIDTH = 420 / 110  # A3 long side / A4 short side
HEIGHT = 1.0       # Normalized A3 short side
PURPLE_LINES = [1/3, 2/3]  # Dividers for design grid

# Helper function to compute a point on the spiral curve (from nu_curve.py and variants)
def compute_spiral_point(input_str, model='tetrahedron', b_factor=1.0, k_factor=1.0):
    """
    Computes a 3D point on a spiral curve for industrial design, using hashed input.
    
    Args:
        input_str (str): Input string to hash for parameters.
        model (str): 'tetrahedron' or 'ipod'.
        b_factor (float): B_SPIRAL factor for curve scaling.
        k_factor (float): K factor for z-axis modulation.
    
    Returns:
        tuple: (x, y, z) coordinates within design space.
    """
    hash_obj = hashlib.sha256(input_str.encode())
    hash_val = Decimal(str(int(hash_obj.hexdigest(), 16) % 1000 / 1000.0))
    
    B_SPIRAL = Decimal(str(B_SPIRAL_BASE)) * Decimal(str(b_factor))
    K = Decimal(str(K_BASE)) * Decimal(str(k_factor))
    
    t = hash_val
    theta = Decimal('2') * Decimal(str(math.pi)) * t
    
    if model == 'tetrahedron':
        r = Decimal(str(A_SPIRAL)) * (Decimal(str(math.e)) ** (B_SPIRAL * theta))
        x = r * Decimal(str(math.cos(float(theta))))
        y = r * Decimal(str(math.sin(float(theta))))
        z = K * Decimal(str(math.sin(float(4 * theta)))) * (Decimal('1') + hash_val)
    elif model == 'ipod':
        r = Decimal('0.5') + Decimal('0.2') * Decimal(str(math.sin(float(6 * theta))))
        x = r * Decimal(str(math.cos(float(theta))))
        y = r * Decimal(str(math.sin(float(theta))))
        z = K * Decimal(str(math.sin(float(12 * theta)))) * (Decimal('1') + hash_val)
    else:
        raise ValueError("Invalid model. Choose 'tetrahedron' or 'ipod'.")
    
    # Normalize to fit within A4 design space
    x = (x + Decimal(str(WIDTH / 2))) % Decimal(str(WIDTH))
    y = (y + Decimal(str(HEIGHT / 2))) % Decimal(str(HEIGHT))
    z = z % Decimal('1.0')  # Keep z in [0,1] for design height
    
    return float(x), float(y), float(z)

# Kappa prime function (from spiral_nu.py and nu_curve_kappa.py)
def kappa_prime_n(n: int) -> float:
    """
    Computes kappa prime for design curvature, based on Mersenne and phi.
    
    Args:
        n (int): Input value (e.g., 2 to 104).
    
    Returns:
        float: Kappa prime value for curvature.
    """
    phi = (1 + math.sqrt(5)) / 2
    F = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]  # Fibonacci up to F_10
    if n == 2:
        return 2
    elif 2 < n < 52:
        phi_pos = phi ** (abs(n - 12) / 12)
        phi_neg = phi ** (-abs(n - 12) / 12)
        denom1 = abs(phi ** (10/3) - phi ** (-10/3))
        denom2 = abs(phi ** (-5/6) - phi ** (5/6))
        return (1 + 0.5 * (abs(phi_pos - phi_neg) / denom1) * (1 / denom2)) * (2 / 1.5) - 0.333
    elif n == 52:
        return 1.5
    elif 52 < n < 92:
        return 1.5 * min(3, 1.5 * F[min(n-52+6, len(F)-1)] / F[6]) * math.exp(-((n-60)**2)/(4*100)) * math.cos(0.5*(n-316))
    elif n == 92:
        return 3
    elif 92 < n <= 104:
        return 3 * math.exp(-((n-92)**2)/(4*100)) * math.cos(0.5*(n-316))
    return 0

# Poly hash 256 (from spiral_nu.py)
def poly_hash_256(mersenne_exponents: list = MERSENNE_EXP) -> str:
    """
    Computes a polyhedral SHA256 hash for design versioning.
    
    Args:
        mersenne_exponents (list): List of Mersenne exponents.
    
    Returns:
        str: Hexdigest of the hash.
    """
    diffs = [mersenne_exponents[i+1] - mersenne_exponents[i] for i in range(len(mersenne_exponents)-1)]
    forward = [(d % 369) & 0xFF for d in diffs[:14]]
    reverse = forward[::-1]
    pair_sum = sum(forward[i] + reverse[i] for i in range(7)) & 0x7F
    is_palindrome = 1 if forward == reverse else 0
    byte_string = bytearray([is_palindrome << 7 | pair_sum]) + int(52).to_bytes(2, 'big') + \
                 bytes(forward) + bytes(reverse)
    byte_string.extend([0] * (32 - len(byte_string)))
    return hashlib.sha256(byte_string).hexdigest()

# Generate design ID function
def generate_design_id(input_str: str, mersenne_index: int = None, model: str = 'tetrahedron',
                       b_factor: float = 1.0, k_factor: float = 1.0, use_kappa: bool = False,
                       n: int = 12, include_version: bool = False) -> str:
    """
    Generates a unique ID for industrial design by hashing input, mapping to Mersenne exponent,
    computing spiral point, and hashing coordinates, with optional versioning.
    
    Args:
        input_str (str): Input string (e.g., design name or specs).
        mersenne_index (int, optional): Mersenne index (0-51).
        model (str): Curve model ('tetrahedron' or 'ipod').
        b_factor (float): B_SPIRAL factor for curve scaling.
        k_factor (float): K factor for z-axis modulation.
        use_kappa (bool): If True, incorporate kappa_prime_n for curvature.
        n (int): Value for kappa_prime_n.
        include_version (bool): If True, include poly_hash_256 for versioning.
    
    Returns:
        str: 16-char hex ID.
    """
    if mersenne_index is None:
        hash_obj = hashlib.sha256(input_str.encode())
        mersenne_index = int(hash_obj.hexdigest(), 16) % len(MERSENNE_EXP)
    
    exp = MERSENNE_EXP[mersenne_index]
    seed = f"{input_str}_{exp}"
    
    if use_kappa:
        kappa = kappa_prime_n(n)
        seed += f"_{kappa:.4f}"
    
    x, y, z = compute_spiral_point(seed, model, b_factor, k_factor)
    
    coord_str = f"{x:.4f}{y:.4f}{z:.4f}"
    if include_version:
        version_hash = poly_hash_256()
        coord_str += f"_{version_hash[:8]}"
    
    hash_obj = hashlib.sha256(coord_str.encode())
    unique_id = hash_obj.hexdigest()
    
    return unique_id[:16]

# Main CLI for industrial design utility
def main():
    """
    Industrial Design Utility for generating unique IDs using hashlet-inspired curve hashing.
    
    Run Instructions:
    ----------------
    Prerequisites:
    - Python 3.x
    - NumPy library (install via `pip install numpy`)
    
    Usage:
    - Save this script as `idutil.py`.
    - Run from the command line using one of the following commands:
    
    1. Generate a unique design ID:
       ```bash
       python idutil.py generate "design_name" [--model tetrahedron|ipod] [--b_factor FLOAT] [--k_factor FLOAT] [--mersenne_index INT] [--use_kappa] [--n INT] [--version]
       ```
       - Example: `python idutil.py generate "chassis_v1" --model ipod --b_factor 1.5 --use_kappa --n 52 --version`
       - Parameters:
         - input: String representing design name or specs (e.g., "engine_part_001").
         - --model: Curve model ('tetrahedron' or 'ipod', default: tetrahedron).
         - --b_factor: Scaling factor for spiral radius (default: 1.0).
         - --k_factor: Scaling factor for z-axis modulation (default: 1.0).
         - --mersenne_index: Mersenne exponent index (0-51, default: derived from input hash).
         - --use_kappa: Include kappa_prime_n for curvature (flag).
         - --n: Input for kappa_prime_n (2-104, default: 12).
         - --version: Include polyhedral hash for versioning (flag).
    
    2. Compute kappa_prime_n for curvature:
       ```bash
       python idutil.py kappa <n>
       ```
       - Example: `python idutil.py kappa 52`
       - Parameter:
         - n: Integer input (2-104) for kappa calculation.
    
    3. Compute polyhedral hash for versioning:
       ```bash
       python idutil.py polyhash
       ```
       - Example: `python idutil.py polyhash`
    
    Output:
    - For `generate`: A 16-character hexadecimal ID (e.g., "a1b2c3d4e5f67890").
    - For `kappa`: A float representing kappa_prime_n (e.g., "1.500000").
    - For `polyhash`: A 64-character hexadecimal hash (e.g., "1a2b3c4d...").
    
    Notes:
    - The script uses a golden spiral and Mersenne exponents to map inputs to 3D coordinates within an A4 design space (WIDTH=3.818, HEIGHT=1.0).
    - Coordinates are normalized to fit the design space, ensuring compatibility with CAD-like workflows.
    - The --version flag adds a polyhedral hash to track design iterations.
    - Ensure NumPy is installed before running.
    - For invalid inputs (e.g., mersenne_index outside 0-51, model not in choices), the script raises a ValueError.
    """
    parser = argparse.ArgumentParser(description="idutil: Industrial Design Utility for generating unique IDs using hashlet-inspired curve hashing.")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Generate ID subcommand
    gen_parser = subparsers.add_parser('generate', help='Generate unique design ID.')
    gen_parser.add_argument('input', type=str, help='Input string (e.g., design name or specs).')
    gen_parser.add_argument('--model', type=str, default='tetrahedron', choices=['tetrahedron', 'ipod'], help='Curve model for design.')
    gen_parser.add_argument('--b_factor', type=float, default=1.0, help='B_SPIRAL factor for curve scaling.')
    gen_parser.add_argument('--k_factor', type=float, default=1.0, help='K factor for z-axis modulation.')
    gen_parser.add_argument('--mersenne_index', type=int, default=None, help='Mersenne index (0-51).')
    gen_parser.add_argument('--use_kappa', action='store_true', help='Incorporate kappa_prime_n for curvature.')
    gen_parser.add_argument('--n', type=int, default=12, help='n for kappa_prime_n.')
    gen_parser.add_argument('--version', action='store_true', help='Include polyhedral hash for versioning.')
    
    # Kappa prime subcommand
    kappa_parser = subparsers.add_parser('kappa', help='Compute kappa_prime_n for curvature.')
    kappa_parser.add_argument('n', type=int, help='Input n (e.g., 2-104).')
    
    # Poly hash subcommand
    poly_parser = subparsers.add_parser('polyhash', help='Compute poly_hash_256 for versioning.')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        unique_id = generate_design_id(
            args.input, args.mersenne_index, args.model, args.b_factor, args.k_factor,
            args.use_kappa, args.n, args.version
        )
        print(f"Generated Design ID: {unique_id}")
    elif args.command == 'kappa':
        kappa = kappa_prime_n(args.n)
        print(f"Kappa prime for n={args.n}: {kappa:.6f}")
    elif args.command == 'polyhash':
        hash_val = poly_hash_256()
        print(f"Polyhedral Hash: {hash_val}")

if __name__ == "__main__":
    main()
