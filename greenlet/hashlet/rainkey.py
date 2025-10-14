# rainkey.py
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Anonymous
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
#
# Notes: Integrates wise_transforms.py for mapping hex color grid based on spiral's hop path vectors and spectrum.
# Applies BitWise for binary pruning, HexWise for reversible rotations, HashWise for immutable entropy.
# Generates a color grid visualization where each hop vector influences RGB values via transformations.
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import math
import argparse
import hashlib
import mpmath

# Set precision for mpmath
mpmath.mp.dps = 19

# wise_transforms functions
def bitwise_transform(data, bits=16):
    """BitWise: Raw binary ops (mask, NOT flip) for -1 pruning."""
    int_data = int.from_bytes(data.encode(), 'big') % (1 << bits)
    mask = (1 << bits) - 1
    mirrored = (~int_data) & mask  # Bitwise mirror
    return bin(mirrored)[2:].zfill(bits)  # Binary string

def hexwise_transform(data, angle=137.5):
    """HexWise: String/hex rotations/mirrors for 0 privacy (reversible)."""
    hex_data = data.encode().hex()
    mirrored = hex_data + hex_data[::-1]  # Palindromic mirror
    shift = int(angle % len(mirrored))
    rotated = mirrored[shift:] + mirrored[:shift]  # Rotate
    return rotated

def hashwise_transform(data):
    """HashWise: SHA1664 sponge perms for +1 culture (immutable entropy)."""
    base_hash = hashlib.sha512(data.encode()).digest()
    mp_state = mpmath.mpf(int(base_hash.hex(), 16))
    for _ in range(4):  # Perms (sqrt * PHI)
        mp_state = mpmath.sqrt(mp_state) * mpmath.phi
    partial = mpmath.nstr(mp_state, 1664 // 4)
    final_hash = hashlib.sha256(partial.encode()).hexdigest()
    entropy = int(mpmath.log(mp_state, 2))
    return final_hash, entropy

# Expanded QWERTY layout (4 rows, including numbers and letters)
qwerty = [
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/']
]

# Strict hex mapping (0-9, a-f only, mapped to all keys)
hex_map = {
    '0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
    'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd', 'E': 'e', 'F': 'f', 'G': 'g', 'H': 'h', 'I': 'i', 'J': 'j',
    'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'O': 'o', 'P': 'p', 'Q': 'q', 'R': 'r', 'S': 's', 'T': 't',
    'U': 'u', 'V': 'v', 'W': 'w', 'X': 'x', 'Y': 'y', 'Z': 'z', ';': 's', ',': 'c', '.': 'd', '/': 'f'
}

# Key positions (row, col)
key_pos = {key: (r, c) for r, row in enumerate(qwerty) for c, key in enumerate(row)}

def generate_spiral_sequence(start_key, time_factor, num_hops=20, kappa=1.0):
    """Generate a forward spiral sequence, unique to the moment."""
    if start_key not in key_pos:
        raise ValueError(f"Invalid start key: {start_key}")
    
    r, c = key_pos[start_key]
    sequence = [start_key]
    theta = 0.0
    visited = {start_key}
    
    for hop in range(1, num_hops):
        theta += (137.5 * math.pi / 180) / (hop * kappa) + time_factor
        distance = hop / kappa
        dx = math.cos(theta) * distance
        dy = math.sin(theta) * distance
        new_r = int((r + dy) % len(qwerty))
        new_c = int((c + dx) % len(qwerty[0]))
        new_key = qwerty[new_r][new_c]
        
        if new_key not in visited and new_key != sequence[0]:
            sequence.append(new_key)
            visited.add(new_key)
            r, c = new_r, new_c
        elif len(visited) < len(key_pos):
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            dr, dc = random.choice(directions)
            adj_r = (r + dr) % len(qwerty)
            adj_c = (c + dc) % len(qwerty[0])
            adj_key = qwerty[adj_r][adj_c]
            if adj_key not in visited and adj_key != sequence[0]:
                sequence.append(adj_key)
                visited.add(adj_key)
                r, c = adj_r, adj_c
        if len(sequence) >= num_hops:
            break
    
    # Pad to num_hops if not reached
    while len(sequence) < num_hops:
        available_keys = [k for k in key_pos if k not in visited and k != start_key]
        if available_keys:
            new_key = random.choice(available_keys)
            sequence.append(new_key)
            visited.add(new_key)
        else:
            sequence.append(random.choice(list(key_pos.keys())))
    
    return sequence

def pollard_kangaroo_on_grid(start_pos, target_pos, steps=800):
    """Enhanced Pollard's Kangaroo with increased steps and dynamic jumps including diagonals."""
    grid_rows, grid_cols = len(qwerty), len(qwerty[0])
    order_approx = grid_rows * grid_cols
    m = int(math.sqrt(order_approx)) + 15  # Increased m for better coverage
    # Jumps including diagonal moves
    jumps = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    tame_map = {}
    x_t, y_t = start_pos
    for i in range(m):
        pos = (x_t % grid_rows, y_t % grid_cols)
        if pos not in tame_map:
            tame_map[pos] = i
        j = hash(str(pos) + str(i)) % len(jumps)
        dr, dc = jumps[j]
        x_t = (x_t + dr) % grid_rows
        y_t = (y_t + dc) % grid_cols
    x_w, y_w = target_pos
    for i in range(m * 4):  # Increased search range
        pos = (x_w % grid_rows, y_w % grid_cols)
        if pos in tame_map:
            return tame_map[pos] + i
        j = hash(str(pos) + str(i)) % len(jumps)
        dr, dc = jumps[j]
        x_w = (x_w + dr) % grid_rows
        y_w = (y_w + dc) % grid_cols
    # Fallback to Manhattan distance if no path found
    return abs(start_pos[0] - target_pos[0]) + abs(start_pos[1] - target_pos[1])

def generate_spectrum_kappa(sequence):
    """Generate strict hex spectrum kappa (0-9, a-f), padded to 16 chars."""
    spectrum = ''.join(hex_map.get(k.upper(), '0') for k in sequence)
    spectrum = (spectrum + '0' * (16 - len(spectrum)))[:16]
    return f"0x{spectrum.lower()}"

def map_hex_color_grid(sequence, hop_distance, spectrum_kappa):
    """Map a hex color grid based on spiral hop path vectors using wise transformations."""
    grid_size = (len(qwerty), len(qwerty[0]))
    color_grid = np.zeros(grid_size + (3,))  # RGB grid
    
    # Apply wise transformations to spectrum_kappa
    bit_wise = bitwise_transform(spectrum_kappa)
    hex_wise = hexwise_transform(spectrum_kappa)
    hash_wise, _ = hashwise_transform(spectrum_kappa)
    
    for i in range(len(sequence) - 1):
        current = sequence[i]
        next_key = sequence[i + 1]
        current_pos = next((r, c) for r, row in enumerate(qwerty) for c, val in enumerate(row) if val == current)
        next_pos = next((r, c) for r, row in enumerate(qwerty) for c, val in enumerate(row) if val == next_key)
        
        # Hop vector (direction and magnitude)
        vector = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])
        vector_str = f"{vector[0]},{vector[1]}"
        
        # Transform vector to color
        hash_color, _ = hashwise_transform(vector_str)
        r = int(hash_color[:2], 16) % 256 / 255  # Red from first two hex digits
        g = int(hash_color[2:4], 16) % 256 / 255  # Green from next two
        b = int(hash_color[4:6], 16) % 256 / 255  # Blue from next two
        color = (r, g, b)
        
        # Apply color to grid at current position
        color_grid[current_pos[0], current_pos[1]] = color
    
    return color_grid

def visualize_color_grid(color_grid):
    """Visualize the hex color grid."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(color_grid)
    ax.set_title("Hex Color Grid based on Spiral Hop Path Vector Spectrum")
    ax.set_xticks(np.arange(-0.5, len(qwerty[0]), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(qwerty), 1), minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Rainkey: QWERTY Hexwise Random Input Generator with Spiral Rainbow and Color Grid")
    parser.add_argument("--start-key", type=str, default="Q", help="Starting key on QWERTY")
    parser.add_argument("--num-hops", type=int, default=20, help="Number of hops in sequence")
    parser.add_argument("--kappa", type=float, default=1.0, help="Kappa for flattening")
    args = parser.parse_args()

    # Time factor unique to moment (forward only, >0)
    time_factor = (time.time() % 1) + 0.01

    # Generate sequence
    sequence = generate_spiral_sequence(args.start_key.upper(), time_factor, args.num_hops, args.kappa)
    print("Generated Sequence:", sequence)

    # Kangaroo hop distance
    start_pos = key_pos[sequence[0]]
    target_pos = key_pos[sequence[-1]]
    hop_distance = pollard_kangaroo_on_grid(start_pos, target_pos)
    print(f"Kangaroo hop distance to end key: {hop_distance}")

    # Spectrum kappa
    spectrum_kappa = generate_spectrum_kappa(sequence)
    print("Spectrum Kappa:", spectrum_kappa)

    # Map and visualize color grid
    color_grid = map_hex_color_grid(sequence, hop_distance, spectrum_kappa)
    visualize_color_grid(color_grid)

if __name__ == "__main__":
    main()
