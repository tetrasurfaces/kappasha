# kappawise.py - Spiral-based grid generation using hash indexing
#
# Copyright (C) Anonymous
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Assuming hash_utils is available; replace with actual implementation if needed.
# For demonstration, we'll mock murmur32 as a simple hash function.

import hashlib

def murmur32(input_str):
    # Mock implementation of a 32-bit hash (replace with actual murmur if available)
    h = hashlib.sha256(input_str.encode()).digest()
    return int.from_bytes(h[:4], 'big')

SEED = 42  # Change if you want forkable worlds; seeded for reproducibility

def kappa_coord(user_id, theta):
    """
    Compute 10-bit coordinates (x, y, z) for a kappa grid point based on user_id and theta.
    Uses a 32-bit hash to derive three 10-bit values.
    """
    input_str = str(user_id) + str(theta) + str(SEED)
    raw = murmur32(input_str)
    x = (raw >> 0) & 1023   # 10 bits
    y = (raw >> 10) & 1023  # 10 bits
    z = (raw >> 20) & 1023  # 10 bits (last 2 bits of 32-bit hash are unused)
    return x, y, z

# Example usage (commented out)
# print(kappa_coord(12345, 3.14159))  # Outputs something like (456, 789, 123)
