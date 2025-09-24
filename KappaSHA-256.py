# KappaSHA-256.py - Full 24-Round Keccak Variant with Kappa Diffusion
# Copyright 2025 xAI
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Note: This file may depend on greenlet components licensed under MIT/PSF. See LICENSE.greenlet.

import hashlib
import math
import mpmath
mpmath.mp.dps = 19  # Precision for φ, π

PHI_FLOAT = (1 + math.sqrt(5)) / 2  # φ ≈1.618
KAPPA_BASE = 0.3536  # Odd Mersenne (m11/107)
MODULO = 369  # Cyclic diffusion
GRID_DIM = 5  # 5x5 lanes
LANE_BITS = 64  # Bits per lane
RATE = 1088  # For 512-bit capacity
CAPACITY = 512  # 256-bit security
OUTPUT_BITS = 256  # 256-bit output
ROUNDS = 24  # Full Keccak rounds

# Mersenne Fluctuation
def mersenne_fluctuation(prime_index=11):
    fluctuation = 0.0027 * (prime_index / 51.0)
    return KAPPA_BASE + fluctuation if prime_index % 2 == 1 else 0.3563 + fluctuation

# Kappa Calculation (Curvature Decay)
def kappa_calc(n, round_idx, prime_index=11):
    kappa_base = mersenne_fluctuation(prime_index)
    abs_n = abs(n - 12) / 12.0
    num = PHI_FLOAT ** abs_n - PHI_FLOAT ** (-abs_n)
    denom = abs(PHI_FLOAT ** (10/3) - PHI_FLOAT ** (-10/3)) * abs(PHI_FLOAT ** (-5/6) - PHI_FLOAT ** (5/6))
    result = (1 + kappa_base * num / denom) * (2 / 1.5) - 0.333 if 2 < n < 52 else max(0, 1.5 * math.exp(-((n - 60) ** 2) / 400.0) * math.cos(0.5 * (n - 316)))
    return result % MODULO

# Kappa Transform (Row-wise Curvature Weighting, Starts Pipeline)
def kappa_transform(state, key, round_idx, prime_index):
    for x in range(GRID_DIM):
        for y in range(GRID_DIM):
            n = x * y
            kappa_val = kappa_calc(n, round_idx, prime_index)
            shift = int(kappa_val % LANE_BITS)
            state[x][y] ^= (key[x][y] >> shift) & ((1 << LANE_BITS) - 1)
    return state

# Standard Keccak Steps (NIST FIPS 202)
def theta(state):
    C = [0] * GRID_DIM
    for x in range(GRID_DIM):
        C[x] = state[x][0] ^ state[x][1] ^ state[x][2] ^ state[x][3] ^ state[x][4]
    D = [0] * GRID_DIM
    for x in range(GRID_DIM):
        D[x] = C[(x - 1) % GRID_DIM] ^ ((C[(x + 1) % GRID_DIM] << 1) | (C[(x + 1) % GRID_DIM] >> 63))
    for x in range(GRID_DIM):
        for y in range(GRID_DIM):
            state[x][y] ^= D[x]
    return state

def rho(state):
    offsets = [[0, 36, 3, 41, 18], [1, 44, 10, 45, 2], [62, 6, 43, 15, 61], [28, 55, 25, 21, 56], [27, 20, 39, 8, 14]]
    for x in range(GRID_DIM):
        for y in range(GRID_DIM):
            state[x][y] = ((state[x][y] << offsets[x][y]) | (state[x][y] >> (LANE_BITS - offsets[x][y]))) & ((1 << LANE_BITS) - 1)
    return state

def pi(state):
    temp = [[0] * GRID_DIM for _ in range(GRID_DIM)]
    for x in range(GRID_DIM):
        for y in range(GRID_DIM):
            temp[x][y] = state[(x + 3 * y) % GRID_DIM][x]
    return temp

def chi(state):
    for x in range(GRID_DIM):
        for y in range(GRID_DIM):
            state[x][y] ^= (~state[(x + 1) % GRID_DIM][y]) & state[(x + 2) % GRID_DIM][y]
    return state

def iota(state, round_idx):
    RC = [
        0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
        0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
        0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
        0x000000008000808b, 0x8000000000000003, 0x8000000000008089, 0x8000000000008002,
        0x8000000000000080, 0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
        0x8000000000008080, 0x0000000080000001, 0x8000000080008008
    ]
    state[0][0] ^= RC[round_idx]
    return state

# Sponge Helpers
def pad_message(msg):
    rate_bytes = RATE // 8
    padded_len = ((len(msg) + rate_bytes - 1) // rate_bytes + 1) * rate_bytes
    padded = msg + b'\x06' + b'\x00' * (padded_len - len(msg) - 2) + b'\x80'
    return padded

def absorb(state, chunk):
    i = 0
    for x in range(GRID_DIM):
        for y in range(GRID_DIM):
            if i < len(chunk):
                state[x][y] ^= int.from_bytes(chunk[i:i+8], 'little')
                i += 8
    return state

def squeeze(state, output_bits=OUTPUT_BITS):
    hash_bytes = b''
    for y in range(GRID_DIM):
        for x in range(GRID_DIM):
            hash_bytes += state[x][y].to_bytes(8, 'little')
    return hash_bytes[:output_bits // 8].hex()

# Division by 180 (Flatten to 0)
def divide_by_180(hash_hex, key_quotient=None):
    H = mpmath.mpf(int(hash_hex, 16))
    pi = mpmath.pi
    divided = H / pi
    modded = divided % MODULO
    flattened = 0 if modded < 1e-10 else modded
    if key_quotient is not None:
        recovered = int((key_quotient * pi) % (1 << OUTPUT_BITS))
        return recovered, flattened
    return flattened

# KappaSHA-256 Hash Function
def kappasha256(message: bytes, key: bytes, prime_index=11):
    state = [[0 for _ in range(GRID_DIM)] for _ in range(GRID_DIM)]
    key_int = int.from_bytes(key, 'big')
    key_lanes = [[(key_int >> (LANE_BITS * (x * GRID_DIM + y))) & ((1 << LANE_BITS) - 1) for y in range(GRID_DIM)] for x in range(GRID_DIM)]
    padded = pad_message(message)
    rate_bytes = RATE // 8
    for i in range(0, len(padded), rate_bytes):
        chunk = padded[i:i + rate_bytes]
        state = absorb(state, chunk)
        for round_idx in range(ROUNDS):
            state = kappa_transform(state, key_lanes, round_idx, prime_index)
            state = theta(state)
            state = rho(state)
            state = pi(state)
            state = chi(state)
            state = iota(state, round_idx)
    hash_hex = squeeze(state)
    H = mpmath.mpf(int(hash_hex, 16))
    quotient = H // mpmath.pi
    flattened = divide_by_180(hash_hex)
    return hash_hex, flattened, quotient

# Example Usage
if __name__ == "__main__":
    message = b"test"
    key = hashlib.sha256(b"secret").digest() * 2  # 512-bit key
    hash_hex, flattened, quotient = kappasha256(message, key)
    print(f"Hash: {hash_hex}\nFlattened: {flattened}\nQuotient: {quotient}")
    recovered, _ = divide_by_180(hash_hex, quotient)
    print(f"Recovered: {recovered}")
