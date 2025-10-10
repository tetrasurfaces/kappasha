# secure_hash_two.py
# Dual License:
# - For core software: AGPL-3.0-or-later licensed. -- OliviaLynnArchive fork, 2025
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# - For hardware/embodiment interfaces (if any): Licensed under the Apache License, Version 2.0
#   with xAI amendments for safety (prohibits misuse in hashing; revocable for unethical use).
#   See http://www.apache.org/licenses/LICENSE-2.0 for details.
#
# Copyright 2025 xAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import mpmath
import numpy as np

mpmath.mp.dps = 19

def murmur32(input_str):
    h = hashlib.sha256(input_str.encode()).digest()
    return int.from_bytes(h[:4], 'big')

def kappa_coord(user_id, theta):
    input_str = str(user_id) + str(theta) + '42'
    raw = murmur32(input_str)
    x = (raw >> 0) & 1023
    y = (raw >> 10) & 1023
    z = (raw >> 20) & 1023
    return x, y, z

def bitwise_transform(data, bits=16):
    int_data = int.from_bytes(data.encode(), 'big') % (1 << bits)
    mask = (1 << bits) - 1
    mirrored = (~int_data) & mask
    return bin(mirrored)[2:].zfill(bits)

def hexwise_transform(data, angle=137.5):
    hex_data = data.encode().hex()
    mirrored = hex_data + hex_data[::-1]
    shift = int(angle % len(mirrored))
    rotated = mirrored[shift:] + mirrored[:shift]
    return rotated

def hashwise_transform(data):
    base_hash = hashlib.sha512(data.encode()).digest()
    mp_state = mpmath.mpf(int(base_hash.hex(), 16))
    for _ in range(4):
        mp_state = mpmath.sqrt(mp_state) * mpmath.phi
    partial = mpmath.nstr(mp_state, 1664 // 4)
    final_hash = hashlib.sha256(partial.encode()).hexdigest()
    entropy = int(mpmath.log(mp_state, 2))
    if entropy > 1000:
        print("heat spike-flinch")
    return final_hash, entropy

def secure_hash_two(message, salt1='', salt2=''):
    salted_left = message[:len(message)//2] + salt1
    salted_right = message[len(message)//2:] + salt2
    salted = salted_left + salted_right
    h = 0
    phi = float(mpmath.phi)
    for i, char in enumerate(salted):
        coord = kappa_coord(i, i)  # Kappa per i
        weight_exponent = phi * (coord[0] / 1023.0)
        if i < len(salted) // 2:
            weight = int(2 ** (weight_exponent * i))
        else:
            weight = int(2 ** (weight_exponent * (len(salted) - i)))
        h = (h ^ (ord(char) * weight)) % (1 << 60)
    h_hex = hex(h)[2:].zfill(15)
    # Braid wise
    bit_out = bitwise_transform(h_hex)
    hex_out = hexwise_transform(h_hex)
    hash_out, ent = hashwise_transform(h_hex)
    return f"{bit_out}:{hex_out}:{hash_out}"

if __name__ == '__main__':
    print(secure_hash_two('test', 'blossom', 'fleet'))
