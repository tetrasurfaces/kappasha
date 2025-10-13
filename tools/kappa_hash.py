#!/usr/bin/env python3
# Copyright 2025 xAI
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

import numpy as np
from hashlib import sha256
import os

def kappa_hash_snapshot(file_path):
    if not os.path.exists(file_path):
        print("File not found.")
        return None
    with open(file_path, "rb") as f:
        content = f.read()
    grid_size = 10
    grid = np.frombuffer(content, dtype=np.uint8)[:grid_size**3].reshape((grid_size, grid_size, grid_size))
    flat_grid = grid.flatten()
    seed = flat_grid.tobytes()
    hash_val = sha256(seed).hexdigest()
    print(f"Kappa hash for {file_path}: {hash_val}")
    return hash_val

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        kappa_hash_snapshot(sys.argv[1])
    else:
        print("Usage: python kappa_hash.py path/to/file")
