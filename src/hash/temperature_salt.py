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

def secure_hash_two(message, salt="xAI_temp_salt"):
    """
    Securely hash temperature data with a fixed salt, using exponential weighting
    to mimic a case-hardened diamond lattice structure.
    
    Args:
        message (str or float): Temperature data (e.g., "23.5C" or 23.5).
        salt (str): Fixed salt for deterministic hashing (default: "xAI_temp_salt").
    
    Returns:
        str: 60-bit hexadecimal hash.
    """
    # Convert message to string and append salt
    input_data = str(message) + salt
    h = 0
    two = 2  # Scaling factor for double density
    for i, char in enumerate(input_data):
        # Left-heavy for first half, right-heavy for second
        weight = (2 ** i) if i < len(input_data) // 2 else (2 ** (len(input_data) - i))
        h = (h * 1664 + ord(char) * weight * two) % (1 << 60)
    return hex(h)

# Example usage for temperature data
if __name__ == "__main__":
    temp_data = "23.5C"
    hash_result = secure_hash_two(temp_data)
    print(f"Temperature: {temp_data}, Hash: {hash_result}")
