# pywise.py - Pi-based kappa indexing with lap reversals
#
# Copyright (C) 2025 Anonymous
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

from math import pi

SEED = int(str(pi)[2:20])  # First 18 digits after decimal as seed
LAP = 18  # Number of laps for reversal

def pywise_kappa(pos):
    """
    Compute a kappa index using pi digits with 18-lap reversals.
    Slices pi digits based on pos, applies reversals, and clamps to 10 bits (0-2047).
    """
    # Get a slice of pi digits (str(pi) is limited; for large pos, use a pi library or precompute)
    s = str(pi)[2:2 + LAP * pos]  # Slice digits; in practice, precompute more digits if needed
    if len(s) > LAP:
        s = s[:LAP]  # Trim to one lap's worth
    # Apply 18 reversals (concatenate reversed strings)
    reversed_s = ''
    for _ in range(LAP):
        s = s[::-1]  # Reverse
        reversed_s += s  # Concatenate
    k = int(reversed_s) % 2048  # Clamp to ~11 bits (for 2048 grid compatibility)
    return k
