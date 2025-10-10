#!/usr/bin/env python3
# dev_utils/hedge.py - Path hedging with ThoughtCurve for kappa awareness.
# Part of Kappasha OS, pure civilian engineering.
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

def hedge(curve, path):
    """Hedge single path with ThoughtCurve tangent check."""
    if len(path) < 2:
        return "hold"
    tangent, _ = curve.spiral_tangent(path[-2], path[-1])
    return "unwind" if tangent else "hold"

def multi_hedge(curve, paths):
    """Hedge multiple paths, suggest alternates on arbitrage."""
    options = []
    for p1, p2 in paths:
        tangent, _ = curve.spiral_tangent(p1, p2)
        options.append(("unwind" if tangent else "hold", p2))
    stable = [p for act, p in options if act == "hold"]
    if not stable:
        return "unwind, suggest alternate paths"
    return f"hold on {stable[0]}, alternates: {', '.join(stable[1:])}"
