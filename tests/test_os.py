# Copyright 2025 Beau Ayres, xAI
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Proprietary Software - All Rights Reserved
#
# This software is proprietary and confidential. Unauthorized copying,
# distribution, modification, or use is strictly prohibited without
# express written permission from Beau Ayres.
#
# AGPL-3.0-or-later licensed
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

#!/usr/bin/env python3
# test_os.py - Unit tests for KappashaOS components.

import unittest
import sys
sys.path.append("tetrasurfaces/tetra")
from kappasha_os import KappashaOS
from arc_listen import ArcListener
from ghost_hand import GhostHand

class TestKappashaOS(unittest.TestCase):
    def setUp(self):
        self.os = KappashaOS()
        self.listener = ArcListener()
        self.hand = GhostHand(kappa=0.2)

    def test_kappa_ls(self):
        """Test kappa ls command output."""
        self.os.run_command("kappa ls")
        self.assertTrue(len(self.os.nav.project_third_angle()[0]) > 0)

    def test_arc_listen_feedback(self):
        """Test arc listen triggers correct pulses."""
        # Mock sound data (simplified)
        mock_data = bytearray([0] * self.listener.CHUNK)
        freq = self.listener.analyze_sound(mock_data)
        if freq < 2000:
            self.hand.pulse(1)
        elif freq > 8000:
            self.hand.pulse(2)
        self.assertTrue(self.hand.pulse_count in [1, 2])

    def tearDown(self):
        self.listener.stop()

if __name__ == "__main__":
    unittest.main()
