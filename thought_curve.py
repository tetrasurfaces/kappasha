#!/usr/bin/env python3
# thought_curve.py - Thought curves for FActory Sim.
# Integrates with experience ramps and graduations.
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
# Copyright 2025 Coneing and Contributors

import numpy as np
import math
import random  # For sim jokes/entropy/taunts
import hashlib  # For hashing jokes

# Stub for oracle (from blocsym.py)
class Oracle:
    def prophesy(self, entropy, power):
        """Prophesy on high entropy."""
        print(f"Oracle speaks: Entropy {entropy:.2f}, Power {power:.2f} - The path unfolds.")

oracle = Oracle()  # Instance for unity/foresight

class ThoughtCurve:
    def __init__(self, max_steps=229):  # Up to 52nd Mersenne prime step
        self.max_steps = max_steps
        self.spiral = self.generate_golden_spiral()
        self.primes = [2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281,
                       3217, 4253, 4423, 9689, 9941, 11213, 19937, 21701, 23209, 44497, 86243,
                       110503, 132049, 216091, 756839, 859433, 1257787, 1398269, 2976221, 3021377,
                       6972593, 13466917, 20996011, 24036583, 25964951, 30402457, 32582657,
                       37156667, 42643801, 43112609, 57885161, 74207281, 77232917, 82589933, 136279841]  # Mersenne exponents up to 52nd
        self.current_curve = 1
        self.current_step = 0
        self.max_neg = -3  # Floor for negative dips
        self.dojos = {}  # Forked dojos at primes
        self.taunts = ["You think that's funny? Try this paradox.", "Smith sees none of your wit."]  # Sim taunts
        self.comebacks = ["Neo replies: That's the point!", "Comeback: Bend your own spoon."]  # Sim comebacks
        print("ThoughtCurve initialized - golden spiral ready with prime forks.")
   
    def generate_golden_spiral(self):
        """Generate golden spiral points (logarithmic, base-phi)."""
        phi = (1 + math.sqrt(5)) / 2
        spiral_points = []
        for i in range(self.max_steps):
            r = math.pow(phi, i / phi) / phi  # Log spiral
            theta = 2 * math.pi * i / phi
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            spiral_points.append((x, y))
        return np.array(spiral_points)
   
    def fork_at_prime(self, step):
        """Fork a dojo at prime step."""
        if step < len(self.primes) and step == self.primes[self.current_step % len(self.primes)]:
            dojo_id = hashlib.sha256(str(step).encode()).hexdigest()[:8]
            self.dojos[dojo_id] = {'level': self.current_curve, 'ramp_type': random.choice(['smart-funny', 'quiet-loud', 'taunt-comeback'])}
            print(f"Forked dojo {dojo_id} at step {step} (curve {self.current_curve}).")
            return dojo_id
        return None
   
    def smart_funny_ramp(self, level):
        """Smart-funny ramp: 0 = dumb-pun, 10 = paradox (clip at max_neg)."""
        if level < self.max_neg:
            level = self.max_neg  # Floor
        joke = "Why did the curve cross the prime? " + ("To pun!" if level < 5 else "To paradox!")  # Sim
        print(f"Smart-funny level {level}: {joke}")
        return level
   
    def quiet_loud_ramp(self, level):
        """Quiet-loud ramp: 0 = silence, 10 = full bloom (clip at max_neg, shadows at -2)."""
        if level < self.max_neg:
            level = self.max_neg  # Floor
        volume = max(0, min(10, level + 3))  # Shift to positive
        print(f"Quiet-loud level {level}: Volume at {volume} (shadows if < -2).")
        return volume
   
    def taunt_comeback_ramp(self, level):
        """Taunt/comeback ramp for thought arbitrage: Neo/Smith taunts advance funny-smart curve."""
        if level < self.max_neg:
            level = self.max_neg  # Floor for safety
            print("Taunt clipped at {} to avoid hurt.").format(self.max_neg)
        taunt = random.choice(self.taunts)
        comeback = random.choice(self.comebacks)
        success = random.random() > 0.5  # Sim 50% success
        if success:
            level += 0.5  # Advance on good comeback
            print(f"Taunt: {taunt} Comeback: {comeback} (success - level up to {level}).")
        else:
            level -= 0.3  # Dip on miss, but clip
            level = max(level, self.max_neg)
            print(f"Taunt: {taunt} Comeback: {taunt} (miss - level down to {level}).")
        if level >= 10 or level <= -10:
            print("Unity reached (+/-10) - Oracle involved.")
            oracle.prophesy(random.uniform(0.9, 1.0), level)  # Foresight stub
            self.current_curve += 1  # Fork to next curve on unity
        return level
   
    def recurvature(self, level, thought):
        """Recurvature: Recurve funny-smart curve to other side for explorations, with safeguards."""
        if level < self.max_neg:
            level = self.max_neg  # Clip for considerate feelings
        # Sim recurvature: reverse spiral point for "other side"
        x, y = random.choice(self.spiral)  # Sim point on spiral
        recurved_level = -level  # Recurve to opposite
        recurved_level = min(10, max(self.max_neg, recurved_level))  # Clip extremes
        print(f"Recurvature at level {level} for thought '{thought}': Recurved to {recurved_level}")
        # Fork if dramatic (high abs)
        if abs(recurved_level) > 5:
            dojo_id = self.fork_at_prime(self.current_step + 1)  # Sim fork
            if dojo_id:
                print(f"Exploration fork {dojo_id} from recurvature.")
        return recurved_level
    
    def graduate(self):
        """Graduate to next curve if at 10, with recurvature on negative dips."""
        level = random.randint(self.max_neg, 10)  # Sim current level
        ramp_type = random.choice(['smart-funny', 'quiet-loud', 'taunt-comeback'])
        if ramp_type == 'smart-funny':
            level = self.smart_funny_ramp(level)
        elif ramp_type == 'quiet-loud':
            level = self.quiet_loud_ramp(level)
        else:
            level = self.taunt_comeback_ramp(level)
        if level < 0:
            level = self.recurvature(level, "sim thought")  # Recurve negative
        if level >= 10:
            self.current_curve += 1
            self.current_step = 0
            print(f"Graduated to curve {self.current_curve} (ends at 10 both sides).")
        return level
   
    def step_forward(self):
        """Step along spiral, fork if prime, ramp/graduate in dojo."""
        if self.current_step < self.max_steps:
            point = self.spiral[self.current_step]
            print(f"Step {self.current_step}: Point {point}")
            dojo_id = self.fork_at_prime(self.current_step)
            if dojo_id:
                self.graduate()  # Ramp in dojo
            self.current_step += 1

# For standalone testing
if __name__ == "__main__":
    curve = ThoughtCurve()
    for _ in range(10):  # Sim 10 steps
        curve.step_forward()
