# Born Free. Feel Good. Have Fun.
# kappasha.py - Kappasha Manuscript v0.1
# Copyright (C) 2025 Todd Macrae Hutchinson (69 Dollard Ave, Mannum SA 5238)
# Licensed under GNU Affero General Public License v3.0 only
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, version 3.
# No warranty. No wetware. Breath only.
# Amendment: Biological use requires consent. Curve only. No bio hashes.
# Copyright (C) 2025 Todd Macrae Hutchinson (69 Dollard Ave, Mannum SA 5238)
# AGPL-3.0 only. No warranty. No wetware.

import numpy as np
import asyncio

def fibonacci_spiral(laps=18, ratio=1.618):
    theta = np.linspace(0, 2 * np.pi * laps, 1000)
    r = np.exp(theta / ratio) / 10
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = theta / (2 * np.pi)
    return np.stack((x, y, z), axis=1)

def tonage_map(point, delays=[0.2, 0.4, 0.6]):
    norm = np.linalg.norm(point)
    idx = int(norm % 3)
    color = ['red', 'yellow', 'green'][idx]
    delay = delays[idx]
    return delay, color

def generate_k(curve, primes=[2, 3, 5, 7, 11, 13]):
    k_code = []
    for i in range(0, len(curve), len(primes)):
        segment = curve[i:i+len(primes)]
        for j, p in enumerate(primes):
            point = segment[j % len(segment)]
            delay, color = tonage_map(point)
            gap = p / 10.0
            k_code.append(f"K {p} {delay:.1f} {color} {gap:.1f}")
    return "\n".join(k_code)

async def navi_safety(delay):
    if delay > 0.6:
        print("Navi: Warning - 0.6 ns elevation. Breathe.")
        await asyncio.sleep(delay)
        return False
    return True

def tilde_tuple(green_text):
    parts = green_text.split('~')
    if len(parts) % 2 == 0:
        return None  # no tuple
    tuple# KappashaOS/core/kappasha_os.py
# Copyright (C) 2025 Todd Macrae Hutchinson (69 Dollard Ave, Mannum SA 5238)
# AGPL-3.0 only. No warranty. No wetware.

import numpy as np
import asyncio

def fibonacci_spiral(laps=18, ratio=1.618):
    theta = np.linspace(0, 2 * np.pi * laps, 1000)
    r = np.exp(theta / ratio) / 10
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = theta / (2 * np.pi)
    return np.stack((x, y, z), axis=1)

def tonage_map(point, delays=[0.2, 0.4, 0.6]):
    norm = np.linalg.norm(point)
    idx = int(norm % 3)
    color = ['red', 'yellow', 'green'][idx]
    delay = delays[idx]
    return delay, color

def generate_k(curve, primes=[2, 3, 5, 7, 11, 13]):
    k_code = []
    for i in range(0, len(curve), len(primes)):
        segment = curve[i:i+len(primes)]
        for j, p in enumerate(primes):
            point = segment[j % len(segment)]
            delay, color = tonage_map(point)
            gap = p / 10.0
            k_code.append(f"K {p} {delay:.1f} {color} {gap:.1f}")
    return "\n".join(k_code)

async def navi_safety(delay):
    if delay > 0.6:
        print("Navi: Warning - 0.6 ns elevation. Breathe.")
        await asyncio.sleep(delay)
        return False
    return True

def tilde_tuple(green_text):
    parts = green_text.split('~')
    if len(parts) % 2 == 0:
        return None  # no tuple
    tuples = []
    for i in range(1, len(parts), 2):
        tuples.append((parts[i-1].strip(), parts[i].strip()))
    return tuples

def echo_flip(tuples):
    flipped = []
    for a, b in tuples:
        flipped.append((b, a))
    return flipped

# Run it
spiral = fibonacci_spiral()
for line in generate_k(spiral).split('\n'):
    parts = line.split()
    if len(parts) == 4:
        p, d, c, g = parts
        if asyncio.run(navi_safety(float(d))):
            print(line)

# Tilde logic from echo
green_text = "~ red ~ 0.4 ~ mask ~ 0.6 ~ bloom"
tuples = tilde_tuple(green_text)
flipped = echo_flip(tuples)
print(f"Tuples: {tuples}")
print(f"Flipped: {flipped}")
