# README for KappashaOS
## Overview
KappashaOS is a GitHub repository evolved from the original hashlet project, now a kappa-tilted operating system blending computational geometry with real-time decision-making for factory environments, design studios, and research. Inspired by tetrahedral meshes, fractal spirals, and the 0.19462501 manuscript constant, it models complex surfaces, tracks curvature hashes, and integrates thought arbitrage—detecting intent-registry divergences. Rooted in repos like `fractal_tetra` and welding simulations, it hashes cryptographic data alongside material phase transformations.

Designed for welders, designers, and innovators, KappashaOS intersects geometry, haptics, and physical simulations. It replaces G-code with k.k (kappa kinematics), optimized in Python with Cython for rhombus voxel grids, porosity modeling, and gyroscopic stabilization.

## Features
- **Kappa-Tilted Navigation**: 3D rhombus voxel grid with real-time kappa adjustments, Cython-optimized.
- **Thought Arbitrage**: Detects intent mismatches, logged with Cython checks.
- **k.k Kinematics**: Replaces G-code with curvature-based commands (e.g., `k0 X0.19462501 Y0.618`).
- **Welding Simulations**: Models sequences—preheat, bead length, arc, voltage, amperage, wind, smoke.
- **Porosity Tracking**: Simulates void growth (up to 30%), hashed per phase.
- **Haptic Feedback**: `ghost_hand` pulses on kappa drifts or arbitrage flags.
- **Modular Components**:
  - `arch_utils/render.py`: Dynamic STL rendering with clash shading.
  - `dev_utils/lockout.py`: Tags incidents (e.g., gas ruptures).
  - `dev_utils/hedge.py`: Suggests alternate paths.
  - `dev_utils/grep.py`: Perl-style log parsing.
  - `dev_utils/thought_arb.py`: Arbitrage logic.
  - `gyrogimbal.py`: Gyro stabilization.
  - `frictionvibe.py`: Vibration damping.
- **CLI Interface**: Commands like `kappa ls`, `kappa decide weld`, `kappa hedge multi [gate,weld]`.
- **Easter Egg**: `Hunt Primes` - offline GIMPS integration, seeds with 0.19462501.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/toddhutchinson/kappasha.git
   cd kappasha

Create and activate a virtual environment:
bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:
bashpip install numpy matplotlib scipy pandas cython

Compile Cython modules:
bashpython setup.py build_ext --inplace

Optional (haptics, 3D export):
bashpip install pyserial opencv-python


Usage
Run from root. Core files: kappasha_os.py, nav3d.py.

Basic Navigation: python kappasha_os.py → kappa ls for grid view.
Welding Sim: python kappasha_os.py → kappa cd weld, dev_utils lockout gas_line.
Decision-Making: kappa decide weld → arbitrage check, ghost_hand pulse.
Path Hedging: kappa hedge multi [gate,weld] → alternate paths.
Hunt Primes: ./easter/hunt-primes.sh → enter creds, hunt Mersenne decimals.

Code Development

Core: nav3d.py, kappasha_os.py, kappasha_os_cython.pyx (kappa tilt, arbitrage).
Adapt: Enhance thought_arb.py (multi-intent), render.py (clash shading).
Extend: Add --env outdoor to kappasha_os.py, --rays to telemetry.py.

License
KappashaOS is licensed under GNU Affero General Public License v3.0 only.

Copyright 2025 Todd Macrae Hutchinson (69 Dollard Ave, Mannum SA 5238) with Beau Ayres.
Amendment: No use in biological synthesis, gene editing, or wetware without explicit, non-coerced organism consent. Violation revokes license. Non-living applications permitted.
See GNU AGPL v3.0.

Contributing
Fork, enhance kappa navigation, arbitrage, or k.k kinematics. Submit pull requests. Share back under AGPL.
Contact
Issues on GitHub or email SENDTOTODDHUTCHINSON@GMAIL.COM (Tetrasurfaces). No DMs.
## Manuscript Constant
0.19462501 - The seed of all curves. Push once to bloom.
