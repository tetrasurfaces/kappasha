# README for Kappasha

## Overview
Kappasha is a GitHub repository forked from the original hashlet project, evolved into a kappa-tilted operating system (KappashaOS) blending computational geometry with real-time decision-making for factory environments. Inspired by tetrahedral meshes and fractal patterns, it models complex surfaces and integrates thought arbitrage—detecting intent-registry divergences—alongside welding simulations and environmental factors. Rooted in repos like `fractal_tetra` and steel case-hardening simulations, "hashing" tracks cryptographic data and material phase transformations.

Designed for developers, welders, and researchers, KappashaOS intersects geometry, haptics, and physical simulations. It features Python scripts with Cython optimization for rhombus voxel grids, porosity modeling, and gyroscopic stabilization, tailored for welding, infrastructure monitoring, and material science.

## Features
- **Kappa-Tilted Navigation**: 3D rhombus voxel grid with real-time kappa adjustments, Cython-optimized for speed.
- **Thought Arbitrage**: Detects mismatches between user intent and factory registry, logged with Cython-enhanced checks.
- **Welding Simulations**: Models sequences with preheating, bead length, arc length, voltage, amperage, and environmental factors (wind, smoke, light).
- **Porosity and Material Tracking**: Simulates void growth during transformations, up to 30% porosity, with phase hashing.
- **Haptic Feedback**: Integrates `ghost_hand` for tactile cues on kappa drifts or arbitrage flags.
- **Modular Components**: 
  - `arch_utils/render.py`: Dynamic STL rendering with decision clash shading.
  - `dev_utils/lockout.py`: Tags incidents like gas ruptures.
  - `dev_utils/hedge.py`: Suggests alternate paths on unwind.
  - `dev_utils/grep.py`: Perl-style regex for log parsing.
  - `dev_utils/thought_arb.py`: Implements thought arbitrage.
  - `gyrogimbal.py`: Stabilizes motion with gyro data.
  - `frictionvibe.py`: Models vibration damping.
  - `telemetry.py`: Logs mesh and stress data.
- **CLI Interface**: Commands like `kappa ls`, `kappa decide weld`, `kappa hedge multi [gate,weld]`, optimized with Cython.
- **Testing and Logging**: Includes test stubs and decision history tracking.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/kappasha.git
   cd kappasha
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install numpy matplotlib scipy pandas pytest simpy cython
   ```

4. Compile Cython modules:
   ```bash
   python setup.py build_ext --inplace
   ```

5. Optional for advanced features (e.g., haptics, 3D export):
   ```bash
   pip install pyserial opencv-python
   ```

## Usage
Run KappashaOS from the root directory. Core functionality spans `nav3d.py` and `kappasha_os.py`, with modules in `arch_utils` and `dev_utils`.

- **Basic Navigation**:
  ```bash
  python kappasha_os.py
  ```
  Use `kappa ls` to view the optimized rhombus grid.

- **Welding Simulation**:
  ```bash
  python kappasha_os.py
  ```
  Run `kappa cd weld` to navigate, `dev_utils lockout gas_line` for incidents.

- **Decision-Making**:
  ```bash
  python kappasha_os.py
  ```
  Type `kappa decide weld` to check arbitrage, feel ghost_hand pulse.

- **Path Hedging**:
  ```bash
  python kappasha_os.py
  ```
  Use `kappa hedge multi [gate,weld]` for alternate suggestions.

- **Log Parsing**:
  ```bash
  python kappasha_os.py
  ```
  Run `kappa grep /gas_rupture/` to filter incidents.

For custom tweaks, edit `arch_utils/render.py` for voxel rendering or `dev_utils/thought_arb.py` for arbitrage logic. Recompile after changes with `python setup.py build_ext --inplace`.

## Code List of Generations and Adaptations Needed for KappashaOS
The repository requires ongoing development:

1. **Core Files Generation**:
   - `nav3d.py`: Rhombus voxel navigation with kappa tilt.
   - `kappasha_os.py`: OS integration with sensor polling and CLI.
   - `kappasha_os_cython.pyx`: Cython-optimized projection and arbitrage.
   - `gyrogimbal.py`: Gyroscopic stabilization for motion.
   - `frictionvibe.py`: Vibration damping for porosity.
   - `telemetry.py`: Log decisions and stresses.

2. **Adaptations**:
   - Enhance `thought_arb.py` with multi-intent arbitrage.
   - Update `render.py` for real-time clash shading.
   - Refine `hedge.py` for dynamic path prioritization.
   - Integrate `grep.py` with sensor data parsing.

3. **Extensions**:
   - Add `--env outdoor` to `kappasha_os.py` for wind/smoke sims.
   - Implement `--rays` in `telemetry.py` for light tracing.
   - Create `weldtest.py` for preheat vs. crack analysis.

4. **Dependencies**:
   - `numpy` for grid math.
   - `matplotlib` for visualization.
   - `scipy` for physics.
   - `simpy` for simulation.
   - `cython` for optimization.
   - `pytest` for testing.

## Testing
Run the test suite:
```bash
cd kappashaos
pytest tests/test_os.py -v
```

Set `PYTHONPATH` if needed:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/kappashaos
```

## License
KappashaOS is dual-licensed under the Apache License 2.0 and GNU Affero General Public License v3.0 or later. See file headers for details. Unauthorized use is prohibited without permission from Beau Ayres.

Copyright 2025 Beau Ayres

## Contributing
Fork the repo, enhance decision-making, welding sims, or Cython optimizations, and submit a pull request. Focus on kappa navigation or arbitrage improvements.

## Contact
Open an issue on GitHub or contact Beau Ayres for support.
