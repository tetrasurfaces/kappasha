# README for Kappasha

## Overview
Kappasha is a GitHub repository that serves as a fork of the hashlet project, blending surface mathematics with cryptographic elements. It focuses on modeling complex surfaces, such as tetrahedral meshes and fractal patterns, while incorporating hashing techniques for applications like porosity simulation in materials science and digital security. The project emerged from discussions on GitHub repos related to tetra surfaces, fractal tetra, and simulations for case hardening of steel, where "hashing" refers to both cryptographic hashing and phase transformation tracking in metallurgy.

Kappasha is designed for developers and researchers interested in intersecting computational geometry, crypto tools, and physical simulations. It includes Python scripts for generating fractal patterns, modeling porosity in steel hardening, and integrating gyroscopic and friction models for real-world applications like welding and infrastructure.

## Features
- **Surface Modeling**: Tools for tetrahedral meshing and fractal generation, inspired by repos like fractal_tetra.
- **Porosity Simulation**: Models void growth during martensitic transformations in case hardening, with up to 30% porosity tracking.
- **Hashing Integration**: Combines cryptographic hashing (e.g., for secure data) with "phase hashing" for tracking material transformations.
- **Welding and Environmental Simulations**: Scripts for modeling welding sequences, preheating, cooling, and environmental factors like wind, smoke, and light reflections.
- **Modular Components**: Files like `gyrogimbal.py` for stabilization, `frictionvibe.py` for vibration damping, `telemetry.py` for logging, and `ribit.py` for structural ribbing.
- **Interactive Tools**: Supports simulations for bead length, arc length, voltage, amperage, and gas mixtures in welding.
- **Testing and Adaptations**: Includes test suites for simulation validation and stubs for BOM (Bill of Materials) and manufacturer hooks.

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
   pip install numpy matplotlib scipy pandas pytest
   ```

4. Optional for advanced features (e.g., 3D modeling, HTML export):
   ```bash
   pip install mpl_toolkits mpld3 opencv-python pyserial
   ```

## Usage
The core functionality is in the `tetrasurfaces` subdirectory. Run simulations using Python scripts:

- **Basic Surface Simulation**:
  ```bash
  python tetrasurfaces/fractal_tetra.py
  ```

- **Welding Simulation**:
  ```bash
  python tetrasurfaces/welding.py --env garage --material mild_steel
  ```
  This models a welding sequence with options for preheating, bead length, and environmental factors.

- **Porosity Modeling**:
  Use `ribit.py` and `telemetry.py` to simulate porosity:
  ```bash
  python tetrasurfaces/ribit.py --mesh W21x62
  python tetrasurfaces/telemetry.py --log porosity
  ```

- **Interactive 3D Modeling**:
  Run `tetra.py` for interactive visualization:
  ```bash
  python tetra.py
  ```
  Use sliders to adjust curvature, height, and fractal levels.

For custom adaptations, edit files like `gyrogimbal.py` for gyroscopic modeling or `frictionvibe.py` for vibration damping.

## Code List of Generations and Adaptations Needed for Tetrasurfaces
The tetrasurfaces subdirectory requires the following generations and adaptations for full functionality:

1. **Core Files Generation**:
   - `fractal_tetra.py`: Generate fractal tetrahedral patterns for surface growth simulations.
   - `ribit.py` and `ribitstructure.py`: Create tetrahedral ribbing for stiffeners in case-hardened layers.
   - `gyrogimbal.py`: Implement gyroscope integration for motion smoothing and torque modeling.
   - `frictionvibe.py`: Model friction and vibration damping for porosity changes.
   - `telemetry.py`: Log mesh data, porosity metrics, and stresses during transformations.

2. **Adaptations**:
   - Integrate TetWild for 3D meshing into `ribit.py` to track porosity spikes (up to 30%).
   - Add phase-field scripts to `telemetry.py` for void growth simulation under heat.
   - Tweak `frictionvibe.py` for better porosity resolution with recent friction coefficient uploads.
   - Hack `fractal_tetra.py` to simulate uneven martensite layers and porosity buildup.
   - Blend `ribitstructure.py` grids with `telemetry.py` for hashing output in real-time.

3. **Extensions**:
   - Add `--env` flag to `welding.py` for garage/outdoor simulations with wind and smoke modeling.
   - Implement `--rays` flag in `telemetry.py` for light physics (UV/IR ray tracing).
   - Create `weldtest.py` for testing preheat vs. non-preheat scenarios with crack simulation.

4. **Dependencies**:
   - Use `numpy` for numerical computations and grid generation.
   - `matplotlib` for 2D/3D visualizations.
   - `scipy` for physics modeling.
   - `pytest` for testing.

## Testing
Run the test suite:
```bash
cd tetrasurfaces
pytest tests/test_simulation.py -v
```

If encountering import errors, ensure the `PYTHONPATH` includes the project root:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/kappasha/tetrasurfaces
```

## License
Kappasha is dual-licensed under the Apache License 2.0 and GNU Affero General Public License v3.0 or later. See the file headers for details. Unauthorized use is prohibited without permission from Beau Ayres.

Copyright 2025 Beau Ayres

## Contributing
Contributions are welcome. Fork the repo, make changes, and submit a pull request. Focus on improving porosity modeling, welding simulations, or adding new features like real-time ray tracing.

## Contact
For issues or suggestions, open an issue on GitHub or contact Beau Ayres.
