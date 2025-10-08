# Kappasha
Kappasha is an interactive Python application for visualizing Mersenne prime curves, golden spirals, and 3D fractal surfaces with curvature modulation. It uses Matplotlib for 2D and 3D plotting, incorporating geometric constructions, harmonic frequency annotations, and user-defined curves. The project supports tools like protractors, rulers, and G-code generation for CNC applications, with a focus on computational geometry and curvature continuity.
Features

Mersenne Prime Curves: Plots 52 curves corresponding to known Mersenne prime exponents, scaled to an A3 landscape layout.
Golden Spiral and Green Segment: Visualizes a golden spiral and a green segment scaled to fit between purple divider lines.
Interactive Tools: Supports drawing, measuring (protractor/ruler), dimensioning, and toggling harmonic frequencies.
3D Surface Generation: Creates 3D models with fractal flower end caps and curvature modulation using a kappa grid.
G-code Export: Generates G-code for 2D curves with variable feed rates for CNC applications.
STL Export: Exports 3D models as STL files for 3D printing.
HTML Export: Saves interactive 2D plots as HTML using mpld3 (optional).

## Installation
Prerequisites

Python 3.8 or higher
pip (Python package manager)
A virtual environment (recommended)

## Dependencies
Install the required Python packages using:
pip install numpy matplotlib mpl_toolkits scipy

Optional dependency for HTML export:
pip install mpld3

## Setup

Clone or download the repository to your local machine:
git clone https://github.com/yourusername/tetrasurfaces.git
cd tetra


Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

If requirements.txt is not provided, manually install the dependencies listed above.

Ensure the tetra directory contains all necessary modules:

tetra.py

KappaSHA256.py

kappa_grid.py

green_curve.py

temperature_salt.py

forge_telemetry.py

tests/test_simulation.py

## Usage
Run the main application:

cd tetra
python tetra.py


## Interactive Controls

R: Toggle draw mode to add kappa nodes for custom curves.
A: Toggle protractor tool for angle measurements.
M: Toggle ruler tool for distance measurements.
D: Toggle dimension tool to label curve lengths.
C: Close the polyhedron manually to generate a 3D model.
G: Convert selected curve to construction geometry.
H: Hide or show selected or all hidden elements.
E: Reset the canvas.
S: Export the 3D model as an STL file.
F: Toggle visibility of harmonic frequency labels.

## Sliders
Adjust parameters in the control window:

Curvature (kappa): Controls the curvature of the green curve.
Height: Sets the height of the 3D model.
Rings: Number of loft rings in the 3D model.
Fractal Level: Depth of fractal flower recursion.
Radial/Tangential/Height Chord: Parameters for flower-like surface modulation.

## Outputs

G-code: Saved as model.gcode when a closed curve is created.
STL: Saved as model.stl when pressing 'S' with a 3D model.
HTML: Saved as mersenne_plot.html if mpld3 is installed.


## Troubleshooting Tests
If you encounter a ModuleNotFoundError for kappa_grid:

Ensure kappa_grid.py exists in the tetra/ directory.
Verify the Python path includes the project root:export PYTHONPATH=$PYTHONPATH:/path/to/kappasha


Install missing dependencies (e.g., mpld3 for HTML export):pip install mpld3



License
Kappasha is dual-licensed under the Apache License, Version 2.0, and the GNU Affero General Public License v3.0 or later. See the license headers in each source file for details.
Copyright © 2025 Beau Ayres
Contributing
Contributions are welcome! Please submit pull requests or open issues on the repository https://github.com/tetrasurfaces/kappasha.
Contact
For questions or support, contact the maintainer at tetrasurfaces or open an issue on the repository.
