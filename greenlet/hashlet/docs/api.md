# BlockChan Greenpaper API Documentation

## Overview
This document outlines the public API for the BlockChan Greenpaper project, organized by module. As of September 21, 2025, the API is based on the modular structure implemented in `src/`.

## Modules

### `src.config`
- **Constants**:
  - `DB_FILE`: str = "blockchan.db"
  - `TRADING_PAIRS`: List[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "~ESC/USDT", "LLLP/GMEx"]
  - `GRID_DIM`: int = 2141
  - `KAPPA_BASE`: float = 0.3536
  - `FEE_RATE`: float = 0.003
  - `MARTINGALE_FACTOR`: float = 2.0
  - `HASHES`: Dict[str, str] = {e.g., "GREEN_TXT_HASH": "a8c2b3f1e5d7a9b0c4e6f2d8a1b3c5e7f9d0a2b4c6e8f0d2a3b5c7e9f1d3a4b6"}

### `src.utils.hash_utils`
- **Class: `HashUtils`**
  - `sha1664(indexed_hash)`: tuple[str, int] - Generates a SHA1664 hash with sponge permutations.
  - `advanced_hash(seed, bits=16, laps=18)`: tuple[int, int, int] - Generates an advanced hash with reversals.
  - `m53_collapse(p, stake=1)`: tuple[int, int] - Computes M53 collapse profit check.

### `src.utils.math_utils`
- `get_kappa_coordinates(radius: int, curvature: float, height: int) -> Dict`: Calculates kappa-based coordinates.
- `kappa_calc(n: int, block_height: int = 0) -> float`: Computes kappa value.
- `compute_fibonacci_spiral_segment(...) -> tuple`: Generates a Fibonacci spiral segment.

### `src.models.bastion`
- **Class: `SHA1664`**
  - `hash_transaction(data: str) -> str`: Generates a SHA-256 hash.
  - `prevent_double_spending(tx_id: str) -> bool`: Checks for double-spending.
  - `receive_gossip(data: Dict, sender: str)`: Logs gossip data.
- **Class: `EphemeralBastion`**
  - `set_ternary_state(state: Any)`: Sets the ternary state.
  - `validate(data: str) -> bool`: Validates data length.

### `src.models.green_models`
- **Class: `Facehuggers`**
  - `process_prompt(prompt: str) -> Dict`: Processes green prompts.
  - `apply_to_curve(curve_points: np.ndarray) -> np.ndarray`: Applies curvature adjustments.
- **Class: `CurvatureVerbismGenerator`**
  - `curve_map_kappa(points: np.ndarray = None, curve_mode: str = 'k_curves') -> np.ndarray`: Computes NURBS curvature.
  - `demo_greenspline_animation(frames: int = 200) -> None`: Generates a greenspline animation.

### `src.visuals.animations`
- `demo_greenspline_animation(frames: int = 200) -> None`: Generates and saves a greenspline animation.
- `animate_logo(frames: int = 200) -> None`: Generates and saves a logo animation.
- `demo_shuttle(frames: int = 200) -> None`: Generates and saves a shuttle animation.
- `create_droplets(ax, positions, alpha=0.7)`: Creates ternary droplets for visualization.

### `src.main`
- **Class: `GreenpaperUX`**
  - `validate_hashes() -> bool`: Validates content hashes.
  - `demo_functions() -> None`: Runs all TOC demos.
  - `demo_<toc_number>()`: Individual demo methods for each TOC section (e.g., `demo_scalability`, `demo_curvature_verbism`).

## Usage
- Run `python src/main.py` to execute all demos.
- Import specific modules (e.g., `from src.utils.hash_utils import HashUtils`) for custom use.

## Notes
- API is subject to change as the project evolves.
- Refer to `src/config.py` for the latest hash values and constants.
- Documentation will be expanded with detailed parameter descriptions in future updates.
