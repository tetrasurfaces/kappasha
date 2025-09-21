# hybrid.py - Hybrid Parser with Cython/Perl Integration
# Integrates Python, Cython, and Perl for parsing and curvature calculations
import numpy as np
from typing import List
class HybridGreenText:
    def __init__(self, sparse_n: int = 50):
        self.sparse_n = sparse_n
        self.perl_script = "green_parser.pl"
    def parse_green_perl(self, text: str) -> str:
        try:
            import subprocess
            result = subprocess.run(['perl', self.perl_script, text], capture_output=True, text=True)
            return result.stdout
        except Exception as e:
            print(f"Perl parsing error: {e}")
            return ""
    def scale_curvature(self, kappa_values: np.ndarray, blue_gold_swap: bool = True) -> np.ndarray:
        from scipy.interpolate import griddata
        sparse_t = np.array([float((k * PHI) % 1) for k in range(self.sparse_n)])
        sparse_kappa = griddata(np.linspace(0, 1, len(kappa_values)), kappa_values, sparse_t, method='linear')
        interpolated = griddata(sparse_t, sparse_kappa, np.linspace(0, 1, len(kappa_values)), method='cubic')
        if blue_gold_swap:
            bands = int(np.mean(interpolated) * PHI)
            interpolated += np.sin(np.linspace(0, 2 * np.pi, len(interpolated))) * bands
        return interpolated
