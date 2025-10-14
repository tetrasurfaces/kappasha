import numpy as np
import hashlib
import subprocess
import logging
from src.config import *
from typing import List, Dict, Any
from src.utils.math_utils import compute_curvature
from src.visuals.plot_utils import create_blob_surface, add_light_slicks
from src.visuals.animations import create_droplets
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import griddata
from matplotlib.colors import LightSource

logger = logging.getLogger(__name__)

class Facehuggers:
    """Facehuggers Logic: Accepts >>>> green prompts as ramps for adjustments."""
    def __init__(self, perl_script="green_parser.pl"):
        self.perl_script = perl_script
        self.prompts = []

    def process_prompt(self, prompt: str) -> Dict[str, Any]:
        """Process >>>> green prompts via Perl parser."""
        try:
            if not prompt.startswith('>>>>'):
                raise ValueError("Prompt must start with >>>>")
            green_text = prompt[4:].strip()
            with open('temp_green.txt', 'w') as f:
                f.write(green_text)
            result = subprocess.run(['perl', self.perl_script, 'temp_green.txt'], capture_output=True, text=True)
            ramp = result.stdout.strip()
            adjustments = {'ramp_factor': 1.618 if 'know thyself' in ramp else 1.0}  # PHI scale for "know thyself"
            mnemonic = hashlib.sha256(ramp.encode()).hexdigest()[:8]
            self.prompts.append(ramp)
            return {'ramp': ramp, 'adjustments': adjustments, 'mnemonic': mnemonic}
        except Exception as e:
            logger.error(f"Process prompt error: {e}")
            return {'ramp': '', 'adjustments': {'ramp_factor': 1.0}, 'mnemonic': '00000000'}

    def apply_to_curve(self, curve_points: np.ndarray) -> np.ndarray:
        """Apply prompt-based curvature adjustments."""
        try:
            if not self.prompts or curve_points.shape[0] < 3:
                return curve_points
            kappa = np.abs(np.diff(curve_points[:, 0]) * np.diff(np.diff(curve_points[:, 1])) - np.diff(curve_points[:, 1]) * np.diff(np.diff(curve_points[:, 0]))) / (np.diff(curve_points[:, 0])**2 + np.diff(curve_points[:, 1])**2)**1.5
            kappa = np.concatenate(([kappa[0]], kappa))
            for prompt in self.prompts:
                factor = 1.618 if 'know thyself' in prompt else 1.0
                if 'query' in prompt:
                    kappa += 0.1 * factor * np.sin(np.linspace(0, 2 * np.pi, len(kappa)))
                elif 'chaos' in prompt:
                    kappa -= 0.1 * factor * np.cos(np.linspace(0, 2 * np.pi, len(kappa)))
            return np.vstack((curve_points[:, 0], curve_points[:, 1] + kappa)).T
        except Exception as e:
            logger.error(f"Apply to curve error: {e}")
            return curve_points

class CurvatureVerbismGenerator:
    def __init__(self, degree: int = 5):
        self.degree = degree
        self.points = np.array([[0, 0], [0.33, 0.1], [0.66, 0.1], [1, 0], [0.5, 0.05], [0.25, 0.02]])  # Robust point set
        self.facehuggers = Facehuggers()

    def curve_map_kappa(self, points: np.ndarray = None, curve_mode: str = 'k_curves') -> np.ndarray:
        """Compute curvature for NURBS curve, ensuring constant curvature with n=5 B-spline."""
        try:
            points = points if points is not None else self.points
            if len(points) <= self.degree:
                logger.warning(f"Insufficient points ({len(points)}) for degree {self.degree}, using default points")
                points = np.array([[0, 0], [0.33, 0.1], [0.66, 0.1], [1, 0], [0.5, 0.05], [0.25, 0.02]])
            tck, u = splprep(points.T, k=min(self.degree, len(points)-1), s=0.01, per=True)  # n=5 constant curvature
            u_fine = np.linspace(0, 1, 1000)  # Fixed to 1000 for interpolation
            x, y = splev(u_fine, tck)
            t_fine = np.linspace(0, 1, 1000)
            kappa, _, _ = compute_curvature(x, y, t_fine)
            prompt = ">>>> be me design logo flip to ternary swap code model rotate 3D know thyself"
            adjustments = self.facehuggers.process_prompt(prompt)
            factor = adjustments['adjustments']['ramp_factor']
            if curve_mode == 'k_curves':
                kappa = np.clip(kappa / (np.max(np.abs(kappa) + 1e-10) * 400) * factor, 0.02, 0.03) * 0.833375
            elif curve_mode == 'arcs':
                kappa = np.clip(kappa / (np.max(np.abs(kappa) + 1e-10) * 300) * factor, 0.02, 0.03) * 0.833375
            elif curve_mode == 'kappa_vectors':
                kappa = np.tan(np.cumsum(kappa) / len(kappa) * 2 * np.pi) * factor
            if not np.all(np.isfinite(kappa)):
                logger.warning("Non-finite kappa values detected, using fallback")
                kappa = np.array([0.02500125] * len(t_fine))
            kappa = np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(kappa)), kappa)  # Interpolate to 100 points
            return kappa
        except Exception as e:
            logger.error(f"Curve map kappa error: {e}")
            return np.array([0.02500125] * 100)

    def demo_greenspline_animation(self, frames: int = 200) -> None:
        """Generate and save greenspline animation with constant curvature and Boas rendering."""
        try:
            fig = plt.figure(figsize=(10, 8), facecolor='black')
            ax = fig.add_subplot(111, projection='3d')
            t = np.linspace(0, 1, 100)
            x, y, nodes = compute_green_segment(t)
            if len(x) == 0:
                logger.error("Empty x array from compute_green_segment, using fallback")
                x, y = np.linspace(0, 1, 100), np.zeros(100)
            z = np.zeros_like(x)
            t_fine = np.linspace(0, 1, len(x))
            kappa = self.curve_map_kappa()
            if len(kappa) != len(x):
                logger.warning(f"Kappa length mismatch ({len(kappa)} vs {len(x)}), resizing kappa")
                kappa = np.interp(np.linspace(0, 1, len(x)), np.linspace(0, 1, len(kappa)), kappa)
            X_blue, Y_blue, Z_blue, X_gold, Y_gold, Z_gold = create_blob_surface(x, y, z, TUBE_RADIUS, NUM_SIDES, kappa)
            light = LightSource(azdeg=315, altdeg=45)
            rgb_blue = np.ones((X_blue.shape[0], X_blue.shape[1], 3)) * [0, 0, 1]  # Blue
            rgb_gold = np.ones((X_gold.shape[0], X_gold.shape[1], 3)) * [1, 0.84, 0]  # Gold
            shaded_blue = light.shade_rgb(rgb_blue, Z_blue)
            shaded_gold = light.shade_rgb(rgb_gold, Z_gold)
            blob_surface_blue = ax.plot_surface(X_blue, Y_blue, Z_blue, facecolors=shaded_blue, edgecolor='none')
            blob_surface_gold = ax.plot_surface(X_gold, Y_gold, Z_gold, facecolors=shaded_gold, edgecolor='none')
            sph_radius = 1.5
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            u, v = np.meshgrid(u, v)
            x_sph = sph_radius * np.cos(u) * np.sin(v)
            y_sph = sph_radius * np.sin(u) * np.sin(v)
            z_sph = sph_radius * np.cos(v)
            ax.plot_wireframe(x_sph, y_sph, z_sph, color='gray', alpha=0.3, rstride=3, cstride=3)
            mersenne_3d = np.array([[0, 0, 1], [0.5, 0, 0], [-0.5, 0, 0]])  # Simplified Mersenne points
            for p in mersenne_3d:
                ax.plot([0, p[0]], [0, p[1]], [0, p[2]], 'r--', alpha=0.5, lw=1)
            printed_frames = set()
            def update(frame):
                ax.clear()
                ax.set_xlim(-2, 2)
                ax.set_ylim(-2, 2)
                ax.set_zlim(-2, 2)
                ax.axis('off')
                ax.set_facecolor('black')
                if frame < 50:
                    ax.view_init(elev=30, azim=frame * 7.2)
                    blob_surface_blue.set_visible(True)
                    blob_surface_gold.set_visible(True)
                elif frame < 150:
                    local_frame = frame - 50
                    swap = np.sin(local_frame / 50 * 2 * np.pi)
                    positions = [
                        [0 + swap * 0.5, 0, 0],
                        [0.5 + swap * -0.5, 0, 0],
                        [-0.5 + swap * 0.5, 0, 0]
                    ]
                    create_droplets(ax, positions, alpha=0.7)
                    ax.plot_wireframe(x_sph, y_sph, z_sph, color='gray', alpha=0.3, rstride=3, cstride=3)
                    for p in mersenne_3d:
                        ax.plot([0, p[0]], [0, p[1]], [0, p[2]], 'r--', alpha=0.5, lw=1)
                    ax.text(0, -1.5, 0, 'Ternary Swap: 0/1/e', fontsize=12, color='white', ha='center', va='center')
                    blob_surface_blue.set_visible(False)
                    blob_surface_gold.set_visible(False)
                else:
                    local_frame = frame - 150
                    ax.view_init(elev=30, azim=local_frame * 7.2 + 360)
                    blob_surface_blue.set_visible(True)
                    blob_surface_gold.set_visible(True)
                    alpha = 1 - (local_frame / 50)
                    if alpha > 0:
                        swap = np.sin((150 + local_frame) / 50 * 2 * np.pi)
                        positions = [[0 + swap * 0.5, 0, 0], [0.5 + swap * -0.5, 0, 0], [-0.5 + swap * 0.5, 0, 0]]
                        create_droplets(ax, positions, alpha=alpha)
                    ax.plot_wireframe(x_sph, y_sph, z_sph, color='gray', alpha=0.3, rstride=3, cstride=3)
                    for p in mersenne_3d:
                        ax.plot([0, p[0]], [0, p[1]], [0, p[2]], 'r--', alpha=0.5 * alpha, lw=1)
                add_light_slicks(ax, alpha=min(1, max(0, (frame - 50) / 100)) if frame < 150 else min(1, max(0, (200 - frame) / 50)))
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                if frame % 45 == 0 and frame not in printed_frames:
                    kappa_hash = hashlib.sha256((str(np.mean(kappa)) + str(frame)).encode()).hexdigest()[:8]
                    print(f"Mnemonic at frame {frame}: {kappa_hash}")
                    printed_frames.add(frame)
                return [blob_surface_blue, blob_surface_gold]
            anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
            anim.save('greenspline_animation.gif', writer='pillow', fps=20)
            print("Simulating greenspline_animation... Saved as 'greenspline_animation.gif'")
        except Exception as e:
            logger.error(f"Demo greenspline animation error: {e}")

class BoasAllocations:
    def __init__(self):
        self.strides = {"blue": 4, "gold": 6}
        self.hybrid_parser = HybridGreenText()

    def allocate(self, hash_str: str, color: str = "blue") -> str:
        """Allocate hash string with blue or gold strides."""
        try:
            stride = self.strides.get(color, 4)
            allocated = ""
            for i in range(0, len(hash_str), stride):
                chunk = hash_str[i:i+stride]
                if chunk.isdigit():
                    vals = np.array(list(map(int, chunk)))
                    smoothed = gaussian_filter1d(vals, sigma=1.0)
                    allocated += ''.join(map(str, smoothed.astype(int)))
                else:
                    allocated += chunk
            return allocated
        except Exception as e:
            logger.error(f"Allocate error: {e}")
            return hash_str

    def compute_curvature(self, s: float, color: str = "gold") -> float:
        """Compute curvature based on stride and position."""
        try:
            stride = self.strides.get(color, 6)
            return math.sin(s * 2 * math.pi / stride) * KAPPA_BASE
        except Exception as e:
            logger.error(f"Compute curvature error: {e}")
            return 0.0

    def scale_curvature_forward(self, kappa_values: np.ndarray, color: str = "blue") -> np.ndarray:
        """Scale curvature values with blue/gold swap option."""
        blue_gold_swap = color == "gold"
        return self.hybrid_parser.scale_curvature(kappa_values, blue_gold_swap)

    def reversal_collapse_curve(self, curve_points: np.ndarray) -> float:
        """Compute reversal collapse kappa for curve points."""
        return self.hybrid_parser.reversal_collapse(curve_points)

class HybridGreenText:
    def __init__(self, sparse_n: int = 50):
        self.sparse_n = sparse_n
        self.perl_script = "green_parser.pl"

    def parse_green_perl(self, text: str) -> str:
        """Parse green text for verb ramps."""
        try:
            lines = text.splitlines()
            ramp_code = []
            for line in lines:
                if line.startswith('>'):
                    verb = line[1:].strip().lower()
                    ramp_code.append(f"# {verb} ramp")
            return "\n".join(ramp_code)
        except Exception as e:
            logger.error(f"Parse green perl error: {e}")
            return ""

    def scale_curvature(self, kappa_values: np.ndarray, blue_gold_swap: bool = True) -> np.ndarray:
        """Scale curvature values with blue/gold swap option."""
        try:
            if not isinstance(kappa_values, np.ndarray) or kappa_values.size < 2:
                logger.error("Invalid kappa_values: empty or insufficient points")
                return np.array([0.02500125] * 1000)
            if not np.all(np.isfinite(kappa_values)):
                logger.error("kappa_values contains NaN or infinite values")
                return np.array([0.02500125] * len(kappa_values))
            sparse_t = np.array([float((k * PHI_FLOAT) % 1) for k in range(self.sparse_n)])
            sparse_kappa = griddata(
                np.linspace(0, 1, len(kappa_values)),
                kappa_values,
                sparse_t,
                method='linear',
                fill_value=0.02500125
            )
            if np.any(np.isnan(sparse_kappa)):
                logger.warning("Sparse interpolation resulted in NaN values, using fallback")
                return np.array([0.02500125] * len(kappa_values))
            interpolated = griddata(
                sparse_t,
                sparse_kappa,
                np.linspace(0, 1, len(kappa_values)),
                method='cubic',
                fill_value=0.02500125
            )
            if np.any(np.isnan(interpolated)):
                logger.warning("Interpolation resulted in NaN values, using fallback")
                return np.array([0.02500125] * len(kappa_values))
            if blue_gold_swap:
                mean_kappa = np.mean(interpolated)
                if np.isnan(mean_kappa):
                    logger.warning("Mean of interpolated values is NaN, skipping blue/gold swap")
                else:
                    bands = int(mean_kappa * PHI_FLOAT)
                    interpolated += np.sin(np.linspace(0, 2 * np.pi, len(interpolated))) * bands
            return interpolated
        except Exception as e:
            logger.error(f"Scale curvature error: {e}")
            return np.array([0.02500125] * len(kappa_values))

    def reversal_collapse(self, curve_points: np.ndarray) -> float:
        """Compute reversal collapse kappa for curve points."""
        try:
            n = curve_points.shape[0]
            if n < 3:
                logger.error("curve_points too small for curvature calculation")
                return 0.0
            l = curve_points[:, 0]
            h = curve_points[:, 1]
            dl = np.diff(l)
            dh = np.diff(h)
            d2l = np.diff(dl)
            d2h = np.diff(dh)
            kappa = np.abs(dl[:-1] * d2h - dh[:-1] * d2l) / (dl[:-1]**2 + dh[:-1]**2)**1.5 * PHI_FLOAT
            kappa_mean = np.mean(kappa)
            if np.isnan(kappa_mean):
                logger.error("Kappa mean is NaN in reversal_collapse")
                return 0.0
            mnemonic = hashlib.sha256(str(kappa_mean).encode()).hexdigest()[:8]
            print(f"Mnemonic: {mnemonic}")
            return kappa_mean
        except Exception as e:
            logger.error(f"Reversal collapse error: {e}")
            return 0.0

class GreenTextLanguage:
    def __init__(self, green_txt: str):
        """Initialize with green text rules."""
        self.rules = green_txt

    def parse(self, text: str) -> List[str]:
        """Parse greentext for verb ramps."""
        try:
            return [line[1:].strip().lower() for line in text.splitlines() if line.startswith('>')]
        except Exception as e:
            logger.error(f"Parse greentext error: {e}")
            return []

    def generate_mnemonic(self, kappa: float) -> str:
        """Generate mnemonic from kappa value."""
        try:
            return hashlib.sha256(str(kappa).encode()).hexdigest()[:8]
        except Exception as e:
            logger.error(f"Generate mnemonic error: {e}")
            return ""

class KeyspaceHUD:
    def __init__(self):
        self.spiral = SpiralNU()

    def map_blind_spots(self, points: np.ndarray) -> np.ndarray:
        """Map blind spots by flipping y-coordinates."""
        try:
            swapped = points.copy()
            swapped[:, 1] = -swapped[:, 1]
            return swapped
        except Exception as e:
            logger.error(f"Map blind spots error: {e}")
            return points

    def spawn_mnemonic(self, kappa: float) -> str:
        """Spawn mnemonic from kappa value."""
        try:
            return self.spiral.poly_hash_256(str(kappa))
        except Exception as e:
            logger.error(f"Spawn mnemonic error: {e}")
            return ""

class OBEWeaving:
    def __init__(self):
        self.ternary_states = []

    def weave_ternary(self, data: str) -> str:
        """Weave ternary state from data hash."""
        try:
            hash_val = hashlib.sha256(data.encode()).hexdigest()
            state = '1' if int(hash_val, 16) % 3 == 0 else '0' if int(hash_val, 16) % 3 == 1 else 'e'
            self.ternary_states.append(state)
            return state
        except Exception as e:
            logger.error(f"Weave ternary error: {e}")
            return ""

    def animate_logo(self, frames: int = 200) -> None:
        """Generate and save logo animation with ternary swap."""
        try:
            fig, ax = plt.subplots(facecolor='black')
            t = np.linspace(0, 1, 100)
            x, y, _ = compute_green_segment(t)
            if len(x) == 0:
                logger.error("Empty x array from compute_green_segment, using fallback")
                x, y = np.linspace(0, 1, 100), np.zeros(100)
            def update(frame):
                ax.clear()
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.axis('off')
                ax.set_facecolor('black')
                if frame < 50:
                    ax.plot(x + frame * 0.02, y, 'b-', label='Grok Logo')
                    ax.scatter([x[0] + frame * 0.02, x[-1] + frame * 0.02], [y[0], y[-1]], c='gold')
                elif frame < 150:
                    local_frame = frame - 50
                    swap = np.sin(local_frame / 50 * 2 * np.pi)
                    pos0 = [0 + swap * 0.5, 0]
                    pos1 = [0.5 + swap * -0.5, 0]
                    pos2 = [-0.5 + swap * 0.5, 0]
                    ax.scatter(pos0[0], pos0[1], c='blue', alpha=0.7, label='0')
                    ax.scatter(pos1[0], pos1[1], c='green', alpha=0.7, label='1')
                    ax.scatter(pos2[0], pos2[1], c='red', alpha=0.7, label='e')
                    ax.text(0, -0.7, 'Ternary Swap', color='white', ha='center', va='center')
                else:
                    local_frame = frame - 150
                    ax.plot(x + local_frame * 0.02, y, 'b-', label='Grok Logo')
                    ax.scatter([x[0] + local_frame * 0.02, x[-1] + local_frame * 0.02], [y[0], y[-1]], c='gold')
                    alpha = 1 - (local_frame / 50)
                    if alpha > 0:
                        swap = np.sin((150 + local_frame) / 50 * 2 * np.pi)
                        pos0 = [0 + swap * 0.5, 0]
                        pos1 = [0.5 + swap * -0.5, 0]
                        pos2 = [-0.5 + swap * 0.5, 0]
                        ax.scatter(pos0[0], pos0[1], c='blue', alpha=alpha * 0.7)
                        ax.scatter(pos1[0], pos1[1], c='green', alpha=alpha * 0.7)
                        ax.scatter(pos2[0], pos2[1], c='red', alpha=alpha * 0.7)
                ax.legend(loc='upper right')
                return []
            anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
            anim.save('logo_animation.gif', writer='pillow', fps=20)
            print("Simulating logo animation... Saved as 'logo_animation.gif'")
        except Exception as e:
            logger.error(f"Animate logo error: {e}")

class SeraphGuardian:
    def __init__(self, threshold=0.69):
        self.threshold = threshold

    def shannon_entropy(self, data):
        counts = np.bincount([ord(c) for c in data])
        probs = counts / len(data)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def test(self, input_mnemonic):
        """Non-reactive test: Yield if 'The One' (entropy > threshold)."""
        echo_hash = hashlib.sha256(input_mnemonic.encode()).hexdigest()
        entropy = self.shannon_entropy(input_mnemonic)
        return True, echo_hash if entropy > self.threshold else (False, "Apology: Not The One")

class BufferWar:
    def mev_opportunity(self, hash_b, window=141, sides=24):
        """Simulate MEV opportunity in buffer window."""
        hashes = range(hash_b - sides, hash_b + sides + 1)
        return [h for h in hashes if h > 3 and h < 145]

    def profitable_arbitrage(self, target, current, volume):
        """Check profitability with martingale factor."""
        return checkProfitable(target, current, volume)

class WeavingUtils:
    def modulate_encode_sequence(self, data, grid_dim=GRID_DIM, float_length=3):
        """Modulate encode sequence with moving heddles and float watermark."""
        chars = list(data)
        heddle_lifts = np.zeros((grid_dim, 3), dtype=int)
        for i, c in enumerate(chars):
            plane = i % 3
            row = ord(c) % grid_dim
            heddle_lifts[row, plane] = 1
        watermarked = ''.join(c + '0' * (float_length - 1) + (d if i + float_length < len(data) else '')
                             for i, (c, d) in enumerate(zip(data[::float_length], data[float_length::float_length])))
        encode = ''.join(c + str(int(heddle_lifts[i % grid_dim, i % 3])) for i, c in enumerate(watermarked))
        return encode, heddle_lifts

class SpiralNU:
    def __init__(self):
        self.exponents = [2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281, 3217, 4253, 4423, 9689, 9941, 11213, 19937, 21701, 23209, 44497, 86243, 110503, 132049, 216091, 756839, 859433, 1257787, 1398269, 2976221, 3021377, 6972593, 13466917, 20996011, 24036583, 25964951, 30402457, 32582657, 37156667, 42643801, 43112609, 57885161, 74207281, 77232917, 82589933, 194062501]

    def kappa_prime_n(self, n: int) -> float:
        """Calculate kappa prime for a given n."""
        try:
            for i, exp in enumerate(self.exponents):
                if n < exp:
                    return 0.03563 if i % 2 == 0 else 0.3536
            return 0.3536
        except Exception as e:
            logger.error(f"Kappa prime n error: {e}")
            return 0.3536

    def vote(self, proposal_id: int, voter_weight: float) -> bool:
        """Simulate voting based on weight."""
        try:
            return voter_weight > 0.5
        except Exception as e:
            logger.error(f"Vote error: {e}")
            return False

    def poly_hash_256(self, data: str, prev_key: str = None) -> str:
        """Generate a SHA-256 hash with optional previous key."""
        try:
            return hashlib.sha256((data + (prev_key or "")).encode()).hexdigest()
        except Exception as e:
            logger.error(f"Poly hash 256 error: {e}")
            return ""

    def run_spiral(self, n_laps: int = 1000) -> List[float]:
        """Run a spiral simulation for flat theta values."""
        try:
            theta = 0
            flat_theta = []
            for _ in range(n_laps):
                theta += 369 / 360 * 2 * math.pi
                if abs(math.sin(theta)) < 0.01:
                    flat_theta.append(theta)
            return flat_theta
        except Exception as e:
            logger.error(f"Run spiral error: {e}")
            return []

    def predict_next_prime(self) -> int:
        """Predict the next Mersenne prime exponent."""
        try:
            target = 1.5 * (2 ** self.exponents[-1] - 1)
            n = self.exponents[-1] + 1
            while True:
                n += 1
                if self.is_prime(2 ** n - 1):
                    if 2 ** n - 1 > target:
                        return n
        except Exception as e:
            logger.error(f"Predict next prime error: {e}")
            return 0

    def is_prime(self, n: int) -> bool:
        """Check if a number is prime."""
        try:
            if n < 2:
                return False
            for i in range(2, int(math.sqrt(n)) + 1):
                if n % i == 0:
                    return False
            return True
        except Exception as e:
            logger.error(f"Is prime error: {e}")
            return False

    def check_m53_candidate(self):
        """Check M53 candidate exponent for Mersenne prime."""
        try:
            p = 194062501
            if not self.is_prime(p):
                return f"Exponent {p} is not prime, so 2^{p}-1 cannot be Mersenne prime."
            remainder = p % 369
            kappa_at_316 = kappa_calc(316)
            return f"Exponent {p} mod 369 = {remainder}. kappa at n=316: {kappa_at_316:.4f} (symmetry point)."
        except Exception as e:
            logger.error(f"Check M53 candidate error: {e}")
            return "M53 check failed."

class ShuttleModel:
    def __init__(self):
        self.wave_packet = np.array([np.exp(-((x - 0.5) ** 2) / 0.1) for x in np.linspace(0, 1, 100)])

    def compute_shuttle_kappa(self, x, y, t):
        """Compute curvature for the shuttle model using greenspline-based kappa."""
        try:
            if len(x) != len(y) or len(x) != len(t):
                logger.error(f"Dimension mismatch: len(x)={len(x)}, len(y)={len(y)}, len(t)={len(t)}")
                return np.array([0.02500125] * len(t))
            dx_dt = np.gradient(x, np.diff(t).mean())
            dy_dt = np.gradient(y, np.diff(t).mean())
            d2x_dt2 = np.gradient(dx_dt, np.diff(t).mean())
            d2y_dt2 = np.gradient(dy_dt, np.diff(t).mean())
            numerator = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2)
            denominator = (dx_dt**2 + dy_dt**2)**1.5 + 1e-12
            kappa = numerator / denominator
            return np.clip(kappa, 0.02, 0.03) * 0.833375  # Align with k_curves normalization
        except Exception as e:
            logger.error(f"Compute shuttle kappa error: {e}")
            return np.array([0.02500125] * len(t))

    def demo_shuttle(self, frames: int = 200) -> None:
        """Generate and save shuttle animation with constant curvature."""
        try:
            fig = plt.figure(figsize=(10, 8), facecolor='black')
            ax = fig.add_subplot(111, projection='3d')
            t = np.linspace(0, 1, 100)
            x, y, _ = compute_green_segment(t)
            if len(x) == 0:
                logger.error("Empty x array from compute_green_segment, using fallback")
                x, y = np.linspace(0, 1, 100), np.zeros(100)
            z = self.wave_packet
            t_fine = np.linspace(0, 1, len(x))
            kappa = self.compute_shuttle_kappa(x, y, t_fine)
            if len(kappa) != len(x):
                logger.warning(f"Kappa length mismatch ({len(kappa)} vs {len(x)}), resizing kappa")
                kappa = np.interp(np.linspace(0, 1, len(x)), np.linspace(0, 1, len(kappa)), kappa)
            X_blue, Y_blue, Z_blue, X_gold, Y_gold, Z_gold = create_blob_surface(x, y, z, TUBE_RADIUS, NUM_SIDES, kappa)
            light = LightSource(azdeg=315, altdeg=45)
            rgb_blue = np.ones((X_blue.shape[0], X_blue.shape[1], 3)) * [0, 0, 1]  # Blue
            rgb_gold = np.ones((X_gold.shape[0], X_gold.shape[1], 3)) * [1, 0.84, 0]  # Gold
            shaded_blue = light.shade_rgb(rgb_blue, Z_blue)
            shaded_gold = light.shade_rgb(rgb_gold, Z_gold)
            blob_surface_blue = ax.plot_surface(X_blue, Y_blue, Z_blue, facecolors=shaded_blue, edgecolor='none')
            blob_surface_gold = ax.plot_surface(X_gold, Y_gold, Z_gold, facecolors=shaded_gold, edgecolor='none')

            def update(frame):
                ax.clear()
                ax.set_xlim(-2, 2)
                ax.set_ylim(-2, 2)
                ax.set_zlim(-2, 2)
                ax.axis('off')
                ax.set_facecolor('black')
                offset = frame * 0.01
                X_blue_shifted = X_blue + offset
                X_gold_shifted = X_gold + offset
                ax.plot_surface(X_blue_shifted, Y_blue, Z_blue, facecolors=shaded_blue, edgecolor='none')
                ax.plot_surface(X_gold_shifted, Y_gold, Z_gold, facecolors=shaded_gold, edgecolor='none')
                add_light_slicks(ax)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title(f"Shuttle Animation: Frame {frame}")
                if frame % 45 == 0:
                    kappa_hash = hashlib.sha256((str(np.mean(kappa)) + str(frame)).encode()).hexdigest()[:8]
                    print(f"Mnemonic at frame {frame}: {kappa_hash}")
                return [blob_surface_blue, blob_surface_gold]

            anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
            anim.save('shuttle_animation.gif', writer='pillow', fps=20)
            print("Simulating shuttle animation... Saved as 'shuttle_animation.gif'")
        except Exception as e:
            logger.error(f"Demo shuttle error: {e}")
