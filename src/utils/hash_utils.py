import hashlib
import mpmath
import numpy as np
import logging
from src.config import GRID_DIM  # Import GRID_DIM specifically

logger = logging.getLogger(__name__)

class HashUtils:
    def __init__(self, grid_dim=GRID_DIM):
        self.grid_dim = grid_dim

    def sha1664(self, indexed_hash):
        """Generate SHA1664 hash with sponge permutations."""
        base_hash = hashlib.sha512(str(indexed_hash).encode()).digest()
        mp_state = mpmath.mpf(int(base_hash.hex(), 16))
        for _ in range(4):
            mp_state = mpmath.sqrt(mp_state) * mpmath.phi
        partial = mpmath.nstr(mp_state, 1664 // 4)
        return hashlib.sha256(partial.encode()).hexdigest(), int(mpmath.log(mp_state, 2))

    def advanced_hash(self, seed, bits=16, laps=18):
        """Generate advanced hash with 18-lap reversals."""
        mask = (1 << bits) - 1
        original = seed & mask
        reverse = (~original) & mask
        left_seq = np.arange(1, laps + 1)
        right_seq = -left_seq
        for lap in range(0, laps, 3):
            left_seq[lap:lap+3] = left_seq[lap:lap+3][::-1]
            right_seq[lap:lap+3] = -right_seq[lap:lap+3][::-1]
        pos_index = sum(left_seq[i % laps] * ((original >> i) & 1) for i in range(bits))
        neg_index = sum(right_seq[i % laps] * ((reverse >> i) & 1) for i in range(bits))
        total_index = pos_index + neg_index
        theta_flat = int(total_index * 0.3536 * mpmath.phi) % 180
        if theta_flat != 0:
            total_index = (total_index // 180) * 180
        return total_index & ((1 << (bits * 2)) - 1), pos_index, neg_index

    def m53_collapse(self, p, stake=1):
        """M53 collapse profit check (converted from MIT green_profit.py)."""
        MOD_BITS = 256
        MOD_SYM = 369
        DIVISOR = 3
        mod_bits = p % MOD_BITS
        mod_sym = p % MOD_SYM
        risk_approx = (1 << mod_bits) - 1
        sym_factor = mod_sym // DIVISOR
        risk_collapsed = risk_approx * sym_factor
        reward = risk_collapsed * stake // DIVISOR
        entropy_bits = int(math.log2(reward)) if reward > 0 else 0
        return reward, entropy_bits
