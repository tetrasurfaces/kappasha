import logging

logger = logging.getLogger(__name__)

class HashSimulator:
    def __init__(self):
        self.sha256_energy = 0.05
        self.sha3_energy = 0.03

    def compare_hashes(self, num_hashes: int = 1000) -> dict:
        """Compare SHA-256 and SHA-3 energy usage."""
        try:
            return {"SHA-256": num_hashes * self.sha256_energy, "SHA-3": num_hashes * self.sha3_energy}
        except Exception as e:
            logger.error(f"Hash comparison error: {e}")
            return {}
