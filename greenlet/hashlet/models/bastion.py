import logging
from typing import Dict  # Add this import for Dict type hint

logger = logging.getLogger(__name__)

class SHA1664:
    def __init__(self):
        self.hash_string = ""
        self.folds = 0

    def hash_transaction(self, data: str) -> str:
        """Generate SHA-256 hash with folding for indexing."""
        try:
            hash_obj = hashlib.sha256(data.encode())
            self.hash_string = hash_obj.hexdigest()
            self.folds += 1
            return self.hash_string
        except Exception as e:
            logger.error(f"Hash transaction error: {e}")
            return ""

    def prevent_double_spending(self, tx_id: str) -> bool:
        """Check for double-spending using transaction cache."""
        try:
            return tx_id not in transaction_cache
        except Exception as e:
            logger.error(f"Prevent double spending error: {e}")
            return False

    def receive_gossip(self, data: Dict, sender: str):
        """Receive and log gossip data from a sender."""
        try:
            logger.info(f"Received gossip from {sender}: {data}")
        except Exception as e:
            logger.error(f"Receive gossip error: {e}")

class EphemeralBastion:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.ternary_state = 0  # 0: pong, 1: ping, e: earth

    def set_ternary_state(self, state: any):
        """Set the ternary state for the bastion."""
        self.ternary_state = state

    def validate(self, data: str) -> bool:
        """Validate data based on length."""
        try:
            return len(data) > 0
        except Exception as e:
            logger.error(f"Validate error: {e}")
            return False
