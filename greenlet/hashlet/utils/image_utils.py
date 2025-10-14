import base64
import logging
from typing import Any

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self, sha1664: Any, bastion: Any, grokwalk: Any):
        """Initialize with hash, bastion, and grokwalk instances for potential integration."""
        self.sha1664 = sha1664
        self.bastion = bastion
        self.grokwalk = grokwalk

    def process_image(self, data: bytes) -> str:
        """Process image data by encoding it to base64 and optionally hashing it."""
        try:
            encoded = base64.b64encode(data).decode('utf-8')
            # Optional: Add hash integration
            hash_str = self.sha1664.hash_transaction(encoded) if hasattr(self.sha1664, 'hash_transaction') else encoded
            return hash_str
        except Exception as e:
            logger.error(f"Image process error: {e}")
            return ""

def encode_image(data: bytes) -> str:
    """Encode image data to base64 string."""
    try:
        return base64.b64encode(data).decode('utf-8')
    except Exception as e:
        logger.error(f"Image encode error: {e}")
        return ""

def decode_image(encoded: str) -> bytes:
    """Decode base64 string to image data."""
    try:
        return base64.b64decode(encoded)
    except Exception as e:
        logger.error(f"Image decode error: {e}")
        return b""
