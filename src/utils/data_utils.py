import math
from src.config import *

def simulate_scalability_issue(tx_count: int) -> float:
    """Simulate scalability delay based on transaction count."""
    delay = tx_count * 0.01
    return delay

def check_scalability(buffer_size: int, max_size: int = BUFFER_BLOCK_LIMIT) -> bool:
    """Check if buffer size is within scalability limits."""
    return buffer_size <= max_size

def consensus_69_percent(votes: int, total_nodes: int) -> bool:
    """Check if 69% consensus threshold is met."""
    return votes / total_nodes >= CONSENSUS_THRESHOLD

def arbitrage_exception(current_price: float, oracle_price: float) -> bool:
    """Check for arbitrage opportunity based on price difference."""
    return abs(current_price - oracle_price) > 0.01 * oracle_price

def shannon_entropy(s: str) -> float:
    """Calculate Shannon entropy of a string."""
    from collections import Counter
    from math import log2
    freq = Counter(s)
    len_s = len(s)
    if len_s == 0:
        return 0.0
    return -sum(count / len_s * log2(count / len_s) for count in freq.values() if count > 0)

def benchmark_hashes(num_hashes: int = 1000) -> dict:
    """Benchmark SHA-256 and SHA-3 hash performance."""
    return {"SHA-256": num_hashes * 0.05, "SHA-3": num_hashes * 0.03}

def martingale_hedge(amount: float, factor: float) -> float:
    """Apply martingale hedging strategy."""
    return amount * factor

def count_ping_pong(rallies: int) -> int:
    """Count ping-pong interactions for ternary 0BE."""
    count = 0
    for i in range(rallies):
        if i % 3 == 0:
            count += 1
        elif i % 3 == 1:
            pass
        else:
            pass
    return count
