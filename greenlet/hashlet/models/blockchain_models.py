import logging
import random  # Kept for potential future use, though replaced in ExperienceRamp
from src.config import *

logger = logging.getLogger(__name__)

class GreedyFillSimulator:
    def __init__(self, target: float):
        self.target = target
        self.total_filled = 0.0
        self.total_fees = 0.0

    def fill_order(self, amount: float, fee_rate: float = FEE_RATE, martingale_factor: float = MARTINGALE_FACTOR) -> bool:
        """Simulate filling an order with martingale weighting."""
        try:
            weighted_amount = amount * martingale_factor
            if self.total_filled + weighted_amount > self.target:
                return False
            self.total_filled += weighted_amount
            self.total_fees += weighted_amount * fee_rate
            return True
        except Exception as e:
            logger.error(f"Fill order error: {e}")
            return False

class PerpLib:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.liquidity_amount = 0.0
        self.interest_rate = 0.05
        self.orders = []

    def add_order(self, price: float, volume: float):
        """Add an order to the perpetual market."""
        try:
            self.orders.append({'price': price, 'volume': volume})
            self.liquidity_amount += volume
            logger.info(f"Added order: {volume} {self.symbol} @ {price}")
        except Exception as e:
            logger.error(f"Add order error: {e}")

    def get_status(self) -> dict:
        """Get the status of the perpetual market."""
        try:
            return {'symbol': self.symbol, 'liquidity_amount': self.liquidity_amount, 'orders': len(self.orders)}
        except Exception as e:
            logger.error(f"Get status error: {e}")
            return {}

class OpsPool:
    def __init__(self):
        self.Ops_pool = []

    def add_Ops(self, ops_instance: PerpLib):
        """Add a PerpLib instance to the operations pool."""
        try:
            self.Ops_pool.append(ops_instance)
            logger.info(f"Added Ops instance for {ops_instance.symbol}")
        except Exception as e:
            logger.error(f"Add Ops error: {e}")

    def redistribute_liquidity(self, redistribution_factor: float = 0.1):
        """Redistribute liquidity across the operations pool."""
        try:
            total_liquidity = sum(r.liquidity_amount for r in self.Ops_pool)
            redistributed_amount = total_liquidity * redistribution_factor
            for ops_instance in self.Ops_pool:
                ops_instance.liquidity_amount += redistributed_amount / len(self.Ops_pool)
            logger.info(f"Redistributed {redistributed_amount} liquidity")
        except Exception as e:
            logger.error(f"Redistribute liquidity error: {e}")

    def calculate_global_interest_rate(self) -> float:
        """Calculate the global interest rate for the pool."""
        try:
            if not self.Ops_pool:
                return 0.0
            return sum(r.interest_rate for r in self.Ops_pool) / len(self.Ops_pool)
        except Exception as e:
            logger.error(f"Calculate global interest rate error: {e}")
            return 0.0

    def get_pool_status(self) -> dict:
        """Get the status of the operations pool."""
        try:
            pool_status = [r.get_status() for r in self.Ops_pool]
            return {
                "Total Liquidity": sum(r["liquidity_amount"] for r in pool_status),
                "Global Interest Rate": self.calculate_global_interest_rate(),
                "Pool Details": pool_status
            }
        except Exception as e:
            logger.error(f"Get pool status error: {e}")
            return {}

    def compute_perpetual_strategy(self, position_size: float, volatility: float, skew: float, funding_rate: float) -> float:
        """Compute a perpetual strategy based on market parameters."""
        try:
            result = position_size * (1 + skew) / (1 + volatility + funding_rate)
            logger.info(f"Computed strategy: {result}")
            return result
        except Exception as e:
            logger.error(f"Compute perpetual strategy error: {e}")
            return 0.0

class ExperienceRamp:
    def __init__(self, ramp_length=50):  # Updated to match 50 TOC demos
        self.ramp = [2 ** i for i in range(ramp_length)]  # 2^0, 2^1, ..., 2^49
        self.kappa = 1.0

    def curve_monster_threat(self, player_exp, max_exp, base_threat=100):
        """Simulate a monster threat level based on player experience."""
        try:
            normalized_exp = player_exp / max_exp
            threat = base_threat
            for i, factor in enumerate(self.ramp):
                if random.random() < factor * self.kappa * normalized_exp:  # Still uses random for variability
                    threat *= (1 - 0.1 * (i + 1) / len(self.ramp))
            return max(0, threat)
        except Exception as e:
            logger.error(f"Curve monster threat error: {e}")
            return 0.0
