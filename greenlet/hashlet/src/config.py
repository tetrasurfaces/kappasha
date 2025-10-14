# Configuration settings and constants
import mpmath

DB_FILE = "blockchan.db"
TRADING_PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "~ESC/USDT", "LLLP/GMEx"]
WIDGETS = ["order_book", "price_panel", "portfolio", "swap", "predictions", "bot-swapper"]
CONSENSUS_THRESHOLD = 0.67
MAX_NODES = 11
MIN_NODES_FOR_GENESIS = 2
UPDATE_INTERVAL = 10
GOSSIP_TIMEOUT = 0.1
GRID_DIM = 2141  # Updated to 2141 with +1 Genesis
BUFFER_BLOCK_LIMIT = 2141
KAPPA_BASE = 0.3536
PHI_FLOAT = float(mpmath.phi)
TICK_SPACING = 0.01
FEE_RATE = 0.003
MARTINGALE_FACTOR = 2.0
FOLD_COUNT = 5
CHANNELS = ["Gossip", "Consensus", "Bastions"]
NUM_SIDES = 64
SPARSE_N = 50

# Hashes (to be validated against content files)
GREEN_TXT_HASH = "b07cfffde1cc3640d7539263944d7a644ccf96622d9d38877ca4bb4549d3c8ef"
HYBRID_CY_HASH = "57cbf50bcab57db2a0a573d5d03c961ff0f007f8926aa8a2a8f901c3c61299ce"
GREENSPLINE_HASH = "b804dac9da90257df5afa686c26f9ed7652234509d131a352c46987569ecb052"
HYBRID_HASH = "eeb1f3b7461cd43feec2ed15f5fdb75fd3e869fb6f58a1c928b62d7b69e09a27"
GREEN_PARSER_HASH = "0f3c41d2d0bc7665f1cf6825db76cc3d8886b0030c8a9c9a29103b21b724556a"
JITHOOK_HASH = "00dcb0c31f58ec0626e201dd327c7dea0cb409044c6a6021acc87ce85a13f297"
LIBRS_RUST_HASH = "872e01fbc3d0a5194bcea0601105966d7dfc87523f6d9f3773ad23402fb28819"

# Sample Data
SAMPLE_DATA = [
    {'date': '08/13/2025', 'open': 191.710, 'high': 204.880, 'low': 191.490, 'close': 201.450, 'vol': 7150000},
    {'date': '08/14/2025', 'open': 201.530, 'high': 209.790, 'low': 187.550, 'close': 192.480, 'vol': 8970000},
    {'date': '08/15/2025', 'open': 192.460, 'high': 198.040, 'low': 183.380, 'close': 185.780, 'vol': 4360000},
]

# Constants from GREENSPLINE_CONTENT
TUBE_RADIUS = 0.05
NUM_SIDES = 64
