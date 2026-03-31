"""
config.py — Central configuration for the Kalshi trading bot.
All risk parameters are SAFE DEFAULTS for a $100 demo account.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "output"
AUTORESEARCH_DIR = PROJECT_ROOT / "autoresearch"
OUTPUT_DIR.mkdir(exist_ok=True)
AUTORESEARCH_DIR.mkdir(exist_ok=True)

# =============================================================================
# KALSHI API
# =============================================================================
KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH", "./kalshi_private_key.pem")
KALSHI_ENV = os.getenv("KALSHI_ENV", "DEMO").upper()

# API base URLs
KALSHI_DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"
KALSHI_PROD_BASE = "https://api.elections.kalshi.com/trade-api/v2"

def get_base_url() -> str:
    if KALSHI_ENV == "PROD":
        return KALSHI_PROD_BASE
    return KALSHI_DEMO_BASE

# =============================================================================
# ANTHROPIC (Claude AI)
# =============================================================================
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-6"

# =============================================================================
# RISK MANAGEMENT — TUNED FOR $580 ACCOUNT (updated 2026-03-28)
# =============================================================================
ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", "580"))

# Maximum percentage of account per single trade (5% = $29 on $580)
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.05"))

# Maximum dollar amount per single trade ($25 = ~4.3% of $580)
MAX_BET_DOLLARS = float(os.getenv("MAX_BET_DOLLARS", "25"))

# Daily loss cap as percentage of account (8% = $46.40 on $580)
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.08"))

# Fractional Kelly criterion multiplier (0.50 = half Kelly, moderate)
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.50"))

# Maximum number of open positions at once
MAX_OPEN_POSITIONS = 20

# Minimum edge required to place a trade (estimated_prob - market_price)
MIN_EDGE_THRESHOLD = 0.05  # 5 cents minimum edge

# =============================================================================
# MARKET FILTERS — High-volume, fast-settling markets only
# =============================================================================
TARGET_MARKET_SERIES = [
    # --- WEATHER (proven profitable, keep active) ---
    "KXHIGHNY",     # NYC daily high temp (settles daily, high volume)
    "KXHIGHCHI",    # Chicago daily high temp (settles daily, high volume)
    "KXHIGHLA",     # LA daily high temp
    "KXHIGHMIA",    # Miami daily high temp
    "KXHIGHDC",     # Washington DC daily high temp
    "KXHIGHDEN",    # Denver daily high temp

    # --- NBA (only KXNBAGAME for underdog strategy) ---
    "KXNBAGAME",     # NBA individual games — underdog YES strategy only (+129% ROI)

    # --- DISABLED LOSING STRATEGIES (2026-03-28 kill switch) ---
    # "KXNBA",       # DISABLED: NBA Extreme NO — -$38.75, -86% ROI, 20% WR. Do not re-enable.
    # "KXNHL",       # DISABLED: No proven edge, removed to reduce noise.
    # "KXMLB",       # DISABLED: No proven edge, removed to reduce noise.
    # "KXNCAAB",     # DISABLED: No proven edge, removed to reduce noise.
    # "KXBTC",       # DISABLED: Crypto directional — -$6.95, -70% ROI, 15% WR. No real-time price feeds.
    # "KXETH",       # DISABLED: Crypto directional — same as BTC, no edge without live data.
    # "KXSOL",       # DISABLED: Crypto directional — same as BTC, no edge without live data.
    # "KXNBAPTS",    # DISABLED: NBA player props — -$18.96, -68% ROI. No player data = no edge.
    # "KXMARMAD",    # DISABLED: March Madness — season over, no proven edge.
]

# Series explicitly BLOCKED from all trading paths (arb, autoresearch, deep ITM, etc.)
# These will be filtered out even if they appear in broad market scans.
# --- PERMANENT KILL LIST (2026-03-28) ---
BLOCKED_SERIES = {
    "KXNBAPTS",      # NBA player props — -$18.96, -68% ROI. No player data = no edge.
    "KXNBA",         # NBA Extreme NO — -$38.75, -86% ROI, 20% WR. Pennies in front of steamrollers.
    "KXBTC",         # Crypto directional — -$6.95, -70% ROI, 15% WR. No real-time price feeds.
    "KXETH",         # Crypto directional — no edge without live price data.
    "KXSOL",         # Crypto directional — no edge without live price data.
    "KXNHL",         # No proven edge. Removed to reduce noise and avoid leaking capital.
    "KXMLB",         # No proven edge. Removed to reduce noise and avoid leaking capital.
    "KXNCAAB",       # No proven edge. Removed to reduce noise and avoid leaking capital.
    "KXMARMAD",      # No proven edge. Season over.
}

# =============================================================================
# ARB STRATEGY — DISABLED BY DEFAULT (2026-03-28)
# =============================================================================
# The YES/NO arb strategy found 0 true arbs in 5,500+ scan cycles.
# Kalshi spreads are too wide for risk-free arb: best_ask(YES) + best_ask(NO)
# is almost always >= 100c after fees. When the scanner DID signal, it was
# using stale mid/last prices, not actual ask prices. This caused one-sided
# fills that created unintended directional exposure instead of risk-free arb.
#
# Set to True to re-enable if Kalshi liquidity improves significantly.
ARB_STRATEGY_ENABLED = False

# Only trade markets with volume above this threshold
MIN_MARKET_VOLUME = 50

# Only trade markets settling within this many hours
MAX_HOURS_TO_SETTLEMENT = 720  # 30 days — filter less aggressively to find active markets

# =============================================================================
# AUTORESEARCH SETTINGS
# =============================================================================
AUTORESEARCH_MAX_ITERATIONS = int(os.getenv("AUTORESEARCH_MAX_ITERATIONS", "50"))
AUTORESEARCH_TARGET_SHARPE = float(os.getenv("AUTORESEARCH_TARGET_SHARPE", "1.2"))
AUTORESEARCH_MAX_DRAWDOWN = float(os.getenv("AUTORESEARCH_MAX_DRAWDOWN", "0.08"))

# =============================================================================
# KALSHI FEE FORMULA (from their docs)
# =============================================================================
# fee = ceil(rate * contracts * price * (1-price))
# maker_rate = 0.0175, taker_rate = 0.07
KALSHI_MAKER_FEE_RATE = 0.0175
KALSHI_TAKER_FEE_RATE = 0.07

def kalshi_fee_cents(contracts: int, price_cents: int, is_maker: bool = True) -> float:
    """Calculate Kalshi fee in cents. Price in cents (1-99)."""
    import math
    rate = KALSHI_MAKER_FEE_RATE if is_maker else KALSHI_TAKER_FEE_RATE
    p = price_cents / 100.0
    return math.ceil(rate * contracts * p * (1 - p) * 100) / 100

# Keep backward-compatible constants (approximate, for simple calculations)
KALSHI_FEE_PER_SIDE_CENTS = 0.5  # avg maker fee at typical prices
KALSHI_FEE_ROUND_TRIP_CENTS = 1.0
KALSHI_FEE_PER_CONTRACT_DOLLARS = 0.01

# =============================================================================
# DISPLAY
# =============================================================================
CHART_THEME = "plotly_dark"
TABLE_FORMAT = "fancy_grid"
