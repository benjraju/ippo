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
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# =============================================================================
# RISK MANAGEMENT — SAFE DEFAULTS FOR $100 ACCOUNT
# =============================================================================
ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", "100"))

# Maximum percentage of account per single trade (2% = $2 on $100)
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.02"))

# Maximum dollar amount per single trade
MAX_BET_DOLLARS = float(os.getenv("MAX_BET_DOLLARS", "2"))

# Daily loss cap as percentage of account (8% = $8 on $100)
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.08"))

# Fractional Kelly criterion multiplier (0.25 = quarter Kelly, very conservative)
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))

# Maximum number of open positions at once
MAX_OPEN_POSITIONS = 5

# Minimum edge required to place a trade (estimated_prob - market_price)
MIN_EDGE_THRESHOLD = 0.05  # 5 cents minimum edge

# =============================================================================
# MARKET FILTERS — High-volume, fast-settling markets only
# =============================================================================
TARGET_MARKET_SERIES = [
    "KXHIGH",       # Daily weather highs (settles daily)
    "KXLOW",        # Daily weather lows (settles daily)
    "NCAA",          # Men's College Basketball
    "NBAMVP",        # NBA markets
    "BTC",           # Bitcoin hourly/daily
    "ETH",           # Ethereum hourly/daily
    "INX",           # S&P 500 range markets
    "NASDAQ",        # Nasdaq range markets
]

# Only trade markets with volume above this threshold
MIN_MARKET_VOLUME = 50

# Only trade markets settling within this many hours
MAX_HOURS_TO_SETTLEMENT = 48

# =============================================================================
# AUTORESEARCH SETTINGS
# =============================================================================
AUTORESEARCH_MAX_ITERATIONS = int(os.getenv("AUTORESEARCH_MAX_ITERATIONS", "50"))
AUTORESEARCH_TARGET_SHARPE = float(os.getenv("AUTORESEARCH_TARGET_SHARPE", "1.2"))
AUTORESEARCH_MAX_DRAWDOWN = float(os.getenv("AUTORESEARCH_MAX_DRAWDOWN", "0.08"))

# =============================================================================
# DISPLAY
# =============================================================================
CHART_THEME = "plotly_dark"
TABLE_FORMAT = "fancy_grid"
