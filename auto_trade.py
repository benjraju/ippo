"""
auto_trade.py -- Fully autonomous daily trading script.

Runs without human intervention. Active strategies:

  1. NBA Underdog  -- buy YES on KXNBAGAME underdogs priced 10-30c (proven edge)
  2. Arb           -- buy both YES+NO when yes+no < $1.00 (mathematical guarantee)
  3. Dutch Book    -- sell YES on all legs when sum(yes_bid) > 102c (guaranteed profit)
  4. Weather Tail  -- buy NO on cheap YES weather markets (proven edge)
  5. Deep ITM      -- buy YES at 95c on near-certain markets (small but consistent)

SAFETY:
  - Balance check before EVERY order
  - 8% daily loss cap
  - Max $2 per trade
  - Limit orders only -- never market orders
  - Every decision logged to output/auto_trade_YYYY-MM-DD.log

Usage:
    python cli.py auto-trade --dry-run   # show what it would do
    python cli.py auto-trade --live       # actually place orders

    # Or standalone:
    python auto_trade.py --dry-run
    python auto_trade.py --live
"""

import json
import logging
import math
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import requests

import config
try:
    from alerts import alert_trade_settled, alert_drawdown, alert_bot_error, alert_big_edge, alert_daily_summary
except (ImportError, OSError):
    alert_trade_settled = alert_drawdown = alert_bot_error = alert_big_edge = alert_daily_summary = None
from kalshi_client import KalshiClient
from risk_manager import RiskManager, TradeProposal
from weather_strategy import (
    NWS_GRID_POINTS,
    normal_cdf,
    calc_bucket_probability,
    calc_above_probability,
    calc_below_probability,
    parse_market_type,
)

try:
    from arb_scanner import ArbScanner, ArbOpportunity
    from market_scanner import MarketScanner
except (ImportError, OSError):
    ArbScanner = None
    MarketScanner = None

try:
    from brier_tracker import BrierTracker
    _brier = BrierTracker()
except ImportError:
    _brier = None

try:
    from weather_tail_strategy import find_weather_tail_trades, tail_risk_budget
except ImportError:
    find_weather_tail_trades = None

try:
    from nba_underdog_strategy import find_nba_underdogs, nba_risk_budget
except Exception as _nba_import_err:
    logging.getLogger("auto_trade").error(
        "NBA underdog strategy import FAILED: %s: %s",
        type(_nba_import_err).__name__, _nba_import_err,
    )
    import traceback as _tb
    logging.getLogger("auto_trade").debug(_tb.format_exc())
    find_nba_underdogs = None
    nba_risk_budget = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# City coordinates for Open-Meteo ensemble API (lat, lon)
CITY_COORDS = {
    "KXHIGHNY":  (40.7829, -73.9654),   # NYC Central Park
    "KXHIGHCHI": (41.7868, -87.7522),   # Chicago Midway
    "KXHIGHMIA": (25.7933, -80.2906),   # Miami Intl Airport
    "KXHIGHLA":  (33.9425, -118.4081),  # Los Angeles LAX
    "KXHIGHDC":  (38.8512, -77.0402),   # Washington DC -- Reagan National
    "KXHIGHDEN": (39.8561, -104.6737),  # Denver Intl Airport
}

CITY_NAMES = {
    "KXHIGHNY":  "NYC",
    "KXHIGHCHI": "Chicago",
    "KXHIGHMIA": "Miami",
    "KXHIGHLA":  "LA",
    "KXHIGHDC":  "DC",
    "KXHIGHDEN": "Denver",
}

# Forecast blend weights by days_out (HRRR + GFS ensemble + NWS official = 1.0)
BLEND_WEIGHTS = {
    0: {"hrrr": 0.40, "gfs": 0.30, "nws": 0.30},   # Day 0: HRRR dominant
    1: {"hrrr": 0.25, "gfs": 0.40, "nws": 0.35},   # Day 1: GFS takes over
    2: {"hrrr": 0.00, "gfs": 0.60, "nws": 0.40},   # Day 2+: no HRRR
}

# Legacy blend weights (used as fallback when HRRR unavailable)
NWS_WEIGHT = 0.40
ENSEMBLE_WEIGHT = 0.60

# Trading limits
WEATHER_EDGE_THRESHOLD_CENTS = 3.0
BTC_EDGE_THRESHOLD_CENTS = 4.0
MAX_DOLLARS_PER_TRADE = 15.0
MAX_DAILY_DEPLOY_PCT = 0.50      # 50% of account for weather
MAX_SPORTS_TRADES_PER_DAY = 15
SPORTS_EDGE_THRESHOLD_CENTS = 5.0
MAX_ARB_TRADES_PER_DAY = 25
ARB_EDGE_THRESHOLD_CENTS = 3.0   # Lower threshold -- arb has built-in edge
DAILY_LOSS_CAP_PCT = 0.08        # 8% of account


# ---------------------------------------------------------------------------
# NBA game dedup helper
# ---------------------------------------------------------------------------

def get_nba_game_id(ticker: str) -> str:
    """Extract game ID from NBA ticker. E.g., KXNBAGAME-26MAR25OKCBOS-BOS -> KXNBAGAME-26MAR25OKCBOS"""
    if "KXNBAGAME" in ticker:
        parts = ticker.rsplit("-", 1)
        if len(parts) == 2:
            return parts[0]
    return ""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(dry_run: bool = True) -> logging.Logger:
    """Create a logger that writes to both console and daily log file."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_dir = config.OUTPUT_DIR
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"auto_trade_{today}.log"

    logger = logging.getLogger("auto_trade")
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicates on re-run
    logger.handlers.clear()

    # File handler -- everything
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)-7s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler -- INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    mode_label = "DRY-RUN" if dry_run else "LIVE"
    logger.info("=" * 60)
    logger.info(f"AUTO-TRADE SESSION START  [{mode_label}]  env={config.KALSHI_ENV}")
    logger.info(f"Log file: {log_path}")
    logger.info("=" * 60)
    return logger


# ---------------------------------------------------------------------------
# Dynamic strategy parameter loading
# ---------------------------------------------------------------------------

def load_strategy_params():
    """Load evolved parameters from candidate_strategy.py, with safe defaults.

    Returns a dict with ALL tunable params. auto_trade.py should use these
    instead of its own hardcoded constants so that AutoResearch mutations
    actually take effect in live trading.
    """
    try:
        import importlib
        # Force reimport to get latest values
        for mod in list(sys.modules.keys()):
            if "candidate_strategy" in mod:
                del sys.modules[mod]
        from autoresearch.candidate_strategy import (
            EDGE_THRESHOLD_CENTS,
            CONTRACTS_PER_TRADE,
            NWS_OFFICIAL_WEIGHT,
            MAX_POSITION_DOLLARS,
            MIN_VOLUME,
            BUCKET_MULTIPLIER,
            THRESHOLD_MULTIPLIER,
            HIGH_CONFIDENCE_EDGE,
            MEDIUM_CONFIDENCE_EDGE,
            TIGHT_ENSEMBLE_THRESHOLD,
            TIGHT_ENSEMBLE_MULTIPLIER,
            CRYPTO_EDGE_THRESHOLD_CENTS as _CRYPTO_EDGE,
            SPORTS_EDGE_THRESHOLD_CENTS as _SPORTS_EDGE,
            ARB_EDGE_THRESHOLD_CENTS as _ARB_EDGE,
            EXIT_EDGE_THRESHOLD_CENTS as _EXIT_EDGE,
            TAIL_FADE_MAX_PRICE,
            TAIL_FADE_MIN_VOLUME,
            TAIL_FADE_WEATHER_ENABLED,
            TAIL_FADE_CRYPTO_ENABLED,
            TAIL_FADE_NBA_ENABLED,
            TAIL_FADE_MID_LOW,
            TAIL_FADE_MID_HIGH,
            get_city_weights,
            get_forecast_stdev,
            get_blend_weights,
        )
        return {
            # Weather edge & sizing
            "edge_threshold": EDGE_THRESHOLD_CENTS,
            "contracts_per_trade": CONTRACTS_PER_TRADE,
            "nws_weight": NWS_OFFICIAL_WEIGHT,
            "max_position": MAX_POSITION_DOLLARS,
            "min_volume": MIN_VOLUME,
            "city_weights": get_city_weights(),
            "forecast_stdev": get_forecast_stdev(),
            # Market type multipliers
            "bucket_multiplier": BUCKET_MULTIPLIER,
            "threshold_multiplier": THRESHOLD_MULTIPLIER,
            # Confidence tiers
            "high_confidence_edge": HIGH_CONFIDENCE_EDGE,
            "medium_confidence_edge": MEDIUM_CONFIDENCE_EDGE,
            # Ensemble tightness
            "tight_ensemble_threshold": TIGHT_ENSEMBLE_THRESHOLD,
            "tight_ensemble_multiplier": TIGHT_ENSEMBLE_MULTIPLIER,
            # Per-strategy edge thresholds
            "crypto_edge_threshold": _CRYPTO_EDGE,
            "sports_edge_threshold": _SPORTS_EDGE,
            "arb_edge_threshold": _ARB_EDGE,
            "exit_edge_threshold": _EXIT_EDGE,
            # Blend weights
            "blend_weights": get_blend_weights(),
            # Tail fade params
            "tail_fade_max_price": TAIL_FADE_MAX_PRICE,
            "tail_fade_min_volume": TAIL_FADE_MIN_VOLUME,
            "tail_fade_weather_enabled": TAIL_FADE_WEATHER_ENABLED,
            "tail_fade_crypto_enabled": TAIL_FADE_CRYPTO_ENABLED,
            "tail_fade_nba_enabled": TAIL_FADE_NBA_ENABLED,
            "tail_fade_mid_low": TAIL_FADE_MID_LOW,
            "tail_fade_mid_high": TAIL_FADE_MID_HIGH,
        }
    except Exception:
        return {
            # Weather edge & sizing
            "edge_threshold": WEATHER_EDGE_THRESHOLD_CENTS,
            "contracts_per_trade": 10,
            "nws_weight": 0.40,
            "max_position": MAX_DOLLARS_PER_TRADE,
            "min_volume": 10,
            "city_weights": {"NYC": 1.0, "Chicago": 0.8, "Miami": 1.0, "LA": 1.0, "DC": 1.0, "Denver": 1.0},
            "forecast_stdev": {0: 1.5, 1: 2.5, 2: 3.5, 3: 4.5},
            # Market type multipliers
            "bucket_multiplier": 1.0,
            "threshold_multiplier": 1.0,
            # Confidence tiers
            "high_confidence_edge": 7.0,
            "medium_confidence_edge": 5.0,
            # Ensemble tightness
            "tight_ensemble_threshold": 2.0,
            "tight_ensemble_multiplier": 1.5,
            # Per-strategy edge thresholds
            "crypto_edge_threshold": BTC_EDGE_THRESHOLD_CENTS,
            "sports_edge_threshold": SPORTS_EDGE_THRESHOLD_CENTS,
            "arb_edge_threshold": ARB_EDGE_THRESHOLD_CENTS,
            "exit_edge_threshold": EXIT_EDGE_THRESHOLD_CENTS,
            # Blend weights
            "blend_weights": dict(BLEND_WEIGHTS),
            # Tail fade params (safe defaults)
            "tail_fade_max_price": 5,
            "tail_fade_min_volume": 0,
            "tail_fade_weather_enabled": 1,
            "tail_fade_crypto_enabled": 1,
            "tail_fade_nba_enabled": 1,
            "tail_fade_mid_low": 30,
            "tail_fade_mid_high": 50,
        }


# ---------------------------------------------------------------------------
# Data classes for trade decisions
# ---------------------------------------------------------------------------

@dataclass
class TradeDecision:
    """Records every decision (trade or skip) for the log."""
    ticker: str
    action: str               # "buy_yes", "buy_no", "skip"
    strategy: str             # "weather" or "btc"
    edge_cents: float
    fair_value_cents: float
    market_price_cents: float
    price_to_pay_cents: int   # limit price
    contracts: int
    max_loss_dollars: float
    reason: str
    placed: bool = False
    order_id: str = ""
    error: str = ""
    forecast_temp: Optional[float] = None  # weather only: blended forecast F


# ---------------------------------------------------------------------------
# Weather forecast fetching (used by exit check + weather_tail_strategy)
# ---------------------------------------------------------------------------

def fetch_nws_forecast(series_ticker: str, logger: logging.Logger) -> dict:
    """
    Fetch NWS official forecast for a city.
    Returns {date_str: temperature_F} mapping.
    """
    grid = NWS_GRID_POINTS.get(series_ticker)
    if not grid:
        logger.debug(f"NWS: no grid point for {series_ticker}")
        return {}

    office, x, y = grid
    url = f"https://api.weather.gov/gridpoints/{office}/{x},{y}/forecast"

    try:
        headers = {"User-Agent": "kalshi-auto-trade/1.0 (contact: bot@example.com)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        result = {}
        for p in data.get("properties", {}).get("periods", []):
            if p.get("isDaytime", False):
                start = p.get("startTime", "")
                if start:
                    date_str = start[:10]
                    result[date_str] = float(p["temperature"])
        logger.debug(f"NWS {series_ticker}: got forecasts for {list(result.keys())}")
        return result
    except Exception as e:
        logger.warning(f"NWS forecast error for {series_ticker}: {e}")
        return {}


def fetch_ensemble_forecast(series_ticker: str, logger: logging.Logger) -> dict:
    """
    Fetch GFS ensemble (30 members) from Open-Meteo for a city.
    Returns {date_str: {"mean": float, "stdev": float, "members": list}} mapping.
    """
    coords = CITY_COORDS.get(series_ticker)
    if not coords:
        logger.debug(f"Ensemble: no coords for {series_ticker}")
        return {}

    lat, lon = coords
    url = (
        f"https://ensemble-api.open-meteo.com/v1/ensemble"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max"
        f"&models=gfs_seamless"
        f"&forecast_days=3"
        f"&temperature_unit=fahrenheit"
    )

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        daily = data.get("daily", {})
        dates = daily.get("time", [])

        # Collect all member columns: temperature_2m_max_member01 .. member30
        member_data = {}  # date -> list of temps
        for key, values in daily.items():
            if key.startswith("temperature_2m_max_member") and values:
                for i, date_str in enumerate(dates):
                    if i < len(values) and values[i] is not None:
                        member_data.setdefault(date_str, []).append(values[i])

        result = {}
        for date_str, temps in member_data.items():
            if len(temps) >= 5:  # Need at least 5 members for meaningful stats
                mean_temp = sum(temps) / len(temps)
                variance = sum((t - mean_temp) ** 2 for t in temps) / len(temps)
                stdev = math.sqrt(variance) if variance > 0 else 1.5
                result[date_str] = {
                    "mean": mean_temp,
                    "stdev": max(stdev, 0.5),  # Floor at 0.5 to avoid div by zero
                    "n_members": len(temps),
                }

        logger.debug(
            f"Ensemble {series_ticker}: {len(result)} dates, "
            + ", ".join(f"{d}: mean={v['mean']:.1f} stdev={v['stdev']:.2f} (n={v['n_members']})"
                        for d, v in result.items())
        )
        return result
    except Exception as e:
        logger.warning(f"Ensemble forecast error for {series_ticker}: {e}")
        return {}


def fetch_hrrr_forecast(series_ticker: str, logger: logging.Logger) -> dict:
    """
    Fetch HRRR (High-Resolution Rapid Refresh) forecast from Open-Meteo.
    HRRR covers CONUS at 3km resolution, best for 0-18 hour forecasts.
    Returns {date_str: {"temp": float}} mapping (today and possibly tomorrow).
    """
    coords = CITY_COORDS.get(series_ticker)
    if not coords:
        logger.debug(f"HRRR: no coords for {series_ticker}")
        return {}

    lat, lon = coords
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max"
        f"&models=hrrr_conus"
        f"&forecast_days=2"
        f"&temperature_unit=fahrenheit"
    )

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        temps = daily.get("temperature_2m_max", [])

        result = {}
        for i, date_str in enumerate(dates):
            if i < len(temps) and temps[i] is not None:
                result[date_str] = {"temp": temps[i]}

        logger.debug(
            f"HRRR {series_ticker}: {len(result)} dates, "
            + ", ".join(f"{d}: {v['temp']:.1f}F" for d, v in result.items())
        )
        return result
    except Exception as e:
        logger.debug(f"HRRR forecast unavailable for {series_ticker}: {e}")
        return {}


def blend_forecasts(
    nws: dict, ensemble: dict, logger: logging.Logger,
    hrrr: dict | None = None,
    strategy: dict | None = None,
) -> dict:
    """
    Blend NWS official + GFS ensemble + HRRR for each date.
    Returns {date_str: {"temp": blended_temp, "stdev": ensemble_stdev, "source": str}}.

    Blend weights vary by days_out (see BLEND_WEIGHTS):
      Day 0: 40% HRRR + 30% GFS + 30% NWS
      Day 1: 25% HRRR + 40% GFS + 35% NWS
      Day 2+: 0% HRRR + 60% GFS + 40% NWS

    If HRRR is unavailable, falls back to GFS+NWS blend with legacy weights.
    Stdev: use ensemble stdev (data-driven, not hardcoded).
    Fallback: if only one source available, use it with default stdev.

    If strategy dict is provided, uses evolved blend_weights and forecast_stdev
    from candidate_strategy.py instead of hardcoded module-level constants.
    """
    if hrrr is None:
        hrrr = {}

    # Use evolved blend weights from strategy if available, else module-level constants
    blend_wts = (strategy or {}).get("blend_weights", BLEND_WEIGHTS)
    # Use evolved forecast stdev from strategy for fallback when ensemble unavailable
    fallback_stdev = (strategy or {}).get("forecast_stdev", {0: 1.5, 1: 2.5, 2: 3.5, 3: 4.5})

    all_dates = set(list(nws.keys()) + list(ensemble.keys()) + list(hrrr.keys()))
    result = {}
    now = datetime.now(timezone.utc)

    for date_str in sorted(all_dates):
        nws_temp = nws.get(date_str)
        ens_data = ensemble.get(date_str)
        hrrr_data = hrrr.get(date_str)

        # Calculate days_out for blend weight selection
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            days_out = max(0, (dt.date() - now.date()).days)
        except Exception:
            days_out = 1

        # Determine stdev (prefer ensemble-derived, fall back to evolved strategy values)
        if ens_data is not None:
            stdev = ens_data["stdev"]
        else:
            stdev = fallback_stdev.get(min(days_out, 3), 4.0)

        # Use HRRR-aware blending when we have HRRR data for day 0 or 1
        if hrrr_data is not None and days_out <= 1:
            weights = blend_wts.get(days_out, blend_wts.get(2, BLEND_WEIGHTS[2]))
            sources = []
            weighted_sum = 0.0
            total_weight = 0.0

            # HRRR component
            weighted_sum += weights["hrrr"] * hrrr_data["temp"]
            total_weight += weights["hrrr"]
            sources.append("hrrr")

            # GFS ensemble component
            if ens_data is not None:
                weighted_sum += weights["gfs"] * ens_data["mean"]
                total_weight += weights["gfs"]
                sources.append("gfs")

            # NWS component
            if nws_temp is not None:
                weighted_sum += weights["nws"] * nws_temp
                total_weight += weights["nws"]
                sources.append("nws")

            if total_weight > 0:
                blended = weighted_sum / total_weight
                source = "+".join(sources)
            else:
                continue
        elif nws_temp is not None and ens_data is not None:
            # Legacy blend: GFS + NWS (no HRRR)
            weights = blend_wts.get(min(days_out, 2), blend_wts.get(2, BLEND_WEIGHTS[2]))
            blended = weights["nws"] * nws_temp + weights["gfs"] * ens_data["mean"]
            # Normalize since hrrr weight is 0 for day 2+ but we want gfs+nws to sum to 1
            total_w = weights["nws"] + weights["gfs"]
            blended = blended / total_w if total_w > 0 else blended
            source = "gfs+nws"
        elif nws_temp is not None:
            blended = nws_temp
            source = "nws-only"
        elif ens_data is not None:
            blended = ens_data["mean"]
            source = "gfs-only"
        else:
            continue

        result[date_str] = {"temp": blended, "stdev": stdev, "source": source}

    return result


# ---------------------------------------------------------------------------
# Account checks
# ---------------------------------------------------------------------------

def get_account_balance(client: KalshiClient, logger: logging.Logger) -> Optional[float]:
    """Fetch current account balance in dollars. Returns None on failure."""
    try:
        bal = client.get_balance()
        # Balance is returned in cents
        balance_dollars = bal.get("balance", 0) / 100.0
        logger.info(f"Account balance: ${balance_dollars:.2f}")
        return balance_dollars
    except Exception as e:
        logger.error(f"Failed to fetch balance: {e}")
        return None


def get_open_positions_value(client: KalshiClient, logger: logging.Logger) -> float:
    """Estimate total capital deployed in open positions."""
    try:
        positions = client.get_positions()
        total = 0.0
        for p in positions.get("market_positions", []):
            # Approximate deployed capital as contracts * average cost
            count = abs(int(p.get("position", 0)))
            # Use market_exposure if available, otherwise estimate
            exposure = float(p.get("market_exposure", 0)) / 100.0
            total += exposure if exposure > 0 else count * 0.50  # Assume 50c avg cost
        return total
    except Exception as e:
        logger.warning(f"Could not fetch positions: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Date parsing from ticker
# ---------------------------------------------------------------------------

MONTH_MAP = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
    "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
    "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}


def parse_ticker_date(ticker: str) -> Optional[str]:
    """
    Extract settlement date from a ticker like KXHIGHCHI-26MAR21-T63.
    Returns 'YYYY-MM-DD' or None.
    """
    match = re.search(r'(\d{2})([A-Z]{3})(\d{2})', ticker)
    if not match:
        return None
    year_short, month_str, day_str = match.group(1), match.group(2), match.group(3)
    month_num = MONTH_MAP.get(month_str)
    if not month_num:
        return None
    return f"20{year_short}-{month_num}-{day_str}"


def hours_until_date(date_str: str) -> float:
    """Hours from now until end of the given date (5 PM local ~ 22 UTC)."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(
            hour=22, minute=0, tzinfo=timezone.utc
        )
        delta = dt - datetime.now(timezone.utc)
        return max(0.1, delta.total_seconds() / 3600)
    except Exception:
        return 24.0


# ---------------------------------------------------------------------------
# Arbitrage trading session
# ---------------------------------------------------------------------------

def run_arb_session(
    client: KalshiClient,
    balance: float,
    dry_run: bool,
    logger: logging.Logger,
) -> list[TradeDecision]:
    """
    Arbitrage scanning session.
    Scans for YES/NO mispricing and cross-event arb, places both sides when found.
    """
    logger.info("--- ARB SESSION START ---")
    decisions = []

    # Load evolved strategy parameters from candidate_strategy.py
    strategy = load_strategy_params()
    arb_edge_threshold = strategy["arb_edge_threshold"]

    # Capital cap: max 20% of account balance deployed in arb trades per cycle
    ARB_CAPITAL_CAP_PCT = 0.20
    arb_capital_cap = balance * ARB_CAPITAL_CAP_PCT
    arb_capital_deployed = 0.0
    logger.info(f"Arb capital cap: ${arb_capital_cap:.2f} (20% of ${balance:.2f})")

    # Track remaining cash balance to avoid 400 errors from insufficient funds
    remaining_balance = balance

    # Only arb on series where we have proven edges or true risk-free arb
    ARB_ALLOWED_SERIES = {
        "KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHDEN", "KXHIGHDC", "KXHIGHLA",
        # KXNBAGAME removed -- NBA games handled exclusively by underdog session
        "KXBTC", "KXETH", "KXSOL",
    }

    if ArbScanner is None:
        logger.warning("arb_scanner not available, skipping arb session")
        logger.info("--- ARB SESSION DONE: 0 decisions, 0 trades ---")
        return decisions

    # 1. Scan for arb opportunities
    try:
        scanner = MarketScanner(client)
        arb = ArbScanner(client)
        markets_df = scanner.scan_all_target_series()
        opps = arb.full_scan(markets_df)
    except Exception as e:
        logger.error(f"Arb scan failed: {e}")
        logger.info("--- ARB SESSION DONE: 0 decisions, 0 trades ---")
        return decisions

    if not opps:
        logger.info("No arbitrage opportunities found")
        logger.info("--- ARB SESSION DONE: 0 decisions, 0 trades ---")
        return decisions

    logger.info(f"Found {len(opps)} arb opportunities | edge_threshold={arb_edge_threshold}c")

    trades_placed = 0
    daily_loss_cap = balance * DAILY_LOSS_CAP_PCT

    for opp in opps:
        if trades_placed >= MAX_ARB_TRADES_PER_DAY:
            logger.info(f"Arb trade cap reached ({MAX_ARB_TRADES_PER_DAY})")
            break

        # Capital cap check: stop placing arb trades once we've deployed enough
        if arb_capital_deployed >= arb_capital_cap:
            logger.info(f"Arb capital cap reached: ${arb_capital_deployed:.2f} >= ${arb_capital_cap:.2f}")
            break

        # Filter: only arb on allowed series
        opp_series = opp.ticker.split("-")[0] if hasattr(opp, "ticker") and "-" in opp.ticker else ""
        if opp_series and opp_series not in ARB_ALLOWED_SERIES:
            continue

        # Skip NBA game winners -- these should ONLY be traded by the NBA underdog session
        if hasattr(opp, "ticker") and opp.ticker.startswith("KXNBAGAME"):
            logger.debug(f"  SKIP ARB {opp.ticker}: NBA games handled by underdog session only")
            continue

        # Only trade high-confidence, high-edge opportunities
        if opp.edge_cents < arb_edge_threshold:
            continue
        if opp.confidence < 0.6:
            logger.debug(f"  SKIP ARB {opp.ticker}: low confidence {opp.confidence:.0%}")
            continue

        # For yes_no_arb: buy BOTH yes and no to lock in guaranteed profit
        if opp.type == "yes_no_arb":
            yes_price = opp.details.get("yes_price", 0)
            no_price = opp.details.get("no_price", 0)
            total_cost_cents = yes_price + no_price
            cost_per_pair = total_cost_cents / 100.0  # dollars per pair

            # Fee calculation: Kalshi charges ~1.75% * price * (1-price) per side
            # For a YES+NO arb we pay fees on BOTH sides
            MAKER_FEE_RATE = 0.0175
            yes_fee_cents = MAKER_FEE_RATE * yes_price * (100 - yes_price) / 100.0
            no_fee_cents = MAKER_FEE_RATE * no_price * (100 - no_price) / 100.0
            total_fee_cents = yes_fee_cents + no_fee_cents
            profit_after_fees_cents = (100 - total_cost_cents) - total_fee_cents

            if profit_after_fees_cents <= 1.0:
                logger.debug(
                    f"  SKIP ARB {opp.ticker}: profit after fees {profit_after_fees_cents:.2f}c <= 1c "
                    f"(gross edge {100 - total_cost_cents}c, fees {total_fee_cents:.2f}c)"
                )
                continue

            # Size: enough pairs to deploy up to $1, capped by daily loss cap AND arb capital cap
            remaining_arb_cap = arb_capital_cap - arb_capital_deployed
            max_dollars = min(1.0, daily_loss_cap, remaining_arb_cap)
            if max_dollars <= 0:
                logger.info(f"  SKIP ARB {opp.ticker}: arb capital cap exhausted")
                break
            pairs = max(1, int(max_dollars / cost_per_pair)) if cost_per_pair > 0 else 0
            actual_cost = pairs * cost_per_pair

            # Create decision for YES side
            decision_yes = TradeDecision(
                ticker=opp.ticker,
                action="buy_yes",
                strategy="arb",
                edge_cents=round(opp.edge_cents / 2, 1),  # Half edge per side
                fair_value_cents=round(50.0, 1),  # Guaranteed $1 payout
                market_price_cents=int(yes_price),
                price_to_pay_cents=int(yes_price),
                contracts=pairs,
                max_loss_dollars=round(pairs * yes_price / 100.0, 2),
                reason=f"ARB: {opp.description} | confidence={opp.confidence:.0%}",
            )

            # Create decision for NO side
            decision_no = TradeDecision(
                ticker=opp.ticker,
                action="buy_no",
                strategy="arb",
                edge_cents=round(opp.edge_cents / 2, 1),
                fair_value_cents=round(50.0, 1),
                market_price_cents=int(no_price),
                price_to_pay_cents=int(no_price),
                contracts=pairs,
                max_loss_dollars=round(pairs * no_price / 100.0, 2),
                reason=f"ARB: {opp.description} | confidence={opp.confidence:.0%}",
            )

            if alert_big_edge and opp.edge_cents > 10:
                alert_big_edge(opp.ticker, opp.edge_cents, "ARB")

            # Pre-flight balance check (local tracking, no API call)
            if actual_cost > remaining_balance:
                decision_yes.reason += " | SKIP: insufficient balance"
                decision_yes.contracts = 0
                decision_no.contracts = 0
                decisions.extend([decision_yes, decision_no])
                logger.warning(
                    f"  SKIP ARB {opp.ticker}: cost ${actual_cost:.2f} > "
                    f"remaining balance ${remaining_balance:.2f}"
                )
                continue

            # Place both sides
            if not dry_run and pairs > 0:
                order_succeeded = True
                for dec, side, price_c in [
                    (decision_yes, "yes", int(yes_price)),
                    (decision_no, "no", int(no_price)),
                ]:
                    side_cost = pairs * price_c / 100.0
                    # Check remaining balance before each individual order
                    if side_cost > remaining_balance:
                        dec.error = f"insufficient balance (${remaining_balance:.2f} < ${side_cost:.2f})"
                        logger.warning(
                            f"  SKIP ARB {side.upper()} {opp.ticker}: "
                            f"cost ${side_cost:.2f} > remaining ${remaining_balance:.2f}"
                        )
                        order_succeeded = False
                        continue
                    try:
                        order_kwargs = {
                            "ticker": opp.ticker,
                            "side": side,
                            "action": "buy",
                            "count": pairs,
                            "type": "limit",
                        }
                        if side == "yes":
                            order_kwargs["yes_price"] = price_c
                        else:
                            order_kwargs["no_price"] = price_c

                        result = client.place_order(**order_kwargs)
                        order_id = result.get("order", {}).get("order_id", "unknown")
                        dec.placed = True
                        dec.order_id = order_id
                        remaining_balance -= side_cost
                        logger.info(
                            f"  ARB ORDER: {side.upper()} {opp.ticker} x{pairs} "
                            f"@ {price_c}c | profit_after_fees +{profit_after_fees_cents:.1f}c "
                            f"| order_id={order_id} | remaining=${remaining_balance:.2f}"
                        )
                    except Exception as e:
                        dec.error = str(e)
                        logger.error(f"  ARB ORDER FAILED: {side} {opp.ticker} -- {e}")
                        order_succeeded = False
                if order_succeeded:
                    arb_capital_deployed += actual_cost
                trades_placed += 1
            else:
                if pairs > 0:
                    trades_placed += 1
                    arb_capital_deployed += actual_cost
                    remaining_balance -= actual_cost
                    logger.info(
                        f"  DRY-RUN ARB: would buy BOTH sides {opp.ticker} x{pairs} "
                        f"(YES@{yes_price}c + NO@{no_price}c = {total_cost_cents}c) "
                        f"| profit_after_fees +{profit_after_fees_cents:.1f}c "
                        f"| arb_deployed=${arb_capital_deployed:.2f}/{arb_capital_cap:.2f}"
                    )

            decisions.extend([decision_yes, decision_no])

        else:
            # For cross_event_arb and wide_spread: log but don't auto-trade
            decision = TradeDecision(
                ticker=opp.ticker,
                action="skip",
                strategy="arb",
                edge_cents=round(opp.edge_cents, 1),
                fair_value_cents=0,
                market_price_cents=0,
                price_to_pay_cents=0,
                contracts=0,
                max_loss_dollars=0,
                reason=(
                    f"ARB ({opp.type}): {opp.description} | "
                    f"confidence={opp.confidence:.0%} -- requires manual execution"
                ),
            )
            decisions.append(decision)
            logger.info(
                f"  ARB SIGNAL ({opp.type}): {opp.ticker} edge={opp.edge_cents:.1f}c "
                f"conf={opp.confidence:.0%} -- logged, not auto-traded"
            )

    logger.info(
        f"--- ARB SESSION DONE: {len(decisions)} decisions, "
        f"{trades_placed} trades, ${arb_capital_deployed:.2f} deployed "
        f"(cap ${arb_capital_cap:.2f}) ---"
    )
    return decisions


# ---------------------------------------------------------------------------
# Position exit check
# ---------------------------------------------------------------------------

EXIT_EDGE_THRESHOLD_CENTS = 2.0  # Exit when edge drops below this


def check_exits(
    client: KalshiClient,
    dry_run: bool,
    logger: logging.Logger,
) -> list[TradeDecision]:
    """
    Check open positions and recommend/execute exits where edge has evaporated.

    For each weather position:
    1. Fetch current forecast data (NWS + ensemble blend)
    2. Re-price the market using current model
    3. If the edge has flipped or dropped below EXIT_EDGE_THRESHOLD_CENTS,
       recommend exit (dry-run) or place a sell order (live)

    Called at the start of each trading session, before placing new trades.
    """
    logger.info("--- EXIT CHECK START ---")
    exit_decisions = []

    # Load evolved strategy params (for blend weights, stdev fallback, exit threshold)
    strategy = load_strategy_params()
    exit_threshold = strategy["exit_edge_threshold"]

    try:
        positions = client.get_positions()
        market_positions = positions.get("market_positions", [])
    except Exception as e:
        logger.error(f"Failed to fetch positions for exit check: {e}")
        return exit_decisions

    # Filter to weather positions only
    weather_positions = []
    for mp in market_positions:
        ticker = mp.get("ticker", "")
        pos = float(mp.get("position_fp", 0))
        if pos == 0:
            continue
        # Check if it's a weather ticker
        is_weather = any(ticker.startswith(prefix) for prefix in [
            "KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHLA", "KXHIGHDEN"
        ])
        if is_weather:
            weather_positions.append(mp)

    if not weather_positions:
        logger.info("  No open weather positions to check")
        logger.info("--- EXIT CHECK DONE: 0 exits ---")
        return exit_decisions

    logger.info(f"  Checking {len(weather_positions)} open weather positions")

    # Pre-fetch forecasts for all relevant series (avoid duplicate API calls)
    forecast_cache = {}
    for mp in weather_positions:
        ticker = mp["ticker"]
        # Extract series ticker (e.g., KXHIGHNY from KXHIGHNY-26MAR21-B57.5)
        series_ticker = ticker.split("-")[0] if "-" in ticker else ticker
        if series_ticker not in forecast_cache:
            nws = fetch_nws_forecast(series_ticker, logger)
            ensemble = fetch_ensemble_forecast(series_ticker, logger)
            forecast_cache[series_ticker] = blend_forecasts(nws, ensemble, logger, strategy=strategy)

    for mp in weather_positions:
        ticker = mp["ticker"]
        pos = float(mp.get("position_fp", 0))
        exposure = float(mp.get("market_exposure_dollars", 0))

        # Determine our side
        if pos > 0:
            our_side = "yes"
            contracts = int(pos)
        else:
            our_side = "no"
            contracts = int(abs(pos))

        series_ticker = ticker.split("-")[0] if "-" in ticker else ticker

        # Get current market price
        try:
            market_data = client.get_market(ticker)
            market_info = market_data.get("market", market_data)
            yes_bid = float(market_info.get("yes_bid_dollars", 0) or 0) * 100
            yes_ask = float(market_info.get("yes_ask_dollars", 0) or 0) * 100
        except Exception as e:
            logger.warning(f"  Could not fetch market data for {ticker}: {e}")
            continue

        # Skip if market is dead/settled
        if yes_bid <= 0 and yes_ask <= 0:
            continue

        # Parse market type from ticker/title
        title = market_info.get("title", "")
        mtype = parse_market_type(ticker, title)
        if not mtype:
            logger.debug(f"  Could not parse market type for {ticker}")
            continue

        # Match to forecast date
        market_date = parse_ticker_date(ticker)
        blended = forecast_cache.get(series_ticker, {})
        if not market_date or market_date not in blended:
            logger.debug(f"  No forecast data for {ticker} date={market_date}")
            continue

        fc = blended[market_date]
        forecast_temp = fc["temp"]
        stdev = fc["stdev"]

        # Re-calculate fair value with current forecast
        if mtype["type"] == "bucket":
            fair_prob = calc_bucket_probability(
                forecast_temp, mtype["low"], mtype["high"], stdev
            )
        elif mtype["type"] == "above":
            fair_prob = calc_above_probability(forecast_temp, mtype["threshold"], stdev)
        elif mtype["type"] == "below":
            fair_prob = calc_below_probability(forecast_temp, mtype["threshold"], stdev)
        else:
            continue

        fair_cents = fair_prob * 100

        # Calculate current edge based on our side
        if our_side == "yes":
            # We hold YES -- edge = fair value - what we'd get selling (yes_bid)
            current_edge = fair_cents - yes_bid if yes_bid > 0 else fair_cents
        else:
            # We hold NO -- edge = (100 - fair) - what we'd get selling NO
            # To sell NO, someone buys our NO at no_bid = 100 - yes_ask
            no_bid = 100 - yes_ask if yes_ask > 0 else 0
            current_edge = (100 - fair_cents) - no_bid if no_bid > 0 else (100 - fair_cents)

        # Check if edge has evaporated or flipped
        should_exit = current_edge < exit_threshold

        if not should_exit:
            logger.debug(
                f"  HOLD {ticker}: {our_side.upper()} x{contracts} "
                f"edge={current_edge:+.1f}c (still good)"
            )
            continue

        # Edge has dropped -- recommend or execute exit
        city = CITY_NAMES.get(series_ticker, series_ticker)

        decision = TradeDecision(
            ticker=ticker,
            action=f"sell_{our_side}",
            strategy="weather_exit",
            edge_cents=round(current_edge, 1),
            fair_value_cents=round(fair_cents, 1),
            market_price_cents=int(yes_bid) if our_side == "yes" else int(100 - yes_ask),
            price_to_pay_cents=int(yes_bid) if our_side == "yes" else int(100 - yes_ask),
            contracts=contracts,
            max_loss_dollars=round(exposure, 2),
            reason=(
                f"EXIT {city} {market_date} | forecast={forecast_temp:.1f}F "
                f"stdev={stdev:.2f} | fair={fair_cents:.1f}c "
                f"edge={current_edge:+.1f}c (below {exit_threshold}c threshold)"
            ),
            forecast_temp=round(forecast_temp, 1),
        )

        if dry_run:
            logger.info(
                f"  EXIT RECOMMENDED: {ticker} -- "
                f"{our_side.upper()} x{contracts} -- "
                f"edge dropped to {current_edge:+.1f}c"
            )
        else:
            # Place a sell order to close the position
            try:
                sell_price = int(yes_bid) if our_side == "yes" else int(100 - yes_ask)
                sell_price = max(1, min(99, sell_price))

                order_kwargs = {
                    "ticker": ticker,
                    "side": our_side,
                    "action": "sell",
                    "count": contracts,
                    "type": "limit",
                }
                if our_side == "yes":
                    order_kwargs["yes_price"] = sell_price
                else:
                    order_kwargs["no_price"] = sell_price

                result = client.place_order(**order_kwargs)
                order_id = result.get("order", {}).get("order_id", "unknown")
                decision.placed = True
                decision.order_id = order_id
                logger.info(
                    f"  EXIT ORDER PLACED: sell {our_side.upper()} {ticker} x{contracts} "
                    f"@ {sell_price}c | edge={current_edge:+.1f}c | order_id={order_id}"
                )
            except Exception as e:
                decision.error = str(e)
                logger.error(f"  EXIT ORDER FAILED: {ticker} -- {e}")

        exit_decisions.append(decision)
        time.sleep(0.5)  # Rate limit

    logger.info(
        f"--- EXIT CHECK DONE: {len(exit_decisions)} exit(s) recommended/placed ---"
    )
    return exit_decisions


# ---------------------------------------------------------------------------
# Dutch Book session (weather arbitrage)
# ---------------------------------------------------------------------------

MAX_DUTCH_BOOK_TRADES = 5  # cap per session


def run_dutch_book_session(
    client: KalshiClient,
    balance: float,
    dry_run: bool,
    logger: logging.Logger,
) -> list[TradeDecision]:
    """
    Dutch Book session -- exploit overpriced weather bucket sets.

    For each weather series, group open markets by event_ticker.
    If an event has exactly 6 markets and sum(yes_bid) > 102c,
    SELL YES on all 6 legs (1 contract each) to lock in guaranteed profit.

    Profit = sum(yes_bid) - 100c per contract set.
    """
    logger.info("--- DUTCH BOOK SESSION START ---")
    decisions = []

    weather_series = ["KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHLA", "KXHIGHDC", "KXHIGHDEN"]
    trades_placed = 0

    for series_ticker in weather_series:
        if trades_placed >= MAX_DUTCH_BOOK_TRADES:
            break

        # Fetch open markets for this series
        try:
            resp = client.get_markets(series_ticker=series_ticker, status="open", limit=100)
            markets = resp.get("markets", [])
        except Exception as e:
            logger.warning(f"  Failed to fetch markets for {series_ticker}: {e}")
            continue

        # Group markets by event_ticker
        events = {}
        for mkt in markets:
            event_ticker = mkt.get("event_ticker", "")
            if event_ticker:
                events.setdefault(event_ticker, []).append(mkt)

        for event_ticker, event_markets in events.items():
            if trades_placed >= MAX_DUTCH_BOOK_TRADES:
                break

            # Dutch book only works on complete bucket sets (exactly 6 markets)
            if len(event_markets) != 6:
                continue

            # Sum yes_bid across all legs
            total_yes_bid = 0
            legs = []
            for mkt in event_markets:
                yes_bid = int(float(mkt.get("yes_bid_dollars", 0) or 0) * 100)
                total_yes_bid += yes_bid
                legs.append({
                    "ticker": mkt.get("ticker", ""),
                    "yes_bid": int(yes_bid),
                    "title": mkt.get("title", ""),
                })

            if total_yes_bid <= 102:
                continue  # Not enough edge

            profit_cents = total_yes_bid - 100
            logger.info(
                f"  DUTCH BOOK FOUND: {event_ticker} | {len(legs)} legs | "
                f"sum(yes_bid)={total_yes_bid:.0f}c | profit={profit_cents:.0f}c/set"
            )

            # Check balance
            current_balance = get_account_balance(client, logger)
            if current_balance is not None and current_balance < 2.0:
                logger.warning(f"  SKIP DUTCH BOOK {event_ticker}: balance too low")
                continue

            event_decisions = []
            all_placed = True

            for leg in legs:
                ticker = leg["ticker"]
                sell_price = leg["yes_bid"]
                if sell_price < 1:
                    all_placed = False
                    continue

                decision = TradeDecision(
                    ticker=ticker,
                    action="sell_yes",
                    strategy="dutch_book",
                    edge_cents=round(profit_cents / len(legs), 1),
                    fair_value_cents=round(100 / len(legs), 1),
                    market_price_cents=sell_price,
                    price_to_pay_cents=sell_price,
                    contracts=1,
                    max_loss_dollars=round((100 - sell_price) / 100.0, 2),
                    reason=(
                        f"DUTCH BOOK: {event_ticker} leg | "
                        f"sell YES@{sell_price}c | total_profit={profit_cents:.0f}c"
                    ),
                )

                if not dry_run:
                    try:
                        result = client.place_order(
                            ticker=ticker,
                            side="yes",
                            action="sell",
                            count=1,
                            type="limit",
                            yes_price=sell_price,
                            post_only=True,
                        )
                        order_id = result.get("order", {}).get("order_id", "unknown")
                        decision.placed = True
                        decision.order_id = order_id
                        logger.info(
                            f"    DUTCH BOOK ORDER: SELL YES {ticker} x1 "
                            f"@ {sell_price}c | order_id={order_id}"
                        )
                    except Exception as e:
                        decision.error = str(e)
                        all_placed = False
                        logger.error(f"    DUTCH BOOK ORDER FAILED: {ticker} -- {e}")
                else:
                    logger.info(
                        f"    DRY-RUN DUTCH BOOK: would SELL YES {ticker} x1 @ {sell_price}c"
                    )

                event_decisions.append(decision)
                time.sleep(0.3)  # Rate limit

            decisions.extend(event_decisions)
            if all_placed and event_decisions:
                trades_placed += 1

    logger.info(
        f"--- DUTCH BOOK SESSION DONE: {len(decisions)} decisions, "
        f"{trades_placed} arb sets ---"
    )
    return decisions


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_auto_trade(dry_run: bool = True, weather: bool = True, btc: bool = True, sports: bool = True, arb: bool = True, copy: bool = True):
    """
    Run the full autonomous trading session.
    Called by CLI or by run_nightly.sh / launchd.
    """
    logger = setup_logger(dry_run=dry_run)

    # Check if trading is paused via Telegram /pause command
    pause_file = config.OUTPUT_DIR / ".trading_paused"
    if pause_file.exists() and not dry_run:
        logger.info("Trading is PAUSED (via /pause command). Skipping session.")
        logger.info("Send /resume in Telegram to re-enable trading.")
        return []

    # Initialize client
    try:
        client = KalshiClient()
        logger.info(f"Kalshi client initialized (env={config.KALSHI_ENV})")
    except Exception as e:
        logger.critical(f"Failed to initialize Kalshi client: {e}")
        logger.critical(traceback.format_exc())
        if alert_bot_error:
            alert_bot_error("auto_trade", str(e))
        return

    # Clean up stale resting orders from previous sessions
    try:
        stale_orders = client._request("GET", "/portfolio/orders", params={"status": "resting", "limit": 200})
        stale = [o for o in stale_orders.get("orders", []) if o.get("remaining_count", 0) == 0]
        if stale:
            for o in stale:
                try:
                    client._request("DELETE", f"/portfolio/orders/{o['order_id']}")
                except Exception:
                    pass
                time.sleep(0.1)
            logger.info(f"Cleaned up {len(stale)} stale resting orders")
    except Exception as e:
        logger.warning(f"Stale order cleanup failed: {e}")

    # Check account balance
    balance = get_account_balance(client, logger)
    if balance is None:
        logger.critical("Cannot proceed without account balance")
        return

    if balance < 0.10:
        logger.critical(f"Account balance too low (${balance:.2f}), aborting")
        return

    # Check daily loss cap
    daily_cap = balance * DAILY_LOSS_CAP_PCT
    logger.info(
        f"Session params: balance=${balance:.2f} daily_cap=${daily_cap:.2f} "
        f"max_deploy={balance * MAX_DAILY_DEPLOY_PCT:.2f} "
        f"max_per_trade=${MAX_DOLLARS_PER_TRADE:.2f}"
    )

    all_decisions = []

    # --- Exit check (runs before placing new trades) ---
    try:
        exit_decisions = check_exits(client, dry_run, logger)
        all_decisions.extend(exit_decisions)
    except Exception as e:
        logger.error(f"Exit check crashed: {e}")
        logger.error(traceback.format_exc())
        if alert_bot_error:
            alert_bot_error("auto_trade", str(e))

    # --- NBA Underdog session (focused, proven edge) ---
    # Buys YES on KXNBAGAME underdogs priced 10-30c.
    # Historical: 29.7% win rate vs 22% implied = +7.7pp edge, ROI +33%, Sharpe 2.64.
    # Conservative sizing: quarter Kelly, max $2/trade, max 5 bets/day.
    if find_nba_underdogs is not None:
        try:
            underdogs = find_nba_underdogs(client)
            budget = nba_risk_budget(balance)
            underdog_count = 0

            # Build set of NBA game IDs we already have positions on to avoid both-sides betting
            nba_game_ids_with_positions = set()
            try:
                positions = client.get_positions()
                for mp in positions.get("market_positions", []):
                    t = mp.get("ticker", "")
                    pos = float(mp.get("position", mp.get("position_fp", 0)))
                    if pos != 0:
                        gid = get_nba_game_id(t)
                        if gid:
                            nba_game_ids_with_positions.add(gid)
                if nba_game_ids_with_positions:
                    logger.info(f"NBA dedup: already have positions on {len(nba_game_ids_with_positions)} games: {nba_game_ids_with_positions}")
            except Exception as e:
                logger.warning(f"NBA dedup: could not fetch positions: {e}")

            for ud in underdogs:
                if underdog_count >= budget.get("max_daily_bets", 5):
                    logger.info(f"NBA underdog daily cap reached ({underdog_count})")
                    break
                # Skip if we already have a position on this game (avoid both-sides betting)
                game_id = get_nba_game_id(ud["ticker"])
                if game_id and game_id in nba_game_ids_with_positions:
                    logger.info(f"NBA dedup: SKIP {ud['ticker']} -- already have position on game {game_id}")
                    continue
                price_cents = int(ud.get("yes_price_cents", 20))
                cost_per = price_cents / 100.0
                max_contracts = int(budget.get("max_bet_dollars", 2.0) / cost_per) if cost_per > 0 else 0
                contracts = max(1, min(ud.get("suggested_contracts", 1), max_contracts))
                if contracts <= 0:
                    continue
                decision = TradeDecision(
                    ticker=ud["ticker"],
                    action="buy_yes",
                    strategy="nba_underdog",
                    edge_cents=ud.get("edge_pp", 0) * 1.0,
                    fair_value_cents=ud.get("estimated_prob", 0.297) * 100,
                    market_price_cents=price_cents,
                    price_to_pay_cents=price_cents,
                    contracts=contracts,
                    max_loss_dollars=contracts * cost_per,
                    reason=(
                        f"NBA underdog: YES@{price_cents}c, "
                        f"impl={ud.get('implied_prob',0)*100:.0f}%, "
                        f"est={ud.get('estimated_prob',0)*100:.1f}%, "
                        f"edge={ud.get('edge_pp',0):+.1f}pp, "
                        f"EV={ud.get('ev_per_contract',0)*100:.1f}c"
                    ),
                    placed=False,
                )
                if not dry_run:
                    try:
                        # Bid 1c below ask to stay as maker order (post_only default)
                        # If ask=13c, we bid 12c and wait for a fill
                        maker_price = max(1, price_cents - 1)
                        result = client.place_order(
                            ticker=decision.ticker,
                            side="yes",
                            action="buy",
                            count=contracts,
                            type="limit",
                            yes_price=maker_price,
                        )
                        decision.placed = True
                        decision.price_to_pay_cents = maker_price
                        decision.order_id = result.get("order", {}).get("order_id", "")
                        logger.info(
                            f"NBA UNDERDOG: {decision.ticker} BUY YES x{contracts} "
                            f"@{maker_price}c (ask={price_cents}c) edge={ud.get('edge_pp',0):+.1f}pp"
                        )
                    except Exception as e:
                        decision.error = str(e)
                        logger.warning(f"NBA underdog order failed: {e}")
                else:
                    logger.info(
                        f"[DRY] NBA UNDERDOG: {decision.ticker} YES x{contracts} "
                        f"@{price_cents}c edge={ud.get('edge_pp',0):+.1f}pp"
                    )
                all_decisions.append(decision)
                underdog_count += 1
        except Exception as e:
            logger.error(f"NBA underdog session crashed: {e}")
            logger.error(traceback.format_exc())
            if alert_bot_error:
                alert_bot_error("auto_trade", str(e))
    else:
        logger.info("--- NBA UNDERDOG SESSION SKIPPED (module not found) ---")

    # --- Arb session ---
    if arb:
        try:
            arb_decisions = run_arb_session(client, balance, dry_run, logger)
            all_decisions.extend(arb_decisions)
        except Exception as e:
            logger.error(f"Arb session crashed: {e}")
            logger.error(traceback.format_exc())
            if alert_bot_error:
                alert_bot_error("auto_trade", str(e))

    # --- Dutch Book session (runs every cycle) ---
    try:
        dutch_book_decisions = run_dutch_book_session(client, balance, dry_run, logger)
        all_decisions.extend(dutch_book_decisions)
    except Exception as e:
        logger.error(f"Dutch book session crashed: {e}")
        logger.error(traceback.format_exc())
        if alert_bot_error:
            alert_bot_error("auto_trade", str(e))

    # --- Weather Tail session (buy NO on cheap YES weather markets) ---
    if find_weather_tail_trades is not None:
        try:
            tail_trades = find_weather_tail_trades(client)
            risk = tail_risk_budget(balance)
            for tt in tail_trades[:5]:  # max 5 tail trades per cycle
                # Use risk budget for contract count (suggested_contracts is never set by weather_tail_strategy)
                max_contracts = min(risk.get("max_contracts", 5), 3)  # cap at 3 per the strategy design
                price_cents = int(tt.get("no_price_cents", 97))
                cost_per = price_cents / 100.0
                contracts_by_budget = int(config.MAX_BET_DOLLARS / cost_per) if cost_per > 0 else 1
                contracts = max(1, min(contracts_by_budget, max_contracts))
                if contracts <= 0:
                    continue
                decision = TradeDecision(
                    ticker=tt["ticker"],
                    action="buy_no",
                    strategy="weather_tail",
                    edge_cents=tt.get("net_edge_cents", 0),
                    fair_value_cents=tt.get("our_no_prob", 0.99) * 100,
                    market_price_cents=tt.get("yes_price_cents", 3),
                    price_to_pay_cents=price_cents,
                    contracts=contracts,
                    max_loss_dollars=contracts * price_cents / 100.0,
                    reason=f"Weather tail: YES@{tt.get('yes_price_cents',0)}c, P(NO)={tt.get('our_no_prob',0.99):.3f}, edge={tt.get('net_edge_cents',0):.1f}c",
                    placed=False,
                )
                if not dry_run:
                    try:
                        result = client.place_order(
                            ticker=decision.ticker,
                            side="no",
                            action="buy",
                            count=contracts,
                            type="limit",
                            no_price=price_cents,
                        )
                        decision.placed = True
                        decision.order_id = result.get("order", {}).get("order_id", "")
                        logger.info(f"WEATHER TAIL: {decision.ticker} BUY NO x{contracts} @{price_cents}c edge={decision.edge_cents:.1f}c")
                    except Exception as e:
                        decision.error = str(e)
                        logger.warning(f"Weather tail order failed: {e}")
                else:
                    logger.info(f"[DRY] WEATHER TAIL: {decision.ticker} NO x{contracts} @{price_cents}c")
                all_decisions.append(decision)
        except Exception as e:
            logger.error(f"Weather tail session crashed: {e}")
            logger.error(traceback.format_exc())

    # --- Deep ITM Maker session (buy YES at 95c on near-certain markets) ---
    try:
        from deep_itm_strategy import find_deep_itm_opportunities, deep_itm_risk_budget
        logger.info("--- DEEP ITM MAKER SESSION START ---")
        itm_opps = find_deep_itm_opportunities(client)
        itm_budget = deep_itm_risk_budget(balance)
        itm_count = 0
        max_itm = itm_budget.get("max_positions", 10)
        for opp in itm_opps[:max_itm]:
            if itm_count >= max_itm:
                break
            contracts = min(opp.get("suggested_contracts", 1), itm_budget.get("max_contracts_per_position", 3))
            if contracts <= 0:
                continue
            bid_price = opp.get("our_bid_cents", 95)
            decision = TradeDecision(
                ticker=opp["ticker"],
                action="buy_yes",
                strategy="deep_itm",
                edge_cents=opp.get("ev_cents", 0),
                fair_value_cents=opp.get("win_rate", 0.996) * 100,
                market_price_cents=opp.get("current_yes_ask_cents", 99),
                price_to_pay_cents=bid_price,
                contracts=contracts,
                max_loss_dollars=contracts * bid_price / 100.0,
                reason=f"Deep ITM: YES@{opp.get('current_yes_ask_cents',99)}c, bid@{bid_price}c, WR={opp.get('win_rate',0.996):.3f}, EV={opp.get('ev_cents',0):.1f}c",
                placed=False,
            )
            if not dry_run:
                try:
                    result = client.place_order(
                        ticker=decision.ticker,
                        side="yes",
                        action="buy",
                        count=contracts,
                        type="limit",
                        yes_price=bid_price,
                        post_only=True,
                    )
                    decision.placed = True
                    decision.order_id = result.get("order", {}).get("order_id", "")
                    logger.info(
                        f"DEEP ITM: {decision.ticker} BUY YES x{contracts} "
                        f"@{bid_price}c EV={opp.get('ev_cents',0):.1f}c"
                    )
                except Exception as e:
                    decision.error = str(e)
                    logger.warning(f"Deep ITM order failed: {e}")
            else:
                logger.info(
                    f"[DRY] DEEP ITM: {decision.ticker} YES x{contracts} @{bid_price}c"
                )
            all_decisions.append(decision)
            itm_count += 1
        logger.info(f"--- DEEP ITM SESSION DONE: {itm_count} orders ---")
    except ImportError:
        logger.info("--- DEEP ITM SESSION SKIPPED (module not found) ---")
    except Exception as e:
        logger.error(f"Deep ITM session crashed: {e}")
        logger.error(traceback.format_exc())

    # --- AutoResearch Evolved Strategy session ---
    # Runs evaluate_market() from candidate_strategy.py on all live markets.
    # This is the bridge between autoresearch discoveries and live trading.
    # evaluate_market() returns astronomical contract counts from backtesting —
    # we cap them to real position sizes via MAX_BET_DOLLARS.
    try:
        from autoresearch.candidate_strategy import evaluate_market
        logger.info("--- AUTORESEARCH STRATEGY SESSION START ---")

        # Collect tickers already traded this session to avoid duplicates
        already_traded = {d.ticker for d in all_decisions if d.placed}

        # Also exclude markets where we already hold positions (from weather_tail_runner, arb_runner, etc.)
        try:
            positions = client.get_positions()
            for mp in positions.get("market_positions", []):
                pos = float(mp.get("position", mp.get("position_fp", 0)))
                if pos != 0:
                    already_traded.add(mp.get("ticker", ""))
            if already_traded:
                logger.info(f"AutoResearch dedup: {len(already_traded)} tickers excluded (session + existing positions)")
        except Exception as e:
            logger.warning(f"AutoResearch dedup: could not fetch positions: {e}")

        # Scan markets across all target series
        ar_count = 0
        ar_max = 20  # cap per cycle
        ar_deployed = 0.0
        ar_budget = balance * 0.20  # max 20% of balance for this session

        for series_prefix in config.TARGET_MARKET_SERIES:
            if ar_count >= ar_max or ar_deployed >= ar_budget:
                break
            try:
                page = client.get_markets(series_ticker=series_prefix, status="open", limit=50)
                markets = page.get("markets", [])
            except Exception:
                continue

            for mkt in markets:
                if ar_count >= ar_max or ar_deployed >= ar_budget:
                    break

                ticker = mkt.get("ticker", "")
                if not ticker or ticker in already_traded:
                    continue

                series = mkt.get("series_ticker", "") or mkt.get("event_ticker", "").split("-")[0]
                # Parse prices — API returns dollar strings like "0.3300"
                try:
                    yes_bid = int(float(mkt.get("yes_bid_dollars", 0) or 0) * 100)
                    yes_ask = int(float(mkt.get("yes_ask_dollars", 0) or 0) * 100)
                    last_price = int(float(mkt.get("last_price_dollars", 0) or 0) * 100)
                    vol = int(float(mkt.get("volume_fp", 0) or mkt.get("volume", 0) or 0))
                except (ValueError, TypeError):
                    continue

                yes_cents = last_price or yes_bid or yes_ask
                if yes_cents <= 0 or yes_cents >= 100:
                    continue

                # Call the evolved strategy function
                oi = int(float(mkt.get("open_interest", 0) or 0))
                lp = float(mkt.get("last_price_dollars", 0) or 0)
                pp = float(mkt.get("previous_yes_bid_dollars", mkt.get("previous_price_dollars", 0)) or 0)
                signal = evaluate_market(
                    ticker=ticker,
                    series=series,
                    yes_cents=yes_cents,
                    ask_cents=yes_ask,
                    bid_cents=yes_bid,
                    volume=vol,
                    open_interest=oi,
                    last_price=lp,
                    previous_price=pp,
                    settled_yes=None,  # live trading
                )

                if not signal or signal.get("action") in (None, "skip"):
                    continue

                action = signal["action"]
                # Cap contracts to real position sizes (backtest uses astronomical numbers)
                raw_contracts = signal.get("contracts", 1)
                if action == "buy_yes":
                    entry_price = yes_ask if yes_ask > 0 else yes_cents
                elif action == "buy_no":
                    entry_price = 100 - (yes_bid if yes_bid > 0 else yes_cents)
                else:
                    continue

                if entry_price <= 0 or entry_price >= 100:
                    continue

                cost_per = entry_price / 100.0
                max_by_budget = int(config.MAX_BET_DOLLARS / cost_per) if cost_per > 0 else 0
                contracts = max(1, min(raw_contracts, max_by_budget, 10))

                total_cost = contracts * cost_per
                if total_cost > config.MAX_BET_DOLLARS:
                    contracts = int(config.MAX_BET_DOLLARS / cost_per)
                if contracts <= 0:
                    continue

                side = "yes" if action == "buy_yes" else "no"
                price_cents = entry_price

                decision = TradeDecision(
                    ticker=ticker,
                    action=action,
                    strategy="autoresearch",
                    edge_cents=0,  # evaluate_market doesn't return edge
                    fair_value_cents=0,
                    market_price_cents=yes_cents,
                    price_to_pay_cents=price_cents,
                    contracts=contracts,
                    max_loss_dollars=contracts * cost_per,
                    reason=f"AutoResearch: {action} {side}@{price_cents}c x{contracts} (evolved strategy)",
                    placed=False,
                )

                if not dry_run:
                    try:
                        order_kwargs = {
                            "ticker": ticker,
                            "side": side,
                            "action": "buy",
                            "count": contracts,
                            "type": "limit",
                        }
                        if side == "yes":
                            order_kwargs["yes_price"] = price_cents
                        else:
                            order_kwargs["no_price"] = price_cents
                        result = client.place_order(**order_kwargs)
                        decision.placed = True
                        decision.order_id = result.get("order", {}).get("order_id", "")
                        ar_deployed += contracts * cost_per
                        ar_count += 1
                        already_traded.add(ticker)
                        logger.info(
                            f"AUTORESEARCH: {ticker} {action.upper()} {side}@{price_cents}c x{contracts}"
                        )
                    except Exception as e:
                        decision.error = str(e)
                        logger.debug(f"AutoResearch order failed: {e}")
                else:
                    ar_count += 1
                    logger.info(f"[DRY] AUTORESEARCH: {ticker} {action.upper()} {side}@{price_cents}c x{contracts}")
                all_decisions.append(decision)

        logger.info(f"--- AUTORESEARCH SESSION DONE: {ar_count} trades, ${ar_deployed:.2f} deployed ---")
    except ImportError:
        logger.info("--- AUTORESEARCH SESSION SKIPPED (candidate_strategy not found) ---")
    except Exception as e:
        logger.error(f"AutoResearch session crashed: {e}")
        logger.error(traceback.format_exc())

    # --- Drawdown check ---
    # Use ACCOUNT_BALANCE (portfolio value) not just cash for drawdown calc.
    # When most capital is deployed in positions, cash is near-zero and any
    # trade would trigger a false drawdown alert.
    total_risk = sum(d.max_loss_dollars for d in all_decisions if d.placed or (dry_run and d.contracts > 0))
    account_value = max(balance, config.ACCOUNT_BALANCE)  # use whichever is higher
    if account_value > 0:
        dd_pct = (total_risk / account_value) * 100
        if alert_drawdown and dd_pct >= 15.0:  # aligned with MAX_DAILY_LOSS_PCT
            alert_drawdown(dd_pct, balance)

    # --- Summary ---
    trades_placed = [d for d in all_decisions if d.placed or (dry_run and d.contracts > 0)]
    trades_skipped = [d for d in all_decisions if d.contracts == 0]
    errors = [d for d in all_decisions if d.error]

    total_deployed = sum(d.max_loss_dollars for d in trades_placed)
    total_edge = sum(d.edge_cents for d in trades_placed)

    logger.info("")
    logger.info("=" * 60)
    logger.info("SESSION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total decisions:  {len(all_decisions)}")
    logger.info(f"Trades placed:    {len(trades_placed)}")
    logger.info(f"Trades skipped:   {len(trades_skipped)}")
    logger.info(f"Errors:           {len(errors)}")
    logger.info(f"Total deployed:   ${total_deployed:.2f}")
    logger.info(f"Total edge:       {total_edge:.1f}c")
    logger.info(f"Mode:             {'DRY-RUN' if dry_run else 'LIVE'}")

    if trades_placed:
        logger.info("")
        logger.info("Trades:")
        for d in trades_placed:
            status = "PLACED" if d.placed else "DRY-RUN"
            logger.info(
                f"  [{status}] {d.action:8s} {d.ticker:30s} x{d.contracts} "
                f"@ {d.price_to_pay_cents}c  edge=+{d.edge_cents:.1f}c  "
                f"cost=${d.max_loss_dollars:.2f}"
            )

    if errors:
        logger.info("")
        logger.info("Errors:")
        for d in errors:
            logger.info(f"  {d.ticker}: {d.error}")

    # Save decisions to JSON for analysis
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    json_path = config.OUTPUT_DIR / f"auto_trade_{today}.json"
    try:
        with open(json_path, "w") as f:
            json.dump(
                {
                    "date": today,
                    "mode": "dry_run" if dry_run else "live",
                    "env": config.KALSHI_ENV,
                    "balance": balance,
                    "total_decisions": len(all_decisions),
                    "trades_placed": len(trades_placed),
                    "total_deployed": total_deployed,
                    "decisions": [asdict(d) for d in all_decisions],
                },
                f,
                indent=2,
            )
        logger.info(f"Decisions saved to: {json_path}")
    except Exception as e:
        logger.warning(f"Failed to save JSON: {e}")

    # --- Send daily summary alert ---
    if alert_daily_summary:
        # Count trades by strategy
        strategy_counts = {}
        for d in trades_placed:
            strategy_counts[d.strategy] = strategy_counts.get(d.strategy, 0) + 1
        strategy_str = ", ".join(f"{k}:{v}" for k, v in sorted(strategy_counts.items()))
        logger.info(f"Strategy breakdown: {strategy_str}")
        alert_daily_summary(len(trades_placed), total_deployed, 0)

    # Record estimates for Brier calibration
    if _brier is not None:
        for d in all_decisions:
            if d.contracts > 0 and d.fair_value_cents > 0:
                try:
                    our_prob = d.fair_value_cents / 100.0
                    market_prob = d.market_price_cents / 100.0
                    _brier.record_estimate(
                        ticker=d.ticker,
                        market_title=d.reason[:80] if d.reason else d.ticker,
                        series=d.ticker.split("-")[0] if "-" in d.ticker else d.strategy,
                        our_probability=our_prob,
                        market_price=market_prob,
                        strategy=d.strategy,
                    )
                except Exception:
                    pass

    logger.info("=" * 60)
    logger.info("AUTO-TRADE SESSION COMPLETE")
    logger.info("=" * 60)

    return all_decisions


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kalshi Auto-Trade (autonomous daily trading)")
    parser.add_argument(
        "--live", action="store_true",
        help="Place real orders (default: dry-run)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Show what would happen without placing orders",
    )
    parser.add_argument(
        "--arb-only", action="store_true",
        help="Run only the arbitrage session",
    )
    parser.add_argument(
        "--no-confirm", action="store_true",
        help="Skip confirmation prompt (for unattended/cron runs)",
    )
    args = parser.parse_args()

    # --live enables live trading; --dry-run overrides it back to dry mode
    # Without any flag, defaults to dry-run (is_live=False)
    is_live = args.live and not args.dry_run
    do_arb = True

    no_confirm = args.no_confirm if hasattr(args, 'no_confirm') else False

    if is_live and not no_confirm:
        print("\n*** LIVE MODE -- Real orders will be placed ***")
        print(f"    Environment: {config.KALSHI_ENV}")
        print(f"    Max per trade: ${MAX_DOLLARS_PER_TRADE}")
        confirm = input("    Type 'YES' to continue: ").strip()
        if confirm != "YES":
            print("Aborted.")
            sys.exit(0)

    run_auto_trade(dry_run=not is_live, arb=do_arb)
