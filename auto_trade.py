"""
auto_trade.py -- Fully autonomous daily trading script.

Runs without human intervention. Four sessions per day:

  MORNING (weather):
    1. Fetch NWS forecasts from exact settlement grid points
    2. Fetch GFS 30-member ensemble from Open-Meteo
    3. Fetch HRRR (3km resolution) from Open-Meteo for day-0/day-1
    4. Blend forecasts with day-dependent weights:
       Day 0: 40% HRRR + 30% GFS + 30% NWS
       Day 1: 25% HRRR + 40% GFS + 35% NWS
       Day 2+: 60% GFS + 40% NWS (no HRRR)
    5. Use ensemble stdev (not hardcoded)
    6. Price all KXHIGH buckets with normal distribution model
    7. Place limit orders on edges >3 cents

  AFTERNOON (CRYPTO: BTC + ETH + SOL):
    1. Fetch price from Coinbase + CoinGecko (cross-verify) for each asset
    2. Calculate 30-day realized vol from CoinGecko history (asset-specific)
    3. Price buckets with log-normal model using asset-specific volatility
    4. Buy NO on overpriced center buckets / YES on underpriced tails (edge >4 cents)

  SPORTS (NBA):
    1. Fetch team stats and today's games from ESPN (free API)
    2. Build win-probability model from efficiency ratings + home court
    3. Compare model probabilities to Kalshi market prices
    4. Place trades on edges >5 cents

  ARB (arbitrage):
    1. Scan all target series for YES/NO mispricing (yes + no < $1.00)
    2. Scan for cross-event arb and wide spreads
    3. For YES/NO arb: buy both sides to lock in guaranteed profit
    4. Log other arb types for manual review

SAFETY:
  - Balance check before EVERY order
  - 8% daily loss cap
  - Max $2 per trade
  - Max 20% of account deployed per day (weather)
  - Max 5 trades per crypto asset per day, max 5 sports/arb trades per day
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
    from sports_strategy import find_nba_edges
except (ImportError, OSError):
    find_nba_edges = None

try:
    from arb_scanner import ArbScanner, ArbOpportunity
    from market_scanner import MarketScanner
except (ImportError, OSError):
    ArbScanner = None
    MarketScanner = None

try:
    from crypto_strategy import CRYPTO_ASSETS
except (ImportError, OSError):
    CRYPTO_ASSETS = {
        "BTC": {"kalshi_series": "KXBTC", "coinbase_pair": "BTC-USD", "coingecko_id": "bitcoin", "default_annual_vol": 0.55},
        "ETH": {"kalshi_series": "KXETH", "coinbase_pair": "ETH-USD", "coingecko_id": "ethereum", "default_annual_vol": 0.65},
        "SOL": {"kalshi_series": "KXSOL", "coinbase_pair": "SOL-USD", "coingecko_id": "solana", "default_annual_vol": 0.85},
    }

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
    from momentum_strategy import detect_momentum_signals, generate_momentum_trades
except ImportError:
    detect_momentum_signals = None
    generate_momentum_trades = None

try:
    from nba_underdog_strategy import find_nba_underdogs, nba_risk_budget
except (ImportError, OSError):
    find_nba_underdogs = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# City coordinates for Open-Meteo ensemble API (lat, lon)
CITY_COORDS = {
    "KXHIGHNY":  (40.7829, -73.9654),   # NYC Central Park
    "KXHIGHCHI": (41.7868, -87.7522),   # Chicago Midway
    "KXHIGHMIA": (25.7933, -80.2906),   # Miami Intl Airport
    "KXHIGHLA":  (33.9425, -118.4081),  # Los Angeles LAX
    "KXHIGHDC":  (38.8512, -77.0402),   # Washington DC — Reagan National
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
# HRRR is best for 0-18h (same-day/next-day), no coverage beyond day 1.
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
MAX_DOLLARS_PER_TRADE = 2.0
MAX_DAILY_DEPLOY_PCT = 0.20      # 20% of account for weather
MAX_BTC_TRADES_PER_DAY = 5
MAX_SPORTS_TRADES_PER_DAY = 5
SPORTS_EDGE_THRESHOLD_CENTS = 5.0
MAX_ARB_TRADES_PER_DAY = 5
ARB_EDGE_THRESHOLD_CENTS = 3.0   # Lower threshold -- arb has built-in edge
MAX_CRYPTO_TRADES_PER_DAY = 5    # Per-asset cap for crypto (BTC/ETH/SOL)
DAILY_LOSS_CAP_PCT = 0.08        # 8% of account

# Open-Meteo GFS ensemble: 30 members + 1 control = 31 total
ENSEMBLE_MEMBERS = 30


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
# Weather forecast fetching
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

    Uses the Open-Meteo Ensemble API:
    https://ensemble-api.open-meteo.com/v1/ensemble?latitude=X&longitude=Y
      &daily=temperature_2m_max&models=gfs_seamless&forecast_days=3
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
# BTC forecast fetching
# ---------------------------------------------------------------------------

def fetch_btc_price_coinbase(logger: logging.Logger) -> Optional[float]:
    """Fetch current BTC/USD spot from Coinbase."""
    try:
        resp = requests.get(
            "https://api.coinbase.com/v2/prices/BTC-USD/spot",
            timeout=10,
        )
        resp.raise_for_status()
        price = float(resp.json()["data"]["amount"])
        logger.debug(f"Coinbase BTC price: ${price:,.2f}")
        return price
    except Exception as e:
        logger.warning(f"Coinbase BTC price error: {e}")
        return None


def fetch_btc_price_coingecko(logger: logging.Logger) -> Optional[float]:
    """Fetch current BTC/USD spot from CoinGecko."""
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price"
            "?ids=bitcoin&vs_currencies=usd",
            timeout=10,
        )
        resp.raise_for_status()
        price = float(resp.json()["bitcoin"]["usd"])
        logger.debug(f"CoinGecko BTC price: ${price:,.2f}")
        return price
    except Exception as e:
        logger.warning(f"CoinGecko BTC price error: {e}")
        return None


def fetch_btc_price(logger: logging.Logger) -> Optional[float]:
    """
    Cross-verified BTC price: average of Coinbase + CoinGecko.
    If they differ by >2%, log a warning and use Coinbase.
    """
    cb = fetch_btc_price_coinbase(logger)
    cg = fetch_btc_price_coingecko(logger)

    if cb is not None and cg is not None:
        spread_pct = abs(cb - cg) / max(cb, cg) * 100
        if spread_pct > 2.0:
            logger.warning(
                f"BTC price discrepancy: Coinbase=${cb:,.0f} CoinGecko=${cg:,.0f} "
                f"({spread_pct:.1f}% spread) -- using Coinbase"
            )
            return cb
        avg = (cb + cg) / 2
        logger.info(f"BTC price: ${avg:,.0f} (CB=${cb:,.0f} CG=${cg:,.0f})")
        return avg
    elif cb is not None:
        logger.info(f"BTC price (Coinbase only): ${cb:,.0f}")
        return cb
    elif cg is not None:
        logger.info(f"BTC price (CoinGecko only): ${cg:,.0f}")
        return cg
    else:
        logger.error("Could not fetch BTC price from any source")
        return None


def fetch_btc_30d_vol(logger: logging.Logger) -> Optional[float]:
    """
    Calculate 30-day realized volatility from CoinGecko daily prices.
    Returns annualized vol as a decimal (e.g., 0.65 = 65%).
    """
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
            "?vs_currency=usd&days=30&interval=daily",
            timeout=15,
        )
        resp.raise_for_status()
        prices = resp.json().get("prices", [])

        if len(prices) < 10:
            logger.warning(f"BTC vol: only {len(prices)} price points, need at least 10")
            return None

        # Calculate daily log returns
        close_prices = [p[1] for p in prices]
        log_returns = [
            math.log(close_prices[i] / close_prices[i - 1])
            for i in range(1, len(close_prices))
            if close_prices[i - 1] > 0
        ]

        if len(log_returns) < 5:
            return None

        # Daily vol -> annualize
        mean_ret = sum(log_returns) / len(log_returns)
        variance = sum((r - mean_ret) ** 2 for r in log_returns) / (len(log_returns) - 1)
        daily_vol = math.sqrt(variance)
        annual_vol = daily_vol * math.sqrt(365)

        logger.info(
            f"BTC 30d vol: daily={daily_vol:.4f} annual={annual_vol:.1%} "
            f"({len(log_returns)} returns)"
        )
        return annual_vol
    except Exception as e:
        logger.warning(f"BTC vol calculation error: {e}")
        return None


# ---------------------------------------------------------------------------
# Generalized crypto price/vol fetching (for ETH, SOL, and future assets)
# ---------------------------------------------------------------------------

def fetch_crypto_price_auto(asset: str, logger: logging.Logger) -> Optional[float]:
    """
    Fetch current price for any crypto asset (BTC, ETH, SOL) from Coinbase + CoinGecko.
    Returns averaged price or None on failure.
    """
    cfg = CRYPTO_ASSETS.get(asset)
    if not cfg:
        logger.error(f"Unknown crypto asset: {asset}")
        return None

    prices = []

    # Source 1: Coinbase
    try:
        pair = cfg.get("coinbase_pair", f"{asset}-USD")
        resp = requests.get(
            f"https://api.coinbase.com/v2/prices/{pair}/spot",
            timeout=10,
        )
        resp.raise_for_status()
        cb_price = float(resp.json()["data"]["amount"])
        prices.append(cb_price)
        logger.debug(f"Coinbase {asset} price: ${cb_price:,.2f}")
    except Exception as e:
        logger.warning(f"Coinbase {asset} price error: {e}")

    # Source 2: CoinGecko
    try:
        cg_id = cfg.get("coingecko_id", asset.lower())
        resp = requests.get(
            f"https://api.coingecko.com/api/v3/simple/price"
            f"?ids={cg_id}&vs_currencies=usd",
            timeout=10,
        )
        resp.raise_for_status()
        cg_price = float(resp.json()[cg_id]["usd"])
        prices.append(cg_price)
        logger.debug(f"CoinGecko {asset} price: ${cg_price:,.2f}")
    except Exception as e:
        logger.warning(f"CoinGecko {asset} price error: {e}")

    if not prices:
        logger.error(f"Could not fetch {asset} price from any source")
        return None

    avg = sum(prices) / len(prices)
    if len(prices) > 1:
        spread_pct = abs(prices[0] - prices[1]) / max(prices) * 100
        if spread_pct > 2.0:
            logger.warning(f"{asset} price discrepancy: {spread_pct:.1f}% -- using Coinbase")
            return prices[0]
        logger.info(f"{asset} price: ${avg:,.2f} ({len(prices)} sources)")
    else:
        logger.info(f"{asset} price (single source): ${avg:,.2f}")
    return avg


def fetch_crypto_30d_vol(asset: str, logger: logging.Logger) -> Optional[float]:
    """
    Calculate 30-day realized volatility for any crypto asset from CoinGecko.
    Returns annualized vol as a decimal.
    """
    cfg = CRYPTO_ASSETS.get(asset)
    if not cfg:
        return None

    cg_id = cfg.get("coingecko_id", asset.lower())
    try:
        resp = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart"
            f"?vs_currency=usd&days=30&interval=daily",
            timeout=15,
        )
        resp.raise_for_status()
        prices = resp.json().get("prices", [])

        if len(prices) < 10:
            logger.warning(f"{asset} vol: only {len(prices)} price points, need at least 10")
            return None

        close_prices = [p[1] for p in prices]
        log_returns = [
            math.log(close_prices[i] / close_prices[i - 1])
            for i in range(1, len(close_prices))
            if close_prices[i - 1] > 0
        ]

        if len(log_returns) < 5:
            return None

        mean_ret = sum(log_returns) / len(log_returns)
        variance = sum((r - mean_ret) ** 2 for r in log_returns) / (len(log_returns) - 1)
        daily_vol = math.sqrt(variance)
        annual_vol = daily_vol * math.sqrt(365)

        logger.info(
            f"{asset} 30d vol: daily={daily_vol:.4f} annual={annual_vol:.1%} "
            f"({len(log_returns)} returns)"
        )
        return annual_vol
    except Exception as e:
        logger.warning(f"{asset} vol calculation error: {e}")
        return None


def parse_bucket_from_title(title: str) -> Optional[dict]:
    """
    Parse actual bucket bounds from a Kalshi market title.
    Handles all crypto assets (BTC, ETH, SOL) with decimal support.

    Title formats:
        "Will Bitcoin be between $87,000 and $87,249?"
        "Will Ether be between $2,190.00 and $2,209.99?"
        "Will Solana be between $140.00 and $141.99?"
        "Bitcoin above $87,500?"  /  "$87,250 or above"
        "$2,190 to $2,209.99"
    """
    title_lower = title.lower()

    # Pattern 1: "between $X and $Y"
    between_match = re.search(
        r'between\s+\$?([\d,.]+)\s+and\s+\$?([\d,.]+)', title_lower
    )
    if between_match:
        low = float(between_match.group(1).replace(",", ""))
        high = float(between_match.group(2).replace(",", ""))
        # Kalshi "between $X and $Y" means [X, Y], add 1 unit for exclusive upper
        return {"type": "bucket", "low": low, "high": high + 0.01}

    # Pattern 2: "X or above"
    above_or_match = re.search(r'\$?([\d,.]+)\s+or\s+above', title_lower)
    if above_or_match:
        threshold = float(above_or_match.group(1).replace(",", ""))
        return {"type": "above", "low": threshold, "high": 1e9}

    # Pattern 3: "above $X" or "> $X"
    if "above" in title_lower or ">" in title:
        match = re.search(r'[\$>]\s*([\d,.]+)', title)
        if match:
            threshold = float(match.group(1).replace(",", ""))
            return {"type": "above", "low": threshold, "high": 1e9}

    # Pattern 4: "X or below"
    below_or_match = re.search(r'\$?([\d,.]+)\s+or\s+below', title_lower)
    if below_or_match:
        threshold = float(below_or_match.group(1).replace(",", ""))
        return {"type": "below", "low": 0, "high": threshold + 0.01}

    # Pattern 5: "below $X" or "< $X"
    if "below" in title_lower or "<" in title:
        match = re.search(r'[\$<]\s*([\d,.]+)', title)
        if match:
            threshold = float(match.group(1).replace(",", ""))
            return {"type": "below", "low": 0, "high": threshold}

    # Pattern 6: "$X to $Y" or "$X - $Y" (with decimal support)
    range_match = re.search(r'\$([\d,.]+)\s*(?:to|-)\s*\$([\d,.]+)', title)
    if range_match:
        low = float(range_match.group(1).replace(",", ""))
        high = float(range_match.group(2).replace(",", ""))
        return {"type": "bucket", "low": low, "high": high}

    return None


# Asset-specific half-widths for ticker-based fallback parsing.
# These are HALF the bucket width: BTC=$500 -> ±250, ETH=$20 -> ±10, SOL=$2 -> ±1
CRYPTO_BUCKET_HALF_WIDTHS = {
    "BTC": 250,
    "ETH": 10,
    "SOL": 1,
}


def parse_crypto_bucket(ticker: str, title: str, asset: str = "BTC") -> Optional[dict]:
    """
    Parse crypto market ticker/title to extract bucket bounds.
    Works for BTC, ETH, SOL with asset-specific bucket widths.

    Prefers title-based parsing (exact bounds) over ticker-based inference.
    """
    # First, try parsing exact bounds from the market title
    result = parse_bucket_from_title(title)
    if result:
        return result

    # Fallback: parse from ticker with asset-specific bucket widths
    half_width = CRYPTO_BUCKET_HALF_WIDTHS.get(asset, 250)

    if "-T" in ticker:
        try:
            threshold = float(ticker.split("-T")[-1])
            title_lower = title.lower()
            if "above" in title_lower or ">" in title_lower:
                return {"type": "above", "low": threshold, "high": 1e9}
            else:
                return {"type": "below", "low": 0, "high": threshold}
        except (ValueError, IndexError):
            pass

    if "-B" in ticker:
        try:
            center = float(ticker.split("-B")[-1])
            return {"type": "bucket", "low": center - half_width, "high": center + half_width}
        except (ValueError, IndexError):
            pass

    return None


# ---------------------------------------------------------------------------
# BTC log-normal pricing
# ---------------------------------------------------------------------------

def lognormal_cdf(x: float, mu: float, sigma: float) -> float:
    """CDF of log-normal distribution."""
    if x <= 0:
        return 0.0
    z = (math.log(x) - mu) / sigma
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def price_btc_bucket(
    current_price: float,
    bucket_low: float,
    bucket_high: float,
    annual_vol: float,
    hours_to_settlement: float,
) -> float:
    """
    Fair probability for BTC landing in [bucket_low, bucket_high] at settlement.
    Uses geometric Brownian motion (log-normal model).

    mu = ln(S) - 0.5 * sigma^2 * t  (risk-neutral drift = 0 for short horizons)
    sigma_period = annual_vol * sqrt(t)
    where t = hours_to_settlement / (365 * 24)
    """
    t = hours_to_settlement / (365.0 * 24.0)
    if t <= 0:
        t = 1.0 / (365.0 * 24.0)  # Minimum 1 hour

    sigma_period = annual_vol * math.sqrt(t)
    mu = math.log(current_price) - 0.5 * sigma_period ** 2

    p_high = lognormal_cdf(bucket_high, mu, sigma_period) if bucket_high < 1e9 else 1.0
    p_low = lognormal_cdf(bucket_low, mu, sigma_period) if bucket_low > 0 else 0.0

    prob = max(0.001, min(0.999, p_high - p_low))
    return prob


def parse_btc_bucket(ticker: str, title: str) -> Optional[dict]:
    """
    Parse BTC market ticker/title to extract bucket bounds.
    BTC tickers look like: KXBTC-26MAR20-T87500 or KXBTC-26MAR20-B87250
    Titles like: "Bitcoin above $87,500?" or "Bitcoin $87,000 to $87,500?"

    NOTE: For multi-asset crypto parsing, prefer parse_crypto_bucket(ticker, title, asset).
    This function is kept for backward compatibility and defaults to BTC bucket widths.
    """
    return parse_crypto_bucket(ticker, title, asset="BTC")


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
# Weather trading session
# ---------------------------------------------------------------------------

def run_weather_session(
    client: KalshiClient,
    balance: float,
    dry_run: bool,
    logger: logging.Logger,
) -> list[TradeDecision]:
    """
    Morning weather trading session.
    Scans all KXHIGH cities, blends NWS + ensemble, finds edges, places orders.
    """
    logger.info("--- WEATHER SESSION START ---")
    decisions = []

    # Load evolved strategy parameters from candidate_strategy.py
    strategy = load_strategy_params()
    logger.info(f"Strategy params: edge_threshold={strategy['edge_threshold']}c, "
                f"nws_weight={strategy['nws_weight']}, max_pos=${strategy['max_position']}")

    # Calculate deployment cap
    max_deploy = balance * MAX_DAILY_DEPLOY_PCT
    deployed_so_far = 0.0
    daily_loss_cap = balance * DAILY_LOSS_CAP_PCT

    # Discover all KXHIGH series by scanning known + discovering new
    series_tickers = list(NWS_GRID_POINTS.keys())

    for series_ticker in series_tickers:
        city = CITY_NAMES.get(series_ticker, series_ticker)
        logger.info(f"Processing {city} ({series_ticker})...")

        # 1. Fetch NWS forecast
        nws = fetch_nws_forecast(series_ticker, logger)

        # 2. Fetch GFS ensemble forecast
        ensemble = fetch_ensemble_forecast(series_ticker, logger)

        # 3. Fetch HRRR forecast (best for day-0 / day-1, silent fallback)
        hrrr = fetch_hrrr_forecast(series_ticker, logger)

        if not nws and not ensemble:
            logger.warning(f"  No forecast data for {city} -- skipping")
            continue

        # 4. Blend forecasts (HRRR + GFS + NWS with day-dependent weights)
        blended = blend_forecasts(nws, ensemble, logger, hrrr=hrrr, strategy=strategy)
        for date_str, fc in blended.items():
            logger.info(
                f"  {date_str}: blended={fc['temp']:.1f}F stdev={fc['stdev']:.2f} "
                f"({fc['source']})"
            )

        # 5. Fetch markets for this series
        try:
            resp = client.get_markets(series_ticker=series_ticker, limit=100, status="open")
            markets = resp.get("markets", [])
            logger.info(f"  Found {len(markets)} open markets")
        except Exception as e:
            logger.error(f"  Failed to fetch markets for {series_ticker}: {e}")
            continue

        # 6. Evaluate each market
        for m in markets:
            ticker = m.get("ticker", "")
            title = m.get("title", "")
            yes_bid = float(m.get("yes_bid_dollars", 0) or 0) * 100
            yes_ask = float(m.get("yes_ask_dollars", 0) or 0) * 100
            no_bid = float(m.get("no_bid_dollars", 0) or 0) * 100
            no_ask = float(m.get("no_ask_dollars", 0) or 0) * 100

            # Skip dead markets
            if yes_bid <= 0 and yes_ask <= 1:
                continue
            if yes_bid >= 99:
                continue

            # Parse market type
            mtype = parse_market_type(ticker, title)
            if not mtype:
                continue

            # Match to forecast date
            market_date = parse_ticker_date(ticker)
            if not market_date or market_date not in blended:
                continue

            fc = blended[market_date]
            forecast_temp = fc["temp"]
            stdev = fc["stdev"]

            # Calculate fair value
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

            # Evaluate both sides
            buy_yes_price = yes_ask if yes_ask > 0 else (yes_bid + 1)
            buy_no_price = no_ask if no_ask > 0 else ((100 - yes_bid) + 1 if yes_bid > 0 else 99)

            buy_yes_edge = fair_cents - buy_yes_price
            buy_no_edge = (100 - fair_cents) - buy_no_price

            # Choose best side (using evolved edge threshold from candidate_strategy)
            if buy_yes_edge >= strategy["edge_threshold"] and buy_yes_edge >= buy_no_edge:
                side = "yes"
                edge = buy_yes_edge
                price_cents = int(math.ceil(buy_yes_price))
            elif buy_no_edge >= strategy["edge_threshold"]:
                side = "no"
                edge = buy_no_edge
                price_cents = int(math.ceil(buy_no_price))
            else:
                # Log skip for markets near the forecast
                if abs(fair_cents - 50) < 30:  # Only log interesting skips
                    logger.debug(
                        f"  SKIP {ticker}: fair={fair_cents:.1f}c "
                        f"yes_edge={buy_yes_edge:+.1f}c no_edge={buy_no_edge:+.1f}c"
                    )
                continue

            # Clamp price to valid range
            price_cents = max(1, min(99, price_cents))

            # Position sizing: use evolved max_position from candidate_strategy
            cost_per_contract = price_cents / 100.0
            max_dollars = min(
                strategy["max_position"],
                max_deploy - deployed_so_far,
                daily_loss_cap,
            )

            if max_dollars < cost_per_contract:
                decisions.append(TradeDecision(
                    ticker=ticker, action=f"buy_{side}", strategy="weather",
                    edge_cents=round(edge, 1), fair_value_cents=round(fair_cents, 1),
                    market_price_cents=price_cents, price_to_pay_cents=price_cents,
                    contracts=0, max_loss_dollars=0, reason="deployment cap reached",
                    forecast_temp=round(forecast_temp, 1),
                ))
                continue

            contracts = max(1, min(
                int(max_dollars / cost_per_contract),
                int(2.0 / cost_per_contract),  # Hard cap $2
            ))
            actual_cost = contracts * cost_per_contract

            decision = TradeDecision(
                ticker=ticker,
                action=f"buy_{side}",
                strategy="weather",
                edge_cents=round(edge, 1),
                fair_value_cents=round(fair_cents, 1),
                market_price_cents=price_cents,
                price_to_pay_cents=price_cents,
                contracts=contracts,
                max_loss_dollars=round(actual_cost, 2),
                reason=(
                    f"{city} {market_date} | forecast={forecast_temp:.1f}F "
                    f"stdev={stdev:.2f} | fair={fair_cents:.1f}c "
                    f"edge=+{edge:.1f}c"
                ),
                forecast_temp=round(forecast_temp, 1),
            )

            if alert_big_edge and edge > 10:
                alert_big_edge(ticker, edge, city)

            # Check balance before placing
            current_balance = get_account_balance(client, logger)
            if current_balance is not None and actual_cost > current_balance:
                decision.reason += " | SKIP: insufficient balance"
                decision.contracts = 0
                decisions.append(decision)
                logger.warning(f"  SKIP {ticker}: cost ${actual_cost:.2f} > balance ${current_balance:.2f}")
                continue

            # Place order
            if not dry_run and contracts > 0:
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
                    order_id = result.get("order", {}).get("order_id", "unknown")
                    decision.placed = True
                    decision.order_id = order_id
                    deployed_so_far += actual_cost
                    logger.info(
                        f"  ORDER PLACED: {side.upper()} {ticker} x{contracts} "
                        f"@ {price_cents}c | edge +{edge:.1f}c | order_id={order_id}"
                    )
                    try:
                        from alerts import _send_telegram
                        _send_telegram(
                            f"<b>TRADE PLACED</b>\n\n"
                            f"{'BUY YES' if side == 'yes' else 'BUY NO'}  x{contracts}\n"
                            f"{ticker}\n"
                            f"Edge: +{edge:.0f}c  |  Cost: ${actual_cost:.2f}\n"
                            f"Forecast: {forecast_temp:.0f}F"
                        )
                    except Exception:
                        pass
                except Exception as e:
                    decision.error = str(e)
                    logger.error(f"  ORDER FAILED: {ticker} -- {e}")
            else:
                if contracts > 0:
                    deployed_so_far += actual_cost
                    logger.info(
                        f"  DRY-RUN: would buy {side.upper()} {ticker} x{contracts} "
                        f"@ {price_cents}c | edge +{edge:.1f}c | cost ${actual_cost:.2f}"
                    )

            decisions.append(decision)

        # Rate limit between cities
        time.sleep(1.0)

    logger.info(
        f"--- WEATHER SESSION DONE: {len(decisions)} decisions, "
        f"${deployed_so_far:.2f} deployed ---"
    )
    return decisions


# ---------------------------------------------------------------------------
# BTC trading session
# ---------------------------------------------------------------------------

def run_crypto_session(
    client: KalshiClient,
    balance: float,
    dry_run: bool,
    logger: logging.Logger,
    assets: list[str] = None,
) -> list[TradeDecision]:
    """
    Crypto trading session -- loops over BTC, ETH, SOL (or specified assets).
    Prices buckets using log-normal model for each asset with asset-specific vol.
    """
    # FORCE DRY-RUN for crypto until bucket parsing is fixed and validated.
    # The crypto bucket parser (parse_crypto_bucket) was broken as of 2026-03-22,
    # producing incorrect bucket boundaries. Trading with bad bucket probabilities
    # would cause systematic losses. Remove this override once the fix is validated.
    if not dry_run:
        logger.warning("CRYPTO FORCED TO DRY-RUN: bucket parsing not yet validated")
        dry_run = True

    if assets is None:
        assets = list(CRYPTO_ASSETS.keys())  # ["BTC", "ETH", "SOL"]

    # Load evolved strategy parameters from candidate_strategy.py
    strategy = load_strategy_params()
    crypto_edge_threshold = strategy["crypto_edge_threshold"]

    logger.info(f"--- CRYPTO SESSION START (assets: {', '.join(assets)}) | edge_threshold={crypto_edge_threshold}c ---")
    all_decisions = []

    for asset in assets:
        cfg = CRYPTO_ASSETS.get(asset, {})
        series = cfg.get("kalshi_series", f"KX{asset}")
        default_vol = cfg.get("default_annual_vol", 0.65)

        logger.info(f"  --- {asset} sub-session ---")

        # 1. Fetch price
        crypto_price = fetch_crypto_price_auto(asset, logger)
        if crypto_price is None:
            logger.error(f"Cannot run {asset} sub-session without price data")
            continue

        # 2. Fetch 30-day realized vol
        annual_vol = fetch_crypto_30d_vol(asset, logger)
        if annual_vol is None:
            logger.warning(f"Using fallback {asset} vol of {default_vol:.0%}")
            annual_vol = default_vol

        # 3. Fetch markets
        try:
            resp = client.get_markets(series_ticker=series, limit=100, status="open")
            markets = resp.get("markets", [])
            logger.info(f"Found {len(markets)} open {asset} markets ({series})")
        except Exception as e:
            logger.error(f"Failed to fetch {series} markets: {e}")
            continue

        if not markets:
            logger.info(f"No open {series} markets found, skipping {asset}")
            continue

        trades_placed = 0
        daily_loss_cap = balance * DAILY_LOSS_CAP_PCT

        for m in markets:
            if trades_placed >= MAX_CRYPTO_TRADES_PER_DAY:
                logger.info(f"{asset} trade cap reached ({MAX_CRYPTO_TRADES_PER_DAY})")
                break

            ticker = m.get("ticker", "")
            title = m.get("title", "")
            yes_bid = float(m.get("yes_bid_dollars", 0) or 0) * 100
            yes_ask = float(m.get("yes_ask_dollars", 0) or 0) * 100

            # Skip dead markets
            if yes_bid <= 0 and yes_ask <= 1:
                continue

            # Parse bucket (pass asset for correct bucket widths)
            bucket = parse_crypto_bucket(ticker, title, asset)
            if not bucket:
                continue

            # Get hours to settlement from close_time
            close_time_str = m.get("close_time", "")
            if close_time_str:
                try:
                    close_dt = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
                    hours_left = max(0.1, (close_dt - datetime.now(timezone.utc)).total_seconds() / 3600)
                except Exception:
                    hours_left = 24.0
            else:
                market_date = parse_ticker_date(ticker)
                hours_left = hours_until_date(market_date) if market_date else 24.0

            # Price the bucket (same lognormal model, asset-specific vol)
            fair_prob = price_btc_bucket(
                crypto_price, bucket["low"], bucket["high"], annual_vol, hours_left
            )
            fair_cents = fair_prob * 100

            market_yes_mid = (yes_bid + yes_ask) / 2 if yes_ask > 0 else yes_bid
            no_fair = 100 - fair_cents

            no_ask_price = float(m.get("no_ask_dollars", 0) or 0) * 100
            if no_ask_price <= 0:
                no_ask_price = 100 - yes_bid if yes_bid > 0 else 99

            buy_no_edge = no_fair - no_ask_price
            buy_yes_edge = fair_cents - yes_ask if yes_ask > 0 else 0

            if buy_no_edge >= crypto_edge_threshold:
                side = "no"
                edge = buy_no_edge
                price_cents = int(math.ceil(no_ask_price))
            elif buy_yes_edge >= crypto_edge_threshold:
                side = "yes"
                edge = buy_yes_edge
                price_cents = int(math.ceil(yes_ask))
            else:
                logger.debug(
                    f"  SKIP {asset} {ticker}: fair_yes={fair_cents:.1f}c "
                    f"mkt_yes={market_yes_mid:.0f}c no_edge={buy_no_edge:+.1f}c "
                    f"yes_edge={buy_yes_edge:+.1f}c"
                )
                continue

            price_cents = max(1, min(99, price_cents))
            cost_per_contract = price_cents / 100.0

            max_dollars = min(1.0, daily_loss_cap)
            contracts = max(1, int(max_dollars / cost_per_contract))
            actual_cost = contracts * cost_per_contract

            strategy_label = asset.lower()  # "btc", "eth", "sol"
            decision = TradeDecision(
                ticker=ticker,
                action=f"buy_{side}",
                strategy=strategy_label,
                edge_cents=round(edge, 1),
                fair_value_cents=round(fair_cents, 1),
                market_price_cents=price_cents,
                price_to_pay_cents=price_cents,
                contracts=contracts,
                max_loss_dollars=round(actual_cost, 2),
                reason=(
                    f"{asset}=${crypto_price:,.2f} vol={annual_vol:.1%} hrs={hours_left:.1f} | "
                    f"bucket=[{bucket['low']:,.0f}-{bucket['high']:,.0f}] | "
                    f"fair={fair_cents:.1f}c edge=+{edge:.1f}c"
                ),
            )

            if alert_big_edge and edge > 10:
                alert_big_edge(ticker, edge, asset)

            current_balance = get_account_balance(client, logger)
            if current_balance is not None and actual_cost > current_balance:
                decision.reason += " | SKIP: insufficient balance"
                decision.contracts = 0
                all_decisions.append(decision)
                logger.warning(f"  SKIP {asset} {ticker}: cost > balance")
                continue

            if not dry_run and contracts > 0:
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
                    order_id = result.get("order", {}).get("order_id", "unknown")
                    decision.placed = True
                    decision.order_id = order_id
                    trades_placed += 1
                    logger.info(
                        f"  {asset} ORDER: {side.upper()} {ticker} x{contracts} "
                        f"@ {price_cents}c | edge +{edge:.1f}c | order_id={order_id}"
                    )
                    try:
                        from alerts import _send_telegram
                        _send_telegram(
                            f"<b>TRADE PLACED</b>\n\n"
                            f"{'BUY YES' if side == 'yes' else 'BUY NO'}  x{contracts}\n"
                            f"{ticker}\n"
                            f"Edge: +{edge:.0f}c  |  Cost: ${actual_cost:.2f}\n"
                            f"{asset}=${crypto_price:,.0f}  |  Vol: {annual_vol:.0%}"
                        )
                    except Exception:
                        pass
                except Exception as e:
                    decision.error = str(e)
                    logger.error(f"  {asset} ORDER FAILED: {ticker} -- {e}")
            else:
                if contracts > 0:
                    trades_placed += 1
                    logger.info(
                        f"  DRY-RUN {asset}: would buy {side.upper()} {ticker} x{contracts} "
                        f"@ {price_cents}c | edge +{edge:.1f}c"
                    )

            all_decisions.append(decision)

        logger.info(f"  --- {asset} sub-session: {trades_placed} trades ---")

    logger.info(
        f"--- CRYPTO SESSION DONE: {len(all_decisions)} decisions ---"
    )
    return all_decisions


def run_btc_session(
    client: KalshiClient,
    balance: float,
    dry_run: bool,
    logger: logging.Logger,
) -> list[TradeDecision]:
    """Legacy wrapper: run crypto session for BTC only."""
    return run_crypto_session(client, balance, dry_run, logger, assets=["BTC"])


# ---------------------------------------------------------------------------
# Sports (NBA) trading session
# ---------------------------------------------------------------------------

def run_sports_session(
    client: KalshiClient,
    balance: float,
    dry_run: bool,
    logger: logging.Logger,
) -> list[TradeDecision]:
    """
    Sports/NBA trading session.
    Uses sports_strategy to find NBA edges on Kalshi, then places trades.
    """
    logger.info("--- SPORTS SESSION START ---")
    decisions = []

    # Load evolved strategy parameters from candidate_strategy.py
    strategy = load_strategy_params()
    sports_edge_threshold = strategy["sports_edge_threshold"]

    if find_nba_edges is None:
        logger.warning("sports_strategy not available, skipping sports session")
        logger.info("--- SPORTS SESSION DONE: 0 decisions, 0 trades ---")
        return decisions

    # 1. Find NBA edges using sports strategy
    try:
        edges = find_nba_edges(kalshi_client=client)
    except Exception as e:
        logger.error(f"Sports edge detection failed: {e}")
        logger.info("--- SPORTS SESSION DONE: 0 decisions, 0 trades ---")
        return decisions

    if not edges:
        logger.info("No NBA edges found")
        logger.info("--- SPORTS SESSION DONE: 0 decisions, 0 trades ---")
        return decisions

    logger.info(f"Found {len(edges)} NBA edges | edge_threshold={sports_edge_threshold}c")

    # ---- UNDERDOG FILTER (2026-03-22) ----
    # Data shows: underdogs (<30c) have 48% win rate with 6:1 payoff.
    # Mid-range and favorites LOSE money. Only trade Winner markets where
    # the market price < 30c (i.e. underdogs).
    pre_filter_count = len(edges)
    filtered_edges = []
    for e in edges:
        # Only trade "Winner" markets (skip Total Points, Spreads, props).
        # The sports_strategy already only returns game winner edges, but
        # double-check by filtering out anything with spread/total/points keywords.
        title_lower = (e.title or "").lower()
        if any(kw in title_lower for kw in ["total", "spread", "points", "over", "under", "prop"]):
            logger.debug(f"  SKIP (not winner market): {e.ticker} | {e.title}")
            continue
        # Only buy YES when market price < 30c (underdogs).
        # For buy_no edges, the market_price is the YES price, so the NO side
        # cost is (100 - market_price). We want the price WE pay to be < 30c.
        if e.side == "buy_yes" and e.market_price >= 30:
            logger.debug(f"  SKIP (not underdog, yes@{e.market_price:.0f}c): {e.ticker}")
            continue
        if e.side == "buy_no" and (100 - e.market_price) >= 30:
            logger.debug(f"  SKIP (not underdog, no@{100 - e.market_price:.0f}c): {e.ticker}")
            continue
        filtered_edges.append(e)
    edges = filtered_edges
    logger.info(f"After underdog filter (<30c): {len(edges)}/{pre_filter_count} edges remain")
    # ---- END UNDERDOG FILTER ----

    trades_placed = 0
    daily_loss_cap = balance * DAILY_LOSS_CAP_PCT

    for edge_obj in edges:
        if trades_placed >= MAX_SPORTS_TRADES_PER_DAY:
            logger.info(f"Sports trade cap reached ({MAX_SPORTS_TRADES_PER_DAY})")
            break

        # Only trade edges above threshold (evolved from candidate_strategy)
        if edge_obj.edge < sports_edge_threshold:
            continue

        ticker = edge_obj.ticker
        side = "yes" if edge_obj.side == "buy_yes" else "no"

        # Determine price to pay
        if side == "yes":
            price_cents = int(math.ceil(edge_obj.market_price))
        else:
            price_cents = int(math.ceil(100 - edge_obj.market_price))

        price_cents = max(1, min(99, price_cents))
        cost_per_contract = price_cents / 100.0

        # Size: max $1 per sports trade (conservative), capped at $2
        max_dollars = min(1.0, MAX_DOLLARS_PER_TRADE, daily_loss_cap)
        contracts = max(1, int(max_dollars / cost_per_contract))
        actual_cost = contracts * cost_per_contract

        decision = TradeDecision(
            ticker=ticker,
            action=f"buy_{side}",
            strategy="sports",
            edge_cents=round(edge_obj.edge, 1),
            fair_value_cents=round(edge_obj.fair_value, 1),
            market_price_cents=price_cents,
            price_to_pay_cents=price_cents,
            contracts=contracts,
            max_loss_dollars=round(actual_cost, 2),
            reason=(
                f"{edge_obj.team_a} vs {edge_obj.team_b} | "
                f"model={edge_obj.model_win_prob:.1%} spread={edge_obj.model_spread:+.1f} | "
                f"fair={edge_obj.fair_value:.1f}c edge=+{edge_obj.edge:.1f}c "
                f"conf={edge_obj.confidence}"
            ),
        )

        # Balance check
        current_balance = get_account_balance(client, logger)
        if current_balance is not None and actual_cost > current_balance:
            decision.reason += " | SKIP: insufficient balance"
            decision.contracts = 0
            decisions.append(decision)
            logger.warning(f"  SKIP SPORTS {ticker}: cost > balance")
            continue

        # Place order
        if not dry_run and contracts > 0:
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
                order_id = result.get("order", {}).get("order_id", "unknown")
                decision.placed = True
                decision.order_id = order_id
                trades_placed += 1
                logger.info(
                    f"  SPORTS ORDER: {side.upper()} {ticker} x{contracts} "
                    f"@ {price_cents}c | edge +{edge_obj.edge:.1f}c | order_id={order_id}"
                )
            except Exception as e:
                decision.error = str(e)
                logger.error(f"  SPORTS ORDER FAILED: {ticker} -- {e}")
        else:
            if contracts > 0:
                trades_placed += 1
                logger.info(
                    f"  DRY-RUN SPORTS: would buy {side.upper()} {ticker} x{contracts} "
                    f"@ {price_cents}c | edge +{edge_obj.edge:.1f}c"
                )

        decisions.append(decision)

    logger.info(
        f"--- SPORTS SESSION DONE: {len(decisions)} decisions, "
        f"{trades_placed} trades ---"
    )
    return decisions


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

    # Only arb on series where we have proven edges or true risk-free arb
    # Exclude KXNBAPTS (no player-level edge) and other unproven series
    ARB_ALLOWED_SERIES = {
        "KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHDEN", "KXHIGHDC", "KXHIGHLA",
        "KXNBAGAME",  # game winners only (props excluded)
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

        # Filter: only arb on allowed series (no KXNBAPTS props, no unproven markets)
        opp_series = opp.ticker.split("-")[0] if hasattr(opp, "ticker") and "-" in opp.ticker else ""
        if opp_series and opp_series not in ARB_ALLOWED_SERIES:
            continue

        # Only trade high-confidence, high-edge opportunities (evolved from candidate_strategy)
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

            # Size: enough pairs to deploy up to $1, capped by daily loss cap
            max_dollars = min(1.0, daily_loss_cap)
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

            # Balance check
            current_balance = get_account_balance(client, logger)
            if current_balance is not None and actual_cost > current_balance:
                decision_yes.reason += " | SKIP: insufficient balance"
                decision_yes.contracts = 0
                decision_no.contracts = 0
                decisions.extend([decision_yes, decision_no])
                logger.warning(f"  SKIP ARB {opp.ticker}: cost ${actual_cost:.2f} > balance")
                continue

            # Place both sides
            if not dry_run and pairs > 0:
                for dec, side, price_c in [
                    (decision_yes, "yes", int(yes_price)),
                    (decision_no, "no", int(no_price)),
                ]:
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
                        logger.info(
                            f"  ARB ORDER: {side.upper()} {opp.ticker} x{pairs} "
                            f"@ {price_c}c | total_edge +{opp.edge_cents:.1f}c | order_id={order_id}"
                        )
                    except Exception as e:
                        dec.error = str(e)
                        logger.error(f"  ARB ORDER FAILED: {side} {opp.ticker} -- {e}")
                trades_placed += 1
            else:
                if pairs > 0:
                    trades_placed += 1
                    logger.info(
                        f"  DRY-RUN ARB: would buy BOTH sides {opp.ticker} x{pairs} "
                        f"(YES@{yes_price}c + NO@{no_price}c = {total_cost_cents}c) "
                        f"| edge +{opp.edge_cents:.1f}c | guaranteed ${opp.edge_cents * pairs / 100:.2f} profit"
                    )

            decisions.extend([decision_yes, decision_no])

        else:
            # For cross_event_arb and wide_spread: log but don't auto-trade
            # (these require more complex execution)
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
        f"{trades_placed} trades ---"
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
                # To sell YES position: sell yes at yes_bid
                # To sell NO position: sell no (= buy yes at yes_ask... actually sell)
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
# Tail Fade session (Strategy 1)
# ---------------------------------------------------------------------------

# Series tickers to scan for tail fade opportunities
TAIL_FADE_SERIES = {
    "weather": ["KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHLA", "KXHIGHDC", "KXHIGHDEN"],
    "crypto":  ["KXBTC", "KXETH", "KXSOL"],
    "nba":     ["KXNBA", "KXNBAGAME"],
}

MAX_TAIL_FADE_TRADES = 10  # cap per session


def run_tail_fade_session(
    client: KalshiClient,
    balance: float,
    dry_run: bool,
    logger: logging.Logger,
) -> list[TradeDecision]:
    """
    Tail Fade session -- buy NO on extreme tails and mid-range overpriced YES.

    Strategy:
      - YES <= TAIL_FADE_MAX_PRICE (e.g. 5c): buy NO @ (100 - yes_ask) -- fade the tail
      - YES in [MID_LOW, MID_HIGH] (e.g. 30-50c): buy NO @ (100 - yes_ask) -- fade mid-range
    All orders are post_only limit orders, max $2 per trade.
    """
    logger.info("--- TAIL FADE SESSION START ---")
    decisions = []

    strategy = load_strategy_params()
    max_price = strategy["tail_fade_max_price"]
    mid_low = strategy["tail_fade_mid_low"]
    mid_high = strategy["tail_fade_mid_high"]
    weather_on = strategy["tail_fade_weather_enabled"]
    crypto_on = strategy["tail_fade_crypto_enabled"]
    nba_on = strategy["tail_fade_nba_enabled"]

    logger.info(
        f"  Params: tail_max={max_price}c mid=[{mid_low}-{mid_high}]c "
        f"weather={'ON' if weather_on else 'OFF'} crypto={'ON' if crypto_on else 'OFF'} "
        f"nba={'ON' if nba_on else 'OFF'}"
    )

    # Build list of series to scan
    series_to_scan = []
    if weather_on:
        series_to_scan.extend(TAIL_FADE_SERIES["weather"])
    if crypto_on:
        series_to_scan.extend(TAIL_FADE_SERIES["crypto"])
    if nba_on:
        series_to_scan.extend(TAIL_FADE_SERIES["nba"])

    if not series_to_scan:
        logger.info("  All categories disabled, nothing to scan")
        logger.info("--- TAIL FADE SESSION DONE: 0 decisions, 0 trades ---")
        return decisions

    trades_placed = 0

    for series_ticker in series_to_scan:
        if trades_placed >= MAX_TAIL_FADE_TRADES:
            break

        # Fetch open markets for this series
        try:
            resp = client.get_markets(series_ticker=series_ticker, status="open", limit=100)
            markets = resp.get("markets", [])
        except Exception as e:
            logger.warning(f"  Failed to fetch markets for {series_ticker}: {e}")
            continue

        for mkt in markets:
            if trades_placed >= MAX_TAIL_FADE_TRADES:
                break

            ticker = mkt.get("ticker", "")
            yes_bid = float(mkt.get("yes_bid", 0) or 0)
            yes_ask = float(mkt.get("yes_ask", 0) or 0)
            volume = int(mkt.get("volume", 0) or 0)

            # Kalshi API returns prices in cents (integer) on some endpoints,
            # dollars on others. Normalize: if < 1.0, it's dollars -> convert.
            if 0 < yes_bid < 1.0:
                yes_bid = yes_bid * 100
            if 0 < yes_ask < 1.0:
                yes_ask = yes_ask * 100

            # Skip dead markets
            if yes_ask <= 0:
                continue

            # Determine fade zone
            fade_zone = None
            if yes_ask <= max_price:
                fade_zone = "tail"
            elif mid_low <= yes_ask <= mid_high:
                fade_zone = "mid"
            else:
                continue

            # NO price = 100 - yes_ask
            no_price_cents = int(100 - yes_ask)
            if no_price_cents < 1 or no_price_cents > 99:
                continue

            # Size: max $2 per trade
            cost_per_contract = no_price_cents / 100.0
            contracts = max(1, int(min(MAX_DOLLARS_PER_TRADE, balance * 0.02) / cost_per_contract))
            actual_cost = contracts * cost_per_contract

            # Edge estimate: for tail fades, edge = (100 - fair) - no_price.
            # We approximate fair YES as ~0 for tails, ~40c for mid-range.
            if fade_zone == "tail":
                approx_fair_yes = 1.0  # near-zero probability
                edge = (100 - approx_fair_yes) - no_price_cents
            else:
                approx_fair_yes = (mid_low + mid_high) / 2.0
                edge = (100 - approx_fair_yes) - no_price_cents

            decision = TradeDecision(
                ticker=ticker,
                action="buy_no",
                strategy="tail_fade",
                edge_cents=round(edge, 1),
                fair_value_cents=round(100 - approx_fair_yes, 1),
                market_price_cents=int(yes_ask),
                price_to_pay_cents=no_price_cents,
                contracts=contracts,
                max_loss_dollars=round(actual_cost, 2),
                reason=f"TAIL_FADE ({fade_zone}): YES@{yes_ask:.0f}c -> BUY NO@{no_price_cents}c | {series_ticker}",
            )

            # Balance check
            current_balance = get_account_balance(client, logger)
            if current_balance is not None and actual_cost > current_balance:
                decision.reason += " | SKIP: insufficient balance"
                decision.contracts = 0
                decisions.append(decision)
                continue

            if not dry_run and contracts > 0:
                try:
                    result = client.place_order(
                        ticker=ticker,
                        side="no",
                        action="buy",
                        count=contracts,
                        type="limit",
                        no_price=no_price_cents,
                        post_only=True,
                    )
                    order_id = result.get("order", {}).get("order_id", "unknown")
                    decision.placed = True
                    decision.order_id = order_id
                    trades_placed += 1
                    logger.info(
                        f"  TAIL_FADE ORDER ({fade_zone}): BUY NO {ticker} x{contracts} "
                        f"@ {no_price_cents}c (YES@{yes_ask:.0f}c) | order_id={order_id}"
                    )
                except Exception as e:
                    decision.error = str(e)
                    logger.error(f"  TAIL_FADE ORDER FAILED: {ticker} -- {e}")
            else:
                if contracts > 0:
                    trades_placed += 1
                    logger.info(
                        f"  DRY-RUN TAIL_FADE ({fade_zone}): would BUY NO {ticker} x{contracts} "
                        f"@ {no_price_cents}c (YES@{yes_ask:.0f}c)"
                    )

            decisions.append(decision)
            time.sleep(0.3)  # Rate limit

    logger.info(
        f"--- TAIL FADE SESSION DONE: {len(decisions)} decisions, "
        f"{trades_placed} trades ---"
    )
    return decisions


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
                yes_bid = float(mkt.get("yes_bid", 0) or 0)
                # Normalize: if < 1.0, it's dollars
                if 0 < yes_bid < 1.0:
                    yes_bid = yes_bid * 100
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

            # Check balance: selling YES requires margin = max loss per leg
            # Max loss on selling 1 YES at price P = (100 - P) cents
            # But since we sell ALL legs, guaranteed payout is 100c, received sum(yes_bid)
            # Net guaranteed profit = sum(yes_bid) - 100c.  No margin needed beyond
            # what Kalshi holds (they net the positions).
            # Conservative: check we have at least $2 available.
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

    if balance < 1.0:
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

    # --- Weather session ---
    # DISABLED 2026-03-23: Mid-range weather has NEGATIVE Kelly (-8.6%).
    # Market is 96.4% calibrated at 40-60c. No edge exists here.
    # Weather tails (buy NO on cheap YES) run separately below.
    if False and weather:
        try:
            weather_decisions = run_weather_session(client, balance, dry_run, logger)
            all_decisions.extend(weather_decisions)
        except Exception as e:
            logger.error(f"Weather session crashed: {e}")
            logger.error(traceback.format_exc())
            if alert_bot_error:
                alert_bot_error("auto_trade", str(e))
    else:
        logger.info("--- WEATHER SESSION SKIPPED (no proven mid-range edge) ---")

    # --- Crypto session (BTC + ETH + SOL) ---
    # DISABLED 2026-03-23: Only 3-5 days of crypto data. Cannot validate edge.
    if False and btc:
        try:
            crypto_decisions = run_crypto_session(client, balance, dry_run, logger)
            all_decisions.extend(crypto_decisions)
        except Exception as e:
            logger.error(f"Crypto session crashed: {e}")
            logger.error(traceback.format_exc())
            if alert_bot_error:
                alert_bot_error("auto_trade", str(e))
    else:
        logger.info("--- CRYPTO SESSION SKIPPED (insufficient data, no proven edge) ---")

    # --- Sports session ---
    # DISABLED 2026-03-23: Sports session places NBA props (no player-level data,
    # -$17 P&L) and non-underdog bets (no proven edge). NBA underdog YES bets
    # are the only proven sports edge — handled by arb + manual for now.
    # TODO: Create a focused underdog-only session once we have n>200 settled.
    if False and sports:
        try:
            sports_decisions = run_sports_session(client, balance, dry_run, logger)
            all_decisions.extend(sports_decisions)
        except Exception as e:
            logger.error(f"Sports session crashed: {e}")
            logger.error(traceback.format_exc())
    else:
        logger.info("--- SPORTS SESSION SKIPPED (props have no edge, underdog-only session needed) ---")

    # --- NBA Underdog session (focused, proven edge) ---
    # Buys YES on KXNBAGAME underdogs priced 10-30c.
    # Historical: 29.7% win rate vs 22% implied = +7.7pp edge, ROI +33%, Sharpe 2.64.
    # Conservative sizing: quarter Kelly, max $2/trade, max 5 bets/day.
    if find_nba_underdogs is not None:
        try:
            underdogs = find_nba_underdogs(client)
            budget = nba_risk_budget(balance)
            underdog_count = 0
            for ud in underdogs:
                if underdog_count >= budget.get("max_daily_bets", 5):
                    logger.info(f"NBA underdog daily cap reached ({underdog_count})")
                    break
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
                        result = client.place_order(
                            ticker=decision.ticker,
                            side="yes",
                            action="buy",
                            count=contracts,
                            type="limit",
                            yes_price=price_cents,
                        )
                        decision.placed = True
                        decision.order_id = result.get("order", {}).get("order_id", "")
                        logger.info(
                            f"NBA UNDERDOG: {decision.ticker} BUY YES x{contracts} "
                            f"@{price_cents}c edge={ud.get('edge_pp',0):+.1f}pp"
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

    # --- Tail Fade session ---
    # DISABLED 2026-03-23: Replaced by weather_tail_strategy which uses
    # NWS forecast + dutch book math for validated edge calculation.
    # Old tail fade traded mid-range (30-50c) which has no proven edge.
    if False:
        try:
            tail_fade_decisions = run_tail_fade_session(client, balance, dry_run, logger)
            all_decisions.extend(tail_fade_decisions)
        except Exception as e:
            logger.error(f"Tail fade session crashed: {e}")
            logger.error(traceback.format_exc())
            if alert_bot_error:
                alert_bot_error("auto_trade", str(e))
    else:
        logger.info("--- TAIL FADE SESSION SKIPPED (replaced by weather_tail_strategy) ---")

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
                contracts = min(tt.get("suggested_contracts", 1), risk.get("max_contracts", 5))
                if contracts <= 0:
                    continue
                price_cents = int(tt.get("no_price_cents", 97))
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

    # --- Momentum session (DRY-RUN data collection only) ---
    # DEPLOYED 2026-03-23: Logs momentum signals from price_snapshots.jsonl.
    # ALWAYS runs in data-collection mode (never places live trades) until
    # we have 7+ days of 60-second NBA snapshot data to validate the strategy.
    # See momentum_strategy.py for details on why backtest is inconclusive.
    if detect_momentum_signals is not None:
        try:
            signals = detect_momentum_signals()
            if signals:
                logger.info(f"--- MOMENTUM SESSION (DRY-RUN ONLY) ---")
                momentum_decisions = generate_momentum_trades(
                    client=client,
                    signals=signals,
                    bankroll=balance,
                    dry_run=True,  # ALWAYS dry-run until data validates strategy
                )
                for md in momentum_decisions:
                    logger.info(
                        f"  [MOMENTUM SIGNAL] {md['action']:8s} {md['ticker']:35s} "
                        f"delta={md['signal']['delta_cents']:+.0f}c "
                        f"({md['signal']['prev_price']:.2f}->{md['signal']['curr_price']:.2f})"
                    )
                    decision = TradeDecision(
                        ticker=md["ticker"],
                        action=md["action"],
                        strategy="momentum",
                        edge_cents=md["edge_cents"],
                        fair_value_cents=md["fair_value_cents"],
                        market_price_cents=md["market_price_cents"],
                        price_to_pay_cents=md["price_to_pay_cents"],
                        contracts=0,  # 0 contracts = signal logged but not traded
                        max_loss_dollars=0,
                        reason=md["reason"],
                        placed=False,
                    )
                    all_decisions.append(decision)
                logger.info(f"--- MOMENTUM SESSION DONE: {len(momentum_decisions)} signals logged (not traded) ---")
            else:
                logger.info("--- MOMENTUM SESSION: no signals detected ---")
        except Exception as e:
            logger.error(f"Momentum session crashed: {e}")
            logger.error(traceback.format_exc())
    else:
        logger.info("--- MOMENTUM SESSION SKIPPED (module not available) ---")

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

    # --- Discovery Trader session ---
    # DISABLED 2026-03-24: Discovery trader places random bets on unproven markets
    # (parlays, props, esports, Trump mentions). -$173 P&L with 33% win rate.
    # No model, no edge — pure speculation. Will be replaced by Karpathy-style
    # autoresearch that discovers and validates strategies before trading them.
    if False:
        try:
            from discovery_trader import execute_discovery_session
            discovery_decisions = execute_discovery_session(client, balance, dry_run, logger)
            for dd in discovery_decisions:
                all_decisions.append(TradeDecision(
                    ticker=dd["ticker"],
                    action=dd["action"],
                    strategy=dd["strategy"],
                    edge_cents=dd.get("edge_cents", 0),
                    fair_value_cents=dd.get("fair_value_cents", 50),
                    market_price_cents=dd.get("market_price_cents", 50),
                    price_to_pay_cents=dd.get("price_to_pay_cents", 50),
                    contracts=dd.get("contracts", 0),
                    max_loss_dollars=dd.get("max_loss_dollars", 0),
                    reason=dd.get("reason", ""),
                    placed=dd.get("placed", False),
                    order_id=dd.get("order_id", ""),
                    error=dd.get("error", ""),
                ))
        except ImportError:
            pass
        except Exception:
            pass
    else:
        logger.info("--- DISCOVERY TRADER SESSION DISABLED (no proven edge, -$173 P&L) ---")

    # --- Copy-trade session ---
    # DISABLED 2026-03-23: Copy-trade has only been simulated, never validated
    # with live data. Re-enable once we have real copy-trade P&L data.
    if False and copy:
        try:
            from copy_trade import get_copy_signals
            copy_signals = get_copy_signals(client)
            if copy_signals:
                logger.info(f"--- COPY-TRADE SESSION START ---")
                copy_count = 0
                for sig in copy_signals:
                    if copy_count >= 5:
                        logger.info("Copy-trade cap reached (5)")
                        break
                    ticker = sig.get("ticker", "")
                    side = sig.get("side", "yes")
                    edge = sig.get("edge", 0)
                    confidence = sig.get("confidence", 0)
                    reason = sig.get("reason", "")
                    source = sig.get("source_trader", "unknown")

                    # Get current market price
                    try:
                        ob = client.get_market_orderbook(ticker, depth=1)
                        book = ob.get("orderbook", {})
                        if side == "yes":
                            price_cents = book.get("yes", [[99]])[0][0] if book.get("yes") else 99
                        else:
                            price_cents = book.get("no", [[99]])[0][0] if book.get("no") else 99
                    except Exception:
                        price_cents = 50

                    contracts = max(1, int(MAX_DOLLARS_PER_TRADE / (price_cents / 100.0))) if price_cents > 0 else 1
                    cost = contracts * price_cents / 100.0
                    if cost > MAX_DOLLARS_PER_TRADE:
                        contracts = max(1, int(MAX_DOLLARS_PER_TRADE / (price_cents / 100.0)))
                        cost = contracts * price_cents / 100.0

                    decision = TradeDecision(
                        ticker=ticker,
                        action=f"buy_{side}",
                        strategy="copy",
                        edge_cents=edge,
                        fair_value_cents=0,
                        market_price_cents=price_cents,
                        price_to_pay_cents=price_cents,
                        contracts=contracts,
                        max_loss_dollars=cost,
                        reason=f"Copy {source} (conf={confidence:.0%}): {reason}",
                    )

                    if not dry_run:
                        try:
                            bal = get_account_balance(client, logger)
                            if bal and bal > cost + 1.0:
                                resp = client.place_order(
                                    ticker=ticker, side=side, action="buy",
                                    count=contracts, type="limit",
                                    **({f"{side}_price": price_cents}),
                                )
                                decision.placed = True
                                decision.order_id = resp.get("order", {}).get("order_id", "")
                                copy_count += 1
                        except Exception as e:
                            decision.error = str(e)
                            logger.error(f"Copy-trade order failed: {e}")
                    else:
                        logger.info(f"  DRY-RUN COPY: {side.upper()} {ticker} x{contracts} @ {price_cents}c | {reason}")
                        copy_count += 1

                    all_decisions.append(decision)
                logger.info(f"--- COPY-TRADE SESSION DONE: {copy_count} trades ---")
        except (ImportError, OSError):
            pass  # copy_trade module not available
        except Exception as e:
            logger.error(f"Copy-trade session crashed: {e}")
            logger.error(traceback.format_exc())
            if alert_bot_error:
                alert_bot_error("auto_trade", str(e))

    # --- Drawdown check ---
    total_risk = sum(d.max_loss_dollars for d in all_decisions if d.placed or (dry_run and d.contracts > 0))
    if balance > 0:
        dd_pct = (total_risk / balance) * 100
        if alert_drawdown and dd_pct >= 5.0:
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
        "--weather-only", action="store_true",
        help="Run only the weather session",
    )
    parser.add_argument(
        "--btc-only", action="store_true",
        help="Run only the BTC session",
    )
    parser.add_argument(
        "--sports-only", action="store_true",
        help="Run only the sports/NBA session",
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
    only_flags = [args.weather_only, args.btc_only, args.sports_only, args.arb_only]
    any_only = any(only_flags)
    do_weather = args.weather_only if any_only else True
    do_btc = args.btc_only if any_only else True
    do_sports = args.sports_only if any_only else True
    do_arb = args.arb_only if any_only else True

    no_confirm = args.no_confirm if hasattr(args, 'no_confirm') else False

    if is_live and not no_confirm:
        print("\n*** LIVE MODE -- Real orders will be placed ***")
        print(f"    Environment: {config.KALSHI_ENV}")
        print(f"    Max per trade: ${MAX_DOLLARS_PER_TRADE}")
        confirm = input("    Type 'YES' to continue: ").strip()
        if confirm != "YES":
            print("Aborted.")
            sys.exit(0)

    run_auto_trade(dry_run=not is_live, weather=do_weather, btc=do_btc, sports=do_sports, arb=do_arb)
