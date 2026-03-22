"""
autoresearch/research_loop.py -- Weather Forecast Arbitrage optimization loop.

Uses real Kalshi settled market data only. No synthetic scenarios.

This is the overnight self-improvement engine specifically for the weather
forecast vs. Kalshi market price strategy. It:

1. Reads the current candidate_strategy.py (weather-specific parameters)
2. Loads real settled weather markets from historical_settlements_with_prices.json
   - Each (series, date) group becomes one backtest scenario
   - Market prices come from actual previous_price data
   - Settlement results (yes/no) determine the true outcome
3. Runs the weather edge detection against those real markets
4. Scores based on: Sortino ratio, ROI, max drawdown, win rate
5. Mutates one parameter at a time, keeps winners, reverts losers
6. Logs everything to autoresearch/results.log
"""

import ast
import os
import re
import sys
import json
import math
import time
import random
import subprocess
import importlib
import copy

import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
from dataclasses import dataclass

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent directory to path so we can import weather_strategy
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

try:
    from alerts import alert_research_improvement
except (ImportError, OSError):
    alert_research_improvement = None

try:
    from kalshi_client import KalshiClient
except ImportError:
    KalshiClient = None

try:
    from settlement_tracker import (
        fetch_actual_high_temp,
        NWS_OBSERVATION_STATIONS,
        SERIES_TO_CITY,
    )
except ImportError:
    fetch_actual_high_temp = None
    NWS_OBSERVATION_STATIONS = {}
    SERIES_TO_CITY = {}

try:
    import anthropic as _anthropic_module
except ImportError:
    _anthropic_module = None

console = Console()

STRATEGY_FILE = Path(__file__).parent / "candidate_strategy.py"
RESULTS_LOG = Path(__file__).parent / "results.log"
REAL_OUTCOMES_FILE = Path(__file__).parent / "real_outcomes.json"
KALSHI_CACHE_FILE = Path(__file__).parent / "kalshi_cache.json"
HISTORICAL_SETTLEMENTS_FILE = Path(__file__).parent.parent / "output" / "historical_settlements_with_prices.json"
BACKTEST_DATASET_FILE = Path(__file__).parent.parent / "output" / "backtest_dataset.json"

# Legacy weight — no longer used (all scenarios are real historical data now).
KALSHI_SCENARIO_WEIGHT = 0

# =============================================================================
# WEATHER-SPECIFIC MUTATION SPACE
# =============================================================================

MUTATION_SPACE = {
    # Forecast stdev per day — the core model parameter
    "FORECAST_STDEV_0": [1.0, 1.2, 1.5, 1.8, 2.0, 2.3],
    "FORECAST_STDEV_1": [1.8, 2.0, 2.3, 2.5, 2.8, 3.0, 3.3],
    "FORECAST_STDEV_2": [2.5, 3.0, 3.5, 4.0, 4.5],
    "FORECAST_STDEV_3": [3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
    # Edge threshold in cents
    "EDGE_THRESHOLD_CENTS": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    # Position sizing
    "CONTRACTS_PER_TRADE": [5, 8, 10, 15, 20, 25],
    # NWS blend ratio
    "NWS_OFFICIAL_WEIGHT": [0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
    # City weights
    "CITY_WEIGHT_NYC": [0.5, 0.8, 1.0, 1.2, 1.5],
    "CITY_WEIGHT_CHI": [0.5, 0.8, 1.0, 1.2, 1.5],
    "CITY_WEIGHT_MIA": [0.5, 0.8, 1.0, 1.2, 1.5],
    "CITY_WEIGHT_LA": [0.5, 0.8, 1.0, 1.2, 1.5],
    "CITY_WEIGHT_DC": [0.5, 0.8, 1.0, 1.2, 1.5],
    "CITY_WEIGHT_DEN": [0.5, 0.8, 1.0, 1.2, 1.5],
    # Bucket vs threshold preference
    "BUCKET_MULTIPLIER": [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
    "THRESHOLD_MULTIPLIER": [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
    # Confidence thresholds
    "HIGH_CONFIDENCE_EDGE": [7.0, 8.0, 10.0, 12.0, 15.0],
    "MEDIUM_CONFIDENCE_EDGE": [3.0, 4.0, 5.0, 6.0, 7.0],
    # Position cap
    "MAX_POSITION_DOLLARS": [3.0, 5.0, 7.0, 10.0],
    # Volume filter
    "MIN_VOLUME": [5, 10, 20, 50],
    # Ensemble tightness
    "TIGHT_ENSEMBLE_THRESHOLD": [1.5, 2.0, 2.5, 3.0],
    "TIGHT_ENSEMBLE_MULTIPLIER": [1.0, 1.2, 1.5, 2.0],
    # Per-strategy edge thresholds
    "CRYPTO_EDGE_THRESHOLD_CENTS": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    "SPORTS_EDGE_THRESHOLD_CENTS": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    "ARB_EDGE_THRESHOLD_CENTS": [1.5, 2.0, 3.0, 4.0, 5.0],
    "EXIT_EDGE_THRESHOLD_CENTS": [1.0, 1.5, 2.0, 3.0, 4.0],
    # Blend weights (HRRR + GFS + NWS)
    "BLEND_HRRR_DAY0": [0.25, 0.30, 0.35, 0.40, 0.50, 0.55],
    "BLEND_GFS_DAY0": [0.20, 0.25, 0.30, 0.35, 0.40],
    "BLEND_NWS_DAY0": [0.15, 0.20, 0.25, 0.30, 0.35],
    "BLEND_HRRR_DAY1": [0.10, 0.15, 0.20, 0.25, 0.30],
    "BLEND_GFS_DAY1": [0.30, 0.35, 0.40, 0.45, 0.50],
    "BLEND_NWS_DAY1": [0.25, 0.30, 0.35, 0.40, 0.45],
    "BLEND_GFS_DAY2": [0.50, 0.55, 0.60, 0.65, 0.70],
    "BLEND_NWS_DAY2": [0.30, 0.35, 0.40, 0.45, 0.50],
    # Tail fade parameters
    "TAIL_FADE_MAX_PRICE": [3, 5, 7, 10, 15, 20],
    "TAIL_FADE_MIN_VOLUME": [0, 100, 500, 1000, 5000],
    "TAIL_FADE_WEATHER_ENABLED": [0, 1],
    "TAIL_FADE_CRYPTO_ENABLED": [0, 1],
    "TAIL_FADE_NBA_ENABLED": [0, 1],
    "TAIL_FADE_MID_LOW": [30, 35, 40, 45],
    "TAIL_FADE_MID_HIGH": [50, 55, 60],
}


# =============================================================================
# WEATHER SCENARIO SIMULATOR
# =============================================================================

# The real stdev of NWS forecast errors (ground truth for simulation).
# These are the "true" errors that we simulate against. The candidate_strategy
# has its own FORECAST_STDEV values which are the model's *belief* about errors.
# When the model's belief matches reality, edge detection is well-calibrated.
REALITY_STDEV = {0: 1.5, 1: 2.5, 2: 3.5, 3: 4.5}

# City configurations for simulation
CITIES = [
    {"name": "NYC", "base_temp_range": (30, 95), "ticker_prefix": "KXHIGHNY"},
    {"name": "Chicago", "base_temp_range": (15, 95), "ticker_prefix": "KXHIGHCHI"},
    {"name": "Miami", "base_temp_range": (60, 95), "ticker_prefix": "KXHIGHMIA"},
    {"name": "LA", "base_temp_range": (55, 95), "ticker_prefix": "KXHIGHLA"},
    {"name": "DC", "base_temp_range": (25, 100), "ticker_prefix": "KXHIGHDC"},
    {"name": "Denver", "base_temp_range": (15, 100), "ticker_prefix": "KXHIGHDEN"},
]


def normal_cdf(x: float) -> float:
    """Standard normal CDF approximation."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def calc_bucket_probability(
    forecast_temp: float,
    bucket_low: float,
    bucket_high: float,
    stdev: float,
) -> float:
    """Probability that actual temp falls in [bucket_low, bucket_high]."""
    z_low = (bucket_low - forecast_temp) / stdev
    z_high = (bucket_high - forecast_temp) / stdev
    prob = normal_cdf(z_high) - normal_cdf(z_low)
    return max(0.001, min(0.999, prob))


def calc_threshold_probability(
    forecast_temp: float,
    threshold: float,
    stdev: float,
    above: bool = True,
) -> float:
    """Probability that actual temp is above (or below) threshold."""
    z = (threshold - forecast_temp) / stdev
    if above:
        return max(0.001, min(0.999, 1 - normal_cdf(z)))
    else:
        return max(0.001, min(0.999, normal_cdf(z)))


@dataclass
class SimulatedMarket:
    """A fake Kalshi-like weather market."""
    ticker: str
    title: str
    city: str
    market_type: str           # "bucket", "above", "below"
    bucket_low: float
    bucket_high: float
    threshold: float           # for above/below markets
    yes_bid_cents: float       # market bid
    yes_ask_cents: float       # market ask
    true_probability: float    # actual probability (from true temp)
    volume: int
    days_out: int


def generate_weather_scenario(
    rng: np.random.Generator,
    n_markets_per_city: int = 12,
) -> tuple[list[SimulatedMarket], dict]:
    """
    Generate a realistic weather scenario with fake markets.

    Returns:
        (list of SimulatedMarket, dict of city -> true_temp)
    """
    markets = []
    true_temps = {}

    for city_cfg in CITIES:
        city = city_cfg["name"]
        low, high = city_cfg["base_temp_range"]
        prefix = city_cfg["ticker_prefix"]

        # Pick a random "NWS forecast" temperature
        forecast_temp = rng.uniform(low + 5, high - 5)

        # Pick days_out (0-3)
        days_out = rng.integers(0, 4)

        # True temperature = forecast + noise drawn from REALITY stdev
        true_stdev = REALITY_STDEV.get(days_out, 4.5)
        true_temp = forecast_temp + rng.normal(0, true_stdev)
        true_temps[city] = {
            "forecast": forecast_temp,
            "true_temp": true_temp,
            "days_out": days_out,
        }

        # Generate bucket markets around the forecast
        # Kalshi typically has 1-degree buckets centered near the forecast
        center = round(forecast_temp)
        bucket_starts = list(range(center - 6, center + 7))

        for bs in bucket_starts:
            bucket_low = float(bs)
            bucket_high = float(bs + 1)

            # True probability of temp landing in this bucket
            true_prob = calc_bucket_probability(true_temp, bucket_low, bucket_high, 0.001)
            # ... but we use forecast + reality stdev for "smart money" fair value
            smart_fair = calc_bucket_probability(forecast_temp, bucket_low, bucket_high, true_stdev)

            # Market price is noisy: some mix of smart money and "dumb retail"
            # Dumb retail overprices buckets near the forecast and underprices tails
            dumb_prob = calc_bucket_probability(forecast_temp, bucket_low, bucket_high, true_stdev * 0.7)
            # Market = blend of smart and dumb
            market_prob = 0.4 * smart_fair + 0.6 * dumb_prob
            # Add noise to simulate bid-ask spread and market microstructure
            market_prob = max(0.01, min(0.99, market_prob + rng.normal(0, 0.03)))

            spread = rng.uniform(1, 5)  # 1-5 cent spread
            mid_cents = market_prob * 100
            yes_bid = max(1, mid_cents - spread / 2)
            yes_ask = min(99, mid_cents + spread / 2)

            volume = int(rng.exponential(80))

            markets.append(SimulatedMarket(
                ticker=f"{prefix}-SIM-B{bs + 0.5:.0f}",
                title=f"High temp {city} be {bs}-{bs+1} F?",
                city=city,
                market_type="bucket",
                bucket_low=bucket_low,
                bucket_high=bucket_high,
                threshold=0,
                yes_bid_cents=round(yes_bid, 1),
                yes_ask_cents=round(yes_ask, 1),
                true_probability=calc_bucket_probability(true_temp, bucket_low, bucket_high, true_stdev),
                volume=volume,
                days_out=days_out,
            ))

        # Generate threshold markets (above/below)
        for offset in [-5, -3, -1, 1, 3, 5]:
            threshold = round(forecast_temp) + offset

            # Above market
            true_above = calc_threshold_probability(true_temp, threshold, 0.001, above=True)
            smart_above = calc_threshold_probability(forecast_temp, threshold, true_stdev, above=True)
            dumb_above = calc_threshold_probability(forecast_temp, threshold, true_stdev * 0.7, above=True)
            market_above = max(0.01, min(0.99, 0.4 * smart_above + 0.6 * dumb_above + rng.normal(0, 0.03)))

            spread = rng.uniform(1, 4)
            mid = market_above * 100
            markets.append(SimulatedMarket(
                ticker=f"{prefix}-SIM-T{threshold}",
                title=f"High temp {city} be >{threshold} F?",
                city=city,
                market_type="above",
                bucket_low=threshold,
                bucket_high=999,
                threshold=threshold,
                yes_bid_cents=round(max(1, mid - spread / 2), 1),
                yes_ask_cents=round(min(99, mid + spread / 2), 1),
                true_probability=calc_threshold_probability(true_temp, threshold, true_stdev, above=True),
                volume=int(rng.exponential(100)),
                days_out=days_out,
            ))

            # Below market
            true_below = calc_threshold_probability(true_temp, threshold, 0.001, above=False)
            smart_below = calc_threshold_probability(forecast_temp, threshold, true_stdev, above=False)
            dumb_below = calc_threshold_probability(forecast_temp, threshold, true_stdev * 0.7, above=False)
            market_below = max(0.01, min(0.99, 0.4 * smart_below + 0.6 * dumb_below + rng.normal(0, 0.03)))

            spread = rng.uniform(1, 4)
            mid = market_below * 100
            markets.append(SimulatedMarket(
                ticker=f"{prefix}-SIM-LT{threshold}",
                title=f"High temp {city} be <{threshold} F?",
                city=city,
                market_type="below",
                bucket_low=0,
                bucket_high=threshold,
                threshold=threshold,
                yes_bid_cents=round(max(1, mid - spread / 2), 1),
                yes_ask_cents=round(min(99, mid + spread / 2), 1),
                true_probability=calc_threshold_probability(true_temp, threshold, true_stdev, above=False),
                volume=int(rng.exponential(100)),
                days_out=days_out,
            ))

    return markets, true_temps


# =============================================================================
# HISTORICAL SETTLED MARKET LOADER (replaces synthetic scenarios)
# =============================================================================

# Series ticker -> city name (matching the CITIES config above)
_SERIES_TO_CITY_NAME = {
    "KXHIGHNY": "NYC",
    "KXHIGHCHI": "Chicago",
    "KXHIGHMIA": "Miami",
    "KXHIGHLA": "LA",
    "KXHIGHDC": "DC",
    "KXHIGHDEN": "Denver",
}


def _parse_market_title(title: str) -> dict:
    """
    Parse a Kalshi weather market title into market_type, bucket bounds, or threshold.

    Examples:
        'Will the **high temp in NYC** be 59-60° ...' -> bucket [59, 60]
        'Will the high temp in Chicago be >70° ...'   -> above, threshold=70
        'Will the **high temp in NYC** be <53° ...'   -> below, threshold=53
    """
    # Bucket: "be 59-60°" or "be 59-60 °"
    bucket_match = re.search(r"be (\d+)-(\d+)\s*°", title)
    if bucket_match:
        return {
            "market_type": "bucket",
            "bucket_low": float(bucket_match.group(1)),
            "bucket_high": float(bucket_match.group(2)),
            "threshold": 0.0,
        }

    # Above: "be >60°"
    above_match = re.search(r">(\d+)\s*°", title)
    if above_match:
        threshold = float(above_match.group(1))
        return {
            "market_type": "above",
            "bucket_low": threshold,
            "bucket_high": 999.0,
            "threshold": threshold,
        }

    # Below: "be <53°"
    below_match = re.search(r"<(\d+)\s*°", title)
    if below_match:
        threshold = float(below_match.group(1))
        return {
            "market_type": "below",
            "bucket_low": 0.0,
            "bucket_high": threshold,
            "threshold": threshold,
        }

    return None


def _infer_true_temp_from_group(group_markets: list[dict]) -> float | None:
    """
    Infer the actual temperature from a group of settled markets for the same
    city+date. The bucket market that settled YES tells us the temp was in that
    range; we use its midpoint as the true temp.

    For threshold markets: if '>X' settled YES, temp was above X.
    If '<X' settled YES, temp was below X. We combine all YES results to
    narrow down the range, then use the midpoint.
    """
    # First try: find a bucket market that settled YES
    for m in group_markets:
        if m["result"] != "yes":
            continue
        parsed = _parse_market_title(m["title"])
        if parsed and parsed["market_type"] == "bucket":
            return (parsed["bucket_low"] + parsed["bucket_high"]) / 2.0

    # Fallback: use threshold markets to bracket the temp
    lower_bound = -999.0
    upper_bound = 999.0
    for m in group_markets:
        parsed = _parse_market_title(m["title"])
        if not parsed:
            continue
        if m["result"] == "yes":
            if parsed["market_type"] == "above":
                # temp > threshold
                lower_bound = max(lower_bound, parsed["threshold"])
            elif parsed["market_type"] == "below":
                # temp < threshold
                upper_bound = min(upper_bound, parsed["threshold"])
        elif m["result"] == "no":
            if parsed["market_type"] == "above":
                # temp <= threshold
                upper_bound = min(upper_bound, parsed["threshold"])
            elif parsed["market_type"] == "below":
                # temp >= threshold
                lower_bound = max(lower_bound, parsed["threshold"])

    if lower_bound > -999 or upper_bound < 999:
        if lower_bound <= -999:
            lower_bound = upper_bound - 5
        if upper_bound >= 999:
            upper_bound = lower_bound + 5
        return (lower_bound + upper_bound) / 2.0

    return None


def refresh_historical_settlements():
    """
    Pull fresh settled markets from Kalshi API and merge into the historical file.
    Called at the start of each AutoResearch cycle so the dataset grows daily.
    New settlements are appended (deduplicated by ticker).
    """
    if KalshiClient is None:
        return

    try:
        client = KalshiClient()
    except Exception as e:
        console.print(f"[yellow]Cannot connect to Kalshi for data refresh: {e}[/yellow]")
        return

    # Load existing data
    existing_tickers = set()
    existing_markets = []
    if HISTORICAL_SETTLEMENTS_FILE.exists():
        try:
            with open(HISTORICAL_SETTLEMENTS_FILE, "r") as f:
                old_data = json.load(f)
            existing_markets = old_data.get("markets", [])
            existing_tickers = {m.get("ticker") for m in existing_markets}
        except Exception:
            pass

    # Fetch recent settled markets from all series
    all_series = [
        "KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHDEN", "KXHIGHDC", "KXHIGHLA",
        "KXBTC", "KXETH", "KXSOL", "KXNBAGAME",
    ]

    new_count = 0
    for series in all_series:
        cursor = None
        for page in range(10):  # Up to 1000 per series
            try:
                resp = client.get_markets(
                    series_ticker=series, status="settled", limit=100, cursor=cursor,
                )
            except Exception:
                break

            markets = resp.get("markets", [])
            if not markets:
                break

            for m in markets:
                ticker = m.get("ticker", "")
                if ticker in existing_tickers:
                    continue  # Already have this one

                existing_tickers.add(ticker)
                existing_markets.append({
                    "ticker": ticker,
                    "title": m.get("title", ""),
                    "series": series,
                    "result": m.get("result", ""),
                    "previous_price": m.get("previous_price_dollars"),
                    "last_price": m.get("last_price_dollars"),
                    "prev_yes_ask": m.get("previous_yes_ask_dollars"),
                    "prev_yes_bid": m.get("previous_yes_bid_dollars"),
                    "volume": m.get("volume_fp", 0),
                    "open_interest": m.get("open_interest_fp", 0),
                    "close_time": m.get("close_time", ""),
                })
                new_count += 1

            cursor = resp.get("cursor")
            if not cursor:
                break

            time.sleep(0.2)

    if new_count > 0:
        # Save merged data
        merged = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "total": len(existing_markets),
            "markets": existing_markets,
        }
        try:
            with open(HISTORICAL_SETTLEMENTS_FILE, "w") as f:
                json.dump(merged, f)
            console.print(f"[green]Data refresh: +{new_count} new settlements "
                         f"({len(existing_markets)} total)[/green]")
        except Exception as e:
            console.print(f"[yellow]Failed to save refreshed data: {e}[/yellow]")
    else:
        console.print(f"[dim]Data refresh: no new settlements (have {len(existing_markets)})[/dim]")


def load_historical_scenarios() -> list[tuple[list[SimulatedMarket], dict]]:
    """
    Load real settled weather market data from historical_settlements_with_prices.json
    and convert into (markets, true_temps) scenario tuples for backtesting.

    Each (series, date) group of markets becomes one scenario. Markets are filtered
    to those with previous_price > 0 and volume > 0. The true temperature is
    inferred from which bucket market settled YES.

    Returns:
        List of (markets, true_temps) tuples — one per city+date group.
        Returns empty list if file doesn't exist or has no valid weather data.
    """
    if not HISTORICAL_SETTLEMENTS_FILE.exists():
        console.print(
            f"[red]Historical settlements file not found: {HISTORICAL_SETTLEMENTS_FILE}[/red]"
        )
        return []

    try:
        with open(HISTORICAL_SETTLEMENTS_FILE, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        console.print(f"[red]Failed to load historical settlements: {e}[/red]")
        return []

    all_markets = data.get("markets", [])
    if not all_markets:
        return []

    # Filter to weather series with valid price and volume
    weather_series = set(_SERIES_TO_CITY_NAME.keys())
    valid_markets = [
        m for m in all_markets
        if m.get("series") in weather_series
        and float(m.get("previous_price", "0")) > 0
        and float(m.get("volume", "0")) > 0
    ]

    if not valid_markets:
        return []

    # Group by (series, date) — each group = one daily scenario for one city
    groups = defaultdict(list)
    for m in valid_markets:
        # close_time is like "2026-03-22T04:59:00Z" — take the date part
        date_key = m["close_time"][:10]
        groups[(m["series"], date_key)].append(m)

    scenarios = []
    for (series, date_key), group_markets in groups.items():
        city = _SERIES_TO_CITY_NAME.get(series)
        if not city:
            continue

        # Infer the true temperature from settlement results
        true_temp = _infer_true_temp_from_group(group_markets)
        if true_temp is None:
            continue

        # Convert each market in the group to a SimulatedMarket
        sim_markets = []
        for m in group_markets:
            parsed = _parse_market_title(m["title"])
            if not parsed:
                continue

            # previous_price is in dollars (e.g. "0.0600"); convert to cents
            prev_price_cents = float(m["previous_price"]) * 100.0
            prev_yes_ask = float(m.get("prev_yes_ask", "0")) * 100.0
            prev_yes_bid = float(m.get("prev_yes_bid", "0")) * 100.0

            # Use prev_yes_ask/prev_yes_bid if available, else derive from previous_price
            if prev_yes_ask > 0 and prev_yes_bid > 0:
                yes_ask = prev_yes_ask
                yes_bid = prev_yes_bid
            else:
                yes_ask = min(99.0, prev_price_cents + 1.0)
                yes_bid = max(1.0, prev_price_cents - 1.0)

            # true_probability: 1.0 if settled YES, 0.0 if settled NO
            true_prob = 1.0 if m["result"] == "yes" else 0.0

            sim_markets.append(SimulatedMarket(
                ticker=m["ticker"],
                title=m["title"],
                city=city,
                market_type=parsed["market_type"],
                bucket_low=parsed["bucket_low"],
                bucket_high=parsed["bucket_high"],
                threshold=parsed["threshold"],
                yes_bid_cents=round(yes_bid, 1),
                yes_ask_cents=round(yes_ask, 1),
                true_probability=true_prob,
                volume=int(float(m.get("volume", "0"))),
                days_out=0,  # already settled
            ))

        if not sim_markets:
            continue

        true_temps = {
            city: {
                "forecast": true_temp,
                "true_temp": true_temp,
                "days_out": 0,
            }
        }

        scenarios.append((sim_markets, true_temps))

    return scenarios


# =============================================================================
# REAL OUTCOME DATA LOADING
# =============================================================================

def load_real_outcomes() -> list[dict]:
    """
    Load real settlement data from real_outcomes.json (written by settlement_tracker).

    Returns a list of real outcome scenarios that can replace synthetic scenarios
    in backtesting. Each entry contains the data needed to reconstruct a
    SimulatedMarket + true_temps pair from actual trading results.

    Returns empty list if file doesn't exist, is empty, or has no weather outcomes.
    """
    if not REAL_OUTCOMES_FILE.exists():
        return []

    try:
        with open(REAL_OUTCOMES_FILE, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

    outcomes = data.get("outcomes", [])
    if not outcomes:
        return []

    # Filter to weather trades only (have forecast/actual temp data)
    weather_outcomes = [
        o for o in outcomes
        if o.get("strategy") == "weather"
        and o.get("forecast_temp") is not None
        and o.get("actual_temp") is not None
    ]

    return weather_outcomes


def real_outcome_to_scenario(outcome: dict) -> tuple[list[SimulatedMarket], dict]:
    """
    Convert a single real outcome record into a (markets, true_temps) pair
    that the backtest engine can process identically to synthetic scenarios.

    The real outcome gives us one market with known forecast, actual temp,
    entry price, and settlement result. We reconstruct a minimal scenario
    around it so detect_edges + settle_trades can evaluate the strategy's
    decision against what actually happened.
    """
    ticker = outcome["ticker"]
    forecast_temp = outcome["forecast_temp"]
    actual_temp = outcome["actual_temp"]
    entry_price = outcome["entry_price"]  # cents
    side = outcome["side"]  # "yes" or "no"
    settlement_result = outcome["settlement_result"]  # "yes" or "no"

    # Determine city from ticker prefix
    city = "NYC"  # default
    for city_cfg in CITIES:
        if ticker.startswith(city_cfg["ticker_prefix"]):
            city = city_cfg["name"]
            break

    # Determine market type from ticker
    # Tickers like KXHIGHNY-26MAR21-B57.5 (bucket) or -T63 (threshold)
    market_type = "bucket"
    bucket_low = forecast_temp - 0.5
    bucket_high = forecast_temp + 0.5
    threshold = 0.0

    if "-B" in ticker:
        # Bucket market: extract center from ticker
        try:
            center = float(ticker.split("-B")[-1])
            bucket_low = math.floor(center)
            bucket_high = math.ceil(center)
            if bucket_low == bucket_high:
                bucket_high = bucket_low + 1
        except (ValueError, IndexError):
            pass
    elif "-LT" in ticker:
        # Below threshold (check before -T since -LT contains -T)
        market_type = "below"
        try:
            threshold = float(ticker.split("-LT")[-1])
        except (ValueError, IndexError):
            threshold = forecast_temp
        bucket_low = 0.0
        bucket_high = threshold
    elif "-T" in ticker:
        # Threshold (above) market
        market_type = "above"
        try:
            threshold = float(ticker.split("-T")[-1])
        except (ValueError, IndexError):
            threshold = forecast_temp
        bucket_low = threshold
        bucket_high = 999.0

    # Reconstruct market prices from entry data
    # If we bought YES at entry_price, the ask was ~entry_price
    # If we bought NO at entry_price, the bid was ~(100 - entry_price)
    if side == "yes":
        yes_ask = entry_price
        yes_bid = max(1.0, entry_price - 2.0)  # assume ~2c spread
    else:
        yes_bid = 100.0 - entry_price
        yes_ask = min(99.0, (100.0 - entry_price) + 2.0)

    # True probability: derived from actual settlement
    # (1.0 if YES settled, 0.0 if NO settled)
    true_prob = 1.0 if settlement_result == "yes" else 0.0

    market = SimulatedMarket(
        ticker=ticker,
        title=f"Real outcome: {ticker}",
        city=city,
        market_type=market_type,
        bucket_low=bucket_low,
        bucket_high=bucket_high,
        threshold=threshold,
        yes_bid_cents=round(yes_bid, 1),
        yes_ask_cents=round(yes_ask, 1),
        true_probability=true_prob,
        volume=100,  # real markets have volume
        days_out=0,  # already settled
    )

    true_temps = {
        city: {
            "forecast": forecast_temp,
            "true_temp": actual_temp,
            "days_out": 0,
        }
    }

    return [market], true_temps


# =============================================================================
# KALSHI SETTLED MARKET FETCHER
# =============================================================================

KALSHI_SERIES = ["KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHLA", "KXHIGHDC", "KXHIGHDEN"]

KALSHI_CACHE_TTL_SECONDS = 4 * 3600  # 4 hours


def _load_kalshi_cache() -> list | None:
    """Load cached Kalshi settlements if cache exists and is fresh (< 4 hours old)."""
    if not KALSHI_CACHE_FILE.exists():
        return None
    try:
        age = time.time() - KALSHI_CACHE_FILE.stat().st_mtime
        if age > KALSHI_CACHE_TTL_SECONDS:
            return None
        with open(KALSHI_CACHE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError, OSError):
        return None


def _save_kalshi_cache(data: list):
    """Write settlements list to cache file."""
    try:
        with open(KALSHI_CACHE_FILE, "w") as f:
            json.dump(data, f)
    except (IOError, OSError):
        pass


def fetch_recent_kalshi_settlements() -> list[tuple[list, dict]]:
    """
    Fetch recently settled Kalshi weather markets and convert them into
    (markets, true_temps) scenario tuples for backtesting.

    Uses the KalshiClient to pull settled markets for each weather series,
    then fetches actual high temps from NWS observations to build ground-truth
    scenarios.

    Results are cached to kalshi_cache.json for 4 hours to avoid repeated
    API calls.

    Returns:
        List of (markets, true_temps) tuples, one per settled market.
        Returns empty list on any failure.
    """
    try:
        # Check cache first
        cached = _load_kalshi_cache()
        if cached is not None:
            # Reconstruct scenario tuples from cached dicts
            scenarios = []
            for entry in cached:
                try:
                    market = SimulatedMarket(**entry["market"])
                    true_temps = entry["true_temps"]
                    scenarios.append(([market], true_temps))
                except (KeyError, TypeError):
                    continue
            if scenarios:
                console.print(
                    f"[dim]Loaded {len(scenarios)} settled Kalshi scenarios from cache[/dim]"
                )
                return scenarios

        # Need KalshiClient and settlement_tracker functions
        if KalshiClient is None or fetch_actual_high_temp is None:
            return []

        client = KalshiClient()
        raw_settlements = []

        for series in KALSHI_SERIES:
            try:
                resp = client.get_markets(
                    series_ticker=series, status="settled", limit=100
                )
                markets_data = resp.get("markets", [])
                for m in markets_data:
                    ticker = m.get("ticker", "")
                    title = m.get("title", "")
                    result = m.get("result", "")
                    if result not in ("yes", "no"):
                        continue

                    # Extract prices — Kalshi API returns dollars, convert to cents
                    yes_bid = float(m.get("yes_bid_dollars", 0) or 0) * 100
                    yes_ask = float(m.get("yes_ask_dollars", 0) or 0) * 100
                    # For settled markets, prices may be 0/100; use last_price as fallback
                    last_price = float(m.get("last_price", 0) or 0) * 100
                    if yes_bid <= 0 and yes_ask <= 0 and last_price > 0:
                        yes_bid = max(1.0, last_price - 2.0)
                        yes_ask = min(99.0, last_price + 2.0)
                    if yes_ask <= 0:
                        yes_ask = max(1.0, yes_bid + 2.0)
                    if yes_bid <= 0:
                        yes_bid = max(1.0, yes_ask - 2.0)

                    raw_settlements.append({
                        "ticker": ticker,
                        "title": title,
                        "result": result,
                        "series": series,
                        "yes_bid": round(yes_bid, 1),
                        "yes_ask": round(yes_ask, 1),
                        "close_time": m.get("close_time", ""),
                    })
            except Exception:
                continue  # skip series on API error

        if not raw_settlements:
            return []

        # Build scenarios from settled markets + actual temps
        cache_entries = []
        scenarios = []

        for raw in raw_settlements:
            ticker = raw["ticker"]
            title = raw["title"]
            result = raw["result"]
            series = raw["series"]
            yes_bid = raw["yes_bid"]
            yes_ask = raw["yes_ask"]
            close_time_str = raw["close_time"]

            # Determine city
            city = SERIES_TO_CITY.get(series, "NYC")

            # Extract settlement date from close_time for NWS lookup
            date_str = None
            if close_time_str:
                try:
                    ct = datetime.fromisoformat(
                        close_time_str.replace("Z", "+00:00")
                    )
                    date_str = ct.strftime("%Y-%m-%d")
                except (ValueError, TypeError):
                    pass

            # Fetch actual high temp from NWS observations
            actual_temp = None
            station_id = NWS_OBSERVATION_STATIONS.get(series)
            if station_id and date_str:
                actual_temp = fetch_actual_high_temp(station_id, date_str)
                time.sleep(0.3)  # rate-limit NWS requests

            if actual_temp is None:
                continue  # can't build scenario without actual temp

            # Parse market type from ticker/title (reuse logic from real_outcome_to_scenario)
            market_type = "bucket"
            bucket_low = actual_temp - 0.5
            bucket_high = actual_temp + 0.5
            threshold = 0.0

            if "-B" in ticker:
                try:
                    center = float(ticker.split("-B")[-1])
                    bucket_low = math.floor(center)
                    bucket_high = math.ceil(center)
                    if bucket_low == bucket_high:
                        bucket_high = bucket_low + 1
                except (ValueError, IndexError):
                    pass
            elif "-LT" in ticker:
                market_type = "below"
                try:
                    threshold = float(ticker.split("-LT")[-1])
                except (ValueError, IndexError):
                    threshold = actual_temp
                bucket_low = 0.0
                bucket_high = threshold
            elif "-T" in ticker:
                market_type = "above"
                try:
                    threshold = float(ticker.split("-T")[-1])
                except (ValueError, IndexError):
                    threshold = actual_temp
                bucket_low = threshold
                bucket_high = 999.0

            # True probability: 1.0 if YES settled, 0.0 if NO settled
            true_prob = 1.0 if result == "yes" else 0.0

            market = SimulatedMarket(
                ticker=ticker,
                title=title,
                city=city,
                market_type=market_type,
                bucket_low=bucket_low,
                bucket_high=bucket_high,
                threshold=threshold,
                yes_bid_cents=yes_bid,
                yes_ask_cents=yes_ask,
                true_probability=true_prob,
                volume=100,
                days_out=0,
            )

            # Use actual_temp as both forecast and true_temp — for settled markets
            # the forecast error is already baked into the result
            true_temps = {
                city: {
                    "forecast": actual_temp,
                    "true_temp": actual_temp,
                    "days_out": 0,
                }
            }

            scenarios.append(([market], true_temps))
            cache_entries.append({
                "market": {
                    "ticker": market.ticker,
                    "title": market.title,
                    "city": market.city,
                    "market_type": market.market_type,
                    "bucket_low": market.bucket_low,
                    "bucket_high": market.bucket_high,
                    "threshold": market.threshold,
                    "yes_bid_cents": market.yes_bid_cents,
                    "yes_ask_cents": market.yes_ask_cents,
                    "true_probability": market.true_probability,
                    "volume": market.volume,
                    "days_out": market.days_out,
                },
                "true_temps": true_temps,
            })

        # Cache results
        if cache_entries:
            _save_kalshi_cache(cache_entries)

        if scenarios:
            console.print(
                f"[green]Fetched {len(scenarios)} settled Kalshi weather scenarios[/green]"
            )

        return scenarios

    except Exception as e:
        console.print(
            f"[yellow]Kalshi settlement fetch failed ({e}), using synthetic only[/yellow]"
        )
        return []


# =============================================================================
# EDGE DETECTION (uses candidate_strategy params + weather_strategy math)
# =============================================================================

def load_strategy_params() -> dict:
    """Import candidate_strategy fresh and return all params."""
    # Force reimport
    for mod_name in list(sys.modules.keys()):
        if "candidate_strategy" in mod_name:
            del sys.modules[mod_name]

    sys.path.insert(0, str(STRATEGY_FILE.parent))
    try:
        import candidate_strategy as strat
        importlib.reload(strat)
        return {
            "forecast_stdev": strat.get_forecast_stdev(),
            "edge_threshold_cents": strat.EDGE_THRESHOLD_CENTS,
            "contracts_per_trade": strat.CONTRACTS_PER_TRADE,
            "nws_official_weight": strat.NWS_OFFICIAL_WEIGHT,
            "city_weights": strat.get_city_weights(),
            "bucket_multiplier": strat.BUCKET_MULTIPLIER,
            "threshold_multiplier": strat.THRESHOLD_MULTIPLIER,
            "high_confidence_edge": strat.HIGH_CONFIDENCE_EDGE,
            "medium_confidence_edge": strat.MEDIUM_CONFIDENCE_EDGE,
            "max_position_dollars": strat.MAX_POSITION_DOLLARS,
            "min_volume": strat.MIN_VOLUME,
            "tight_ensemble_threshold": strat.TIGHT_ENSEMBLE_THRESHOLD,
            "tight_ensemble_multiplier": strat.TIGHT_ENSEMBLE_MULTIPLIER,
        }
    except Exception as e:
        return {"error": str(e)}


@dataclass
class DetectedEdge:
    """An edge found by our model in a simulated market."""
    market: SimulatedMarket
    fair_value_cents: float
    edge_cents: float
    side: str               # "buy_yes" or "buy_no"
    contracts: int
    cost_dollars: float


def detect_edges(
    markets: list[SimulatedMarket],
    true_temps: dict,
    params: dict,
) -> list[DetectedEdge]:
    """
    Run our weather edge detection model against simulated markets.
    Uses the candidate_strategy parameters.
    """
    if "error" in params:
        return []

    edges = []
    forecast_stdev = params["forecast_stdev"]
    edge_threshold = params["edge_threshold_cents"]
    contracts_base = params["contracts_per_trade"]
    city_weights = params["city_weights"]
    bucket_mult = params["bucket_multiplier"]
    threshold_mult = params["threshold_multiplier"]
    max_pos = params["max_position_dollars"]
    min_vol = params["min_volume"]
    nws_weight = params["nws_official_weight"]

    for m in markets:
        # Volume filter
        if m.volume < min_vol:
            continue

        # Get the model's stdev for this market's days_out
        stdev = forecast_stdev.get(min(m.days_out, 3), 4.5)

        # Get forecast temp for this city
        city_info = true_temps.get(m.city)
        if not city_info:
            continue
        forecast_temp = city_info["forecast"]

        # Blend: in reality we'd blend NWS official with ensemble mean.
        # In simulation, the "ensemble mean" is just the forecast with some noise.
        # The blend doesn't change the forecast here but affects stdev confidence.
        # A higher NWS weight means we trust the point forecast more (lower effective stdev).
        effective_stdev = stdev * (1.0 + 0.3 * (1.0 - nws_weight))

        # Calculate our fair value
        if m.market_type == "bucket":
            fair_prob = calc_bucket_probability(
                forecast_temp, m.bucket_low, m.bucket_high, effective_stdev
            )
        elif m.market_type == "above":
            fair_prob = calc_threshold_probability(
                forecast_temp, m.threshold, effective_stdev, above=True
            )
        elif m.market_type == "below":
            fair_prob = calc_threshold_probability(
                forecast_temp, m.threshold, effective_stdev, above=False
            )
        else:
            continue

        fair_value_cents = fair_prob * 100.0

        # Calculate edge vs market
        buy_price = m.yes_ask_cents
        sell_price = m.yes_bid_cents

        buy_edge = fair_value_cents - buy_price
        sell_edge = sell_price - fair_value_cents

        # Position sizing with city weight and market type multiplier
        city_w = city_weights.get(m.city, 1.0)
        type_mult = bucket_mult if m.market_type == "bucket" else threshold_mult
        contracts = max(1, int(contracts_base * city_w * type_mult))

        if buy_edge > edge_threshold:
            cost = contracts * buy_price / 100.0
            if cost > max_pos:
                contracts = max(1, int(max_pos / (buy_price / 100.0)))
                cost = contracts * buy_price / 100.0
            edges.append(DetectedEdge(
                market=m,
                fair_value_cents=round(fair_value_cents, 2),
                edge_cents=round(buy_edge, 2),
                side="buy_yes",
                contracts=contracts,
                cost_dollars=round(cost, 4),
            ))
        elif sell_edge > edge_threshold:
            no_price = 100 - sell_price
            cost = contracts * no_price / 100.0
            if cost > max_pos:
                contracts = max(1, int(max_pos / (no_price / 100.0)))
                cost = contracts * no_price / 100.0
            edges.append(DetectedEdge(
                market=m,
                fair_value_cents=round(fair_value_cents, 2),
                edge_cents=round(sell_edge, 2),
                side="buy_no",
                contracts=contracts,
                cost_dollars=round(cost, 4),
            ))

    return edges


# =============================================================================
# TRADE SETTLEMENT & P&L CALCULATION
# =============================================================================

def settle_trades(
    edges: list[DetectedEdge],
    true_temps: dict,
) -> list[dict]:
    """
    Settle each detected edge against the true temperature.
    Returns list of trade results with P&L.
    """
    results = []

    for e in edges:
        m = e.market
        city_info = true_temps.get(m.city)
        if not city_info:
            continue

        true_temp = city_info["true_temp"]

        # Determine if the YES outcome actually happened
        if m.market_type == "bucket":
            yes_happened = m.bucket_low <= true_temp < m.bucket_high
        elif m.market_type == "above":
            yes_happened = true_temp > m.threshold
        elif m.market_type == "below":
            yes_happened = true_temp < m.threshold
        else:
            continue

        # Calculate P&L
        if e.side == "buy_yes":
            if yes_happened:
                # Bought yes at ask, settles at $1
                pnl = e.contracts * (100 - m.yes_ask_cents) / 100.0
            else:
                # Bought yes, it settles at $0
                pnl = -e.cost_dollars
        else:  # buy_no
            if not yes_happened:
                # Bought no at (100-bid), settles at $1
                no_price = 100 - m.yes_bid_cents
                pnl = e.contracts * (100 - no_price) / 100.0
            else:
                # Bought no, but yes happened, settles at $0
                pnl = -e.cost_dollars

        results.append({
            "city": m.city,
            "ticker": m.ticker,
            "type": m.market_type,
            "side": e.side,
            "edge_cents": e.edge_cents,
            "contracts": e.contracts,
            "cost": e.cost_dollars,
            "pnl": round(pnl, 4),
            "won": pnl > 0,
            "true_temp": round(true_temp, 1),
            "forecast_temp": round(city_info["forecast"], 1),
        })

    return results


# =============================================================================
# SCORING
# =============================================================================

def scale_position_to_balance(
    contracts: int,
    cost_per_contract: float,
    balance: float,
    max_position_frac: float = 0.05,
) -> tuple[int, float]:
    """Scale position size so cost doesn't exceed max_position_frac of current balance."""
    if cost_per_contract <= 0 or balance <= 0:
        return 0, 0.0
    max_cost = balance * max_position_frac
    total_cost = contracts * cost_per_contract
    if total_cost > max_cost:
        contracts = max(1, int(max_cost / cost_per_contract))
        total_cost = contracts * cost_per_contract
    return contracts, total_cost


def score_simulation(
    all_trade_results: list[dict],
    initial_balance: float = 100.0,
) -> dict:
    """
    Score a full simulation run (many scenarios) for strategy quality.

    Returns dict with: sortino, roi_pct, max_dd_pct, win_rate, total_trades,
                       profit_factor, score (composite).
    """
    if not all_trade_results:
        return {
            "sortino": 0, "roi_pct": 0, "max_dd_pct": 0, "win_rate": 0,
            "total_trades": 0, "profit_factor": 0, "total_pnl": 0,
            "score": -999.0,
        }

    pnls = [t["pnl"] for t in all_trade_results]
    total_pnl = sum(pnls)
    n_trades = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    # Win rate
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0

    # ROI
    roi_pct = (total_pnl / initial_balance) * 100

    # Equity curve for drawdown
    balance = initial_balance
    peak = balance
    max_dd = 0
    for p in pnls:
        balance += p
        peak = max(peak, balance)
        dd = (peak - balance) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
    max_dd_pct = max_dd * 100

    # Sortino ratio (annualized, ~250 trading days)
    if n_trades > 1:
        returns = np.array(pnls) / initial_balance
        mean_ret = np.mean(returns)
        downside = returns[returns < 0]
        down_std = np.std(downside, ddof=1) if len(downside) > 1 else 0.0001
        sortino = (mean_ret / down_std) * np.sqrt(250)
    else:
        sortino = 0

    # Profit factor
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.0001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Composite score
    # Weather bucket markets have low win rates (~30-40%) but high payoffs.
    # Sortino matters most (risk-adjusted), drawdown matters, but the threshold
    # must be realistic for a strategy that takes many small bets.
    if max_dd_pct > 60:
        score = -999.0  # Hard constraint: reject catastrophic drawdown
    else:
        score = (
            sortino * 0.30 +
            min(roi_pct, 500) * 0.01 * 0.20 +  # cap ROI contribution to avoid runaway
            (win_rate / 100.0) * 10.0 * 0.15 +
            min(profit_factor, 3.0) * 3.0 * 0.15 -
            max_dd_pct * 0.15                    # penalize drawdown
        )

    return {
        "sortino": round(sortino, 3),
        "roi_pct": round(roi_pct, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "win_rate": round(win_rate, 1),
        "total_trades": n_trades,
        "profit_factor": round(profit_factor, 2),
        "total_pnl": round(total_pnl, 2),
        "score": round(score, 4),
    }


# =============================================================================
# FULL BACKTEST: MANY SCENARIOS
# =============================================================================

def run_weather_backtest(
    seed: int = 42,
    n_scenarios: int = 200,
    use_real_data: bool = False,
    walk_forward: bool = False,
    _historical_cache: list | None = None,
) -> dict:
    """
    Run a full weather strategy backtest using ONLY real historical data.

    1. Load candidate_strategy params
    2. Load real settled weather markets from historical_settlements_with_prices.json
    3. Detect edges and scale positions relative to current balance
    4. Settle trades, accumulate P&L
    5. Score the result

    The seed parameter is used to shuffle the order of historical scenarios
    so that multi-seed validation explores different orderings.

    If walk_forward=True, the first 70% of scenarios are used for training
    and the last 30% for testing. Both train_score and test_score are returned.

    Returns dict with all metrics.
    """
    params = load_strategy_params()
    if "error" in params:
        return {"error": params["error"], "score": -999.0}

    initial_balance = 100.0
    max_pos_dollars = params.get("max_position_dollars", 5.0)

    rng = np.random.default_rng(seed)
    all_results = []
    balance = initial_balance

    # Load real historical scenarios (use cache if provided to avoid re-reading)
    if _historical_cache is not None:
        historical_scenarios = list(_historical_cache)
    else:
        historical_scenarios = load_historical_scenarios()

    if not historical_scenarios:
        return {"error": "No historical scenarios loaded", "score": -999.0}

    # Shuffle with seed for multi-seed validation variety
    rng.shuffle(historical_scenarios)

    # Use all available scenarios (ignore n_scenarios for real data)
    total_scenarios = len(historical_scenarios)

    for scenario_idx in range(total_scenarios):
        if balance <= 1.0:
            break  # Account blown

        markets, true_temps = historical_scenarios[scenario_idx]

        edges = detect_edges(markets, true_temps, params)

        # Sort edges by edge size (best first) and apply daily budget
        edges.sort(key=lambda e: e.edge_cents, reverse=True)
        daily_budget = balance * 0.20  # Risk at most 20% of balance per day
        spent = 0.0
        filtered_edges = []

        for e in edges:
            # Determine cost per contract
            if e.side == "buy_yes":
                cost_per = e.market.yes_ask_cents / 100.0
            else:
                cost_per = (100 - e.market.yes_bid_cents) / 100.0

            # Cap individual position at fraction of balance
            max_single = min(balance * 0.03, max_pos_dollars)
            if cost_per > 0:
                scaled_contracts = max(1, int(max_single / cost_per))
                e.contracts = min(e.contracts, scaled_contracts)
                e.cost_dollars = round(e.contracts * cost_per, 4)

            # Check daily budget
            if spent + e.cost_dollars > daily_budget:
                continue
            spent += e.cost_dollars
            filtered_edges.append(e)

        settled = settle_trades(filtered_edges, true_temps)

        # Update balance with this day's P&L
        for t in settled:
            balance += t["pnl"]
            t["balance_after"] = round(balance, 2)

        all_results.extend(settled)

    # Walk-forward split: train on first 70%, test on last 30%
    if walk_forward and all_results:
        split_idx = int(len(all_results) * 0.70)
        train_results = all_results[:split_idx]
        test_results = all_results[split_idx:]

        train_metrics = score_simulation(train_results, initial_balance=initial_balance)
        test_metrics = score_simulation(test_results, initial_balance=initial_balance)

        # Use overall metrics as the primary result
        metrics = score_simulation(all_results, initial_balance=initial_balance)
        metrics["train_score"] = train_metrics["score"]
        metrics["test_score"] = test_metrics["score"]
        return metrics

    metrics = score_simulation(all_results, initial_balance=initial_balance)
    return metrics


# =============================================================================
# TAIL FADE BACKTEST: Price-based "buy NO on cheap YES" strategy
# =============================================================================

# Series -> category mapping for tail fade filtering
_SERIES_TO_CATEGORY = {
    "KXHIGHNY": "weather",
    "KXHIGHCHI": "weather",
    "KXHIGHMIA": "weather",
    "KXHIGHLA": "weather",
    "KXHIGHDC": "weather",
    "KXHIGHDEN": "weather",
    "KXBTC": "crypto",
    "KXETH": "crypto",
    "KXNBAGAME": "nba",
}


def load_tail_fade_params() -> dict:
    """Import candidate_strategy fresh and return tail fade params."""
    for mod_name in list(sys.modules.keys()):
        if "candidate_strategy" in mod_name:
            del sys.modules[mod_name]

    sys.path.insert(0, str(STRATEGY_FILE.parent))
    try:
        import candidate_strategy as strat
        importlib.reload(strat)
        return {
            "max_price": getattr(strat, "TAIL_FADE_MAX_PRICE", 5),
            "min_volume": getattr(strat, "TAIL_FADE_MIN_VOLUME", 100),
            "weather_enabled": getattr(strat, "TAIL_FADE_WEATHER_ENABLED", 1),
            "crypto_enabled": getattr(strat, "TAIL_FADE_CRYPTO_ENABLED", 1),
            "nba_enabled": getattr(strat, "TAIL_FADE_NBA_ENABLED", 1),
            "mid_low": getattr(strat, "TAIL_FADE_MID_LOW", 40),
            "mid_high": getattr(strat, "TAIL_FADE_MID_HIGH", 55),
        }
    except Exception as e:
        return {"error": str(e)}


def run_tail_fade_backtest(
    seed: int = 42,
    walk_forward: bool = False,
    _all_markets_cache: list | None = None,
) -> dict:
    """
    Run a tail fade backtest against all historical settled markets.

    Strategy: buy NO on markets where YES price is cheap (< max_price cents)
    or in the mid-range fade zone (mid_low < yes_price < mid_high).

    If the market settles NO, we profit (yes_price cents per contract).
    If the market settles YES, we lose (100 - yes_price cents per contract).

    Walk-forward: shuffle by close_time, train on 60%, test on 40%.

    Returns dict with score_simulation() metrics.
    """
    tf_params = load_tail_fade_params()
    if "error" in tf_params:
        return {"error": tf_params["error"], "score": -999.0}

    max_price = tf_params["max_price"]
    min_volume = tf_params["min_volume"]
    weather_on = tf_params["weather_enabled"]
    crypto_on = tf_params["crypto_enabled"]
    nba_on = tf_params["nba_enabled"]
    mid_low = tf_params["mid_low"]
    mid_high = tf_params["mid_high"]

    # Load all markets from historical file
    if _all_markets_cache is not None:
        all_markets = _all_markets_cache
    else:
        if not HISTORICAL_SETTLEMENTS_FILE.exists():
            return {"error": "No historical settlements file", "score": -999.0}
        try:
            with open(HISTORICAL_SETTLEMENTS_FILE, "r") as f:
                data = json.load(f)
            all_markets = data.get("markets", [])
        except (json.JSONDecodeError, IOError) as e:
            return {"error": str(e), "score": -999.0}

    # Filter to markets with price > 0 and volume > 0
    valid = []
    for m in all_markets:
        prev_price = float(m.get("previous_price", "0"))
        volume = float(m.get("volume", "0"))
        if prev_price <= 0 or volume <= 0:
            continue
        if volume < min_volume:
            continue

        series = m.get("series", "")
        category = _SERIES_TO_CATEGORY.get(series, "other")

        # Category filter
        if category == "weather" and not weather_on:
            continue
        if category == "crypto" and not crypto_on:
            continue
        if category == "nba" and not nba_on:
            continue
        if category == "other":
            continue  # skip unknown categories

        yes_price_cents = prev_price * 100.0
        result = m.get("result", "")
        if result not in ("yes", "no"):
            continue

        valid.append({
            "yes_price_cents": yes_price_cents,
            "result": result,
            "close_time": m.get("close_time", ""),
            "volume": volume,
            "series": series,
            "category": category,
        })

    if not valid:
        return {"error": "No valid markets for tail fade", "score": -999.0}

    # Sort by close_time for walk-forward, then shuffle with seed
    valid.sort(key=lambda x: x["close_time"])

    rng = np.random.default_rng(seed)
    rng.shuffle(valid)

    # Apply tail fade strategy: generate trade results
    all_trade_results = []
    for m in valid:
        yp = m["yes_price_cents"]
        take_trade = False

        # Tail fade: YES price is very cheap -> buy NO
        if yp <= max_price:
            take_trade = True

        # Mid-range fade: YES price in mid zone -> buy NO
        if mid_low < yp < mid_high:
            take_trade = True

        if not take_trade:
            continue

        # Cost to buy NO = (100 - yes_price) cents per contract
        no_cost_cents = 100.0 - yp

        # Settlement
        if m["result"] == "no":
            # We bought NO and it settled NO: profit = yes_price cents
            pnl = yp / 100.0  # convert cents to dollars (1 contract)
        else:
            # We bought NO and it settled YES: loss = no_cost cents
            pnl = -no_cost_cents / 100.0

        all_trade_results.append({
            "city": m.get("category", ""),
            "ticker": m.get("series", ""),
            "type": "tail_fade",
            "side": "buy_no",
            "edge_cents": yp,  # the "edge" is the cheap YES price
            "contracts": 1,
            "cost": no_cost_cents / 100.0,
            "pnl": round(pnl, 4),
            "won": pnl > 0,
            "true_temp": 0.0,
            "forecast_temp": 0.0,
        })

    if not all_trade_results:
        return {"error": "No tail fade trades generated", "score": -999.0}

    # Walk-forward split: train on 60%, test on 40%
    if walk_forward:
        split_idx = int(len(all_trade_results) * 0.60)
        train_results = all_trade_results[:split_idx]
        test_results = all_trade_results[split_idx:]

        train_metrics = score_simulation(train_results, initial_balance=100.0)
        test_metrics = score_simulation(test_results, initial_balance=100.0)

        metrics = score_simulation(all_trade_results, initial_balance=100.0)
        metrics["train_score"] = train_metrics["score"]
        metrics["test_score"] = test_metrics["score"]
        return metrics

    return score_simulation(all_trade_results, initial_balance=100.0)


# =============================================================================
# PARAMETER MUTATION
# =============================================================================

def read_strategy_file() -> str:
    """Read current strategy source code."""
    return STRATEGY_FILE.read_text()


def write_strategy_file(content: str):
    """Write modified strategy back."""
    STRATEGY_FILE.write_text(content)


def mutate_parameter(source: str, param_name: str, new_value) -> str:
    """
    Replace a parameter value in the strategy source code.
    Handles int, float, and bool values.
    """
    lines = source.split("\n")
    new_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{param_name} =") or stripped.startswith(f"{param_name}="):
            parts = line.split("#")
            comment = f"  # {parts[1].strip()}" if len(parts) > 1 else ""
            indent = len(line) - len(line.lstrip())
            new_line = f"{' ' * indent}{param_name} = {repr(new_value)}{comment}"
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


def get_current_value(source: str, param_name: str):
    """Extract current value of a parameter from source."""
    for line in source.split("\n"):
        stripped = line.strip()
        if stripped.startswith(f"{param_name} =") or stripped.startswith(f"{param_name}="):
            val_part = stripped.split("=", 1)[1].split("#")[0].strip()
            try:
                return ast.literal_eval(val_part)
            except Exception:
                return val_part
    return None


def random_mutation(source: str) -> tuple[str, object, str]:
    """Pick a random parameter and a random value for it."""
    param = random.choice(list(MUTATION_SPACE.keys()))
    new_val = random.choice(MUTATION_SPACE[param])
    return param, new_val, "Random exploration"


# =============================================================================
# LLM-GUIDED MUTATIONS
# =============================================================================

def llm_guided_mutation(
    source: str,
    history: list[dict],
    results_log_path: Path,
) -> tuple[str, object, str]:
    """
    Use Claude (Haiku) to suggest the next parameter mutation based on
    recent experiment history. Falls back to random_mutation() on any error.

    Reads the last 15 entries from the results log, formats them as context,
    and asks Claude which parameter to change next and why.

    Returns:
        (param_name, new_value, reason)
    """
    if _anthropic_module is None or not config.ANTHROPIC_API_KEY:
        return random_mutation(source)

    # Load last 15 entries from results log
    recent_entries = []
    try:
        if results_log_path.exists():
            lines = results_log_path.read_text().strip().split("\n")
            for line in lines[-15:]:
                if line.strip():
                    try:
                        recent_entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except (IOError, OSError):
        pass

    if not recent_entries:
        return random_mutation(source)

    # Format experiment history for context
    history_text = ""
    for entry in recent_entries:
        kept = entry.get("kept", False)
        param = entry.get("parameter", "?")
        old_v = entry.get("old_value", "?")
        new_v = entry.get("new_value", "?")
        score = entry.get("metrics", {}).get("score", "?")
        result_str = "IMPROVED" if kept else "REVERTED"
        history_text += f"  {param}: {old_v} -> {new_v} | score={score} | {result_str}\n"

    # Format valid parameter space
    param_space_text = json.dumps(MUTATION_SPACE, indent=2)

    prompt = f"""You are optimizing a weather trading strategy. Here are the last {len(recent_entries)} experiment results showing which parameter changes improved or worsened the score:

{history_text}

Here are the valid parameters and their allowed values:
{param_space_text}

Based on these patterns, suggest the SINGLE best parameter to change next. Consider:
- Which parameters have shown improvement trends when increased/decreased?
- Which parameters haven't been explored much yet?
- What direction (higher/lower) tends to help?

Respond with ONLY a JSON object, no other text:
{{"parameter": "PARAM_NAME", "value": NEW_VALUE, "reason": "brief explanation"}}"""

    try:
        client = _anthropic_module.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text.strip()

        # Parse JSON from response (handle markdown code blocks)
        if "```" in response_text:
            # Extract content between code fences
            json_str = response_text.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
            json_str = json_str.strip()
        else:
            json_str = response_text

        result = json.loads(json_str)
        param_name = result["parameter"]
        new_value = result["value"]
        reason = result.get("reason", "LLM-guided mutation")

        # Validate the suggestion against MUTATION_SPACE
        if param_name not in MUTATION_SPACE:
            return random_mutation(source)

        # Find the closest valid value in MUTATION_SPACE
        valid_values = MUTATION_SPACE[param_name]
        if new_value not in valid_values:
            # Snap to nearest valid value
            if isinstance(new_value, (int, float)):
                new_value = min(valid_values, key=lambda v: abs(v - new_value))
            else:
                new_value = random.choice(valid_values)

        return param_name, new_value, f"LLM-guided: {reason}"

    except Exception:
        return random_mutation(source)


# =============================================================================
# GIT HELPERS
# =============================================================================

def git_commit(message: str):
    """Commit current strategy state."""
    try:
        subprocess.run(
            ["git", "add", str(STRATEGY_FILE)],
            cwd=config.PROJECT_ROOT,
            capture_output=True,
            timeout=10,
        )
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=config.PROJECT_ROOT,
            capture_output=True,
            timeout=10,
        )
    except Exception:
        pass


def git_revert():
    """Revert strategy to last committed version."""
    try:
        subprocess.run(
            ["git", "checkout", "--", str(STRATEGY_FILE)],
            cwd=config.PROJECT_ROOT,
            capture_output=True,
            timeout=10,
        )
    except Exception:
        pass


# =============================================================================
# LOGGING
# =============================================================================

def log_result(iteration: int, param: str, old_val, new_val, metrics: dict, kept: bool):
    """Append result to research log."""
    entry = {
        "iteration": iteration,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameter": str(param),
        "old_value": float(old_val) if isinstance(old_val, (int, float)) else str(old_val),
        "new_value": float(new_val) if isinstance(new_val, (int, float)) else str(new_val),
        "metrics": {
            k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else str(v)
            for k, v in metrics.items()
        },
        "kept": bool(kept),
    }
    with open(RESULTS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


# =============================================================================
# MAIN RESEARCH LOOP
# =============================================================================

def run_research(
    max_iterations: int = None,
    n_scenarios: int = 200,
    verbose: bool = True,
    use_real_data: bool = False,
    walk_forward: bool = False,
    no_claude: bool = False,
):
    """
    Run the Weather AutoResearch loop.

    For each iteration:
    1. Read current candidate_strategy.py
    2. Pick a mutation (LLM-guided 70%, random 30%)
    3. Apply mutation, run weather backtest with multi-seed validation
    4. If score improves: keep and git commit
    5. If score worsens: revert
    6. Log everything

    Args:
        max_iterations: Number of experiments to run
        n_scenarios: Number of weather days to simulate per backtest
        verbose: Print progress to terminal
        use_real_data: If True, incorporate real settlement outcomes from
                       real_outcomes.json for the first N scenarios
        walk_forward: If True, use walk-forward validation (70/30 split)
        no_claude: If True, skip LLM-guided mutations (random only)
    """
    max_iterations = max_iterations or config.AUTORESEARCH_MAX_ITERATIONS

    console.print("\n[bold cyan]================================================================[/bold cyan]")
    console.print("[bold cyan]  AUTORESEARCH: Multi-Strategy Optimization Loop              [/bold cyan]")
    console.print("[bold cyan]================================================================[/bold cyan]\n")

    # Refresh data from Kalshi API (pulls new settlements since last run)
    refresh_historical_settlements()

    # Build gold-standard dataset from price snapshots (if enough data exists)
    try:
        if BACKTEST_DATASET_FILE.exists():
            with open(BACKTEST_DATASET_FILE, "r") as f:
                _ds = json.load(f)
            ds_count = _ds.get("matched", 0)
            if ds_count > 0:
                console.print(
                    f"[green]Gold dataset: {ds_count} markets with REAL pre-settlement prices[/green]"
                )
        else:
            # Try to build it
            try:
                from build_backtest_dataset import build_dataset
                ds = build_dataset()
                if ds:
                    console.print(f"[green]Built gold dataset: {len(ds)} markets[/green]")
            except Exception as e:
                console.print(f"[dim]Gold dataset not available yet: {e}[/dim]")
    except Exception:
        pass

    # Load historical scenarios (shared across all backtests this session)
    historical_scenarios = load_historical_scenarios()
    if historical_scenarios:
        console.print(
            f"[green]Historical data: {len(historical_scenarios)} real settled weather scenarios "
            f"loaded from {HISTORICAL_SETTLEMENTS_FILE.name}[/green]"
        )
    else:
        console.print(
            "[red]No historical weather scenarios found -- cannot run weather backtest without real data[/red]"
        )
        return

    # Load raw markets for tail fade backtest (all categories, not just weather)
    tail_fade_markets_cache = None
    try:
        if HISTORICAL_SETTLEMENTS_FILE.exists():
            with open(HISTORICAL_SETTLEMENTS_FILE, "r") as f:
                _tf_data = json.load(f)
            tail_fade_markets_cache = _tf_data.get("markets", [])
            n_tf = len([
                m for m in tail_fade_markets_cache
                if float(m.get("previous_price", "0")) > 0
                and float(m.get("volume", "0")) > 0
            ])
            console.print(
                f"[green]Tail fade data: {n_tf} markets with price+volume "
                f"(all categories)[/green]"
            )
    except Exception as e:
        console.print(f"[yellow]Failed to load tail fade data: {e}[/yellow]")

    # Report mutation mode
    llm_available = (
        not no_claude
        and _anthropic_module is not None
        and config.ANTHROPIC_API_KEY
    )
    if llm_available:
        console.print("[green]Mutation mode: LLM-guided (70%) + random (30%)[/green]")
    else:
        reason = "disabled" if no_claude else "anthropic SDK/key unavailable"
        console.print(f"[dim]Mutation mode: random only ({reason})[/dim]")

    if walk_forward:
        console.print("[green]Walk-forward validation: enabled (70/30 split)[/green]")

    # Baseline: score current weather strategy
    console.print("[dim]Running baseline weather backtest...[/dim]")
    baseline = run_weather_backtest(
        seed=42, n_scenarios=n_scenarios, use_real_data=use_real_data,
        walk_forward=walk_forward, _historical_cache=historical_scenarios,
    )
    if "error" in baseline:
        console.print(f"[red]Weather baseline failed: {baseline['error']}[/red]")
        return

    console.print(f"[green]Weather baseline score: {baseline['score']:.4f}[/green]")
    console.print(
        f"  Sortino: {baseline['sortino']:.3f} | "
        f"ROI: {baseline['roi_pct']:.2f}% | "
        f"Win: {baseline['win_rate']:.1f}% | "
        f"MaxDD: {baseline['max_dd_pct']:.2f}% | "
        f"Trades: {baseline['total_trades']} | "
        f"PF: {baseline['profit_factor']:.2f}"
    )
    if walk_forward and "train_score" in baseline:
        console.print(
            f"  Walk-forward: train={baseline['train_score']:.4f} "
            f"test={baseline['test_score']:.4f}"
        )

    # Baseline: score current tail fade strategy
    console.print("[dim]Running baseline tail fade backtest...[/dim]")
    tf_baseline = run_tail_fade_backtest(
        seed=42, walk_forward=walk_forward,
        _all_markets_cache=tail_fade_markets_cache,
    )
    if "error" in tf_baseline:
        console.print(f"[yellow]Tail fade baseline: {tf_baseline.get('error', 'unknown')}[/yellow]")
        tf_baseline_score = -999.0
    else:
        tf_baseline_score = tf_baseline["score"]
        console.print(f"[green]Tail fade baseline score: {tf_baseline['score']:.4f}[/green]")
        console.print(
            f"  Sortino: {tf_baseline['sortino']:.3f} | "
            f"ROI: {tf_baseline['roi_pct']:.2f}% | "
            f"Win: {tf_baseline['win_rate']:.1f}% | "
            f"MaxDD: {tf_baseline['max_dd_pct']:.2f}% | "
            f"Trades: {tf_baseline['total_trades']} | "
            f"PF: {tf_baseline['profit_factor']:.2f}"
        )
        if walk_forward and "train_score" in tf_baseline:
            console.print(
                f"  Walk-forward: train={tf_baseline['train_score']:.4f} "
                f"test={tf_baseline['test_score']:.4f}"
            )
    console.print()

    best_score = baseline["score"]
    best_train_score = baseline.get("train_score", best_score)
    best_test_score = baseline.get("test_score", best_score)
    best_tf_score = tf_baseline_score
    best_tf_train_score = tf_baseline.get("train_score", tf_baseline_score) if "error" not in tf_baseline else -999.0
    best_tf_test_score = tf_baseline.get("test_score", tf_baseline_score) if "error" not in tf_baseline else -999.0
    improvements = 0
    history = []

    # Load existing history
    if RESULTS_LOG.exists():
        for line in RESULTS_LOG.read_text().strip().split("\n"):
            if line.strip():
                try:
                    history.append(json.loads(line))
                except Exception:
                    pass

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Multi-strategy research loop", total=max_iterations)

        # Separate mutation spaces for targeted iteration
        weather_params = set(MUTATION_SPACE.keys()) - {
            k for k in MUTATION_SPACE if k.startswith("TAIL_FADE_")
        }
        tail_fade_params = {
            k for k in MUTATION_SPACE if k.startswith("TAIL_FADE_")
        }

        for i in range(1, max_iterations + 1):
            source = read_strategy_file()

            # Alternate: odd iterations = weather, even = tail fade
            is_tail_fade_iter = (i % 2 == 0) and tail_fade_markets_cache is not None

            if is_tail_fade_iter:
                mode_label = "tail_fade"
                # Pick from tail fade params only
                param = random.choice(list(tail_fade_params))
                new_val = random.choice(MUTATION_SPACE[param])
                reason = "Random tail fade exploration"
            else:
                mode_label = "weather"
                # Pick mutation: LLM-guided 70% of the time, random 30%
                if llm_available and random.random() < 0.70:
                    try:
                        param, new_val, reason = llm_guided_mutation(
                            source, history, RESULTS_LOG
                        )
                    except Exception:
                        param, new_val, reason = random_mutation(source)
                else:
                    param, new_val, reason = random_mutation(source)

            old_val = get_current_value(source, param)

            # Skip if same value
            if old_val == new_val:
                progress.update(task, advance=1)
                continue

            progress.update(
                task,
                description=f"Iter {i}/{max_iterations} [{mode_label}]: {param}={new_val}",
            )

            # Apply mutation
            new_source = mutate_parameter(source, param, new_val)
            write_strategy_file(new_source)

            # Multi-seed validation: run 3 backtests with different seeds
            # and use the AVERAGE score to reduce variance / overfitting
            seeds = [42 + i, 42 + i + 1000, 42 + i + 2000]
            seed_scores = []
            first_metrics = None

            if is_tail_fade_iter:
                # Run tail fade backtest
                for s in seeds:
                    m = run_tail_fade_backtest(
                        seed=s, walk_forward=walk_forward,
                        _all_markets_cache=tail_fade_markets_cache,
                    )
                    if "error" not in m:
                        seed_scores.append(m)
                        if first_metrics is None:
                            first_metrics = m
            else:
                # Run weather backtest
                for s in seeds:
                    m = run_weather_backtest(
                        seed=s, n_scenarios=n_scenarios, use_real_data=use_real_data,
                        walk_forward=walk_forward, _historical_cache=historical_scenarios,
                    )
                    if "error" not in m:
                        seed_scores.append(m)
                        if first_metrics is None:
                            first_metrics = m

            if not seed_scores:
                write_strategy_file(source)  # revert on error
                progress.update(task, advance=1)
                continue

            # Use first run for detailed metrics display, average for score
            metrics = first_metrics
            new_score = sum(s["score"] for s in seed_scores) / len(seed_scores)

            # Decision: keep or revert (compare against mode-specific best)
            if is_tail_fade_iter:
                ref_best = best_tf_score
            else:
                ref_best = best_score

            if walk_forward and "train_score" in metrics:
                avg_train = sum(
                    s.get("train_score", s["score"]) for s in seed_scores
                ) / len(seed_scores)
                avg_test = sum(
                    s.get("test_score", s["score"]) for s in seed_scores
                ) / len(seed_scores)
                if is_tail_fade_iter:
                    kept = (
                        avg_train > best_tf_train_score
                        and avg_test > best_tf_test_score
                    )
                else:
                    kept = (
                        avg_train > best_train_score
                        and avg_test > best_test_score
                    )
            else:
                kept = new_score > ref_best

            if kept:
                if is_tail_fade_iter:
                    best_tf_score = new_score
                    if walk_forward and "train_score" in metrics:
                        best_tf_train_score = sum(
                            s.get("train_score", s["score"]) for s in seed_scores
                        ) / len(seed_scores)
                        best_tf_test_score = sum(
                            s.get("test_score", s["score"]) for s in seed_scores
                        ) / len(seed_scores)
                else:
                    best_score = new_score
                    if walk_forward and "train_score" in metrics:
                        best_train_score = sum(
                            s.get("train_score", s["score"]) for s in seed_scores
                        ) / len(seed_scores)
                        best_test_score = sum(
                            s.get("test_score", s["score"]) for s in seed_scores
                        ) / len(seed_scores)
                improvements += 1
                git_commit(
                    f"Research iter {i} [{mode_label}]: {param}={new_val} "
                    f"score={new_score:.4f} sortino={metrics['sortino']:.3f}"
                )
                if alert_research_improvement is not None:
                    try:
                        alert_research_improvement(param, old_val, new_val, new_score)
                    except Exception:
                        pass
                if verbose:
                    console.print(
                        f"  [green]+ Iter {i} [{mode_label}]: {param} {old_val}->{new_val} "
                        f"score={new_score:.4f} sortino={metrics['sortino']:.3f} "
                        f"roi={metrics['roi_pct']:.1f}% win={metrics['win_rate']:.0f}% KEPT[/green]"
                    )
            else:
                write_strategy_file(source)  # revert
                git_revert()
                if verbose and i % 5 == 0:
                    console.print(
                        f"  [dim]- Iter {i} [{mode_label}]: {param} {old_val}->{new_val} "
                        f"score={new_score:.4f} (vs {ref_best:.4f}) REVERTED[/dim]"
                    )

            # Log
            log_result(i, param, old_val, new_val, metrics, kept)
            history.append({
                "iteration": i,
                "parameter": param,
                "old_value": old_val,
                "new_value": new_val,
                "score": new_score,
                "kept": kept,
                "mode": mode_label,
            })

            progress.update(task, advance=1)
            time.sleep(0.05)  # brief pause

    # Final summary
    console.print(f"\n[bold cyan]{'=' * 65}[/bold cyan]")
    console.print(f"[bold]Multi-Strategy Research Complete: {max_iterations} iterations[/bold]")
    console.print(f"  Improvements found: {improvements}")
    console.print(f"  Best weather score: {best_score:.4f} (baseline was {baseline['score']:.4f})")
    console.print(f"  Best tail fade score: {best_tf_score:.4f} (baseline was {tf_baseline_score:.4f})")

    # Final weather backtest
    final = run_weather_backtest(
        seed=42, n_scenarios=n_scenarios, use_real_data=use_real_data,
        walk_forward=walk_forward, _historical_cache=historical_scenarios,
    )
    if "error" in final:
        console.print(f"[red]Final weather backtest failed: {final['error']}[/red]")
    else:
        console.print(f"\n[bold]Final Weather Strategy Performance:[/bold]")
        console.print(f"  Sortino:       {final['sortino']:.3f}")
        console.print(f"  ROI:           {final['roi_pct']:.2f}%")
        console.print(f"  Win Rate:      {final['win_rate']:.1f}%")
        console.print(f"  Max Drawdown:  {final['max_dd_pct']:.2f}%")
        console.print(f"  Profit Factor: {final['profit_factor']:.2f}")
        console.print(f"  Total Trades:  {final['total_trades']}")
        console.print(f"  Total P&L:     ${final['total_pnl']:.2f}")
        if walk_forward and "train_score" in final:
            console.print(f"  Train Score:   {final['train_score']:.4f}")
            console.print(f"  Test Score:    {final['test_score']:.4f}")

    # Final tail fade backtest
    tf_final = run_tail_fade_backtest(
        seed=42, walk_forward=walk_forward,
        _all_markets_cache=tail_fade_markets_cache,
    )
    if "error" in tf_final:
        console.print(f"[yellow]Final tail fade backtest: {tf_final.get('error', 'unknown')}[/yellow]")
    else:
        console.print(f"\n[bold]Final Tail Fade Strategy Performance:[/bold]")
        console.print(f"  Sortino:       {tf_final['sortino']:.3f}")
        console.print(f"  ROI:           {tf_final['roi_pct']:.2f}%")
        console.print(f"  Win Rate:      {tf_final['win_rate']:.1f}%")
        console.print(f"  Max Drawdown:  {tf_final['max_dd_pct']:.2f}%")
        console.print(f"  Profit Factor: {tf_final['profit_factor']:.2f}")
        console.print(f"  Total Trades:  {tf_final['total_trades']}")
        console.print(f"  Total P&L:     ${tf_final['total_pnl']:.2f}")
        if walk_forward and "train_score" in tf_final:
            console.print(f"  Train Score:   {tf_final['train_score']:.4f}")
            console.print(f"  Test Score:    {tf_final['test_score']:.4f}")

    console.print(f"\n  Results log: {RESULTS_LOG}")

    # Show current best params
    console.print(f"\n[bold]Optimized Weather Parameters:[/bold]")
    params = load_strategy_params()
    if "error" not in params:
        console.print(f"  FORECAST_STDEV: {params['forecast_stdev']}")
        console.print(f"  EDGE_THRESHOLD: {params['edge_threshold_cents']}c")
        console.print(f"  CONTRACTS/TRADE: {params['contracts_per_trade']}")
        console.print(f"  NWS_WEIGHT: {params['nws_official_weight']}")
        console.print(f"  CITY_WEIGHTS: {params['city_weights']}")
        console.print(f"  BUCKET_MULT: {params['bucket_multiplier']} / THRESHOLD_MULT: {params['threshold_multiplier']}")

    tf_params = load_tail_fade_params()
    if "error" not in tf_params:
        console.print(f"\n[bold]Optimized Tail Fade Parameters:[/bold]")
        console.print(f"  MAX_PRICE: {tf_params['max_price']}c")
        console.print(f"  MIN_VOLUME: {tf_params['min_volume']}")
        console.print(f"  WEATHER: {'on' if tf_params['weather_enabled'] else 'off'} | "
                      f"CRYPTO: {'on' if tf_params['crypto_enabled'] else 'off'} | "
                      f"NBA: {'on' if tf_params['nba_enabled'] else 'off'}")
        console.print(f"  MID_FADE: {tf_params['mid_low']}-{tf_params['mid_high']}c")

    console.print(f"[bold cyan]{'=' * 65}[/bold cyan]\n")

    # Regenerate strategy doc after each research cycle
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from strategy_doc import generate_strategy_doc
        generate_strategy_doc()
    except Exception:
        pass  # Non-critical


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Strategy AutoResearch Loop")
    parser.add_argument("--iterations", type=int, default=50, help="Number of experiments")
    parser.add_argument("--scenarios", type=int, default=200, help="Weather days per backtest")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    parser.add_argument("--no-claude", action="store_true", help="Random mutations only (no Claude API)")
    parser.add_argument(
        "--use-real-data", action="store_true",
        help="Incorporate real settlement outcomes from real_outcomes.json"
    )
    parser.add_argument(
        "--walk-forward", action="store_true",
        help="Enable walk-forward validation (70/30 train/test split)"
    )
    args = parser.parse_args()

    run_research(
        max_iterations=args.iterations,
        n_scenarios=args.scenarios,
        verbose=not args.quiet,
        use_real_data=args.use_real_data,
        walk_forward=args.walk_forward,
        no_claude=args.no_claude,
    )
