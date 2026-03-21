"""
autoresearch/research_loop.py -- Weather Forecast Arbitrage optimization loop.

This is the overnight self-improvement engine specifically for the weather
forecast vs. Kalshi market price strategy. It:

1. Reads the current candidate_strategy.py (weather-specific parameters)
2. Simulates realistic weather scenarios:
   - True temp = NWS forecast + normal noise (configurable stdev)
   - Fake Kalshi bucket/threshold markets around the true temp
   - Market prices = noisy implied probabilities from a "dumb" model
3. Runs the weather edge detection (from weather_strategy.py) against those markets
4. Scores based on: Sortino ratio, ROI, max drawdown, win rate
5. Mutates one parameter at a time, keeps winners, reverts losers
6. Logs everything to autoresearch/results.log
"""

import os
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

console = Console()

STRATEGY_FILE = Path(__file__).parent / "candidate_strategy.py"
RESULTS_LOG = Path(__file__).parent / "results.log"
REAL_OUTCOMES_FILE = Path(__file__).parent / "real_outcomes.json"
KALSHI_CACHE_FILE = Path(__file__).parent / "kalshi_cache.json"

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
) -> dict:
    """
    Run a full weather strategy backtest:
    1. Load candidate_strategy params
    2. Generate n_scenarios random weather days
    3. Detect edges and scale positions relative to current balance
    4. Settle trades, accumulate P&L
    5. Score the result

    If use_real_data=True and real_outcomes.json exists, the FIRST N scenarios
    (where N = number of real outcomes) use real settlement data. The remaining
    scenarios are filled with synthetic data. This ensures the strategy is
    optimized against real market behavior first, then explored with simulation.

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

    # Load real outcomes if requested
    real_scenarios = []
    if use_real_data:
        real_outcomes = load_real_outcomes()
        if real_outcomes:
            for outcome in real_outcomes:
                try:
                    markets, true_temps = real_outcome_to_scenario(outcome)
                    real_scenarios.append((markets, true_temps))
                except Exception:
                    continue  # skip malformed entries

    # Fetch settled Kalshi markets (always attempted — cached for 4 hours)
    kalshi_scenarios = fetch_recent_kalshi_settlements()
    # Weight real Kalshi scenarios 3x since they represent actual market behavior
    weighted_kalshi = kalshi_scenarios * 3

    # Build combined scenario list: real outcomes first, then Kalshi 3x, then synthetic
    all_real = real_scenarios + weighted_kalshi
    n_real = len(all_real)
    n_synthetic = max(0, n_scenarios - n_real)

    for scenario_idx in range(n_real + n_synthetic):
        if balance <= 1.0:
            break  # Account blown

        # Use real/Kalshi scenario data first, then synthetic
        if scenario_idx < n_real:
            markets, true_temps = all_real[scenario_idx]
        else:
            markets, true_temps = generate_weather_scenario(rng)

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

    metrics = score_simulation(all_results, initial_balance=initial_balance)
    return metrics


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
                return eval(val_part)
            except Exception:
                return val_part
    return None


def random_mutation(source: str) -> tuple[str, object, str]:
    """Pick a random parameter and a random value for it."""
    param = random.choice(list(MUTATION_SPACE.keys()))
    new_val = random.choice(MUTATION_SPACE[param])
    return param, new_val, "Random exploration"


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
):
    """
    Run the Weather AutoResearch loop.

    For each iteration:
    1. Read current candidate_strategy.py
    2. Pick a random parameter mutation
    3. Apply mutation, run weather backtest
    4. If score improves: keep and git commit
    5. If score worsens: revert
    6. Log everything

    Args:
        max_iterations: Number of experiments to run
        n_scenarios: Number of weather days to simulate per backtest
        verbose: Print progress to terminal
        use_real_data: If True, incorporate real settlement outcomes from
                       real_outcomes.json for the first N scenarios
    """
    max_iterations = max_iterations or config.AUTORESEARCH_MAX_ITERATIONS

    console.print("\n[bold cyan]================================================================[/bold cyan]")
    console.print("[bold cyan]  WEATHER AUTORESEARCH: Forecast Arbitrage Optimization Loop  [/bold cyan]")
    console.print("[bold cyan]================================================================[/bold cyan]\n")

    # Report real data status
    if use_real_data:
        real_outcomes = load_real_outcomes()
        if real_outcomes:
            console.print(
                f"[green]Real data mode: {len(real_outcomes)} real outcomes loaded "
                f"from {REAL_OUTCOMES_FILE.name}[/green]"
            )
        else:
            console.print(
                "[yellow]Real data mode requested but no weather outcomes found -- "
                "falling back to 100% synthetic[/yellow]"
            )

    # Report Kalshi settled data status (always fetched, independent of --use-real-data)
    kalshi_preview = fetch_recent_kalshi_settlements()
    if kalshi_preview:
        console.print(
            f"[green]Kalshi settled markets: {len(kalshi_preview)} scenarios "
            f"(weighted 3x = {len(kalshi_preview) * 3} effective scenarios)[/green]"
        )
    else:
        console.print(
            "[dim]No settled Kalshi weather markets available -- using synthetic only[/dim]"
        )

    # Baseline: score current strategy
    console.print("[dim]Running baseline weather backtest...[/dim]")
    baseline = run_weather_backtest(
        seed=42, n_scenarios=n_scenarios, use_real_data=use_real_data
    )
    if "error" in baseline:
        console.print(f"[red]Baseline failed: {baseline['error']}[/red]")
        return

    console.print(f"[green]Baseline score: {baseline['score']:.4f}[/green]")
    console.print(
        f"  Sortino: {baseline['sortino']:.3f} | "
        f"ROI: {baseline['roi_pct']:.2f}% | "
        f"Win: {baseline['win_rate']:.1f}% | "
        f"MaxDD: {baseline['max_dd_pct']:.2f}% | "
        f"Trades: {baseline['total_trades']} | "
        f"PF: {baseline['profit_factor']:.2f}\n"
    )

    best_score = baseline["score"]
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
        task = progress.add_task("Weather research loop", total=max_iterations)

        for i in range(1, max_iterations + 1):
            source = read_strategy_file()

            # Pick mutation
            param, new_val, reason = random_mutation(source)
            old_val = get_current_value(source, param)

            # Skip if same value
            if old_val == new_val:
                progress.update(task, advance=1)
                continue

            progress.update(task, description=f"Iter {i}/{max_iterations}: {param}={new_val}")

            # Apply mutation
            new_source = mutate_parameter(source, param, new_val)
            write_strategy_file(new_source)

            # Run weather backtest with slightly different seed per iteration
            metrics = run_weather_backtest(
                seed=42 + i, n_scenarios=n_scenarios, use_real_data=use_real_data
            )
            if "error" in metrics:
                write_strategy_file(source)  # revert on error
                progress.update(task, advance=1)
                continue

            new_score = metrics["score"]

            # Decision: keep or revert
            kept = new_score > best_score

            if kept:
                best_score = new_score
                improvements += 1
                git_commit(
                    f"WeatherResearch iter {i}: {param}={new_val} "
                    f"score={new_score:.4f} sortino={metrics['sortino']:.3f}"
                )
                if alert_research_improvement is not None:
                    try:
                        alert_research_improvement(param, old_val, new_val, new_score)
                    except Exception:
                        pass
                if verbose:
                    console.print(
                        f"  [green]+ Iter {i}: {param} {old_val}->{new_val} "
                        f"score={new_score:.4f} sortino={metrics['sortino']:.3f} "
                        f"roi={metrics['roi_pct']:.1f}% win={metrics['win_rate']:.0f}% KEPT[/green]"
                    )
            else:
                write_strategy_file(source)  # revert
                git_revert()
                if verbose and i % 5 == 0:
                    console.print(
                        f"  [dim]- Iter {i}: {param} {old_val}->{new_val} "
                        f"score={new_score:.4f} (vs {best_score:.4f}) REVERTED[/dim]"
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
            })

            progress.update(task, advance=1)
            time.sleep(0.05)  # brief pause

    # Final summary
    console.print(f"\n[bold cyan]{'=' * 65}[/bold cyan]")
    console.print(f"[bold]Weather Research Complete: {max_iterations} iterations[/bold]")
    console.print(f"  Improvements found: {improvements}")
    console.print(f"  Best score: {best_score:.4f} (baseline was {baseline['score']:.4f})")

    final = run_weather_backtest(
        seed=42, n_scenarios=n_scenarios, use_real_data=use_real_data
    )
    if "error" in final:
        console.print(f"[red]Final backtest failed: {final['error']}[/red]")
    else:
        console.print(f"\n[bold]Final Strategy Performance:[/bold]")
        console.print(f"  Sortino:       {final['sortino']:.3f}")
        console.print(f"  ROI:           {final['roi_pct']:.2f}%")
        console.print(f"  Win Rate:      {final['win_rate']:.1f}%")
        console.print(f"  Max Drawdown:  {final['max_dd_pct']:.2f}%")
        console.print(f"  Profit Factor: {final['profit_factor']:.2f}")
        console.print(f"  Total Trades:  {final['total_trades']}")
        console.print(f"  Total P&L:     ${final['total_pnl']:.2f}")
    console.print(f"\n  Results log: {RESULTS_LOG}")

    # Show current best params
    console.print(f"\n[bold]Optimized Parameters:[/bold]")
    params = load_strategy_params()
    if "error" not in params:
        console.print(f"  FORECAST_STDEV: {params['forecast_stdev']}")
        console.print(f"  EDGE_THRESHOLD: {params['edge_threshold_cents']}c")
        console.print(f"  CONTRACTS/TRADE: {params['contracts_per_trade']}")
        console.print(f"  NWS_WEIGHT: {params['nws_official_weight']}")
        console.print(f"  CITY_WEIGHTS: {params['city_weights']}")
        console.print(f"  BUCKET_MULT: {params['bucket_multiplier']} / THRESHOLD_MULT: {params['threshold_multiplier']}")

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
    parser = argparse.ArgumentParser(description="Weather AutoResearch Loop")
    parser.add_argument("--iterations", type=int, default=50, help="Number of experiments")
    parser.add_argument("--scenarios", type=int, default=200, help="Weather days per backtest")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    parser.add_argument("--no-claude", action="store_true", help="Random mutations only (no Claude API)")
    parser.add_argument(
        "--use-real-data", action="store_true",
        help="Incorporate real settlement outcomes from real_outcomes.json"
    )
    args = parser.parse_args()

    run_research(
        max_iterations=args.iterations,
        n_scenarios=args.scenarios,
        verbose=not args.quiet,
        use_real_data=args.use_real_data,
    )
