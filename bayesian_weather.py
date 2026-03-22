"""
bayesian_weather.py -- Bayesian Forecast Combination for Weather Markets.

MATHEMATICAL FOUNDATION
=======================

Traditional approach (WRONG):
    T_blend = w1*T_nws + w2*T_gfs + w3*T_hrrr
    Problem: averages TEMPERATURES, not DISTRIBUTIONS.
    Ignores that different sources have different uncertainties.
    A tight HRRR forecast (std=1.5F) should dominate a loose GFS ensemble
    (std=3.0F), but ad-hoc weights treat them equally.

Bayesian approach (CORRECT):
    Each forecast source provides a Gaussian likelihood:
        Source i: N(mu_i, sigma_i^2)
    Bayesian Gaussian conjugate update combines them:
        precision_i   = 1 / sigma_i^2
        posterior_prec = SUM(precision_i)
        posterior_mean = SUM(precision_i * mu_i) / posterior_prec
        posterior_std  = sqrt(1 / posterior_prec)

    This is mathematically optimal for Gaussian distributions.
    It TIGHTENS uncertainty (lower std than any single source) and SHIFTS
    the mean toward the most precise sources automatically.

    Key properties:
    - More sources always reduce uncertainty (precision is additive)
    - The most precise source gets the highest implicit weight
    - Posterior std is always <= min(source stds)
    - Equivalent to sequential Bayesian updating (order doesn't matter)

EDGE CALCULATION
================

    EV = p_hat - p_market
    where:
        p_hat    = Bayesian posterior probability of bucket
        p_market = Kalshi market price / 100

    Kelly sizing: f* = (b*p - q) / b
    where b = payout odds, p = win probability, q = 1 - p

    We use quarter-Kelly (f*/4) for conservative sizing.

Reference: "Real-Time Bayesian Signal Processing Agent Decision
Architecture" (QR-PM-2026-0041, Formulas 2-4)
"""

import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests
from rich.console import Console
from rich.table import Table

import config
from kalshi_client import KalshiClient
from weather_strategy import (
    NWS_GRID_POINTS,
    FORECAST_STDEV,
    parse_market_type,
)

console = Console()


# =============================================================================
# City coordinates (matching auto_trade.py exactly)
# =============================================================================

CITY_COORDS = {
    "KXHIGHNY":  (40.7829, -73.9654),   # NYC Central Park
    "KXHIGHCHI": (41.7868, -87.7522),    # Chicago Midway
    "KXHIGHMIA": (25.7933, -80.2906),    # Miami Intl Airport
    "KXHIGHLA":  (33.9425, -118.4081),   # Los Angeles LAX
    "KXHIGHDC":  (38.8512, -77.0402),    # Washington DC -- Reagan National
    "KXHIGHDEN": (39.8561, -104.6737),   # Denver Intl Airport
}

CITY_NAMES = {
    "KXHIGHNY":  "NYC",
    "KXHIGHCHI": "Chicago",
    "KXHIGHMIA": "Miami",
    "KXHIGHLA":  "LA",
    "KXHIGHDC":  "DC",
    "KXHIGHDEN": "Denver",
}

# HRRR fixed std: 1.5F for day 0-1 (very accurate short-range model, 3km res)
HRRR_FIXED_STD = 1.5


# =============================================================================
# Core data structures
# =============================================================================

@dataclass
class BayesianForecast:
    """Result of Bayesian combination of multiple forecast sources."""
    city: str
    date: str
    posterior_mean: float       # Bayesian combined forecast temperature (F)
    posterior_std: float        # Bayesian combined uncertainty (F)
    sources: list[dict]         # [{name, mean, std, precision, weight}]
    n_sources: int
    precision_gain: float       # How much uncertainty reduced vs best single source
                                # = 1 - (posterior_std / min_source_std)
                                # Higher = more information gain from combining


# =============================================================================
# Bayesian math: normal CDF and conjugate Gaussian update
# =============================================================================

def normal_cdf(x: float) -> float:
    """Standard normal CDF using math.erf (no scipy needed)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bayesian_combine(sources: list[dict]) -> Optional[BayesianForecast]:
    """
    Combine multiple forecast sources using Bayesian Gaussian conjugate update.

    Each source is a dict: {"name": str, "mean": float, "std": float}
      - mean: forecast temperature in Fahrenheit
      - std:  forecast standard deviation in Fahrenheit

    Returns a BayesianForecast with the posterior distribution, or None if
    no valid sources provided.

    Mathematical derivation:
        For N independent Gaussian observations of the same quantity:
            X_i ~ N(mu_i, sigma_i^2)

        The posterior (assuming flat/improper prior) is:
            precision_post = sum(1/sigma_i^2)
            mean_post      = sum(mu_i / sigma_i^2) / precision_post
            sigma_post     = sqrt(1 / precision_post)

        This is equivalent to the product of N Gaussian PDFs, normalized.
        The posterior precision is the SUM of individual precisions --
        more sources always tighten the estimate.
    """
    if not sources:
        return None

    # Filter out sources with invalid std
    valid = [s for s in sources if s.get("std", 0) > 0]
    if not valid:
        return None

    # Compute precisions
    precisions = []
    for s in valid:
        prec = 1.0 / (s["std"] ** 2)
        precisions.append(prec)

    total_precision = sum(precisions)

    # Posterior mean: precision-weighted average
    weighted_mean = sum(
        s["mean"] / (s["std"] ** 2) for s in valid
    ) / total_precision

    # Posterior std: inverse sqrt of total precision
    posterior_std = math.sqrt(1.0 / total_precision)

    # Build annotated source list with implicit Bayesian weights
    source_details = []
    for s, prec in zip(valid, precisions):
        source_details.append({
            "name": s["name"],
            "mean": round(s["mean"], 2),
            "std": round(s["std"], 2),
            "precision": round(prec, 4),
            "weight": round(prec / total_precision, 4),  # Bayesian implicit weight
        })

    # Precision gain: reduction in uncertainty vs best single source
    min_source_std = min(s["std"] for s in valid)
    precision_gain = 1.0 - (posterior_std / min_source_std)

    return BayesianForecast(
        city="",          # Caller sets this
        date="",          # Caller sets this
        posterior_mean=round(weighted_mean, 2),
        posterior_std=round(posterior_std, 3),
        sources=source_details,
        n_sources=len(valid),
        precision_gain=round(precision_gain, 4),
    )


# =============================================================================
# Probability calculations using the posterior distribution
# =============================================================================

def bayesian_bucket_probability(
    forecast: BayesianForecast,
    bucket_low: float,
    bucket_high: float,
) -> float:
    """
    Calculate probability that the true temperature falls in [bucket_low, bucket_high]
    using the Bayesian posterior distribution.

    P(low <= T <= high) = Phi((high - mu) / sigma) - Phi((low - mu) / sigma)

    where Phi is the standard normal CDF, mu is the posterior mean, and sigma
    is the posterior standard deviation.
    """
    if forecast.posterior_std <= 0:
        # Degenerate case: point estimate
        return 1.0 if bucket_low <= forecast.posterior_mean <= bucket_high else 0.0

    z_low = (bucket_low - forecast.posterior_mean) / forecast.posterior_std
    z_high = (bucket_high - forecast.posterior_mean) / forecast.posterior_std
    prob = normal_cdf(z_high) - normal_cdf(z_low)
    return max(0.001, min(0.999, prob))


def bayesian_above_probability(
    forecast: BayesianForecast,
    threshold: float,
) -> float:
    """P(T > threshold) using posterior distribution."""
    if forecast.posterior_std <= 0:
        return 1.0 if forecast.posterior_mean > threshold else 0.0

    z = (threshold - forecast.posterior_mean) / forecast.posterior_std
    prob = 1.0 - normal_cdf(z)
    return max(0.001, min(0.999, prob))


def bayesian_below_probability(
    forecast: BayesianForecast,
    threshold: float,
) -> float:
    """P(T < threshold) using posterior distribution."""
    if forecast.posterior_std <= 0:
        return 1.0 if forecast.posterior_mean < threshold else 0.0

    z = (threshold - forecast.posterior_mean) / forecast.posterior_std
    prob = normal_cdf(z)
    return max(0.001, min(0.999, prob))


# =============================================================================
# Edge and Kelly sizing calculations
# =============================================================================

def bayesian_edge(
    forecast: BayesianForecast,
    bucket_low: float,
    bucket_high: float,
    market_price_cents: float,
) -> dict:
    """
    Calculate expected value and Kelly sizing for a weather market.

    Core formula (Document Formula 4):
        EV = p_hat - p_market

    where:
        p_hat    = Bayesian posterior probability for the bucket
        p_market = market price expressed as probability (cents / 100)

    Kelly criterion for binary outcome:
        f* = (b * p - q) / b
    where:
        b = payout odds = (1 - p_market) / p_market  (for YES)
        p = our estimated win probability
        q = 1 - p

    We use quarter-Kelly (f*/4) for conservative sizing that accounts for
    model uncertainty not captured in the Gaussian posterior.

    Returns dict with: p_hat, p_market, ev, edge_cents, side,
                       full_kelly, quarter_kelly, confidence
    """
    p_hat = bayesian_bucket_probability(forecast, bucket_low, bucket_high)
    p_market = market_price_cents / 100.0

    # Clamp market probability away from extremes
    p_market = max(0.01, min(0.99, p_market))

    ev = p_hat - p_market  # This IS the edge (Formula 4)

    if ev > 0:
        # Buy YES: we think the event is more likely than the market
        b = (1.0 - p_market) / p_market  # odds offered
        full_kelly = (b * p_hat - (1.0 - p_hat)) / b
    else:
        # Buy NO: we think the event is less likely than the market
        p_no_hat = 1.0 - p_hat
        p_no_market = 1.0 - p_market
        b = (1.0 - p_no_market) / p_no_market  # NO odds
        full_kelly = (b * p_no_hat - (1.0 - p_no_hat)) / b

    quarter_kelly = max(0.0, full_kelly * 0.25)

    # Confidence tiers based on edge magnitude
    abs_ev = abs(ev)
    if abs_ev > 0.07:
        confidence = "high"
    elif abs_ev > 0.04:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "p_hat": round(p_hat, 4),
        "p_market": round(p_market, 4),
        "ev": round(ev, 4),
        "edge_cents": round(ev * 100, 2),
        "side": "buy_yes" if ev > 0 else "buy_no",
        "full_kelly": round(max(0.0, full_kelly), 4),
        "quarter_kelly": round(quarter_kelly, 4),
        "confidence": confidence,
    }


# =============================================================================
# Forecast fetching: NWS, GFS Ensemble, HRRR
# =============================================================================

def _fetch_nws_forecast(series_ticker: str) -> dict:
    """
    Fetch NWS official forecast for a city.
    Returns {date_str: temperature_F} mapping.

    NWS forecasts are free, updated ~hourly, and are the "ground truth"
    for Kalshi settlement. They use a deterministic model with human
    forecaster adjustments.
    """
    grid = NWS_GRID_POINTS.get(series_ticker)
    if not grid:
        return {}

    office, x, y = grid
    url = f"https://api.weather.gov/gridpoints/{office}/{x},{y}/forecast"

    try:
        headers = {"User-Agent": "ippo-bayesian-weather/1.0 (trading bot)"}
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
        return result
    except Exception as e:
        console.print(f"[yellow]NWS error ({series_ticker}): {e}[/yellow]")
        return {}


def _fetch_gfs_ensemble(series_ticker: str) -> dict:
    """
    Fetch GFS 31-member ensemble (30 perturbed + 1 control) from Open-Meteo.

    Returns {date_str: {"mean": float, "std": float, "n_members": int}}.

    The ensemble spread gives us a DATA-DRIVEN uncertainty estimate:
    we compute the std directly from the 31 member forecasts rather
    than using a hardcoded value.

    API: https://ensemble-api.open-meteo.com/v1/ensemble
    Returns temperature in Fahrenheit (temperature_unit=fahrenheit).
    """
    coords = CITY_COORDS.get(series_ticker)
    if not coords:
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
        member_data: dict[str, list[float]] = {}
        for key, values in daily.items():
            if key.startswith("temperature_2m_max_member") and values:
                for i, date_str in enumerate(dates):
                    if i < len(values) and values[i] is not None:
                        member_data.setdefault(date_str, []).append(values[i])

        result = {}
        for date_str, temps in member_data.items():
            if len(temps) >= 5:  # Need >= 5 members for meaningful stats
                mean_temp = sum(temps) / len(temps)
                variance = sum((t - mean_temp) ** 2 for t in temps) / len(temps)
                stdev = math.sqrt(variance) if variance > 0 else 1.5
                result[date_str] = {
                    "mean": mean_temp,
                    "std": max(stdev, 0.5),  # Floor at 0.5F to avoid degenerate precision
                    "n_members": len(temps),
                }

        return result
    except Exception as e:
        console.print(f"[yellow]GFS ensemble error ({series_ticker}): {e}[/yellow]")
        return {}


def _fetch_hrrr(series_ticker: str) -> dict:
    """
    Fetch HRRR (High-Resolution Rapid Refresh) from Open-Meteo.

    HRRR is a 3km-resolution CONUS model, updated hourly. It's the most
    accurate short-range forecast (0-18 hours) but has no coverage beyond
    ~48 hours. We only use it for day 0 and day 1.

    Returns {date_str: {"temp": float}} with temperatures in Fahrenheit.
    """
    coords = CITY_COORDS.get(series_ticker)
    if not coords:
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

        return result
    except Exception as e:
        console.print(f"[yellow]HRRR error ({series_ticker}): {e}[/yellow]")
        return {}


def fetch_all_forecasts(series_ticker: str, date_str: str) -> list[dict]:
    """
    Fetch forecasts from all available sources for a given city and date.

    Returns a list of source dicts ready for bayesian_combine():
        [{"name": "NWS", "mean": 72.0, "std": 2.5}, ...]

    Source uncertainties:
        NWS:  from candidate_strategy.py FORECAST_STDEV (per days_out)
        GFS:  computed from ensemble member spread (data-driven)
        HRRR: fixed 1.5F (empirically very accurate for short range)
    """
    now = datetime.now(timezone.utc)
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        days_out = max(0, (dt.date() - now.date()).days)
    except Exception:
        days_out = 1

    sources = []

    # --- Source 1: NWS official forecast ---
    nws_data = _fetch_nws_forecast(series_ticker)
    if date_str in nws_data:
        # NWS std from candidate_strategy or fallback to weather_strategy defaults
        try:
            from autoresearch.candidate_strategy import get_forecast_stdev
            stdev_map = get_forecast_stdev()
        except Exception:
            stdev_map = FORECAST_STDEV

        nws_std = stdev_map.get(min(days_out, 3), 4.0)
        sources.append({
            "name": "NWS",
            "mean": nws_data[date_str],
            "std": nws_std,
        })

    time.sleep(0.5)  # Rate limit

    # --- Source 2: GFS ensemble ---
    gfs_data = _fetch_gfs_ensemble(series_ticker)
    if date_str in gfs_data:
        sources.append({
            "name": "GFS_ensemble",
            "mean": gfs_data[date_str]["mean"],
            "std": gfs_data[date_str]["std"],
        })

    time.sleep(0.5)  # Rate limit

    # --- Source 3: HRRR (day 0-1 only) ---
    if days_out <= 1:
        hrrr_data = _fetch_hrrr(series_ticker)
        if date_str in hrrr_data:
            sources.append({
                "name": "HRRR",
                "mean": hrrr_data[date_str]["temp"],
                "std": HRRR_FIXED_STD,
            })

    return sources


# =============================================================================
# Full pipeline: scan all weather markets for Bayesian edges
# =============================================================================

def scan_bayesian_edges(client: KalshiClient = None) -> list[dict]:
    """
    Full Bayesian edge scanning pipeline:

    1. For each weather city series with open markets on Kalshi
    2. Fetch all forecast sources (NWS, GFS ensemble, HRRR)
    3. Combine Bayesianly (precision-weighted, not ad-hoc average)
    4. Calculate bucket probabilities from the posterior distribution
    5. Compare to market prices to find edges (EV = p_hat - p_market)
    6. Size with quarter-Kelly

    Returns list of edge dicts sorted by |edge| descending.
    """
    client = client or KalshiClient()
    all_edges = []

    for series_ticker in CITY_COORDS:
        city = CITY_NAMES.get(series_ticker, series_ticker)
        console.print(f"[dim]Scanning {city}...[/dim]")

        # Fetch all markets for this city series
        try:
            resp = client.get_markets(series_ticker=series_ticker, limit=50, status="open")
            markets = resp.get("markets", [])
        except Exception as e:
            console.print(f"[yellow]Kalshi API error for {series_ticker}: {e}[/yellow]")
            continue

        if not markets:
            continue

        # Group markets by settlement date to batch forecast fetches
        markets_by_date: dict[str, list[dict]] = {}
        for m in markets:
            ticker = m.get("ticker", "")
            # Extract date from ticker: KXHIGHCHI-26MAR21-T63 -> 2026-03-21
            date_match = re.search(r'(\d{2})([A-Z]{3})(\d{2})', ticker)
            if not date_match:
                continue

            year_short = date_match.group(1)
            month_str = date_match.group(2)
            day_str = date_match.group(3)

            month_map = {
                "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
                "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
                "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
            }
            month_num = month_map.get(month_str)
            if not month_num:
                continue

            market_date = f"20{year_short}-{month_num}-{day_str}"
            markets_by_date.setdefault(market_date, []).append(m)

        # For each unique date, fetch forecasts once and price all markets
        for market_date, date_markets in markets_by_date.items():
            # Fetch & combine forecasts for this city+date
            sources = fetch_all_forecasts(series_ticker, market_date)
            if not sources:
                continue

            forecast = bayesian_combine(sources)
            if forecast is None:
                continue

            # Set city/date on the forecast
            forecast.city = city
            forecast.date = market_date

            # Price each market using the Bayesian posterior
            for m in date_markets:
                ticker = m.get("ticker", "")
                title = m.get("title", "")
                yes_bid = float(m.get("yes_bid_dollars", 0) or 0) * 100
                yes_ask = float(m.get("yes_ask_dollars", 0) or 0) * 100
                volume = int(float(m.get("volume_fp", 0) or 0))

                # Skip settled or illiquid
                if yes_bid <= 0 and yes_ask <= 1:
                    continue
                if yes_bid >= 99:
                    continue

                # Parse market type (bucket, above, below)
                mtype = parse_market_type(ticker, title)
                if not mtype:
                    continue

                # Determine bucket bounds for edge calculation
                if mtype["type"] == "bucket":
                    b_low = mtype["low"]
                    b_high = mtype["high"]
                elif mtype["type"] == "above":
                    # P(T > threshold): use [threshold, +inf] approximated as [threshold, threshold+100]
                    edge_result = bayesian_edge(
                        forecast, mtype["threshold"], mtype["threshold"] + 100,
                        yes_ask if yes_ask > 0 else yes_bid + 1,
                    )
                    # Override p_hat with the proper above-probability
                    p_hat = bayesian_above_probability(forecast, mtype["threshold"])
                    p_market = (yes_ask if yes_ask > 0 else yes_bid + 1) / 100.0
                    p_market = max(0.01, min(0.99, p_market))
                    ev = p_hat - p_market

                    if ev > 0:
                        b_odds = (1.0 - p_market) / p_market
                        fk = (b_odds * p_hat - (1.0 - p_hat)) / b_odds
                    else:
                        p_no_hat = 1.0 - p_hat
                        p_no_mkt = 1.0 - p_market
                        b_odds = (1.0 - p_no_mkt) / p_no_mkt
                        fk = (b_odds * p_no_hat - (1.0 - p_no_hat)) / b_odds

                    abs_ev = abs(ev)
                    edge_result = {
                        "p_hat": round(p_hat, 4),
                        "p_market": round(p_market, 4),
                        "ev": round(ev, 4),
                        "edge_cents": round(ev * 100, 2),
                        "side": "buy_yes" if ev > 0 else "buy_no",
                        "full_kelly": round(max(0.0, fk), 4),
                        "quarter_kelly": round(max(0.0, fk * 0.25), 4),
                        "confidence": "high" if abs_ev > 0.07 else "medium" if abs_ev > 0.04 else "low",
                    }
                    _append_edge(all_edges, edge_result, ticker, title, city,
                                 market_date, forecast, mtype["threshold"], 999,
                                 yes_bid, yes_ask, volume)
                    continue

                elif mtype["type"] == "below":
                    # P(T < threshold): use [-100, threshold] approximated
                    p_hat = bayesian_below_probability(forecast, mtype["threshold"])
                    p_market = (yes_ask if yes_ask > 0 else yes_bid + 1) / 100.0
                    p_market = max(0.01, min(0.99, p_market))
                    ev = p_hat - p_market

                    if ev > 0:
                        b_odds = (1.0 - p_market) / p_market
                        fk = (b_odds * p_hat - (1.0 - p_hat)) / b_odds
                    else:
                        p_no_hat = 1.0 - p_hat
                        p_no_mkt = 1.0 - p_market
                        b_odds = (1.0 - p_no_mkt) / p_no_mkt
                        fk = (b_odds * p_no_hat - (1.0 - p_no_hat)) / b_odds

                    abs_ev = abs(ev)
                    edge_result = {
                        "p_hat": round(p_hat, 4),
                        "p_market": round(p_market, 4),
                        "ev": round(ev, 4),
                        "edge_cents": round(ev * 100, 2),
                        "side": "buy_yes" if ev > 0 else "buy_no",
                        "full_kelly": round(max(0.0, fk), 4),
                        "quarter_kelly": round(max(0.0, fk * 0.25), 4),
                        "confidence": "high" if abs_ev > 0.07 else "medium" if abs_ev > 0.04 else "low",
                    }
                    _append_edge(all_edges, edge_result, ticker, title, city,
                                 market_date, forecast, 0, mtype["threshold"],
                                 yes_bid, yes_ask, volume)
                    continue
                else:
                    continue

                # Bucket market: use standard bayesian_edge()
                market_price = yes_ask if yes_ask > 0 else yes_bid + 1
                edge_result = bayesian_edge(forecast, b_low, b_high, market_price)

                _append_edge(all_edges, edge_result, ticker, title, city,
                             market_date, forecast, b_low, b_high,
                             yes_bid, yes_ask, volume)

        time.sleep(0.5)  # Rate limit between cities

    # Sort by absolute edge size descending
    all_edges.sort(key=lambda e: abs(e["edge_cents"]), reverse=True)

    # Display results
    _display_bayesian_edges(all_edges)

    return all_edges


def _append_edge(
    all_edges: list[dict],
    edge_result: dict,
    ticker: str,
    title: str,
    city: str,
    market_date: str,
    forecast: BayesianForecast,
    bucket_low: float,
    bucket_high: float,
    yes_bid: float,
    yes_ask: float,
    volume: int,
) -> None:
    """Append an edge to the results list if it meets minimum threshold (2 cents)."""
    if abs(edge_result["edge_cents"]) < 2.0:
        return

    all_edges.append({
        "ticker": ticker,
        "title": title[:60],
        "city": city,
        "date": market_date,
        "posterior_mean": forecast.posterior_mean,
        "posterior_std": forecast.posterior_std,
        "n_sources": forecast.n_sources,
        "precision_gain": forecast.precision_gain,
        "sources": forecast.sources,
        "bucket_low": bucket_low,
        "bucket_high": bucket_high,
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "volume": volume,
        **edge_result,
    })


# =============================================================================
# Display
# =============================================================================

def _display_bayesian_edges(edges: list[dict]) -> None:
    """Pretty-print Bayesian edge scan results using Rich."""
    if not edges:
        console.print("\n[yellow]No Bayesian weather edges found.[/yellow]")
        return

    console.print()
    table = Table(
        title="Bayesian Weather Edges (precision-weighted forecast combination)",
        show_lines=False,
        title_style="bold cyan",
    )
    table.add_column("City", style="cyan", width=8)
    table.add_column("Date", style="dim", width=10)
    table.add_column("Market", style="white", width=30)
    table.add_column("Forecast", justify="right", style="green", width=12)
    table.add_column("p_hat", justify="right", style="yellow", width=7)
    table.add_column("p_mkt", justify="right", width=7)
    table.add_column("Edge", justify="right", style="bold", width=8)
    table.add_column("Side", style="bold", width=8)
    table.add_column("Kelly", justify="right", width=7)
    table.add_column("Srcs", justify="center", width=4)
    table.add_column("Conf", width=6)

    for e in edges:
        # Color-code the edge
        edge_val = e["edge_cents"]
        if abs(edge_val) >= 7:
            edge_color = "bold green" if edge_val > 0 else "bold red"
        elif abs(edge_val) >= 4:
            edge_color = "green" if edge_val > 0 else "red"
        else:
            edge_color = "dim"

        side_color = "green" if e["side"] == "buy_yes" else "red"
        side_text = "YES" if e["side"] == "buy_yes" else "NO"

        conf_color = {
            "high": "bold green",
            "medium": "yellow",
            "low": "dim",
        }.get(e["confidence"], "dim")

        forecast_str = f"{e['posterior_mean']:.1f}F +/-{e['posterior_std']:.1f}"

        table.add_row(
            e["city"],
            e["date"],
            e["title"][:30],
            forecast_str,
            f"{e['p_hat']:.0%}",
            f"{e['p_market']:.0%}",
            f"[{edge_color}]{edge_val:+.1f}c[/{edge_color}]",
            f"[{side_color}]{side_text}[/{side_color}]",
            f"{e['quarter_kelly']:.1%}",
            str(e["n_sources"]),
            f"[{conf_color}]{e['confidence']}[/{conf_color}]",
        )

    console.print(table)

    # Summary stats
    high_conf = [e for e in edges if e["confidence"] == "high"]
    total_ev = sum(abs(e["edge_cents"]) for e in edges)
    console.print(
        f"\n[dim]Found {len(edges)} edges "
        f"({len(high_conf)} high confidence). "
        f"Total |edge| = {total_ev:.1f} cents.[/dim]"
    )

    # Show source breakdown for the top edge
    if edges:
        top = edges[0]
        console.print(f"\n[bold]Top edge detail: {top['ticker']}[/bold]")
        console.print(f"  Posterior: {top['posterior_mean']:.2f}F +/- {top['posterior_std']:.2f}F")
        console.print(f"  Precision gain: {top['precision_gain']:.1%} reduction in uncertainty")
        for src in top.get("sources", []):
            console.print(
                f"  [{src['name']}] mean={src['mean']:.1f}F "
                f"std={src['std']:.1f}F "
                f"weight={src['weight']:.0%}"
            )


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    console.print("[bold cyan]Bayesian Weather Edge Scanner[/bold cyan]")
    console.print("[dim]Combining NWS + GFS ensemble + HRRR with Bayesian updating[/dim]\n")

    edges = scan_bayesian_edges()

    if edges:
        console.print(f"\n[bold green]Scan complete: {len(edges)} edges found.[/bold green]")
    else:
        console.print("\n[yellow]Scan complete: no edges found.[/yellow]")
