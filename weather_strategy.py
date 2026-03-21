"""
weather_strategy.py — Weather forecast vs. market price edge detection.

STRATEGY: Compare NWS (National Weather Service) forecasts to Kalshi weather
market prices. NWS forecasts are free, updated hourly, and surprisingly accurate
for 1-2 day horizons. When the market price diverges from the forecast-implied
probability, we have an edge.

KEY INSIGHT: Weather bucket markets (e.g., "will temp be 57-58°?") are priced
by retail traders who often don't check the latest NWS forecast. The NWS forecast
updates every few hours and is quite accurate for next-day highs (±2-3°F).

EDGE SOURCES:
1. Market hasn't caught up to latest forecast update
2. Bucket probabilities don't properly account for forecast uncertainty (±2-3°F)
3. Tail markets (>X° or <X°) are often mispriced vs cumulative forecast distribution
"""

import json
import math
from dataclasses import dataclass
from typing import Optional

import requests
from rich.console import Console
from rich.table import Table

import config
from kalshi_client import KalshiClient

console = Console()

# NWS forecast grid points for each city
# These are the grid coordinates for the NWS API
NWS_GRID_POINTS = {
    "KXHIGHNY":  ("OKX", 34, 38),     # NYC Central Park (exact Kalshi settlement location)
    "KXHIGHCHI": ("LOT", 72, 69),     # Chicago Midway Airport (exact Kalshi settlement location)
    "KXHIGHMIA": ("MFL", 106, 51),    # Miami Intl Airport (exact Kalshi settlement location)
    "KXHIGHLA":  ("LOX", 154, 44),    # Los Angeles (TBD - needs verification)
    "KXHIGHDC":  ("LWX", 97, 69),     # Washington DC — Reagan National (KDCA)
    "KXHIGHDEN": ("BOU", 74, 66),     # Denver Intl Airport (KDEN)
}

# Forecast uncertainty: standard deviation in degrees F for NWS forecasts
# Day 1: ±2°F, Day 2: ±3°F, Day 3: ±4°F (empirical estimates)
FORECAST_STDEV = {
    0: 1.5,   # Same day (very accurate)
    1: 2.5,   # Tomorrow
    2: 3.5,   # Day after
    3: 4.5,   # 3 days out
}


@dataclass
class WeatherEdge:
    """A detected weather market edge."""
    ticker: str
    title: str
    city: str
    market_yes_price: float   # Current yes price in cents
    market_yes_ask: float     # Current ask in cents
    fair_value: float         # Our estimated fair value in cents
    edge: float               # fair_value - market_price (positive = buy yes)
    forecast_temp: float      # NWS forecast temperature
    bucket_low: float         # Low end of temperature bucket
    bucket_high: float        # High end of temperature bucket
    side: str                 # "buy_yes" or "buy_no"
    confidence: str           # "high", "medium", "low"
    volume: int


def fetch_nws_forecast(series_ticker: str) -> Optional[dict]:
    """
    Fetch forecast from the free NWS API.
    Returns dict mapping date string (YYYY-MM-DD) to forecast high temp.
    This avoids the day-offset bug by matching on actual dates.
    """
    grid = NWS_GRID_POINTS.get(series_ticker)
    if not grid:
        return None

    office, x, y = grid
    url = f"https://api.weather.gov/gridpoints/{office}/{x},{y}/forecast"

    try:
        headers = {"User-Agent": "kalshi-weather-bot/1.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # Build a date -> high temp mapping from daytime periods
        forecasts_by_date = {}
        for p in data.get("properties", {}).get("periods", []):
            if p.get("isDaytime", False):
                # Extract date from startTime (e.g., "2026-03-21T06:00:00-05:00")
                start = p.get("startTime", "")
                if start:
                    date_str = start[:10]  # "2026-03-21"
                    forecasts_by_date[date_str] = {
                        "temperature": p["temperature"],
                        "name": p["name"],
                        "shortForecast": p.get("shortForecast", ""),
                    }
        return forecasts_by_date
    except Exception as e:
        console.print(f"[yellow]NWS forecast error for {series_ticker}: {e}[/yellow]")
        return None


def normal_cdf(x: float) -> float:
    """Standard normal CDF approximation."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def calc_bucket_probability(
    forecast_temp: float,
    bucket_low: float,
    bucket_high: float,
    stdev: float,
) -> float:
    """
    Calculate probability that actual temp falls in [bucket_low, bucket_high].
    Uses normal distribution centered on forecast with given stdev.
    """
    z_low = (bucket_low - forecast_temp) / stdev
    z_high = (bucket_high - forecast_temp) / stdev
    prob = normal_cdf(z_high) - normal_cdf(z_low)
    return max(0.001, min(0.999, prob))


def calc_above_probability(
    forecast_temp: float,
    threshold: float,
    stdev: float,
) -> float:
    """Probability that actual temp is above threshold."""
    z = (threshold - forecast_temp) / stdev
    return max(0.001, min(0.999, 1 - normal_cdf(z)))


def calc_below_probability(
    forecast_temp: float,
    threshold: float,
    stdev: float,
) -> float:
    """Probability that actual temp is below threshold."""
    z = (threshold - forecast_temp) / stdev
    return max(0.001, min(0.999, normal_cdf(z)))


def parse_market_type(ticker: str, title: str) -> Optional[dict]:
    """
    Parse a weather market ticker/title to extract:
    - type: "bucket" (57-58°), "above" (>60°), "below" (<53°)
    - threshold or bucket bounds
    """
    title_lower = title.lower()

    if "be >" in title_lower or "be >" in title:
        # Above threshold: "be >60°"
        for part in title.split(">"):
            part = part.strip().rstrip("°").rstrip("?").strip()
            try:
                threshold = float(part.split("°")[0].split()[0])
                return {"type": "above", "threshold": threshold}
            except (ValueError, IndexError):
                continue

    elif "be <" in title_lower or "be <" in title:
        # Below threshold: "be <53°"
        for part in title.split("<"):
            part = part.strip().rstrip("°").rstrip("?").strip()
            try:
                threshold = float(part.split("°")[0].split()[0])
                return {"type": "below", "threshold": threshold}
            except (ValueError, IndexError):
                continue

    elif "-" in title and "°" in title:
        # Bucket: "be 57-58°"
        # Try to extract from ticker: B57.5 means bucket centered at 57.5 (so 57-58)
        if "-B" in ticker:
            try:
                center = float(ticker.split("-B")[-1])
                return {"type": "bucket", "low": center - 0.5, "high": center + 0.5}
            except (ValueError, IndexError):
                pass

        # Fallback: parse from title
        import re
        match = re.search(r'(\d+)-(\d+)°', title)
        if match:
            low = float(match.group(1))
            high = float(match.group(2))
            return {"type": "bucket", "low": low, "high": high}

    # Try parsing threshold from ticker
    if "-T" in ticker:
        try:
            threshold = float(ticker.split("-T")[-1])
            # Determine above/below from title
            if "<" in title:
                return {"type": "below", "threshold": threshold}
            else:
                return {"type": "above", "threshold": threshold}
        except (ValueError, IndexError):
            pass

    return None


def find_weather_edges(client: KalshiClient = None) -> list[WeatherEdge]:
    """
    Main strategy function: find all weather market edges.

    1. Fetch NWS forecasts for each city
    2. Get Kalshi market prices
    3. Calculate fair values using forecast + uncertainty model
    4. Return edges sorted by magnitude
    """
    client = client or KalshiClient()
    all_edges = []

    for series_ticker, (office, x, y) in NWS_GRID_POINTS.items():
        city = {
            "KXHIGHNY": "NYC",
            "KXHIGHCHI": "Chicago",
            "KXHIGHMIA": "Miami",
            "KXHIGHLA": "LA",
            "KXHIGHDC": "DC",
            "KXHIGHDEN": "Denver",
        }.get(series_ticker, series_ticker)

        # Get forecast
        forecast_periods = fetch_nws_forecast(series_ticker)
        if not forecast_periods:
            continue

        # Get markets
        try:
            resp = client.get_markets(series_ticker=series_ticker, limit=50, status="open")
            markets = resp.get("markets", [])
        except Exception:
            continue

        # Match each market to a forecast period
        for m in markets:
            ticker = m.get("ticker", "")
            title = m.get("title", "")
            yes_bid = float(m.get("yes_bid_dollars", 0) or 0) * 100
            yes_ask = float(m.get("yes_ask_dollars", 0) or 0) * 100
            volume = int(float(m.get("volume_fp", 0) or 0))

            # Skip settled or illiquid markets
            if yes_bid <= 0 and yes_ask <= 1:
                continue
            if yes_bid >= 99:
                continue

            # Parse market type
            mtype = parse_market_type(ticker, title)
            if not mtype:
                continue

            # Extract the settlement date from the ticker
            # Tickers look like: KXHIGHCHI-26MAR21-T63
            # "26MAR21" = 2026-03-21
            import re
            from datetime import datetime, timezone, timedelta
            date_match = re.search(r'(\d{2})([A-Z]{3})(\d{2})', ticker)
            if not date_match:
                continue

            year_short = date_match.group(1)  # "26"
            month_str = date_match.group(2)   # "MAR"
            day_str = date_match.group(3)     # "21"

            month_map = {"JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
                         "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
                         "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"}
            month_num = month_map.get(month_str)
            if not month_num:
                continue

            market_date = f"20{year_short}-{month_num}-{day_str}"  # "2026-03-21"

            # Look up forecast for this exact date
            if market_date not in forecast_periods:
                continue

            forecast_temp = forecast_periods[market_date]["temperature"]

            # Calculate stdev based on how far out the date is
            now = datetime.now(timezone.utc)
            try:
                market_dt = datetime.strptime(market_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                days_out = max(0, (market_dt.date() - now.date()).days)
            except Exception:
                days_out = 1

            stdev = FORECAST_STDEV.get(min(days_out, 3), 4.0)

            # Calculate fair value
            if mtype["type"] == "bucket":
                fair_prob = calc_bucket_probability(
                    forecast_temp, mtype["low"], mtype["high"], stdev
                )
                bucket_low = mtype["low"]
                bucket_high = mtype["high"]
            elif mtype["type"] == "above":
                fair_prob = calc_above_probability(forecast_temp, mtype["threshold"], stdev)
                bucket_low = mtype["threshold"]
                bucket_high = 999
            elif mtype["type"] == "below":
                fair_prob = calc_below_probability(forecast_temp, mtype["threshold"], stdev)
                bucket_low = 0
                bucket_high = mtype["threshold"]
            else:
                continue

            fair_value_cents = fair_prob * 100

            # Calculate edge
            # Use the ask price for buying, bid price for selling
            buy_price = yes_ask if yes_ask > 0 else yes_bid + 1
            sell_price = yes_bid

            buy_edge = fair_value_cents - buy_price
            sell_edge = sell_price - fair_value_cents

            if buy_edge > 3:  # At least 3 cents edge to buy YES
                all_edges.append(WeatherEdge(
                    ticker=ticker,
                    title=title[:60],
                    city=city,
                    market_yes_price=yes_bid,
                    market_yes_ask=yes_ask,
                    fair_value=round(fair_value_cents, 1),
                    edge=round(buy_edge, 1),
                    forecast_temp=forecast_temp,
                    bucket_low=bucket_low,
                    bucket_high=bucket_high,
                    side="buy_yes",
                    confidence="high" if buy_edge > 10 else "medium" if buy_edge > 5 else "low",
                    volume=volume,
                ))
            elif sell_edge > 3:  # At least 3 cents edge to sell (buy NO)
                all_edges.append(WeatherEdge(
                    ticker=ticker,
                    title=title[:60],
                    city=city,
                    market_yes_price=yes_bid,
                    market_yes_ask=yes_ask,
                    fair_value=round(fair_value_cents, 1),
                    edge=round(sell_edge, 1),
                    forecast_temp=forecast_temp,
                    bucket_low=bucket_low,
                    bucket_high=bucket_high,
                    side="buy_no",
                    confidence="high" if sell_edge > 10 else "medium" if sell_edge > 5 else "low",
                    volume=volume,
                ))

        import time
        time.sleep(0.8)

    # Sort by edge size
    all_edges.sort(key=lambda e: e.edge, reverse=True)
    return all_edges


def display_edges(edges: list[WeatherEdge]):
    """Pretty-print detected weather edges."""
    if not edges:
        console.print("[yellow]No weather edges found right now.[/yellow]")
        return

    table = Table(title="Weather Forecast vs. Market — Detected Edges", show_lines=False)
    table.add_column("City", style="cyan", width=8)
    table.add_column("Market", style="white", width=35)
    table.add_column("Forecast", justify="right", style="green", width=8)
    table.add_column("Mkt Price", justify="right", width=9)
    table.add_column("Fair Val", justify="right", style="yellow", width=8)
    table.add_column("Edge", justify="right", style="bold green", width=7)
    table.add_column("Action", style="bold", width=10)
    table.add_column("Conf", width=6)

    for e in edges:
        conf_color = "green" if e.confidence == "high" else "yellow" if e.confidence == "medium" else "dim"
        action_color = "green" if e.side == "buy_yes" else "red"
        action_text = "BUY YES" if e.side == "buy_yes" else "BUY NO"
        price_text = f"{e.market_yes_ask:.0f}c" if e.side == "buy_yes" else f"{100-e.market_yes_price:.0f}c"

        table.add_row(
            e.city,
            e.title[:35],
            f"{e.forecast_temp:.0f}°F",
            price_text,
            f"{e.fair_value:.0f}c",
            f"+{e.edge:.0f}c",
            f"[{action_color}]{action_text}[/{action_color}]",
            f"[{conf_color}]{e.confidence}[/{conf_color}]",
        )

    console.print(table)
    console.print(f"\n[dim]Found {len(edges)} edges. Forecast source: NWS (weather.gov)[/dim]")


if __name__ == "__main__":
    edges = find_weather_edges()
    display_edges(edges)
