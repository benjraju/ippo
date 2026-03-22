"""
real_data_collector.py — Fetch REAL settled Kalshi weather market data for backtesting.

Pulls all settled markets from Kalshi weather series (PROD API) and matches
them with actual observed temperatures from NWS observation stations.

Output: output/kalshi_settlements.json
"""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from rich.console import Console

import config
from kalshi_client import KalshiClient

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEATHER_SERIES = [
    "KXHIGHNY",
    "KXHIGHCHI",
    "KXHIGHMIA",
    "KXHIGHLA",
    "KXHIGHDC",
    "KXHIGHDEN",
]

# ICAO station IDs matching Kalshi's settlement locations
# (mirrored from settlement_tracker.py)
NWS_OBSERVATION_STATIONS = {
    "KXHIGHNY":  "KNYC",
    "KXHIGHCHI": "KMDW",
    "KXHIGHMIA": "KMIA",
    "KXHIGHLA":  "KLAX",
    "KXHIGHDC":  "KDCA",
    "KXHIGHDEN": "KDEN",
}

SERIES_TO_CITY = {
    "KXHIGHNY":  "NYC",
    "KXHIGHCHI": "Chicago",
    "KXHIGHMIA": "Miami",
    "KXHIGHLA":  "LA",
    "KXHIGHDC":  "DC",
    "KXHIGHDEN": "Denver",
}

OUTPUT_PATH = config.OUTPUT_DIR / "kalshi_settlements.json"

# ---------------------------------------------------------------------------
# Title Parsing
# ---------------------------------------------------------------------------

def parse_market_title(title: str) -> dict:
    """
    Parse market type (bucket / above / below) and thresholds from the title.

    Examples:
        "be 59-60"   -> bucket, low=59, high=60
        "be >61"     -> above, threshold=61
        "be above 61"-> above, threshold=61
        "be <54"     -> below, threshold=54
        "be below 54"-> below, threshold=54
    """
    # Strip markdown bold markers
    clean = title.replace("**", "")

    result = {
        "market_type": "unknown",
        "bucket_low": None,
        "bucket_high": None,
        "threshold": None,
    }

    # Bucket: "be 59-60" or "be 59 - 60" (with optional degree symbol)
    m = re.search(r'be\s+(-?\d+(?:\.\d+)?)\s*[-\u2013]\s*(-?\d+(?:\.\d+)?)', clean, re.IGNORECASE)
    if m:
        result["market_type"] = "bucket"
        result["bucket_low"] = float(m.group(1))
        result["bucket_high"] = float(m.group(2))
        return result

    # Above: "be >61" or "be above 61" or ">= 61"
    m = re.search(r'(?:be\s+)?(?:>|above|at or above|at least)\s*=?\s*(-?\d+(?:\.\d+)?)', clean, re.IGNORECASE)
    if m:
        result["market_type"] = "above"
        result["threshold"] = float(m.group(1))
        return result

    # Below: "be <54" or "be below 54" or "<= 54"
    m = re.search(r'(?:be\s+)?(?:<|below|at or below|at most)\s*=?\s*(-?\d+(?:\.\d+)?)', clean, re.IGNORECASE)
    if m:
        result["market_type"] = "below"
        result["threshold"] = float(m.group(1))
        return result

    return result


def dollars_to_cents(val) -> Optional[int]:
    """Convert a dollar float to integer cents. Returns None if input is None."""
    if val is None:
        return None
    try:
        return int(round(float(val) * 100))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# NWS Observation Fetching
# ---------------------------------------------------------------------------

def fetch_actual_high_temp(station_id: str, date_str: str) -> Optional[float]:
    """
    Fetch the actual recorded high temperature from NWS observations.

    Args:
        station_id: ICAO station ID (e.g. "KNYC")
        date_str: Date in YYYY-MM-DD format

    Returns:
        High temperature in Fahrenheit, or None if unavailable.
    """
    start = f"{date_str}T00:00:00Z"
    # End = next day
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None
    from datetime import timedelta
    end_date = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
    end = f"{end_date}T00:00:00Z"

    url = f"https://api.weather.gov/stations/{station_id}/observations"
    params = {"start": start, "end": end}
    headers = {"User-Agent": "ippo-backtest-collector/1.0"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        features = data.get("features", [])
        if not features:
            return None

        # Find the maximum temperature from maxTemperatureLast24Hours
        max_temp_c = None
        for obs in features:
            props = obs.get("properties", {})
            temp = props.get("maxTemperatureLast24Hours", {})
            if temp and temp.get("value") is not None:
                val = temp["value"]
                if max_temp_c is None or val > max_temp_c:
                    max_temp_c = val

        # Fallback: use the max of all individual temperature readings
        if max_temp_c is None:
            temps = []
            for obs in features:
                props = obs.get("properties", {})
                temp = props.get("temperature", {})
                if temp and temp.get("value") is not None:
                    temps.append(temp["value"])
            if temps:
                max_temp_c = max(temps)

        if max_temp_c is not None:
            return round(max_temp_c * 9 / 5 + 32, 1)

        return None

    except Exception as e:
        console.print(f"[dim]NWS fetch failed for {station_id} on {date_str}: {e}[/dim]")
        return None


# ---------------------------------------------------------------------------
# Date Extraction from Event Ticker
# ---------------------------------------------------------------------------

def extract_date_from_event(event_ticker: str) -> Optional[str]:
    """
    Extract a YYYY-MM-DD date string from an event ticker like KXHIGHNY-26MAR21.

    Format: SERIES-YYMONDD  (e.g. 26MAR21 -> 2026-03-21)
    """
    m = re.search(r'-(\d{2})([A-Z]{3})(\d{2})$', event_ticker)
    if not m:
        return None

    yy, mon_str, dd = m.group(1), m.group(2), m.group(3)

    months = {
        "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
        "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
        "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
    }
    mm = months.get(mon_str)
    if not mm:
        return None

    return f"20{yy}-{mm}-{dd}"


# ---------------------------------------------------------------------------
# Kalshi Market Fetcher
# ---------------------------------------------------------------------------

def fetch_settled_markets(client: KalshiClient, series: str) -> list[dict]:
    """
    Fetch ALL settled markets for a given series, handling pagination via cursor.
    """
    all_markets = []
    cursor = None

    while True:
        try:
            params = {
                "series_ticker": series,
                "status": "settled",
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor

            resp = client._request("GET", "/markets", params=params)
            markets = resp.get("markets", [])
            all_markets.extend(markets)

            # Check for next page
            cursor = resp.get("cursor")
            if not cursor or not markets:
                break

            time.sleep(0.3)  # Rate limit between Kalshi pages

        except Exception as e:
            console.print(f"[red]Error fetching {series}: {e}[/red]")
            break

    return all_markets


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    console.print("[bold cyan]Ippo Real Data Collector[/bold cyan]")
    console.print("[dim]Fetching settled Kalshi weather markets + NWS observed temps[/dim]\n")

    # Force PROD for real settlement data
    original_env = config.KALSHI_ENV
    config.KALSHI_ENV = "PROD"

    try:
        client = KalshiClient()
    except Exception as e:
        console.print(f"[red]Failed to initialize Kalshi client: {e}[/red]")
        return

    # Restore env
    config.KALSHI_ENV = original_env

    all_records = []

    # --- Phase 1: Fetch settled markets from Kalshi ---
    console.print("[bold]Phase 1: Fetching settled markets from Kalshi (PROD)[/bold]")

    for series in WEATHER_SERIES:
        city = SERIES_TO_CITY[series]
        console.print(f"  Fetching [cyan]{series}[/cyan] ({city})...", end=" ")

        markets = fetch_settled_markets(client, series)
        console.print(f"[green]{len(markets)} markets[/green]")

        for mkt in markets:
            ticker = mkt.get("ticker", "")
            title = mkt.get("title", "")
            event_ticker = mkt.get("event_ticker", "")

            # Parse market type from title
            parsed = parse_market_title(title)

            # Determine result
            result = mkt.get("result", "")

            record = {
                "ticker": ticker,
                "title": title,
                "series": series,
                "city": city,
                "event_ticker": event_ticker,
                "market_type": parsed["market_type"],
                "bucket_low": parsed["bucket_low"],
                "bucket_high": parsed["bucket_high"],
                "threshold": parsed["threshold"],
                "result": result,
                "last_price_cents": dollars_to_cents(mkt.get("last_price")),
                "yes_bid_cents": dollars_to_cents(mkt.get("yes_bid")),
                "yes_ask_cents": dollars_to_cents(mkt.get("yes_ask")),
                "no_bid_cents": dollars_to_cents(mkt.get("no_bid")),
                "no_ask_cents": dollars_to_cents(mkt.get("no_ask")),
                "volume": mkt.get("volume"),
                "open_interest": mkt.get("open_interest"),
                "close_time": mkt.get("close_time"),
                "settled_time": mkt.get("settle_timer_expiration_time") or mkt.get("expiration_time"),
                "actual_temp_f": None,
                "nws_station": NWS_OBSERVATION_STATIONS.get(series),
            }

            all_records.append(record)

        time.sleep(0.3)  # Rate limit between series

    console.print(f"\n  Total settled markets: [bold green]{len(all_records)}[/bold green]\n")

    # --- Phase 2: Fetch actual observed temperatures from NWS ---
    console.print("[bold]Phase 2: Fetching NWS observed temperatures[/bold]")

    # Group by (station, date) to avoid duplicate NWS calls
    nws_cache: dict[tuple[str, str], Optional[float]] = {}
    nws_lookups = 0
    nws_hits = 0

    for i, record in enumerate(all_records):
        series = record["series"]
        event_ticker = record["event_ticker"]
        station_id = NWS_OBSERVATION_STATIONS.get(series)

        if not station_id:
            continue

        date_str = extract_date_from_event(event_ticker)
        if not date_str:
            continue

        cache_key = (station_id, date_str)

        if cache_key in nws_cache:
            record["actual_temp_f"] = nws_cache[cache_key]
            continue

        # Fetch from NWS
        nws_lookups += 1
        if nws_lookups % 25 == 0 or nws_lookups == 1:
            console.print(
                f"  NWS request {nws_lookups}: "
                f"[cyan]{station_id}[/cyan] on {date_str}..."
            )

        temp = fetch_actual_high_temp(station_id, date_str)
        nws_cache[cache_key] = temp
        record["actual_temp_f"] = temp

        if temp is not None:
            nws_hits += 1

        time.sleep(0.5)  # NWS rate limit

    console.print(
        f"\n  NWS lookups: {nws_lookups} | "
        f"Temps found: [green]{nws_hits}[/green] | "
        f"Missing: [yellow]{nws_lookups - nws_hits}[/yellow]\n"
    )

    # --- Phase 3: Save output ---
    output = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "total_markets": len(all_records),
        "series_counts": {
            series: sum(1 for r in all_records if r["series"] == series)
            for series in WEATHER_SERIES
        },
        "markets": all_records,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    console.print(f"[bold green]Saved {len(all_records)} markets to {OUTPUT_PATH}[/bold green]")

    # Summary stats
    with_temps = sum(1 for r in all_records if r["actual_temp_f"] is not None)
    yes_results = sum(1 for r in all_records if r["result"] == "yes")
    no_results = sum(1 for r in all_records if r["result"] == "no")
    bucket_count = sum(1 for r in all_records if r["market_type"] == "bucket")
    above_count = sum(1 for r in all_records if r["market_type"] == "above")
    below_count = sum(1 for r in all_records if r["market_type"] == "below")

    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Markets with NWS temps: {with_temps}/{len(all_records)}")
    console.print(f"  Results: [green]{yes_results} yes[/green] / [red]{no_results} no[/red]")
    console.print(f"  Types: {bucket_count} bucket / {above_count} above / {below_count} below")


if __name__ == "__main__":
    main()
