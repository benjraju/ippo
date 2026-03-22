"""
historical_backtest.py — Pull ALL settled Kalshi markets, backtest every strategy
against real outcomes, and feed results to AutoResearch.

Usage:
    python historical_backtest.py fetch       # Pull all settled markets from Kalshi
    python historical_backtest.py backtest    # Run models against historical data
    python historical_backtest.py full        # fetch + backtest + save for autoresearch

This replaces synthetic "dumb retail" scenarios with REAL market data.
"""

import json
import math
import sys
import time
import re
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Optional

import config
from kalshi_client import KalshiClient

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUT_DIR = config.OUTPUT_DIR
SETTLEMENTS_FILE = OUTPUT_DIR / "historical_settlements.json"
BACKTEST_RESULTS_FILE = OUTPUT_DIR / "historical_backtest_results.json"
AUTORESEARCH_SCENARIOS_FILE = Path("autoresearch") / "real_market_scenarios.json"
LOG_FILE = OUTPUT_DIR / "historical_backtest.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, mode="a")],
)
logger = logging.getLogger("historical_backtest")

# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def lognormal_bucket_prob(spot, low, high, vol, hours):
    if hours <= 0:
        hours = 0.1
    t = hours / (365 * 24)
    sigma = vol * math.sqrt(t)
    if sigma < 0.001:
        return 1.0 if low <= spot <= high else 0.0
    z_high = math.log(high / spot) / sigma
    z_low = math.log(low / spot) / sigma
    return max(0.0, normal_cdf(z_high) - normal_cdf(z_low))


def normal_bucket_prob(forecast, low, high, stdev):
    if stdev < 0.01:
        return 1.0 if low <= forecast <= high else 0.0
    return max(0.0, normal_cdf((high - forecast) / stdev) - normal_cdf((low - forecast) / stdev))


# ---------------------------------------------------------------------------
# STEP 1: Fetch ALL settled markets from Kalshi
# ---------------------------------------------------------------------------

ALL_SERIES = [
    # Weather
    "KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHLA", "KXHIGHDC", "KXHIGHDEN",
    # Crypto
    "KXBTC", "KXETH", "KXSOL",
    # Sports
    "KXNBAGAME", "KXNBAPTS",
]


def fetch_all_settled_markets():
    """Paginate through ALL settled markets for every series."""
    logger.info("=" * 60)
    logger.info("FETCHING ALL SETTLED MARKETS FROM KALSHI")
    logger.info("=" * 60)

    client = KalshiClient()
    all_markets = []

    for series in ALL_SERIES:
        cursor = None
        series_count = 0

        for page in range(50):  # Max 50 pages per series
            try:
                resp = client.get_markets(
                    series_ticker=series,
                    status="settled",
                    limit=100,
                    cursor=cursor,
                )
            except Exception as e:
                logger.warning(f"  {series} page {page} failed: {e}")
                break

            markets = resp.get("markets", [])
            if not markets:
                break

            for mkt in markets:
                all_markets.append({
                    "ticker": mkt.get("ticker", ""),
                    "title": mkt.get("title", ""),
                    "series_ticker": series,
                    "status": mkt.get("status", ""),
                    "result": mkt.get("result", ""),
                    "close_time": mkt.get("close_time", ""),
                    "yes_bid": mkt.get("yes_bid"),
                    "yes_ask": mkt.get("yes_ask"),
                    "no_bid": mkt.get("no_bid"),
                    "no_ask": mkt.get("no_ask"),
                    "last_price": mkt.get("last_price"),
                    "volume": mkt.get("volume", 0),
                    "open_time": mkt.get("open_time", ""),
                })
                series_count += 1

            cursor = resp.get("cursor")
            if not cursor:
                break

            time.sleep(0.3)  # Rate limit

        logger.info(f"  {series}: {series_count} settled markets")

    # Classify each market
    for mkt in all_markets:
        ticker = mkt["ticker"].upper()
        if "KXHIGH" in ticker:
            mkt["category"] = "weather"
            for c in ["NY", "CHI", "MIA", "LA", "DC", "DEN"]:
                if c in ticker:
                    mkt["sub_category"] = c
                    break
            else:
                mkt["sub_category"] = "unknown"
        elif "KXBTC" in ticker:
            mkt["category"] = "crypto"
            mkt["sub_category"] = "BTC"
        elif "KXETH" in ticker:
            mkt["category"] = "crypto"
            mkt["sub_category"] = "ETH"
        elif "KXSOL" in ticker:
            mkt["category"] = "crypto"
            mkt["sub_category"] = "SOL"
        elif "KXNBAGAME" in ticker:
            mkt["category"] = "sports"
            mkt["sub_category"] = "nba_winner"
        elif "KXNBAPTS" in ticker:
            mkt["category"] = "sports"
            mkt["sub_category"] = "nba_props"
        else:
            mkt["category"] = "other"
            mkt["sub_category"] = "unknown"

        # Parse bucket bounds from title
        mkt["bucket"] = parse_bounds(mkt["title"], mkt["ticker"])

    # Save
    data = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "total_markets": len(all_markets),
        "markets": all_markets,
    }

    with open(SETTLEMENTS_FILE, "w") as f:
        json.dump(data, f, indent=2)

    # Summary
    cats = defaultdict(int)
    for m in all_markets:
        cats[f"{m['category']}/{m['sub_category']}"] += 1

    logger.info(f"\nTotal: {len(all_markets)} settled markets")
    for k in sorted(cats.keys()):
        logger.info(f"  {k}: {cats[k]}")

    return all_markets


def parse_bounds(title: str, ticker: str) -> Optional[dict]:
    """Parse bucket/threshold bounds from title or ticker."""
    if not title:
        return None

    def clean(s):
        return float(s.replace(",", "").replace("$", "").replace("°", "").strip())

    # "$X to $Y" or "X to Y"
    m = re.search(r'\$?([\d,.]+)\s+to\s+\$?([\d,.]+)', title)
    if m:
        return {"type": "bucket", "low": clean(m.group(1)), "high": clean(m.group(2))}

    # "between $X and $Y"
    m = re.search(r'between\s+\$?([\d,.]+)\s+and\s+\$?([\d,.]+)', title, re.I)
    if m:
        return {"type": "bucket", "low": clean(m.group(1)), "high": clean(m.group(2))}

    # "be X-Y" (weather degrees)
    m = re.search(r'be\s+([\d.]+)[-–]([\d.]+)', title)
    if m:
        return {"type": "bucket", "low": float(m.group(1)), "high": float(m.group(2))}

    # ">X" / "above X"
    m = re.search(r'(?:>|above|or above|at least|or more)\s*\$?([\d,.]+)', title, re.I)
    if m:
        return {"type": "above", "low": clean(m.group(1)), "high": 1e9}

    # "<X" / "below X"
    m = re.search(r'(?:<|below|under|or below|less than)\s*\$?([\d,.]+)', title, re.I)
    if m:
        return {"type": "below", "low": 0, "high": clean(m.group(1))}

    # Ticker-based fallback for weather: -B57.5 or -T63
    if "-B" in ticker:
        try:
            center = float(ticker.split("-B")[-1])
            return {"type": "bucket", "low": math.floor(center), "high": math.ceil(center)}
        except ValueError:
            pass
    if "-T" in ticker and "-LT" not in ticker:
        try:
            thresh = float(ticker.split("-T")[-1])
            return {"type": "above", "low": thresh, "high": 1e9}
        except ValueError:
            pass

    return None


# ---------------------------------------------------------------------------
# STEP 2: Fetch actual temperatures for weather markets
# ---------------------------------------------------------------------------

NWS_STATIONS = {
    "NY": "KNYC", "CHI": "KORD", "MIA": "KMIA",
    "LA": "KLAX", "DC": "KDCA", "DEN": "KDEN",
}


def fetch_actual_temp(city: str, date_str: str) -> Optional[float]:
    """Fetch actual high temp from NWS observations API."""
    import requests

    station = NWS_STATIONS.get(city)
    if not station:
        return None

    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        start = dt.strftime("%Y-%m-%dT00:00:00Z")
        end = (dt + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")

        url = f"https://api.weather.gov/stations/{station}/observations"
        resp = requests.get(url, params={"start": start, "end": end},
                          headers={"User-Agent": "ippo-backtest"}, timeout=10)
        data = resp.json()

        temps = []
        for obs in data.get("features", []):
            t = obs.get("properties", {}).get("temperature", {}).get("value")
            if t is not None:
                temps.append(t * 9 / 5 + 32)

        return max(temps) if temps else None
    except Exception:
        return None


def extract_date_from_ticker(ticker: str) -> Optional[str]:
    """Extract settlement date from ticker like KXHIGHNY-26MAR21-B57.5."""
    months = {"JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05",
              "JUN": "06", "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10",
              "NOV": "11", "DEC": "12"}

    m = re.search(r'(\d{2})([A-Z]{3})(\d{2})', ticker)
    if m:
        yy, mon, dd = m.group(1), m.group(2), m.group(3)
        mm = months.get(mon)
        if mm:
            return f"20{yy}-{mm}-{dd}"
    return None


# ---------------------------------------------------------------------------
# STEP 3: Fetch historical crypto prices
# ---------------------------------------------------------------------------

def fetch_historical_crypto_price(asset: str, timestamp_iso: str) -> Optional[float]:
    """Fetch crypto price at a specific historical time from CoinGecko."""
    import requests

    cg_map = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana"}
    cg_id = cg_map.get(asset)
    if not cg_id:
        return None

    try:
        dt = datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00"))
        # CoinGecko /history endpoint uses dd-mm-yyyy
        date_str = dt.strftime("%d-%m-%Y")
        resp = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{cg_id}/history",
            params={"date": date_str},
            timeout=10,
        )
        data = resp.json()
        return data.get("market_data", {}).get("current_price", {}).get("usd")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# STEP 4: Backtest all strategies against historical settlements
# ---------------------------------------------------------------------------

def backtest_all():
    """Run every model against historical settlements and grade accuracy."""
    logger.info("=" * 60)
    logger.info("HISTORICAL BACKTEST")
    logger.info("=" * 60)

    if not SETTLEMENTS_FILE.exists():
        logger.error("No settlements file. Run 'fetch' first.")
        return

    with open(SETTLEMENTS_FILE) as f:
        data = json.load(f)

    markets = data["markets"]
    logger.info(f"Loaded {len(markets)} settled markets")

    results = {
        "backtest_time": datetime.now(timezone.utc).isoformat(),
        "weather": backtest_weather(markets),
        "crypto": backtest_crypto(markets),
        "sports": backtest_sports(markets),
    }

    # Save results
    with open(BACKTEST_RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # Save scenarios for AutoResearch
    save_autoresearch_scenarios(markets, results)

    # Print report
    print_report(results)

    return results


def backtest_weather(markets: list) -> dict:
    """Backtest weather model against settled weather markets."""
    weather = [m for m in markets if m["category"] == "weather" and m.get("result")]
    logger.info(f"Weather: {len(weather)} settled markets to backtest")

    if not weather:
        return {"total": 0, "message": "No settled weather markets found"}

    try:
        from autoresearch.candidate_strategy import get_forecast_stdev
        stdevs = get_forecast_stdev()
    except Exception:
        stdevs = {0: 1.5, 1: 2.5, 2: 3.5, 3: 4.5}

    # Fetch actual temps (batch by city+date)
    temp_cache = {}
    predictions = []
    cities_needing_temps = defaultdict(set)

    for mkt in weather:
        city = mkt.get("sub_category", "")
        date = extract_date_from_ticker(mkt["ticker"])
        if city and date:
            cities_needing_temps[city].add(date)

    # Fetch temps
    for city, dates in cities_needing_temps.items():
        for date in sorted(dates):
            key = f"{city}_{date}"
            if key not in temp_cache:
                temp = fetch_actual_temp(city, date)
                if temp is not None:
                    temp_cache[key] = temp
                    logger.debug(f"  {city} {date}: {temp:.1f}°F")
                time.sleep(0.2)

    logger.info(f"  Fetched {len(temp_cache)} actual temps")

    # Now backtest each market
    for mkt in weather:
        city = mkt.get("sub_category", "")
        date = extract_date_from_ticker(mkt["ticker"])
        bucket = mkt.get("bucket")
        result = mkt["result"]  # "yes" or "no"
        last_price = mkt.get("last_price") or 50

        if not bucket or not date:
            continue

        actual_temp = temp_cache.get(f"{city}_{date}")
        if actual_temp is None:
            continue

        # Use actual temp as "forecast" (best case — what if forecast was perfect?)
        # Then vary stdev to see how model performs
        for stdev_label, stdev in [("tight_1.5", 1.5), ("medium_3.0", 3.0), ("loose_4.5", 4.5)]:
            if bucket["type"] == "bucket":
                fair = normal_bucket_prob(actual_temp, bucket["low"], bucket["high"], stdev)
            elif bucket["type"] == "above":
                fair = 1.0 - normal_cdf((bucket["low"] - actual_temp) / max(stdev, 0.01))
            elif bucket["type"] == "below":
                fair = normal_cdf((bucket["high"] - actual_temp) / max(stdev, 0.01))
            else:
                continue

            fair_cents = fair * 100
            market_cents = last_price

            # Would we have traded?
            edge = abs(fair_cents - market_cents)
            if edge < 3:
                side = "no_trade"
                correct = None
            elif fair_cents > market_cents:
                side = "buy_yes"
                correct = (result == "yes")
            else:
                side = "buy_no"
                correct = (result == "no")

            predictions.append({
                "ticker": mkt["ticker"],
                "city": city,
                "date": date,
                "stdev_label": stdev_label,
                "stdev": stdev,
                "actual_temp": actual_temp,
                "bucket": bucket,
                "fair_cents": round(fair_cents, 1),
                "market_cents": market_cents,
                "edge_cents": round(edge, 1),
                "side": side,
                "result": result,
                "correct": correct,
            })

    # Score by stdev
    scores = {}
    for stdev_label in ["tight_1.5", "medium_3.0", "loose_4.5"]:
        subset = [p for p in predictions if p["stdev_label"] == stdev_label and p["correct"] is not None]
        if not subset:
            continue
        n = len(subset)
        wins = sum(1 for p in subset if p["correct"])
        avg_edge = sum(p["edge_cents"] for p in subset) / n
        scores[stdev_label] = {
            "trades": n, "wins": wins,
            "win_rate": round(wins / n, 3) if n > 0 else 0,
            "avg_edge": round(avg_edge, 1),
        }

    return {
        "total_markets": len(weather),
        "temps_found": len(temp_cache),
        "predictions": len(predictions),
        "scores_by_stdev": scores,
    }


def backtest_crypto(markets: list) -> dict:
    """Backtest crypto log-normal model against settled crypto markets."""
    crypto = [m for m in markets if m["category"] == "crypto" and m.get("result")]
    logger.info(f"Crypto: {len(crypto)} settled markets to backtest")

    if not crypto:
        return {"total": 0, "message": "No settled crypto markets found"}

    BUCKET_WIDTHS = {"BTC": 500, "ETH": 20, "SOL": 2}
    predictions = []

    # Group by settlement time to batch price fetches
    by_date = defaultdict(list)
    for mkt in crypto:
        close_time = mkt.get("close_time", "")
        if close_time:
            by_date[close_time[:10]].append(mkt)

    # Fetch historical prices
    price_cache = {}
    for date_str, mkts in sorted(by_date.items()):
        for asset in ["BTC", "ETH", "SOL"]:
            key = f"{asset}_{date_str}"
            if key in price_cache:
                continue
            asset_mkts = [m for m in mkts if m["sub_category"] == asset]
            if not asset_mkts:
                continue
            price = fetch_historical_crypto_price(asset, f"{date_str}T12:00:00Z")
            if price:
                price_cache[key] = price
                logger.debug(f"  {asset} {date_str}: ${price:,.2f}")
            time.sleep(1.1)  # CoinGecko rate limit

    logger.info(f"  Fetched {len(price_cache)} historical prices")

    # Backtest each market
    for mkt in crypto:
        asset = mkt["sub_category"]
        bucket = mkt.get("bucket")
        result = mkt["result"]
        last_price = mkt.get("last_price") or 50
        close_time = mkt.get("close_time", "")

        if not bucket or not close_time:
            continue

        date_str = close_time[:10]
        spot = price_cache.get(f"{asset}_{date_str}")
        if not spot:
            continue

        bucket_width = BUCKET_WIDTHS.get(asset, 500)

        # Test with different vol assumptions
        for vol_label, vol in [("low_0.3", 0.30), ("med_0.5", 0.50), ("high_0.8", 0.80)]:
            # Assume we're looking ~4 hours before settlement
            hours = 4.0

            if bucket["type"] == "bucket":
                fair = lognormal_bucket_prob(spot, bucket["low"], bucket["high"], vol, hours)
            elif bucket["type"] == "above":
                sigma = vol * math.sqrt(hours / (365 * 24))
                if sigma > 0.001 and bucket["low"] > 0 and spot > 0:
                    fair = 1.0 - normal_cdf(math.log(bucket["low"] / spot) / sigma)
                else:
                    fair = 1.0 if spot >= bucket["low"] else 0.0
            elif bucket["type"] == "below":
                sigma = vol * math.sqrt(hours / (365 * 24))
                if sigma > 0.001 and bucket["high"] > 0 and spot > 0:
                    fair = normal_cdf(math.log(bucket["high"] / spot) / sigma)
                else:
                    fair = 1.0 if spot <= bucket["high"] else 0.0
            else:
                continue

            fair_cents = fair * 100
            market_cents = last_price
            edge = abs(fair_cents - market_cents)

            if edge < 3:
                side = "no_trade"
                correct = None
            elif fair_cents > market_cents:
                side = "buy_yes"
                correct = (result == "yes")
            else:
                side = "buy_no"
                correct = (result == "no")

            # Calculate P&L per contract
            if correct is True:
                pnl = (100 - market_cents) if side == "buy_yes" else market_cents
            elif correct is False:
                pnl = -market_cents if side == "buy_yes" else -(100 - market_cents)
            else:
                pnl = 0

            predictions.append({
                "ticker": mkt["ticker"],
                "asset": asset,
                "date": date_str,
                "vol_label": vol_label,
                "vol": vol,
                "spot": spot,
                "bucket": bucket,
                "bucket_width": bucket_width,
                "fair_cents": round(fair_cents, 1),
                "market_cents": market_cents,
                "edge_cents": round(edge, 1),
                "side": side,
                "result": result,
                "correct": correct,
                "pnl_cents": round(pnl, 1),
            })

    # Score by vol + asset
    scores = {}
    for asset in ["BTC", "ETH", "SOL"]:
        for vol_label in ["low_0.3", "med_0.5", "high_0.8"]:
            subset = [p for p in predictions
                     if p["asset"] == asset and p["vol_label"] == vol_label and p["correct"] is not None]
            if not subset:
                continue
            n = len(subset)
            wins = sum(1 for p in subset if p["correct"])
            total_pnl = sum(p["pnl_cents"] for p in subset)
            key = f"{asset}/{vol_label}"
            scores[key] = {
                "trades": n, "wins": wins,
                "win_rate": round(wins / n, 3) if n > 0 else 0,
                "total_pnl_cents": round(total_pnl, 1),
                "avg_pnl_cents": round(total_pnl / n, 1) if n > 0 else 0,
            }

    return {
        "total_markets": len(crypto),
        "prices_found": len(price_cache),
        "predictions": len(predictions),
        "scores_by_vol_asset": scores,
    }


def backtest_sports(markets: list) -> dict:
    """Analyze sports market patterns from settlement data."""
    sports = [m for m in markets if m["category"] == "sports" and m.get("result")]
    logger.info(f"Sports: {len(sports)} settled markets")

    if not sports:
        return {"total": 0, "message": "No settled sports markets found"}

    # Analyze: what % of markets settle YES vs NO?
    # And what's the average last_price for YES vs NO settlements?
    yes_settled = [m for m in sports if m["result"] == "yes"]
    no_settled = [m for m in sports if m["result"] == "no"]

    # Price distribution of YES settlements
    yes_prices = [m.get("last_price", 50) for m in yes_settled if m.get("last_price")]
    no_prices = [m.get("last_price", 50) for m in no_settled if m.get("last_price")]

    # Underdog analysis: markets where last_price was low but settled YES
    upsets = [m for m in yes_settled if (m.get("last_price") or 50) < 30]

    return {
        "total_markets": len(sports),
        "yes_settlements": len(yes_settled),
        "no_settlements": len(no_settled),
        "yes_pct": round(len(yes_settled) / len(sports) * 100, 1) if sports else 0,
        "avg_yes_price": round(sum(yes_prices) / len(yes_prices), 1) if yes_prices else 0,
        "avg_no_price": round(sum(no_prices) / len(no_prices), 1) if no_prices else 0,
        "upsets_under_30c": len(upsets),
        "upset_rate": round(len(upsets) / len(sports) * 100, 1) if sports else 0,
    }


# ---------------------------------------------------------------------------
# STEP 5: Save scenarios for AutoResearch
# ---------------------------------------------------------------------------

def save_autoresearch_scenarios(markets: list, results: dict):
    """Convert historical settlements into scenarios AutoResearch can use."""
    scenarios = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "weather_scenarios": [],
        "crypto_scenarios": [],
        "sports_scenarios": [],
    }

    # Weather scenarios: each settled market with actual temp
    weather = [m for m in markets if m["category"] == "weather" and m.get("result")]
    for mkt in weather:
        bucket = mkt.get("bucket")
        date = extract_date_from_ticker(mkt["ticker"])
        if not bucket or not date:
            continue

        scenarios["weather_scenarios"].append({
            "ticker": mkt["ticker"],
            "city": mkt.get("sub_category", ""),
            "date": date,
            "bucket": bucket,
            "result": mkt["result"],
            "last_price": mkt.get("last_price"),
            "volume": mkt.get("volume", 0),
        })

    # Crypto scenarios
    crypto = [m for m in markets if m["category"] == "crypto" and m.get("result")]
    for mkt in crypto:
        bucket = mkt.get("bucket")
        if not bucket:
            continue

        scenarios["crypto_scenarios"].append({
            "ticker": mkt["ticker"],
            "asset": mkt.get("sub_category", ""),
            "close_time": mkt.get("close_time", ""),
            "bucket": bucket,
            "result": mkt["result"],
            "last_price": mkt.get("last_price"),
            "volume": mkt.get("volume", 0),
        })

    # Sports scenarios
    sports = [m for m in markets if m["category"] == "sports" and m.get("result")]
    for mkt in sports:
        scenarios["sports_scenarios"].append({
            "ticker": mkt["ticker"],
            "title": mkt["title"],
            "result": mkt["result"],
            "last_price": mkt.get("last_price"),
            "volume": mkt.get("volume", 0),
        })

    with open(AUTORESEARCH_SCENARIOS_FILE, "w") as f:
        json.dump(scenarios, f, indent=2)

    logger.info(f"Saved {len(scenarios['weather_scenarios'])} weather, "
                f"{len(scenarios['crypto_scenarios'])} crypto, "
                f"{len(scenarios['sports_scenarios'])} sports scenarios for AutoResearch")


# ---------------------------------------------------------------------------
# STEP 6: Report
# ---------------------------------------------------------------------------

def print_report(results: dict):
    """Print comprehensive backtest report."""
    print()
    print("=" * 80)
    print("HISTORICAL BACKTEST REPORT")
    print("=" * 80)

    # Weather
    w = results.get("weather", {})
    print(f"\n--- WEATHER ({w.get('total_markets', 0)} settled markets, "
          f"{w.get('temps_found', 0)} temps fetched) ---")
    scores = w.get("scores_by_stdev", {})
    if scores:
        print(f"  {'Stdev':>12s}  {'Trades':>6s}  {'Wins':>5s}  {'Rate':>6s}  {'Avg Edge':>8s}")
        for label, sc in sorted(scores.items()):
            print(f"  {label:>12s}  {sc['trades']:>6d}  {sc['wins']:>5d}  "
                  f"{sc['win_rate']:>5.0%}  {sc['avg_edge']:>+7.1f}c")
        # Best stdev
        best = max(scores.items(), key=lambda x: x[1]["win_rate"])
        print(f"  BEST: {best[0]} ({best[1]['win_rate']:.0%} win rate)")
    else:
        print("  No backtest data available")

    # Crypto
    c = results.get("crypto", {})
    print(f"\n--- CRYPTO ({c.get('total_markets', 0)} settled markets, "
          f"{c.get('prices_found', 0)} prices fetched) ---")
    scores = c.get("scores_by_vol_asset", {})
    if scores:
        print(f"  {'Asset/Vol':>15s}  {'Trades':>6s}  {'Wins':>5s}  {'Rate':>6s}  "
              f"{'Total P&L':>10s}  {'Avg P&L':>8s}")
        for key, sc in sorted(scores.items()):
            print(f"  {key:>15s}  {sc['trades']:>6d}  {sc['wins']:>5d}  "
                  f"{sc['win_rate']:>5.0%}  {sc['total_pnl_cents']:>+9.1f}c  "
                  f"{sc['avg_pnl_cents']:>+7.1f}c")
        # Best combo
        profitable = {k: v for k, v in scores.items() if v["avg_pnl_cents"] > 0}
        if profitable:
            best = max(profitable.items(), key=lambda x: x[1]["avg_pnl_cents"])
            print(f"  BEST: {best[0]} ({best[1]['win_rate']:.0%} win rate, "
                  f"+{best[1]['avg_pnl_cents']:.1f}c/trade)")
        else:
            print("  NO profitable configuration found")
    else:
        print("  No backtest data available")

    # Sports
    s = results.get("sports", {})
    print(f"\n--- SPORTS ({s.get('total_markets', 0)} settled markets) ---")
    print(f"  YES settlements: {s.get('yes_settlements', 0)} ({s.get('yes_pct', 0):.1f}%)")
    print(f"  NO settlements:  {s.get('no_settlements', 0)}")
    print(f"  Avg price when YES won: {s.get('avg_yes_price', 0):.1f}c")
    print(f"  Avg price when NO won:  {s.get('avg_no_price', 0):.1f}c")
    print(f"  Upsets (won at <30c):   {s.get('upsets_under_30c', 0)} "
          f"({s.get('upset_rate', 0):.1f}% of all markets)")

    print()
    print(f"Results saved to {BACKTEST_RESULTS_FILE}")
    print(f"AutoResearch scenarios saved to {AUTORESEARCH_SCENARIOS_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python historical_backtest.py [fetch|backtest|full]")
        print("  fetch     — Pull all settled markets from Kalshi API")
        print("  backtest  — Run models against historical data")
        print("  full      — fetch + backtest (complete run)")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "fetch":
        fetch_all_settled_markets()
    elif cmd == "backtest":
        backtest_all()
    elif cmd == "full":
        fetch_all_settled_markets()
        backtest_all()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
