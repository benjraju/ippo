"""
strategy_tester.py — Comprehensive strategy testing engine.

Scans ALL Kalshi markets, runs every model, logs predictions, grades
against real settlements, and reports which strategies actually have edge.

Usage:
    python strategy_tester.py scan      # Log predictions for all open markets
    python strategy_tester.py grade     # Grade settled predictions against outcomes
    python strategy_tester.py report    # Print accuracy scorecard
    python strategy_tester.py run       # scan + grade + report (full cycle)

Run every hour via systemd timer. After 14 days you'll know which
strategies are real and which are noise.
"""

import json
import logging
import math
import sys
import time
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import Optional

import config
from kalshi_client import KalshiClient

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PREDICTIONS_FILE = config.OUTPUT_DIR / "strategy_predictions.jsonl"
SCORECARD_FILE = config.OUTPUT_DIR / "strategy_scorecard.json"
LOG_FILE = config.OUTPUT_DIR / "strategy_tester.log"

# How often to re-scan (don't log duplicate predictions within this window)
DEDUP_WINDOW_HOURS = 2

# Minimum edge to log a prediction (cents)
MIN_EDGE_TO_LOG = 3.0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger("strategy_tester")

# ---------------------------------------------------------------------------
# Prediction record
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    timestamp: str
    ticker: str
    title: str
    strategy: str          # weather, crypto, sports, arb
    sub_strategy: str      # city name, asset, market type
    side: str              # buy_yes, buy_no
    market_price_cents: float
    model_fair_value_cents: float
    edge_cents: float
    model_details: dict    # forecast_temp, volatility, etc.
    # Filled in by grading
    settled: bool = False
    settlement_result: str = ""  # yes, no
    pnl_per_contract_cents: float = 0.0
    correct: bool = False


def append_prediction(pred: Prediction):
    with open(PREDICTIONS_FILE, "a") as f:
        f.write(json.dumps(asdict(pred)) + "\n")


def load_predictions() -> list[dict]:
    if not PREDICTIONS_FILE.exists():
        return []
    preds = []
    for line in PREDICTIONS_FILE.read_text().strip().split("\n"):
        if line.strip():
            try:
                preds.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return preds


def save_predictions(preds: list[dict]):
    with open(PREDICTIONS_FILE, "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")


def get_recent_tickers(preds: list[dict], hours: int = DEDUP_WINDOW_HOURS) -> set:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    return {p["ticker"] for p in preds if p.get("timestamp", "") > cutoff}


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def lognormal_bucket_prob(spot: float, low: float, high: float, vol: float, hours: float) -> float:
    if hours <= 0:
        hours = 0.1
    t = hours / (365 * 24)
    sigma = vol * math.sqrt(t)
    if sigma < 0.001:
        return 1.0 if low <= spot <= high else 0.0
    z_high = math.log(high / spot) / sigma
    z_low = math.log(low / spot) / sigma
    return max(0.0, normal_cdf(z_high) - normal_cdf(z_low))


def normal_bucket_prob(forecast: float, low: float, high: float, stdev: float) -> float:
    if stdev < 0.01:
        return 1.0 if low <= forecast <= high else 0.0
    z_high = (high - forecast) / stdev
    z_low = (low - forecast) / stdev
    return max(0.0, normal_cdf(z_high) - normal_cdf(z_low))


def binomial_p_value(n: int, k: int, p0: float) -> float:
    """One-sided binomial test: P(X >= k) under null H0: p = p0."""
    if n == 0:
        return 1.0
    # Normal approximation for large n
    mu = n * p0
    sigma = math.sqrt(n * p0 * (1 - p0))
    if sigma < 0.01:
        return 0.0 if k > mu else 1.0
    z = (k - 0.5 - mu) / sigma
    return 1 - normal_cdf(z)


# ---------------------------------------------------------------------------
# Market parsing helpers
# ---------------------------------------------------------------------------

def parse_bucket_from_title(title: str) -> Optional[dict]:
    """Parse exact bucket bounds from market title."""
    if not title:
        return None

    def clean_num(s):
        return float(s.replace(",", "").replace("$", "").strip())

    # "$X to $Y" or "$X to Y"
    m = re.search(r'\$([\d,.]+)\s+to\s+\$?([\d,.]+)', title)
    if m:
        return {"type": "bucket", "low": clean_num(m.group(1)), "high": clean_num(m.group(2))}

    # "between $X and $Y"
    m = re.search(r'between\s+\$?([\d,.]+)\s+and\s+\$?([\d,.]+)', title, re.I)
    if m:
        return {"type": "bucket", "low": clean_num(m.group(1)), "high": clean_num(m.group(2))}

    # "above $X" / "$X or above"
    m = re.search(r'(?:above|or above|or more|at least)\s+\$?([\d,.]+)', title, re.I)
    if m:
        return {"type": "above", "low": clean_num(m.group(1)), "high": clean_num(m.group(1)) * 10}

    # "below $X" / "under $X"
    m = re.search(r'(?:below|under|or below|less than)\s+\$?([\d,.]+)', title, re.I)
    if m:
        return {"type": "below", "low": 0, "high": clean_num(m.group(1))}

    # Weather: "be X-Y" degrees
    m = re.search(r'be\s+([\d.]+)-([\d.]+)', title)
    if m:
        return {"type": "bucket", "low": float(m.group(1)), "high": float(m.group(2))}

    # Weather: ">X" or "<X" degrees
    m = re.search(r'be\s*[>≥]\s*([\d.]+)', title)
    if m:
        return {"type": "above", "low": float(m.group(1)), "high": float(m.group(1)) + 50}
    m = re.search(r'be\s*[<≤]\s*([\d.]+)', title)
    if m:
        return {"type": "below", "low": float(m.group(1)) - 50, "high": float(m.group(1))}

    return None


def classify_market(ticker: str, title: str) -> tuple[str, str]:
    """Returns (strategy, sub_strategy) for a market."""
    t = ticker.upper()
    tl = title.lower()

    if "KXHIGH" in t:
        city = "unknown"
        for c in ["NY", "CHI", "MIA", "LA", "DC", "DEN"]:
            if c in t:
                city = c
                break
        return "weather", city

    if "KXBTC" in t:
        return "crypto", "BTC"
    if "KXETH" in t:
        return "crypto", "ETH"
    if "KXSOL" in t:
        return "crypto", "SOL"

    if "KXNBAGAME" in t:
        return "sports", "nba_winner"
    if "KXNBAPTS" in t:
        return "sports", "nba_props"
    if any(x in t for x in ["KXNHL", "KXMLB", "KXNCAA"]):
        return "sports", "other_sports"

    return "other", "unknown"


def hours_to_settlement(market: dict) -> float:
    """Calculate hours until market settles."""
    close_time = market.get("close_time") or market.get("expected_expiration_time", "")
    if not close_time:
        return 24.0
    try:
        close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
        delta = close_dt - datetime.now(timezone.utc)
        return max(0.1, delta.total_seconds() / 3600)
    except (ValueError, TypeError):
        return 24.0


# ---------------------------------------------------------------------------
# WEATHER MODEL
# ---------------------------------------------------------------------------

def fetch_nws_forecast_simple(city: str) -> Optional[dict]:
    """Fetch NWS forecast. Returns {date_str: {"high": temp_f, "stdev": uncertainty}}."""
    import requests

    grid_points = {
        "NY": ("OKX", 33, 37),
        "CHI": ("LOT", 76, 73),
        "MIA": ("MFL", 75, 67),
        "LA": ("LOX", 154, 44),
        "DC": ("LWX", 97, 71),
        "DEN": ("BOU", 62, 60),
    }
    if city not in grid_points:
        return None

    office, x, y = grid_points[city]
    try:
        from autoresearch.candidate_strategy import get_forecast_stdev
        stdevs = get_forecast_stdev()
    except Exception:
        stdevs = {0: 1.5, 1: 2.5, 2: 3.5, 3: 4.5}

    try:
        url = f"https://api.weather.gov/gridpoints/{office}/{x},{y}/forecast"
        resp = requests.get(url, headers={"User-Agent": "ippo-strategy-tester"}, timeout=10)
        data = resp.json()
        forecasts = {}
        now = datetime.now(timezone.utc)
        for period in data.get("properties", {}).get("periods", []):
            if not period.get("isDaytime", False):
                continue
            try:
                start = datetime.fromisoformat(period["startTime"])
                days_out = (start.date() - now.date()).days
                if 0 <= days_out <= 3:
                    date_str = start.strftime("%Y-%m-%d")
                    forecasts[date_str] = {
                        "high": period["temperature"],
                        "stdev": stdevs.get(days_out, 4.5),
                        "days_out": days_out,
                    }
            except (KeyError, ValueError):
                continue
        return forecasts if forecasts else None
    except Exception as e:
        logger.debug(f"NWS fetch failed for {city}: {e}")
        return None


def scan_weather(client: KalshiClient, recent_tickers: set) -> list[Prediction]:
    """Scan all weather markets and generate predictions."""
    preds = []
    cities = {"NY": "KXHIGHNY", "CHI": "KXHIGHCHI", "MIA": "KXHIGHMIA",
              "LA": "KXHIGHLA", "DC": "KXHIGHDC", "DEN": "KXHIGHDEN"}

    for city, series in cities.items():
        forecast = fetch_nws_forecast_simple(city)
        if not forecast:
            logger.debug(f"No forecast for {city}")
            continue

        try:
            resp = client.get_markets(series_ticker=series, limit=100, status="open")
            markets = resp.get("markets", [])
        except Exception as e:
            logger.warning(f"Failed to fetch {series}: {e}")
            continue

        for mkt in markets:
            ticker = mkt.get("ticker", "")
            if ticker in recent_tickers:
                continue

            title = mkt.get("title", "")
            yes_price = mkt.get("yes_ask", 0) or 0
            if yes_price == 0:
                yes_price = mkt.get("last_price", 50) or 50

            # Parse bucket from title
            bucket = parse_bucket_from_title(title)
            if not bucket:
                continue

            # Find matching forecast date
            settle_date = None
            for date_str in forecast:
                # Try to match date in ticker
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                    ticker_date_patterns = [
                        dt.strftime("%y%b%d").upper(),  # 26MAR21
                        dt.strftime("%d%b%y").upper(),  # 21MAR26
                    ]
                    if any(p in ticker.upper() for p in ticker_date_patterns):
                        settle_date = date_str
                        break
                except ValueError:
                    continue

            if not settle_date or settle_date not in forecast:
                # Try first available forecast
                settle_date = min(forecast.keys())

            fc = forecast[settle_date]
            temp = fc["high"]
            stdev = fc["stdev"]

            # Calculate fair value
            if bucket["type"] == "bucket":
                fair = normal_bucket_prob(temp, bucket["low"], bucket["high"], stdev)
            elif bucket["type"] == "above":
                fair = 1.0 - normal_cdf((bucket["low"] - temp) / max(stdev, 0.01))
            elif bucket["type"] == "below":
                fair = normal_cdf((bucket["high"] - temp) / max(stdev, 0.01))
            else:
                continue

            fair_cents = fair * 100
            market_cents = yes_price

            # Determine side and edge
            if fair_cents > market_cents + MIN_EDGE_TO_LOG:
                side = "buy_yes"
                edge = fair_cents - market_cents
            elif fair_cents < market_cents - MIN_EDGE_TO_LOG:
                side = "buy_no"
                edge = market_cents - fair_cents
            else:
                continue

            preds.append(Prediction(
                timestamp=datetime.now(timezone.utc).isoformat(),
                ticker=ticker,
                title=title,
                strategy="weather",
                sub_strategy=city,
                side=side,
                market_price_cents=market_cents,
                model_fair_value_cents=round(fair_cents, 1),
                edge_cents=round(edge, 1),
                model_details={
                    "forecast_temp": temp,
                    "stdev": stdev,
                    "settle_date": settle_date,
                    "days_out": fc["days_out"],
                    "bucket_low": bucket["low"],
                    "bucket_high": bucket["high"],
                    "bucket_type": bucket["type"],
                },
            ))

    logger.info(f"Weather: {len(preds)} predictions across {len(cities)} cities")
    return preds


# ---------------------------------------------------------------------------
# CRYPTO MODEL
# ---------------------------------------------------------------------------

def fetch_crypto_price(asset: str) -> Optional[float]:
    import requests
    try:
        r = requests.get(f"https://api.coinbase.com/v2/prices/{asset}-USD/spot", timeout=5)
        return float(r.json()["data"]["amount"])
    except Exception:
        pass
    cg_map = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana"}
    try:
        r = requests.get(f"https://api.coingecko.com/api/v3/simple/price",
                        params={"ids": cg_map.get(asset, asset.lower()), "vs_currencies": "usd"}, timeout=5)
        return list(r.json().values())[0]["usd"]
    except Exception:
        return None


def fetch_crypto_vol(asset: str) -> float:
    import requests
    cg_map = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana"}
    try:
        r = requests.get(f"https://api.coingecko.com/api/v3/coins/{cg_map.get(asset, asset.lower())}/market_chart",
                        params={"vs_currency": "usd", "days": "30"}, timeout=10)
        prices = [p[1] for p in r.json()["prices"]]
        returns = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]
        daily_vol = (sum(r ** 2 for r in returns) / len(returns)) ** 0.5
        return daily_vol * math.sqrt(365)
    except Exception:
        return {"BTC": 0.50, "ETH": 0.70, "SOL": 0.80}.get(asset, 0.60)


CRYPTO_BUCKET_WIDTHS = {"BTC": 500, "ETH": 20, "SOL": 2}
CRYPTO_SERIES = {"BTC": "KXBTC", "ETH": "KXETH", "SOL": "KXSOL"}


def scan_crypto(client: KalshiClient, recent_tickers: set) -> list[Prediction]:
    """Scan all crypto markets and generate predictions."""
    preds = []

    for asset in ["BTC", "ETH", "SOL"]:
        spot = fetch_crypto_price(asset)
        if not spot:
            logger.debug(f"No price for {asset}")
            continue

        vol = fetch_crypto_vol(asset)
        series = CRYPTO_SERIES[asset]
        bucket_width = CRYPTO_BUCKET_WIDTHS[asset]

        try:
            resp = client.get_markets(series_ticker=series, limit=100, status="open")
            markets = resp.get("markets", [])
        except Exception as e:
            logger.warning(f"Failed to fetch {series}: {e}")
            continue

        for mkt in markets:
            ticker = mkt.get("ticker", "")
            if ticker in recent_tickers:
                continue

            title = mkt.get("title", "")
            yes_price = mkt.get("yes_ask", 0) or mkt.get("last_price", 50) or 50
            hrs = hours_to_settlement(mkt)

            # Parse bucket — prefer title, fall back to ticker
            bucket = parse_bucket_from_title(title)
            if not bucket and "-B" in ticker:
                try:
                    center = float(ticker.split("-B")[-1])
                    half = bucket_width / 2
                    bucket = {"type": "bucket", "low": center - half, "high": center + half}
                except (ValueError, IndexError):
                    continue

            if not bucket:
                continue

            # Calculate fair value
            if bucket["type"] == "bucket":
                fair = lognormal_bucket_prob(spot, bucket["low"], bucket["high"], vol, hrs)
            elif bucket["type"] == "above":
                fair = 1.0 - normal_cdf(math.log(bucket["low"] / spot) / (vol * math.sqrt(max(hrs, 0.1) / (365 * 24))))
            elif bucket["type"] == "below":
                fair = normal_cdf(math.log(bucket["high"] / spot) / (vol * math.sqrt(max(hrs, 0.1) / (365 * 24))))
            else:
                continue

            fair_cents = fair * 100
            market_cents = yes_price

            if fair_cents > market_cents + MIN_EDGE_TO_LOG:
                side = "buy_yes"
                edge = fair_cents - market_cents
            elif fair_cents < market_cents - MIN_EDGE_TO_LOG:
                side = "buy_no"
                edge = market_cents - fair_cents
            else:
                continue

            preds.append(Prediction(
                timestamp=datetime.now(timezone.utc).isoformat(),
                ticker=ticker,
                title=title,
                strategy="crypto",
                sub_strategy=asset,
                side=side,
                market_price_cents=market_cents,
                model_fair_value_cents=round(fair_cents, 1),
                edge_cents=round(edge, 1),
                model_details={
                    "spot": spot,
                    "vol": round(vol, 4),
                    "hours_to_settle": round(hrs, 1),
                    "bucket_low": bucket["low"],
                    "bucket_high": bucket["high"],
                    "bucket_type": bucket["type"],
                    "bucket_width": bucket_width,
                },
            ))

    logger.info(f"Crypto: {len(preds)} predictions across BTC/ETH/SOL")
    return preds


# ---------------------------------------------------------------------------
# ARB MODEL
# ---------------------------------------------------------------------------

def scan_arb(client: KalshiClient, recent_tickers: set) -> list[Prediction]:
    """Scan all markets for YES+NO arbitrage opportunities."""
    preds = []
    fee_per_side = config.KALSHI_FEE_PER_SIDE_CENTS

    all_series = config.TARGET_MARKET_SERIES
    seen = set()

    for series in all_series:
        try:
            resp = client.get_markets(series_ticker=series, limit=100, status="open")
            markets = resp.get("markets", [])
        except Exception:
            continue

        for mkt in markets:
            ticker = mkt.get("ticker", "")
            if ticker in seen or ticker in recent_tickers:
                continue
            seen.add(ticker)

            try:
                book = client.get_market_orderbook(ticker, depth=1)
            except Exception:
                continue

            yes_ask = None
            no_ask = None

            yes_orders = book.get("orderbook", {}).get("yes", [])
            no_orders = book.get("orderbook", {}).get("no", [])

            if yes_orders and len(yes_orders) > 0:
                # Asks are lowest price to buy
                if isinstance(yes_orders[0], list):
                    yes_ask = yes_orders[0][0]
                elif isinstance(yes_orders[0], (int, float)):
                    yes_ask = yes_orders[0]

            if no_orders and len(no_orders) > 0:
                if isinstance(no_orders[0], list):
                    no_ask = no_orders[0][0]
                elif isinstance(no_orders[0], (int, float)):
                    no_ask = no_orders[0]

            if yes_ask is None or no_ask is None:
                continue

            total_cost = yes_ask + no_ask
            total_fees = 2 * fee_per_side * 2  # fees on both legs, both sides
            edge_net = 100 - total_cost - total_fees

            if edge_net > 0:
                preds.append(Prediction(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    ticker=ticker,
                    title=mkt.get("title", ""),
                    strategy="arb",
                    sub_strategy="yes_no",
                    side="buy_both",
                    market_price_cents=total_cost,
                    model_fair_value_cents=100.0,
                    edge_cents=round(edge_net, 1),
                    model_details={
                        "yes_ask": yes_ask,
                        "no_ask": no_ask,
                        "total_cost": total_cost,
                        "total_fees": total_fees,
                        "net_profit_cents": round(edge_net, 1),
                    },
                ))

            time.sleep(0.15)  # Rate limit orderbook calls

    logger.info(f"Arb: {len(preds)} opportunities found")
    return preds


# ---------------------------------------------------------------------------
# SPORTS MODEL (NBA underdogs)
# ---------------------------------------------------------------------------

def scan_sports(client: KalshiClient, recent_tickers: set) -> list[Prediction]:
    """Scan NBA game winner markets for underdog opportunities."""
    preds = []

    try:
        from sports_strategy import find_nba_edges
        edges = find_nba_edges(client)
    except Exception as e:
        logger.warning(f"Sports scan failed: {e}")
        return preds

    for edge in edges:
        ticker = edge.ticker
        if ticker in recent_tickers:
            continue

        market_price = edge.market_price * 100 if edge.market_price < 1 else edge.market_price
        fair_value = edge.fair_value * 100 if edge.fair_value < 1 else edge.fair_value
        edge_cents = abs(fair_value - market_price)

        # Log ALL sports edges for testing, not just underdogs
        if edge_cents < MIN_EDGE_TO_LOG:
            continue

        title = getattr(edge, 'title', '') or ticker
        is_winner = 'winner' in title.lower() or 'KXNBAGAME' in ticker.upper()
        is_underdog = market_price < 30

        preds.append(Prediction(
            timestamp=datetime.now(timezone.utc).isoformat(),
            ticker=ticker,
            title=title,
            strategy="sports",
            sub_strategy="nba_underdog" if (is_winner and is_underdog) else "nba_other",
            side=edge.side if hasattr(edge, 'side') else "buy_yes",
            market_price_cents=round(market_price, 1),
            model_fair_value_cents=round(fair_value, 1),
            edge_cents=round(edge_cents, 1),
            model_details={
                "is_winner_market": is_winner,
                "is_underdog": is_underdog,
                "model_spread": getattr(edge, 'model_spread', 0),
                "confidence": getattr(edge, 'confidence', 'unknown'),
            },
        ))

    logger.info(f"Sports: {len(preds)} predictions")
    return preds


# ---------------------------------------------------------------------------
# NBA PROPS MODEL
# ---------------------------------------------------------------------------

def parse_nba_prop(title: str) -> Optional[dict]:
    """Parse player name and threshold from NBA prop title.

    Examples:
        "Jalen Brunson: 20+ points" → {"player": "Jalen Brunson", "threshold": 20}
        "LeBron James Over 25.5 Points" → {"player": "LeBron James", "threshold": 25.5}
        "Stephen Curry: 30+ points" → {"player": "Stephen Curry", "threshold": 30}
    """
    if not title:
        return None

    # Pattern: "Player Name: X+ points"
    m = re.search(r'(.+?):\s*([\d.]+)\+?\s*points', title, re.I)
    if m:
        return {"player": m.group(1).strip(), "threshold": float(m.group(2))}

    # Pattern: "Player Name Over X.X Points"
    m = re.search(r'(.+?)\s+(?:over|under)\s+([\d.]+)\s*points', title, re.I)
    if m:
        return {"player": m.group(1).strip(), "threshold": float(m.group(2))}

    # Pattern: "Player Name X+ pts"
    m = re.search(r'(.+?)[\s:]+(\d+)\+?\s*(?:pts|points)', title, re.I)
    if m:
        return {"player": m.group(1).strip(), "threshold": float(m.group(2))}

    return None


def scan_nba_props(client: KalshiClient, recent_tickers: set) -> list[Prediction]:
    """Scan KXNBAPTS series for open prop markets and log predictions."""
    preds = []

    try:
        resp = client.get_markets(series_ticker="KXNBAPTS", limit=100, status="open")
        markets = resp.get("markets", [])
    except Exception as e:
        logger.warning(f"Failed to fetch KXNBAPTS: {e}")
        return preds

    for mkt in markets:
        ticker = mkt.get("ticker", "")
        if ticker in recent_tickers:
            continue

        title = mkt.get("title", "")
        yes_ask = mkt.get("yes_ask", 0) or 0
        yes_bid = mkt.get("yes_bid", 0) or 0
        volume = mkt.get("volume", 0) or 0
        event_ticker = mkt.get("event_ticker", "")

        if yes_ask == 0:
            yes_ask = mkt.get("last_price", 50) or 50

        # Parse player + threshold from title
        prop = parse_nba_prop(title)
        player = prop["player"] if prop else "unknown"
        threshold = prop["threshold"] if prop else 0

        sub_strat = f"{player}_{int(threshold)}" if prop else "unknown"

        # Simple edge heuristic: extreme prices suggest mispricing
        if yes_ask < 30:
            side = "buy_yes"
            edge = 30 - yes_ask  # underdog prop bet
            fair_cents = 30.0  # assume at least 30c fair for any listed prop
        elif yes_ask > 70:
            side = "buy_no"
            edge = yes_ask - 70  # overpriced prop
            fair_cents = 70.0
        else:
            continue  # no strong signal in the middle range

        preds.append(Prediction(
            timestamp=datetime.now(timezone.utc).isoformat(),
            ticker=ticker,
            title=title,
            strategy="nba_props",
            sub_strategy=sub_strat,
            side=side,
            market_price_cents=yes_ask,
            model_fair_value_cents=fair_cents,
            edge_cents=round(edge, 1),
            model_details={
                "player": player,
                "threshold": threshold,
                "yes_ask": yes_ask,
                "yes_bid": yes_bid,
                "volume": volume,
                "event_ticker": event_ticker,
            },
        ))

    logger.info(f"NBA Props: {len(preds)} predictions")
    return preds


# ---------------------------------------------------------------------------
# CRYPTO MOMENTUM MODEL
# ---------------------------------------------------------------------------

def scan_crypto_momentum(client: KalshiClient, recent_tickers: set) -> list[Prediction]:
    """Scan KXBTC/KXETH settled + open markets for momentum signals.

    Strategy: find which bucket won in the most recent settled event,
    determine if it was above or below the range midpoint, then predict
    the same direction continues in the next open event.
    """
    preds = []

    for asset, series in [("BTC", "KXBTC"), ("ETH", "KXETH")]:
        try:
            # Get settled events to find recent winners
            settled_resp = client.get_markets(series_ticker=series, limit=50, status="settled")
            settled_markets = settled_resp.get("markets", [])

            # Get open events to place predictions on
            open_resp = client.get_markets(series_ticker=series, limit=50, status="open")
            open_markets = open_resp.get("markets", [])
        except Exception as e:
            logger.warning(f"Crypto momentum fetch failed for {series}: {e}")
            continue

        if not settled_markets or not open_markets:
            continue

        # Group settled markets by event_ticker
        settled_by_event = defaultdict(list)
        for mkt in settled_markets:
            et = mkt.get("event_ticker", "")
            if et:
                settled_by_event[et].append(mkt)

        # Find the most recently settled event (by close_time)
        latest_event = None
        latest_close = ""
        for et, mkts in settled_by_event.items():
            for m in mkts:
                ct = m.get("close_time", "") or m.get("expected_expiration_time", "")
                if ct > latest_close:
                    latest_close = ct
                    latest_event = et

        if not latest_event:
            continue

        # Find the winner in the settled event
        winner_market = None
        all_buckets = []
        for mkt in settled_by_event[latest_event]:
            bucket = parse_bucket_from_title(mkt.get("title", ""))
            if bucket:
                all_buckets.append(bucket)
            if mkt.get("result") == "yes":
                winner_market = mkt

        if not winner_market or not all_buckets:
            continue

        # Determine range midpoint across all buckets in that event
        all_lows = [b["low"] for b in all_buckets if b.get("type") == "bucket"]
        all_highs = [b["high"] for b in all_buckets if b.get("type") == "bucket"]
        if not all_lows or not all_highs:
            continue

        range_low = min(all_lows)
        range_high = max(all_highs)
        range_mid = (range_low + range_high) / 2

        # Where did the winner land?
        winner_bucket = parse_bucket_from_title(winner_market.get("title", ""))
        if not winner_bucket:
            continue

        if winner_bucket.get("type") == "bucket":
            winner_center = (winner_bucket["low"] + winner_bucket["high"]) / 2
        elif winner_bucket.get("type") == "above":
            winner_center = winner_bucket["low"] + 100  # above range
        elif winner_bucket.get("type") == "below":
            winner_center = winner_bucket["high"] - 100  # below range
        else:
            continue

        prev_winner_side = "above" if winner_center > range_mid else "below"

        # Calculate hours between settled event and now
        try:
            settled_dt = datetime.fromisoformat(latest_close.replace("Z", "+00:00"))
            hours_between = (datetime.now(timezone.utc) - settled_dt).total_seconds() / 3600
        except (ValueError, TypeError):
            hours_between = 24.0

        # Group open markets by event_ticker
        open_by_event = defaultdict(list)
        for mkt in open_markets:
            et = mkt.get("event_ticker", "")
            if et:
                open_by_event[et].append(mkt)

        # For the next open event, predict same direction continues
        for et, mkts in open_by_event.items():
            # Determine which bucket(s) are above/below midpoint
            for mkt in mkts:
                ticker = mkt.get("ticker", "")
                if ticker in recent_tickers:
                    continue

                title = mkt.get("title", "")
                bucket = parse_bucket_from_title(title)
                if not bucket or bucket.get("type") != "bucket":
                    continue

                bucket_center = (bucket["low"] + bucket["high"]) / 2
                yes_ask = mkt.get("yes_ask", 0) or mkt.get("last_price", 50) or 50

                # Predict momentum: same direction continues
                bucket_side = "above" if bucket_center > range_mid else "below"

                if bucket_side == prev_winner_side and yes_ask < 50:
                    # Momentum favors this bucket, it's underpriced
                    side = "buy_yes"
                    # Estimate a slight edge from momentum
                    fair_cents = yes_ask + 5  # momentum adds ~5c of edge
                    edge = fair_cents - yes_ask
                elif bucket_side != prev_winner_side and yes_ask > 50:
                    # Counter-momentum bucket is overpriced
                    side = "buy_no"
                    fair_cents = yes_ask - 5
                    edge = yes_ask - fair_cents
                else:
                    continue

                if edge < MIN_EDGE_TO_LOG:
                    continue

                preds.append(Prediction(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    ticker=ticker,
                    title=title,
                    strategy="crypto_momentum",
                    sub_strategy=asset,
                    side=side,
                    market_price_cents=yes_ask,
                    model_fair_value_cents=round(fair_cents, 1),
                    edge_cents=round(edge, 1),
                    model_details={
                        "prev_winner_side": prev_winner_side,
                        "prev_winner_position": round(winner_center, 2),
                        "range_mid": round(range_mid, 2),
                        "bucket_center": round(bucket_center, 2),
                        "hours_between": round(hours_between, 1),
                        "settled_event": latest_event,
                    },
                ))

    logger.info(f"Crypto Momentum: {len(preds)} predictions across BTC/ETH")
    return preds


# ---------------------------------------------------------------------------
# MARKET MAKER OPPORTUNITY SCANNER
# ---------------------------------------------------------------------------

def scan_market_maker(client: KalshiClient, recent_tickers: set) -> list[Prediction]:
    """Scan high-volume open markets for viable market-making opportunities.

    Logs markets where spread >= 3c and volume >= 100.
    """
    preds = []

    all_series = config.TARGET_MARKET_SERIES

    for series in all_series:
        try:
            resp = client.get_markets(series_ticker=series, limit=100, status="open")
            markets = resp.get("markets", [])
        except Exception:
            continue

        for mkt in markets:
            ticker = mkt.get("ticker", "")
            if ticker in recent_tickers:
                continue

            title = mkt.get("title", "")
            yes_ask = mkt.get("yes_ask", 0) or 0
            yes_bid = mkt.get("yes_bid", 0) or 0
            volume = mkt.get("volume", 0) or 0

            if yes_ask == 0 or yes_bid == 0:
                continue

            spread = yes_ask - yes_bid
            hrs = hours_to_settlement(mkt)

            # Only log viable MM opportunities
            if spread < 3 or volume < 100:
                continue

            strategy_name, sub_strat = classify_market(ticker, title)

            # Fair value = midpoint of spread
            midpoint = (yes_ask + yes_bid) / 2

            preds.append(Prediction(
                timestamp=datetime.now(timezone.utc).isoformat(),
                ticker=ticker,
                title=title,
                strategy="market_maker",
                sub_strategy=f"{strategy_name}_{sub_strat}",
                side="buy_both",  # MM is delta-neutral
                market_price_cents=midpoint,
                model_fair_value_cents=midpoint,
                edge_cents=round(spread / 2, 1),  # half-spread is theoretical edge per side
                model_details={
                    "yes_ask": yes_ask,
                    "yes_bid": yes_bid,
                    "spread": spread,
                    "volume": volume,
                    "hours_to_settlement": round(hrs, 1),
                    "series": series,
                },
            ))

    logger.info(f"Market Maker: {len(preds)} viable opportunities (spread>=3c, vol>=100)")
    return preds


# ---------------------------------------------------------------------------
# SCAN: Run all models, log predictions
# ---------------------------------------------------------------------------

def scan_all():
    """Run all strategy models against all open markets and log predictions."""
    logger.info("=" * 60)
    logger.info("STRATEGY TESTER: SCAN")
    logger.info("=" * 60)

    client = KalshiClient()
    existing = load_predictions()
    recent = get_recent_tickers(existing)
    logger.info(f"Loaded {len(existing)} existing predictions, {len(recent)} recent (dedup window)")

    all_preds = []

    # Weather
    try:
        weather_preds = scan_weather(client, recent)
        all_preds.extend(weather_preds)
    except Exception as e:
        logger.error(f"Weather scan failed: {e}")

    # Crypto
    try:
        crypto_preds = scan_crypto(client, recent)
        all_preds.extend(crypto_preds)
    except Exception as e:
        logger.error(f"Crypto scan failed: {e}")

    # Arb
    try:
        arb_preds = scan_arb(client, recent)
        all_preds.extend(arb_preds)
    except Exception as e:
        logger.error(f"Arb scan failed: {e}")

    # Sports
    try:
        sports_preds = scan_sports(client, recent)
        all_preds.extend(sports_preds)
    except Exception as e:
        logger.error(f"Sports scan failed: {e}")

    # NBA Props
    try:
        nba_props_preds = scan_nba_props(client, recent)
        all_preds.extend(nba_props_preds)
    except Exception as e:
        logger.error(f"NBA Props scan failed: {e}")

    # Crypto Momentum
    try:
        crypto_momentum_preds = scan_crypto_momentum(client, recent)
        all_preds.extend(crypto_momentum_preds)
    except Exception as e:
        logger.error(f"Crypto Momentum scan failed: {e}")

    # Market Maker
    try:
        mm_preds = scan_market_maker(client, recent)
        all_preds.extend(mm_preds)
    except Exception as e:
        logger.error(f"Market Maker scan failed: {e}")

    # Save new predictions
    for p in all_preds:
        append_prediction(p)

    logger.info(f"Logged {len(all_preds)} new predictions")
    logger.info(f"  Weather:          {sum(1 for p in all_preds if p.strategy == 'weather')}")
    logger.info(f"  Crypto:           {sum(1 for p in all_preds if p.strategy == 'crypto')}")
    logger.info(f"  Arb:              {sum(1 for p in all_preds if p.strategy == 'arb')}")
    logger.info(f"  Sports:           {sum(1 for p in all_preds if p.strategy == 'sports')}")
    logger.info(f"  NBA Props:        {sum(1 for p in all_preds if p.strategy == 'nba_props')}")
    logger.info(f"  Crypto Momentum:  {sum(1 for p in all_preds if p.strategy == 'crypto_momentum')}")
    logger.info(f"  Market Maker:     {sum(1 for p in all_preds if p.strategy == 'market_maker')}")

    return all_preds


# ---------------------------------------------------------------------------
# GRADE: Check settled markets, update predictions with outcomes
# ---------------------------------------------------------------------------

def grade_predictions():
    """Grade all ungraded predictions against actual market settlements."""
    logger.info("=" * 60)
    logger.info("STRATEGY TESTER: GRADE")
    logger.info("=" * 60)

    preds = load_predictions()
    ungraded = [p for p in preds if not p.get("settled", False)]
    if not ungraded:
        logger.info("No ungraded predictions to check")
        return

    client = KalshiClient()
    graded_count = 0
    market_cache = {}

    for p in preds:
        if p.get("settled", False):
            continue

        ticker = p["ticker"]
        if ticker not in market_cache:
            try:
                result = client.get_market(ticker)
                market_cache[ticker] = result.get("market", {})
                time.sleep(0.2)
            except Exception:
                continue

        mkt = market_cache.get(ticker, {})
        status = mkt.get("status", "")
        result = mkt.get("result", "")

        if status != "settled" or not result:
            continue

        p["settled"] = True
        p["settlement_result"] = result

        # Calculate P&L and correctness
        side = p.get("side", "")
        market_price = p.get("market_price_cents", 50)

        if side == "buy_yes":
            if result == "yes":
                p["pnl_per_contract_cents"] = 100 - market_price
                p["correct"] = True
            else:
                p["pnl_per_contract_cents"] = -market_price
                p["correct"] = False
        elif side == "buy_no":
            no_price = 100 - market_price
            if result == "no":
                p["pnl_per_contract_cents"] = 100 - no_price
                p["correct"] = True
            else:
                p["pnl_per_contract_cents"] = -no_price
                p["correct"] = False
        elif side == "buy_both":
            # Arb always "wins" (pays $1)
            total_cost = p.get("model_details", {}).get("total_cost", 100)
            p["pnl_per_contract_cents"] = 100 - total_cost
            p["correct"] = True

        graded_count += 1

    save_predictions(preds)
    logger.info(f"Graded {graded_count} predictions")
    return graded_count


# ---------------------------------------------------------------------------
# REPORT: Print accuracy scorecard
# ---------------------------------------------------------------------------

def generate_report():
    """Generate comprehensive accuracy report by strategy."""
    preds = load_predictions()
    settled = [p for p in preds if p.get("settled", False)]
    ungraded = [p for p in preds if not p.get("settled", False)]

    print()
    print("=" * 80)
    print(f"STRATEGY TESTER SCORECARD — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Total predictions: {len(preds)} | Settled: {len(settled)} | Pending: {len(ungraded)}")
    print("=" * 80)

    if not settled:
        print("\nNo settled predictions yet. Run 'scan' to log predictions,")
        print("then wait for markets to settle and run 'grade'.")
        return

    # Group by strategy + sub_strategy
    groups = defaultdict(list)
    for p in settled:
        key = f"{p['strategy']}/{p.get('sub_strategy', 'all')}"
        groups[key].append(p)

    print(f"\n{'Strategy':<25s} {'Trades':>6s} {'Wins':>5s} {'Rate':>6s} {'Avg Edge':>8s} "
          f"{'Avg P&L':>8s} {'Total P&L':>10s} {'p-value':>8s} {'Verdict':>10s}")
    print("-" * 96)

    scorecard = {}
    for key in sorted(groups.keys()):
        trades = groups[key]
        n = len(trades)
        wins = sum(1 for t in trades if t.get("correct", False))
        win_rate = wins / n if n > 0 else 0
        avg_edge = sum(t.get("edge_cents", 0) for t in trades) / n
        avg_pnl = sum(t.get("pnl_per_contract_cents", 0) for t in trades) / n
        total_pnl = sum(t.get("pnl_per_contract_cents", 0) for t in trades)

        # Average market implied probability (null hypothesis)
        avg_implied = sum(
            t.get("market_price_cents", 50) / 100 if t.get("side") == "buy_yes"
            else (100 - t.get("market_price_cents", 50)) / 100
            for t in trades
        ) / n

        pval = binomial_p_value(n, wins, avg_implied)

        if pval < 0.01 and avg_pnl > 0:
            verdict = "EDGE"
        elif pval < 0.05 and avg_pnl > 0:
            verdict = "PROMISING"
        elif n < 10:
            verdict = "TOO FEW"
        elif avg_pnl > 0:
            verdict = "WEAK"
        else:
            verdict = "NO EDGE"

        print(f"{key:<25s} {n:>6d} {wins:>5d} {win_rate:>5.0%} {avg_edge:>+7.1f}c "
              f"{avg_pnl:>+7.1f}c {total_pnl:>+9.1f}c {pval:>8.4f} {verdict:>10s}")

        scorecard[key] = {
            "trades": n, "wins": wins, "win_rate": round(win_rate, 3),
            "avg_edge": round(avg_edge, 1), "avg_pnl": round(avg_pnl, 1),
            "total_pnl": round(total_pnl, 1), "p_value": round(pval, 4),
            "verdict": verdict,
        }

    # Overall
    print("-" * 96)
    n = len(settled)
    wins = sum(1 for t in settled if t.get("correct", False))
    total_pnl = sum(t.get("pnl_per_contract_cents", 0) for t in settled)
    print(f"{'TOTAL':<25s} {n:>6d} {wins:>5d} {wins/n:>5.0%} {'':>8s} "
          f"{'':>8s} {total_pnl:>+9.1f}c")

    # Save scorecard
    scorecard["_meta"] = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "total_predictions": len(preds),
        "total_settled": len(settled),
        "total_pending": len(ungraded),
    }
    with open(SCORECARD_FILE, "w") as f:
        json.dump(scorecard, f, indent=2)

    print(f"\nScorecard saved to {SCORECARD_FILE}")

    # Recommendations
    print("\n--- RECOMMENDATIONS ---")
    for key, sc in scorecard.items():
        if key.startswith("_"):
            continue
        if sc.get("verdict") == "EDGE":
            print(f"  GO LIVE: {key} — {sc['trades']} trades, {sc['win_rate']:.0%} win rate, p={sc['p_value']:.4f}")
        elif sc.get("verdict") == "NO EDGE":
            print(f"  KILL:    {key} — losing money, no statistical edge")
        elif sc.get("verdict") == "TOO FEW":
            print(f"  WAIT:    {key} — need more data ({sc['trades']} trades, need 10+)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python strategy_tester.py [scan|grade|report|run]")
        print("  scan   — Log predictions for all open markets")
        print("  grade  — Grade settled predictions against outcomes")
        print("  report — Print accuracy scorecard")
        print("  run    — Full cycle: scan + grade + report")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "scan":
        scan_all()
    elif cmd == "grade":
        grade_predictions()
    elif cmd == "report":
        generate_report()
    elif cmd == "run":
        scan_all()
        grade_predictions()
        generate_report()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
