"""
crypto_strategy.py -- Crypto daily range edge detection vs. Kalshi markets.

Supports BTC, ETH, and SOL.

STRATEGY: Compare model-implied bucket probabilities (from realized volatility
and current price) to Kalshi crypto daily range market prices. When the market
price diverges from our fair value by >4 cents, we flag it as an edge.

KEY INSIGHT: Kalshi crypto daily range markets (KXBTC/KXETH/KXSOL series) divide
the next 24h price into buckets. Retail traders systematically overprice the center
buckets ("crypto will stay where it is") and underprice the tails. Our model uses
actual realized volatility to calculate log-normal probabilities for each bucket.

MODEL OVERVIEW:
1. Fetch current price from Coinbase + CoinGecko (cross-verify).
2. Fetch 30-day price history from CoinGecko for realized volatility.
3. Compute 30-day and 7-day annualized realized vol; use the higher one (conservative).
4. Price at settlement follows: ln(S_t/S_0) ~ N(0, sigma * sqrt(t))
5. Period vol = daily_vol / sqrt(24) * sqrt(hours_to_settlement)
6. P(bucket) = Phi(ln(high/S)/sigma_period) - Phi(ln(low/S)/sigma_period)
7. Edge = fair_NO - market_NO_ask (for overpriced center buckets)
   or fair_YES - market_YES_ask (for underpriced tail buckets)

SETTLEMENT: 5:00 PM EDT daily.

EDGE SOURCES:
1. Center buckets are overpriced (retail bias: "crypto stays put")
2. Tail buckets are underpriced (retail underestimates crypto volatility)
3. Vol regime changes: 7-day vol spikes aren't reflected in prices fast enough
"""

import math
import time as _time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests
from rich.console import Console
from rich.table import Table

import config
from kalshi_client import KalshiClient

console = Console()

# =============================================================================
# CONSTANTS
# =============================================================================
KALSHI_BTC_SERIES = "KXBTC"
BUCKET_WIDTH = 250  # $250 per bucket (BTC default)
SETTLEMENT_HOUR_EDT = 17  # 5 PM EDT = 21:00 UTC (or 22:00 during EST)
MIN_EDGE_CENTS = 4  # Minimum edge to flag (in cents)

# CoinGecko free API (no key required, 10-30 calls/min)
COINGECKO_PRICE_URL = "https://api.coingecko.com/api/v3/simple/price"
COINGECKO_HISTORY_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

# Coinbase public API (no key required)
COINBASE_PRICE_URL = "https://api.coinbase.com/v2/prices/BTC-USD/spot"

# ---- Multi-asset crypto configuration ----
CRYPTO_ASSETS = {
    "BTC": {
        "kalshi_series": "KXBTC",
        "coinbase_pair": "BTC-USD",
        "coingecko_id": "bitcoin",
        "binance_symbol": "BTCUSDT",
        "default_annual_vol": 0.55,  # ~55% fallback
        "bucket_width": 250,
        "name": "Bitcoin",
    },
    "ETH": {
        "kalshi_series": "KXETH",
        "coinbase_pair": "ETH-USD",
        "coingecko_id": "ethereum",
        "binance_symbol": "ETHUSDT",
        "default_annual_vol": 0.65,  # ~65% fallback
        "bucket_width": 25,  # ETH buckets are narrower ($25)
        "name": "Ethereum",
    },
    "SOL": {
        "kalshi_series": "KXSOL",
        "coinbase_pair": "SOL-USD",
        "coingecko_id": "solana",
        "binance_symbol": "SOLUSDT",
        "default_annual_vol": 0.85,  # ~85% fallback
        "bucket_width": 2,  # SOL buckets are narrow ($2)
        "name": "Solana",
    },
}


@dataclass
class CryptoEdge:
    """A detected crypto market edge (BTC, ETH, SOL)."""
    ticker: str
    asset: str              # "BTC", "ETH", or "SOL"
    bucket_low: float
    bucket_high: float
    side: str               # "buy_yes" or "buy_no"
    entry_price: float      # Price we'd pay (cents)
    fair_value: float       # Our model's fair value (cents)
    edge: float             # fair_value - entry_price (positive = edge)
    win_prob: float         # Model probability that this side wins
    current_price: float    # Current asset price used in calculation
    vol_used: float         # Annualized vol used (as decimal, e.g. 0.55 = 55%)


# Keep backward compat alias
BtcEdge = CryptoEdge


# =============================================================================
# GENERALIZED MULTI-ASSET DATA FETCHING
# =============================================================================

def fetch_crypto_price(asset: str = "BTC") -> Optional[dict]:
    """
    Fetch current price for any supported crypto asset from Coinbase + CoinGecko + Binance.
    Cross-verifies sources and flags if spread > threshold.

    Args:
        asset: "BTC", "ETH", or "SOL"

    Returns:
        dict with keys: price, coinbase, coingecko, binance, spread, sources_agree
        or None on failure.
    """
    cfg = CRYPTO_ASSETS.get(asset)
    if not cfg:
        console.print(f"[red]Unknown crypto asset: {asset}[/red]")
        return None

    prices = {}

    # Source 1: Coinbase
    try:
        resp = requests.get(
            f"https://api.coinbase.com/v2/prices/{cfg['coinbase_pair']}/spot",
            timeout=10,
        )
        resp.raise_for_status()
        prices["coinbase"] = float(resp.json()["data"]["amount"])
    except Exception as e:
        console.print(f"[yellow]Coinbase {asset} price fetch failed: {e}[/yellow]")

    # Source 2: CoinGecko
    try:
        resp = requests.get(
            COINGECKO_PRICE_URL,
            params={"ids": cfg["coingecko_id"], "vs_currencies": "usd"},
            timeout=10,
        )
        resp.raise_for_status()
        prices["coingecko"] = float(resp.json()[cfg["coingecko_id"]]["usd"])
    except Exception as e:
        console.print(f"[yellow]CoinGecko {asset} price fetch failed: {e}[/yellow]")

    # Source 3: Binance (additional cross-check)
    try:
        resp = requests.get(
            f"https://api.binance.com/api/v3/ticker/price?symbol={cfg['binance_symbol']}",
            timeout=10,
        )
        resp.raise_for_status()
        prices["binance"] = float(resp.json()["price"])
    except Exception as e:
        console.print(f"[yellow]Binance {asset} price fetch failed: {e}[/yellow]")

    if not prices:
        console.print(f"[red]All {asset} price sources failed. Cannot proceed.[/red]")
        return None

    # Average available sources
    price_values = list(prices.values())
    avg_price = sum(price_values) / len(price_values)

    # Max spread between any two sources
    spread = max(price_values) - min(price_values) if len(price_values) > 1 else 0.0
    # Threshold scales with price level (0.5% of price)
    spread_threshold = avg_price * 0.005
    sources_agree = spread <= spread_threshold

    if not sources_agree:
        console.print(
            f"[bold red]WARNING: {asset} price sources diverge by ${spread:.2f}! "
            f"{prices}[/bold red]"
        )

    return {
        "price": avg_price,
        "coinbase": prices.get("coinbase"),
        "coingecko": prices.get("coingecko"),
        "binance": prices.get("binance"),
        "spread": spread,
        "sources_agree": sources_agree,
    }


def fetch_crypto_volatility(asset: str = "BTC") -> Optional[dict]:
    """
    Fetch 30-day price history from CoinGecko and compute realized volatility
    for any supported crypto asset.

    Args:
        asset: "BTC", "ETH", or "SOL"

    Returns:
        dict with vol data, or None on failure.
    """
    cfg = CRYPTO_ASSETS.get(asset)
    if not cfg:
        return None

    coingecko_id = cfg["coingecko_id"]
    history_url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}/market_chart"

    try:
        resp = requests.get(
            history_url,
            params={"vs_currency": "usd", "days": "30", "interval": "daily"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        prices_raw = data.get("prices", [])

        if len(prices_raw) < 8:
            console.print(f"[red]Not enough {asset} price history for vol calculation.[/red]")
            return None

        closes = [p[1] for p in prices_raw]
        log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]

        daily_vol_30d = _stdev(log_returns)
        annualized_vol_30d = daily_vol_30d * math.sqrt(365)

        recent_returns = log_returns[-7:] if len(log_returns) >= 7 else log_returns
        daily_vol_7d = _stdev(recent_returns)
        annualized_vol_7d = daily_vol_7d * math.sqrt(365)

        vol_used = max(annualized_vol_30d, annualized_vol_7d)
        # Floor at default vol for the asset (prevents underestimation)
        vol_used = max(vol_used, cfg["default_annual_vol"] * 0.5)
        daily_vol = vol_used / math.sqrt(365)

        hours_to_settle = _hours_to_next_settlement()
        period_vol = daily_vol * math.sqrt(hours_to_settle / 24.0)

        return {
            "daily_vol_30d": daily_vol_30d,
            "daily_vol_7d": daily_vol_7d,
            "annualized_vol_30d": annualized_vol_30d,
            "annualized_vol_7d": annualized_vol_7d,
            "vol_used": vol_used,
            "daily_vol": daily_vol,
            "hours_to_settle": hours_to_settle,
            "period_vol": period_vol,
        }
    except Exception as e:
        console.print(f"[red]{asset} volatility fetch error: {e}[/red]")
        return None


def _parse_crypto_bucket(ticker: str, title: str, asset: str = "BTC") -> Optional[dict]:
    """
    Parse a crypto market ticker/title to extract bucket bounds.
    Works for BTC, ETH, SOL -- all use similar title patterns on Kalshi.
    """
    import re

    title_lower = title.lower()

    # Try "between $X and $Y" pattern
    between_match = re.search(
        r'between\s+\$?([\d,.]+)\s+and\s+\$?([\d,.]+)', title_lower
    )
    if between_match:
        low = float(between_match.group(1).replace(",", ""))
        high = float(between_match.group(2).replace(",", ""))
        return {"type": "bucket", "low": low, "high": high + 1}

    # Try "X or above" pattern
    above_match = re.search(r'\$?([\d,.]+)\s+or\s+above', title_lower)
    if above_match:
        threshold = float(above_match.group(1).replace(",", ""))
        return {"type": "above", "low": threshold, "high": threshold * 5}

    # Try "X or below" pattern
    below_match = re.search(r'\$?([\d,.]+)\s+or\s+below', title_lower)
    if below_match:
        threshold = float(below_match.group(1).replace(",", ""))
        return {"type": "below", "low": 0, "high": threshold + 1}

    # Try range: "$X to $Y" or "$X - $Y"
    range_match = re.search(r'\$([\d,.]+)\s*(?:to|-)\s*\$([\d,.]+)', title)
    if range_match:
        low = float(range_match.group(1).replace(",", ""))
        high = float(range_match.group(2).replace(",", ""))
        return {"type": "bucket", "low": low, "high": high}

    # Try "above $X" / "below $X"
    if "above" in title_lower or ">" in title:
        match = re.search(r'[\$>]\s*([\d,.]+)', title)
        if match:
            threshold = float(match.group(1).replace(",", ""))
            return {"type": "above", "low": threshold, "high": threshold * 5}

    if "below" in title_lower or "<" in title:
        match = re.search(r'[\$<]\s*([\d,.]+)', title)
        if match:
            threshold = float(match.group(1).replace(",", ""))
            return {"type": "below", "low": 0, "high": threshold + 1}

    # Fallback: parse from ticker
    cfg = CRYPTO_ASSETS.get(asset, CRYPTO_ASSETS["BTC"])
    bucket_width = cfg["bucket_width"]

    if "-B" in ticker:
        try:
            center = float(ticker.split("-B")[-1])
            half = bucket_width / 2
            return {"type": "bucket", "low": center - half, "high": center + half}
        except (ValueError, IndexError):
            pass

    if "-T" in ticker:
        try:
            threshold = float(ticker.split("-T")[-1])
            if "above" in title_lower or "higher" in title_lower:
                return {"type": "above", "low": threshold, "high": threshold * 5}
            elif "below" in title_lower or "lower" in title_lower:
                return {"type": "below", "low": 0, "high": threshold}
            else:
                return {"type": "above", "low": threshold, "high": threshold * 5}
        except (ValueError, IndexError):
            pass

    return None


def find_crypto_edges(asset: str = "BTC", client: KalshiClient = None) -> list[CryptoEdge]:
    """
    Main strategy function: find all daily range market edges for a crypto asset.

    Args:
        asset: "BTC", "ETH", or "SOL"
        client: KalshiClient instance (created if not provided)

    Returns:
        List of CryptoEdge objects sorted by edge magnitude.
    """
    cfg = CRYPTO_ASSETS.get(asset)
    if not cfg:
        console.print(f"[red]Unknown asset: {asset}[/red]")
        return []

    client = client or KalshiClient()
    series = cfg["kalshi_series"]
    asset_name = cfg["name"]

    # Step 1: Get current price
    price_data = fetch_crypto_price(asset)
    if not price_data:
        return []
    current_price = price_data["price"]

    console.print(f"  {asset} Price: [bold]${current_price:,.2f}[/bold]", end="")
    source_parts = []
    for src in ["coinbase", "coingecko", "binance"]:
        if price_data.get(src):
            source_parts.append(f"{src.title()}: ${price_data[src]:,.2f}")
    if source_parts:
        console.print(f"  ({' | '.join(source_parts)} | Spread: ${price_data['spread']:.2f})")
    else:
        console.print()

    # Step 2: Compute volatility
    vol_data = fetch_crypto_volatility(asset)
    if not vol_data:
        # Use default vol for this asset
        default_vol = cfg["default_annual_vol"]
        console.print(f"  [yellow]Using default {asset} vol: {default_vol:.0%}[/yellow]")
        hours_to_settle = _hours_to_next_settlement()
        daily_vol = default_vol / math.sqrt(365)
        period_vol = daily_vol * math.sqrt(hours_to_settle / 24.0)
        vol_data = {
            "vol_used": default_vol,
            "period_vol": period_vol,
            "hours_to_settle": hours_to_settle,
            "annualized_vol_30d": default_vol,
            "annualized_vol_7d": default_vol,
        }

    period_vol = vol_data["period_vol"]
    vol_annualized = vol_data["vol_used"]
    hours_to_settle = vol_data["hours_to_settle"]

    console.print(
        f"  Vol: [bold]{vol_annualized:.0%}[/bold] annualized "
        f"(30d: {vol_data['annualized_vol_30d']:.0%}, "
        f"7d: {vol_data['annualized_vol_7d']:.0%}) | "
        f"Period vol: {period_vol:.4f} | "
        f"Settlement in {hours_to_settle:.1f}h"
    )

    if not price_data["sources_agree"]:
        console.print(f"[bold red]  {asset} price source disagreement -- edges may be unreliable![/bold red]")

    # Step 3: Get Kalshi markets
    all_edges = []
    try:
        resp = client.get_markets(series_ticker=series, limit=100, status="open")
        markets = resp.get("markets", [])
    except Exception as e:
        console.print(f"[red]Failed to fetch {series} markets: {e}[/red]")
        return []

    if not markets:
        console.print(f"[yellow]No open {series} markets found.[/yellow]")
        return []

    console.print(f"  Found {len(markets)} open {asset} markets\n")

    # Step 4: Calculate edges
    for m in markets:
        ticker = m.get("ticker", "")
        title = m.get("title", "")
        yes_bid = float(m.get("yes_bid_dollars", 0) or 0) * 100
        yes_ask = float(m.get("yes_ask_dollars", 0) or 0) * 100
        no_bid = float(m.get("no_bid_dollars", 0) or 0) * 100
        no_ask = float(m.get("no_ask_dollars", 0) or 0) * 100

        if yes_bid <= 0 and yes_ask <= 1:
            continue
        if yes_bid >= 99:
            continue

        bucket = _parse_crypto_bucket(ticker, title, asset)
        if not bucket:
            continue

        bucket_low = bucket["low"]
        bucket_high = bucket["high"]

        fair_yes_prob = calc_bucket_probability(current_price, bucket_low, bucket_high, period_vol)
        fair_yes_cents = fair_yes_prob * 100
        fair_no_cents = 100 - fair_yes_cents

        # Check for BUY YES edge (underpriced tail buckets)
        if yes_ask > 0:
            buy_yes_edge = fair_yes_cents - yes_ask
            if buy_yes_edge > MIN_EDGE_CENTS:
                all_edges.append(CryptoEdge(
                    ticker=ticker,
                    asset=asset,
                    bucket_low=bucket_low,
                    bucket_high=bucket_high,
                    side="buy_yes",
                    entry_price=round(yes_ask, 1),
                    fair_value=round(fair_yes_cents, 1),
                    edge=round(buy_yes_edge, 1),
                    win_prob=round(fair_yes_prob, 4),
                    current_price=current_price,
                    vol_used=vol_annualized,
                ))

        # Check for BUY NO edge (overpriced center buckets)
        if no_ask > 0:
            buy_no_edge = fair_no_cents - no_ask
            if buy_no_edge > MIN_EDGE_CENTS:
                all_edges.append(CryptoEdge(
                    ticker=ticker,
                    asset=asset,
                    bucket_low=bucket_low,
                    bucket_high=bucket_high,
                    side="buy_no",
                    entry_price=round(no_ask, 1),
                    fair_value=round(fair_no_cents, 1),
                    edge=round(buy_no_edge, 1),
                    win_prob=round(1 - fair_yes_prob, 4),
                    current_price=current_price,
                    vol_used=vol_annualized,
                ))

    all_edges.sort(key=lambda e: e.edge, reverse=True)
    return all_edges


def display_crypto_edges(edges: list[CryptoEdge], asset: str = None):
    """Pretty-print detected crypto edges."""
    if not edges:
        label = f"{asset} " if asset else "crypto "
        console.print(f"[yellow]No {label}edges found right now.[/yellow]")
        return

    # Determine title from edges
    assets_found = sorted(set(e.asset for e in edges))
    title_label = "/".join(assets_found) if assets_found else "Crypto"
    table = Table(title=f"{title_label} Daily Range -- Detected Edges", show_lines=False)
    table.add_column("Ticker", style="cyan", width=28)
    table.add_column("Asset", style="bold white", width=5)
    table.add_column("Bucket", justify="right", style="white", width=22)
    table.add_column("Action", style="bold", width=10)
    table.add_column("Entry", justify="right", width=8)
    table.add_column("Fair Val", justify="right", style="yellow", width=8)
    table.add_column("Edge", justify="right", style="bold green", width=7)
    table.add_column("Win %", justify="right", style="green", width=7)
    table.add_column("Vol", justify="right", style="dim", width=6)

    for e in edges:
        action_color = "green" if e.side == "buy_yes" else "red"
        action_text = "BUY YES" if e.side == "buy_yes" else "BUY NO"

        if e.bucket_high >= e.current_price * 4:
            bucket_text = f">= ${e.bucket_low:,.0f}"
        elif e.bucket_low <= 0:
            bucket_text = f"<= ${e.bucket_high:,.0f}"
        else:
            bucket_text = f"${e.bucket_low:,.0f} - ${e.bucket_high:,.0f}"

        table.add_row(
            e.ticker,
            e.asset,
            bucket_text,
            f"[{action_color}]{action_text}[/{action_color}]",
            f"{e.entry_price:.0f}c",
            f"{e.fair_value:.0f}c",
            f"+{e.edge:.0f}c",
            f"{e.win_prob:.0%}",
            f"{e.vol_used:.0%}",
        )

    console.print(table)
    console.print(
        f"\n[dim]Found {len(edges)} edges (min {MIN_EDGE_CENTS}c threshold). "
        f"Sources: Coinbase + CoinGecko + Binance (price), CoinGecko (vol). "
        f"Settlement: 5pm EDT.[/dim]"
    )


# =============================================================================
# LEGACY BTC-ONLY DATA FETCHING (kept for backward compatibility)
# =============================================================================

def fetch_btc_price() -> Optional[dict]:
    """
    Fetch current BTC price from Coinbase + CoinGecko.
    Cross-verifies sources and flags if spread > $100.

    Returns:
        dict with keys: price, coinbase, coingecko, spread, sources_agree
        or None on failure.
    """
    coinbase_price = None
    coingecko_price = None

    # Source 1: Coinbase
    try:
        resp = requests.get(COINBASE_PRICE_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        coinbase_price = float(data["data"]["amount"])
    except Exception as e:
        console.print(f"[yellow]Coinbase price fetch failed: {e}[/yellow]")

    # Source 2: CoinGecko
    try:
        resp = requests.get(
            COINGECKO_PRICE_URL,
            params={"ids": "bitcoin", "vs_currencies": "usd"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        coingecko_price = float(data["bitcoin"]["usd"])
    except Exception as e:
        console.print(f"[yellow]CoinGecko price fetch failed: {e}[/yellow]")

    # Need at least one source
    if coinbase_price is None and coingecko_price is None:
        console.print("[red]Both price sources failed. Cannot proceed.[/red]")
        return None

    # Average available sources
    prices = [p for p in [coinbase_price, coingecko_price] if p is not None]
    avg_price = sum(prices) / len(prices)

    spread = abs(coinbase_price - coingecko_price) if (coinbase_price and coingecko_price) else 0.0
    sources_agree = spread <= 100.0

    if not sources_agree:
        console.print(
            f"[bold red]WARNING: Price sources diverge by ${spread:.0f}! "
            f"Coinbase=${coinbase_price:.0f}, CoinGecko=${coingecko_price:.0f}[/bold red]"
        )

    return {
        "price": avg_price,
        "coinbase": coinbase_price,
        "coingecko": coingecko_price,
        "spread": spread,
        "sources_agree": sources_agree,
    }


def fetch_btc_volatility() -> Optional[dict]:
    """
    Fetch 30-day BTC price history from CoinGecko and compute realized volatility.

    Returns:
        dict with keys: daily_vol_30d, daily_vol_7d, annualized_vol_30d,
                        annualized_vol_7d, vol_used (annualized, the higher one),
                        hours_to_settle, period_vol
        or None on failure.
    """
    try:
        resp = requests.get(
            COINGECKO_HISTORY_URL,
            params={"vs_currency": "usd", "days": "30", "interval": "daily"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        prices_raw = data.get("prices", [])

        if len(prices_raw) < 8:
            console.print("[red]Not enough price history for vol calculation.[/red]")
            return None

        # Extract closing prices (each entry is [timestamp_ms, price])
        closes = [p[1] for p in prices_raw]

        # Log returns
        log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]

        # 30-day realized vol (daily, then annualized)
        daily_vol_30d = _stdev(log_returns)
        annualized_vol_30d = daily_vol_30d * math.sqrt(365)

        # 7-day realized vol (last 7 returns, weighted more heavily for regime detection)
        recent_returns = log_returns[-7:] if len(log_returns) >= 7 else log_returns
        daily_vol_7d = _stdev(recent_returns)
        annualized_vol_7d = daily_vol_7d * math.sqrt(365)

        # Conservative: use the HIGHER of the two
        vol_used = max(annualized_vol_30d, annualized_vol_7d)
        daily_vol = vol_used / math.sqrt(365)

        # Hours to settlement: next 5 PM EDT
        hours_to_settle = _hours_to_next_settlement()

        # Period vol: scale daily vol to remaining hours
        # daily_vol is for 24 hours; scale to hours_to_settle
        period_vol = daily_vol * math.sqrt(hours_to_settle / 24.0)

        return {
            "daily_vol_30d": daily_vol_30d,
            "daily_vol_7d": daily_vol_7d,
            "annualized_vol_30d": annualized_vol_30d,
            "annualized_vol_7d": annualized_vol_7d,
            "vol_used": vol_used,
            "daily_vol": daily_vol,
            "hours_to_settle": hours_to_settle,
            "period_vol": period_vol,
        }
    except Exception as e:
        console.print(f"[red]Volatility fetch error: {e}[/red]")
        return None


def _stdev(values: list[float]) -> float:
    """Sample standard deviation."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance)


def _hours_to_next_settlement() -> float:
    """
    Calculate hours until next 5 PM EDT settlement.
    EDT = UTC-4, EST = UTC-5. We approximate with UTC-4 (EDT).
    If settlement already passed today, target tomorrow.
    """
    now_utc = datetime.now(timezone.utc)
    # 5 PM EDT = 21:00 UTC (during EDT, roughly Apr-Nov)
    # 5 PM EST = 22:00 UTC (during EST, roughly Nov-Mar)
    # Use 21:00 UTC as default (EDT); close enough for vol scaling
    settlement_utc_hour = 21

    today_settlement = now_utc.replace(
        hour=settlement_utc_hour, minute=0, second=0, microsecond=0
    )

    if now_utc >= today_settlement:
        # Settlement already passed, target tomorrow
        next_settlement = today_settlement + timedelta(days=1)
    else:
        next_settlement = today_settlement

    delta = next_settlement - now_utc
    hours = delta.total_seconds() / 3600.0
    return max(0.5, hours)  # Floor at 0.5h to avoid near-zero vol


# =============================================================================
# PROBABILITY MODEL
# =============================================================================

def normal_cdf(x: float) -> float:
    """Standard normal CDF approximation."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def calc_bucket_probability(
    current_price: float,
    bucket_low: float,
    bucket_high: float,
    period_vol: float,
) -> float:
    """
    Calculate probability that BTC price at settlement falls in [bucket_low, bucket_high].

    Uses log-normal model: ln(S_t / S_0) ~ N(0, sigma_period)
    P(low < S_t < high) = Phi(ln(high/S_0) / sigma) - Phi(ln(low/S_0) / sigma)

    For boundary buckets (e.g., ">$90,000"), bucket_high should be very large.
    """
    if period_vol <= 0:
        # Zero vol: all probability on current price bucket
        if bucket_low <= current_price <= bucket_high:
            return 0.999
        return 0.001

    # Handle extreme bucket bounds
    if bucket_low <= 0:
        # "Below X" bucket: P(S_t < bucket_high)
        z_high = math.log(bucket_high / current_price) / period_vol
        prob = normal_cdf(z_high)
    elif bucket_high >= current_price * 5:
        # "Above X" bucket: P(S_t > bucket_low)
        z_low = math.log(bucket_low / current_price) / period_vol
        prob = 1.0 - normal_cdf(z_low)
    else:
        # Standard bucket
        z_low = math.log(bucket_low / current_price) / period_vol
        z_high = math.log(bucket_high / current_price) / period_vol
        prob = normal_cdf(z_high) - normal_cdf(z_low)

    return max(0.001, min(0.999, prob))


# =============================================================================
# MARKET PARSING
# =============================================================================

def _parse_btc_bucket(ticker: str, title: str) -> Optional[dict]:
    """
    Parse a BTC market ticker/title to extract bucket bounds.

    Kalshi BTC tickers look like:
        KXBTC-26MAR21-T87500   (above/below threshold)
        KXBTC-26MAR21-B87250   (bucket centered at 87250, so 87125-87375)
    Titles look like:
        "Will Bitcoin be between $87,000 and $87,249?"
        "Will Bitcoin be $87,250 or above?"
        "Will Bitcoin be $86,749 or below?"
    """
    import re

    title_lower = title.lower()

    # Try "between $X and $Y" pattern
    between_match = re.search(
        r'between\s+\$?([\d,]+)\s+and\s+\$?([\d,]+)', title_lower
    )
    if between_match:
        low = float(between_match.group(1).replace(",", ""))
        high = float(between_match.group(2).replace(",", ""))
        # Kalshi uses inclusive ranges like "$87,000 and $87,249"
        # The actual boundary is high + 1 (i.e., 87,250)
        return {"type": "bucket", "low": low, "high": high + 1}

    # Try "X or above" pattern
    above_match = re.search(r'\$?([\d,]+)\s+or\s+above', title_lower)
    if above_match:
        threshold = float(above_match.group(1).replace(",", ""))
        return {"type": "above", "low": threshold, "high": threshold * 5}

    # Try "X or below" pattern
    below_match = re.search(r'\$?([\d,]+)\s+or\s+below', title_lower)
    if below_match:
        threshold = float(below_match.group(1).replace(",", ""))
        return {"type": "below", "low": 0, "high": threshold + 1}

    # Fallback: parse from ticker
    if "-B" in ticker:
        try:
            center = float(ticker.split("-B")[-1])
            half = BUCKET_WIDTH / 2
            return {"type": "bucket", "low": center - half, "high": center + half}
        except (ValueError, IndexError):
            pass

    if "-T" in ticker:
        try:
            threshold = float(ticker.split("-T")[-1])
            if "above" in title_lower or "higher" in title_lower:
                return {"type": "above", "low": threshold, "high": threshold * 5}
            elif "below" in title_lower or "lower" in title_lower:
                return {"type": "below", "low": 0, "high": threshold}
            else:
                # Default: treat as above threshold
                return {"type": "above", "low": threshold, "high": threshold * 5}
        except (ValueError, IndexError):
            pass

    return None


# =============================================================================
# EDGE DETECTION
# =============================================================================

def find_btc_edges(client: KalshiClient = None) -> list[CryptoEdge]:
    """
    Legacy wrapper: find BTC daily range market edges.
    Delegates to find_crypto_edges("BTC").
    """
    return find_crypto_edges("BTC", client=client)


# =============================================================================
# DISPLAY
# =============================================================================

def display_edges(edges: list[CryptoEdge]):
    """Pretty-print detected crypto edges (legacy wrapper)."""
    display_crypto_edges(edges)


if __name__ == "__main__":
    import sys as _sys

    # Support: python crypto_strategy.py [BTC|ETH|SOL|all]
    asset_arg = _sys.argv[1].upper() if len(_sys.argv) > 1 else "all"

    if asset_arg == "ALL":
        all_edges = []
        for asset_key in CRYPTO_ASSETS:
            console.print(f"\n[bold cyan]--- {CRYPTO_ASSETS[asset_key]['name']} ({asset_key}) ---[/bold cyan]")
            edges = find_crypto_edges(asset_key)
            all_edges.extend(edges)
        display_crypto_edges(all_edges)
    else:
        edges = find_crypto_edges(asset_arg)
        display_crypto_edges(edges, asset=asset_arg)
