"""
latency_arb.py -- Latency arbitrage engine for KXBTC markets on Kalshi.

STRATEGY: Exploit the speed differential between crypto exchange price feeds
and Kalshi prediction market repricing. When BTC moves sharply on Coinbase/
Binance/CoinGecko, Kalshi bucket probabilities lag behind. We detect the
stale pricing and trade before the market catches up.

INSPIRATION: The 0x8dxd Polymarket bot that turned $313 into $2.3M by
watching crypto prices on Binance/Coinbase and trading prediction market
contracts before prices update.

HOW IT WORKS:
1. Fetch BTC price from 3 sources simultaneously (Coinbase, Binance, CoinGecko)
   to get the freshest possible price with cross-verification.
2. Fetch all open KXBTC markets and their orderbooks from Kalshi.
3. Use the REAL price + recent volatility to calculate true probability of
   BTC landing in each bucket via log-normal model (same math as crypto_strategy).
4. Compare our fair value to Kalshi's current pricing. When BTC has moved
   but Kalshi hasn't repriced, the stale buckets are tradeable.
5. Fire trade signals when edge > 5 cents, with urgency scoring.

KEY INSIGHT: Crypto prices update every second. Kalshi orderbooks update
when traders manually adjust. After a $500+ move in 5 minutes, there can
be a 1-5 minute window where bucket probabilities are stale.

Usage:
    python latency_arb.py --dry-run     # monitor and log signals
    python latency_arb.py --live        # monitor and place trades
"""

import argparse
import json
import logging
import math
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import requests

import config
from kalshi_client import KalshiClient

try:
    from alerts import alert_big_edge, alert_bot_error
except (ImportError, OSError):
    alert_big_edge = alert_bot_error = None


# =============================================================================
# CONSTANTS
# =============================================================================

KALSHI_BTC_SERIES = "KXBTC"

# Price source URLs
COINBASE_URL = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
BINANCE_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
COINGECKO_HISTORY_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

# Latency arb thresholds
MIN_PRICE_MOVE_PCT = 0.005      # 0.5% move triggers a scan
MIN_EDGE_CENTS = 5              # 5-cent minimum edge to signal
POLL_INTERVAL_SECONDS = 30      # How often to check prices
MAX_TRADES_PER_SESSION = 10     # Hard cap on trades per run
MAX_DOLLARS_PER_TRADE = 2.0     # Conservative: $1-2 per trade
DEFAULT_TRADE_DOLLARS = 1.0     # Default size per latency arb trade
DAILY_LOSS_CAP_PCT = 0.08       # 8% of account

# Urgency thresholds (higher = more urgent, trade faster)
URGENCY_HIGH = 0.8              # Edge > 15c or price moved > 2%
URGENCY_MEDIUM = 0.5            # Edge > 10c or price moved > 1%
URGENCY_LOW = 0.3               # Edge > 5c


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PriceSnapshot:
    """A point-in-time BTC price from multiple sources."""
    timestamp: float              # Unix timestamp
    coinbase: Optional[float]
    binance: Optional[float]
    coingecko: Optional[float]
    consensus: float              # Median of available sources
    source_count: int
    max_spread_pct: float         # Max pct spread between sources

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    @property
    def is_fresh(self) -> bool:
        return self.age_seconds < 60  # Fresh if < 1 minute old


@dataclass
class LatencySignal:
    """A detected latency arbitrage opportunity."""
    ticker: str
    title: str
    side: str                     # "buy_yes" or "buy_no"
    bucket_low: float
    bucket_high: float
    fair_value_cents: float       # Our model's fair value
    market_price_cents: float     # Current Kalshi ask price
    edge_cents: float             # fair - market (positive = edge)
    btc_price: float              # Real-time BTC price used
    price_move_pct: float         # How much BTC moved since last Kalshi update
    urgency: float                # 0-1 urgency score
    vol_used: float               # Annualized vol used
    hours_to_settle: float
    timestamp: float

    @property
    def is_actionable(self) -> bool:
        return self.edge_cents >= MIN_EDGE_CENTS and self.urgency >= URGENCY_LOW


# =============================================================================
# PRICE FETCHING (speed is everything here)
# =============================================================================

def _fetch_coinbase() -> Optional[float]:
    """Fetch BTC/USD from Coinbase. Typically fastest source."""
    try:
        resp = requests.get(COINBASE_URL, timeout=5)
        resp.raise_for_status()
        return float(resp.json()["data"]["amount"])
    except Exception:
        return None


def _fetch_binance() -> Optional[float]:
    """Fetch BTCUSDT from Binance. Often first to reflect moves."""
    try:
        resp = requests.get(BINANCE_URL, timeout=5)
        resp.raise_for_status()
        return float(resp.json()["price"])
    except Exception:
        return None


def _fetch_coingecko() -> Optional[float]:
    """Fetch BTC/USD from CoinGecko. Aggregated price, sometimes lags."""
    try:
        resp = requests.get(COINGECKO_URL, timeout=5)
        resp.raise_for_status()
        return float(resp.json()["bitcoin"]["usd"])
    except Exception:
        return None


def fetch_prices_parallel(logger: logging.Logger) -> Optional[PriceSnapshot]:
    """
    Fetch BTC price from all 3 sources simultaneously.
    Returns a PriceSnapshot with the consensus (median) price.
    Speed matters: uses ThreadPoolExecutor for parallel requests.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_fetch_coinbase): "coinbase",
            executor.submit(_fetch_binance): "binance",
            executor.submit(_fetch_coingecko): "coingecko",
        }
        for future in as_completed(futures):
            source = futures[future]
            try:
                price = future.result()
                if price and price > 0:
                    results[source] = price
            except Exception:
                pass

    if not results:
        logger.error("All 3 price sources failed")
        return None

    prices = list(results.values())
    prices.sort()
    # Use median for robustness against one outlier
    if len(prices) >= 3:
        consensus = prices[1]  # median of 3
    elif len(prices) == 2:
        consensus = sum(prices) / 2  # average of 2
    else:
        consensus = prices[0]  # single source

    # Calculate max spread between sources
    max_spread_pct = 0.0
    if len(prices) >= 2:
        max_spread_pct = (max(prices) - min(prices)) / min(prices) * 100

    snapshot = PriceSnapshot(
        timestamp=time.time(),
        coinbase=results.get("coinbase"),
        binance=results.get("binance"),
        coingecko=results.get("coingecko"),
        consensus=consensus,
        source_count=len(results),
        max_spread_pct=max_spread_pct,
    )

    sources_str = " | ".join(
        f"{src}=${p:,.0f}" for src, p in sorted(results.items())
    )
    logger.debug(
        f"BTC prices: {sources_str} | consensus=${consensus:,.0f} "
        f"spread={max_spread_pct:.2f}%"
    )

    return snapshot


def fetch_btc_volatility(logger: logging.Logger) -> Optional[dict]:
    """
    Fetch 30-day BTC history from CoinGecko and compute realized volatility.
    Returns dict with annualized vol, daily vol, and period vol.
    """
    try:
        resp = requests.get(
            COINGECKO_HISTORY_URL,
            params={"vs_currency": "usd", "days": "30", "interval": "daily"},
            timeout=15,
        )
        resp.raise_for_status()
        prices_raw = resp.json().get("prices", [])

        if len(prices_raw) < 8:
            logger.warning("Not enough price history for vol calculation")
            return None

        closes = [p[1] for p in prices_raw]
        log_returns = [
            math.log(closes[i] / closes[i - 1])
            for i in range(1, len(closes))
            if closes[i - 1] > 0
        ]

        if len(log_returns) < 5:
            return None

        # 30-day vol
        mean_ret = sum(log_returns) / len(log_returns)
        variance_30d = sum((r - mean_ret) ** 2 for r in log_returns) / (len(log_returns) - 1)
        daily_vol_30d = math.sqrt(variance_30d)
        annual_vol_30d = daily_vol_30d * math.sqrt(365)

        # 7-day vol (more reactive to recent regime changes)
        recent = log_returns[-7:] if len(log_returns) >= 7 else log_returns
        mean_recent = sum(recent) / len(recent)
        variance_7d = sum((r - mean_recent) ** 2 for r in recent) / max(1, len(recent) - 1)
        daily_vol_7d = math.sqrt(variance_7d)
        annual_vol_7d = daily_vol_7d * math.sqrt(365)

        # Conservative: use the higher of 30d and 7d
        annual_vol = max(annual_vol_30d, annual_vol_7d)
        daily_vol = annual_vol / math.sqrt(365)

        logger.info(
            f"BTC vol: 30d={annual_vol_30d:.0%} 7d={annual_vol_7d:.0%} "
            f"using={annual_vol:.0%}"
        )

        return {
            "annual_vol": annual_vol,
            "daily_vol": daily_vol,
            "annual_vol_30d": annual_vol_30d,
            "annual_vol_7d": annual_vol_7d,
        }
    except Exception as e:
        logger.warning(f"Volatility fetch failed: {e}")
        return None


# =============================================================================
# PROBABILITY MODEL (same log-normal math as crypto_strategy.py)
# =============================================================================

def normal_cdf(x: float) -> float:
    """Standard normal CDF approximation."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def lognormal_cdf(x: float, mu: float, sigma: float) -> float:
    """CDF of log-normal distribution."""
    if x <= 0:
        return 0.0
    z = (math.log(x) - mu) / sigma
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def calc_bucket_probability(
    current_price: float,
    bucket_low: float,
    bucket_high: float,
    annual_vol: float,
    hours_to_settlement: float,
) -> float:
    """
    Fair probability for BTC landing in [bucket_low, bucket_high] at settlement.
    Uses geometric Brownian motion / log-normal model.

    mu = ln(S) - 0.5 * sigma^2 * t  (zero-drift for short horizons)
    sigma_period = annual_vol * sqrt(t)
    t = hours_to_settlement / (365 * 24)
    """
    t = hours_to_settlement / (365.0 * 24.0)
    if t <= 0:
        t = 1.0 / (365.0 * 24.0)  # floor at 1 hour

    sigma_period = annual_vol * math.sqrt(t)
    mu = math.log(current_price) - 0.5 * sigma_period ** 2

    p_high = lognormal_cdf(bucket_high, mu, sigma_period) if bucket_high < 1e9 else 1.0
    p_low = lognormal_cdf(bucket_low, mu, sigma_period) if bucket_low > 0 else 0.0

    return max(0.001, min(0.999, p_high - p_low))


# =============================================================================
# MARKET PARSING (reused from auto_trade.py patterns)
# =============================================================================

def parse_btc_bucket(ticker: str, title: str) -> Optional[dict]:
    """
    Parse BTC market ticker/title to extract bucket bounds.

    Handles formats like:
        "Will Bitcoin be between $87,000 and $87,249?"
        "Will Bitcoin be $87,250 or above?"
        "Bitcoin above $87,500?"
        "Bitcoin $87,000 to $87,500?"
    """
    title_lower = title.lower()

    # Pattern: "between $X and $Y" (supports decimals)
    between_match = re.search(
        r'between\s+\$?([\d,.]+)\s+and\s+\$?([\d,.]+)', title_lower
    )
    if between_match:
        low = float(between_match.group(1).replace(",", ""))
        high = float(between_match.group(2).replace(",", ""))
        # Kalshi "between $X and $Y" is inclusive; add epsilon for exclusive upper
        return {"type": "bucket", "low": low, "high": high + 0.01}

    # Pattern: "X or above" or "above $X"
    if "above" in title_lower or ">" in title:
        match = re.search(r'[\$>]\s*([\d,.]+)', title)
        if match:
            threshold = float(match.group(1).replace(",", ""))
            return {"type": "above", "low": threshold, "high": 1e9}
        above_match = re.search(r'\$?([\d,.]+)\s+or\s+above', title_lower)
        if above_match:
            threshold = float(above_match.group(1).replace(",", ""))
            return {"type": "above", "low": threshold, "high": 1e9}

    # Pattern: "X or below" or "below $X"
    if "below" in title_lower or "<" in title:
        match = re.search(r'[\$<]\s*([\d,.]+)', title)
        if match:
            threshold = float(match.group(1).replace(",", ""))
            return {"type": "below", "low": 0, "high": threshold}
        below_match = re.search(r'\$?([\d,.]+)\s+or\s+below', title_lower)
        if below_match:
            threshold = float(below_match.group(1).replace(",", ""))
            return {"type": "below", "low": 0, "high": threshold + 0.01}

    # Pattern: "$X to $Y" range (supports decimals)
    range_match = re.search(r'\$([\d,.]+)\s*(?:to|-)\s*\$([\d,.]+)', title)
    if range_match:
        low = float(range_match.group(1).replace(",", ""))
        high = float(range_match.group(2).replace(",", ""))
        return {"type": "bucket", "low": low, "high": high}

    # Fallback: parse from ticker
    if "-T" in ticker:
        try:
            threshold = float(ticker.split("-T")[-1])
            if "above" in title_lower or ">" in title_lower:
                return {"type": "above", "low": threshold, "high": 1e9}
            else:
                return {"type": "below", "low": 0, "high": threshold}
        except (ValueError, IndexError):
            pass

    if "-B" in ticker:
        try:
            center = float(ticker.split("-B")[-1])
            # BTC buckets are $500 wide (center ± 250)
            return {"type": "bucket", "low": center - 250, "high": center + 250}
        except (ValueError, IndexError):
            pass

    return None


def hours_to_settlement(market: dict) -> float:
    """Extract hours to settlement from a market dict."""
    close_time_str = market.get("close_time", "")
    if close_time_str:
        try:
            close_dt = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
            hours = (close_dt - datetime.now(timezone.utc)).total_seconds() / 3600
            return max(0.1, hours)
        except Exception:
            pass

    # Fallback: parse date from ticker
    ticker = market.get("ticker", "")
    month_map = {
        "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
        "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
        "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
    }
    match = re.search(r'(\d{2})([A-Z]{3})(\d{2})', ticker)
    if match:
        day, mon_str, year_short = match.group(1), match.group(2), match.group(3)
        mon = month_map.get(mon_str)
        if mon:
            try:
                dt = datetime.strptime(
                    f"20{year_short}-{mon}-{day}", "%Y-%m-%d"
                ).replace(hour=22, minute=0, tzinfo=timezone.utc)
                hours = (dt - datetime.now(timezone.utc)).total_seconds() / 3600
                return max(0.1, hours)
            except Exception:
                pass

    return 24.0  # Default: 24 hours


# =============================================================================
# LATENCY EDGE DETECTION (the core engine)
# =============================================================================

def scan_for_latency_edges(
    price_snapshot: PriceSnapshot,
    last_price: Optional[float],
    annual_vol: float,
    client: KalshiClient,
    logger: logging.Logger,
) -> list[LatencySignal]:
    """
    Core latency arb scanner: compare real-time BTC price against Kalshi
    bucket pricing. When price has moved but Kalshi hasn't repriced,
    we find the edge.

    Args:
        price_snapshot: Current multi-source BTC price
        last_price: Previous BTC price (to measure move)
        annual_vol: Annualized realized volatility
        client: Authenticated Kalshi client
        logger: Logger instance

    Returns:
        List of LatencySignal objects sorted by edge size
    """
    current_price = price_snapshot.consensus
    signals = []

    # Calculate price move since last check
    if last_price and last_price > 0:
        price_move_pct = (current_price - last_price) / last_price
    else:
        price_move_pct = 0.0

    abs_move_pct = abs(price_move_pct)

    # Log the move
    if abs_move_pct > 0.001:
        direction = "UP" if price_move_pct > 0 else "DOWN"
        logger.info(
            f"BTC moved {direction} {abs_move_pct:.2%}: "
            f"${last_price:,.0f} -> ${current_price:,.0f} "
            f"(${current_price - last_price:+,.0f})"
        )

    # Fetch all open KXBTC markets
    try:
        resp = client.get_markets(series_ticker=KALSHI_BTC_SERIES, limit=200, status="open")
        markets = resp.get("markets", [])
    except Exception as e:
        logger.error(f"Failed to fetch KXBTC markets: {e}")
        return []

    if not markets:
        logger.warning("No open KXBTC markets found")
        return []

    logger.debug(f"Scanning {len(markets)} KXBTC markets for latency edges")

    for m in markets:
        ticker = m.get("ticker", "")
        title = m.get("title", "")

        # Parse market prices (Kalshi returns dollars, convert to cents)
        yes_bid = float(m.get("yes_bid_dollars", 0) or 0) * 100
        yes_ask = float(m.get("yes_ask_dollars", 0) or 0) * 100
        no_bid = float(m.get("no_bid_dollars", 0) or 0) * 100
        no_ask = float(m.get("no_ask_dollars", 0) or 0) * 100

        # Skip dead/illiquid markets
        if yes_bid <= 0 and yes_ask <= 1:
            continue
        if yes_bid >= 99:
            continue

        # Parse bucket
        bucket = parse_btc_bucket(ticker, title)
        if not bucket:
            continue

        bucket_low = bucket["low"]
        bucket_high = bucket["high"]

        # Get hours to settlement for this specific market
        hrs = hours_to_settlement(m)

        # Calculate our fair probability using CURRENT real-time price
        fair_yes_prob = calc_bucket_probability(
            current_price, bucket_low, bucket_high, annual_vol, hrs
        )
        fair_yes_cents = fair_yes_prob * 100
        fair_no_cents = 100 - fair_yes_cents

        # --- Check BUY YES edge (bucket is underpriced) ---
        if yes_ask > 0 and yes_ask < 99:
            edge = fair_yes_cents - yes_ask
            if edge >= MIN_EDGE_CENTS:
                urgency = _calc_urgency(edge, abs_move_pct, hrs)
                signals.append(LatencySignal(
                    ticker=ticker,
                    title=title,
                    side="buy_yes",
                    bucket_low=bucket_low,
                    bucket_high=bucket_high,
                    fair_value_cents=round(fair_yes_cents, 1),
                    market_price_cents=round(yes_ask, 1),
                    edge_cents=round(edge, 1),
                    btc_price=current_price,
                    price_move_pct=round(price_move_pct * 100, 2),
                    urgency=urgency,
                    vol_used=annual_vol,
                    hours_to_settle=hrs,
                    timestamp=time.time(),
                ))

        # --- Check BUY NO edge (bucket is overpriced) ---
        if no_ask > 0 and no_ask < 99:
            edge = fair_no_cents - no_ask
            if edge >= MIN_EDGE_CENTS:
                urgency = _calc_urgency(edge, abs_move_pct, hrs)
                signals.append(LatencySignal(
                    ticker=ticker,
                    title=title,
                    side="buy_no",
                    bucket_low=bucket_low,
                    bucket_high=bucket_high,
                    fair_value_cents=round(fair_no_cents, 1),
                    market_price_cents=round(no_ask, 1),
                    edge_cents=round(edge, 1),
                    btc_price=current_price,
                    price_move_pct=round(price_move_pct * 100, 2),
                    urgency=urgency,
                    vol_used=annual_vol,
                    hours_to_settle=hrs,
                    timestamp=time.time(),
                ))

    # Sort by edge size, descending
    signals.sort(key=lambda s: s.edge_cents, reverse=True)
    return signals


def _calc_urgency(edge_cents: float, abs_move_pct: float, hours_left: float) -> float:
    """
    Calculate urgency score (0.0 - 1.0).
    Higher urgency = bigger edge, bigger price move, less time.

    Factors:
    - Edge size: bigger edge = more urgency (it will get arbed away)
    - Price move: bigger move = more likely Kalshi is stale
    - Time to settle: less time = probabilities more sensitive
    """
    # Edge component (0 to 0.4)
    edge_score = min(0.4, (edge_cents - MIN_EDGE_CENTS) / 25.0 * 0.4)

    # Move component (0 to 0.4)
    move_score = min(0.4, abs_move_pct / 0.03 * 0.4)  # Max at 3% move

    # Time component (0 to 0.2): more urgent when closer to settlement
    if hours_left < 2:
        time_score = 0.2
    elif hours_left < 6:
        time_score = 0.15
    elif hours_left < 12:
        time_score = 0.1
    else:
        time_score = 0.05

    return min(1.0, edge_score + move_score + time_score + URGENCY_LOW)


# =============================================================================
# INTEGRATION FUNCTION (for auto_trade.py)
# =============================================================================

def find_latency_edges(kalshi_client: KalshiClient) -> list[dict]:
    """
    Integration function for auto_trade.py main loop.

    Called from the trading orchestrator to check for latency arb
    opportunities. Returns a list of dicts compatible with the existing
    trade decision flow.

    Args:
        kalshi_client: Authenticated KalshiClient instance

    Returns:
        List of dicts with keys:
            ticker, side, action, edge_cents, fair_value_cents,
            market_price_cents, urgency, btc_price, reason
    """
    logger = logging.getLogger("latency_arb")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)-7s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    # Fetch current price
    snapshot = fetch_prices_parallel(logger)
    if not snapshot:
        return []

    # Fetch volatility
    vol_data = fetch_btc_volatility(logger)
    annual_vol = vol_data["annual_vol"] if vol_data else 0.65  # 65% fallback

    # Scan for edges (no last_price reference in one-shot mode, so use None)
    signals = scan_for_latency_edges(
        price_snapshot=snapshot,
        last_price=None,
        annual_vol=annual_vol,
        client=kalshi_client,
        logger=logger,
    )

    # Convert to dicts for the auto_trade integration
    edges = []
    for sig in signals:
        if not sig.is_actionable:
            continue

        side = "yes" if sig.side == "buy_yes" else "no"
        edges.append({
            "ticker": sig.ticker,
            "title": sig.title,
            "side": side,
            "action": "buy",
            "edge_cents": sig.edge_cents,
            "fair_value_cents": sig.fair_value_cents,
            "market_price_cents": sig.market_price_cents,
            "urgency": sig.urgency,
            "btc_price": sig.btc_price,
            "price_move_pct": sig.price_move_pct,
            "vol_used": sig.vol_used,
            "hours_to_settle": sig.hours_to_settle,
            "reason": (
                f"LATENCY ARB: BTC=${sig.btc_price:,.0f} "
                f"move={sig.price_move_pct:+.1f}% "
                f"edge=+{sig.edge_cents:.1f}c "
                f"urgency={sig.urgency:.0%}"
            ),
            "strategy": "latency_arb",
        })

    return edges


# =============================================================================
# TRADE EXECUTION
# =============================================================================

def execute_signal(
    signal: LatencySignal,
    client: KalshiClient,
    dry_run: bool,
    logger: logging.Logger,
    balance: Optional[float] = None,
) -> dict:
    """
    Execute a single latency arb trade signal.
    Returns a dict recording the trade decision.

    Conservative sizing: $1-2 per trade, limit orders only.
    """
    side = "yes" if signal.side == "buy_yes" else "no"
    price_cents = int(math.ceil(signal.market_price_cents))
    price_cents = max(1, min(99, price_cents))
    cost_per_contract = price_cents / 100.0

    # Size conservatively: $1 default, up to $2
    max_dollars = DEFAULT_TRADE_DOLLARS
    if signal.urgency >= URGENCY_HIGH and signal.edge_cents >= 15:
        max_dollars = MAX_DOLLARS_PER_TRADE  # Scale up for high-conviction

    contracts = max(1, int(max_dollars / cost_per_contract))
    actual_cost = contracts * cost_per_contract

    # Ensure we don't exceed max
    if actual_cost > MAX_DOLLARS_PER_TRADE:
        contracts = max(1, int(MAX_DOLLARS_PER_TRADE / cost_per_contract))
        actual_cost = contracts * cost_per_contract

    result = {
        "ticker": signal.ticker,
        "side": side,
        "action": f"buy_{side}",
        "contracts": contracts,
        "price_cents": price_cents,
        "cost_dollars": round(actual_cost, 2),
        "edge_cents": signal.edge_cents,
        "fair_value_cents": signal.fair_value_cents,
        "urgency": signal.urgency,
        "btc_price": signal.btc_price,
        "price_move_pct": signal.price_move_pct,
        "placed": False,
        "order_id": "",
        "error": "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Balance check
    if balance is not None and actual_cost > balance:
        result["error"] = f"Insufficient balance: need ${actual_cost:.2f}, have ${balance:.2f}"
        logger.warning(
            f"SKIP {signal.ticker}: cost ${actual_cost:.2f} > balance ${balance:.2f}"
        )
        return result

    if dry_run:
        logger.info(
            f"[DRY-RUN] Would BUY {side.upper()} {signal.ticker} "
            f"x{contracts} @ {price_cents}c | "
            f"edge=+{signal.edge_cents:.1f}c urgency={signal.urgency:.0%} | "
            f"cost=${actual_cost:.2f}"
        )
        return result

    # Place live order
    try:
        order_kwargs = {
            "ticker": signal.ticker,
            "side": side,
            "action": "buy",
            "count": contracts,
            "type": "limit",
        }
        if side == "yes":
            order_kwargs["yes_price"] = price_cents
        else:
            order_kwargs["no_price"] = price_cents

        resp = client.place_order(**order_kwargs)
        order_id = resp.get("order", {}).get("order_id", "unknown")
        result["placed"] = True
        result["order_id"] = order_id

        logger.info(
            f"ORDER PLACED: BUY {side.upper()} {signal.ticker} "
            f"x{contracts} @ {price_cents}c | "
            f"edge=+{signal.edge_cents:.1f}c | "
            f"order_id={order_id}"
        )

        if alert_big_edge and signal.edge_cents > 20:
            alert_big_edge(signal.ticker, signal.edge_cents, "LATENCY_ARB")

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"ORDER FAILED: {signal.ticker} -- {e}")
        if alert_bot_error:
            alert_bot_error("latency_arb", str(e))

    return result


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logger(dry_run: bool = True) -> logging.Logger:
    """Create a logger for the latency arb engine."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_dir = config.OUTPUT_DIR
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"latency_arb_{today}.log"

    logger = logging.getLogger("latency_arb")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    mode_label = "DRY-RUN" if dry_run else "LIVE"
    logger.info("=" * 60)
    logger.info(f"LATENCY ARB ENGINE START  [{mode_label}]  env={config.KALSHI_ENV}")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Poll interval: {POLL_INTERVAL_SECONDS}s")
    logger.info(f"Min edge: {MIN_EDGE_CENTS}c | Min move: {MIN_PRICE_MOVE_PCT:.1%}")
    logger.info(f"Max per trade: ${MAX_DOLLARS_PER_TRADE} | Daily cap: {DAILY_LOSS_CAP_PCT:.0%}")
    logger.info("=" * 60)

    return logger


# =============================================================================
# DISPLAY
# =============================================================================

def display_signals(signals: list[LatencySignal]):
    """Pretty-print latency arb signals to console."""
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
    except (ImportError, OSError):
        # Fallback to plain print
        for s in signals:
            print(
                f"  {s.side:8s} {s.ticker:30s} "
                f"edge=+{s.edge_cents:.1f}c  fair={s.fair_value_cents:.0f}c  "
                f"mkt={s.market_price_cents:.0f}c  urgency={s.urgency:.0%}"
            )
        return

    if not signals:
        console.print("[yellow]No latency arb signals detected.[/yellow]")
        return

    table = Table(
        title="Latency Arb Signals -- BTC Price vs Kalshi",
        show_lines=False,
    )
    table.add_column("Ticker", style="cyan", width=28)
    table.add_column("Bucket", justify="right", style="white", width=22)
    table.add_column("Action", style="bold", width=10)
    table.add_column("Mkt", justify="right", width=7)
    table.add_column("Fair", justify="right", style="yellow", width=7)
    table.add_column("Edge", justify="right", style="bold green", width=7)
    table.add_column("BTC Move", justify="right", width=8)
    table.add_column("Urgency", justify="right", width=8)
    table.add_column("Settle", justify="right", style="dim", width=7)

    for s in signals:
        action_color = "green" if s.side == "buy_yes" else "red"
        action_text = "BUY YES" if s.side == "buy_yes" else "BUY NO"

        if s.bucket_high >= 1e8:
            bucket_text = f">= ${s.bucket_low:,.0f}"
        elif s.bucket_low <= 0:
            bucket_text = f"<= ${s.bucket_high:,.0f}"
        else:
            bucket_text = f"${s.bucket_low:,.0f}-${s.bucket_high:,.0f}"

        urgency_color = (
            "bold red" if s.urgency >= URGENCY_HIGH
            else "yellow" if s.urgency >= URGENCY_MEDIUM
            else "dim"
        )

        table.add_row(
            s.ticker,
            bucket_text,
            f"[{action_color}]{action_text}[/{action_color}]",
            f"{s.market_price_cents:.0f}c",
            f"{s.fair_value_cents:.0f}c",
            f"+{s.edge_cents:.0f}c",
            f"{s.price_move_pct:+.1f}%",
            f"[{urgency_color}]{s.urgency:.0%}[/{urgency_color}]",
            f"{s.hours_to_settle:.1f}h",
        )

    console.print(table)
    top = signals[0]
    console.print(
        f"\n[dim]Found {len(signals)} signals (min {MIN_EDGE_CENTS}c edge). "
        f"BTC=${top.btc_price:,.0f}. "
        f"Sources: Coinbase + Binance + CoinGecko (parallel). "
        f"Vol: {top.vol_used:.0%} annualized.[/dim]"
    )


# =============================================================================
# MAIN LOOP (standalone monitoring + trading)
# =============================================================================

def run_latency_arb(dry_run: bool = True):
    """
    Main monitoring loop for the latency arb engine.

    Polls BTC price every 30 seconds. When a significant move is detected,
    scans KXBTC markets for stale pricing and optionally places trades.
    """
    logger = setup_logger(dry_run=dry_run)

    # Initialize Kalshi client
    try:
        client = KalshiClient()
        logger.info(f"Kalshi client initialized (env={config.KALSHI_ENV})")
    except Exception as e:
        logger.critical(f"Failed to initialize Kalshi client: {e}")
        logger.critical(traceback.format_exc())
        if alert_bot_error:
            alert_bot_error("latency_arb", str(e))
        return

    # Check balance
    balance = None
    try:
        bal = client.get_balance()
        balance = bal.get("balance", 0) / 100.0
        logger.info(f"Account balance: ${balance:.2f}")
    except Exception as e:
        logger.warning(f"Could not fetch balance: {e} (will skip balance checks)")

    if balance is not None and balance < 1.0:
        logger.critical(f"Account balance too low: ${balance:.2f}")
        return

    # Fetch initial volatility (only need this once per session)
    vol_data = fetch_btc_volatility(logger)
    annual_vol = vol_data["annual_vol"] if vol_data else 0.65
    logger.info(f"Using annualized vol: {annual_vol:.0%}")

    # State tracking
    last_price: Optional[float] = None
    trades_this_session = 0
    total_deployed = 0.0
    daily_cap = (balance or config.ACCOUNT_BALANCE) * DAILY_LOSS_CAP_PCT
    all_trades: list[dict] = []
    cycle_count = 0

    logger.info(f"Daily loss cap: ${daily_cap:.2f}")
    logger.info(f"Starting monitoring loop (Ctrl+C to stop)...")
    logger.info("")

    try:
        while True:
            cycle_count += 1
            cycle_start = time.time()

            # --- 1. Fetch real-time prices ---
            snapshot = fetch_prices_parallel(logger)
            if not snapshot:
                logger.warning("Price fetch failed, retrying next cycle")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            current_price = snapshot.consensus

            # Log price on every cycle
            sources_str = []
            if snapshot.coinbase:
                sources_str.append(f"CB=${snapshot.coinbase:,.0f}")
            if snapshot.binance:
                sources_str.append(f"BN=${snapshot.binance:,.0f}")
            if snapshot.coingecko:
                sources_str.append(f"CG=${snapshot.coingecko:,.0f}")

            logger.info(
                f"[Cycle {cycle_count}] BTC=${current_price:,.0f} "
                f"({' '.join(sources_str)}) "
                f"spread={snapshot.max_spread_pct:.2f}%"
            )

            # Warn on large source disagreement
            if snapshot.max_spread_pct > 1.0:
                logger.warning(
                    f"Large price spread between sources: {snapshot.max_spread_pct:.1f}%"
                )

            # --- 2. Check for price move ---
            if last_price is not None:
                move_pct = abs(current_price - last_price) / last_price
                move_dollar = current_price - last_price

                if move_pct >= MIN_PRICE_MOVE_PCT:
                    direction = "UP" if move_dollar > 0 else "DOWN"
                    logger.info(
                        f"*** PRICE MOVE DETECTED: {direction} {move_pct:.2%} "
                        f"(${move_dollar:+,.0f}) -- scanning for stale Kalshi prices ***"
                    )

            # --- 3. Scan for latency edges ---
            signals = scan_for_latency_edges(
                price_snapshot=snapshot,
                last_price=last_price,
                annual_vol=annual_vol,
                client=client,
                logger=logger,
            )

            actionable = [s for s in signals if s.is_actionable]

            if actionable:
                logger.info(f"Found {len(actionable)} actionable signals!")
                display_signals(actionable)

                # --- 4. Execute trades (if within limits) ---
                for signal in actionable:
                    # Check session trade cap
                    if trades_this_session >= MAX_TRADES_PER_SESSION:
                        logger.info(
                            f"Session trade cap reached ({MAX_TRADES_PER_SESSION}), "
                            "skipping remaining signals"
                        )
                        break

                    # Check daily loss cap
                    if total_deployed >= daily_cap:
                        logger.info(
                            f"Daily deployment cap reached "
                            f"(${total_deployed:.2f} >= ${daily_cap:.2f}), "
                            "skipping remaining signals"
                        )
                        break

                    trade_result = execute_signal(
                        signal=signal,
                        client=client,
                        dry_run=dry_run,
                        logger=logger,
                        balance=balance,
                    )
                    all_trades.append(trade_result)

                    if trade_result.get("placed") or (dry_run and not trade_result.get("error")):
                        trades_this_session += 1
                        total_deployed += trade_result["cost_dollars"]
            else:
                logger.debug("No actionable signals this cycle")

            # Update last price
            last_price = current_price

            # --- 5. Wait for next cycle ---
            elapsed = time.time() - cycle_start
            sleep_time = max(1, POLL_INTERVAL_SECONDS - elapsed)
            logger.debug(
                f"Cycle {cycle_count} done in {elapsed:.1f}s. "
                f"Trades: {trades_this_session}/{MAX_TRADES_PER_SESSION}. "
                f"Deployed: ${total_deployed:.2f}/${daily_cap:.2f}. "
                f"Next scan in {sleep_time:.0f}s."
            )
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("\nShutting down latency arb engine (Ctrl+C)")

    # --- Session summary ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("LATENCY ARB SESSION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Cycles run:       {cycle_count}")
    logger.info(f"Trades executed:  {trades_this_session}")
    logger.info(f"Total deployed:   ${total_deployed:.2f}")
    logger.info(f"Mode:             {'DRY-RUN' if dry_run else 'LIVE'}")

    if all_trades:
        logger.info("")
        logger.info("Trades:")
        for t in all_trades:
            status = "PLACED" if t.get("placed") else ("ERROR" if t.get("error") else "DRY-RUN")
            logger.info(
                f"  [{status}] {t['action']:8s} {t['ticker']:30s} "
                f"x{t['contracts']} @ {t['price_cents']}c | "
                f"edge=+{t['edge_cents']:.1f}c cost=${t['cost_dollars']:.2f}"
            )

    # Save results
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    json_path = config.OUTPUT_DIR / f"latency_arb_{today}.json"
    try:
        with open(json_path, "w") as f:
            json.dump({
                "date": today,
                "mode": "dry_run" if dry_run else "live",
                "env": config.KALSHI_ENV,
                "cycles": cycle_count,
                "trades": trades_this_session,
                "total_deployed": total_deployed,
                "trade_log": all_trades,
            }, f, indent=2)
        logger.info(f"Results saved to: {json_path}")
    except Exception as e:
        logger.warning(f"Failed to save results: {e}")

    logger.info("=" * 60)
    logger.info("LATENCY ARB ENGINE STOPPED")
    logger.info("=" * 60)


# =============================================================================
# STANDALONE ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Latency Arbitrage Engine for Kalshi KXBTC markets. "
            "Monitors real-time BTC prices from Coinbase/Binance/CoinGecko "
            "and trades mispriced Kalshi buckets before the market catches up."
        )
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Place real orders (default: dry-run)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Monitor and log signals without placing orders (default)",
    )
    parser.add_argument(
        "--interval", type=int, default=POLL_INTERVAL_SECONDS,
        help=f"Poll interval in seconds (default: {POLL_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--min-edge", type=float, default=MIN_EDGE_CENTS,
        help=f"Minimum edge in cents to signal (default: {MIN_EDGE_CENTS})",
    )
    parser.add_argument(
        "--no-confirm", action="store_true",
        help="Skip confirmation prompt for live mode",
    )
    args = parser.parse_args()

    # Apply overrides
    if args.interval != POLL_INTERVAL_SECONDS:
        POLL_INTERVAL_SECONDS = args.interval
    if args.min_edge != MIN_EDGE_CENTS:
        MIN_EDGE_CENTS = args.min_edge

    is_live = args.live

    if is_live and not args.no_confirm:
        print("\n*** LIVE MODE -- Real orders will be placed ***")
        print(f"    Environment: {config.KALSHI_ENV}")
        print(f"    Max per trade: ${MAX_DOLLARS_PER_TRADE}")
        print(f"    Daily cap: {DAILY_LOSS_CAP_PCT:.0%} of account")
        print(f"    Poll interval: {POLL_INTERVAL_SECONDS}s")
        confirm = input("    Type 'YES' to continue: ").strip()
        if confirm != "YES":
            print("Aborted.")
            sys.exit(0)

    run_latency_arb(dry_run=not is_live)
