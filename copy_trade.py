"""
copy_trade.py -- Copy-trade module for the Kalshi trading bot.

Monitors top Polymarket traders (especially 0x8dxd) and mirrors their
crypto trades to equivalent Kalshi markets.

APPROACH:
  1. Fetch recent trades from Polymarket's public REST APIs (no web3 needed).
  2. Match Polymarket crypto markets to Kalshi equivalents (KXBTC/KXETH/KXSOL).
  3. Generate trade signals with confidence scoring.
  4. Track performance per trader for accuracy/PnL analytics.

POLYMARKET DATA SOURCES:
  - Activity API: https://data-api.polymarket.com/activity
  - Positions API: https://data-api.polymarket.com/positions
  - Market Info API: https://gamma-api.polymarket.com/markets

SIZING: Conservative -- $1-2 per copy trade regardless of whale sizing.

Usage:
    python copy_trade.py --watch           # Poll every 60s, show new trades
    python copy_trade.py --positions       # Show current positions of tracked wallets
    python copy_trade.py --signals         # Generate copy signals for Kalshi
    python copy_trade.py --stats           # Show copy-trade performance stats
    python copy_trade.py --add-wallet ADDR NAME  # Add a wallet to track
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import requests
from rich.console import Console
from rich.table import Table
from rich.live import Live

import config
from kalshi_client import KalshiClient

try:
    from alerts import alert_big_edge, alert_bot_error
except (ImportError, OSError):
    alert_big_edge = alert_bot_error = None

console = Console()
logger = logging.getLogger("copy_trade")

# =============================================================================
# OUTPUT PATHS
# =============================================================================
OUTPUT_DIR = config.OUTPUT_DIR
SEEN_FILE = OUTPUT_DIR / "copytrade_seen.json"
MATCHES_FILE = OUTPUT_DIR / "copytrade_matches.json"
LOG_FILE = OUTPUT_DIR / "copytrade_log.json"
WALLETS_FILE = OUTPUT_DIR / "copytrade_wallets.json"

# =============================================================================
# TRACKED WALLETS
# =============================================================================
TRACKED_WALLETS = {
    "0x8dxd": {
        "address": "0x63ce342161250d705dc0b16df89036c8e5f9ba9a",
        "weight": 1.0,  # How much to trust this trader (for AutoResearch)
        "focus": ["crypto"],  # What categories to copy
    },
}

# =============================================================================
# POLYMARKET API ENDPOINTS
# =============================================================================
POLY_DATA_API = "https://data-api.polymarket.com"
POLY_GAMMA_API = "https://gamma-api.polymarket.com"

# =============================================================================
# CRYPTO KEYWORD -> KALSHI SERIES MAPPING
# =============================================================================
CRYPTO_KEYWORDS = {
    "bitcoin": "KXBTC",
    "btc": "KXBTC",
    "ethereum": "KXETH",
    "ether": "KXETH",
    "eth": "KXETH",
    "solana": "KXSOL",
    "sol": "KXSOL",
}

# Conservative sizing for copy trades
COPY_TRADE_MAX_DOLLARS = 2.0
COPY_TRADE_MIN_DOLLARS = 1.0
COPY_TRADE_MIN_CONFIDENCE = 0.8

# How far back to look for new trades (minutes)
TRADE_LOOKBACK_MINUTES = 30

# Poll interval for --watch mode (seconds)
WATCH_POLL_INTERVAL = 60


# =============================================================================
# POLYMARKET DATA FETCHING
# =============================================================================

def fetch_recent_trades(wallet_address: str, limit: int = 20) -> list[dict]:
    """
    Fetch recent trades from Polymarket Data API.

    GET https://data-api.polymarket.com/activity?user={wallet}&type=TRADE&limit={limit}

    Returns list of dicts with: trade_id, market_question, side, amount_usd,
    price, timestamp, condition_id
    """
    try:
        resp = requests.get(
            f"{POLY_DATA_API}/activity",
            params={
                "user": wallet_address,
                "type": "TRADE",
                "limit": limit,
            },
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Polymarket activity fetch failed: {e}")
        return []
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Polymarket activity parse error: {e}")
        return []

    trades = []
    items = raw if isinstance(raw, list) else raw.get("data", raw.get("activities", []))
    for item in items:
        try:
            # Polymarket amounts are in micros (divide by 1e6 for USD)
            raw_amount = float(item.get("amount", 0) or item.get("size", 0) or 0)
            # Heuristic: if amount > 10_000 it's likely in micros
            amount_usd = raw_amount / 1e6 if raw_amount > 10_000 else raw_amount

            trade = {
                "trade_id": str(item.get("id", item.get("tradeId", ""))),
                "market_question": item.get("title", item.get("question", item.get("market", ""))),
                "side": item.get("side", item.get("outcome", "")).upper(),
                "amount_usd": round(amount_usd, 2),
                "price": float(item.get("price", 0) or 0),
                "timestamp": item.get("timestamp", item.get("createdAt", "")),
                "condition_id": item.get("conditionId", item.get("condition_id", "")),
            }

            # Normalize side to YES/NO
            if trade["side"] not in ("YES", "NO"):
                if trade["side"] in ("1", "TRUE", "BUY"):
                    trade["side"] = "YES"
                elif trade["side"] in ("0", "FALSE", "SELL"):
                    trade["side"] = "NO"

            trades.append(trade)
        except (KeyError, ValueError, TypeError) as e:
            logger.debug(f"Skipping malformed trade entry: {e}")
            continue

    return trades


def fetch_positions(wallet_address: str) -> list[dict]:
    """
    Fetch current open positions from Polymarket Data API.

    GET https://data-api.polymarket.com/positions?user={wallet}&sizeThreshold=1.0&limit=100

    Returns list of dicts with: condition_id, question, side, size_usd, current_price
    """
    try:
        resp = requests.get(
            f"{POLY_DATA_API}/positions",
            params={
                "user": wallet_address,
                "sizeThreshold": "1.0",
                "limit": 100,
            },
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Polymarket positions fetch failed: {e}")
        return []
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Polymarket positions parse error: {e}")
        return []

    positions = []
    items = raw if isinstance(raw, list) else raw.get("data", raw.get("positions", []))
    for item in items:
        try:
            raw_size = float(item.get("size", 0) or item.get("amount", 0) or 0)
            size_usd = raw_size / 1e6 if raw_size > 10_000 else raw_size

            pos = {
                "condition_id": item.get("conditionId", item.get("condition_id", "")),
                "question": item.get("title", item.get("question", item.get("market", ""))),
                "side": item.get("side", item.get("outcome", "")).upper(),
                "size_usd": round(size_usd, 2),
                "current_price": float(item.get("curPrice", item.get("price", 0)) or 0),
            }
            positions.append(pos)
        except (KeyError, ValueError, TypeError) as e:
            logger.debug(f"Skipping malformed position entry: {e}")
            continue

    return positions


def fetch_market_info(condition_id: str) -> Optional[dict]:
    """
    Get market metadata from Polymarket Gamma API.

    GET https://gamma-api.polymarket.com/markets?condition_id={condition_id}

    Returns: question, outcomes, end_date, slug (or None on failure)
    """
    if not condition_id:
        return None

    try:
        resp = requests.get(
            f"{POLY_GAMMA_API}/markets",
            params={"condition_id": condition_id},
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Polymarket market info fetch failed for {condition_id}: {e}")
        return None
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Polymarket market info parse error for {condition_id}: {e}")
        return None

    # API may return a list or a single object
    if isinstance(raw, list):
        if not raw:
            return None
        market = raw[0]
    else:
        market = raw

    return {
        "question": market.get("question", market.get("title", "")),
        "outcomes": market.get("outcomes", []),
        "end_date": market.get("endDate", market.get("end_date", "")),
        "slug": market.get("slug", ""),
    }


# =============================================================================
# MARKET MATCHING: POLYMARKET -> KALSHI
# =============================================================================

def _load_match_cache() -> dict:
    """Load cached market matches."""
    if MATCHES_FILE.exists():
        try:
            return json.loads(MATCHES_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_match_cache(cache: dict):
    """Save market matches to cache."""
    try:
        MATCHES_FILE.write_text(json.dumps(cache, indent=2))
    except OSError as e:
        logger.error(f"Failed to save match cache: {e}")


def _extract_price_threshold(question: str) -> Optional[float]:
    """
    Extract a price threshold from a Polymarket market question.

    Examples:
        "Will Bitcoin be above $87,500 at 5pm ET?" -> 87500.0
        "ETH above $2,100?" -> 2100.0
        "Will the price of SOL be above 140.50?" -> 140.50
    """
    # Match dollar amounts with optional commas and decimals
    patterns = [
        r'\$\s*([\d,]+(?:\.\d+)?)',           # $87,500 or $87500.50
        r'above\s+([\d,]+(?:\.\d+)?)',         # above 87500
        r'below\s+([\d,]+(?:\.\d+)?)',         # below 87500
        r'between\s+([\d,]+(?:\.\d+)?)',       # between 87000
        r'reach\s+\$?([\d,]+(?:\.\d+)?)',      # reach $87500
        r'hit\s+\$?([\d,]+(?:\.\d+)?)',        # hit $87500
        r'exceed\s+\$?([\d,]+(?:\.\d+)?)',     # exceed $87500
    ]
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                continue
    return None


def _identify_crypto_asset(question: str) -> Optional[str]:
    """Identify which crypto asset a question is about. Returns KXBTC/KXETH/KXSOL or None."""
    q_lower = question.lower()
    for keyword, series in CRYPTO_KEYWORDS.items():
        if keyword in q_lower:
            return series
    return None


def match_to_kalshi(poly_question: str, kalshi_client: KalshiClient) -> dict:
    """
    Match a Polymarket market to a Kalshi market.

    For crypto markets, uses keyword matching: "Bitcoin" -> KXBTC, etc.
    For price bucket markets, extracts the price threshold and finds the
    matching Kalshi bucket.

    Returns:
        dict with: kalshi_ticker, confidence (0-1), match_reason
        Returns empty kalshi_ticker and 0 confidence if no match found.
    """
    result = {
        "kalshi_ticker": "",
        "confidence": 0.0,
        "match_reason": "no match",
    }

    # Check match cache first
    cache = _load_match_cache()
    cache_key = poly_question.strip().lower()[:200]
    if cache_key in cache:
        return cache[cache_key]

    # Step 1: Identify the crypto asset
    kalshi_series = _identify_crypto_asset(poly_question)
    if not kalshi_series:
        result["match_reason"] = "not a crypto market or unrecognized asset"
        return result

    # Step 2: Extract price threshold
    threshold = _extract_price_threshold(poly_question)

    # Step 3: Find matching Kalshi market
    try:
        resp = kalshi_client.get_markets(series_ticker=kalshi_series, limit=100, status="open")
        kalshi_markets = resp.get("markets", [])
    except Exception as e:
        logger.error(f"Failed to fetch Kalshi {kalshi_series} markets: {e}")
        result["match_reason"] = f"Kalshi API error: {e}"
        return result

    if not kalshi_markets:
        result["match_reason"] = f"no open {kalshi_series} markets found"
        return result

    best_match = None
    best_confidence = 0.0
    best_reason = ""

    q_lower = poly_question.lower()
    is_above = any(w in q_lower for w in ["above", "over", "higher", "exceed", "reach", "hit"])
    is_below = any(w in q_lower for w in ["below", "under", "lower", "drop", "fall"])

    for km in kalshi_markets:
        ticker = km.get("ticker", "")
        title = km.get("title", "").lower()

        # If we have a threshold, try to match to the right bucket
        if threshold is not None:
            # Check if the Kalshi market title references a similar threshold
            kalshi_threshold = _extract_price_threshold(km.get("title", ""))

            if kalshi_threshold is not None:
                # Close threshold match (within 1% of each other)
                pct_diff = abs(kalshi_threshold - threshold) / max(threshold, 1)
                if pct_diff < 0.01:
                    confidence = 0.95
                    reason = f"exact threshold match: ${threshold:,.0f} -> {ticker}"
                elif pct_diff < 0.03:
                    confidence = 0.85
                    reason = f"close threshold match: ${threshold:,.0f} ~= ${kalshi_threshold:,.0f} -> {ticker}"
                elif pct_diff < 0.05:
                    confidence = 0.70
                    reason = f"approximate threshold: ${threshold:,.0f} ~= ${kalshi_threshold:,.0f} -> {ticker}"
                else:
                    continue

                if confidence > best_confidence:
                    best_match = ticker
                    best_confidence = confidence
                    best_reason = reason

            # Check "between X and Y" style Kalshi markets
            between = re.search(r'between\s+\$?([\d,]+)\s+and\s+\$?([\d,]+)', title)
            if between:
                low = float(between.group(1).replace(",", ""))
                high = float(between.group(2).replace(",", ""))
                if low <= threshold <= high:
                    confidence = 0.80
                    reason = f"threshold ${threshold:,.0f} falls in bucket [{low:,.0f}-{high:,.0f}] -> {ticker}"
                    if confidence > best_confidence:
                        best_match = ticker
                        best_confidence = confidence
                        best_reason = reason

        else:
            # No threshold extracted -- try broader title similarity
            # Check if both are about the same direction
            kalshi_is_above = any(w in title for w in ["above", "or above", "higher"])
            kalshi_is_below = any(w in title for w in ["below", "or below", "lower"])

            if (is_above and kalshi_is_above) or (is_below and kalshi_is_below):
                confidence = 0.60
                reason = f"directional match ({kalshi_series}) -> {ticker}"
                if confidence > best_confidence:
                    best_match = ticker
                    best_confidence = confidence
                    best_reason = reason

    if best_match:
        result["kalshi_ticker"] = best_match
        result["confidence"] = best_confidence
        result["match_reason"] = best_reason
    else:
        result["match_reason"] = (
            f"crypto asset identified ({kalshi_series}) but no matching Kalshi bucket "
            f"for threshold={threshold}"
        )

    # Cache the result
    cache[cache_key] = result
    _save_match_cache(cache)

    return result


# =============================================================================
# SEEN TRADES TRACKING (deduplication)
# =============================================================================

def _load_seen_trades() -> set:
    """Load set of previously seen trade IDs."""
    if SEEN_FILE.exists():
        try:
            data = json.loads(SEEN_FILE.read_text())
            return set(data)
        except (json.JSONDecodeError, OSError):
            pass
    return set()


def _save_seen_trades(seen: set):
    """Save seen trade IDs."""
    try:
        # Keep only last 5000 entries to prevent unbounded growth
        seen_list = sorted(seen)[-5000:]
        SEEN_FILE.write_text(json.dumps(seen_list, indent=2))
    except OSError as e:
        logger.error(f"Failed to save seen trades: {e}")


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def generate_copy_signals(kalshi_client: KalshiClient) -> list[dict]:
    """
    Main function: generate copy-trade signals.

    For each tracked wallet:
      1. Fetch recent trades (last 30 min)
      2. Filter to trades we haven't seen before
      3. Match each to a Kalshi market
      4. For high-confidence matches (>0.8), generate a trade signal

    Returns list of dicts:
        {ticker, side, confidence, source_trader, source_trade_id,
         amount_suggested, reason}
    """
    wallets = _get_all_wallets()
    seen = _load_seen_trades()
    signals = []
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=TRADE_LOOKBACK_MINUTES)

    for name, wallet_cfg in wallets.items():
        address = wallet_cfg["address"]
        focus_categories = wallet_cfg.get("focus", ["crypto"])
        weight = wallet_cfg.get("weight", 1.0)

        logger.info(f"Checking wallet: {name} ({address[:10]}...)")

        trades = fetch_recent_trades(address, limit=20)
        if not trades:
            logger.debug(f"  No recent trades for {name}")
            continue

        for trade in trades:
            trade_id = trade["trade_id"]

            # Skip already-seen trades
            if trade_id in seen:
                continue

            # Mark as seen
            seen.add(trade_id)

            # Check timestamp freshness
            ts_str = trade.get("timestamp", "")
            if ts_str:
                try:
                    # Handle various timestamp formats
                    if isinstance(ts_str, (int, float)):
                        trade_time = datetime.fromtimestamp(ts_str, tz=timezone.utc)
                    elif "T" in str(ts_str):
                        ts_clean = str(ts_str).replace("Z", "+00:00")
                        trade_time = datetime.fromisoformat(ts_clean)
                    else:
                        trade_time = cutoff  # Can't parse, treat as recent
                    if trade_time < cutoff:
                        continue
                except (ValueError, TypeError, OverflowError):
                    pass  # Can't parse timestamp, process anyway

            # Check if this is a crypto trade (our focus)
            question = trade.get("market_question", "")
            if "crypto" in focus_categories:
                asset = _identify_crypto_asset(question)
                if not asset:
                    logger.debug(f"  Skipping non-crypto trade: {question[:60]}")
                    continue
            else:
                continue

            # Match to Kalshi
            match = match_to_kalshi(question, kalshi_client)
            confidence = match["confidence"] * weight

            if confidence < COPY_TRADE_MIN_CONFIDENCE:
                logger.info(
                    f"  Low confidence match ({confidence:.2f}): "
                    f"{question[:60]} -> {match['match_reason']}"
                )
                continue

            # Map Polymarket side to Kalshi side
            poly_side = trade.get("side", "YES")
            kalshi_side = "yes" if poly_side == "YES" else "no"

            # Conservative sizing: $1-2 regardless of whale size
            suggested_amount = min(
                COPY_TRADE_MAX_DOLLARS,
                max(COPY_TRADE_MIN_DOLLARS, trade["amount_usd"] * 0.01),
            )

            signal = {
                "ticker": match["kalshi_ticker"],
                "side": kalshi_side,
                "confidence": round(confidence, 3),
                "source_trader": name,
                "source_trade_id": trade_id,
                "amount_suggested": round(suggested_amount, 2),
                "reason": (
                    f"Copy {name} ({confidence:.0%} confidence): "
                    f"{poly_side} on '{question[:80]}' "
                    f"-> Kalshi {match['kalshi_ticker']} | {match['match_reason']}"
                ),
                "poly_question": question,
                "poly_price": trade.get("price", 0),
                "poly_amount_usd": trade.get("amount_usd", 0),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            signals.append(signal)

            logger.info(
                f"  SIGNAL: {name} {poly_side} '{question[:50]}' "
                f"-> {match['kalshi_ticker']} ({confidence:.0%})"
            )

            # Alert on big whale moves
            if alert_big_edge and trade.get("amount_usd", 0) > 500:
                alert_big_edge(
                    match["kalshi_ticker"],
                    confidence * 100,
                    f"COPY:{name}",
                )

    _save_seen_trades(seen)
    return signals


def get_copy_signals(kalshi_client: KalshiClient) -> list[dict]:
    """
    Integration function for auto_trade.py.

    Returns actionable signals for the trading loop.
    Each signal has: ticker, side, confidence, source_trader,
    source_trade_id, amount_suggested, reason
    """
    try:
        signals = generate_copy_signals(kalshi_client)
        # Log all signals
        tracker = CopyTradeTracker()
        for s in signals:
            tracker.log_signal(s)
        return signals
    except Exception as e:
        logger.error(f"Copy signal generation failed: {e}")
        if alert_bot_error:
            alert_bot_error("copy_trade", str(e))
        return []


# =============================================================================
# PERFORMANCE TRACKING
# =============================================================================

class CopyTradeTracker:
    """Track copy-trade performance: signals, trades, and outcomes."""

    def __init__(self):
        self.log_path = LOG_FILE
        self._ensure_log()

    def _ensure_log(self):
        """Create log file if it doesn't exist."""
        if not self.log_path.exists():
            self.log_path.write_text("[]")

    def _read_log(self) -> list[dict]:
        """Read the log file."""
        try:
            return json.loads(self.log_path.read_text())
        except (json.JSONDecodeError, OSError):
            return []

    def _write_log(self, entries: list[dict]):
        """Write to the log file."""
        try:
            self.log_path.write_text(json.dumps(entries, indent=2))
        except OSError as e:
            logger.error(f"Failed to write copy trade log: {e}")

    def log_signal(self, signal: dict):
        """Log a generated copy signal."""
        entries = self._read_log()
        entry = {
            "type": "signal",
            "timestamp": signal.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "ticker": signal.get("ticker", ""),
            "side": signal.get("side", ""),
            "confidence": signal.get("confidence", 0),
            "source_trader": signal.get("source_trader", ""),
            "source_trade_id": signal.get("source_trade_id", ""),
            "amount_suggested": signal.get("amount_suggested", 0),
            "reason": signal.get("reason", ""),
            "traded": False,
            "outcome": None,
            "pnl": 0.0,
        }
        entries.append(entry)
        # Keep last 1000 entries
        entries = entries[-1000:]
        self._write_log(entries)

    def mark_traded(self, source_trade_id: str, order_id: str = ""):
        """Mark a signal as having been traded."""
        entries = self._read_log()
        for entry in entries:
            if entry.get("source_trade_id") == source_trade_id:
                entry["traded"] = True
                entry["order_id"] = order_id
                entry["traded_at"] = datetime.now(timezone.utc).isoformat()
                break
        self._write_log(entries)

    def mark_outcome(self, source_trade_id: str, won: bool, pnl: float):
        """Record the outcome of a copy trade."""
        entries = self._read_log()
        for entry in entries:
            if entry.get("source_trade_id") == source_trade_id:
                entry["outcome"] = "won" if won else "lost"
                entry["pnl"] = round(pnl, 2)
                entry["settled_at"] = datetime.now(timezone.utc).isoformat()
                break
        self._write_log(entries)

    def get_trader_stats(self) -> dict:
        """
        Calculate performance stats per trader.

        Returns dict keyed by trader name with:
            win_rate, total_trades, total_pnl, signals_generated,
            signals_traded, avg_confidence
        """
        entries = self._read_log()
        stats = {}

        for entry in entries:
            trader = entry.get("source_trader", "unknown")
            if trader not in stats:
                stats[trader] = {
                    "signals_generated": 0,
                    "signals_traded": 0,
                    "wins": 0,
                    "losses": 0,
                    "total_trades": 0,
                    "total_pnl": 0.0,
                    "win_rate": 0.0,
                    "confidences": [],
                    "avg_confidence": 0.0,
                }

            s = stats[trader]
            s["signals_generated"] += 1
            s["confidences"].append(entry.get("confidence", 0))

            if entry.get("traded"):
                s["signals_traded"] += 1

            if entry.get("outcome") is not None:
                s["total_trades"] += 1
                s["total_pnl"] += entry.get("pnl", 0)
                if entry["outcome"] == "won":
                    s["wins"] += 1
                else:
                    s["losses"] += 1

        # Calculate derived metrics
        for trader, s in stats.items():
            if s["total_trades"] > 0:
                s["win_rate"] = round(s["wins"] / s["total_trades"], 3)
            s["total_pnl"] = round(s["total_pnl"], 2)
            if s["confidences"]:
                s["avg_confidence"] = round(
                    sum(s["confidences"]) / len(s["confidences"]), 3
                )
            del s["confidences"]  # Don't include raw list in output

        return stats


# =============================================================================
# WALLET MANAGEMENT
# =============================================================================

def _get_all_wallets() -> dict:
    """Get all tracked wallets (built-in + user-added)."""
    wallets = dict(TRACKED_WALLETS)

    # Load user-added wallets
    if WALLETS_FILE.exists():
        try:
            user_wallets = json.loads(WALLETS_FILE.read_text())
            wallets.update(user_wallets)
        except (json.JSONDecodeError, OSError):
            pass

    return wallets


def add_wallet(address: str, name: str, focus: list[str] = None, weight: float = 0.5):
    """Add a new wallet to track."""
    if focus is None:
        focus = ["crypto"]

    # Load existing user wallets
    user_wallets = {}
    if WALLETS_FILE.exists():
        try:
            user_wallets = json.loads(WALLETS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    user_wallets[name] = {
        "address": address.lower(),
        "weight": weight,
        "focus": focus,
    }

    try:
        WALLETS_FILE.write_text(json.dumps(user_wallets, indent=2))
        console.print(f"[green]Added wallet: {name} ({address[:10]}...)[/green]")
    except OSError as e:
        console.print(f"[red]Failed to save wallet: {e}[/red]")


# =============================================================================
# CLI DISPLAY
# =============================================================================

def display_trades(trades: list[dict], trader_name: str):
    """Display recent trades in a table."""
    if not trades:
        console.print(f"[yellow]No recent trades for {trader_name}[/yellow]")
        return

    table = Table(title=f"Recent Trades: {trader_name}", show_lines=False)
    table.add_column("Time", style="dim", width=20)
    table.add_column("Market", style="cyan", width=50)
    table.add_column("Side", style="bold", width=6)
    table.add_column("Amount", justify="right", style="green", width=12)
    table.add_column("Price", justify="right", width=8)

    for t in trades:
        side_color = "green" if t["side"] == "YES" else "red"
        ts = t.get("timestamp", "")
        if isinstance(ts, str) and len(ts) > 19:
            ts = ts[:19]

        table.add_row(
            ts,
            t["market_question"][:50],
            f"[{side_color}]{t['side']}[/{side_color}]",
            f"${t['amount_usd']:,.2f}",
            f"{t['price']:.2f}" if t["price"] else "-",
        )

    console.print(table)


def display_positions(positions: list[dict], trader_name: str):
    """Display current positions in a table."""
    if not positions:
        console.print(f"[yellow]No open positions for {trader_name}[/yellow]")
        return

    table = Table(title=f"Open Positions: {trader_name}", show_lines=False)
    table.add_column("Market", style="cyan", width=50)
    table.add_column("Side", style="bold", width=6)
    table.add_column("Size", justify="right", style="green", width=12)
    table.add_column("Price", justify="right", width=8)

    for p in positions:
        side_color = "green" if p["side"] == "YES" else "red"
        table.add_row(
            p["question"][:50],
            f"[{side_color}]{p['side']}[/{side_color}]",
            f"${p['size_usd']:,.2f}",
            f"{p['current_price']:.2f}" if p["current_price"] else "-",
        )

    console.print(table)


def display_signals(signals: list[dict]):
    """Display generated copy signals."""
    if not signals:
        console.print("[yellow]No copy signals generated.[/yellow]")
        return

    table = Table(title="Copy-Trade Signals", show_lines=False)
    table.add_column("Kalshi Ticker", style="cyan", width=28)
    table.add_column("Side", style="bold", width=6)
    table.add_column("Confidence", justify="right", width=12)
    table.add_column("Source", style="dim", width=12)
    table.add_column("Suggested $", justify="right", style="green", width=12)
    table.add_column("Reason", width=50)

    for s in signals:
        side_color = "green" if s["side"] == "yes" else "red"
        conf_color = "green" if s["confidence"] >= 0.9 else "yellow"
        table.add_row(
            s["ticker"],
            f"[{side_color}]{s['side'].upper()}[/{side_color}]",
            f"[{conf_color}]{s['confidence']:.0%}[/{conf_color}]",
            s["source_trader"],
            f"${s['amount_suggested']:.2f}",
            s["reason"][:50],
        )

    console.print(table)


def display_stats():
    """Display copy-trade performance stats."""
    tracker = CopyTradeTracker()
    stats = tracker.get_trader_stats()

    if not stats:
        console.print("[yellow]No copy-trade data yet.[/yellow]")
        return

    table = Table(title="Copy-Trade Performance", show_lines=True)
    table.add_column("Trader", style="cyan", width=15)
    table.add_column("Signals", justify="right", width=10)
    table.add_column("Traded", justify="right", width=10)
    table.add_column("Settled", justify="right", width=10)
    table.add_column("Win Rate", justify="right", width=10)
    table.add_column("PnL", justify="right", width=10)
    table.add_column("Avg Conf", justify="right", width=10)

    for trader, s in stats.items():
        pnl_color = "green" if s["total_pnl"] >= 0 else "red"
        wr_color = "green" if s["win_rate"] >= 0.5 else "red"
        wr_str = f"{s['win_rate']:.0%}" if s["total_trades"] > 0 else "-"

        table.add_row(
            trader,
            str(s["signals_generated"]),
            str(s["signals_traded"]),
            str(s["total_trades"]),
            f"[{wr_color}]{wr_str}[/{wr_color}]",
            f"[{pnl_color}]${s['total_pnl']:+.2f}[/{pnl_color}]",
            f"{s['avg_confidence']:.0%}",
        )

    console.print(table)


# =============================================================================
# CLI COMMANDS
# =============================================================================

def cmd_watch(poll_interval: int = WATCH_POLL_INTERVAL):
    """Poll tracked wallets every N seconds, show new trades."""
    wallets = _get_all_wallets()
    seen = _load_seen_trades()

    console.print(
        f"[bold cyan]Watching {len(wallets)} wallets "
        f"(poll every {poll_interval}s). Ctrl+C to stop.[/bold cyan]\n"
    )

    for name, cfg in wallets.items():
        console.print(f"  {name}: {cfg['address'][:12]}... (weight={cfg.get('weight', 1.0)})")
    console.print()

    try:
        while True:
            now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
            new_count = 0

            for name, cfg in wallets.items():
                trades = fetch_recent_trades(cfg["address"], limit=10)
                new_trades = [t for t in trades if t["trade_id"] not in seen]

                if new_trades:
                    console.print(
                        f"\n[bold green][{now}] {len(new_trades)} new trade(s) "
                        f"from {name}:[/bold green]"
                    )
                    display_trades(new_trades, name)
                    new_count += len(new_trades)

                    for t in new_trades:
                        seen.add(t["trade_id"])

            if new_count == 0:
                console.print(f"[dim][{now}] No new trades[/dim]", end="\r")

            _save_seen_trades(seen)
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        console.print("\n[yellow]Watch stopped.[/yellow]")
        _save_seen_trades(seen)


def cmd_positions():
    """Show current positions of all tracked wallets."""
    wallets = _get_all_wallets()

    for name, cfg in wallets.items():
        console.print(f"\n[bold cyan]--- {name} ---[/bold cyan]")
        positions = fetch_positions(cfg["address"])
        display_positions(positions, name)


def cmd_signals():
    """Generate and display copy signals."""
    try:
        client = KalshiClient()
        signals = generate_copy_signals(client)
        display_signals(signals)

        if signals:
            tracker = CopyTradeTracker()
            for s in signals:
                tracker.log_signal(s)
            console.print(
                f"\n[dim]{len(signals)} signal(s) logged to {LOG_FILE}[/dim]"
            )
    except Exception as e:
        console.print(f"[red]Signal generation failed: {e}[/red]")
        logger.error(f"Signal generation error: {e}")


def cmd_stats():
    """Show copy-trade performance stats."""
    display_stats()


def cmd_add_wallet(address: str, name: str):
    """Add a wallet to track."""
    add_wallet(address, name)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Copy-trade module: mirror top Polymarket traders to Kalshi"
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="Poll every 60s, show new trades from tracked wallets",
    )
    parser.add_argument(
        "--positions", action="store_true",
        help="Show current positions of tracked wallets",
    )
    parser.add_argument(
        "--signals", action="store_true",
        help="Generate copy signals for current Kalshi markets",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show copy-trade performance stats",
    )
    parser.add_argument(
        "--add-wallet", nargs=2, metavar=("ADDRESS", "NAME"),
        help="Add a wallet to track",
    )
    parser.add_argument(
        "--poll-interval", type=int, default=WATCH_POLL_INTERVAL,
        help=f"Poll interval in seconds for --watch (default: {WATCH_POLL_INTERVAL})",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.add_wallet:
        cmd_add_wallet(args.add_wallet[0], args.add_wallet[1])
    elif args.watch:
        cmd_watch(poll_interval=args.poll_interval)
    elif args.positions:
        cmd_positions()
    elif args.signals:
        cmd_signals()
    elif args.stats:
        cmd_stats()
    else:
        # Default: show summary
        console.print("[bold cyan]Copy-Trade Module[/bold cyan]")
        console.print()

        wallets = _get_all_wallets()
        console.print(f"Tracked wallets: {len(wallets)}")
        for name, cfg in wallets.items():
            console.print(
                f"  {name}: {cfg['address'][:12]}... "
                f"(weight={cfg.get('weight', 1.0)}, focus={cfg.get('focus', [])})"
            )

        console.print()
        console.print("Commands:")
        console.print("  --watch       Poll for new trades")
        console.print("  --positions   Show current positions")
        console.print("  --signals     Generate Kalshi signals")
        console.print("  --stats       Performance stats")
        console.print("  --add-wallet  Add a new wallet")


if __name__ == "__main__":
    main()
