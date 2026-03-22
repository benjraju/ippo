"""
live_trader.py -- Unified trade executor for all 4 proven strategies.

Strategies:
  1. Tail Fade      — Buy NO on extreme-tail markets (yes_ask <= 5c or 30-50c range)
  2. NBA Underdogs  — Buy YES on underdog NBA game winners (10-30c)
  3. Weather Dutch Book — Sell YES on all 6 legs when sum(yes_bid) > 102c
  4. NBA Favorite Fade  — Buy NO on heavy favorites (70-80c), half size

Usage:
  python live_trader.py              # dry-run (default)
  python live_trader.py --live       # real orders
  python live_trader.py --dry-run    # explicit dry-run
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config
from kalshi_client import KalshiClient
from risk_manager import RiskManager, TradeProposal

# ---------------------------------------------------------------------------
# Tail Fade params — loaded from candidate_strategy.py with safe fallbacks
# ---------------------------------------------------------------------------
try:
    from autoresearch.candidate_strategy import (
        TAIL_FADE_MAX_PRICE,
        TAIL_FADE_MIN_VOLUME,
        TAIL_FADE_MID_LOW,
        TAIL_FADE_MID_HIGH,
        TAIL_FADE_WEATHER_ENABLED,
        TAIL_FADE_CRYPTO_ENABLED,
        TAIL_FADE_NBA_ENABLED,
    )
except ImportError:
    TAIL_FADE_MAX_PRICE = 5
    TAIL_FADE_MIN_VOLUME = 0
    TAIL_FADE_MID_LOW = 30
    TAIL_FADE_MID_HIGH = 50
    TAIL_FADE_WEATHER_ENABLED = 1
    TAIL_FADE_CRYPTO_ENABLED = 1
    TAIL_FADE_NBA_ENABLED = 1

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WEATHER_SERIES = [
    "KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA",
    "KXHIGHLA", "KXHIGHDC", "KXHIGHDEN",
]

TRADE_LOG_PATH = config.OUTPUT_DIR / "live_trades.jsonl"
ORDER_DELAY_SECONDS = 0.3  # rate limiting between order placements


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TradeSignal:
    """A trading opportunity detected by one of the 4 strategies."""
    strategy: str          # "tail_fade" | "nba_underdog" | "weather_dutch" | "nba_fav_fade"
    ticker: str            # market ticker
    side: str              # "yes" or "no"
    action: str            # "buy" or "sell"
    price_cents: int       # limit price in cents (1-99)
    contracts: int         # number of contracts
    max_dollars: float     # max cost in dollars
    reason: str = ""
    event_ticker: str = ""


@dataclass
class TradeResult:
    """Result of attempting to execute a TradeSignal."""
    signal: TradeSignal
    executed: bool = False
    order_id: str = ""
    error: str = ""
    dry_run: bool = True


# ---------------------------------------------------------------------------
# Price helpers
# ---------------------------------------------------------------------------
def dollars_to_cents(val) -> int:
    """Convert API dollar string/float to integer cents."""
    try:
        return int(round(float(val) * 100))
    except (TypeError, ValueError):
        return 0


def cents_to_contracts(max_dollars: float, price_cents: int) -> int:
    """How many contracts can we buy for max_dollars at price_cents?"""
    if price_cents <= 0 or price_cents >= 100:
        return 0
    cost_per = price_cents / 100.0
    return max(1, int(max_dollars / cost_per))


# ---------------------------------------------------------------------------
# Market fetcher — paginate through all open markets for a series
# ---------------------------------------------------------------------------
def fetch_all_markets(client: KalshiClient, series_ticker: str) -> list[dict]:
    """Fetch all open markets for a series, handling pagination."""
    markets = []
    cursor = None
    for _ in range(50):  # safety cap
        resp = client.get_markets(
            limit=100,
            cursor=cursor,
            series_ticker=series_ticker,
            status="open",
        )
        batch = resp.get("markets", [])
        if not batch:
            break
        markets.extend(batch)
        cursor = resp.get("cursor")
        if not cursor:
            break
    return markets


def fetch_all_open_markets(client: KalshiClient) -> list[dict]:
    """Fetch ALL open markets across every series (no series filter)."""
    markets = []
    cursor = None
    for _ in range(200):  # safety cap — Kalshi has thousands of markets
        resp = client.get_markets(limit=100, cursor=cursor, status="open")
        batch = resp.get("markets", [])
        if not batch:
            break
        markets.extend(batch)
        cursor = resp.get("cursor")
        if not cursor:
            break
    return markets


# ===========================================================================
# STRATEGY 1: TAIL FADE
# ===========================================================================
def scan_tail_fade(all_markets: list[dict]) -> list[TradeSignal]:
    """
    Scan all open markets for tail-fade opportunities.
    - yes_ask <= TAIL_FADE_MAX_PRICE cents  ->  BUY NO
    - TAIL_FADE_MID_LOW <= yes_ask <= TAIL_FADE_MID_HIGH  ->  BUY NO
    """
    signals = []
    max_dollars = config.MAX_BET_DOLLARS  # $2

    for m in all_markets:
        ticker = m.get("ticker", "")
        yes_ask = dollars_to_cents(m.get("yes_ask_dollars", 0))
        volume = int(m.get("volume", 0) or 0)

        if yes_ask <= 0 or yes_ask >= 100:
            continue
        if volume < TAIL_FADE_MIN_VOLUME:
            continue

        # Determine NO price: if we BUY NO, we pay (100 - yes_ask) cents
        no_price = 100 - yes_ask
        if no_price <= 0 or no_price >= 100:
            continue

        hit = False
        reason = ""

        # Extreme tail: yes_ask very cheap -> event very unlikely -> buy NO
        if yes_ask <= TAIL_FADE_MAX_PRICE:
            hit = True
            reason = f"tail_extreme: yes_ask={yes_ask}c <= {TAIL_FADE_MAX_PRICE}c"

        # Mid-range fade
        elif TAIL_FADE_MID_LOW <= yes_ask <= TAIL_FADE_MID_HIGH:
            hit = True
            reason = f"tail_mid: yes_ask={yes_ask}c in [{TAIL_FADE_MID_LOW},{TAIL_FADE_MID_HIGH}]"

        if hit:
            contracts = cents_to_contracts(max_dollars, no_price)
            signals.append(TradeSignal(
                strategy="tail_fade",
                ticker=ticker,
                side="no",
                action="buy",
                price_cents=no_price,
                contracts=contracts,
                max_dollars=max_dollars,
                reason=reason,
            ))

    return signals


# ===========================================================================
# STRATEGY 2: NBA UNDERDOGS
# ===========================================================================
def scan_nba_underdogs(client: KalshiClient) -> list[TradeSignal]:
    """
    Scan KXNBAGAME for underdog Winner markets with yes_ask 10-30c.
    Action: BUY YES at yes_ask.
    """
    signals = []
    max_dollars = config.MAX_BET_DOLLARS  # $2
    markets = fetch_all_markets(client, "KXNBAGAME")

    for m in markets:
        ticker = m.get("ticker", "")
        title = m.get("title", "")
        yes_ask = dollars_to_cents(m.get("yes_ask_dollars", 0))

        if "Winner" not in title:
            continue
        if not (10 <= yes_ask <= 30):
            continue

        contracts = cents_to_contracts(max_dollars, yes_ask)
        signals.append(TradeSignal(
            strategy="nba_underdog",
            ticker=ticker,
            side="yes",
            action="buy",
            price_cents=yes_ask,
            contracts=contracts,
            max_dollars=max_dollars,
            reason=f"underdog: yes_ask={yes_ask}c, title=\"{title}\"",
        ))

    return signals


# ===========================================================================
# STRATEGY 3: WEATHER DUTCH BOOK
# ===========================================================================
def scan_weather_dutch(client: KalshiClient) -> list[TradeSignal]:
    """
    For each weather series, group markets by event_ticker.
    If an event has exactly 6 markets (complete bucket set) and
    sum(yes_bid) > 102c, SELL YES on all 6 legs.
    """
    signals = []

    for series in WEATHER_SERIES:
        markets = fetch_all_markets(client, series)
        if not markets:
            continue

        # Group by event_ticker
        events: dict[str, list[dict]] = defaultdict(list)
        for m in markets:
            evt = m.get("event_ticker", "")
            if evt:
                events[evt].append(m)

        for event_ticker, legs in events.items():
            if len(legs) != 6:
                continue  # not a complete bucket set

            # Sum all yes_bid prices
            total_yes_bid = 0
            valid = True
            for leg in legs:
                yb = dollars_to_cents(leg.get("yes_bid_dollars", 0))
                if yb <= 0:
                    valid = False
                    break
                total_yes_bid += yb

            if not valid:
                continue
            if total_yes_bid <= 102:
                continue  # no profit after fees

            profit_cents = total_yes_bid - 100
            reason = (
                f"dutch_book: {len(legs)} legs, sum(yes_bid)={total_yes_bid}c, "
                f"profit={profit_cents}c, event={event_ticker}"
            )

            for leg in legs:
                yb = dollars_to_cents(leg.get("yes_bid_dollars", 0))
                signals.append(TradeSignal(
                    strategy="weather_dutch",
                    ticker=leg.get("ticker", ""),
                    side="yes",
                    action="sell",
                    price_cents=yb,
                    contracts=1,  # 1 contract per leg for safety
                    max_dollars=yb / 100.0,
                    reason=reason,
                    event_ticker=event_ticker,
                ))

    return signals


# ===========================================================================
# STRATEGY 4: NBA FAVORITE FADE (half size)
# ===========================================================================
def scan_nba_fav_fade(client: KalshiClient) -> list[TradeSignal]:
    """
    Scan KXNBAGAME Winner markets where yes_ask 70-80c.
    Action: BUY NO. Half size ($1 max) because small sample.
    """
    signals = []
    max_dollars = 1.0  # half size — collecting data
    markets = fetch_all_markets(client, "KXNBAGAME")

    for m in markets:
        ticker = m.get("ticker", "")
        title = m.get("title", "")
        yes_ask = dollars_to_cents(m.get("yes_ask_dollars", 0))

        if "Winner" not in title:
            continue
        if not (70 <= yes_ask <= 80):
            continue

        no_price = 100 - yes_ask
        if no_price <= 0 or no_price >= 100:
            continue

        contracts = cents_to_contracts(max_dollars, no_price)
        signals.append(TradeSignal(
            strategy="nba_fav_fade",
            ticker=ticker,
            side="no",
            action="buy",
            price_cents=no_price,
            contracts=contracts,
            max_dollars=max_dollars,
            reason=f"fav_fade: yes_ask={yes_ask}c, no_price={no_price}c, title=\"{title}\"",
        ))

    return signals


# ===========================================================================
# UNIFIED SCANNER
# ===========================================================================
def scan_all_opportunities(client: KalshiClient) -> list[TradeSignal]:
    """
    Run all 4 strategy scanners and return combined signal list.
    """
    print("=" * 70)
    print("SCANNING ALL STRATEGIES")
    print("=" * 70)

    # Fetch all open markets once for tail fade (scans everything)
    print("\n[1/4] Fetching all open markets for Tail Fade scan...")
    all_markets = fetch_all_open_markets(client)
    print(f"      Found {len(all_markets)} open markets total")
    tail_signals = scan_tail_fade(all_markets)
    print(f"      Tail Fade signals: {len(tail_signals)}")

    # NBA Underdogs
    print("\n[2/4] Scanning NBA Underdogs (KXNBAGAME)...")
    nba_under_signals = scan_nba_underdogs(client)
    print(f"      NBA Underdog signals: {len(nba_under_signals)}")

    # Weather Dutch Book
    print("\n[3/4] Scanning Weather Dutch Book...")
    dutch_signals = scan_weather_dutch(client)
    print(f"      Weather Dutch signals: {len(dutch_signals)}")

    # NBA Favorite Fade
    print("\n[4/4] Scanning NBA Favorite Fade (KXNBAGAME)...")
    fav_fade_signals = scan_nba_fav_fade(client)
    print(f"      NBA Fav Fade signals: {len(fav_fade_signals)}")

    all_signals = tail_signals + nba_under_signals + dutch_signals + fav_fade_signals
    print(f"\n>>> Total signals: {len(all_signals)}")
    return all_signals


# ===========================================================================
# SIGNAL TABLE
# ===========================================================================
def print_signal_table(signals: list[TradeSignal]):
    """Print a summary table of all detected signals."""
    if not signals:
        print("\nNo signals found.")
        return

    print("\n" + "=" * 110)
    print(f"{'Strategy':<18} {'Ticker':<28} {'Action':<6} {'Side':<5} "
          f"{'Price':>6} {'Qty':>4} {'MaxCost':>8}  Reason")
    print("-" * 110)

    for s in signals:
        cost = s.contracts * s.price_cents / 100.0
        print(f"{s.strategy:<18} {s.ticker:<28} {s.action:<6} {s.side:<5} "
              f"{s.price_cents:>5}c {s.contracts:>4} "
              f"${cost:>6.2f}  {s.reason[:40]}")

    print("=" * 110)


# ===========================================================================
# TRADE EXECUTOR
# ===========================================================================
def execute_trades(
    client: KalshiClient,
    signals: list[TradeSignal],
    risk_mgr: RiskManager,
    dry_run: bool = True,
) -> list[TradeResult]:
    """
    Execute trade signals through the Kalshi API.
    Checks risk_manager before every trade. Logs to live_trades.jsonl.
    """
    results = []
    executed_count = 0
    skipped_count = 0

    for sig in signals:
        # ----- Risk check -----
        # Build a TradeProposal for the risk manager.
        # For strategies without a model probability, use a synthetic edge
        # based on the strategy type so risk manager doesn't block on min_edge.
        if sig.strategy == "tail_fade":
            # Buying NO when YES is cheap -> high implied prob for NO
            estimated_prob = 0.97 if sig.price_cents >= 90 else 0.80
            market_price = sig.price_cents
        elif sig.strategy == "nba_underdog":
            # We think underdog is underpriced by at least 10%
            estimated_prob = (sig.price_cents / 100.0) + 0.10
            market_price = sig.price_cents
        elif sig.strategy == "weather_dutch":
            # Dutch book = guaranteed profit, synthetic high edge
            estimated_prob = 0.99
            market_price = sig.price_cents
        elif sig.strategy == "nba_fav_fade":
            # Buying NO when favorite is 70-80c -> NO is 20-30c
            estimated_prob = (sig.price_cents / 100.0) + 0.10
            market_price = sig.price_cents
        else:
            estimated_prob = 0.60
            market_price = sig.price_cents

        edge = estimated_prob - (market_price / 100.0)

        proposal = TradeProposal(
            ticker=sig.ticker,
            side=sig.side,
            action=sig.action,
            estimated_prob=estimated_prob,
            market_price=market_price,
            edge=edge,
            reason=sig.reason,
        )

        approved = risk_mgr.evaluate_trade(proposal)
        if approved is None:
            # Risk manager rejected — check if it's the daily cap
            status = risk_mgr.get_status()
            if status["is_blocked"]:
                reason = "RISK: daily loss cap hit"
            elif status["open_positions"] >= config.MAX_OPEN_POSITIONS:
                reason = "RISK: max open positions"
            elif abs(edge) < config.MIN_EDGE_THRESHOLD:
                reason = f"RISK: edge {edge:.3f} < min {config.MIN_EDGE_THRESHOLD}"
            else:
                reason = "RISK: rejected (Kelly/sizing)"

            results.append(TradeResult(
                signal=sig,
                executed=False,
                error=reason,
                dry_run=dry_run,
            ))
            skipped_count += 1
            continue

        # ----- Execute -----
        if dry_run:
            results.append(TradeResult(
                signal=sig,
                executed=True,
                order_id="DRY_RUN",
                dry_run=True,
            ))
            executed_count += 1
            _log_trade(sig, "DRY_RUN", dry_run=True)
        else:
            try:
                # Build order kwargs
                order_kwargs = {
                    "ticker": sig.ticker,
                    "side": sig.side,
                    "action": sig.action,
                    "count": sig.contracts,
                    "type": "limit",
                    "post_only": True,
                }

                # Price: Kalshi expects yes_price or no_price in cents
                if sig.side == "yes":
                    order_kwargs["yes_price"] = sig.price_cents
                else:
                    order_kwargs["no_price"] = sig.price_cents

                resp = client.place_order(**order_kwargs)
                order_id = resp.get("order", {}).get("order_id", "unknown")

                results.append(TradeResult(
                    signal=sig,
                    executed=True,
                    order_id=order_id,
                    dry_run=False,
                ))
                executed_count += 1
                _log_trade(sig, order_id, dry_run=False)

            except Exception as e:
                results.append(TradeResult(
                    signal=sig,
                    executed=False,
                    error=str(e),
                    dry_run=False,
                ))
                skipped_count += 1

            # Rate limit between orders
            time.sleep(ORDER_DELAY_SECONDS)

    print(f"\nExecution summary: {executed_count} executed, {skipped_count} skipped")
    return results


def _log_trade(sig: TradeSignal, order_id: str, dry_run: bool = True):
    """Append trade to output/live_trades.jsonl."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategy": sig.strategy,
        "ticker": sig.ticker,
        "side": sig.side,
        "action": sig.action,
        "price_cents": sig.price_cents,
        "contracts": sig.contracts,
        "max_dollars": sig.max_dollars,
        "order_id": order_id,
        "dry_run": dry_run,
        "reason": sig.reason,
        "event_ticker": sig.event_ticker,
    }
    try:
        with open(TRADE_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as e:
        print(f"WARNING: could not write trade log: {e}")


# ===========================================================================
# RESULT TABLE
# ===========================================================================
def print_result_table(results: list[TradeResult]):
    """Print execution results."""
    if not results:
        print("\nNo trades attempted.")
        return

    mode = "DRY-RUN" if results[0].dry_run else "LIVE"
    print(f"\n{'=' * 100}")
    print(f"EXECUTION RESULTS ({mode})")
    print(f"{'=' * 100}")
    print(f"{'Strategy':<18} {'Ticker':<28} {'Action':<6} {'Side':<5} "
          f"{'Price':>6} {'Qty':>4} {'Status':<10} OrderID/Error")
    print("-" * 100)

    for r in results:
        s = r.signal
        status = "OK" if r.executed else "SKIP"
        info = r.order_id if r.executed else r.error[:30]
        print(f"{s.strategy:<18} {s.ticker:<28} {s.action:<6} {s.side:<5} "
              f"{s.price_cents:>5}c {s.contracts:>4} {status:<10} {info}")

    print("=" * 100)

    # Summary by strategy
    by_strat: dict[str, dict] = defaultdict(lambda: {"exec": 0, "skip": 0})
    for r in results:
        key = r.signal.strategy
        if r.executed:
            by_strat[key]["exec"] += 1
        else:
            by_strat[key]["skip"] += 1

    print("\nSummary by strategy:")
    for strat, counts in sorted(by_strat.items()):
        print(f"  {strat:<18} executed={counts['exec']}, skipped={counts['skip']}")


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================
def run_live_trader(dry_run: bool = True):
    """
    Main entry point. Scans all strategies, prints signals, executes trades.
    """
    mode = "DRY-RUN" if dry_run else "LIVE"
    print(f"\n{'#' * 70}")
    print(f"  IPPO LIVE TRADER — {mode} MODE")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  ENV: {config.KALSHI_ENV}")
    print(f"{'#' * 70}")

    if not dry_run and config.KALSHI_ENV != "PROD":
        print("\nWARNING: --live flag set but KALSHI_ENV is not PROD.")
        print("Orders will go to DEMO API.\n")

    # Initialize
    client = KalshiClient()
    risk_mgr = RiskManager()

    # Show risk status
    status = risk_mgr.get_status()
    print(f"\nRisk status: balance=${status['balance']}, "
          f"daily_pnl=${status['daily_pnl']}, "
          f"budget_remaining=${status['daily_budget_remaining']}, "
          f"positions={status['open_positions']}/{config.MAX_OPEN_POSITIONS}, "
          f"blocked={status['is_blocked']}")

    if status["is_blocked"]:
        print("\nDAILY LOSS CAP HIT — no trades will be placed.")
        return

    # Scan
    signals = scan_all_opportunities(client)
    print_signal_table(signals)

    if not signals:
        print("\nNo opportunities found. Exiting.")
        return

    # Execute
    results = execute_trades(client, signals, risk_mgr, dry_run=dry_run)
    print_result_table(results)

    # Log path
    print(f"\nTrade log: {TRADE_LOG_PATH}")


# ===========================================================================
# CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Ippo Live Trader — unified executor for 4 proven strategies"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Place real orders (default is dry-run)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Simulate trades without placing orders (default)",
    )
    args = parser.parse_args()

    # --live overrides --dry-run
    dry_run = not args.live

    run_live_trader(dry_run=dry_run)


if __name__ == "__main__":
    main()
