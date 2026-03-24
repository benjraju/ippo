#!/usr/bin/env python3
"""
weather_tail_runner.py — Fast 30-minute weather tail scanner.

Runs independently from auto_trade.py on a 30-min cycle.
Only does ONE thing: find weather tail NO opportunities and place maker orders.

This catches fresh NWS forecast updates faster than the 2h auto_trade cycle.
Weather forecasts update every few hours — scanning every 30 min means we
catch mispricing within minutes of a forecast change, before the market corrects.

Usage:
    python weather_tail_runner.py --live     # Place real orders
    python weather_tail_runner.py            # Dry run (default)
"""

import argparse
import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from kalshi_client import KalshiClient

# Import weather tail strategy
try:
    from weather_tail_strategy import find_weather_tail_trades, tail_risk_budget
except ImportError:
    print("ERROR: weather_tail_strategy.py not found")
    sys.exit(1)

# Setup logging
LOG_FILE = config.OUTPUT_DIR / "weather_tail_runner.log"
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("weather_tail_runner")


def get_balance(client):
    """Get current account balance in dollars."""
    try:
        bal = client.get_balance()
        return bal.get("balance", 0) / 100.0
    except Exception as e:
        log.error(f"Balance fetch failed: {e}")
        return None


def get_existing_positions(client):
    """Get set of tickers we already have positions on."""
    try:
        pos = client.get_positions()
        existing = set()
        for p in pos.get("market_positions", []):
            fp = float(p.get("position_fp", 0) or 0)
            if fp != 0:
                existing.add(p.get("ticker", ""))
        return existing
    except Exception as e:
        log.error(f"Positions fetch failed: {e}")
        return set()


def run(dry_run=True):
    """Scan for weather tail NOs and place orders."""
    log.info(f"{'[DRY RUN] ' if dry_run else ''}Weather tail scan starting")

    client = KalshiClient()
    balance = get_balance(client)
    if balance is None:
        log.error("Could not get balance, aborting")
        return

    log.info(f"Balance: ${balance:.2f}")

    # Don't trade if balance is too low
    if balance < 2.0:
        log.warning(f"Balance ${balance:.2f} too low, skipping")
        return

    # Get existing positions to avoid duplicates
    existing = get_existing_positions(client)
    log.info(f"Existing positions: {len(existing)}")

    # Find opportunities
    try:
        tail_trades = find_weather_tail_trades(client)
    except Exception as e:
        log.error(f"Scan failed: {e}")
        log.error(traceback.format_exc())
        return

    if not tail_trades:
        log.info("No tail opportunities found")
        return

    log.info(f"Found {len(tail_trades)} tail opportunities")

    # Risk budget
    risk = tail_risk_budget(balance)
    max_per_cycle = 8  # cap per 30-min cycle
    remaining_balance = balance
    placed = 0

    for tt in tail_trades:
        if placed >= max_per_cycle:
            break

        ticker = tt["ticker"]

        # Skip if we already have a position
        if ticker in existing:
            log.debug(f"  SKIP {ticker}: already have position")
            continue

        contracts = min(
            tt.get("suggested_contracts", 1),
            risk.get("max_contracts", 5),
            3,  # hard cap per trade
        )
        if contracts <= 0:
            continue

        price_cents = int(tt.get("no_price_cents", 97))
        cost = contracts * price_cents / 100.0

        # Balance check
        if cost > remaining_balance - 2.0:  # keep $2 reserve
            log.info(f"  SKIP {ticker}: cost ${cost:.2f} > remaining ${remaining_balance:.2f}")
            continue

        edge = tt.get("net_edge_cents", 0)
        yes_price = tt.get("yes_price_cents", 0)

        if not dry_run:
            try:
                result = client.place_order(
                    ticker=ticker,
                    side="no",
                    action="buy",
                    count=contracts,
                    type="limit",
                    no_price=price_cents,
                )
                order_id = result.get("order", {}).get("order_id", "")
                remaining_balance -= cost
                placed += 1
                log.info(
                    f"  PLACED: NO {ticker} x{contracts} @ {price_cents}c | "
                    f"YES@{yes_price}c edge={edge:.1f}c | ${remaining_balance:.2f} left"
                )
            except Exception as e:
                log.warning(f"  FAILED: {ticker} — {e}")
        else:
            placed += 1
            remaining_balance -= cost
            log.info(
                f"  [DRY] NO {ticker} x{contracts} @ {price_cents}c | "
                f"YES@{yes_price}c edge={edge:.1f}c"
            )

    log.info(f"Done: {placed} orders placed, ${remaining_balance:.2f} remaining")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Place real orders")
    parser.add_argument("--no-confirm", action="store_true")
    args = parser.parse_args()
    run(dry_run=not args.live)
