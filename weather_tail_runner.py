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

# Import settlement timing strategy
try:
    from settlement_timing_strategy import find_settlement_timing_trades
except ImportError:
    find_settlement_timing_trades = None

# Setup logging
LOG_FILE = config.OUTPUT_DIR / "weather_tail_runner.log"
log = logging.getLogger("weather_tail_runner")
log.setLevel(logging.INFO)
log.propagate = False
if not log.handlers:
    _fmt = logging.Formatter("[%(asctime)s] %(levelname)-7s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    _fh = logging.FileHandler(LOG_FILE)
    _fh.setFormatter(_fmt)
    _sh = logging.StreamHandler()
    _sh.setFormatter(_fmt)
    log.handlers = [_fh, _sh]


def get_balance(client):
    """Get current account balance in dollars."""
    try:
        bal = client.get_balance()
        return bal.get("balance", 0) / 100.0
    except Exception as e:
        log.error(f"Balance fetch failed: {e}")
        return None


def get_existing_positions(client):
    """Get dict of tickers we already have positions on, with contract counts."""
    try:
        pos = client.get_positions()
        existing = {}
        for p in pos.get("market_positions", []):
            fp = float(p.get("position_fp", 0) or 0)
            if fp != 0:
                existing[p.get("ticker", "")] = int(abs(fp))
        return existing
    except Exception as e:
        log.error(f"Positions fetch failed: {e}")
        return {}


def run(dry_run=True):
    """Scan for weather tail NOs and place orders."""
    log.info(f"{'[DRY RUN] ' if dry_run else ''}Weather tail scan starting")

    client = KalshiClient()

    # Cancel stale resting orders before placing new ones (frees capital)
    try:
        from auto_trade import cleanup_stale_orders
        cleanup_stale_orders(client, log, max_age_hours=3.0, dry_run=dry_run)
    except Exception as e:
        log.warning(f"Stale order cleanup failed: {e}")

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

    # Cap: max 8 contracts per ticker across all cycles (prevent accumulation)
    MAX_CONTRACTS_PER_TICKER = 8

    for tt in tail_trades:
        if placed >= max_per_cycle:
            break

        ticker = tt["ticker"]

        # Skip if we already have enough contracts on this ticker
        current_contracts = existing.get(ticker, 0)
        if current_contracts >= MAX_CONTRACTS_PER_TICKER:
            log.debug(f"  SKIP {ticker}: already {current_contracts} contracts (max {MAX_CONTRACTS_PER_TICKER})")
            continue

        # Only add enough to reach the cap
        room = MAX_CONTRACTS_PER_TICKER - current_contracts

        # Scale contract count by confidence (same logic as auto_trade.py)
        our_no_prob = tt.get("our_no_prob", 0.98)
        budget_max = risk.get("max_contracts", 5)
        if our_no_prob >= 0.995:
            prob_max = budget_max  # Near-certain
        elif our_no_prob >= 0.98:
            prob_max = max(1, int(budget_max * 0.75))
        else:
            prob_max = max(1, int(budget_max * 0.5))

        # Dutch book sizing: scale by market overpricing signal
        db_total = tt.get("dutch_book_total", 100)
        if db_total < 102:
            prob_max = max(1, int(prob_max * 0.5))
        elif db_total < 110:
            scale = 0.5 + 0.5 * (db_total - 102) / 8
            prob_max = max(1, int(prob_max * scale))

        contracts = min(
            tt.get("suggested_contracts", 1),
            prob_max,
            4,  # hard cap per trade
            room,  # don't exceed per-ticker cap
        )
        if contracts <= 0:
            continue

        raw_no_price = int(tt.get("no_price_cents", 97))
        # Bid 1c below the NO ask to stay as maker (avoid "post only cross")
        price_cents = max(1, raw_no_price - 1)
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

    log.info(f"Weather tail done: {placed} orders placed, ${remaining_balance:.2f} remaining")

    # --- Settlement timing (near-risk-free, after 2 PM local) ---
    if find_settlement_timing_trades is not None:
        try:
            log.info("--- Settlement timing scan ---")
            st_trades = find_settlement_timing_trades(client)
            if not st_trades:
                log.info("No settlement timing opportunities (wrong time or no known outcomes)")
            else:
                log.info(f"Found {len(st_trades)} settlement timing opportunities")
                MAX_ST_PER_TICKER = 5
                st_placed = 0

                for st in st_trades[:8]:
                    if st_placed >= 8:
                        break

                    ticker = st.ticker
                    current = existing.get(ticker, 0)
                    if current >= MAX_ST_PER_TICKER:
                        log.debug(f"  SKIP {ticker}: already {current} contracts")
                        continue
                    room = MAX_ST_PER_TICKER - current

                    contracts = min(st.suggested_contracts, room)
                    if contracts <= 0:
                        continue

                    if st.side == "yes":
                        price_cents = max(1, st.yes_price_cents - 1)
                    else:
                        price_cents = max(1, st.no_price_cents - 1)

                    cost = contracts * price_cents / 100.0
                    if cost > remaining_balance - 2.0:
                        log.info(f"  SKIP {ticker}: cost ${cost:.2f} > remaining ${remaining_balance:.2f}")
                        continue

                    if not dry_run:
                        try:
                            order_kwargs = {
                                "ticker": ticker,
                                "side": st.side,
                                "action": "buy",
                                "count": contracts,
                                "type": "limit",
                            }
                            if st.side == "yes":
                                order_kwargs["yes_price"] = price_cents
                            else:
                                order_kwargs["no_price"] = price_cents
                            result = client.place_order(**order_kwargs)
                            remaining_balance -= cost
                            st_placed += 1
                            log.info(
                                f"  PLACED: {st.side.upper()} {ticker} x{contracts} @{price_cents}c | "
                                f"obs={st.observed_high_f:.1f}F thresh={st.threshold_f:.0f}F "
                                f"buf={st.buffer_f:+.1f}F conf={st.confidence}"
                            )
                        except Exception as e:
                            log.warning(f"  FAILED: {ticker} — {e}")
                    else:
                        st_placed += 1
                        remaining_balance -= cost
                        log.info(
                            f"  [DRY] {st.side.upper()} {ticker} x{contracts} @{price_cents}c | "
                            f"obs={st.observed_high_f:.1f}F thresh={st.threshold_f:.0f}F"
                        )

                log.info(f"Settlement timing done: {st_placed} orders, ${remaining_balance:.2f} remaining")
        except Exception as e:
            log.error(f"Settlement timing scan failed: {e}")
            log.error(traceback.format_exc())
    else:
        log.info("Settlement timing module not available")

    log.info(f"All done: ${remaining_balance:.2f} remaining")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Place real orders")
    parser.add_argument("--no-confirm", action="store_true")
    args = parser.parse_args()
    run(dry_run=not args.live)
