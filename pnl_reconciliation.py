"""
pnl_reconciliation.py -- Daily P&L reconciliation from Kalshi API.

Source of truth: Kalshi account balance + fills API.
Never trusts CSV cumulative totals. Recomputes everything from API data.

Outputs:
    output/pnl_daily.csv   -- Append-only daily reconciliation log
    (dict return)          -- Latest snapshot for Telegram /pnl command

Usage:
    python pnl_reconciliation.py              # Run daily reconciliation
    python pnl_reconciliation.py --history    # Rebuild from all fills
"""

import csv
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
import config
from kalshi_client import KalshiClient

log = logging.getLogger("pnl_reconciliation")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEPOSITS = [
    ("2026-03-21", 98.00),
    ("2026-03-27", 490.00),
]
TOTAL_DEPOSITED = sum(amt for _, amt in DEPOSITS)

PNL_DAILY_CSV = config.OUTPUT_DIR / "pnl_daily.csv"
PNL_DAILY_FIELDS = [
    "date",
    "timestamp",
    "cash_balance",
    "portfolio_value",
    "total_equity",
    "total_deposited",
    "net_pnl",
    "net_pnl_pct",
    "num_fills_total",
    "num_open_positions",
    "realized_pnl_from_fills",
    "strategy_breakdown",
]


# ---------------------------------------------------------------------------
# Strategy classification (lightweight, matches telegram_bot.py patterns)
# ---------------------------------------------------------------------------

def _classify_fill(ticker: str, side: str, price_cents: float) -> str:
    """Classify a fill into a strategy bucket."""
    t = ticker.upper()
    if t.startswith("KXNBAGAME"):
        if side == "yes" and price_cents <= 30:
            return "nba_underdog"
        return "nba_other"
    if t.startswith("KXNBAPTS"):
        return "nba_props"
    if t.startswith(("KXNBA", "KXNHL", "KXMLB", "KXNCAA", "KXMARMAD")):
        return "sports"
    if t.startswith("KXHIGH"):
        if (side == "no" and price_cents >= 85) or (side == "yes" and price_cents <= 15):
            return "weather_tail"
        return "weather"
    if t.startswith(("KXBTC", "KXETH", "KXSOL")):
        return "crypto"
    return "other"


# ---------------------------------------------------------------------------
# Core reconciliation logic
# ---------------------------------------------------------------------------

def _fetch_balance(client: KalshiClient) -> dict:
    """Get cash balance and portfolio value from Kalshi API.

    Returns dict with keys: cash, portfolio_value, total_equity (all in dollars).
    """
    bal = client.get_balance()
    cash_cents = bal.get("balance", 0)
    portfolio_cents = bal.get("portfolio_value", 0) or 0
    cash = cash_cents / 100.0
    portfolio_value = portfolio_cents / 100.0
    return {
        "cash": cash,
        "portfolio_value": portfolio_value,
        "total_equity": cash + portfolio_value,
    }


def _fetch_all_fills(client: KalshiClient, max_pages: int = 20) -> list[dict]:
    """Paginate through all fills from the Kalshi API.

    Returns list of raw fill dicts, oldest first.
    """
    all_fills = []
    cursor = None
    for _ in range(max_pages):
        try:
            resp = client.get_fills(limit=100, cursor=cursor)
        except Exception as e:
            log.error(f"Error fetching fills page: {e}")
            break

        fills = resp.get("fills", [])
        if not fills:
            break
        all_fills.extend(fills)
        cursor = resp.get("cursor")
        if not cursor:
            break
        time.sleep(0.3)  # rate limit

    # Oldest first for chronological processing
    all_fills.reverse()
    return all_fills


def _compute_realized_pnl(fills: list[dict], client: KalshiClient) -> dict:
    """Compute realized P&L from settled fills.

    Groups fills by ticker, checks market settlement status,
    computes P&L for each settled position.

    Returns:
        {
            "total_realized_pnl": float,
            "by_strategy": {"weather_tail": {"pnl": x, "trades": n, "wins": w}, ...},
            "settled_trades": [{"ticker": ..., "pnl": ..., "strategy": ..., "side": ...}, ...],
            "open_tickers": int,
        }
    """
    # Group fills by ticker
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for fill in fills:
        ticker = fill.get("ticker", "")
        if ticker:
            by_ticker[ticker].append(fill)

    total_realized = 0.0
    by_strategy: dict[str, dict] = defaultdict(lambda: {"pnl": 0.0, "trades": 0, "wins": 0})
    settled_trades = []
    open_count = 0

    # Cache market lookups
    market_cache: dict[str, Optional[dict]] = {}

    for ticker, ticker_fills in sorted(by_ticker.items()):
        # Compute net position
        net_yes = 0
        total_cost_dollars = 0.0
        dominant_side = "yes"
        dominant_price = 50.0

        for fill in ticker_fills:
            side = fill.get("side", "")
            action = fill.get("action", "")
            count_raw = fill.get("count_fp") or fill.get("count", "0")
            count = int(float(count_raw))
            price_dollars = float(
                fill.get("yes_price_dollars") or fill.get("yes_price", "0")
            )
            price_cents = price_dollars * 100

            if action == "buy" and side == "yes":
                net_yes += count
                total_cost_dollars += count * price_dollars
                dominant_side = "yes"
                dominant_price = price_cents
            elif action == "sell" and side == "yes":
                net_yes -= count
                total_cost_dollars -= count * price_dollars
            elif action == "buy" and side == "no":
                net_yes -= count
                total_cost_dollars += count * (1.0 - price_dollars)
                dominant_side = "no"
                dominant_price = (1.0 - price_dollars) * 100
            elif action == "sell" and side == "no":
                net_yes += count
                total_cost_dollars -= count * (1.0 - price_dollars)

        abs_contracts = abs(net_yes)
        if abs_contracts == 0:
            continue

        actual_side = "yes" if net_yes > 0 else "no"
        avg_entry_cents = abs(total_cost_dollars / abs_contracts * 100) if abs_contracts else 0

        # Check settlement status
        if ticker not in market_cache:
            try:
                resp = client.get_market(ticker)
                market_cache[ticker] = resp.get("market", {})
            except Exception:
                market_cache[ticker] = None
            time.sleep(0.15)  # rate limit

        market = market_cache[ticker]
        if not market:
            open_count += 1
            continue

        status = market.get("status", "")
        result = market.get("result", "")

        if status not in ("settled", "finalized") or not result:
            open_count += 1
            continue

        # Compute P&L for settled position
        if result == "yes":
            # YES wins: YES holders get $1, NO holders lose
            if net_yes > 0:
                pnl = abs_contracts * (100 - avg_entry_cents) / 100.0
            else:
                pnl = -abs_contracts * (100 - avg_entry_cents) / 100.0
        elif result == "no":
            # NO wins: NO holders get $1, YES holders lose
            if net_yes > 0:
                pnl = -abs_contracts * avg_entry_cents / 100.0
            else:
                pnl = abs_contracts * avg_entry_cents / 100.0
        else:
            pnl = 0.0

        strategy = _classify_fill(ticker, actual_side, avg_entry_cents)
        total_realized += pnl
        by_strategy[strategy]["pnl"] += pnl
        by_strategy[strategy]["trades"] += 1
        if pnl > 0:
            by_strategy[strategy]["wins"] += 1

        settled_trades.append({
            "ticker": ticker,
            "side": actual_side,
            "contracts": abs_contracts,
            "entry_cents": round(avg_entry_cents, 1),
            "result": result,
            "pnl": round(pnl, 4),
            "strategy": strategy,
        })

    # Round strategy totals
    for s in by_strategy.values():
        s["pnl"] = round(s["pnl"], 4)

    return {
        "total_realized_pnl": round(total_realized, 4),
        "by_strategy": dict(by_strategy),
        "settled_trades": settled_trades,
        "open_tickers": open_count,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reconcile(client: KalshiClient = None, compute_fills: bool = True) -> dict:
    """Run daily P&L reconciliation.

    This is the main entry point. It:
    1. Queries Kalshi for current balance (source of truth).
    2. Optionally recomputes realized P&L from fills.
    3. Appends a row to pnl_daily.csv.
    4. Returns a snapshot dict for callers (Telegram bot, ss.py, etc).

    Args:
        client: KalshiClient instance (created if None).
        compute_fills: If True, also fetch fills and compute realized P&L.
            Set False for a quick balance-only snapshot.

    Returns:
        dict with keys: date, cash, portfolio_value, total_equity,
        total_deposited, net_pnl, net_pnl_pct, realized_pnl_from_fills,
        strategy_breakdown, num_open_positions, num_fills_total.
    """
    if client is None:
        client = KalshiClient()

    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")

    # Step 1: Balance from API (source of truth)
    balance = _fetch_balance(client)
    net_pnl = balance["total_equity"] - TOTAL_DEPOSITED
    net_pnl_pct = (net_pnl / TOTAL_DEPOSITED * 100) if TOTAL_DEPOSITED > 0 else 0.0

    # Step 2: Realized P&L from fills (optional, slower)
    fills_data = {
        "total_realized_pnl": 0.0,
        "by_strategy": {},
        "settled_trades": [],
        "open_tickers": 0,
    }
    num_fills = 0
    if compute_fills:
        fills = _fetch_all_fills(client)
        num_fills = len(fills)
        fills_data = _compute_realized_pnl(fills, client)

    # Step 3: Build snapshot
    snapshot = {
        "date": today,
        "timestamp": now.isoformat(),
        "cash": round(balance["cash"], 2),
        "portfolio_value": round(balance["portfolio_value"], 2),
        "total_equity": round(balance["total_equity"], 2),
        "total_deposited": TOTAL_DEPOSITED,
        "net_pnl": round(net_pnl, 2),
        "net_pnl_pct": round(net_pnl_pct, 2),
        "realized_pnl_from_fills": fills_data["total_realized_pnl"],
        "strategy_breakdown": fills_data["by_strategy"],
        "settled_trades": fills_data["settled_trades"],
        "num_open_positions": fills_data["open_tickers"],
        "num_fills_total": num_fills,
    }

    # Step 4: Append to CSV
    _append_to_csv(snapshot)

    return snapshot


def quick_balance(client: KalshiClient = None) -> dict:
    """Fast balance-only check, no fill computation.

    Use this for Telegram commands that need sub-second response.
    """
    return reconcile(client=client, compute_fills=False)


def format_pnl_report(snapshot: dict) -> str:
    """Format a snapshot dict as an HTML Telegram message."""
    net = snapshot["net_pnl"]
    sign = "+" if net >= 0 else ""
    equity = snapshot["total_equity"]
    cash = snapshot["cash"]
    positions = snapshot["portfolio_value"]
    deposited = snapshot["total_deposited"]
    pct = snapshot["net_pnl_pct"]

    msg = (
        f"<b>P&amp;L RECONCILIATION</b>\n"
        f"<i>{snapshot['date']}</i>\n\n"
        f"Cash:        ${cash:.2f}\n"
        f"Positions:   ${positions:.2f}\n"
        f"Equity:      <b>${equity:.2f}</b>\n"
        f"Deposited:   ${deposited:.2f}\n"
        f"Net P&amp;L:    <b>{sign}${net:.2f}</b> ({sign}{pct:.1f}%)\n"
    )

    # Strategy breakdown from fills
    breakdown = snapshot.get("strategy_breakdown", {})
    if breakdown:
        msg += "\n<b>By Strategy (settled):</b>\n"

        # Sort by P&L descending
        for strat, data in sorted(
            breakdown.items(), key=lambda x: x[1]["pnl"], reverse=True
        ):
            p = data["pnl"]
            t = data["trades"]
            w = data["wins"]
            wr = w / t * 100 if t > 0 else 0
            s = "+" if p >= 0 else ""
            msg += f"  {strat:15s} {s}${p:.2f}  ({t}t, {wr:.0f}%W)\n"

    open_pos = snapshot.get("num_open_positions", 0)
    if open_pos:
        msg += f"\nOpen positions: {open_pos}"

    msg += (
        "\n\n<i>Source: Kalshi API balance</i>"
    )

    return msg


# ---------------------------------------------------------------------------
# CSV persistence (append-only)
# ---------------------------------------------------------------------------

def _append_to_csv(snapshot: dict):
    """Append one row to pnl_daily.csv.

    If a row for today already exists, update it (replace last occurrence).
    Otherwise, append.
    """
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    csv_path = PNL_DAILY_CSV

    # Prepare the row
    row = {
        "date": snapshot["date"],
        "timestamp": snapshot["timestamp"],
        "cash_balance": snapshot["cash"],
        "portfolio_value": snapshot["portfolio_value"],
        "total_equity": snapshot["total_equity"],
        "total_deposited": snapshot["total_deposited"],
        "net_pnl": snapshot["net_pnl"],
        "net_pnl_pct": snapshot["net_pnl_pct"],
        "num_fills_total": snapshot["num_fills_total"],
        "num_open_positions": snapshot["num_open_positions"],
        "realized_pnl_from_fills": snapshot["realized_pnl_from_fills"],
        "strategy_breakdown": json.dumps(snapshot.get("strategy_breakdown", {})),
    }

    # Read existing rows (if file exists)
    existing_rows = []
    if csv_path.exists():
        try:
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
        except Exception:
            existing_rows = []

    # Check if today already has an entry -- update the last one
    today = snapshot["date"]
    updated = False
    for i in range(len(existing_rows) - 1, -1, -1):
        if existing_rows[i].get("date") == today:
            existing_rows[i] = row
            updated = True
            break

    if not updated:
        existing_rows.append(row)

    # Write all rows
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PNL_DAILY_FIELDS)
        writer.writeheader()
        for r in existing_rows:
            writer.writerow(r)

    log.info(f"P&L reconciliation saved to {csv_path} ({snapshot['date']})")


# ---------------------------------------------------------------------------
# Load historical data from CSV
# ---------------------------------------------------------------------------

def load_daily_history() -> list[dict]:
    """Load all rows from pnl_daily.csv as a list of dicts."""
    if not PNL_DAILY_CSV.exists():
        return []
    rows = []
    with open(PNL_DAILY_CSV, "r") as f:
        for row in csv.DictReader(f):
            # Parse numeric fields
            for field in [
                "cash_balance", "portfolio_value", "total_equity",
                "total_deposited", "net_pnl", "net_pnl_pct",
                "realized_pnl_from_fills",
            ]:
                try:
                    row[field] = float(row.get(field, 0) or 0)
                except (ValueError, TypeError):
                    row[field] = 0.0
            for field in ["num_fills_total", "num_open_positions"]:
                try:
                    row[field] = int(row.get(field, 0) or 0)
                except (ValueError, TypeError):
                    row[field] = 0
            # Parse strategy breakdown JSON
            try:
                row["strategy_breakdown"] = json.loads(
                    row.get("strategy_breakdown", "{}")
                )
            except (json.JSONDecodeError, TypeError):
                row["strategy_breakdown"] = {}
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for manual reconciliation."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Ippo P&L Reconciliation")
    parser.add_argument(
        "--history", action="store_true",
        help="Rebuild realized P&L from all fills (slower, more accurate)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Balance-only check, skip fill computation",
    )
    args = parser.parse_args()

    print("\n  IPPO P&L RECONCILIATION")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    if args.quick:
        snapshot = quick_balance()
    else:
        snapshot = reconcile(compute_fills=not args.quick)

    # Display results
    print(f"\n  Cash:              ${snapshot['cash']:.2f}")
    print(f"  Positions:         ${snapshot['portfolio_value']:.2f}")
    print(f"  Total equity:      ${snapshot['total_equity']:.2f}")
    print(f"  Total deposited:   ${TOTAL_DEPOSITED:.2f}")
    net = snapshot["net_pnl"]
    sign = "+" if net >= 0 else ""
    print(f"  Net P&L:           {sign}${net:.2f} ({sign}{snapshot['net_pnl_pct']:.1f}%)")

    if snapshot.get("strategy_breakdown"):
        print(f"\n  {'Strategy':18s} {'P&L':>10s} {'Trades':>7s} {'Win%':>6s}")
        print(f"  {'-'*18} {'-'*10} {'-'*7} {'-'*6}")
        for strat, data in sorted(
            snapshot["strategy_breakdown"].items(),
            key=lambda x: x[1]["pnl"],
            reverse=True,
        ):
            p = data["pnl"]
            t = data["trades"]
            w = data["wins"]
            wr = w / t * 100 if t > 0 else 0
            s = "+" if p >= 0 else ""
            print(f"  {strat:18s} {s}${p:9.2f} {t:7d} {wr:5.0f}%")

    print(f"\n  Open positions:    {snapshot['num_open_positions']}")
    print(f"  Total fills:       {snapshot['num_fills_total']}")
    print(f"\n  Saved to: {PNL_DAILY_CSV}")
    print()


if __name__ == "__main__":
    main()
