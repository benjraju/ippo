#!/usr/bin/env python3
"""
autoresearch/backtest_harness.py — Fixed evaluation harness for strategy backtesting.

DO NOT MODIFY THIS FILE. This is the ground truth evaluator, equivalent to
Karpathy's prepare.py. The autoresearch agent modifies candidate_strategy.py
and runs this script to evaluate changes.

Loads 40K+ settled Kalshi markets, applies the strategy from candidate_strategy.py,
does a 70/30 chronological train/test split, and reports standardized metrics.

Usage:
    python autoresearch/backtest_harness.py              # Full evaluation
    python autoresearch/backtest_harness.py --refresh     # Pull fresh data first
    python autoresearch/backtest_harness.py --series KXBTC  # Single series
"""

import argparse
import importlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# ---------------------------------------------------------------------------
# Constants (DO NOT MODIFY)
# ---------------------------------------------------------------------------
SETTLEMENTS_FILE = config.OUTPUT_DIR / "historical_settlements_with_prices.json"
MAKER_FEE_RATE = 0.0175   # 1.75% maker fee
TAKER_FEE_RATE = 0.07     # 7% taker fee
TRAIN_FRACTION = 0.70     # 70% train, 30% test (chronological split)
# For time-based split: test on most recent N months of data
TEST_MONTHS = 3           # Use last 3 months as out-of-sample test set
MIN_TRADES_SIGNIFICANT = 30  # Need at least 30 trades for z-test


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_settlements(series_filter: str = None) -> list[dict]:
    """Load settled markets from the historical dataset."""
    if not SETTLEMENTS_FILE.exists():
        print(f"ERROR: {SETTLEMENTS_FILE} not found. Run settlement_tracker first.")
        sys.exit(1)

    with open(SETTLEMENTS_FILE) as f:
        data = json.load(f)

    markets = data.get("markets", [])

    # Filter to settled markets with price data
    settled = []
    for m in markets:
        if not m.get("result"):
            continue
        if not m.get("previous_price") and not m.get("prev_yes_ask"):
            continue

        # Parse series from ticker
        ticker = m.get("ticker", "")
        series = ticker.rsplit("-", 2)[0] if "-" in ticker else ticker

        if series_filter and not series.startswith(series_filter):
            continue

        m["series"] = series
        m["yes_price"] = float(m.get("previous_price", 0) or 0)
        m["yes_ask"] = float(m.get("prev_yes_ask", 0) or 0)
        m["yes_bid"] = float(m.get("prev_yes_bid", 0) or 0)
        m["vol"] = float(m.get("volume", 0) or 0)
        m["open_interest"] = float(m.get("open_interest", 0) or 0)
        m["last_price_dollars"] = float(m.get("last_price", 0) or 0)
        m["settled_yes"] = m["result"] == "yes"

        settled.append(m)

    # Sort by close_time for chronological split
    settled.sort(key=lambda x: x.get("close_time", ""))
    return settled


def refresh_settlements():
    """Pull fresh settlement data from Kalshi API."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from kalshi_client import KalshiClient
        import config

        client = KalshiClient()
        print("Refreshing settlement data from Kalshi API...")

        all_markets = []
        for series in config.TARGET_MARKET_SERIES:
            cursor = None
            for _ in range(20):  # max 20 pages per series
                kwargs = {
                    "series_ticker": series,
                    "status": "settled",
                    "limit": 200,
                }
                if cursor:
                    kwargs["cursor"] = cursor
                try:
                    resp = client.get_markets(**kwargs)
                except Exception as e:
                    print(f"  {series}: error {e}")
                    break

                markets = resp.get("markets", [])
                if not markets:
                    break

                for m in markets:
                    record = {
                        "ticker": m.get("ticker", ""),
                        "series": m.get("series_ticker", ""),
                        "title": m.get("title", ""),
                        "result": m.get("result", ""),
                        "previous_price": str(m.get("previous_yes_price", m.get("last_price_dollars", 0)) or 0),
                        "prev_yes_ask": str(m.get("previous_yes_ask", m.get("yes_ask_dollars", 0)) or 0),
                        "prev_yes_bid": str(m.get("previous_yes_bid", m.get("yes_bid_dollars", 0)) or 0),
                        "volume": str(m.get("volume", m.get("volume_fp", 0)) or 0),
                        "open_interest": str(m.get("open_interest", 0) or 0),
                        "last_price": str(m.get("last_price_dollars", 0) or 0),
                        "close_time": m.get("close_time", ""),
                    }
                    if record["result"]:
                        all_markets.append(record)

                cursor = resp.get("cursor")
                if not cursor:
                    break

            print(f"  {series}: {sum(1 for m in all_markets if m.get('series', '').startswith(series))} settled markets")

        # Merge with existing data (keep markets not in this fetch)
        existing = {}
        if SETTLEMENTS_FILE.exists():
            with open(SETTLEMENTS_FILE) as f:
                data = json.load(f)
            for m in data.get("markets", []):
                existing[m.get("ticker", "")] = m

        # Update with fresh data
        for m in all_markets:
            existing[m["ticker"]] = m

        output = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "total": len(existing),
            "markets": list(existing.values()),
        }

        with open(SETTLEMENTS_FILE, "w") as f:
            json.dump(output, f)

        print(f"Done. {len(existing)} total markets ({len(all_markets)} refreshed).")

    except Exception as e:
        print(f"Cannot refresh: {e}")
        import traceback
        traceback.print_exc()
        print("Using existing data.")


# ---------------------------------------------------------------------------
# Fee Calculation
# ---------------------------------------------------------------------------
def maker_fee_cents(price_cents: float) -> float:
    """Calculate maker fee in cents for a given price."""
    price_frac = price_cents / 100.0
    fee_frac = MAKER_FEE_RATE * price_frac * (1 - price_frac)
    return fee_frac * 100.0


# ---------------------------------------------------------------------------
# Strategy Evaluation
# ---------------------------------------------------------------------------
def evaluate_strategy(markets: list[dict]) -> dict:
    """
    Run the strategy from candidate_strategy.py against historical markets.

    For each market, the strategy decides: buy_yes, buy_no, or skip.
    We simulate the trade and track P&L.

    Returns dict with metrics.
    """
    # Import strategy module fresh (picks up any changes)
    import importlib
    strat_path = Path(__file__).parent / "candidate_strategy.py"
    spec = importlib.util.spec_from_file_location("candidate_strategy", strat_path)
    strat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strat)

    trades = []

    for m in markets:
        ticker = m["ticker"]
        series = m["series"]
        yes_price = m["yes_price"]  # in dollars (0.00 - 1.00)
        yes_ask = m["yes_ask"]
        yes_bid = m["yes_bid"]
        settled_yes = m["settled_yes"]
        vol = m["vol"]

        yes_cents = yes_price * 100 if yes_price < 1.5 else yes_price
        ask_cents = yes_ask * 100 if yes_ask < 1.5 else yes_ask
        bid_cents = yes_bid * 100 if yes_bid < 1.5 else yes_bid

        # Skip dead markets
        if ask_cents <= 0 and bid_cents <= 0:
            continue

        # Call the strategy's evaluate function if it exists
        decision = None
        oi = m.get("open_interest", 0)
        last_price = m.get("last_price_dollars", 0)
        prev_price = m.get("yes_price", 0)  # previous_price (before settlement)
        if hasattr(strat, "evaluate_market"):
            try:
                decision = strat.evaluate_market(
                    ticker=ticker,
                    series=series,
                    yes_cents=yes_cents,
                    ask_cents=ask_cents,
                    bid_cents=bid_cents,
                    volume=vol,
                    open_interest=oi,
                    last_price=last_price,
                    previous_price=prev_price,
                    settled_yes=settled_yes,
                )
            except TypeError:
                # Fallback for old signature
                try:
                    decision = strat.evaluate_market(
                        ticker=ticker,
                        series=series,
                        yes_cents=yes_cents,
                        ask_cents=ask_cents,
                        bid_cents=bid_cents,
                        volume=vol,
                        open_interest=oi,
                        settled_yes=settled_yes,
                    )
                except TypeError:
                    decision = strat.evaluate_market(
                        ticker=ticker,
                        series=series,
                        yes_cents=yes_cents,
                        ask_cents=ask_cents,
                        bid_cents=bid_cents,
                        volume=vol,
                        settled_yes=settled_yes,
                    )

        if decision is None:
            # Fallback: use the built-in tail/underdog logic from params
            decision = _default_strategy(strat, ticker, series,
                                         yes_cents, ask_cents, bid_cents, vol)

        if decision is None or decision.get("action") == "skip":
            continue

        # Simulate the trade
        side = decision["action"]  # "buy_yes" or "buy_no"
        contracts = decision.get("contracts", 1)

        if side == "buy_yes":
            entry_cents = ask_cents if ask_cents > 0 else yes_cents
            fee = maker_fee_cents(entry_cents)
            cost = (entry_cents + fee) * contracts
            if settled_yes:
                pnl = (100 - entry_cents - fee) * contracts
            else:
                pnl = -(entry_cents + fee) * contracts
        elif side == "buy_no":
            no_price = 100 - bid_cents if bid_cents > 0 else 100 - yes_cents
            fee = maker_fee_cents(no_price)
            cost = (no_price + fee) * contracts
            if not settled_yes:
                pnl = (100 - no_price - fee) * contracts
            else:
                pnl = -(no_price + fee) * contracts
        else:
            continue

        trades.append({
            "ticker": ticker,
            "series": series,
            "side": side,
            "entry_cents": entry_cents if side == "buy_yes" else no_price,
            "contracts": contracts,
            "pnl_cents": pnl,
            "won": pnl > 0,
            "close_time": m.get("close_time", ""),
        })

    return _compute_metrics(trades)


def _default_strategy(strat, ticker, series, yes_cents, ask_cents, bid_cents, vol):
    """
    Default strategy logic using candidate_strategy.py parameters.
    Implements weather tail NO and NBA underdog YES.
    """
    # Weather tail NO: buy NO on cheap YES weather markets
    if series.startswith("KXHIGH"):
        max_yes = getattr(strat, "WEATHER_TAIL_MAX_YES", 15)
        min_no_prob = getattr(strat, "WEATHER_TAIL_MIN_NO_PROB", 0.90)
        max_contracts = getattr(strat, "WEATHER_TAIL_MAX_CONTRACTS", 3)

        if 0 < yes_cents <= max_yes and vol >= 10:
            return {"action": "buy_no", "contracts": max_contracts}

    # Crypto tail NO
    if series.startswith(("KXBTC", "KXETH", "KXSOL")):
        max_yes = getattr(strat, "WEATHER_TAIL_MAX_YES", 15)
        if 0 < yes_cents <= max_yes and vol >= 10:
            return {"action": "buy_no", "contracts": 1}

    # NBA underdog YES
    if series.startswith("KXNBAGAME"):
        min_price = getattr(strat, "UNDERDOG_MIN_PRICE", 8)
        max_price = getattr(strat, "UNDERDOG_MAX_PRICE", 20)
        max_contracts = getattr(strat, "UNDERDOG_MAX_CONTRACTS", 20)

        # Only game winner markets (not props/totals/spreads)
        if any(kw in ticker.upper() for kw in ["PTS", "TOTAL", "SPREAD", "MENTION"]):
            return None

        if min_price <= yes_cents <= max_price and vol >= 50:
            return {"action": "buy_yes", "contracts": min(max_contracts, 5)}

    return None


def _compute_metrics(trades: list[dict]) -> dict:
    """Compute standardized metrics from a list of simulated trades."""
    if not trades:
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "total_pnl_cents": 0, "total_pnl_dollars": 0,
            "avg_pnl_cents": 0, "sharpe": 0, "sortino": 0,
            "max_drawdown_cents": 0, "z_score": 0, "profit_factor": 0,
            "trades_by_series": {},
        }

    pnls = [t["pnl_cents"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p <= 0)
    total_pnl = sum(pnls)
    avg_pnl = np.mean(pnls)
    std_pnl = np.std(pnls, ddof=1) if len(pnls) > 1 else 1.0

    # Sharpe ratio
    sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0

    # Sortino ratio (downside deviation only)
    downside = [p for p in pnls if p < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1.0
    sortino = avg_pnl / downside_std if downside_std > 0 else 0

    # Max drawdown
    cumsum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumsum)
    drawdowns = running_max - cumsum
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0

    # Z-score: is the mean P&L significantly > 0?
    z_score = avg_pnl / (std_pnl / math.sqrt(len(pnls))) if std_pnl > 0 and len(pnls) > 0 else 0

    # Profit factor
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Per-series breakdown
    by_series = defaultdict(list)
    for t in trades:
        by_series[t["series"]].append(t["pnl_cents"])

    series_metrics = {}
    for s, s_pnls in by_series.items():
        s_wins = sum(1 for p in s_pnls if p > 0)
        series_metrics[s] = {
            "trades": len(s_pnls),
            "pnl_cents": sum(s_pnls),
            "win_rate": s_wins / len(s_pnls) * 100 if s_pnls else 0,
        }

    return {
        "total_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(trades) * 100 if trades else 0,
        "total_pnl_cents": total_pnl,
        "total_pnl_dollars": total_pnl / 100,
        "avg_pnl_cents": avg_pnl,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_cents": max_dd,
        "z_score": z_score,
        "profit_factor": profit_factor,
        "trades_by_series": series_metrics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Backtest harness for autoresearch")
    parser.add_argument("--refresh", action="store_true", help="Pull fresh data from Kalshi")
    parser.add_argument("--series", type=str, default=None, help="Filter to specific series")
    args = parser.parse_args()

    if args.refresh:
        refresh_settlements()

    print("=" * 70)
    print("  BACKTEST HARNESS — Strategy Evaluation")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 70)

    # Load data
    markets = load_settlements(args.series)
    print(f"\nLoaded {len(markets)} settled markets")

    if len(markets) < 100:
        print("ERROR: Not enough data for meaningful backtest")
        sys.exit(1)

    # Time-based train/test split: test on most recent N months
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    test_cutoff = (now - timedelta(days=TEST_MONTHS * 30)).strftime("%Y-%m-%d")

    train_markets = [m for m in markets if m.get("close_time", "")[:10] < test_cutoff]
    test_markets = [m for m in markets if m.get("close_time", "")[:10] >= test_cutoff]

    # Fallback to fraction-based if time split gives too few test samples
    if len(test_markets) < 100:
        split_idx = int(len(markets) * TRAIN_FRACTION)
        train_markets = markets[:split_idx]
        test_markets = markets[split_idx:]

    train_date_range = (train_markets[0].get("close_time", "?")[:10],
                        train_markets[-1].get("close_time", "?")[:10]) if train_markets else ("?", "?")
    test_date_range = (test_markets[0].get("close_time", "?")[:10],
                       test_markets[-1].get("close_time", "?")[:10]) if test_markets else ("?", "?")

    print(f"Train: {len(train_markets)} markets ({train_date_range[0]} to {train_date_range[1]})")
    print(f"Test:  {len(test_markets)} markets ({test_date_range[0]} to {test_date_range[1]})")

    # Evaluate on train
    print("\n--- IN-SAMPLE (Train) ---")
    train_metrics = evaluate_strategy(train_markets)
    _print_metrics(train_metrics, "TRAIN")

    # Evaluate on test
    print("\n--- OUT_OF_SAMPLE (Test) ---")
    test_metrics = evaluate_strategy(test_markets)
    _print_metrics(test_metrics, "OUT_OF_SAMPLE")

    # Verdict
    print("\n" + "=" * 70)
    _print_verdict(train_metrics, test_metrics)
    print("=" * 70)

    # Machine-readable summary
    print(f"\n---")
    print(f"train_pnl_dollars:   {train_metrics['total_pnl_dollars']:.2f}")
    print(f"test_pnl_dollars:    {test_metrics['total_pnl_dollars']:.2f}")
    print(f"test_trades:         {test_metrics['total_trades']}")
    print(f"test_win_rate:       {test_metrics['win_rate']:.1f}")
    print(f"test_sharpe:         {test_metrics['sharpe']:.4f}")
    print(f"test_z_score:        {test_metrics['z_score']:.4f}")
    print(f"test_profit_factor:  {test_metrics['profit_factor']:.4f}")


def _print_metrics(m: dict, label: str):
    """Print formatted metrics."""
    print(f"  {label} Trades:        {m['total_trades']} ({m['wins']}W / {m['losses']}L)")
    print(f"  {label} Win Rate:      {m['win_rate']:.1f}%")
    print(f"  {label} P&L:           ${m['total_pnl_dollars']:+.2f} ({m['total_pnl_cents']:+.0f}c)")
    print(f"  {label} Avg P&L/trade: {m['avg_pnl_cents']:+.2f}c")
    print(f"  {label} Sharpe:        {m['sharpe']:.4f}")
    print(f"  {label} Sortino:       {m['sortino']:.4f}")
    print(f"  {label} Z-Score:       {m['z_score']:.4f}")
    print(f"  {label} Profit Factor: {m['profit_factor']:.2f}")
    print(f"  {label} Max Drawdown:  {m['max_drawdown_cents']:.0f}c")

    if m["trades_by_series"]:
        print(f"  {label} By Series:")
        for s, sm in sorted(m["trades_by_series"].items(), key=lambda x: x[1]["pnl_cents"], reverse=True):
            print(f"    {s:<12} {sm['trades']:>4} trades  {sm['win_rate']:>5.1f}%  {sm['pnl_cents']:>+8.0f}c")


def _print_verdict(train: dict, test: dict):
    """Print VERDICT: whether this strategy change should be kept."""
    issues = []

    if test["total_trades"] < MIN_TRADES_SIGNIFICANT:
        issues.append(f"Too few test trades ({test['total_trades']} < {MIN_TRADES_SIGNIFICANT})")

    if test["total_pnl_cents"] <= 0:
        issues.append(f"Negative out-of-sample P&L (${test['total_pnl_dollars']:.2f})")

    if test["z_score"] < 1.5:
        issues.append(f"Z-score too low ({test['z_score']:.2f} < 1.5)")

    if train["total_pnl_cents"] > 0 and test["total_pnl_cents"] <= 0:
        issues.append("Train profitable but test negative (overfit)")

    if issues:
        print("  VERDICT: DISCARD")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  VERDICT: KEEP")
        print(f"    + Out-of-sample P&L: ${test['total_pnl_dollars']:+.2f}")
        print(f"    + Z-score: {test['z_score']:.2f} (p < {_z_to_p(test['z_score']):.4f})")
        print(f"    + {test['total_trades']} trades, {test['win_rate']:.1f}% win rate")


def _z_to_p(z: float) -> float:
    """Convert z-score to one-tailed p-value."""
    return 0.5 * (1 + math.erf(-z / math.sqrt(2)))


if __name__ == "__main__":
    main()
