"""
daily_recap.py -- Generates and sends a daily KPI recap via Telegram.

Summarises account status, trades, research progress, and strategy changes
into a single end-of-day message.

Usage:
    python daily_recap.py
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import config
from kalshi_client import KalshiClient
from alerts import _send_telegram

OUTPUT = config.OUTPUT_DIR
RESULTS_LOG = config.AUTORESEARCH_DIR / "results.log"
TRADE_HISTORY = OUTPUT / "trade_history.csv"


def _read_trade_history() -> list[dict]:
    """Read all rows from trade_history.csv."""
    if not TRADE_HISTORY.exists():
        return []
    try:
        with open(TRADE_HISTORY) as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _get_account_equity() -> tuple[float, int]:
    """Return (equity_dollars, open_position_count)."""
    try:
        client = KalshiClient()
        bal = client.get_balance()
        pos = client.get_positions()
        cash = bal.get("balance", 0) / 100.0
        portfolio = bal.get("portfolio_value", 0) / 100.0
        open_positions = [
            mp for mp in pos.get("market_positions", [])
            if float(mp.get("position_fp", 0)) != 0
        ]
        return cash + portfolio, len(open_positions)
    except Exception:
        return 0.0, 0


def _get_day_pnl(rows: list[dict], today: str) -> float:
    """Sum pnl for rows matching today's date."""
    return sum(
        float(r.get("pnl", 0))
        for r in rows
        if r.get("date", "") == today and r.get("settlement_result") != "open"
    )


def _get_today_trades(rows: list[dict], today: str) -> list[dict]:
    """Filter rows placed today."""
    return [r for r in rows if r.get("date", "") == today]


def _get_settled_today(rows: list[dict], today: str) -> list[dict]:
    """Filter rows that settled today (date matches and not open)."""
    return [
        r for r in rows
        if r.get("date", "") == today and r.get("settlement_result") != "open"
    ]


def _win_rate(rows: list[dict]) -> tuple[int, int, float]:
    """Return (wins, losses, win_rate_pct) for settled rows."""
    settled = [r for r in rows if r.get("settlement_result") != "open"]
    wins = sum(1 for r in settled if float(r.get("pnl", 0)) > 0)
    losses = sum(1 for r in settled if float(r.get("pnl", 0)) <= 0)
    total = wins + losses
    rate = (wins / total * 100) if total > 0 else 0.0
    return wins, losses, rate


def _best_worst_trade(settled: list[dict]) -> tuple[str, str]:
    """Return formatted best and worst trade strings."""
    if not settled:
        return "n/a", "n/a"

    best = max(settled, key=lambda r: float(r.get("pnl", 0)))
    worst = min(settled, key=lambda r: float(r.get("pnl", 0)))

    def _fmt(row: dict) -> str:
        pnl = float(row.get("pnl", 0))
        sign = "+" if pnl >= 0 else ""
        ticker = row.get("ticker", "???")
        # Truncate long tickers
        if len(ticker) > 20:
            ticker = ticker[:17] + "..."
        return f"{sign}${pnl:.2f} ({ticker})"

    return _fmt(best), _fmt(worst)


def _get_research_stats(today: str) -> dict:
    """Parse autoresearch/results.log for experiment counts and strategy changes."""
    stats = {
        "total_experiments": 0,
        "total_improvements": 0,
        "today_experiments": 0,
        "today_improvements": 0,
        "strategy_changes": [],
    }

    if not RESULTS_LOG.exists():
        return stats

    try:
        for line in RESULTS_LOG.read_text().strip().split("\n"):
            if not line.strip():
                continue
            stats["total_experiments"] += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            kept = entry.get("kept", False)
            if kept:
                stats["total_improvements"] += 1

            ts = entry.get("timestamp", "")
            if ts.startswith(today):
                stats["today_experiments"] += 1
                if kept:
                    stats["today_improvements"] += 1
                    param = entry.get("parameter", "?")
                    old = entry.get("old_value", "?")
                    new = entry.get("new_value", "?")
                    stats["strategy_changes"].append(
                        f"{param}: {old} -> {new}"
                    )
    except Exception:
        pass

    return stats


def _fmt_sign(val: float) -> str:
    """Format a number with explicit +/- sign."""
    sign = "+" if val >= 0 else ""
    return f"{sign}${val:.2f}"


def send_daily_recap():
    """Build and send the daily KPI recap via Telegram."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    display_date = datetime.now(timezone.utc).strftime("%B %d, %Y")

    # --- Account ---
    equity, open_count = _get_account_equity()

    # --- Trades ---
    all_rows = _read_trade_history()
    day_pnl = _get_day_pnl(all_rows, today)
    today_trades = _get_today_trades(all_rows, today)
    settled_today = _get_settled_today(all_rows, today)
    placed_count = len(today_trades)
    settled_count = len(settled_today)

    wins_today, losses_today, wr_today = _win_rate(settled_today)
    best, worst = _best_worst_trade(settled_today)

    # --- All time ---
    all_settled = [r for r in all_rows if r.get("settlement_result") != "open"]
    total_pnl = sum(float(r.get("pnl", 0)) for r in all_settled)
    _, _, wr_all = _win_rate(all_rows)
    total_trades = len(all_rows)

    # --- Research ---
    research = _get_research_stats(today)

    # --- Build message ---
    lines = [
        f"<b>DAILY RECAP</b> -- {display_date}",
        "",
        "<b>ACCOUNT</b>",
        f"  Equity:     ${equity:.2f}",
        f"  Day P&L:    {_fmt_sign(day_pnl)}",
        f"  Open:       {open_count} positions",
        "",
        "<b>TRADES TODAY</b>",
        f"  Placed: {placed_count}  |  Settled: {settled_count}",
        f"  Won: {wins_today}  |  Lost: {losses_today}",
        f"  Win Rate: {wr_today:.0f}%",
        f"  Best:  {best}",
        f"  Worst: {worst}",
        "",
        "<b>RESEARCH</b>",
        f"  Experiments today: {research['today_experiments']:,}",
        f"  Improvements: {research['today_improvements']}",
    ]

    if research["strategy_changes"]:
        lines.append("  Strategy changes:")
        for change in research["strategy_changes"]:
            lines.append(f"    - {change}")
    else:
        lines.append("  Strategy changes: none")

    lines += [
        "",
        "<b>ALL TIME</b>",
        f"  Total P&L:    {_fmt_sign(total_pnl)}",
        f"  Win Rate:     {wr_all:.0f}%",
        f"  Total Trades: {total_trades}",
    ]

    message = "\n".join(lines)
    _send_telegram(message)


if __name__ == "__main__":
    send_daily_recap()
