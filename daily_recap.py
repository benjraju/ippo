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


def _get_positions_by_strategy() -> dict:
    """Group open positions by strategy with cost basis from trade history."""
    rows = _read_trade_history()
    open_rows = [r for r in rows if r.get("settlement_result") == "open"]
    buckets: dict[str, dict] = {}
    for r in open_rows:
        strat = r.get("strategy", "other")
        if strat not in buckets:
            buckets[strat] = {"count": 0, "cost": 0.0}
        buckets[strat]["count"] += int(r.get("contracts", 1))
        entry = float(r.get("entry_price", 0))
        contracts = int(r.get("contracts", 1))
        buckets[strat]["cost"] += (entry / 100.0) * contracts
    return buckets


def _get_strategy_hit_rates(rows: list[dict]) -> dict:
    """Compute hit rate per strategy across all settled trades."""
    strats: dict[str, dict] = {}
    for r in rows:
        if r.get("settlement_result") == "open":
            continue
        strat = r.get("strategy", "other")
        if strat not in strats:
            strats[strat] = {"wins": 0, "total": 0}
        strats[strat]["total"] += 1
        if float(r.get("pnl", 0)) > 0:
            strats[strat]["wins"] += 1
    return strats


def _strategy_label(key: str) -> str:
    """Human-readable strategy name."""
    labels = {
        "weather": "Weather",
        "crypto": "Crypto",
        "btc": "Crypto",
        "sports": "Sports",
        "nba": "NBA",
        "arb": "Arbitrage",
        "other": "Other",
    }
    return labels.get(key, key.title())


def send_daily_recap():
    """Build and send the daily KPI recap via Telegram."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    display_date = datetime.now(timezone.utc).strftime("%B %d, %Y")

    # ── Account ──────────────────────────────────
    equity, open_count = _get_account_equity()
    try:
        client = KalshiClient()
        bal = client.get_balance()
        cash = bal.get("balance", 0) / 100.0
        portfolio = bal.get("portfolio_value", 0) / 100.0
    except Exception:
        cash, portfolio = equity, 0.0

    # ── Trades ───────────────────────────────────
    all_rows = _read_trade_history()
    day_pnl = _get_day_pnl(all_rows, today)
    today_trades = _get_today_trades(all_rows, today)
    settled_today = _get_settled_today(all_rows, today)
    placed_count = len(today_trades)
    settled_count = len(settled_today)

    wins_today, losses_today, _ = _win_rate(settled_today)

    # Day P&L percentage
    base = equity - day_pnl if equity - day_pnl != 0 else 1
    day_pct = (day_pnl / base) * 100

    # MTD return
    month_start = today[:8] + "01"
    mtd_settled = [
        r for r in all_rows
        if r.get("date", "") >= month_start
        and r.get("settlement_result") != "open"
    ]
    mtd_pnl = sum(float(r.get("pnl", 0)) for r in mtd_settled)
    mtd_base = equity - mtd_pnl if equity - mtd_pnl != 0 else 1
    mtd_pct = (mtd_pnl / mtd_base) * 100

    # Deployment ratio
    deployed = portfolio
    deploy_pct = (deployed / equity * 100) if equity > 0 else 0

    # ── Winners / Losers ─────────────────────────
    winners_pnl = sum(float(r.get("pnl", 0)) for r in settled_today if float(r.get("pnl", 0)) > 0)
    losers_pnl = sum(float(r.get("pnl", 0)) for r in settled_today if float(r.get("pnl", 0)) <= 0)
    net_settled = winners_pnl + losers_pnl

    # ── Top movers ───────────────────────────────
    sorted_settled = sorted(settled_today, key=lambda r: float(r.get("pnl", 0)), reverse=True)
    top_movers_lines = []
    for r in sorted_settled[:3]:
        pnl_val = float(r.get("pnl", 0))
        ticker = r.get("ticker", "???")
        side = r.get("side", "").upper()
        # Shorten ticker for display
        short = ticker
        if len(short) > 22:
            short = short[:19] + "..."
        arrow = "\u2191" if pnl_val > 0 else "\u2193"
        result = "won" if pnl_val > 0 else "lost"
        top_movers_lines.append(
            f"  {arrow} {short} {side:<4} {_fmt_sign(pnl_val):>8}  ({result})"
        )

    # ── Book composition ─────────────────────────
    pos_buckets = _get_positions_by_strategy()
    book_lines = []
    total_deployed_cost = 0.0
    for strat, data in sorted(pos_buckets.items()):
        label = _strategy_label(strat)
        count = data["count"]
        cost = data["cost"]
        total_deployed_cost += cost
        book_lines.append(f"  {label:<16} {count:>3} contracts  ${cost:>7.2f}")
    book_lines.append(f"  {'Cash':<16}               ${cash:>7.2f}")

    # ── Strategy alpha ───────────────────────────
    strat_rates = _get_strategy_hit_rates(all_rows)
    alpha_lines = []
    for strat, data in sorted(strat_rates.items()):
        label = _strategy_label(strat)
        total = data["total"]
        wins = data["wins"]
        rate = (wins / total * 100) if total > 0 else 0
        alpha_lines.append(f"  {label:<16} {rate:>3.0f}% hit rate  ({wins}/{total} settled)")

    # ── Research ─────────────────────────────────
    research = _get_research_stats(today)
    research_lines = [
        f"  Experiments: {research['today_experiments']:,}"
        f" | Improvements: {research['today_improvements']}"
    ]
    if research["strategy_changes"]:
        latest = research["strategy_changes"][-1]
        research_lines.append(f"  Latest: {latest}")

    # ── All-time stats ───────────────────────────
    all_settled = [r for r in all_rows if r.get("settlement_result") != "open"]
    total_pnl = sum(float(r.get("pnl", 0)) for r in all_settled)
    _, _, wr_all = _win_rate(all_rows)
    total_trades = len(all_settled)

    # ── Max open risk ────────────────────────────
    open_rows = [r for r in all_rows if r.get("settlement_result") == "open"]
    max_open_risk = sum(
        (float(r.get("entry_price", 0)) / 100.0) * int(r.get("contracts", 1))
        for r in open_rows
    )

    # ══════════════════════════════════════════════
    # BUILD MESSAGE
    # ══════════════════════════════════════════════
    sep = "\u2501" * 22
    lines = [
        f"<pre>",
        sep,
        f"  IPPO CAPITAL \u2014 DAILY BRIEF",
        f"  {display_date}",
        sep,
        f"",
        f"FUND OVERVIEW",
        f"  NAV:          ${equity:>10.2f}",
        f"  Day P&L:      {_fmt_sign(day_pnl):>7} ({day_pct:+.1f}%)",
        f"  MTD Return:   {_fmt_sign(mtd_pnl):>7} ({mtd_pct:+.1f}%)",
        f"  Deployment:   {deploy_pct:.0f}% (${deployed:.2f} of ${equity:.2f})",
        f"",
        f"BOOK COMPOSITION",
    ]
    lines.extend(book_lines)
    lines += [
        f"",
        f"TODAY'S ACTIVITY",
        f"  Placed:    {placed_count:>3} orders",
        f"  Settled:   {settled_count:>3} ({wins_today}W / {losses_today}L)",
        f"  Winners:   {_fmt_sign(winners_pnl):>8}",
        f"  Losers:    {_fmt_sign(losers_pnl):>8}",
        f"  Net:       {_fmt_sign(net_settled):>8}",
    ]
    if top_movers_lines:
        lines += [f"", f"TOP MOVERS"]
        lines.extend(top_movers_lines)
    if alpha_lines:
        lines += [f"", f"STRATEGY ALPHA"]
        lines.extend(alpha_lines)
    lines += [f"", f"RESEARCH"]
    lines.extend(research_lines)
    lines += [
        f"",
        f"ALL-TIME",
        f"  Total P&L:    {_fmt_sign(total_pnl):>8}",
        f"  Win Rate:     {wr_all:>7.0f}%",
        f"  Settled:      {total_trades:>7}",
        f"",
        sep,
        f"  Risk: {deploy_pct:.0f}% deployed"
        f" | Max loss: {_fmt_sign(-max_open_risk)}",
        sep,
        f"</pre>",
    ]

    message = "\n".join(lines)
    _send_telegram(message)

    # ── Strategy summary (second message) ────────
    try:
        from strategy_doc import generate_strategy_doc
        generate_strategy_doc()

        strategy_path = Path(__file__).parent / "STRATEGY.md"
        if strategy_path.exists():
            doc = strategy_path.read_text()
            sections = doc.split("\n## ")
            summary_parts = ["<b>STRATEGY SUMMARY</b>\n"]

            for section in sections:
                title = section.split("\n")[0].strip()
                if title in ("How Ippo Makes Money", "Current Strategy Settings",
                             "What AutoResearch Has Learned"):
                    content = "\n".join(section.split("\n")[1:]).strip()
                    if len(content) > 600:
                        content = content[:597] + "..."
                    summary_parts.append(f"<b>{title}</b>\n{content}\n")

            strategy_msg = "\n".join(summary_parts)
            if len(strategy_msg) > 4000:
                strategy_msg = strategy_msg[:3997] + "..."
            _send_telegram(strategy_msg)
    except Exception:
        pass


if __name__ == "__main__":
    send_daily_recap()
