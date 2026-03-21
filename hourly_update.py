"""
hourly_update.py -- Sends a concise hourly Telegram status update.

Tracks deltas since last run via a state file so "new trades" and
"new settlements" reflect what happened in the last hour, not all-time.

Usage:
    python hourly_update.py
    python cli.py hourly-update
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
TRADE_HISTORY = OUTPUT / "trade_history.csv"
STATE_FILE = OUTPUT / ".hourly_state.json"


def _load_state() -> dict:
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text())
    except Exception:
        pass
    return {"last_total_trades": 0, "last_settled_today": 0}


def _save_state(state: dict):
    try:
        STATE_FILE.write_text(json.dumps(state))
    except Exception:
        pass


def _read_trade_history() -> list[dict]:
    if not TRADE_HISTORY.exists():
        return []
    try:
        with open(TRADE_HISTORY) as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _get_account_info() -> dict:
    try:
        client = KalshiClient()
        bal = client.get_balance()
        pos = client.get_positions()

        cash = bal.get("balance", 0) / 100.0
        portfolio = bal.get("portfolio_value", 0) / 100.0
        equity = cash + portfolio

        positions = [
            p for p in pos.get("market_positions", [])
            if float(p.get("position_fp", 0)) != 0
        ]

        has_weather = any(
            p.get("ticker", "").startswith(s)
            for p in positions
            for s in ["KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHLA", "KXHIGHDC", "KXHIGHDEN"]
        )
        has_crypto = any(
            p.get("ticker", "").startswith(s)
            for p in positions
            for s in ["KXBTC", "KXETH", "KXSOL"]
        )
        has_sports = any(
            p.get("ticker", "").startswith(s)
            for p in positions
            for s in ["KXNBA", "KXNFL", "KXMLB", "KXNHL"]
        )

        return {
            "equity": equity,
            "position_count": len(positions),
            "has_weather": has_weather,
            "has_crypto": has_crypto,
            "has_sports": has_sports,
        }
    except Exception as e:
        return {"error": str(e)}


def _strategy_active(rows: list[dict], account: dict, today: str) -> dict:
    """
    A strategy is "active" if it traded today OR has open positions.
    Strategy field values from settlement_tracker: weather / btc / sports.
    """
    today_strategies = {r.get("strategy", "") for r in rows if r.get("date", "") == today}
    return {
        "weather": "weather" in today_strategies or account.get("has_weather", False),
        "crypto": "btc" in today_strategies or account.get("has_crypto", False),
        "sports": "sports" in today_strategies or account.get("has_sports", False),
    }


def send_hourly_update():
    """Build and send the hourly status update via Telegram."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    hour = datetime.now(timezone.utc).strftime("%H:%M UTC")

    state = _load_state()
    account = _get_account_info()
    all_rows = _read_trade_history()

    today_rows = [r for r in all_rows if r.get("date", "") == today]
    settled_today = [
        r for r in today_rows
        if r.get("settlement_result") not in ("open", "", None)
    ]

    wins = sum(1 for r in settled_today if float(r.get("pnl", 0)) > 0)
    losses = len(settled_today) - wins
    day_pnl = sum(float(r.get("pnl", 0)) for r in settled_today)

    # Deltas since last hourly run
    new_trades = max(0, len(all_rows) - state.get("last_total_trades", 0))
    new_settled = max(0, len(settled_today) - state.get("last_settled_today", 0))

    # Build message parts
    if "error" in account:
        balance_line = "Balance: unavailable"
        pnl_line = ""
        strategies_line = ""
    else:
        equity = account["equity"]
        day_sign = "+" if day_pnl >= 0 else ""
        day_pct = (day_pnl / equity * 100) if equity > 0 else 0.0
        balance_line = f"Balance: ${equity:.2f} | Positions: {account['position_count']} open"
        pnl_line = f"Today: {day_sign}${day_pnl:.2f} ({day_sign}{day_pct:.1f}%)"

        active = _strategy_active(all_rows, account, today)
        w = "✅" if active["weather"] else "💤"
        c = "✅" if active["crypto"] else "💤"
        s = "✅" if active["sports"] else "💤"
        strategies_line = f"Strategies: weather {w} crypto {c} sports {s}"

    settled_line = f"Settled: {wins}W / {losses}L"
    if new_settled > 0:
        settled_line += f" (+{new_settled} new)"

    trades_line = f"New trades: {new_trades} placed"

    paused = (OUTPUT / ".trading_paused").exists()
    paused_note = "\n⏸ <b>Trading paused</b>" if paused else ""

    msg = f"🕐 <b>Hourly Update</b>  {hour}\n{balance_line}\n"
    if pnl_line:
        msg += f"{pnl_line}\n"
    msg += f"{settled_line}\n{trades_line}\n"
    if strategies_line:
        msg += strategies_line
    msg += paused_note

    _send_telegram(msg)

    _save_state({
        "last_total_trades": len(all_rows),
        "last_settled_today": len(settled_today),
    })


if __name__ == "__main__":
    send_hourly_update()
