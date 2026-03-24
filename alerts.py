"""
alerts.py -- Minimal notification helpers for the Kalshi trading bot.

Provides send_telegram() for other modules. All per-event spam
(trade alerts, research improvements, big edges) has been removed.
The telegram_bot.py 30-minute scheduled update is the notification system.

Functions kept for backward compatibility (called from auto_trade.py, arb_runner.py):
  - alert_trade_settled   -> logs to file only
  - alert_drawdown        -> sends telegram (safety-critical)
  - alert_bot_error       -> sends telegram (errors matter)
  - alert_big_edge        -> logs to file only
  - alert_daily_summary   -> logs to file only
  - alert_research_improvement -> logs to file only
  - send_trade_alert      -> logs to file only
  - send_settlement_alert -> logs to file only
  - notify                -> logs to file only
"""

import json
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

ALERT_LOG = Path(__file__).parent / "output" / "alerts.log"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Kept for backward compat with telegram_bot.py /set command (now removed,
# but auto_trade.py may still reference it)
_PARAM_EXPLAIN = {
    "EDGE_THRESHOLD_CENTS": {"name": "Minimum edge to trade"},
    "NWS_OFFICIAL_WEIGHT": {"name": "Weather source blend"},
    "CONTRACTS_PER_TRADE": {"name": "Trade size"},
}


def _log(title: str, message: str):
    """Append to alerts.log."""
    try:
        entry = {"ts": datetime.now().isoformat(), "title": title, "message": message}
        with open(ALERT_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def send_telegram(text: str):
    """Send a Telegram message. Used by other modules that need to notify."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        import requests
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception:
        pass


# Kept as _send_telegram alias for auto_trade.py backward compat
_send_telegram = send_telegram


# ── backward-compatible stubs (log only, no telegram spam) ───

def notify(title: str, message: str, sound: bool = False):
    _log(title, message)


def alert_trade_settled(ticker: str, side: str, pnl: float, won: bool):
    _log("Settlement", f"{'WIN' if won else 'LOSS'} {ticker} {side} {pnl:+.2f}")


def alert_drawdown(current_dd_pct: float, balance: float):
    """Drawdown is safety-critical -- still sends telegram."""
    _log("Drawdown Warning", f"{current_dd_pct:.1f}% -- ${balance:.2f}")
    send_telegram(
        f"<b>DRAWDOWN WARNING</b>\n\n"
        f"Account is down {current_dd_pct:.1f}% today\n"
        f"Balance: ${balance:.2f}\n\n"
        f"Auto-pause at 8% daily loss."
    )


def alert_bot_error(component: str, error: str):
    """Errors are important -- still sends telegram."""
    _log(f"Error: {component}", error[:200])
    send_telegram(f"<b>ERROR</b> in {component}\n\n{error[:200]}")


def alert_big_edge(ticker: str, edge_cents: float, city: str = ""):
    _log("Big Edge", f"{ticker}: +{edge_cents:.0f}c")


def alert_daily_summary(trades: int, pnl: float, win_rate: float):
    _log("Daily Summary", f"{trades} trades, ${pnl:+.2f}")


def alert_research_improvement(param: str, old_val, new_val, score: float):
    _log("Research", f"{param}: {old_val} -> {new_val} (score: {score:.2f})")


def send_trade_alert(trade_details: dict):
    _log("Trade", str(trade_details)[:200])


def send_settlement_alert(trade_details: dict):
    _log("Settlement", str(trade_details)[:200])
