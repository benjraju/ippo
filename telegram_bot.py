"""
telegram_bot.py -- Telegram bot for Ippo trading bot.

30-minute scheduled portfolio updates + simple commands.
Commands: /status, /pnl, /pause, /resume, /help
"""

import csv
import json
import logging
import os
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import requests as http_requests

sys.path.insert(0, str(Path(__file__).parent))
import config
from kalshi_client import KalshiClient

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("telegram_bot")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
API = f"https://api.telegram.org/bot{TOKEN}"
OUTPUT = config.OUTPUT_DIR
AUTORESEARCH = config.AUTORESEARCH_DIR
PAUSE_FILE = OUTPUT / ".trading_paused"
TRADE_HISTORY = OUTPUT / "trade_history.csv"
RESULTS_LOG = AUTORESEARCH / "results.log"

UPDATE_INTERVAL = 30 * 60  # 30 minutes


# ── helpers ──────────────────────────────────────────────────

def send(text: str):
    """Send a Telegram message."""
    if not TOKEN or not CHAT_ID:
        return
    try:
        http_requests.post(f"{API}/sendMessage", json={
            "chat_id": CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
        }, timeout=10)
    except Exception as e:
        log.error(f"Send failed: {e}")


def _classify_strategy(ticker: str, strategy: str, side: str) -> str:
    """Map a trade to a human-readable strategy tag."""
    t = ticker.upper()
    s = strategy.lower() if strategy else ""
    if "arb" in s:
        return "Arb"
    if "dutch" in s:
        return "Dutch Book"
    if "deep_itm" in s:
        return "Deep ITM"
    if t.startswith("KXNBA") or "nba" in s or "sports" in s:
        return "NBA Underdog"
    if t.startswith(("KXBTC", "KXETH", "KXSOL")):
        return "Crypto Tail"
    if t.startswith("KXHIGH"):
        if "tail" in s:
            return "Weather Tail"
        return "Weather"
    return "Other"


def _trade_reason(ticker: str, strategy: str, side: str, price: float) -> str:
    """One-line explanation of why we took this trade."""
    tag = _classify_strategy(ticker, strategy, side)
    price_c = int(price) if price else 0

    if tag == "Arb":
        return "YES+NO sum < $1, guaranteed profit"
    if tag == "Dutch Book":
        return "Multi-outcome arb, sum(YES) > $1"
    if tag == "Deep ITM":
        return "Near-certain outcome, maker limit at 95c"
    if tag == "NBA Underdog":
        if side.lower() == "yes":
            return f"Underdog at {price_c}c, favorite-longshot bias edge"
        return f"Underdog at {100 - price_c}c, favorite-longshot bias edge"
    if tag == "Crypto Tail":
        if side.lower() == "no":
            yes_c = 100 - price_c
            return f"YES at {yes_c}c, tail overpriced, model says <1%"
        return f"Cheap tail at {price_c}c, asymmetric upside"
    if tag == "Weather Tail":
        if side.lower() == "no":
            yes_c = 100 - price_c
            return f"YES at {yes_c}c, tail overpriced, model says <1%"
        return f"Cheap tail at {price_c}c, asymmetric upside"
    if tag == "Weather":
        return f"Forecast disagrees with market price at {price_c}c"
    return f"Edge detected at {price_c}c"


def _shorten_ticker(ticker: str) -> str:
    """Make a ticker human-readable. e.g. KXNBAGAME-26MAR23HOUCHI-CHI -> Houston@Chicago"""
    t = ticker.upper()
    # NBA games
    if "NBAGAME" in t:
        parts = t.split("-")
        if len(parts) >= 3:
            # Last part is the side (team picked)
            return parts[-1]
        return t
    # Weather: KXHIGHNY-26MAR24-B48.5 -> NYC 48-49
    if t.startswith("KXHIGH"):
        city_map = {"NY": "NYC", "CHI": "Chicago", "MIA": "Miami",
                    "LA": "LA", "DC": "DC", "DEN": "Denver"}
        rest = t.replace("KXHIGH", "")
        for code, name in city_map.items():
            if rest.startswith(code):
                # Extract temp from the last part
                parts = t.split("-")
                temp_part = parts[-1] if len(parts) >= 3 else ""
                return f"{name} {temp_part}"
        return t
    # Crypto: KXBTC-26MAR2417-B71525 -> BTC range
    for prefix, name in [("KXBTC", "BTC"), ("KXETH", "ETH"), ("KXSOL", "SOL")]:
        if t.startswith(prefix):
            return f"{name} range"
    return ticker


def _get_research_status() -> str:
    """One-line research summary."""
    try:
        if not RESULTS_LOG.exists():
            return "idle | No data"
        lines = RESULTS_LOG.read_text().strip().split("\n")
        total = len(lines)
        improvements = sum(1 for l in lines if '"kept": true' in l)

        # Find last improvement time
        last_time = "unknown"
        for line in reversed(lines):
            try:
                entry = json.loads(line)
                if entry.get("kept"):
                    ts = entry.get("timestamp", "")
                    if ts:
                        dt = datetime.fromisoformat(ts)
                        ago = datetime.now() - dt.replace(tzinfo=None)
                        hours = int(ago.total_seconds() / 3600)
                        if hours < 1:
                            mins = int(ago.total_seconds() / 60)
                            last_time = f"{mins}m ago"
                        else:
                            last_time = f"{hours}h ago"
                    break
            except (json.JSONDecodeError, ValueError):
                continue

        # Check if autoresearch is running
        status = "running" if total > 0 else "idle"
        return f"{status} | Last: {last_time} | Improvements: {improvements}"
    except Exception:
        return "idle | No data"


# ── portfolio update (the main feature) ──────────────────────

def build_portfolio_update() -> str:
    """Build the 30-minute portfolio update message."""
    try:
        now = datetime.now()
        time_str = now.strftime("%-I:%M %p")

        # Get balance from Kalshi
        client = KalshiClient()
        bal = client.get_balance()
        pos = client.get_positions()

        cash = bal.get("balance", 0) / 100.0
        portfolio_val = bal.get("portfolio_value", 0) / 100.0
        equity = cash + portfolio_val

        # Today's trades from trade_history.csv
        today = now.strftime("%Y-%m-%d")
        today_trades = []
        if TRADE_HISTORY.exists():
            with open(TRADE_HISTORY) as f:
                for row in csv.DictReader(f):
                    if row.get("date") == today:
                        today_trades.append(row)

        # Settled today
        settled = [t for t in today_trades if t.get("settlement_result") in ("yes", "no")]
        wins = [t for t in settled if float(t.get("pnl", 0)) > 0]
        losses = [t for t in settled if float(t.get("pnl", 0)) < 0]
        today_pnl = sum(float(t.get("pnl", 0)) for t in settled)

        # Open positions today
        open_trades = [t for t in today_trades if t.get("settlement_result") == "open"]

        # Build message
        msg = f"<b>IPPO</b> -- {time_str}\n\n"
        msg += (
            f"Cash: ${cash:.2f} | Positions: ${portfolio_val:.2f} | "
            f"Equity: <b>${equity:.2f}</b>\n\n"
        )

        # Today's P&L section
        if settled:
            sign = "+" if today_pnl >= 0 else ""
            msg += (
                f"<b>Today:</b> {sign}${today_pnl:.2f} "
                f"({len(wins)}W/{len(losses)}L)\n"
            )
            for t in settled:
                pnl = float(t.get("pnl", 0))
                ticker = t.get("ticker", "?")
                side = t.get("side", "?").upper()
                contracts = t.get("contracts", "1")
                price = float(t.get("entry_price", 0))
                strategy = t.get("strategy", "")
                tag = _classify_strategy(ticker, strategy, side)
                reason = _trade_reason(ticker, strategy, side, price)
                icon = "+" if pnl >= 0 else ""
                win_icon = "WIN" if pnl >= 0 else "LOSS"
                short = _shorten_ticker(ticker)
                msg += (
                    f"  {win_icon} {icon}${pnl:.2f} {short} {side} "
                    f"{contracts}x@{int(price)}c [{tag}]\n"
                    f"     {reason}\n"
                )
            msg += "\n"
        else:
            msg += "<b>Today:</b> No settlements yet\n\n"

        # Open positions (top 5 by exposure)
        if open_trades:
            # Sort by dollar exposure (contracts * price)
            def exposure(t):
                c = int(t.get("contracts", 1))
                p = float(t.get("entry_price", 50))
                side = t.get("side", "yes").lower()
                if side == "no":
                    return c * (100 - p) / 100.0
                return c * p / 100.0

            sorted_open = sorted(open_trades, key=exposure, reverse=True)[:5]
            msg += f"<b>Open</b> (top {min(5, len(open_trades))} of {len(open_trades)}):\n"
            for t in sorted_open:
                ticker = t.get("ticker", "?")
                side = t.get("side", "?").upper()
                contracts = t.get("contracts", "1")
                price = float(t.get("entry_price", 50))
                strategy = t.get("strategy", "")
                tag = _classify_strategy(ticker, strategy, side)
                short = _shorten_ticker(ticker)
                exp = exposure(t)
                # Category icon
                if "NBA" in tag or "Sport" in tag:
                    icon = ""
                elif "Weather" in tag:
                    icon = ""
                elif "Crypto" in tag or "BTC" in tag:
                    icon = ""
                else:
                    icon = ""
                msg += (
                    f"  {icon} {short} {side} {contracts}x@{int(price)}c "
                    f"${exp:.2f} [{tag}]\n"
                )
            msg += "\n"

        # Research line
        research = _get_research_status()
        msg += f"Research: {research}"

        # Paused indicator
        if PAUSE_FILE.exists():
            msg += "\n\nTrading: PAUSED"

        return msg

    except Exception as e:
        log.error(f"Portfolio update error: {e}")
        return f"Portfolio update failed: {e}"


def scheduled_update():
    """Send portfolio update every 30 minutes."""
    try:
        msg = build_portfolio_update()
        send(msg)
        log.info("Sent scheduled portfolio update")
    except Exception as e:
        log.error(f"Scheduled update error: {e}")
    # Schedule next
    t = threading.Timer(UPDATE_INTERVAL, scheduled_update)
    t.daemon = True
    t.start()


# ── commands ─────────────────────────────────────────────────

def cmd_status():
    return build_portfolio_update()


def cmd_pnl():
    """All-time P&L breakdown."""
    try:
        if not TRADE_HISTORY.exists():
            return "No trade history yet."
        with open(TRADE_HISTORY) as f:
            records = list(csv.DictReader(f))
        settled = [r for r in records if r.get("settlement_result") in ("yes", "no")]
        if not settled:
            return "No settled trades yet."

        total_pnl = sum(float(r.get("pnl", 0)) for r in settled)
        wins = [r for r in settled if float(r.get("pnl", 0)) > 0]
        losses = [r for r in settled if float(r.get("pnl", 0)) < 0]
        win_rate = len(wins) / len(settled) * 100 if settled else 0

        # By strategy
        strats = {}
        for r in settled:
            tag = _classify_strategy(
                r.get("ticker", ""), r.get("strategy", ""), r.get("side", "")
            )
            if tag not in strats:
                strats[tag] = {"pnl": 0, "trades": 0, "wins": 0}
            strats[tag]["pnl"] += float(r.get("pnl", 0))
            strats[tag]["trades"] += 1
            if float(r.get("pnl", 0)) > 0:
                strats[tag]["wins"] += 1

        sign = "+" if total_pnl >= 0 else ""
        msg = (
            f"<b>P&amp;L REPORT</b>\n\n"
            f"Total: <b>{sign}${total_pnl:.2f}</b>\n"
            f"Win rate: {win_rate:.0f}% ({len(wins)}W/{len(losses)}L)\n"
            f"Settled: {len(settled)} trades\n\n"
        )
        for tag, d in sorted(strats.items(), key=lambda x: x[1]["pnl"], reverse=True):
            p = d["pnl"]
            s = "+" if p >= 0 else ""
            wr = d["wins"] / d["trades"] * 100 if d["trades"] > 0 else 0
            msg += f"  {tag:15s} {s}${p:.2f}  ({d['trades']}t, {wr:.0f}%)\n"
        return msg
    except Exception as e:
        return f"Error: {e}"


def cmd_pause():
    PAUSE_FILE.touch()
    return "Trading PAUSED. /resume to restart."


def cmd_resume():
    try:
        PAUSE_FILE.unlink()
    except FileNotFoundError:
        pass
    return "Trading RESUMED."


def cmd_help():
    return (
        "<b>IPPO COMMANDS</b>\n\n"
        "/status   Portfolio snapshot\n"
        "/pnl      All-time P&amp;L by strategy\n"
        "/pause    Stop trading\n"
        "/resume   Start trading again\n"
        "/help     This message\n\n"
        "Auto-updates every 30 min."
    )


# ── routing ──────────────────────────────────────────────────

COMMANDS = {
    "/status": cmd_status,
    "/pnl": cmd_pnl,
    "/pause": cmd_pause,
    "/resume": cmd_resume,
    "/help": cmd_help,
    "/start": cmd_help,
}


def handle_message(text: str):
    cmd = text.strip().lower().split("@")[0]
    if cmd in COMMANDS:
        return COMMANDS[cmd]()
    return None


# ── main loop ────────────────────────────────────────────────

def poll():
    offset = 0
    log.info("Ippo Telegram bot started")

    # Start scheduled updates
    t = threading.Timer(5, scheduled_update)  # first one in 5 seconds
    t.daemon = True
    t.start()

    while True:
        try:
            resp = http_requests.get(
                f"{API}/getUpdates",
                params={"offset": offset, "timeout": 30},
                timeout=35,
            )
            updates = resp.json().get("result", [])

            for update in updates:
                offset = update["update_id"] + 1
                msg = update.get("message", {})
                text = msg.get("text", "").strip()
                chat_id = str(msg.get("chat", {}).get("id", ""))

                if chat_id != CHAT_ID or not text:
                    continue

                log.info(f"Message: {text}")
                response = handle_message(text)

                if response:
                    send(response)
                elif text.startswith("/"):
                    send("Unknown command. /help for options.")

        except http_requests.exceptions.Timeout:
            continue
        except Exception as e:
            log.error(f"Poll error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    if not TOKEN or not CHAT_ID:
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
        sys.exit(1)
    poll()
