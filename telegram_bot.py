"""
telegram_bot.py -- Interactive Telegram bot for Ippo trading bot.

Query commands:
    /status    - Account balance, positions, equity
    /trades    - Recent trade decisions
    /pnl       - P&L breakdown by strategy
    /research  - AutoResearch progress
    /health    - Service health check
    /strategy  - Current strategy explained in plain English
    /help      - Show all commands

Control commands:
    /pause     - Pause all trading
    /resume    - Resume trading
    /set edge 5       - Change min edge threshold (cents)
    /set city miami 1.5  - Change city weight
    /set nws 0.4      - Change NWS vs ensemble blend
    /run trade        - Trigger auto-trade now
    /run research     - Trigger autoresearch now
    /run settle       - Trigger settlement check now
"""

import json
import os
import re
import subprocess
import sys
import time
import csv
import logging
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
STRATEGY_FILE = AUTORESEARCH / "candidate_strategy.py"
PAUSE_FILE = OUTPUT / ".trading_paused"


def send(text: str):
    try:
        http_requests.post(f"{API}/sendMessage", json={
            "chat_id": CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
        }, timeout=10)
    except Exception as e:
        log.error(f"Send failed: {e}")


# ─── QUERY COMMANDS ──────────────────────────────────────────

def cmd_status():
    try:
        client = KalshiClient()
        bal = client.get_balance()
        pos = client.get_positions()

        cash = bal.get("balance", 0) / 100.0
        portfolio = bal.get("portfolio_value", 0) / 100.0
        equity = cash + portfolio
        pnl_pct = ((equity - 100) / 100) * 100

        positions = [
            p for p in pos.get("market_positions", [])
            if float(p.get("position_fp", 0)) != 0
        ]

        weather = [p for p in positions if any(
            p.get("ticker", "").startswith(s) for s in
            ["KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHLA", "KXHIGHDC", "KXHIGHDEN"]
        )]
        crypto = [p for p in positions if any(
            p.get("ticker", "").startswith(s) for s in ["KXBTC", "KXETH", "KXSOL"]
        )]

        paused = PAUSE_FILE.exists()

        msg = (
            f"<b>ACCOUNT STATUS</b>\n\n"
            f"Cash:       ${cash:.2f}\n"
            f"Positions:  ${portfolio:.2f}  ({len(positions)} open)\n"
            f"Equity:     <b>${equity:.2f}</b>  ({pnl_pct:+.1f}%)\n"
        )

        if paused:
            msg += "\nTrading: PAUSED\n"

        if weather:
            msg += f"\nWeather ({len(weather)}):\n"
            for p in weather:
                t = p.get("ticker", "?")
                qty = int(float(p.get("position_fp", 0)))
                side = "YES" if qty > 0 else "NO"
                # Make ticker readable
                short = t.replace("KXHIGH", "").split("-")
                city = short[0] if short else "?"
                rest = "-".join(short[1:]) if len(short) > 1 else ""
                msg += f"  {side} x{abs(qty)}  {city} {rest}\n"

        if crypto:
            msg += f"\nCrypto ({len(crypto)}):\n"
            for p in crypto:
                t = p.get("ticker", "?")
                qty = int(float(p.get("position_fp", 0)))
                side = "YES" if qty > 0 else "NO"
                short = t.replace("KX", "")
                msg += f"  {side} x{abs(qty)}  {short}\n"

        return msg
    except Exception as e:
        return f"Error: {e}"


def cmd_trades():
    try:
        json_files = sorted(OUTPUT.glob("auto_trade_*.json"), reverse=True)
        if not json_files:
            return "No trade logs yet."

        with open(json_files[0]) as f:
            data = json.load(f)

        decisions = data.get("decisions", [])
        placed = [d for d in decisions if d.get("placed")]
        skipped = [d for d in decisions if d.get("contracts", 0) > 0 and not d.get("placed")]
        date = data.get("date", "?")

        msg = f"<b>TRADES</b>  ({date})\n\n"

        if placed:
            msg += f"<b>{len(placed)} trades placed:</b>\n\n"
            for d in placed[:8]:
                action = d.get("action", "?").replace("buy_", "BUY ").replace("sell_", "SELL ").upper()
                ticker = d.get("ticker", "?")
                edge = d.get("edge_cents", 0)
                cost = d.get("max_loss_dollars", 0)
                contracts = d.get("contracts", 0)
                msg += (
                    f"  {action} x{contracts}  ${cost:.2f}\n"
                    f"  {ticker}\n"
                    f"  Edge: +{edge:.0f}c\n\n"
                )
        else:
            msg += "No trades placed this session.\n\n"

        if skipped:
            msg += f"{len(skipped)} dry-run (would have traded):\n"
            for d in skipped[:3]:
                msg += f"  {d.get('ticker', '?')}  +{d.get('edge_cents', 0):.0f}c\n"

        return msg
    except Exception as e:
        return f"Error: {e}"


def cmd_pnl():
    try:
        path = OUTPUT / "trade_history.csv"
        if not path.exists():
            return "No trade history yet."

        with open(path) as f:
            records = list(csv.DictReader(f))

        settled = [r for r in records if r.get("settlement_result") in ("yes", "no")]
        if not settled:
            return "No settled trades yet. Positions are still open."

        total_pnl = sum(float(r.get("pnl", 0)) for r in settled)
        wins = [r for r in settled if float(r.get("pnl", 0)) > 0]
        losses = [r for r in settled if float(r.get("pnl", 0)) < 0]
        win_rate = len(wins) / len(settled) * 100 if settled else 0

        strategies = {}
        for r in settled:
            s = r.get("strategy", "other")
            if s not in strategies:
                strategies[s] = {"pnl": 0, "trades": 0, "wins": 0}
            strategies[s]["pnl"] += float(r.get("pnl", 0))
            strategies[s]["trades"] += 1
            if float(r.get("pnl", 0)) > 0:
                strategies[s]["wins"] += 1

        sign = "+" if total_pnl >= 0 else ""

        msg = (
            f"<b>P&L REPORT</b>\n\n"
            f"Total P&L:  <b>{sign}${total_pnl:.2f}</b>\n"
            f"Win Rate:   {win_rate:.0f}%  ({len(wins)}W / {len(losses)}L)\n"
            f"Trades:     {len(settled)} settled\n\n"
        )

        for s, data in sorted(strategies.items(), key=lambda x: x[1]["pnl"], reverse=True):
            pnl = data["pnl"]
            trades = data["trades"]
            wr = data["wins"] / trades * 100 if trades > 0 else 0
            sign = "+" if pnl >= 0 else ""
            msg += f"  {s:10s}  {sign}${pnl:.2f}  ({trades} trades, {wr:.0f}%)\n"

        return msg
    except Exception as e:
        return f"Error: {e}"


def cmd_research():
    try:
        results_log = AUTORESEARCH / "results.log"
        if not results_log.exists():
            return "No research results yet."

        lines = results_log.read_text().strip().split("\n")
        total = len(lines)
        improvements = sum(1 for l in lines if '"kept": true' in l)
        hit_rate = improvements / total * 100 if total > 0 else 0

        recent = []
        for line in reversed(lines):
            try:
                entry = json.loads(line)
                if entry.get("kept"):
                    recent.append(entry)
                    if len(recent) >= 5:
                        break
            except json.JSONDecodeError:
                continue

        msg = (
            f"<b>RESEARCH PROGRESS</b>\n\n"
            f"Experiments:   {total:,}\n"
            f"Improvements:  {improvements}\n"
            f"Hit rate:      {hit_rate:.1f}%\n\n"
        )

        if recent:
            best = recent[0]
            sortino = best.get("metrics", {}).get("sortino", 0)
            roi = best.get("metrics", {}).get("roi_pct", 0)
            msg += (
                f"<b>Latest performance:</b>\n"
                f"  Sortino ratio: {sortino:.1f}\n"
                f"  Backtest ROI:  {roi:.0f}%\n\n"
                f"<b>Recent improvements:</b>\n"
            )
            from alerts import _PARAM_EXPLAIN
            for entry in recent:
                param = entry.get("parameter", "?")
                name = _PARAM_EXPLAIN.get(param, {}).get("name", param)
                old = entry.get("old_value", "?")
                new = entry.get("new_value", "?")
                msg += f"  {name}: {old} -> {new}\n"

        return msg
    except Exception as e:
        return f"Error: {e}"


def cmd_strategy():
    try:
        if not STRATEGY_FILE.exists():
            return "Strategy file not found."

        text = STRATEGY_FILE.read_text()
        params = {}
        for line in text.split("\n"):
            if "=" in line and not line.strip().startswith(("#", "def", '"', "'")):
                parts = line.split("=", 1)
                name = parts[0].strip()
                try:
                    val = parts[1].strip().split("#")[0].strip()
                    params[name] = val
                except (IndexError):
                    pass

        edge = params.get("EDGE_THRESHOLD_CENTS", "3")
        nws = float(params.get("NWS_OFFICIAL_WEIGHT", "0.4"))
        ens = round(1 - nws, 1)
        contracts = params.get("CONTRACTS_PER_TRADE", "10")

        msg = (
            f"<b>CURRENT STRATEGY</b>\n\n"
            f"<b>How we trade:</b>\n"
            f"We compare weather forecasts to Kalshi market prices.\n"
            f"When our forecast disagrees with the market by more than "
            f"{edge} cents, we place a trade betting our forecast is right.\n\n"
            f"<b>Forecast blend:</b>\n"
            f"  {nws:.0%} official NWS forecast\n"
            f"  {ens:.0%} GFS ensemble model (30 simulations)\n\n"
            f"<b>Trade sizing:</b>\n"
            f"  {contracts} contracts per trade\n"
            f"  Max $2 per trade, 8% daily loss cap\n\n"
            f"<b>City weights:</b>\n"
        )

        for city, key in [("NYC", "CITY_WEIGHT_NYC"), ("Chicago", "CITY_WEIGHT_CHI"),
                          ("Miami", "CITY_WEIGHT_MIA"), ("LA", "CITY_WEIGHT_LA"),
                          ("DC", "CITY_WEIGHT_DC"), ("Denver", "CITY_WEIGHT_DEN")]:
            w = float(params.get(key, "1.0"))
            if w > 1.1:
                msg += f"  {city}: {w}x (aggressive)\n"
            elif w < 0.9:
                msg += f"  {city}: {w}x (cautious)\n"
            else:
                msg += f"  {city}: {w}x\n"

        paused = PAUSE_FILE.exists()
        msg += f"\nTrading: {'PAUSED' if paused else 'ACTIVE'}"

        return msg
    except Exception as e:
        return f"Error: {e}"


def cmd_health():
    try:
        resp = http_requests.get("http://localhost:8787/health", timeout=5)
        d = resp.json()

        msg = f"<b>SYSTEM HEALTH</b>\n\n"

        for name, info in d.get("services", {}).items():
            running = info.get("running", False)
            label = name.replace("_", " ").replace("arb runner", "Arb Scanner").replace("auto trade", "Auto Trade").replace("autoresearch", "Research").replace("settlement", "Settlement")
            msg += f"  {label:15s}  {'ON' if running else 'OFF'}\n"

        msg += (
            f"\nBalance:  ${d.get('balance', 0):.2f}\n"
            f"Errors:   {d.get('errors_today', 0)}\n"
        )

        ar = d.get("autoresearch", {})
        msg += f"Research: {ar.get('iterations', 0):,} experiments, {ar.get('improvements', 0)} improvements\n"

        return msg
    except Exception as e:
        return f"Health check failed: {e}"


# ─── CONTROL COMMANDS ────────────────────────────────────────

def cmd_pause():
    PAUSE_FILE.touch()
    return "Trading PAUSED. Auto-trade will skip placing orders until you /resume."


def cmd_resume():
    try:
        PAUSE_FILE.unlink()
    except FileNotFoundError:
        pass
    return "Trading RESUMED. Next auto-trade session will place orders."


def cmd_set(args: str):
    """Handle /set commands to modify strategy parameters."""
    parts = args.strip().split()
    if len(parts) < 2:
        return (
            "<b>Usage:</b>\n"
            "/set edge 5        - Min edge (cents)\n"
            "/set nws 0.4       - NWS weight (0-1)\n"
            "/set city miami 1.5 - City weight\n"
        )

    target = parts[0].lower()

    if target == "edge":
        try:
            val = float(parts[1])
            if val < 1 or val > 20:
                return "Edge must be 1-20 cents."
            return _update_strategy_param("EDGE_THRESHOLD_CENTS", val)
        except ValueError:
            return "Invalid number. Example: /set edge 5"

    elif target == "nws":
        try:
            val = float(parts[1])
            if val < 0 or val > 1:
                return "NWS weight must be 0-1."
            return _update_strategy_param("NWS_OFFICIAL_WEIGHT", val)
        except ValueError:
            return "Invalid number. Example: /set nws 0.4"

    elif target == "city":
        if len(parts) < 3:
            return "Usage: /set city miami 1.5"
        city_map = {
            "nyc": "CITY_WEIGHT_NYC", "newyork": "CITY_WEIGHT_NYC",
            "chicago": "CITY_WEIGHT_CHI", "chi": "CITY_WEIGHT_CHI",
            "miami": "CITY_WEIGHT_MIA", "mia": "CITY_WEIGHT_MIA",
            "la": "CITY_WEIGHT_LA", "losangeles": "CITY_WEIGHT_LA",
            "dc": "CITY_WEIGHT_DC", "washington": "CITY_WEIGHT_DC",
            "denver": "CITY_WEIGHT_DEN", "den": "CITY_WEIGHT_DEN",
        }
        city_key = city_map.get(parts[1].lower())
        if not city_key:
            return f"Unknown city. Options: {', '.join(set(city_map.values()))}"
        try:
            val = float(parts[2])
            if val < 0.1 or val > 3:
                return "City weight must be 0.1-3.0"
            return _update_strategy_param(city_key, val)
        except ValueError:
            return "Invalid number. Example: /set city miami 1.5"

    return "Unknown setting. Type /set to see options."


def _update_strategy_param(param: str, value: float) -> str:
    """Update a parameter in candidate_strategy.py."""
    try:
        text = STRATEGY_FILE.read_text()
        lines = text.split("\n")
        updated = False

        for i, line in enumerate(lines):
            if line.startswith(f"{param} ") or line.startswith(f"{param}="):
                # Preserve any inline comment
                comment = ""
                if "#" in line:
                    comment = "  #" + line.split("#", 1)[1]
                lines[i] = f"{param} = {value}{comment}"
                updated = True
                break

        if not updated:
            return f"Parameter {param} not found in strategy file."

        STRATEGY_FILE.write_text("\n".join(lines))

        from alerts import _PARAM_EXPLAIN
        name = _PARAM_EXPLAIN.get(param, {}).get("name", param)
        return f"Updated <b>{name}</b> to {value}\n\nThis will take effect on the next trading session."

    except Exception as e:
        return f"Error updating: {e}"


def cmd_run(args: str):
    """Trigger a service run manually."""
    target = args.strip().lower()

    if target in ("trade", "auto-trade", "autotrade"):
        subprocess.Popen(
            ["systemctl", "start", "ippo-auto-trade"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return "Auto-trade triggered. Results will appear in a few minutes."

    elif target in ("research", "autoresearch"):
        subprocess.Popen(
            ["systemctl", "restart", "ippo-autoresearch"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return "AutoResearch restarted with fresh cycle."

    elif target in ("settle", "settlement", "settlements"):
        subprocess.Popen(
            ["systemctl", "start", "ippo-settlement"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return "Settlement check triggered. P&L will update shortly."

    return (
        "<b>Usage:</b>\n"
        "/run trade     - Run auto-trade now\n"
        "/run research  - Restart autoresearch\n"
        "/run settle    - Check settlements now\n"
    )


def cmd_help():
    return (
        "<b>IPPO COMMANDS</b>\n\n"
        "<b>Info:</b>\n"
        "/status    Balance & positions\n"
        "/trades    Recent trades\n"
        "/pnl       Profit & loss report\n"
        "/strategy  How the bot trades (plain English)\n"
        "/research  AutoResearch progress\n"
        "/health    System health\n\n"
        "<b>Control:</b>\n"
        "/pause     Stop trading\n"
        "/resume    Start trading again\n"
        "/set edge 5       Change min edge\n"
        "/set city miami 1.5  Change city weight\n"
        "/set nws 0.4      Change forecast blend\n"
        "/run trade    Trigger trade session\n"
        "/run settle   Check settlements\n"
    )


# ─── ROUTING ─────────────────────────────────────────────────

SIMPLE_COMMANDS = {
    "/status": cmd_status,
    "/trades": cmd_trades,
    "/pnl": cmd_pnl,
    "/research": cmd_research,
    "/strategy": cmd_strategy,
    "/health": cmd_health,
    "/pause": cmd_pause,
    "/resume": cmd_resume,
    "/help": cmd_help,
    "/start": cmd_help,
}


def handle_message(text: str):
    """Route a message to the right handler."""
    text = text.strip()
    cmd = text.lower().split("@")[0]  # strip @botname

    # Simple commands
    if cmd in SIMPLE_COMMANDS:
        return SIMPLE_COMMANDS[cmd]()

    # Commands with arguments
    if cmd.startswith("/set"):
        args = text[4:].strip()
        return cmd_set(args)

    if cmd.startswith("/run"):
        args = text[4:].strip()
        return cmd_run(args)

    return None


def poll():
    offset = 0
    log.info("Ippo Telegram bot started")

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
                    send("Unknown command. Type /help to see options.")

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
