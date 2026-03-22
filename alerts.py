"""
alerts.py -- Cross-platform notification system for the Kalshi trading bot.

Sends alerts via Telegram bot (if configured) and logs to file.
Research improvements are batched and explained in plain English.
"""

import json
import os
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

ALERT_LOG = Path(__file__).parent / "output" / "alerts.log"
RESULTS_LOG = Path(__file__).parent / "autoresearch" / "results.log"
STRATEGY_FILE = Path(__file__).parent / "autoresearch" / "candidate_strategy.py"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

_research_batch = []
_RESEARCH_BATCH_SIZE = 10

# What each parameter means and how changes affect trading
_PARAM_EXPLAIN = {
    "FORECAST_STDEV_0": {
        "name": "Same-day forecast confidence",
        "up": "Being more cautious about today's forecast accuracy. Will need bigger edges to trade.",
        "down": "Trusting today's forecast more. Will find more edges and trade more aggressively.",
    },
    "FORECAST_STDEV_1": {
        "name": "Tomorrow forecast confidence",
        "up": "Being more cautious about tomorrow's forecast. Wider uncertainty = fewer but safer trades.",
        "down": "Trusting tomorrow's forecast more. Tighter model = more trades on tomorrow's markets.",
    },
    "FORECAST_STDEV_2": {
        "name": "2-day forecast confidence",
        "up": "Less confident in 2-day forecasts. Will trade these markets less aggressively.",
        "down": "More confident in 2-day forecasts. Will find more edges on day-after-tomorrow markets.",
    },
    "FORECAST_STDEV_3": {
        "name": "3-day forecast confidence",
        "up": "Less confident in 3-day forecasts. More conservative on longer-dated markets.",
        "down": "More confident in 3-day forecasts. Will trade longer-dated markets more.",
    },
    "EDGE_THRESHOLD_CENTS": {
        "name": "Minimum edge to trade",
        "up": "Raising the bar — only trading when we have a bigger advantage. Fewer trades, higher quality.",
        "down": "Lowering the bar — taking smaller edges too. More trades, but each one has less advantage.",
    },
    "CONTRACTS_PER_TRADE": {
        "name": "Trade size",
        "up": "Betting more per trade. Higher risk, higher reward.",
        "down": "Betting less per trade. Safer, more conservative.",
    },
    "NWS_OFFICIAL_WEIGHT": {
        "name": "Weather source blend",
        "up": "Trusting the official NWS forecast more vs. the ensemble model.",
        "down": "Trusting the GFS ensemble model more vs. the official NWS forecast.",
    },
    "CITY_WEIGHT_NYC": {
        "name": "NYC trading weight",
        "up": "Trading NYC weather markets more aggressively.",
        "down": "Trading NYC weather markets less aggressively.",
    },
    "CITY_WEIGHT_CHI": {
        "name": "Chicago trading weight",
        "up": "Trading Chicago weather more aggressively.",
        "down": "Scaling back Chicago weather trades — forecasts may be less reliable there.",
    },
    "CITY_WEIGHT_MIA": {
        "name": "Miami trading weight",
        "up": "Trading Miami weather more aggressively.",
        "down": "Scaling back Miami weather trades.",
    },
    "CITY_WEIGHT_LA": {
        "name": "LA trading weight",
        "up": "Trading LA weather more aggressively.",
        "down": "Scaling back LA weather trades.",
    },
    "CITY_WEIGHT_DC": {
        "name": "DC trading weight",
        "up": "Trading DC weather more aggressively.",
        "down": "Scaling back DC weather trades.",
    },
    "CITY_WEIGHT_DEN": {
        "name": "Denver trading weight",
        "up": "Trading Denver weather more aggressively.",
        "down": "Scaling back Denver weather trades.",
    },
    "BUCKET_MULTIPLIER": {
        "name": "Bucket market preference",
        "up": "Favoring 'temperature falls in X-Y range' markets.",
        "down": "Reducing bets on range-bucket markets.",
    },
    "THRESHOLD_MULTIPLIER": {
        "name": "Threshold market preference",
        "up": "Favoring 'temperature above/below X' markets.",
        "down": "Reducing bets on above/below threshold markets.",
    },
    "HIGH_CONFIDENCE_EDGE": {
        "name": "High confidence threshold",
        "up": "Harder to reach 'high confidence' — more conservative sizing on big edges.",
        "down": "Easier to reach 'high confidence' — will size up more on good edges.",
    },
    "MEDIUM_CONFIDENCE_EDGE": {
        "name": "Medium confidence threshold",
        "up": "Harder to reach 'medium confidence' — skipping more borderline trades.",
        "down": "Taking more medium-confidence trades.",
    },
    "TIGHT_ENSEMBLE_THRESHOLD": {
        "name": "Ensemble agreement threshold",
        "up": "Requiring forecasts to agree more before betting big.",
        "down": "Betting bigger even when forecasts disagree slightly.",
    },
    "TIGHT_ENSEMBLE_MULTIPLIER": {
        "name": "Ensemble agreement bet boost",
        "up": "Betting even bigger when all forecast models agree.",
        "down": "Less aggressive even when models agree. More conservative overall.",
    },
    "COPY_DELAY_SECONDS": {
        "name": "Copy-trade reaction speed",
        "up": "Waiting longer after whale trades before copying. More cautious.",
        "down": "Copying whale trades faster. More aggressive.",
    },
    "COPY_MIN_CONFIDENCE": {
        "name": "Copy-trade confidence filter",
        "up": "Only copying highest-conviction whale trades.",
        "down": "Copying more whale trades, including lower-conviction ones.",
    },
}


def _get_total_experiments() -> int:
    """Count total experiments from results.log."""
    try:
        if RESULTS_LOG.exists():
            return sum(1 for _ in open(RESULTS_LOG))
    except Exception:
        pass
    return 0


def _get_total_improvements() -> int:
    """Count total improvements from results.log."""
    try:
        if RESULTS_LOG.exists():
            return sum(1 for line in open(RESULTS_LOG) if '"kept": true' in line)
    except Exception:
        pass
    return 0


def _get_strategy_summary() -> str:
    """Read current strategy params and summarize in plain English."""
    try:
        if not STRATEGY_FILE.exists():
            return ""
        text = STRATEGY_FILE.read_text()
        params = {}
        for line in text.split("\n"):
            if "=" in line and not line.strip().startswith("#") and not line.strip().startswith("def") and not line.strip().startswith("\""):
                parts = line.split("=", 1)
                name = parts[0].strip()
                try:
                    val = float(parts[1].strip().split("#")[0].strip())
                    params[name] = val
                except (ValueError, IndexError):
                    pass

        edge = params.get("EDGE_THRESHOLD_CENTS", 3)
        nws_w = params.get("NWS_OFFICIAL_WEIGHT", 0.4)
        ens_w = round(1 - nws_w, 1)

        cities = []
        for city, key in [("NYC", "CITY_WEIGHT_NYC"), ("CHI", "CITY_WEIGHT_CHI"),
                          ("MIA", "CITY_WEIGHT_MIA"), ("LA", "CITY_WEIGHT_LA"),
                          ("DC", "CITY_WEIGHT_DC"), ("DEN", "CITY_WEIGHT_DEN")]:
            w = params.get(key, 1.0)
            if w > 1.1:
                cities.append(f"{city} (aggressive)")
            elif w < 0.9:
                cities.append(f"{city} (cautious)")

        summary = f"Min edge: {edge:.0f}c | Blend: {nws_w:.0%} NWS + {ens_w:.0%} ensemble"
        if cities:
            summary += f"\nCity adjustments: {', '.join(cities)}"
        return summary
    except Exception:
        return ""


def _explain_change(param: str, old_val, new_val) -> str:
    """Generate plain English explanation of a parameter change."""
    info = _PARAM_EXPLAIN.get(param)
    if not info:
        return f"{param} changed from {old_val} to {new_val}"

    try:
        direction = "up" if float(new_val) > float(old_val) else "down"
    except (ValueError, TypeError):
        direction = "up"

    return info[direction]


def _send_telegram(text: str):
    """Send a Telegram message, splitting at 4000 chars if needed."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        import requests
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

        MAX_LEN = 4000
        if len(text) <= MAX_LEN:
            requests.post(url, json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
            }, timeout=10)
            return

        # Split long messages on newline boundaries
        chunks = []
        remaining = text
        while remaining:
            if len(remaining) <= MAX_LEN:
                chunks.append(remaining)
                break
            # Find last newline before the limit
            split_at = remaining.rfind("\n", 0, MAX_LEN)
            if split_at == -1:
                split_at = MAX_LEN
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:].lstrip("\n")

        for chunk in chunks:
            if not chunk.strip():
                continue
            requests.post(url, json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": chunk,
                "parse_mode": "HTML",
            }, timeout=10)
    except Exception:
        pass


def _send_macos(title: str, message: str, sound: bool = False):
    if platform.system() != "Darwin":
        return
    try:
        script = f'display notification "{message}" with title "{title}"'
        if sound:
            script += ' sound name "Glass"'
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)
    except Exception:
        pass


def _log_to_file(title: str, message: str):
    log_entry = {"ts": datetime.now().isoformat(), "title": title, "message": message}
    try:
        with open(ALERT_LOG, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass


def notify(title: str, message: str, sound: bool = False):
    _log_to_file(title, message)
    _send_telegram(f"<b>{title}</b>\n{message}")
    _send_macos(title, message, sound)


def alert_trade_settled(ticker: str, side: str, pnl: float, won: bool):
    sign = "+" if pnl >= 0 else ""
    _log_to_file("Trade Settled", f"{ticker} {side} {sign}${pnl:.2f}")
    _send_telegram(
        f"{'<b>WIN</b>' if won else '<b>LOSS</b>'}  {sign}${pnl:.2f}\n"
        f"{ticker}\n"
        f"Side: {side.upper()}"
    )
    _send_macos(f"Trade {'WON' if won else 'LOST'}", f"{sign}${pnl:.2f}", sound=won)


def alert_drawdown(current_dd_pct: float, balance: float):
    _log_to_file("Drawdown Warning", f"{current_dd_pct:.1f}% — ${balance:.2f}")
    _send_telegram(
        f"<b>DRAWDOWN WARNING</b>\n\n"
        f"Account is down {current_dd_pct:.1f}% today\n"
        f"Balance: ${balance:.2f}\n\n"
        f"Trading will automatically pause at 8% daily loss."
    )


def alert_bot_error(component: str, error: str):
    _log_to_file(f"Bot Error: {component}", error[:200])
    _send_telegram(
        f"<b>ERROR</b> in {component}\n\n"
        f"{error[:200]}"
    )


def alert_research_improvement(param: str, old_val, new_val, score: float):
    global _research_batch

    _research_batch.append({
        "param": param,
        "old": old_val,
        "new": new_val,
        "score": score,
        "explanation": _explain_change(param, old_val, new_val),
    })

    _log_to_file("Research Improvement", f"{param}: {old_val} -> {new_val} (score: {score:.2f})")

    if len(_research_batch) >= _RESEARCH_BATCH_SIZE:
        _flush_research_batch()


def _flush_research_batch():
    global _research_batch
    if not _research_batch:
        return

    total_experiments = _get_total_experiments()
    total_improvements = _get_total_improvements()
    best = max(_research_batch, key=lambda x: x["score"])

    info = _PARAM_EXPLAIN.get(best["param"], {})
    best_name = info.get("name", best["param"])

    msg = (
        f"<b>STRATEGY UPDATE</b>\n\n"
        f"Ran {total_experiments:,} experiments so far\n"
        f"Found {total_improvements} total improvements ({len(_research_batch)} new)\n\n"
        f"<b>Best finding:</b>\n"
        f"{best_name}: {best['old']} -> {best['new']}\n"
        f"{best['explanation']}\n\n"
    )

    if len(_research_batch) > 1:
        msg += "<b>Other improvements:</b>\n"
        for item in _research_batch:
            if item is not best:
                name = _PARAM_EXPLAIN.get(item["param"], {}).get("name", item["param"])
                msg += f"  {name}: {item['old']} -> {item['new']}\n"
        msg += "\n"

    strategy = _get_strategy_summary()
    if strategy:
        msg += f"<b>Current strategy:</b>\n{strategy}"

    _send_telegram(msg)
    _research_batch = []


def alert_big_edge(ticker: str, edge_cents: float, city: str = ""):
    where = f" in {city}" if city else ""
    _log_to_file("Big Edge", f"{ticker}: +{edge_cents:.0f}c")
    _send_telegram(
        f"<b>BIG EDGE FOUND</b>{where}\n\n"
        f"Our model sees a +{edge_cents:.0f} cent advantage\n"
        f"on {ticker}\n\n"
        f"That means the market is mispriced by {edge_cents:.0f}% "
        f"compared to our forecast."
    )


def alert_daily_summary(trades: int, pnl: float, win_rate: float):
    sign = "+" if pnl >= 0 else ""
    _flush_research_batch()
    _log_to_file("Daily Summary", f"{trades} trades, {sign}${pnl:.2f}")

    total_experiments = _get_total_experiments()
    total_improvements = _get_total_improvements()

    _send_telegram(
        f"<b>DAILY SUMMARY</b>\n\n"
        f"Trades placed: {trades}\n"
        f"Deployed: {sign}${pnl:.2f}\n\n"
        f"Research: {total_experiments:,} experiments, "
        f"{total_improvements} improvements found"
    )


def send_trade_alert(trade_details: dict):
    """Send a single-trade execution notification.

    Expected keys in trade_details:
        side:       "yes" or "no"
        ticker:     market ticker string
        price:      entry price in cents (e.g. 25)
        contracts:  number of contracts
        risk:       max loss in dollars
        target:     max gain in dollars
        strategy:   strategy name (e.g. "weather", "crypto")
        edge:       edge in cents (optional)
    """
    side = trade_details.get("side", "?").upper()
    ticker = trade_details.get("ticker", "???")
    price = trade_details.get("price", 0)
    contracts = trade_details.get("contracts", 1)
    risk = trade_details.get("risk", 0)
    target = trade_details.get("target", 0)
    strategy = trade_details.get("strategy", "")
    edge = trade_details.get("edge", None)

    strat_line = strategy.title() if strategy else "Manual"
    if edge is not None:
        strat_line += f" (edge: {edge:.0f}c)"

    msg = (
        f"<pre>"
        f"TRADE EXECUTED\n"
        f"  BUY {side} {ticker}\n"
        f"  @ {price:.0f}c x {contracts}\n"
        f"  Risk: ${risk:.2f} | Target: +${target:.2f}\n"
        f"  Strategy: {strat_line}"
        f"</pre>"
    )

    _log_to_file("Trade Executed", f"{side} {ticker} @ {price}c x{contracts}")
    _send_telegram(msg)


def send_settlement_alert(trade_details: dict):
    """Send a settlement notification for a resolved position.

    Expected keys in trade_details:
        ticker:     market ticker string
        side:       "yes" or "no"
        won:        bool -- True if position won
        pnl:        realized P&L in dollars (signed)
        entry:      entry price in cents
        nav:        current NAV after settlement (optional)
        day_pnl:    day's running P&L (optional)
    """
    ticker = trade_details.get("ticker", "???")
    side = trade_details.get("side", "?").upper()
    won = trade_details.get("won", False)
    pnl = trade_details.get("pnl", 0.0)
    entry = trade_details.get("entry", 0)
    nav = trade_details.get("nav", None)
    day_pnl = trade_details.get("day_pnl", None)

    if won:
        header = "SETTLEMENT \u2014 WIN"
        pnl_pct = ((1.0 / (entry / 100.0)) - 1) * 100 if entry > 0 else 0
        detail = f"  {ticker} {side} settled @ $1.00"
        pnl_line = f"  P&L: +${pnl:.2f} (+{pnl_pct:.0f}%)"
    else:
        header = "SETTLEMENT \u2014 LOSS"
        detail = f"  {ticker} {side} expired worthless"
        pnl_line = f"  P&L: -${abs(pnl):.2f}"

    book_line = ""
    if nav is not None:
        sign = "+" if (day_pnl or pnl) >= 0 else ""
        day_val = day_pnl if day_pnl is not None else pnl
        book_line = f"\n  Book: ${nav:.2f} NAV ({sign}${day_val:.2f} today)"

    msg = (
        f"<pre>"
        f"{header}\n"
        f"{detail}\n"
        f"{pnl_line}\n"
        f"{book_line}"
        f"</pre>"
    )

    _log_to_file("Settlement", f"{'WIN' if won else 'LOSS'} {ticker} {side} {pnl:+.2f}")
    _send_telegram(msg)
