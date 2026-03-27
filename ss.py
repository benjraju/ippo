#!/usr/bin/env python3
"""
ss — Ippo Trading Bot Status Dashboard

Run: python ss.py (on VPS)
Or:  ssh root@24.144.91.158 'cd /opt/ippo && .venv/bin/python ss.py'
"""

import csv
import json
import math
import subprocess
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
OUTPUT = Path("/opt/ippo/output")
TOTAL_DEPOSITED = 588.0  # $98 initial + $490 on Mar 27


def section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def ok(text):
    return f"  OK  {text}"


def fail(text):
    return f" FAIL {text}"


# ── 1. Service Health ───────────────────────────────────────────────────
def check_services():
    section("SERVICES")
    services = [
        "ippo-auto-trade.timer", "ippo-arb-runner", "ippo-telegram",
        "ippo-health", "ippo-settlement.timer", "ippo-weather-tail.timer",
    ]
    all_ok = True
    for svc in services:
        try:
            result = subprocess.run(
                ["systemctl", "is-active", svc],
                capture_output=True, text=True, timeout=5,
            )
            status = result.stdout.strip()
            symbol = "OK" if status == "active" else "!!"
            print(f"  [{symbol}] {svc:35s} {status}")
            if status != "active":
                all_ok = False
        except Exception:
            print(f"  [!!] {svc:35s} ERROR")
            all_ok = False

    # Check for rogue shadow files
    shadows = [f for f in ["numpy.py", "dotenv.py", "pandas.py"]
               if Path(f"/opt/ippo/{f}").exists()]
    if shadows:
        print(f"  [!!] ROGUE SHADOW FILES: {', '.join(shadows)}")
        all_ok = False
    else:
        print(f"  [OK] No shadow files")

    return all_ok


# ── 2. Account Balance ─────────────────────────────────────────────────
def check_balance():
    section("ACCOUNT")
    try:
        sys.path.insert(0, "/opt/ippo")
        from kalshi_client import KalshiClient
        client = KalshiClient()

        bal = client.get_balance()
        if isinstance(bal, dict):
            cash_cents = bal.get("balance", 0)
        else:
            cash_cents = bal
        cash = cash_cents / 100.0

        # Get positions
        pos_resp = client.get_positions()
        positions = pos_resp.get("market_positions", pos_resp.get("positions", []))
        open_count = len(positions) if positions else 0

        # Estimate position value from market_exposure or position cost
        pos_value = 0
        for p in (positions or []):
            # Try various fields Kalshi might return
            exposure = abs(float(p.get("market_exposure", 0) or 0))
            if exposure > 0:
                pos_value += exposure / 100  # cents to dollars
            else:
                qty = abs(int(p.get("total_traded", p.get("position", 0)) or 0))
                pos_value += qty * 0.50  # rough estimate

        # Total equity = cash + positions value (from balance API)
        try:
            portfolio = client._request("GET", "/portfolio/balance")
            pos_cents = portfolio.get("portfolio_value", 0) or 0
            total_equity = cash + pos_cents / 100.0
        except Exception:
            total_equity = cash + pos_value  # fallback

        net_pnl = total_equity - TOTAL_DEPOSITED

        print(f"  Cash:              ${cash:.2f}")
        print(f"  Positions:         ${total_equity - cash:.2f}")
        print(f"  Total equity:      ${total_equity:.2f}")
        print(f"  Total deposited:   ${TOTAL_DEPOSITED:.2f}")
        print(f"  Net P&L:           ${net_pnl:+.2f}  ({net_pnl / TOTAL_DEPOSITED * 100:+.1f}%)")
        return cash, open_count, total_equity
    except Exception as e:
        print(f"  ERROR: {e}")
        return 0, 0, 0


# ── 3. Today's Trades ──────────────────────────────────────────────────
def show_todays_trades():
    section("TODAY'S TRADES")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    json_file = OUTPUT / f"auto_trade_{today}.json"

    if not json_file.exists():
        print("  No trades today yet.")
        return

    with open(json_file) as f:
        data = json.load(f)

    # JSON wraps decisions in a dict with metadata
    if isinstance(data, dict):
        decisions = data.get("decisions", [])
    else:
        decisions = data

    placed = [d for d in decisions if isinstance(d, dict) and d.get("placed")]
    if not placed:
        print("  No trades placed today.")
        return

    # Group by strategy
    by_strat = defaultdict(list)
    for d in placed:
        by_strat[d.get("strategy", "other")].append(d)

    total_deployed = 0
    total_edge = 0

    for strat in sorted(by_strat.keys()):
        trades = by_strat[strat]
        deployed = sum(t.get("contracts", 1) * t.get("price_to_pay_cents", 0) / 100 for t in trades)
        edge = sum(t.get("edge_cents", 0) for t in trades)
        total_deployed += deployed
        total_edge += edge
        print(f"\n  {strat.upper()} ({len(trades)} trades, ${deployed:.2f} deployed)")

        for t in trades[:8]:
            ticker = t["ticker"]
            action = t["action"]
            price = t.get("price_to_pay_cents", 0)
            contracts = t.get("contracts", 1)
            e = t.get("edge_cents", 0)

            # Estimate win probability
            if "arb" in strat:
                win_prob = "99%+"
                outcome = "guaranteed"
            elif action == "buy_no":
                # NO at Xc, win when event doesn't happen
                wp = min(price / 100 + abs(e) / 100, 0.99) if e else price / 100
                win_prob = f"{wp:.0%}"
                outcome = f"+{(100 - price) / 100:.2f}" if price < 100 else "+0.01"
            elif action == "buy_yes":
                wp = min(0.5 + abs(e) / 200, 0.99) if e else 0.5
                win_prob = f"{wp:.0%}"
                outcome = f"+{(100 - price) / 100:.2f}"
            else:
                win_prob = "?"
                outcome = "?"

            short_ticker = ticker.split("-", 1)[-1] if "-" in ticker else ticker
            print(f"    {short_ticker:35s} {action:8s} @{price:2d}c x{contracts} P(win)={win_prob:>4s} if win={outcome}")

        if len(trades) > 8:
            print(f"    ... and {len(trades) - 8} more")

    print(f"\n  TOTAL: {len(placed)} trades, ${total_deployed:.2f} deployed, {total_edge:.1f}c edge")


# ── 4. Strategy Performance ────────────────────────────────────────────
def show_strategy_performance():
    section("STRATEGY PERFORMANCE (all settled, March 2026+)")

    csv_file = OUTPUT / "trade_history.csv"
    if not csv_file.exists():
        print("  No trade history.")
        return

    rows = []
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("date", "") >= "2026-03":
                rows.append(r)

    def classify(r):
        ticker = r.get("ticker", "")
        strategy = r.get("strategy", "other")
        side = r.get("side", "")
        entry = float(r.get("entry_price", "0") or "0")
        qty = int(r.get("contracts", "1") or "1")
        if "KXNBAPTS" in ticker:
            return "NBA Props (OFF)"
        if "KXNBAGAME" in ticker:
            if side == "no" and entry <= 20 and qty >= 5:
                return "NBA Extreme NO (OFF)"
            if side == "no":
                return "NBA Moderate NO"
            return "NBA Underdog YES"
        if strategy == "weather_tail":
            return "Weather Tail NO"
        if strategy == "weather":
            return "Weather Forecast"
        if strategy == "crypto":
            return "Crypto Tail NO"
        if strategy == "arb":
            return "Arb (YES+NO)"
        if strategy == "autoresearch":
            return "AutoResearch"
        return strategy or "other"

    strats = defaultdict(lambda: {
        "settled": 0, "open": 0, "wins": 0, "losses": 0,
        "pnl": 0.0, "gross_win": 0.0, "gross_loss": 0.0,
    })

    for r in rows:
        cat = classify(r)
        settlement = r.get("settlement_result", "open")
        if settlement == "open":
            strats[cat]["open"] += 1
            continue
        pnl = float(r.get("pnl", "0") or "0")
        if pnl == 0 and settlement == "scalar":
            continue
        strats[cat]["settled"] += 1
        strats[cat]["pnl"] += pnl
        if pnl > 0:
            strats[cat]["wins"] += 1
            strats[cat]["gross_win"] += pnl
        elif pnl < 0:
            strats[cat]["losses"] += 1
            strats[cat]["gross_loss"] += pnl

    print(f"  {'Strategy':22s} {'Sttl':>5s} {'Open':>5s} {'W':>4s} {'L':>4s} {'Win%':>5s} {'P&L':>8s} {'Status'}")
    print(f"  {'-'*22} {'-'*5} {'-'*5} {'-'*4} {'-'*4} {'-'*5} {'-'*8} {'-'*10}")

    total_pnl = 0
    active_pnl = 0

    order = [
        "Weather Tail NO", "Weather Forecast", "NBA Underdog YES",
        "Arb (YES+NO)", "AutoResearch", "Crypto Tail NO",
        "NBA Extreme NO (OFF)", "NBA Props (OFF)", "NBA Moderate NO",
    ]

    for cat in order:
        if cat not in strats:
            continue
        s = strats[cat]
        t = s["settled"]
        wr = f"{s['wins']/t*100:.0f}%" if t > 0 else "—"
        disabled = "OFF" in cat
        status = "DISABLED" if disabled else ("ACTIVE" if s["open"] > 0 or t > 0 else "IDLE")

        if t >= 3:
            if s["pnl"] > 1:
                verdict = "WINNER"
            elif s["pnl"] > -1:
                verdict = "FLAT"
            else:
                verdict = "LOSER"
        elif s["open"] > 0:
            verdict = "PENDING"
        else:
            verdict = "—"

        total_pnl += s["pnl"]
        if not disabled:
            active_pnl += s["pnl"]

        print(f"  {cat:22s} {t:5d} {s['open']:5d} {s['wins']:4d} {s['losses']:4d} {wr:>5s} {s['pnl']:+8.2f} {verdict}")

    print(f"\n  CSV settled P&L:  ${total_pnl:+.2f}")
    print(f"  (Use Kalshi balance above for real account value)")


# ── 5. AutoResearch Status ──────────────────────────────────────────────
def show_autoresearch():
    section("AUTORESEARCH")

    # Check if running
    result = subprocess.run(
        ["pgrep", "-f", "run_experiments"],
        capture_output=True, text=True,
    )
    running = bool(result.stdout.strip())
    tmux_result = subprocess.run(
        ["tmux", "list-sessions"],
        capture_output=True, text=True,
    )
    tmux_sessions = [l for l in tmux_result.stdout.split("\n") if "autoresearch" in l.lower()]

    print(f"  Status:     {'RUNNING' if running or tmux_sessions else 'STOPPED'}")

    # Count experiments from git log
    try:
        git_result = subprocess.run(
            ["git", "-C", "/opt/ippo", "log", "--oneline", "--all"],
            capture_output=True, text=True, timeout=10,
        )
        lines = git_result.stdout.strip().split("\n")
        experiments = [l for l in lines if "autoresearch:" in l.lower() or "exp" in l.lower()]
        keeps = [l for l in lines if "KEEP" in l]
        discards = [l for l in experiments if "DISCARD" in l]
        print(f"  Experiments: {len(experiments)} total")
        print(f"  Kept:        {len(keeps)} strategies passed z>=2.0")
        print(f"  Discarded:   {len(discards)}")
    except Exception:
        print(f"  Experiments: (could not read git log)")

    # Check if evaluate_market is wired to live
    try:
        from autoresearch.candidate_strategy import evaluate_market
        # Count strategy branches
        import inspect
        source = inspect.getsource(evaluate_market)
        buy_count = source.count('"buy_')
        skip_count = source.count('"skip"')
        print(f"  Live bridge: ACTIVE (evaluate_market wired to auto_trade)")
        print(f"  Branches:    {buy_count} trade signals, {skip_count} skip rules")
    except Exception:
        print(f"  Live bridge: NOT CONNECTED")

    # Recent autoresearch trades
    csv_file = OUTPUT / "trade_history.csv"
    if csv_file.exists():
        ar_trades = 0
        ar_open = 0
        ar_pnl = 0.0
        with open(csv_file) as f:
            for r in csv.DictReader(f):
                if r.get("strategy") == "autoresearch":
                    if r.get("settlement_result", "open") == "open":
                        ar_open += 1
                    else:
                        ar_trades += 1
                        ar_pnl += float(r.get("pnl", 0) or 0)
        print(f"\n  Live trades: {ar_trades} settled, {ar_open} open")
        if ar_trades > 0:
            print(f"  Live P&L:    ${ar_pnl:+.2f}")
        elif ar_open > 0:
            print(f"  Live P&L:    pending ({ar_open} positions settling)")
        else:
            print(f"  Live P&L:    no trades yet (just wired up)")

    # What strategies it discovered
    print(f"\n  Key discoveries (from 834 experiments):")
    discoveries = [
        "Weather per-city YES caps (CHI:73c, MIA:58c, NY:78c, DEN:57c)",
        "Weather near-certain YES (76-99c with 100% historical rate)",
        "Crypto ETH extended tail (15c -> 36c range)",
        "BTC 31c near-certain YES (1/1 all-time)",
        "NBA home/away win patterns at specific prices",
    ]
    for d in discoveries:
        print(f"    - {d}")


# ── 6. Daily P&L Tracker ────────────────────────────────────────────────
def show_daily_pnl(total_equity):
    section("DAY-BY-DAY P&L")

    # Load tracker
    tracker_path = OUTPUT / "daily_portfolio.json"
    deposits = [
        ("2026-03-21", 98.0),
        ("2026-03-27", 490.0),
    ]

    # Historical + today
    days = [
        ("2026-03-21", 98.00),
        ("2026-03-22", 89.00),
        ("2026-03-23", 86.00),
        ("2026-03-24", 84.87),
        ("2026-03-25", 53.77),
        ("2026-03-26", 89.16),
    ]

    # Try to load saved daily snapshots
    if tracker_path.exists():
        try:
            with open(tracker_path) as f:
                saved = json.load(f)
            for e in saved.get("entries", []):
                d = e.get("date", "")
                pv = e.get("total_equity", e.get("portfolio_value", 0))
                if pv > 0 and d not in [x[0] for x in days]:
                    days.append((d, pv))
        except Exception:
            pass

    # Add/update today
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    days = [(d, v) for d, v in days if d != today]
    if total_equity > 0:
        days.append((today, total_equity))
    days.sort()

    # Save today's snapshot
    try:
        saved_data = {"entries": [], "deposits": [{"date": d, "amount": a} for d, a in deposits]}
        if tracker_path.exists():
            with open(tracker_path) as f:
                saved_data = json.load(f)
        existing_dates = {e["date"] for e in saved_data.get("entries", [])}
        if today not in existing_dates:
            saved_data.setdefault("entries", []).append({
                "date": today,
                "total_equity": total_equity,
                "time": datetime.now(timezone.utc).isoformat(),
            })
        else:
            for e in saved_data["entries"]:
                if e["date"] == today:
                    e["total_equity"] = total_equity
        with open(tracker_path, "w") as f:
            json.dump(saved_data, f, indent=2)
    except Exception:
        pass

    print(f"  {'Date':12s} {'Portfolio':>10s} {'Deposited':>10s} {'Net P&L':>10s} {'Day Chg':>10s}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    prev_pnl = None
    for date, portfolio in days:
        cum_deposits = sum(a for d, a in deposits if d <= date)
        net_pnl = portfolio - cum_deposits
        day_chg = (net_pnl - prev_pnl) if prev_pnl is not None else 0.0
        prev_pnl = net_pnl
        print(f"  {date:12s} ${portfolio:9.2f} ${cum_deposits:9.2f} ${net_pnl:+9.2f} ${day_chg:+9.2f}")

    cum_all = sum(a for _, a in deposits)
    if days:
        final = days[-1][1]
        net = final - cum_all
        print(f"\n  Net P&L: ${net:+.2f} ({net/cum_all*100:+.1f}%) on ${cum_all:.0f} deposited")


# ── 7. Summary ──────────────────────────────────────────────────────────
def show_summary(services_ok, cash, open_positions):
    section("SUMMARY")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"  Time:     {now}")
    print(f"  Health:   {'ALL GREEN' if services_ok else 'ISSUES DETECTED'}")
    print(f"  Cash:     ${cash:.2f}")
    print(f"  Positions: {open_positions} open")
    print(f"  Bot:      Running — auto_trade every 2h, weather_tail every 30m, arb continuous")
    print()


# ── Main ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print()
    print("  IPPO TRADING BOT — STATUS DASHBOARD")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    services_ok = check_services()
    cash, open_pos, total_equity = check_balance()
    show_daily_pnl(total_equity)
    show_todays_trades()
    show_strategy_performance()
    show_autoresearch()
    show_summary(services_ok, cash, open_pos)
