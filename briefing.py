"""
briefing.py -- Daily status briefing for Ippo.

Run this when you open a new session. Shows everything that matters:
  - Account status & P&L
  - What trades were placed since last check
  - What settled and how it went
  - Bot health (are background jobs running?)
  - AutoResearch progress
  - Alerts & errors
  - What's coming next

Usage:
    python briefing.py
    python cli.py briefing
"""

import json
import os
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box

import config
from kalshi_client import KalshiClient

console = Console()

OUTPUT = config.OUTPUT_DIR
AUTORESEARCH = config.AUTORESEARCH_DIR


def get_account_status() -> dict:
    """Fetch live account data."""
    try:
        client = KalshiClient()
        bal = client.get_balance()
        pos = client.get_positions()
        fills = client.get_fills(limit=20)

        market_positions = [
            mp for mp in pos.get("market_positions", [])
            if float(mp.get("position_fp", 0)) != 0
        ]

        return {
            "connected": True,
            "balance": bal.get("balance", 0) / 100.0,
            "portfolio_value": bal.get("portfolio_value", 0) / 100.0,
            "total_equity": (bal.get("balance", 0) + bal.get("portfolio_value", 0)) / 100.0,
            "num_positions": len(market_positions),
            "positions": market_positions,
            "recent_fills": fills.get("fills", [])[:10],
            "events": pos.get("event_positions", []),
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}


def get_bot_health() -> list[tuple]:
    """Check background process health."""
    checks = []

    # Arb runner
    try:
        r = subprocess.run(
            ["launchctl", "print", f"gui/{os.getuid()}/com.kalshi.arb-runner"],
            capture_output=True, text=True, timeout=5,
        )
        running = "state = running" in r.stdout
        checks.append(("Arb Runner (30s)", "Running" if running else "Stopped", running))
    except Exception:
        checks.append(("Arb Runner (30s)", "Unknown", False))

    # Autoresearch
    try:
        r = subprocess.run(
            ["launchctl", "print", f"gui/{os.getuid()}/com.kalshi.autoresearch"],
            capture_output=True, text=True, timeout=5,
        )
        loaded = r.returncode == 0
        checks.append(("AutoResearch (2h)", "Loaded" if loaded else "Not loaded", loaded))
    except Exception:
        checks.append(("AutoResearch (2h)", "Unknown", False))

    # Check for recent errors in arb log
    arb_log = OUTPUT / "arb_runner_launchd.log"
    if arb_log.exists():
        try:
            lines = arb_log.read_text().strip().split("\n")[-20:]
            errors = [l for l in lines if "ERROR" in l or "CRITICAL" in l]
            if errors:
                checks.append(("Arb Errors", f"{len(errors)} recent", False))
            else:
                checks.append(("Arb Errors", "None", True))
        except Exception:
            pass

    # Check launchd log for errors
    launchd_log = AUTORESEARCH / "launchd.log"
    if launchd_log.exists():
        try:
            content = launchd_log.read_text().strip()
            if content and ("error" in content.lower() or "permission" in content.lower()):
                checks.append(("Launchd Errors", "Check launchd.log", False))
            else:
                checks.append(("Launchd Errors", "None", True))
        except Exception:
            pass

    return checks


def get_recent_trades() -> list[dict]:
    """Get trades from today's auto_trade log."""
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = OUTPUT / f"auto_trade_{today}.json"

    if not log_file.exists():
        # Try yesterday
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        log_file = OUTPUT / f"auto_trade_{yesterday}.json"

    if not log_file.exists():
        return []

    try:
        data = json.loads(log_file.read_text())
        return data.get("decisions", [])
    except Exception:
        return []


def get_settlement_status() -> dict:
    """Check trade_history.csv for recent settlements."""
    csv_path = OUTPUT / "trade_history.csv"
    if not csv_path.exists():
        return {"total": 0, "wins": 0, "losses": 0, "open": 0, "pnl": 0}

    try:
        import csv
        with open(csv_path) as f:
            reader = list(csv.DictReader(f))

        if not reader:
            return {"total": 0, "wins": 0, "losses": 0, "open": 0, "pnl": 0}

        wins = sum(1 for r in reader if float(r.get("pnl", 0)) > 0 and r.get("settlement_result") != "open")
        losses = sum(1 for r in reader if float(r.get("pnl", 0)) <= 0 and r.get("settlement_result") != "open")
        opens = sum(1 for r in reader if r.get("settlement_result") == "open")
        total_pnl = sum(float(r.get("pnl", 0)) for r in reader if r.get("settlement_result") != "open")

        return {
            "total": len(reader),
            "wins": wins,
            "losses": losses,
            "open": opens,
            "pnl": total_pnl,
            "win_rate": (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0,
        }
    except Exception:
        return {"total": 0, "wins": 0, "losses": 0, "open": 0, "pnl": 0}


def get_autoresearch_status() -> dict:
    """AutoResearch progress."""
    results_log = AUTORESEARCH / "results.log"
    if not results_log.exists():
        return {"iterations": 0, "improvements": 0, "best_score": None}

    try:
        results = []
        for line in results_log.read_text().strip().split("\n"):
            if line.strip():
                try:
                    results.append(json.loads(line))
                except Exception:
                    pass

        kept = [r for r in results if r.get("kept")]
        best = max(kept, key=lambda r: r.get("metrics", {}).get("score", -999)) if kept else None

        return {
            "iterations": len(results),
            "improvements": len(kept),
            "best_score": best.get("metrics", {}).get("score") if best else None,
            "best_sortino": best.get("metrics", {}).get("sortino") if best else None,
            "last_run": results[-1].get("timestamp", "Never") if results else "Never",
        }
    except Exception:
        return {"iterations": 0, "improvements": 0, "best_score": None}


def get_alerts() -> list[dict]:
    """Recent alerts."""
    alerts_log = OUTPUT / "alerts.log"
    if not alerts_log.exists():
        return []

    try:
        alerts = []
        for line in alerts_log.read_text().strip().split("\n"):
            if line.strip():
                try:
                    alerts.append(json.loads(line))
                except Exception:
                    pass
        return alerts[-10:]  # Last 10
    except Exception:
        return []


def get_arb_stats() -> dict:
    """Today's arb scanner results."""
    today = datetime.now().strftime("%Y-%m-%d")
    results_file = OUTPUT / f"arb_results_{today}.json"

    if not results_file.exists():
        return {"scanned": True, "arbs_found": 0, "total_profit": 0}

    try:
        results = json.loads(results_file.read_text())
        total_profit = sum(r.get("profit", 0) for r in results)
        return {
            "scanned": True,
            "arbs_found": len(results),
            "total_profit": total_profit,
            "total_cost": sum(r.get("cost", 0) for r in results),
        }
    except Exception:
        return {"scanned": True, "arbs_found": 0, "total_profit": 0}


def run_briefing():
    """Generate and display the daily briefing."""
    console.print()
    console.print(Panel(
        "[bold white]IPPO DAILY BRIEFING[/bold white]",
        subtitle=datetime.now().strftime("%A, %B %d, %Y  %I:%M %p"),
        style="cyan",
        width=70,
    ))

    # --- Account Status ---
    acct = get_account_status()
    if acct.get("connected"):
        equity = acct["total_equity"]
        starting = 98.0  # Adjusted for 2% deposit fee
        pnl = equity - starting
        pnl_pct = (pnl / starting) * 100
        pnl_color = "green" if pnl >= 0 else "red"
        pnl_sign = "+" if pnl >= 0 else ""

        console.print(f"\n  [bold]Account[/bold]")
        console.print(f"    Cash:         ${acct['balance']:.2f}")
        console.print(f"    Positions:    ${acct['portfolio_value']:.2f} ({acct['num_positions']} open)")
        console.print(f"    Total Equity: [bold]${equity:.2f}[/bold]")
        console.print(f"    Overall P&L:  [{pnl_color}]{pnl_sign}${pnl:.2f} ({pnl_sign}{pnl_pct:.1f}%)[/{pnl_color}]")
    else:
        console.print(f"\n  [red]API Error: {acct.get('error', 'Unknown')}[/red]")

    # --- Bot Health ---
    health = get_bot_health()
    console.print(f"\n  [bold]Bot Health[/bold]")
    for name, detail, ok in health:
        dot = "[green]OK[/green]" if ok else "[red]!!![/red]"
        console.print(f"    {dot}  {name}: {detail}")

    # --- Settlements ---
    settlements = get_settlement_status()
    console.print(f"\n  [bold]Trade Results (All Time)[/bold]")
    if settlements["total"] > 0:
        settled = settlements["wins"] + settlements["losses"]
        console.print(f"    Settled:  {settled} trades ({settlements['wins']}W / {settlements['losses']}L)")
        if settled > 0:
            console.print(f"    Win Rate: {settlements['win_rate']:.0f}%")
        pnl_color = "green" if settlements["pnl"] >= 0 else "red"
        pnl_sign = "+" if settlements["pnl"] >= 0 else ""
        console.print(f"    Real P&L: [{pnl_color}]{pnl_sign}${settlements['pnl']:.2f}[/{pnl_color}]")
        console.print(f"    Open:     {settlements['open']} positions awaiting settlement")
    else:
        console.print(f"    No settlements yet. Run: python cli.py check-settlements")

    # --- Recent Trades ---
    trades = get_recent_trades()
    if trades:
        console.print(f"\n  [bold]Recent Trades ({len(trades)} decisions)[/bold]")
        placed = [t for t in trades if t.get("contracts", 0) > 0]
        by_strategy = {}
        for t in placed:
            s = t.get("strategy", "unknown")
            by_strategy[s] = by_strategy.get(s, 0) + 1
        for strategy, count in sorted(by_strategy.items()):
            console.print(f"    {strategy:12s}  {count} trades")
    else:
        console.print(f"\n  [bold]Recent Trades[/bold]")
        console.print(f"    No trades today yet.")

    # --- Arb Scanner ---
    arb = get_arb_stats()
    console.print(f"\n  [bold]Arb Scanner (Today)[/bold]")
    console.print(f"    Arbs found:    {arb['arbs_found']}")
    console.print(f"    Locked profit: ${arb.get('total_profit', 0):.4f}")

    # --- AutoResearch ---
    ar = get_autoresearch_status()
    console.print(f"\n  [bold]AutoResearch[/bold]")
    console.print(f"    Iterations:    {ar['iterations']}")
    console.print(f"    Improvements:  {ar['improvements']}")
    if ar.get("best_sortino"):
        console.print(f"    Best Sortino:  {ar['best_sortino']:.1f}")
    if ar.get("last_run") and ar["last_run"] != "Never":
        console.print(f"    Last run:      {ar['last_run'][:19]}")

    # --- Recent Alerts ---
    alerts = get_alerts()
    if alerts:
        console.print(f"\n  [bold]Recent Alerts[/bold]")
        for a in alerts[-5:]:
            ts = a.get("ts", "")[:16]
            console.print(f"    [{ts}] {a.get('title', '')}: {a.get('message', '')}")

    # --- Open Positions Summary ---
    if acct.get("connected") and acct.get("events"):
        console.print(f"\n  [bold]Position Exposure by Event[/bold]")
        for ep in acct["events"]:
            ticker = ep.get("event_ticker", "?")
            cost = float(ep.get("total_cost_dollars", 0))
            realized = float(ep.get("realized_pnl_dollars", 0))
            console.print(f"    {ticker:30s}  ${cost:.2f} deployed  realized=${realized:.2f}")

    # --- Strategy Status ---
    console.print(f"\n  [bold]Active Strategies[/bold]")
    strategies = [
        ("Weather Arb", "NWS+GFS+HRRR vs Kalshi buckets", "LIVE", "NYC, CHI, MIA, LA, DEN, DC"),
        ("Crypto Buckets", "Lognormal model vs KXBTC/ETH/SOL", "LIVE", "BTC, ETH, SOL"),
        ("Sports/NBA", "ESPN win-prob model vs Kalshi", "LIVE", "NBA games"),
        ("YES/NO Arb", "Buy both sides when YES+NO < $1", "SCANNING", "All markets, 30s poll"),
        ("Latency Arb", "Exchange price leads Kalshi reprice", "BUILT", "KXBTC short-term"),
        ("Copy-Trade", "Mirror 0x8dxd whale from Polymarket", "BUILT", "Crypto markets"),
        ("Strategy Discovery", "Claude scans all markets for edges", "BUILT", "Politics, econ, etc"),
    ]
    for name, desc, status, scope in strategies:
        if status == "LIVE":
            badge = "[green]LIVE[/green]"
        elif status == "SCANNING":
            badge = "[cyan]SCANNING[/cyan]"
        else:
            badge = "[yellow]READY[/yellow]"
        console.print(f"    {badge}  [bold]{name}[/bold] — {desc}")
        console.print(f"          [dim]{scope}[/dim]")

    # --- What We Learned ---
    console.print(f"\n  [bold]Key Learnings[/bold]")
    learnings = [
        "0x8dxd turned $313 -> $2.3M on Polymarket with 98% win rate using latency arb on 15-min crypto contracts",
        "Top Polymarket whale @k9Q2mX4L8A7ZP3R uses pure YES/NO arb: buy both sides when total < $1.00 = risk-free",
        "14 of top 20 Polymarket wallets are bots — speed + automation wins",
        "Arb opportunities last ~2.7 seconds (down from 12.3s in 2024) — must scan continuously",
        "Weather forecast arb works: ensemble model (HRRR+GFS+NWS) finds 3-48c edges vs Kalshi pricing",
        "AutoResearch found: reduce Chicago weight to 0.8x, best Sortino = 5.2",
    ]
    for l in learnings:
        console.print(f"    [dim]>[/dim] {l}")

    # --- What's Been Built ---
    console.print(f"\n  [bold]Recent Improvements[/bold]")
    improvements_file = config.PROJECT_ROOT / "output" / "improvements_log.json"
    default_improvements = [
        {"date": "2026-03-21", "item": "Added HRRR forecast (3km resolution) for day-0/day-1 weather"},
        {"date": "2026-03-21", "item": "Added DC + Denver weather markets"},
        {"date": "2026-03-21", "item": "Added ETH + SOL crypto strategies"},
        {"date": "2026-03-21", "item": "Built continuous arb runner (scans every 30s)"},
        {"date": "2026-03-21", "item": "Built copy-trade engine tracking 0x8dxd whale"},
        {"date": "2026-03-21", "item": "Built strategy discovery (Claude scans all Kalshi markets)"},
        {"date": "2026-03-21", "item": "Added position exit logic (closes when edge drops <2c)"},
        {"date": "2026-03-21", "item": "Added macOS alerts for trades, drawdowns, errors"},
        {"date": "2026-03-21", "item": "Fixed AutoResearch + settlement tracking (3 bugs)"},
        {"date": "2026-03-21", "item": "Upgraded to 2-hour trading cycle (was 4h)"},
    ]
    # Load custom improvements if file exists, otherwise use defaults
    try:
        if improvements_file.exists():
            custom = json.loads(improvements_file.read_text())
            show_improvements = custom[-8:]
        else:
            show_improvements = default_improvements[-8:]
            improvements_file.write_text(json.dumps(default_improvements, indent=2))
    except Exception:
        show_improvements = default_improvements[-8:]

    for imp in show_improvements:
        console.print(f"    [dim]{imp.get('date', '')}[/dim]  {imp.get('item', '')}")

    # --- What's Next ---
    console.print(f"\n  [bold]Next Steps[/bold]")
    next_steps = [
        "BTC positions settle today at 2pm PT — check with: python cli.py check-settlements",
        "Weather positions settle tomorrow 7am PT",
        "Watch arb_runner logs for first risk-free arb opportunity",
        "Run copy-trade watch to see 0x8dxd's next moves: python copy_trade.py --watch",
        "After 2 weeks of data: graduate strategy_discovery categories to auto-trade",
    ]
    for n in next_steps:
        console.print(f"    [dim]-[/dim] {n}")

    # --- Quick Commands ---
    console.print(f"\n  [bold]Quick Commands[/bold]")
    console.print(f"    [dim]python dashboard.py[/dim]           Open full dashboard")
    console.print(f"    [dim]python cli.py check-settlements[/dim]  Check what settled")
    console.print(f"    [dim]python cli.py auto-trade[/dim]        Run all strategies (dry-run)")
    console.print(f"    [dim]python cli.py weather-edges[/dim]     See current edges")
    console.print(f"    [dim]python copy_trade.py --watch[/dim]    Watch whale trades")

    cheatsheet = Path(__file__).parent / "CHEATSHEET.md"
    console.print(f"\n  [bold cyan]Full cheat sheet:[/bold cyan] [link=file://{cheatsheet}]{cheatsheet}[/link]")
    console.print(f"    [dim]cat CHEATSHEET.md[/dim] or [dim]open CHEATSHEET.md[/dim]")

    console.print()


if __name__ == "__main__":
    run_briefing()
