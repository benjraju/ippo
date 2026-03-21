"""
dashboard.py -- Live HTML dashboard for the Kalshi trading bot.

Fetches account data, positions, fills, AutoResearch status, and bot health.
Generates a self-contained HTML page and opens it in the browser.

Usage:
    python dashboard.py              # Generate and open dashboard
    python cli.py dashboard          # Same, via CLI
"""

import json
import os
import subprocess
import webbrowser
from datetime import datetime, timezone, timedelta
from pathlib import Path

import config
from kalshi_client import KalshiClient

OUTPUT_PATH = config.OUTPUT_DIR / "dashboard.html"
RESULTS_LOG = config.AUTORESEARCH_DIR / "results.log"
LAUNCHD_LOG = config.AUTORESEARCH_DIR / "launchd.log"


def fetch_all_data() -> dict:
    """Fetch everything from Kalshi API + local state."""
    client = KalshiClient()

    # Core account data
    balance_resp = client.get_balance()
    positions_resp = client.get_positions()
    fills_resp = client.get_fills(limit=50)
    orders_resp = client.get_orders(status="resting")

    balance_cents = balance_resp.get("balance", 0)
    portfolio_value_cents = balance_resp.get("portfolio_value", 0)

    # Build positions from API data (no per-position orderbook calls to avoid rate limits)
    market_positions = positions_resp.get("market_positions", [])
    enriched_positions = []

    for mp in market_positions:
        pos = float(mp.get("position_fp", 0))
        if pos == 0:
            continue  # Skip zero positions

        ticker = mp["ticker"]
        exposure = float(mp.get("market_exposure_dollars", 0))
        fees = float(mp.get("fees_paid_dollars", 0))

        if pos > 0:
            side = "YES"
            contracts = int(pos)
        else:
            side = "NO"
            contracts = int(abs(pos))

        # Parse ticker for display
        strategy = "BTC" if "BTC" in ticker else "Weather"
        city = ""
        if "HIGHNY" in ticker:
            city = "NYC"
        elif "HIGHCHI" in ticker:
            city = "Chicago"
        elif "HIGHMIA" in ticker:
            city = "Miami"
        elif "HIGHLA" in ticker:
            city = "LA"
        elif "HIGHDEN" in ticker:
            city = "Denver"

        enriched_positions.append({
            "ticker": ticker,
            "strategy": strategy,
            "city": city,
            "side": side,
            "contracts": contracts,
            "entry_cost": exposure,
            "fees": fees,
            "last_updated": mp.get("last_updated_ts", ""),
        })

    # Event-level summary
    event_positions = positions_resp.get("event_positions", [])

    # Fills
    fills = fills_resp.get("fills", [])

    # Open orders
    orders = orders_resp.get("orders", [])

    # AutoResearch status
    autoresearch = get_autoresearch_status()

    # Bot health
    health = get_bot_health()

    # Calculate KPIs
    total_exposure = sum(p["entry_cost"] for p in enriched_positions)
    total_fees = sum(p["fees"] for p in enriched_positions)
    total_realized_pnl = sum(float(ep.get("realized_pnl_dollars", 0)) for ep in event_positions)
    # Kalshi provides portfolio_value which is the mark-to-market value of all positions
    total_unrealized_pnl = portfolio_value_cents / 100.0 - total_exposure

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "env": config.KALSHI_ENV,
        "balance": balance_cents / 100.0,
        "portfolio_value": portfolio_value_cents / 100.0,
        "total_equity": (balance_cents + portfolio_value_cents) / 100.0,
        "positions": enriched_positions,
        "event_positions": event_positions,
        "fills": fills[:20],  # Last 20
        "open_orders": orders,
        "total_exposure": total_exposure,
        "total_current_value": portfolio_value_cents / 100.0,
        "total_unrealized_pnl": total_unrealized_pnl,
        "total_realized_pnl": total_realized_pnl,
        "total_fees": total_fees,
        "num_positions": len(enriched_positions),
        "num_open_orders": len(orders),
        "autoresearch": autoresearch,
        "health": health,
    }


def get_autoresearch_status() -> dict:
    """Check AutoResearch loop status."""
    status = {
        "total_iterations": 0,
        "improvements": 0,
        "last_run": "Never",
        "best_score": None,
        "best_sortino": None,
        "recent_results": [],
    }

    if not RESULTS_LOG.exists():
        return status

    results = []
    for line in RESULTS_LOG.read_text().strip().split("\n"):
        if line.strip():
            try:
                results.append(json.loads(line))
            except Exception:
                pass

    if not results:
        return status

    status["total_iterations"] = len(results)
    status["improvements"] = sum(1 for r in results if r.get("kept"))
    status["last_run"] = results[-1].get("timestamp", "Unknown")

    kept_results = [r for r in results if r.get("kept")]
    if kept_results:
        best = max(kept_results, key=lambda r: r.get("metrics", {}).get("score", -999))
        status["best_score"] = best.get("metrics", {}).get("score")
        status["best_sortino"] = best.get("metrics", {}).get("sortino")

    status["recent_results"] = results[-10:]
    return status


def get_bot_health() -> dict:
    """Check bot automation health."""
    health = {
        "launchd_loaded": False,
        "launchd_running": False,
        "last_launchd_error": None,
        "nightly_script_ok": False,
        "venv_ok": False,
        "api_connected": True,  # We got this far
    }

    # Check launchd
    try:
        result = subprocess.run(
            ["launchctl", "print", f"gui/{os.getuid()}/com.kalshi.autoresearch"],
            capture_output=True, text=True, timeout=5,
        )
        health["launchd_loaded"] = result.returncode == 0
        if "state = running" in result.stdout:
            health["launchd_running"] = True
    except Exception:
        pass

    # Check launchd log for errors
    if LAUNCHD_LOG.exists():
        try:
            content = LAUNCHD_LOG.read_text().strip()
            if content:
                last_lines = content.split("\n")[-3:]
                errors = [l for l in last_lines if "error" in l.lower() or "permission" in l.lower()]
                if errors:
                    health["last_launchd_error"] = errors[-1][:120]
        except Exception:
            pass

    # Check nightly script
    nightly = config.PROJECT_ROOT / "run_nightly.sh"
    health["nightly_script_ok"] = nightly.exists() and nightly.stat().st_size > 0

    # Check venv
    venv_python = config.PROJECT_ROOT / ".venv" / "bin" / "python"
    health["venv_ok"] = venv_python.exists()

    return health


def generate_html(data: dict) -> str:
    """Generate the dashboard HTML."""

    # Build positions table rows
    pos_rows = ""
    for p in sorted(data["positions"], key=lambda x: x["entry_cost"], reverse=True):
        side_class = "side-yes" if p["side"] == "YES" else "side-no"
        time_str = p.get("last_updated", "")[:19].replace("T", " ")
        pos_rows += f"""
        <tr>
            <td><span class="mono">{p['ticker']}</span></td>
            <td>{p['strategy']}</td>
            <td>{p['city']}</td>
            <td><span class="{side_class}">{p['side']}</span></td>
            <td class="num">{p['contracts']}</td>
            <td class="num">${p['entry_cost']:.2f}</td>
            <td class="num">${p['fees']:.2f}</td>
            <td class="mono dim">{time_str}</td>
        </tr>"""

    # Build fills table rows
    fill_rows = ""
    for f in data["fills"]:
        side_class = "side-yes" if f.get("side") == "yes" else "side-no"
        time_str = f.get("created_time", "")[:19].replace("T", " ")
        cost = float(f.get("no_price_dollars", 0)) if f.get("side") == "no" else float(f.get("yes_price_dollars", 0))
        fill_rows += f"""
        <tr>
            <td class="mono">{time_str}</td>
            <td><span class="mono">{f.get('ticker', '')}</span></td>
            <td>{f.get('action', '').upper()}</td>
            <td><span class="{side_class}">{f.get('side', '').upper()}</span></td>
            <td class="num">{float(f.get('count_fp', 0)):.0f}</td>
            <td class="num">${cost:.2f}</td>
            <td class="num">${float(f.get('fee_cost', 0)):.2f}</td>
        </tr>"""

    # Build open orders rows
    order_rows = ""
    for o in data["open_orders"]:
        side_class = "side-yes" if o.get("side") == "yes" else "side-no"
        price = float(o.get("no_price_dollars", 0)) if o.get("side") == "no" else float(o.get("yes_price_dollars", 0))
        remaining = float(o.get("remaining_count_fp", 0))
        order_rows += f"""
        <tr>
            <td><span class="mono">{o.get('ticker', '')}</span></td>
            <td>{o.get('action', '').upper()}</td>
            <td><span class="{side_class}">{o.get('side', '').upper()}</span></td>
            <td class="num">{remaining:.0f}</td>
            <td class="num">${price:.2f}</td>
            <td class="status-pill status-resting">RESTING</td>
        </tr>"""

    # AutoResearch recent results
    ar = data["autoresearch"]
    ar_rows = ""
    for r in reversed(ar.get("recent_results", [])):
        kept_class = "positive" if r.get("kept") else "dim"
        kept_label = "KEPT" if r.get("kept") else "reverted"
        metrics = r.get("metrics", {})
        ar_rows += f"""
        <tr class="{kept_class}">
            <td class="num">{r.get('iteration', '')}</td>
            <td>{r.get('parameter', '')}</td>
            <td class="num">{r.get('old_value', '')} &rarr; {r.get('new_value', '')}</td>
            <td class="num">{metrics.get('score', 0):.2f}</td>
            <td class="num">{metrics.get('sortino', 0):.1f}</td>
            <td class="num">{metrics.get('win_rate', 0):.0f}%</td>
            <td class="num">{metrics.get('max_dd_pct', 0):.0f}%</td>
            <td><span class="status-pill {'status-ok' if r.get('kept') else 'status-dim'}">{kept_label}</span></td>
        </tr>"""

    # Health indicators
    h = data["health"]
    health_items = [
        ("Kalshi API", data.get("env", "?"), h["api_connected"]),
        ("Launchd Job", "Loaded" if h["launchd_loaded"] else "Not loaded", h["launchd_loaded"]),
        ("Nightly Script", "OK" if h["nightly_script_ok"] else "Missing/empty", h["nightly_script_ok"]),
        ("Python Venv", "OK" if h["venv_ok"] else "Missing", h["venv_ok"]),
    ]

    health_html = ""
    for name, detail, ok in health_items:
        cls = "health-ok" if ok else "health-err"
        dot = "&#9679;" if ok else "&#9679;"
        health_html += f'<div class="health-item {cls}"><span class="health-dot">{dot}</span> {name} <span class="health-detail">{detail}</span></div>'

    if h.get("last_launchd_error"):
        health_html += f'<div class="health-item health-err"><span class="health-dot">&#9679;</span> Last Error <span class="health-detail">{h["last_launchd_error"]}</span></div>'

    # Total P&L color
    total_pnl = data["total_unrealized_pnl"]
    pnl_class = "positive" if total_pnl >= 0 else "negative"
    pnl_sign = "+" if total_pnl >= 0 else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ippo Dashboard</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', system-ui, sans-serif;
        background: #0a0a0f;
        color: #e0e0e8;
        padding: 24px 32px;
        min-height: 100vh;
    }}
    .header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 28px;
        padding-bottom: 16px;
        border-bottom: 1px solid #1a1a2e;
    }}
    .header h1 {{
        font-size: 22px;
        font-weight: 600;
        color: #fff;
        letter-spacing: -0.5px;
    }}
    .header h1 span {{ color: #6c63ff; }}
    .header-meta {{
        font-size: 13px;
        color: #666;
    }}
    .header-meta .env {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-left: 8px;
    }}
    .env-PROD {{ background: #2d1515; color: #ff6b6b; }}
    .env-DEMO {{ background: #152d1f; color: #6bffa8; }}
    .refresh-btn {{
        background: #1a1a2e;
        color: #888;
        border: 1px solid #2a2a3e;
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 12px;
        cursor: pointer;
        transition: all 0.15s;
    }}
    .refresh-btn:hover {{ background: #2a2a3e; color: #fff; }}

    /* KPI Cards */
    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin-bottom: 28px;
    }}
    .kpi-card {{
        background: #111118;
        border: 1px solid #1a1a2e;
        border-radius: 10px;
        padding: 16px 18px;
    }}
    .kpi-label {{
        font-size: 11px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }}
    .kpi-value {{
        font-size: 24px;
        font-weight: 700;
        color: #fff;
        letter-spacing: -0.5px;
    }}
    .kpi-value.positive {{ color: #4ade80; }}
    .kpi-value.negative {{ color: #f87171; }}
    .kpi-sub {{
        font-size: 12px;
        color: #555;
        margin-top: 4px;
    }}

    /* Sections */
    .section {{
        margin-bottom: 28px;
    }}
    .section-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }}
    .section-title {{
        font-size: 14px;
        font-weight: 600;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .section-badge {{
        font-size: 11px;
        color: #666;
        background: #1a1a2e;
        padding: 2px 8px;
        border-radius: 10px;
    }}

    /* Tables */
    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }}
    th {{
        text-align: left;
        padding: 8px 12px;
        color: #555;
        font-weight: 500;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-bottom: 1px solid #1a1a2e;
    }}
    td {{
        padding: 8px 12px;
        border-bottom: 1px solid #111118;
        color: #ccc;
    }}
    tr:hover td {{ background: #111118; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .mono {{ font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px; }}
    .positive {{ color: #4ade80; }}
    .negative {{ color: #f87171; }}
    .dim {{ color: #444; }}
    .side-yes {{ color: #60a5fa; font-weight: 600; }}
    .side-no {{ color: #f472b6; font-weight: 600; }}

    .status-pill {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }}
    .status-ok {{ background: #0f2f1a; color: #4ade80; }}
    .status-resting {{ background: #1a1a2e; color: #60a5fa; }}
    .status-dim {{ background: #1a1a1a; color: #555; }}

    /* Health bar */
    .health-bar {{
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        margin-bottom: 28px;
        padding: 12px 16px;
        background: #111118;
        border: 1px solid #1a1a2e;
        border-radius: 10px;
    }}
    .health-item {{
        font-size: 12px;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .health-dot {{ font-size: 8px; }}
    .health-ok .health-dot {{ color: #4ade80; }}
    .health-err .health-dot {{ color: #f87171; }}
    .health-detail {{ color: #555; font-size: 11px; }}

    /* Two-column layout for smaller tables */
    .two-col {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 24px;
    }}
    @media (max-width: 1000px) {{
        .two-col {{ grid-template-columns: 1fr; }}
    }}

    /* Totals row */
    .totals-row td {{
        border-top: 2px solid #2a2a3e;
        font-weight: 600;
        color: #fff;
    }}
</style>
</head>
<body>

<div class="header">
    <div>
        <h1><span>Ippo</span> Trading Dashboard</h1>
    </div>
    <div class="header-meta">
        {data['timestamp']}
        <span class="env env-{data['env']}">{data['env']}</span>
        <button class="refresh-btn" onclick="location.reload()" style="margin-left: 12px;">Refresh</button>
    </div>
</div>

<div class="health-bar">
    {health_html}
</div>

<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Cash Balance</div>
        <div class="kpi-value">${data['balance']:.2f}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Portfolio Value</div>
        <div class="kpi-value">${data['portfolio_value']:.2f}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Total Equity</div>
        <div class="kpi-value">${data['total_equity']:.2f}</div>
        <div class="kpi-sub">cash + positions</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Capital Deployed</div>
        <div class="kpi-value">${data['total_exposure']:.2f}</div>
        <div class="kpi-sub">{data['num_positions']} positions</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Unrealized P&L</div>
        <div class="kpi-value {pnl_class}">{pnl_sign}${abs(total_pnl):.2f}</div>
        <div class="kpi-sub">fees: ${data['total_fees']:.2f}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Open Orders</div>
        <div class="kpi-value">{data['num_open_orders']}</div>
        <div class="kpi-sub">resting limit orders</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">AutoResearch</div>
        <div class="kpi-value">{ar['improvements']}/{ar['total_iterations']}</div>
        <div class="kpi-sub">improvements found</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Best Sortino</div>
        <div class="kpi-value">{f"{ar['best_sortino']:.1f}" if ar['best_sortino'] else "N/A"}</div>
        <div class="kpi-sub">from autoresearch</div>
    </div>
</div>

<!-- Positions -->
<div class="section">
    <div class="section-header">
        <span class="section-title">Open Positions</span>
        <span class="section-badge">{data['num_positions']} active</span>
    </div>
    <table>
        <thead>
            <tr>
                <th>Ticker</th>
                <th>Strategy</th>
                <th>City</th>
                <th>Side</th>
                <th class="num">Contracts</th>
                <th class="num">Cost</th>
                <th class="num">Fees</th>
                <th>Opened</th>
            </tr>
        </thead>
        <tbody>
            {pos_rows}
            <tr class="totals-row">
                <td colspan="5">TOTAL</td>
                <td class="num">${data['total_exposure']:.2f}</td>
                <td class="num">${data['total_fees']:.2f}</td>
                <td></td>
            </tr>
        </tbody>
    </table>
</div>

<div class="two-col">
    <!-- Recent Fills -->
    <div class="section">
        <div class="section-header">
            <span class="section-title">Recent Fills</span>
            <span class="section-badge">last {len(data['fills'])}</span>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Ticker</th>
                    <th>Action</th>
                    <th>Side</th>
                    <th class="num">Qty</th>
                    <th class="num">Price</th>
                    <th class="num">Fee</th>
                </tr>
            </thead>
            <tbody>{fill_rows}</tbody>
        </table>
    </div>

    <!-- Open Orders -->
    <div class="section">
        <div class="section-header">
            <span class="section-title">Open Orders</span>
            <span class="section-badge">{data['num_open_orders']} resting</span>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Ticker</th>
                    <th>Action</th>
                    <th>Side</th>
                    <th class="num">Qty</th>
                    <th class="num">Price</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>{order_rows if order_rows else '<tr><td colspan="6" style="color:#444;text-align:center;">No open orders</td></tr>'}</tbody>
        </table>
    </div>
</div>

<!-- AutoResearch -->
<div class="section">
    <div class="section-header">
        <span class="section-title">AutoResearch History</span>
        <span class="section-badge">{ar['improvements']} improvements / {ar['total_iterations']} iterations</span>
    </div>
    <table>
        <thead>
            <tr>
                <th class="num">Iter</th>
                <th>Parameter</th>
                <th class="num">Change</th>
                <th class="num">Score</th>
                <th class="num">Sortino</th>
                <th class="num">Win Rate</th>
                <th class="num">Max DD</th>
                <th>Result</th>
            </tr>
        </thead>
        <tbody>{ar_rows if ar_rows else '<tr><td colspan="8" style="color:#444;text-align:center;">No research runs yet</td></tr>'}</tbody>
    </table>
</div>

<div style="text-align:center; color:#333; font-size:11px; margin-top:32px; padding-top:16px; border-top:1px solid #1a1a2e;">
    Ippo &middot; Kalshi Self-Improving Bot &middot; Generated {data['timestamp']}
    &middot; <a href="javascript:location.reload()" style="color:#555;">Regenerate to refresh</a>
</div>

</body>
</html>"""


def run_dashboard():
    """Generate dashboard and open in browser."""
    from rich.console import Console
    console = Console()
    console.print("[cyan]Fetching live data from Kalshi...[/cyan]")

    try:
        data = fetch_all_data()
    except Exception as e:
        console.print(f"[red]Error fetching data: {e}[/red]")
        return

    html = generate_html(data)
    OUTPUT_PATH.write_text(html)
    console.print(f"[green]Dashboard saved to: {OUTPUT_PATH}[/green]")
    webbrowser.open(f"file://{OUTPUT_PATH}")


if __name__ == "__main__":
    run_dashboard()
