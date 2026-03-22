"""
edge_logger.py -- Daily edge existence tracker.

Records what edges the model finds in real Kalshi markets,
then checks settlement outcomes to measure if edges are real.

This is the most important test: proving edges exist in real markets.

Usage:
    python edge_logger.py --log       # Log today's edges (run daily)
    python edge_logger.py --check     # Check settled edges for accuracy
    python edge_logger.py --report    # Full accuracy report
"""

import sys
import re
import json
import math
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

import config
from kalshi_client import KalshiClient
from weather_strategy import (
    NWS_GRID_POINTS,
    FORECAST_STDEV,
    fetch_nws_forecast,
    parse_market_type,
    calc_bucket_probability,
    calc_above_probability,
    calc_below_probability,
    normal_cdf,
)

# Import FORECAST_STDEV from candidate_strategy if available (AutoResearch-tuned values)
sys.path.insert(0, str(config.AUTORESEARCH_DIR))
try:
    from candidate_strategy import get_forecast_stdev
    TUNED_STDEV = get_forecast_stdev()
except ImportError:
    TUNED_STDEV = FORECAST_STDEV

from rich.console import Console
from rich.table import Table

console = Console()

# ── paths ────────────────────────────────────────────────────────────────────
EDGE_LOG_PATH = config.OUTPUT_DIR / "edge_log.jsonl"

# City label mapping (matches weather_strategy.py)
CITY_LABELS = {
    "KXHIGHNY": "NYC",
    "KXHIGHCHI": "Chicago",
    "KXHIGHMIA": "Miami",
    "KXHIGHLA": "LA",
    "KXHIGHDC": "DC",
    "KXHIGHDEN": "Denver",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _read_log() -> list[dict]:
    """Read all entries from the JSONL log."""
    if not EDGE_LOG_PATH.exists():
        return []
    entries = []
    with open(EDGE_LOG_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _write_log(entries: list[dict]):
    """Overwrite the JSONL log with updated entries."""
    with open(EDGE_LOG_PATH, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _append_entries(entries: list[dict]):
    """Append new entries to the JSONL log."""
    with open(EDGE_LOG_PATH, "a") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _parse_ticker_date(ticker: str) -> str | None:
    """Extract settlement date from ticker like KXHIGHCHI-26MAR21-T63 -> '2026-03-21'."""
    date_match = re.search(r'(\d{2})([A-Z]{3})(\d{2})', ticker)
    if not date_match:
        return None
    year_short = date_match.group(1)
    month_str = date_match.group(2)
    day_str = date_match.group(3)
    month_map = {
        "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
        "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
        "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
    }
    month_num = month_map.get(month_str)
    if not month_num:
        return None
    return f"20{year_short}-{month_num}-{day_str}"


def _binomial_p_value(accuracy: float, n: int) -> float:
    """One-sided p-value for accuracy > 0.5 using normal approximation of binomial."""
    if n == 0:
        return 1.0
    z = (accuracy - 0.5) / math.sqrt(0.5 * 0.5 / n)
    p = 1 - normal_cdf(z)
    return p


# ── log mode ─────────────────────────────────────────────────────────────────

def do_log():
    """Fetch current markets and forecasts, log all edges."""
    try:
        client = KalshiClient()
    except (FileNotFoundError, Exception) as e:
        console.print(f"[red]Cannot initialize KalshiClient: {e}[/red]")
        console.print("[dim]Make sure KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH "
                       "are set in .env[/dim]")
        sys.exit(1)

    now = datetime.now(timezone.utc)
    new_entries = []
    total_markets = 0

    for series_ticker in NWS_GRID_POINTS:
        city = CITY_LABELS.get(series_ticker, series_ticker)

        # Fetch NWS forecast
        forecast_periods = fetch_nws_forecast(series_ticker)
        if not forecast_periods:
            console.print(f"[yellow]No forecast for {city}, skipping[/yellow]")
            continue

        # Fetch open markets
        try:
            resp = client.get_markets(series_ticker=series_ticker, limit=100, status="open")
            markets = resp.get("markets", [])
        except Exception as e:
            console.print(f"[yellow]Failed to fetch markets for {city}: {e}[/yellow]")
            continue

        for m in markets:
            ticker = m.get("ticker", "")
            title = m.get("title", "")
            yes_bid = float(m.get("yes_bid_dollars", 0) or 0) * 100
            yes_ask = float(m.get("yes_ask_dollars", 0) or 0) * 100

            # Use midpoint as market price; fall back to whichever side is available
            if yes_bid > 0 and yes_ask > 0:
                market_price = (yes_bid + yes_ask) / 2.0
            elif yes_ask > 0:
                market_price = yes_ask
            elif yes_bid > 0:
                market_price = yes_bid
            else:
                continue  # no price at all

            # Parse market type
            mtype = parse_market_type(ticker, title)
            if not mtype:
                continue

            # Extract settlement date from ticker
            market_date = _parse_ticker_date(ticker)
            if not market_date or market_date not in forecast_periods:
                continue

            forecast_temp = forecast_periods[market_date]["temperature"]

            # Days out
            try:
                market_dt = datetime.strptime(market_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                days_out = max(0, (market_dt.date() - now.date()).days)
            except Exception:
                days_out = 1

            stdev = TUNED_STDEV.get(min(days_out, 3), 4.0)

            # Calculate model fair value
            if mtype["type"] == "bucket":
                fair_prob = calc_bucket_probability(forecast_temp, mtype["low"], mtype["high"], stdev)
                bucket_low = mtype["low"]
                bucket_high = mtype["high"]
            elif mtype["type"] == "above":
                fair_prob = calc_above_probability(forecast_temp, mtype["threshold"], stdev)
                bucket_low = mtype["threshold"]
                bucket_high = None
            elif mtype["type"] == "below":
                fair_prob = calc_below_probability(forecast_temp, mtype["threshold"], stdev)
                bucket_low = None
                bucket_high = mtype["threshold"]
            else:
                continue

            fair_value_cents = round(fair_prob * 100, 2)
            edge_cents = round(fair_value_cents - market_price, 2)

            # Determine side: positive edge -> buy_yes, negative edge -> buy_no
            if edge_cents >= 0:
                side = "buy_yes"
            else:
                side = "buy_no"

            entry = {
                "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "ticker": ticker,
                "city": city,
                "market_type": mtype["type"],
                "market_yes_price": round(market_price, 2),
                "model_fair_value": fair_value_cents,
                "edge_cents": round(abs(edge_cents), 2),
                "side": side,
                "forecast_temp": forecast_temp,
                "bucket_low": bucket_low,
                "bucket_high": bucket_high,
                "days_out": days_out,
                "settled": False,
                "settlement_result": None,
                "edge_was_correct": None,
            }
            new_entries.append(entry)
            total_markets += 1

        # Rate-limit between series (NWS + Kalshi)
        time.sleep(0.8)

    # Append to log
    _append_entries(new_entries)

    console.print(f"[green]Logged {total_markets} markets across "
                  f"{len(NWS_GRID_POINTS)} cities[/green]")
    console.print(f"[dim]Log file: {EDGE_LOG_PATH}[/dim]")

    # Show summary of largest edges
    edges_with_edge = [e for e in new_entries if e["edge_cents"] > 3]
    if edges_with_edge:
        edges_with_edge.sort(key=lambda e: e["edge_cents"], reverse=True)
        table = Table(title="Top Edges Found", show_lines=False)
        table.add_column("Ticker", style="cyan", width=30)
        table.add_column("City", width=8)
        table.add_column("Mkt Price", justify="right", width=10)
        table.add_column("Fair Val", justify="right", width=10)
        table.add_column("Edge", justify="right", style="bold green", width=8)
        table.add_column("Side", width=10)

        for e in edges_with_edge[:15]:
            side_color = "green" if e["side"] == "buy_yes" else "red"
            table.add_row(
                e["ticker"],
                e["city"],
                f'{e["market_yes_price"]:.1f}c',
                f'{e["model_fair_value"]:.1f}c',
                f'+{e["edge_cents"]:.1f}c',
                f'[{side_color}]{e["side"]}[/{side_color}]',
            )
        console.print(table)
    else:
        console.print("[dim]No edges > 3c found right now.[/dim]")


# ── check mode ───────────────────────────────────────────────────────────────

def do_check():
    """Check settled markets and update log with outcomes."""
    entries = _read_log()
    if not entries:
        console.print("[yellow]No entries in edge log yet. Run --log first.[/yellow]")
        return

    try:
        client = KalshiClient()
    except (FileNotFoundError, Exception) as e:
        console.print(f"[red]Cannot initialize KalshiClient: {e}[/red]")
        sys.exit(1)

    unsettled = [e for e in entries if not e["settled"]]
    if not unsettled:
        console.print("[green]All entries already settled.[/green]")
        return

    console.print(f"[dim]Checking {len(unsettled)} unsettled entries...[/dim]")

    # Group by ticker to avoid duplicate API calls
    tickers_to_check = set(e["ticker"] for e in unsettled)
    settlement_cache = {}

    for ticker in tickers_to_check:
        try:
            resp = client.get_market(ticker)
            market = resp.get("market", resp)  # Handle both wrapped and unwrapped responses
            status = market.get("status", "")
            result = market.get("result", "")
            settlement_cache[ticker] = {
                "status": status,
                "result": result,
            }
        except Exception as e:
            console.print(f"[yellow]Could not check {ticker}: {e}[/yellow]")
        time.sleep(0.3)  # Rate limit

    # Update entries
    settled_count = 0
    for entry in entries:
        if entry["settled"]:
            continue

        ticker = entry["ticker"]
        if ticker not in settlement_cache:
            continue

        info = settlement_cache[ticker]
        if info["status"] not in ("settled", "finalized", "closed"):
            continue

        result = info["result"]  # "yes" or "no"
        if not result:
            continue

        entry["settled"] = True
        entry["settlement_result"] = result

        # Determine if edge was correct
        if entry["side"] == "buy_yes" and result == "yes":
            entry["edge_was_correct"] = True
        elif entry["side"] == "buy_no" and result == "no":
            entry["edge_was_correct"] = True
        elif entry["side"] == "buy_yes" and result == "no":
            entry["edge_was_correct"] = False
        elif entry["side"] == "buy_no" and result == "yes":
            entry["edge_was_correct"] = False
        else:
            entry["edge_was_correct"] = None

        settled_count += 1

    _write_log(entries)
    console.print(f"[green]Updated {settled_count} entries with settlement results.[/green]")


# ── report mode ──────────────────────────────────────────────────────────────

def do_report():
    """Print full accuracy report."""
    entries = _read_log()
    if not entries:
        console.print("[yellow]No entries in edge log. Run --log first.[/yellow]")
        return

    total = len(entries)
    settled = [e for e in entries if e["settled"]]
    unsettled_count = total - len(settled)

    console.print()
    console.print("[bold]EDGE ACCURACY REPORT[/bold]")
    console.print("=" * 50)
    console.print(f"Total edges logged:  {total}")
    console.print(f"Settled:             {len(settled)}")
    console.print(f"Unsettled:           {unsettled_count}")
    console.print()

    if not settled:
        console.print("[yellow]No settled entries yet. Run --check after markets settle.[/yellow]")
        return

    # Accuracy by edge threshold
    thresholds = [0, 3, 5, 7, 10]
    table = Table(title="Accuracy by Edge Size", show_lines=False)
    table.add_column("Min Edge", justify="right", width=10)
    table.add_column("Correct", justify="right", width=10)
    table.add_column("Total", justify="right", width=10)
    table.add_column("Accuracy", justify="right", width=10)
    table.add_column("p-value", justify="right", width=12)
    table.add_column("Sig?", width=15)

    for threshold in thresholds:
        subset = [e for e in settled if e["edge_cents"] > threshold and e["edge_was_correct"] is not None]
        if not subset:
            table.add_row(f">{threshold}c", "0", "0", "-", "-", "-")
            continue
        correct = sum(1 for e in subset if e["edge_was_correct"])
        n = len(subset)
        accuracy = correct / n
        p = _binomial_p_value(accuracy, n)
        sig = "[bold green]SIGNIFICANT[/bold green]" if p < 0.05 else "[dim]not sig[/dim]"
        table.add_row(
            f">{threshold}c",
            str(correct),
            str(n),
            f"{accuracy:.1%}",
            f"{p:.4f}",
            sig,
        )

    console.print(table)
    console.print()

    # By city
    cities = sorted(set(e["city"] for e in settled))
    if cities:
        table = Table(title="Accuracy by City", show_lines=False)
        table.add_column("City", width=10)
        table.add_column("Correct", justify="right", width=10)
        table.add_column("Total", justify="right", width=10)
        table.add_column("Accuracy", justify="right", width=10)

        for city in cities:
            subset = [e for e in settled if e["city"] == city and e["edge_cents"] > 3 and e["edge_was_correct"] is not None]
            if not subset:
                continue
            correct = sum(1 for e in subset if e["edge_was_correct"])
            n = len(subset)
            accuracy = correct / n
            table.add_row(city, str(correct), str(n), f"{accuracy:.1%}")

        console.print(table)
        console.print()

    # By days out
    days_values = sorted(set(e["days_out"] for e in settled))
    if days_values:
        table = Table(title="Accuracy by Days Out", show_lines=False)
        table.add_column("Days Out", justify="right", width=10)
        table.add_column("Correct", justify="right", width=10)
        table.add_column("Total", justify="right", width=10)
        table.add_column("Accuracy", justify="right", width=10)

        for d in days_values:
            subset = [e for e in settled if e["days_out"] == d and e["edge_cents"] > 3 and e["edge_was_correct"] is not None]
            if not subset:
                continue
            correct = sum(1 for e in subset if e["edge_was_correct"])
            n = len(subset)
            accuracy = correct / n
            table.add_row(f"Day {d}", str(correct), str(n), f"{accuracy:.1%}")

        console.print(table)
        console.print()

    # By market type
    market_types = sorted(set(e["market_type"] for e in settled))
    if len(market_types) > 1:
        table = Table(title="Accuracy by Market Type", show_lines=False)
        table.add_column("Type", width=10)
        table.add_column("Correct", justify="right", width=10)
        table.add_column("Total", justify="right", width=10)
        table.add_column("Accuracy", justify="right", width=10)

        for mt in market_types:
            subset = [e for e in settled if e["market_type"] == mt and e["edge_cents"] > 3 and e["edge_was_correct"] is not None]
            if not subset:
                continue
            correct = sum(1 for e in subset if e["edge_was_correct"])
            n = len(subset)
            accuracy = correct / n
            table.add_row(mt, str(correct), str(n), f"{accuracy:.1%}")

        console.print(table)
        console.print()

    # Final verdict
    sig_subset = [e for e in settled if e["edge_cents"] > 3 and e["edge_was_correct"] is not None]
    if sig_subset:
        correct = sum(1 for e in sig_subset if e["edge_was_correct"])
        n = len(sig_subset)
        accuracy = correct / n
        p = _binomial_p_value(accuracy, n)

        console.print("-" * 50)
        if accuracy > 0.55 and p < 0.05:
            console.print("[bold green]VERDICT: EDGES ARE REAL[/bold green]")
            console.print(f"[green]Edge >3c accuracy: {accuracy:.1%} (n={n}, p={p:.4f})[/green]")
        elif accuracy > 0.50 and p < 0.10:
            console.print("[bold yellow]VERDICT: PROMISING BUT NEED MORE DATA[/bold yellow]")
            console.print(f"[yellow]Edge >3c accuracy: {accuracy:.1%} (n={n}, p={p:.4f})[/yellow]")
        else:
            console.print("[bold red]VERDICT: EDGES ARE NOISE[/bold red]")
            console.print(f"[red]Edge >3c accuracy: {accuracy:.1%} (n={n}, p={p:.4f})[/red]")
    else:
        console.print("[yellow]Not enough settled data for a verdict.[/yellow]")

    console.print()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Edge Logger -- Track whether model edges are real.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python edge_logger.py --log       # Log today's edges
  python edge_logger.py --check     # Update settled markets
  python edge_logger.py --report    # Full accuracy report
  python edge_logger.py --log --report  # Log + report in one go
        """,
    )
    parser.add_argument("--log", action="store_true", help="Log current edges from live markets")
    parser.add_argument("--check", action="store_true", help="Check settled markets and update outcomes")
    parser.add_argument("--report", action="store_true", help="Print accuracy report")

    args = parser.parse_args()

    if not (args.log or args.check or args.report):
        parser.print_help()
        sys.exit(0)

    if args.log:
        do_log()

    if args.check:
        do_check()

    if args.report:
        do_report()


if __name__ == "__main__":
    main()
