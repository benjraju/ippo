"""
strategy_discovery.py -- Strategy discovery engine for the Kalshi trading bot.

Scans ALL high-volume Kalshi markets (not just weather/BTC/sports),
uses Claude to estimate probabilities on uncovered categories, tracks
accuracy over time, and "graduates" categories that prove profitable.

Workflow:
  1. Fetch all open markets via paginated API calls
  2. Categorize each market (weather, crypto, sports, politics, economics, fed_rates, other)
  3. Identify "discovery candidates" -- categories without existing strategies
  4. Run Claude probability estimation on top candidates (cost-controlled)
  5. Log results to output/strategy_lab.json
  6. When markets settle, compare predictions to outcomes
  7. Graduate categories with >10 settled, >55% accuracy, >5c avg edge

CLI:
    python strategy_discovery.py --scan          # List all discovery candidates
    python strategy_discovery.py --analyze       # Claude-analyze top 10 candidates
    python strategy_discovery.py --lab-status    # Show accuracy per category
    python strategy_discovery.py --find-edges    # Markets with >10c edge only

Integration:
    from strategy_discovery import get_graduated_edges
    edges = get_graduated_edges(kalshi_client)  # Returns edges from proven categories
"""

import argparse
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

import config
from kalshi_client import KalshiClient
from claude_agent import ClaudeAgent

console = Console()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STRATEGY_LAB_PATH = config.OUTPUT_DIR / "strategy_lab.json"
CACHE_PATH = config.OUTPUT_DIR / "discovery_cache.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_ANALYSIS_PER_SCAN = 20       # Cap Claude calls per run (cost control)
CACHE_TTL_SECONDS = 3600         # 1 hour -- don't re-analyze the same market
MIN_VOLUME = 50                  # Minimum volume filter
MAX_SETTLE_DAYS = 30             # Only markets settling within 30 days
GRADUATION_MIN_PREDICTIONS = 10  # Need 10+ settled predictions
GRADUATION_MIN_ACCURACY = 0.55   # Need >55% accuracy
GRADUATION_MIN_EDGE = 0.05       # Need >5c average edge

# Categories that already have dedicated strategies
COVERED_CATEGORIES = {"weather", "crypto", "sports"}

# ---------------------------------------------------------------------------
# Market categorization
# ---------------------------------------------------------------------------

CATEGORY_RULES = {
    "weather": [
        "KXHIGH", "KXLOW", "temperature", "weather", "rain", "snow",
        "hurricane", "tornado", "wind", "forecast", "degrees",
    ],
    "crypto": [
        "KXBTC", "KXETH", "bitcoin", "ethereum", "crypto", "BTC", "ETH",
        "solana", "SOL", "dogecoin",
    ],
    "sports": [
        "KXNBA", "KXNHL", "KXMLB", "KXNFL", "KXNCAA", "KXMARMAD",
        "KXNBAGAME", "KXNBAPTS", "KXMLS", "KXSOCCER",
        "NBA", "NHL", "MLB", "NFL", "NCAA", "soccer", "football",
        "basketball", "baseball", "hockey", "Super Bowl", "World Series",
        "March Madness", "game", "team", "player", "points", "touchdowns",
    ],
    "politics": [
        "president", "election", "senate", "congress", "governor",
        "democrat", "republican", "Trump", "Biden", "Harris",
        "vote", "ballot", "poll", "primary", "cabinet", "impeach",
        "nominee", "political", "White House",
    ],
    "economics": [
        "GDP", "unemployment", "jobs", "nonfarm", "payroll", "CPI",
        "inflation", "retail sales", "housing", "consumer", "ISM",
        "PMI", "trade deficit", "economic", "recession",
    ],
    "fed_rates": [
        "Fed", "FOMC", "interest rate", "rate cut", "rate hike",
        "federal reserve", "basis points", "fed funds", "monetary policy",
        "KXFED", "fed rate",
    ],
}


def categorize_market(ticker: str, title: str, series: str) -> str:
    """Categorize a market based on ticker, title, and series."""
    combined = f"{ticker} {title} {series}".lower()

    # Check series/ticker prefixes first (most reliable)
    ticker_upper = ticker.upper()
    series_upper = series.upper()

    for cat, keywords in CATEGORY_RULES.items():
        for kw in keywords:
            if kw.startswith("KX") and (ticker_upper.startswith(kw) or series_upper.startswith(kw)):
                return cat

    # Then check title keywords
    for cat, keywords in CATEGORY_RULES.items():
        for kw in keywords:
            if kw.lower() in combined:
                return cat

    return "other"


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def _load_cache() -> dict:
    """Load analysis cache (avoids re-analyzing the same market within TTL)."""
    if not CACHE_PATH.exists():
        return {}
    try:
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict):
    """Persist analysis cache to disk."""
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def _is_cached(cache: dict, ticker: str) -> bool:
    """Check if a market was analyzed within the cache TTL."""
    entry = cache.get(ticker)
    if not entry:
        return False
    cached_at = datetime.fromisoformat(entry["timestamp"])
    age = (datetime.now(timezone.utc) - cached_at).total_seconds()
    return age < CACHE_TTL_SECONDS


# ---------------------------------------------------------------------------
# Strategy lab file
# ---------------------------------------------------------------------------

def _load_lab() -> list[dict]:
    """Load all strategy lab entries."""
    if not STRATEGY_LAB_PATH.exists():
        return []
    try:
        with open(STRATEGY_LAB_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_lab(entries: list[dict]):
    """Persist strategy lab entries."""
    with open(STRATEGY_LAB_PATH, "w") as f:
        json.dump(entries, f, indent=2)


def _append_lab_entry(entry: dict):
    """Append a single entry to the lab file."""
    entries = _load_lab()
    # Avoid duplicates -- update if same ticker already exists with same timestamp date
    entry_date = entry["timestamp"][:10]
    entries = [
        e for e in entries
        if not (e["ticker"] == entry["ticker"] and e["timestamp"][:10] == entry_date)
    ]
    entries.append(entry)
    _save_lab(entries)


# ---------------------------------------------------------------------------
# Market scanning
# ---------------------------------------------------------------------------

def scan_all_markets(client: KalshiClient) -> list[dict]:
    """
    Fetch ALL open markets from Kalshi with pagination.
    Filters to volume > MIN_VOLUME and settling within MAX_SETTLE_DAYS.
    Returns list of parsed market dicts with category field.
    """
    all_markets = []
    cursor = None
    max_pages = 10  # Up to 1000 markets
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(days=MAX_SETTLE_DAYS)

    console.print("[dim]Fetching all open markets from Kalshi...[/dim]")

    for page in range(max_pages):
        try:
            resp = client.get_markets(
                status="open",
                limit=100,
                cursor=cursor,
            )
        except Exception as e:
            console.print(f"[yellow]API error on page {page + 1}: {e}[/yellow]")
            break

        markets_raw = resp.get("markets", [])
        if not markets_raw:
            break

        for m in markets_raw:
            parsed = _parse_market(m, now, cutoff)
            if parsed:
                all_markets.append(parsed)

        cursor = resp.get("cursor")
        if not cursor:
            break

        time.sleep(1.0)  # Rate limiting

    console.print(f"  Fetched {len(all_markets)} markets across {page + 1} pages")
    return all_markets


def _parse_market(raw: dict, now: datetime, cutoff: datetime) -> Optional[dict]:
    """Parse and filter a single market. Returns None if filtered out."""
    try:
        close_time_str = raw.get("close_time", "")
        if not close_time_str:
            return None
        close_time = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))

        # Filter: must settle before cutoff
        if close_time > cutoff:
            return None

        hours_to_settlement = (close_time - now).total_seconds() / 3600
        if hours_to_settlement < 0.5:
            return None  # About to close

        # Parse prices (API v2 returns dollar strings)
        def to_cents(val):
            if val is None:
                return 0
            try:
                return int(float(val) * 100)
            except (ValueError, TypeError):
                return 0

        yes_price = to_cents(raw.get("yes_bid_dollars") or raw.get("yes_bid") or 0)
        no_price = to_cents(raw.get("no_bid_dollars") or raw.get("no_bid") or 0)
        yes_ask = to_cents(raw.get("yes_ask_dollars") or raw.get("yes_ask") or 0)

        # Volume
        vol_raw = raw.get("volume_fp") or raw.get("volume_24h_fp") or raw.get("volume") or 0
        try:
            volume = int(float(vol_raw))
        except (ValueError, TypeError):
            volume = 0

        # Filter: minimum volume
        if volume < MIN_VOLUME:
            return None

        ticker = raw.get("ticker", "")
        title = raw.get("title", "")
        series = raw.get("series_ticker", "")

        category = categorize_market(ticker, title, series)

        return {
            "ticker": ticker,
            "title": title[:80],
            "series": series,
            "event_ticker": raw.get("event_ticker", ""),
            "category": category,
            "yes_price": yes_price,
            "no_price": no_price,
            "yes_ask": yes_ask,
            "spread": (yes_ask - yes_price) if (yes_ask and yes_price) else 99,
            "volume": volume,
            "hours_to_settlement": round(hours_to_settlement, 1),
            "close_time": close_time.isoformat(),
            "result": raw.get("result", ""),
            "status": raw.get("status", ""),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Discovery identification
# ---------------------------------------------------------------------------

def identify_candidates(markets: list[dict]) -> list[dict]:
    """
    Filter to discovery candidates -- markets in categories
    NOT already covered by existing strategies (weather, crypto, sports).
    Sorted by volume descending.
    """
    candidates = [m for m in markets if m["category"] not in COVERED_CATEGORIES]
    candidates.sort(key=lambda m: m["volume"], reverse=True)
    return candidates


def summarize_by_category(markets: list[dict]) -> dict:
    """Count markets per category."""
    summary = {}
    for m in markets:
        cat = m["category"]
        if cat not in summary:
            summary[cat] = {"count": 0, "total_volume": 0, "is_covered": cat in COVERED_CATEGORIES}
        summary[cat]["count"] += 1
        summary[cat]["total_volume"] += m["volume"]
    return summary


# ---------------------------------------------------------------------------
# Claude analysis on discovery candidates
# ---------------------------------------------------------------------------

def analyze_candidates(
    candidates: list[dict],
    max_analyze: int = MAX_ANALYSIS_PER_SCAN,
) -> list[dict]:
    """
    Run Claude probability estimation on top candidates.
    Respects cache TTL and API cost limits.
    Returns list of analysis results.
    """
    if not config.ANTHROPIC_API_KEY:
        console.print("[yellow]ANTHROPIC_API_KEY not set. Skipping Claude analysis.[/yellow]")
        console.print("[dim]Set the key to enable probability estimation.[/dim]")
        return []

    agent = ClaudeAgent()
    cache = _load_cache()
    results = []
    analyzed = 0

    for market in candidates:
        if analyzed >= max_analyze:
            console.print(f"[dim]Reached analysis limit ({max_analyze}). Stopping.[/dim]")
            break

        ticker = market["ticker"]

        # Check cache
        if _is_cached(cache, ticker):
            cached_result = cache[ticker]
            results.append(cached_result)
            continue

        # Skip markets with no meaningful price
        if market["yes_price"] <= 0 and market["yes_ask"] <= 0:
            continue

        # Use best available price
        price = market["yes_price"] if market["yes_price"] > 0 else market["yes_ask"]

        console.print(f"  Analyzing: [cyan]{ticker}[/cyan] - {market['title']}")

        estimate = agent.estimate_probability(
            market_title=market["title"],
            yes_price=price,
            no_price=market["no_price"],
            volume=market["volume"],
            hours_to_settlement=market["hours_to_settlement"],
            additional_context=f"Category: {market['category']}. "
                              f"Spread: {market['spread']}c. "
                              f"Settling in {market['hours_to_settlement']:.0f} hours.",
        )

        if estimate:
            market_price_decimal = price / 100.0
            claude_prob = estimate["estimated_probability"]
            edge = abs(claude_prob - market_price_decimal)

            result = {
                "ticker": ticker,
                "title": market["title"],
                "category": market["category"],
                "claude_estimate": claude_prob,
                "market_price": price,
                "market_price_decimal": market_price_decimal,
                "edge": round(edge, 4),
                "edge_cents": round(edge * 100, 1),
                "recommended_side": estimate.get("recommended_side", "yes"),
                "confidence": estimate.get("confidence", "low"),
                "reasoning": estimate.get("reasoning", ""),
                "volume": market["volume"],
                "hours_to_settlement": market["hours_to_settlement"],
                "close_time": market["close_time"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "settled": False,
                "actual_result": None,
                "prediction_correct": None,
            }

            results.append(result)

            # Update cache
            cache[ticker] = result
            _save_cache(cache)

            # Save to lab
            _append_lab_entry(result)

            analyzed += 1
            time.sleep(0.5)  # Rate limit Claude calls

    console.print(f"  Analyzed {analyzed} new markets ({len(results)} total with cache)")
    return results


# ---------------------------------------------------------------------------
# Settlement tracking and accuracy
# ---------------------------------------------------------------------------

def update_settlements(client: KalshiClient):
    """
    Check lab entries for settled markets and update actual outcomes.
    Compares Claude's prediction to the real result.
    """
    entries = _load_lab()
    if not entries:
        console.print("[dim]No lab entries to check.[/dim]")
        return

    updated = 0
    for entry in entries:
        if entry.get("settled"):
            continue  # Already resolved

        ticker = entry["ticker"]
        try:
            market_data = client.get_market(ticker)
            market = market_data.get("market", market_data)
            result = market.get("result", "")
            status = market.get("status", "")

            if status in ("settled", "closed") and result in ("yes", "no"):
                entry["settled"] = True
                entry["actual_result"] = result

                # Did Claude predict correctly?
                claude_prob = entry["claude_estimate"]
                predicted_yes = claude_prob >= 0.5
                actual_yes = result == "yes"
                entry["prediction_correct"] = predicted_yes == actual_yes
                updated += 1

        except Exception:
            continue  # Market may not exist anymore

        time.sleep(0.3)  # Rate limit

    if updated > 0:
        _save_lab(entries)
        console.print(f"  Updated {updated} settled markets")
    else:
        console.print("  [dim]No new settlements found[/dim]")


def calculate_accuracy_by_category() -> dict:
    """
    Calculate prediction accuracy per category from lab data.
    Returns dict keyed by category with stats.
    """
    entries = _load_lab()
    categories = {}

    for entry in entries:
        cat = entry.get("category", "other")
        if cat not in categories:
            categories[cat] = {
                "total_predictions": 0,
                "settled": 0,
                "correct": 0,
                "incorrect": 0,
                "pending": 0,
                "total_edge": 0.0,
                "edges": [],
                "accuracy": 0.0,
                "avg_edge": 0.0,
                "graduated": False,
            }

        stats = categories[cat]
        stats["total_predictions"] += 1
        stats["total_edge"] += entry.get("edge", 0)
        stats["edges"].append(entry.get("edge", 0))

        if entry.get("settled"):
            stats["settled"] += 1
            if entry.get("prediction_correct"):
                stats["correct"] += 1
            else:
                stats["incorrect"] += 1
        else:
            stats["pending"] += 1

    # Calculate derived metrics and graduation status
    for cat, stats in categories.items():
        if stats["settled"] > 0:
            stats["accuracy"] = round(stats["correct"] / stats["settled"], 4)
        stats["avg_edge"] = round(
            stats["total_edge"] / stats["total_predictions"], 4
        ) if stats["total_predictions"] > 0 else 0.0

        # Graduation check
        stats["graduated"] = (
            stats["settled"] >= GRADUATION_MIN_PREDICTIONS
            and stats["accuracy"] >= GRADUATION_MIN_ACCURACY
            and stats["avg_edge"] >= GRADUATION_MIN_EDGE
        )

        # Clean up -- don't persist the raw edges list
        del stats["edges"]

    return categories


def get_graduated_categories() -> list[str]:
    """Return list of category names that have graduated."""
    accuracy = calculate_accuracy_by_category()
    return [cat for cat, stats in accuracy.items() if stats["graduated"]]


# ---------------------------------------------------------------------------
# Integration function for auto_trade.py
# ---------------------------------------------------------------------------

def get_graduated_edges(kalshi_client: KalshiClient) -> list[dict]:
    """
    Main integration point for auto_trade.py.
    Scans markets, runs Claude analysis on graduated categories only,
    returns list of edge dicts ready for trading.

    Each returned dict has:
        ticker, side, action, estimated_prob, market_price, edge, reason, category

    Uses same risk parameters as other strategies:
        - $2/trade max
        - Conservative sizing
        - Only trades graduated (proven) categories
    """
    graduated = get_graduated_categories()
    if not graduated:
        return []

    # Scan all markets
    markets = scan_all_markets(kalshi_client)

    # Filter to graduated categories only
    grad_markets = [m for m in markets if m["category"] in graduated]
    if not grad_markets:
        return []

    # Sort by volume, take top candidates
    grad_markets.sort(key=lambda m: m["volume"], reverse=True)
    grad_markets = grad_markets[:MAX_ANALYSIS_PER_SCAN]

    # Analyze with Claude
    results = analyze_candidates(grad_markets, max_analyze=MAX_ANALYSIS_PER_SCAN)

    # Filter to tradeable edges (>5c edge, medium+ confidence)
    tradeable = []
    for r in results:
        edge_cents = r.get("edge_cents", 0)
        confidence = r.get("confidence", "low")

        if edge_cents < 5:
            continue
        if confidence == "low":
            continue

        claude_prob = r["claude_estimate"]
        market_decimal = r["market_price_decimal"]

        # Determine trade side
        if claude_prob > market_decimal:
            side = "yes"
            action = "buy"
        else:
            side = "no"
            action = "buy"

        tradeable.append({
            "ticker": r["ticker"],
            "title": r["title"],
            "category": r["category"],
            "side": side,
            "action": action,
            "estimated_prob": claude_prob,
            "market_price": r["market_price"],
            "edge": r["edge"],
            "edge_cents": edge_cents,
            "confidence": confidence,
            "reason": f"[graduated:{r['category']}] {r.get('reasoning', '')}",
        })

    return tradeable


# ---------------------------------------------------------------------------
# CLI display helpers
# ---------------------------------------------------------------------------

def display_scan_results(markets: list[dict]):
    """Show categorized market scan results."""
    summary = summarize_by_category(markets)
    candidates = identify_candidates(markets)

    # Category summary table
    cat_table = Table(title="Market Categories", show_lines=False, padding=(0, 1))
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Markets", justify="right", style="white")
    cat_table.add_column("Volume", justify="right", style="yellow")
    cat_table.add_column("Status", style="white")

    for cat, stats in sorted(summary.items(), key=lambda x: x[1]["total_volume"], reverse=True):
        status = "[green]COVERED[/green]" if stats["is_covered"] else "[magenta]DISCOVERY[/magenta]"
        cat_table.add_row(
            cat,
            str(stats["count"]),
            f"{stats['total_volume']:,}",
            status,
        )

    console.print(cat_table)

    # Discovery candidates table
    if candidates:
        disc_table = Table(
            title=f"\nDiscovery Candidates ({len(candidates)} markets)",
            show_lines=False, padding=(0, 1),
        )
        disc_table.add_column("Ticker", style="cyan", max_width=30)
        disc_table.add_column("Title", style="white", max_width=50)
        disc_table.add_column("Category", style="magenta")
        disc_table.add_column("Yes", justify="right", style="green")
        disc_table.add_column("Vol", justify="right", style="yellow")
        disc_table.add_column("Hrs", justify="right", style="dim")

        for m in candidates[:30]:
            disc_table.add_row(
                m["ticker"][:30],
                m["title"][:50],
                m["category"],
                f"{m['yes_price']}c",
                f"{m['volume']:,}",
                f"{m['hours_to_settlement']:.0f}",
            )

        console.print(disc_table)
    else:
        console.print("[dim]No discovery candidates found (all markets are in covered categories).[/dim]")


def display_analysis_results(results: list[dict]):
    """Show Claude analysis results with edges."""
    if not results:
        console.print("[dim]No analysis results to display.[/dim]")
        return

    table = Table(title="Claude Analysis Results", show_lines=False, padding=(0, 1))
    table.add_column("Ticker", style="cyan", max_width=25)
    table.add_column("Category", style="magenta")
    table.add_column("Mkt", justify="right", style="white")
    table.add_column("Claude", justify="right", style="green")
    table.add_column("Edge", justify="right")
    table.add_column("Conf", style="white")
    table.add_column("Side", style="white")
    table.add_column("Reasoning", style="dim", max_width=40)

    for r in sorted(results, key=lambda x: x.get("edge", 0), reverse=True):
        edge_cents = r.get("edge_cents", 0)
        edge_color = "green" if edge_cents >= 10 else "yellow" if edge_cents >= 5 else "dim"
        table.add_row(
            r["ticker"][:25],
            r["category"],
            f"{r['market_price']}c",
            f"{r['claude_estimate']:.0%}",
            f"[{edge_color}]{edge_cents:.1f}c[/{edge_color}]",
            r.get("confidence", "?"),
            r.get("recommended_side", "?"),
            (r.get("reasoning", "") or "")[:40],
        )

    console.print(table)


def display_lab_status():
    """Show accuracy per category with graduation status."""
    accuracy = calculate_accuracy_by_category()

    if not accuracy:
        console.print("[dim]No lab data yet. Run --analyze first.[/dim]")
        return

    table = Table(title="Strategy Lab -- Category Accuracy", show_lines=True, padding=(0, 1))
    table.add_column("Category", style="cyan")
    table.add_column("Total", justify="right", style="white")
    table.add_column("Settled", justify="right", style="white")
    table.add_column("Correct", justify="right", style="green")
    table.add_column("Accuracy", justify="right")
    table.add_column("Avg Edge", justify="right", style="yellow")
    table.add_column("Pending", justify="right", style="dim")
    table.add_column("Status")

    for cat, stats in sorted(accuracy.items(), key=lambda x: x[1]["accuracy"], reverse=True):
        acc = stats["accuracy"]
        acc_color = "green" if acc >= 0.60 else "yellow" if acc >= 0.50 else "red"

        if stats["graduated"]:
            status = "[bold green]GRADUATED[/bold green]"
        elif stats["settled"] >= GRADUATION_MIN_PREDICTIONS:
            if acc >= GRADUATION_MIN_ACCURACY:
                status = "[yellow]EDGE TOO LOW[/yellow]"
            else:
                status = "[red]INACCURATE[/red]"
        else:
            remaining = GRADUATION_MIN_PREDICTIONS - stats["settled"]
            status = f"[dim]need {remaining} more[/dim]"

        table.add_row(
            cat,
            str(stats["total_predictions"]),
            str(stats["settled"]),
            str(stats["correct"]),
            f"[{acc_color}]{acc:.0%}[/{acc_color}]" if stats["settled"] > 0 else "-",
            f"{stats['avg_edge'] * 100:.1f}c",
            str(stats["pending"]),
            status,
        )

    console.print(table)

    graduated = [cat for cat, s in accuracy.items() if s["graduated"]]
    if graduated:
        console.print(
            Panel(
                f"[bold green]Graduated categories: {', '.join(graduated)}[/bold green]\n"
                "These will be auto-traded when integrated with auto_trade.py",
                title="Graduation Status",
                border_style="green",
            )
        )
    else:
        console.print(
            "\n[dim]No categories have graduated yet. "
            f"Need: {GRADUATION_MIN_PREDICTIONS}+ settled, "
            f"{GRADUATION_MIN_ACCURACY:.0%}+ accuracy, "
            f"{GRADUATION_MIN_EDGE * 100:.0f}c+ avg edge[/dim]"
        )


def display_edges(results: list[dict], min_edge_cents: float = 10.0):
    """Show only markets with significant edges."""
    big_edges = [r for r in results if r.get("edge_cents", 0) >= min_edge_cents]

    if not big_edges:
        console.print(f"[dim]No markets with >{min_edge_cents:.0f}c edge found.[/dim]")
        return

    big_edges.sort(key=lambda x: x["edge_cents"], reverse=True)

    table = Table(
        title=f"Markets with >{min_edge_cents:.0f}c Edge ({len(big_edges)} found)",
        show_lines=False, padding=(0, 1),
    )
    table.add_column("Ticker", style="cyan", max_width=25)
    table.add_column("Title", style="white", max_width=40)
    table.add_column("Category", style="magenta")
    table.add_column("Mkt", justify="right")
    table.add_column("Claude", justify="right", style="green")
    table.add_column("Edge", justify="right", style="bold green")
    table.add_column("Side", style="white")
    table.add_column("Vol", justify="right", style="yellow")

    for r in big_edges:
        table.add_row(
            r["ticker"][:25],
            (r.get("title", "") or "")[:40],
            r["category"],
            f"{r['market_price']}c",
            f"{r['claude_estimate']:.0%}",
            f"{r['edge_cents']:.1f}c",
            r.get("recommended_side", "?"),
            f"{r.get('volume', 0):,}",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Strategy Discovery Engine for Kalshi Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python strategy_discovery.py --scan          Scan and categorize all markets
  python strategy_discovery.py --analyze       Claude-analyze top 10 candidates
  python strategy_discovery.py --analyze -n 5  Analyze top 5 candidates
  python strategy_discovery.py --lab-status    Show accuracy per category
  python strategy_discovery.py --find-edges    Show markets with >10c edge
  python strategy_discovery.py --settle        Check for settled markets
        """,
    )
    parser.add_argument("--scan", action="store_true", help="Scan and show all discovery candidates")
    parser.add_argument("--analyze", action="store_true", help="Run Claude on top N candidates")
    parser.add_argument("--lab-status", action="store_true", help="Show accuracy per category")
    parser.add_argument("--find-edges", action="store_true", help="Show markets with >10c edge")
    parser.add_argument("--settle", action="store_true", help="Check for settled markets and update lab")
    parser.add_argument("-n", "--num", type=int, default=10, help="Number of candidates to analyze (default: 10)")

    args = parser.parse_args()

    # Default to --scan if no action specified
    if not any([args.scan, args.analyze, args.lab_status, args.find_edges, args.settle]):
        parser.print_help()
        return

    # Actions that don't need the API client
    if args.lab_status:
        console.print(Panel("[bold]Strategy Lab Status[/bold]", border_style="cyan"))
        display_lab_status()
        return

    # Actions that need the Kalshi client
    try:
        client = KalshiClient()
    except FileNotFoundError:
        console.print("[red]Private key not found. Check KALSHI_PRIVATE_KEY_PATH in .env[/red]")
        return
    except Exception as e:
        console.print(f"[red]Failed to initialize Kalshi client: {e}[/red]")
        return

    if args.settle:
        console.print(Panel("[bold]Checking Settlements[/bold]", border_style="cyan"))
        update_settlements(client)
        display_lab_status()
        return

    if args.scan or args.analyze or args.find_edges:
        console.print(Panel("[bold]Strategy Discovery Engine[/bold]", border_style="cyan"))

        # Scan all markets
        markets = scan_all_markets(client)
        if not markets:
            console.print("[red]No markets found. Check API connection.[/red]")
            return

        if args.scan:
            display_scan_results(markets)

        if args.analyze or args.find_edges:
            candidates = identify_candidates(markets)
            if not candidates:
                console.print("[dim]No discovery candidates found.[/dim]")
                return

            max_analyze = min(args.num, MAX_ANALYSIS_PER_SCAN)
            console.print(f"\n[cyan]Analyzing top {max_analyze} candidates with Claude...[/cyan]")
            results = analyze_candidates(candidates[:max_analyze], max_analyze=max_analyze)

            if args.analyze:
                display_analysis_results(results)

            if args.find_edges:
                display_edges(results, min_edge_cents=10.0)


if __name__ == "__main__":
    main()
