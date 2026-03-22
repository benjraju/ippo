"""
polytope_scanner.py -- Marginal Polytope Arbitrage Scanner for Kalshi

Applies the mathematical framework from "Unravelling the Probabilistic Forest"
to Kalshi prediction markets. Specifically targets weather bucket markets where
logical dependencies create exploitable mispricing.

Three types of arbitrage detected:

1. BUCKET SUM ARB: All buckets for one event must sum to ~$1.00
   If sum < $1.00 → buy all buckets (guaranteed profit)
   If sum > $1.00 → sell all / buy all NOs (guaranteed profit)

2. THRESHOLD-BUCKET DEPENDENCY: "High > 70F" must equal sum of all buckets above 70
   If they disagree → trade the difference

3. CROSS-EVENT ARB: Related events (same city, adjacent days) with logical constraints

Usage:
    python polytope_scanner.py                # Full scan
    python polytope_scanner.py --series KXHIGHNY  # Single city
    python polytope_scanner.py --deep         # Deep scan with dependency analysis
"""

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

import config
from kalshi_client import KalshiClient

console = Console()

WEATHER_SERIES = ["KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHLA", "KXHIGHDC", "KXHIGHDEN"]

CITY_NAMES = {
    "KXHIGHNY": "NYC",
    "KXHIGHCHI": "Chicago",
    "KXHIGHMIA": "Miami",
    "KXHIGHLA": "LA",
    "KXHIGHDC": "DC",
    "KXHIGHDEN": "Denver",
}

# Kalshi fee per contract (round-trip, in dollars), from central config
from config import KALSHI_FEE_PER_CONTRACT_DOLLARS as KALSHI_FEE_PER_CONTRACT


@dataclass
class MarketData:
    """Parsed Kalshi market data."""
    ticker: str
    title: str
    series: str
    event_ticker: str
    yes_bid: float  # cents
    yes_ask: float  # cents
    no_bid: float   # cents
    no_ask: float   # cents
    volume: int
    open_interest: int
    status: str
    # Parsed weather fields
    market_type: str = ""  # "bucket", "above", "below"
    bucket_low: float = 0.0
    bucket_high: float = 0.0
    threshold: float = 0.0


@dataclass
class ArbOpportunity:
    """A detected arbitrage opportunity."""
    arb_type: str  # "bucket_sum", "threshold_dependency", "cross_event"
    description: str
    markets_involved: list
    total_cost: float  # dollars to execute
    guaranteed_payout: float  # dollars guaranteed
    gross_profit: float  # before fees
    net_profit: float  # after fees
    num_contracts: int
    confidence: str  # "high", "medium", "low"
    details: dict = field(default_factory=dict)


def parse_weather_market(title: str) -> dict:
    """
    Parse a Kalshi weather market title to extract type and bounds.

    Examples:
        "High temp in NYC be 72° to 73°?" -> bucket, low=72, high=73
        "High temp in NYC be above 70°?"   -> above, threshold=70
        "High temp in NYC be 67° or less?" -> below, threshold=67
        "High temp in NYC be 68° or more?" -> above, threshold=68
    """
    # Strip markdown bold markers
    clean = title.replace("**", "")
    clean_lower = clean.lower()

    # Bucket: "60-61°" or "60° to 61°" or "be 60-61"
    bucket_match = re.search(r'(\d+)°?\s*(?:to|[-–])\s*(\d+)°?', clean)
    if bucket_match:
        low = float(bucket_match.group(1))
        high = float(bucket_match.group(2))
        return {"type": "bucket", "low": low, "high": high, "threshold": 0}

    # Above: ">61°" or "above 61" or "or more" or "at least"
    # Kalshi format: "be >61°" or "be above 61°"
    above_match = re.search(r'(?:>|above\s*|at least\s*|or more.*?)(\d+)', clean_lower)
    if above_match and '<' not in clean_lower.split(above_match.group(0))[0][-5:]:
        threshold = float(above_match.group(1))
        return {"type": "above", "low": threshold, "high": 999, "threshold": threshold}

    # Below: "<54°" or "below 54" or "or less"
    # Kalshi format: "be <54°"
    below_match = re.search(r'(?:<|below\s*|under\s*)(\d+)', clean_lower)
    if not below_match:
        below_match = re.search(r'(\d+)°?\s*(?:or less|or below|or under)', clean_lower)
    if below_match:
        threshold = float(below_match.group(1))
        return {"type": "below", "low": 0, "high": threshold, "threshold": threshold}

    return {"type": "unknown", "low": 0, "high": 0, "threshold": 0}


def fetch_markets_for_series(client: KalshiClient, series: str) -> list[MarketData]:
    """Fetch all active markets for a weather series."""
    markets = []
    cursor = None

    while True:
        params = {
            "series_ticker": series,
            "status": "open",
            "limit": 200,
        }
        if cursor:
            params["cursor"] = cursor

        try:
            resp = client.get_markets(**params)
        except Exception as e:
            console.print(f"[yellow]API error for {series}: {e}[/yellow]")
            break

        for m in resp.get("markets", []):
            # Kalshi API v2 uses dollar fields: yes_bid_dollars, yes_ask_dollars, etc.
            yes_bid = float(m.get("yes_bid_dollars", 0) or 0) * 100  # convert to cents
            yes_ask = float(m.get("yes_ask_dollars", 0) or 0) * 100
            no_bid = float(m.get("no_bid_dollars", 0) or 0) * 100
            no_ask = float(m.get("no_ask_dollars", 0) or 0) * 100

            parsed = parse_weather_market(m.get("title", ""))

            md = MarketData(
                ticker=m.get("ticker", ""),
                title=m.get("title", ""),
                series=series,
                event_ticker=m.get("event_ticker", ""),
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                no_bid=no_bid,
                no_ask=no_ask,
                volume=int(float(m.get("volume_fp", 0) or 0)),
                open_interest=int(float(m.get("open_interest_fp", 0) or 0)),
                status=m.get("status", ""),
                market_type=parsed["type"],
                bucket_low=parsed["low"],
                bucket_high=parsed["high"],
                threshold=parsed["threshold"],
            )
            markets.append(md)

        cursor = resp.get("cursor")
        if not cursor:
            break

    return markets


def scan_bucket_sum_arb(markets: list[MarketData]) -> list[ArbOpportunity]:
    """
    Type 1: Bucket Sum Arbitrage

    For each event, all bucket markets must sum to ~$1.00.
    If the sum of YES ask prices < $1.00 → buy all buckets.
    If the sum of YES bid prices > $1.00 → sell all (buy all NOs).

    This is the marginal polytope constraint: the convex hull of valid
    outcomes requires exactly one bucket to be TRUE.
    """
    opps = []

    # Group bucket markets by event
    events = {}
    for m in markets:
        if m.market_type != "bucket":
            continue
        if m.event_ticker not in events:
            events[m.event_ticker] = []
        events[m.event_ticker].append(m)

    for event_ticker, buckets in events.items():
        if len(buckets) < 3:
            continue  # need multiple buckets for meaningful check

        # Sort by bucket_low for display
        buckets.sort(key=lambda b: b.bucket_low)

        # Strategy 1: Buy all YES (if sum of asks < 100)
        sum_yes_ask = sum(b.yes_ask for b in buckets)
        # Strategy 2: Buy all NO (if sum of no_asks < (n-1)*100 ... actually simpler:
        #   if sum of yes_bids > 100, sell all YES / buy all NO)
        sum_yes_bid = sum(b.yes_bid for b in buckets)

        # Calculate midpoints for display
        sum_mid = sum((b.yes_bid + b.yes_ask) / 2 for b in buckets if b.yes_bid > 0 and b.yes_ask > 0)

        n = len(buckets)
        city = CITY_NAMES.get(buckets[0].series, "Unknown")

        # --- Buy-all-YES arbitrage ---
        if sum_yes_ask > 0 and sum_yes_ask < 100:
            cost_per_set = sum_yes_ask / 100.0  # dollars to buy 1 of each
            payout = 1.00  # guaranteed $1 (exactly one bucket settles YES)
            fees = n * KALSHI_FEE_PER_CONTRACT  # fee per contract bought
            gross_profit = payout - cost_per_set
            net_profit = gross_profit - fees

            if net_profit > 0:
                opps.append(ArbOpportunity(
                    arb_type="bucket_sum_buy",
                    description=f"{city}: {n} buckets sum to {sum_yes_ask:.1f}c (ask) < 100c. "
                                f"Buy all for ${cost_per_set:.2f}, guaranteed $1.00 payout.",
                    markets_involved=[b.ticker for b in buckets],
                    total_cost=cost_per_set,
                    guaranteed_payout=payout,
                    gross_profit=gross_profit,
                    net_profit=net_profit,
                    num_contracts=n,
                    confidence="high" if net_profit > 0.10 else "medium",
                    details={
                        "event": event_ticker,
                        "city": city,
                        "n_buckets": n,
                        "sum_ask_cents": round(sum_yes_ask, 1),
                        "sum_bid_cents": round(sum_yes_bid, 1),
                        "sum_mid_cents": round(sum_mid, 1),
                        "buckets": [
                            {"ticker": b.ticker, "range": f"{b.bucket_low}-{b.bucket_high}",
                             "bid": b.yes_bid, "ask": b.yes_ask, "vol": b.volume}
                            for b in buckets
                        ],
                    },
                ))
            elif gross_profit > 0:
                # Profitable before fees but not after
                opps.append(ArbOpportunity(
                    arb_type="bucket_sum_buy_marginal",
                    description=f"{city}: {n} buckets sum to {sum_yes_ask:.1f}c (ask). "
                                f"Gross profit ${gross_profit:.3f} but fees eat it (${fees:.2f}).",
                    markets_involved=[b.ticker for b in buckets],
                    total_cost=cost_per_set,
                    guaranteed_payout=payout,
                    gross_profit=gross_profit,
                    net_profit=net_profit,
                    num_contracts=n,
                    confidence="low",
                    details={
                        "event": event_ticker,
                        "sum_ask_cents": round(sum_yes_ask, 1),
                        "fees": round(fees, 3),
                    },
                ))

        # --- Sell-all-YES (buy all NO) arbitrage ---
        if sum_yes_bid > 100:
            # Sell all YES at bid = buy all NO
            # Cost = sum of (100 - yes_bid) for each / 100 in dollars... no.
            # Actually: sell YES at bid price. Revenue = sum(yes_bid)/100.
            # Payout obligation: exactly one settles YES, pay $1.
            # Net = revenue - $1.00
            revenue_per_set = sum_yes_bid / 100.0
            payout_obligation = 1.00
            fees = n * KALSHI_FEE_PER_CONTRACT
            gross_profit = revenue_per_set - payout_obligation
            net_profit = gross_profit - fees

            if net_profit > 0:
                opps.append(ArbOpportunity(
                    arb_type="bucket_sum_sell",
                    description=f"{city}: {n} buckets sum to {sum_yes_bid:.1f}c (bid) > 100c. "
                                f"Sell all YES for ${revenue_per_set:.2f}, pay $1.00 settlement.",
                    markets_involved=[b.ticker for b in buckets],
                    total_cost=payout_obligation,
                    guaranteed_payout=revenue_per_set,
                    gross_profit=gross_profit,
                    net_profit=net_profit,
                    num_contracts=n,
                    confidence="high" if net_profit > 0.10 else "medium",
                    details={
                        "event": event_ticker,
                        "sum_bid_cents": round(sum_yes_bid, 1),
                    },
                ))

    return opps


def scan_threshold_dependency_arb(markets: list[MarketData]) -> list[ArbOpportunity]:
    """
    Type 2: Threshold-Bucket Dependency Arbitrage

    "High > 70F" must equal the sum of all buckets with low >= 70.
    If the threshold market price differs from the bucket sum, there's
    a dependency arbitrage.

    This is the marginal polytope insight: these markets share outcomes,
    creating constraints that prices must satisfy.
    """
    opps = []

    # Group by event
    events = {}
    for m in markets:
        if m.market_type in ("bucket", "above", "below"):
            if m.event_ticker not in events:
                events[m.event_ticker] = {"buckets": [], "above": [], "below": []}
            events[m.event_ticker][m.market_type if m.market_type != "bucket" else "buckets"].append(m)

    for event_ticker, grouped in events.items():
        buckets = sorted(grouped["buckets"], key=lambda b: b.bucket_low)
        above_markets = grouped["above"]
        below_markets = grouped["below"]

        if not buckets:
            continue

        # Check each "above" threshold against bucket sum
        for above in above_markets:
            threshold = above.threshold
            # Sum of all bucket YES prices where bucket_low >= threshold
            relevant_buckets = [b for b in buckets if b.bucket_low >= threshold]
            if not relevant_buckets:
                continue

            bucket_sum_mid = sum(
                (b.yes_bid + b.yes_ask) / 2
                for b in relevant_buckets
                if b.yes_bid > 0 and b.yes_ask > 0
            )
            above_mid = (above.yes_bid + above.yes_ask) / 2 if above.yes_bid > 0 and above.yes_ask > 0 else 0

            if above_mid <= 0 or bucket_sum_mid <= 0:
                continue

            discrepancy = abs(above_mid - bucket_sum_mid)

            if discrepancy > 5:  # 5 cent discrepancy threshold
                city = CITY_NAMES.get(above.series, "Unknown")
                direction = "above overpriced" if above_mid > bucket_sum_mid else "buckets overpriced"

                opps.append(ArbOpportunity(
                    arb_type="threshold_dependency",
                    description=f"{city}: 'High >{threshold}F' priced at {above_mid:.0f}c "
                                f"but bucket sum = {bucket_sum_mid:.0f}c. "
                                f"Discrepancy: {discrepancy:.1f}c ({direction}).",
                    markets_involved=[above.ticker] + [b.ticker for b in relevant_buckets],
                    total_cost=0,  # depends on which side you take
                    guaranteed_payout=0,
                    gross_profit=discrepancy / 100.0,
                    net_profit=(discrepancy / 100.0) - (len(relevant_buckets) + 1) * KALSHI_FEE_PER_CONTRACT,
                    num_contracts=len(relevant_buckets) + 1,
                    confidence="medium" if discrepancy > 8 else "low",
                    details={
                        "event": event_ticker,
                        "threshold": threshold,
                        "above_mid": round(above_mid, 1),
                        "bucket_sum_mid": round(bucket_sum_mid, 1),
                        "discrepancy_cents": round(discrepancy, 1),
                        "direction": direction,
                        "n_buckets": len(relevant_buckets),
                    },
                ))

        # Same logic for "below" thresholds
        for below in below_markets:
            threshold = below.threshold
            relevant_buckets = [b for b in buckets if b.bucket_high <= threshold]
            if not relevant_buckets:
                continue

            bucket_sum_mid = sum(
                (b.yes_bid + b.yes_ask) / 2
                for b in relevant_buckets
                if b.yes_bid > 0 and b.yes_ask > 0
            )
            below_mid = (below.yes_bid + below.yes_ask) / 2 if below.yes_bid > 0 and below.yes_ask > 0 else 0

            if below_mid <= 0 or bucket_sum_mid <= 0:
                continue

            discrepancy = abs(below_mid - bucket_sum_mid)

            if discrepancy > 5:
                city = CITY_NAMES.get(below.series, "Unknown")
                direction = "below overpriced" if below_mid > bucket_sum_mid else "buckets overpriced"

                opps.append(ArbOpportunity(
                    arb_type="threshold_dependency",
                    description=f"{city}: 'High <{threshold}F' priced at {below_mid:.0f}c "
                                f"but bucket sum = {bucket_sum_mid:.0f}c. "
                                f"Discrepancy: {discrepancy:.1f}c ({direction}).",
                    markets_involved=[below.ticker] + [b.ticker for b in relevant_buckets],
                    total_cost=0,
                    guaranteed_payout=0,
                    gross_profit=discrepancy / 100.0,
                    net_profit=(discrepancy / 100.0) - (len(relevant_buckets) + 1) * KALSHI_FEE_PER_CONTRACT,
                    num_contracts=len(relevant_buckets) + 1,
                    confidence="medium" if discrepancy > 8 else "low",
                    details={
                        "event": event_ticker,
                        "threshold": threshold,
                        "below_mid": round(below_mid, 1),
                        "bucket_sum_mid": round(bucket_sum_mid, 1),
                        "discrepancy_cents": round(discrepancy, 1),
                        "direction": direction,
                    },
                ))

    return opps


def display_market_summary(all_markets: dict[str, list[MarketData]]):
    """Show what markets we found."""
    table = Table(title="Market Summary", show_lines=True)
    table.add_column("City", style="cyan")
    table.add_column("Events", justify="right")
    table.add_column("Buckets", justify="right")
    table.add_column("Above", justify="right")
    table.add_column("Below", justify="right")
    table.add_column("Total", justify="right", style="bold")

    for series in WEATHER_SERIES:
        markets = all_markets.get(series, [])
        city = CITY_NAMES.get(series, series)
        events = len(set(m.event_ticker for m in markets))
        buckets = len([m for m in markets if m.market_type == "bucket"])
        above = len([m for m in markets if m.market_type == "above"])
        below = len([m for m in markets if m.market_type == "below"])
        table.add_row(city, str(events), str(buckets), str(above), str(below), str(len(markets)))

    console.print(table)


def display_bucket_breakdown(all_markets: dict[str, list[MarketData]]):
    """Show bucket sum analysis per event."""
    table = Table(title="Bucket Sum Analysis (Marginal Polytope Check)", show_lines=True)
    table.add_column("Event", style="cyan", width=30)
    table.add_column("City", width=8)
    table.add_column("Buckets", justify="right")
    table.add_column("Sum(Ask)", justify="right")
    table.add_column("Sum(Bid)", justify="right")
    table.add_column("Sum(Mid)", justify="right")
    table.add_column("Status", justify="center")

    events = {}
    for series, markets in all_markets.items():
        for m in markets:
            if m.market_type == "bucket":
                key = m.event_ticker
                if key not in events:
                    events[key] = {"city": CITY_NAMES.get(series, series), "buckets": []}
                events[key]["buckets"].append(m)

    for event_ticker, data in sorted(events.items()):
        buckets = data["buckets"]
        city = data["city"]
        n = len(buckets)

        sum_ask = sum(b.yes_ask for b in buckets if b.yes_ask > 0)
        sum_bid = sum(b.yes_bid for b in buckets if b.yes_bid > 0)
        sum_mid = sum((b.yes_bid + b.yes_ask) / 2 for b in buckets if b.yes_bid > 0 and b.yes_ask > 0)

        if sum_ask == 0 and sum_bid == 0:
            status = "[dim]no quotes[/dim]"
        elif sum_ask < 100:
            profit = (100 - sum_ask) / 100
            fees = n * KALSHI_FEE_PER_CONTRACT
            net = profit - fees
            if net > 0:
                status = f"[bold green]BUY ARB +${net:.3f}[/bold green]"
            elif profit > 0:
                status = f"[yellow]edge but fees eat it[/yellow]"
            else:
                status = "[dim]OK[/dim]"
        elif sum_bid > 100:
            profit = (sum_bid - 100) / 100
            fees = n * KALSHI_FEE_PER_CONTRACT
            net = profit - fees
            if net > 0:
                status = f"[bold green]SELL ARB +${net:.3f}[/bold green]"
            else:
                status = "[dim]OK[/dim]"
        else:
            status = "[dim]OK[/dim]"

        table.add_row(
            event_ticker[:30],
            city,
            str(n),
            f"{sum_ask:.1f}c" if sum_ask > 0 else "-",
            f"{sum_bid:.1f}c" if sum_bid > 0 else "-",
            f"{sum_mid:.1f}c" if sum_mid > 0 else "-",
            status,
        )

    console.print(table)


def display_opportunities(opps: list[ArbOpportunity]):
    """Display found arbitrage opportunities."""
    if not opps:
        console.print(Panel(
            "[dim]No arbitrage opportunities found above fee threshold.\n"
            "This means Kalshi's weather markets are currently well-priced.[/dim]",
            title="Scan Result",
            border_style="yellow",
        ))
        return

    profitable = [o for o in opps if o.net_profit > 0]
    marginal = [o for o in opps if o.gross_profit > 0 and o.net_profit <= 0]

    if profitable:
        table = Table(title=f"PROFITABLE ARBITRAGE ({len(profitable)} found)", show_lines=True)
        table.add_column("Type", style="cyan", width=20)
        table.add_column("Description", width=60)
        table.add_column("Gross", justify="right", style="green")
        table.add_column("Fees", justify="right", style="red")
        table.add_column("Net", justify="right", style="bold green")
        table.add_column("Conf", justify="center")

        for o in sorted(profitable, key=lambda x: x.net_profit, reverse=True):
            fees = o.gross_profit - o.net_profit
            table.add_row(
                o.arb_type,
                o.description,
                f"${o.gross_profit:.3f}",
                f"${fees:.3f}",
                f"${o.net_profit:.3f}",
                f"[green]{o.confidence}[/green]" if o.confidence == "high" else o.confidence,
            )

        console.print(table)

    if marginal:
        table = Table(title=f"Marginal Opportunities ({len(marginal)} - fees eat the edge)", show_lines=True)
        table.add_column("Type", style="cyan", width=20)
        table.add_column("Description", width=60)
        table.add_column("Gross", justify="right")
        table.add_column("Net", justify="right", style="red")

        for o in sorted(marginal, key=lambda x: x.gross_profit, reverse=True)[:10]:
            table.add_row(o.arb_type, o.description, f"${o.gross_profit:.3f}", f"${o.net_profit:.3f}")

        console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Marginal Polytope Arbitrage Scanner")
    parser.add_argument("--series", type=str, help="Scan single series (e.g., KXHIGHNY)")
    parser.add_argument("--deep", action="store_true", help="Deep scan with dependency analysis")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    console.print(Panel(
        "[bold]Marginal Polytope Arbitrage Scanner[/bold]\n"
        "Scanning Kalshi weather markets for cross-market dependencies\n"
        "Based on: 'Unravelling the Probabilistic Forest' (arXiv:2508.03474)",
        border_style="cyan",
    ))

    # Initialize client
    try:
        client = KalshiClient()
    except Exception as e:
        console.print(f"[red]Failed to initialize Kalshi client: {e}[/red]")
        console.print("[yellow]Make sure your .env has valid KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH[/yellow]")
        sys.exit(1)

    # Determine which series to scan
    series_list = [args.series.upper()] if args.series else WEATHER_SERIES

    # Fetch all markets
    console.print("[dim]Fetching markets...[/dim]")
    all_markets = {}
    total = 0
    for series in series_list:
        markets = fetch_markets_for_series(client, series)
        all_markets[series] = markets
        total += len(markets)
        console.print(f"  {CITY_NAMES.get(series, series)}: {len(markets)} markets")
        time.sleep(0.3)  # rate limit

    console.print(f"\n[bold]Total markets: {total}[/bold]\n")

    if total == 0:
        console.print("[yellow]No open markets found. Markets may be closed.[/yellow]")
        return

    # Display market summary
    display_market_summary(all_markets)

    # Flatten for scanning
    flat_markets = []
    for markets in all_markets.values():
        flat_markets.extend(markets)

    # Display bucket breakdown
    display_bucket_breakdown(all_markets)

    # Run scans
    console.print("\n[dim]Running arbitrage scans...[/dim]\n")
    all_opps = []

    # Type 1: Bucket sum arbitrage
    bucket_opps = scan_bucket_sum_arb(flat_markets)
    all_opps.extend(bucket_opps)

    # Type 2: Threshold-bucket dependency
    if args.deep or True:  # always run dependency scan
        dep_opps = scan_threshold_dependency_arb(flat_markets)
        all_opps.extend(dep_opps)

    # Display results
    display_opportunities(all_opps)

    # Summary
    profitable = [o for o in all_opps if o.net_profit > 0]
    console.print(f"\n[bold]Scan complete.[/bold]")
    console.print(f"  Markets scanned: {total}")
    console.print(f"  Opportunities found: {len(all_opps)}")
    console.print(f"  Profitable after fees: {len(profitable)}")
    if profitable:
        total_net = sum(o.net_profit for o in profitable)
        console.print(f"  Total extractable profit: [bold green]${total_net:.2f}[/bold green]")

    # JSON output
    if args.json:
        output = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "markets_scanned": total,
            "opportunities": [
                {
                    "type": o.arb_type,
                    "description": o.description,
                    "gross_profit": o.gross_profit,
                    "net_profit": o.net_profit,
                    "confidence": o.confidence,
                    "details": o.details,
                }
                for o in all_opps
            ],
        }
        out_path = config.OUTPUT_DIR / "polytope_scan.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        console.print(f"\nJSON saved to {out_path}")


if __name__ == "__main__":
    main()
