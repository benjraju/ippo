"""
dutch_book.py -- Multi-outcome arbitrage scanner for Kalshi.

Only scans mutually exclusive outcome events (weather buckets, crypto buckets).
NBA props are NOT mutually exclusive (multiple thresholds can settle YES) and
are excluded. For example, "10+ points" and "15+ points" can both settle YES
simultaneously, which breaks the Dutch book assumption that exactly one outcome
wins.

Dutch book principle: if an event has N mutually exclusive, exhaustive outcomes,
the sum of all YES prices must equal $1.00. Any deviation is free money:

  - Sum < $1.00: buy all YES contracts -> guaranteed $1.00 payout for less
  - Sum > $1.00: sell all YES contracts -> collect more than $1.00, pay out exactly $1.00

Valid event types: weather temp buckets (e.g. "57-58°F", "58-59°F") and
crypto price range buckets (e.g. "$80K-$82K", "$82K-$84K"). These are
truly mutually exclusive — exactly one bucket settles YES.

Usage:
    python dutch_book.py                          # Scan all series
    python dutch_book.py --series KXHIGHNY        # Single series
    python dutch_book.py --min-profit 5           # Min 5c profit threshold
    python dutch_book.py --json                   # Output JSON
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

# Series to scan -- ONLY mutually exclusive outcome events.
# Sports props (NBA, NHL, MLB, etc.) are excluded because they use cumulative
# thresholds ("10+ points", "15+ points") where multiple outcomes can settle YES
# simultaneously, violating the Dutch book assumption.
MULTI_OUTCOME_SERIES = [
    # Weather -- exclusive temp buckets (e.g. "57-58°F", "58-59°F")
    "KXHIGHNY",
    "KXHIGHCHI",
    "KXHIGHMIA",
    "KXHIGHLA",
    "KXHIGHDC",
    "KXHIGHDEN",
    # Crypto -- exclusive price range buckets (e.g. "$80K-$82K")
    "KXBTC",
    "KXETH",
    "KXSOL",
]


@dataclass
class DutchBookOpportunity:
    """A detected Dutch book arbitrage opportunity."""
    event_ticker: str
    event_title: str
    arb_type: str           # "buy_all" or "sell_all"
    markets: list           # list of market dicts with ticker, title, price info
    total_cost_cents: float # sum of all YES ask prices (buy_all) or 100*n - sum bids (sell_all)
    payout_cents: float     # guaranteed payout in cents (always 100)
    gross_profit_cents: float
    total_fees_cents: float
    net_profit_cents: float
    num_legs: int
    series: str


@dataclass
class MarketLeg:
    """One leg of a Dutch book trade."""
    ticker: str
    title: str
    yes_bid_cents: float
    yes_ask_cents: float
    volume: int


def to_cents(val) -> float:
    """Convert Kalshi dollar string (e.g. '0.3500') to cents."""
    if val is None:
        return 0.0
    try:
        return round(float(val) * 100, 2)
    except (ValueError, TypeError):
        return 0.0


def kalshi_fee_for_leg(price_cents: float, is_maker: bool = False) -> float:
    """
    Calculate Kalshi fee in cents for 1 contract at given price.

    Uses the actual Kalshi fee formula:
        fee = ceil(rate * contracts * price * (1-price))
    where price is in [0, 1].

    For Dutch book we use taker rate since we're buying at the ask.
    """
    rate = config.KALSHI_MAKER_FEE_RATE if is_maker else config.KALSHI_TAKER_FEE_RATE
    p = price_cents / 100.0
    if p <= 0 or p >= 1:
        return 0.0
    # fee formula returns cents
    return math.ceil(rate * 1 * p * (1 - p) * 100) / 100


def fetch_event_markets(client: KalshiClient, series_ticker: str) -> dict[str, list[MarketLeg]]:
    """
    Fetch all open markets for a series and group by event_ticker.

    Returns: {event_ticker: [MarketLeg, ...]}
    """
    events: dict[str, list[MarketLeg]] = {}
    cursor = None
    max_pages = 10

    for _ in range(max_pages):
        try:
            resp = client.get_markets(
                series_ticker=series_ticker,
                status="open",
                limit=200,
                cursor=cursor,
            )
        except Exception as e:
            console.print(f"[yellow]API error for {series_ticker}: {e}[/yellow]")
            break

        for m in resp.get("markets", []):
            event_ticker = m.get("event_ticker", "")
            if not event_ticker:
                continue

            yes_bid = to_cents(m.get("yes_bid_dollars") or m.get("yes_bid") or 0)
            yes_ask = to_cents(m.get("yes_ask_dollars") or m.get("yes_ask") or 0)

            vol_raw = m.get("volume_fp") or m.get("volume_24h_fp") or m.get("volume") or 0
            try:
                volume = int(float(vol_raw))
            except (ValueError, TypeError):
                volume = 0

            leg = MarketLeg(
                ticker=m.get("ticker", ""),
                title=(m.get("title", "") or "")[:80],
                yes_bid_cents=yes_bid,
                yes_ask_cents=yes_ask,
                volume=volume,
            )

            if event_ticker not in events:
                events[event_ticker] = []
            events[event_ticker].append(leg)

        cursor = resp.get("cursor")
        if not cursor:
            break
        time.sleep(0.5)  # rate limiting

    return events


def _is_cumulative_threshold(title: str) -> bool:
    """
    Heuristic: detect if a market title indicates cumulative thresholds
    rather than mutually exclusive buckets.

    Cumulative patterns (NOT mutually exclusive):
      - "10+ points", "250+ yards", "5+ rebounds"
      - "10 or more points"
      - "at least 10 points"
      - "over 200 yards"
      - "under 10 points" (also cumulative from the other direction)

    Exclusive bucket patterns (OK for Dutch book):
      - "57-58 degrees", "58-59 degrees"
      - "$80,000 - $82,000"
    """
    t = title.lower()
    # Patterns that indicate cumulative / non-exclusive thresholds
    cumulative_patterns = [
        r'\d+\+',            # "10+", "250+"
        r'\bor more\b',      # "10 or more"
        r'\bat least\b',     # "at least 10"
        r'\bover\b',         # "over 200"
        r'\bunder\b',        # "under 10"
        r'\bor fewer\b',     # "10 or fewer"
        r'\bor less\b',      # "10 or less"
        r'\bno more than\b', # "no more than 5"
    ]
    return any(re.search(pat, t) for pat in cumulative_patterns)


def _event_has_cumulative_markets(legs: list[MarketLeg]) -> bool:
    """
    Check if any market in the event uses cumulative threshold titles.
    If ANY leg looks cumulative, the event is not mutually exclusive.
    """
    return any(_is_cumulative_threshold(l.title) for l in legs)


def check_dutch_book(
    event_ticker: str,
    legs: list[MarketLeg],
    min_profit_cents: float = 2.0,
) -> Optional[DutchBookOpportunity]:
    """
    Check a single event for Dutch book arbitrage.

    For mutually exclusive outcomes:
      - Buy all: sum(yes_ask) < 100 -> profit = 100 - sum(asks) - fees
      - Sell all: sum(yes_bid) > 100 -> profit = sum(bids) - 100 - fees

    Validates that markets are mutually exclusive by checking for cumulative
    threshold patterns in titles (e.g. "10+" or "or more"). Events with
    cumulative thresholds are skipped since multiple outcomes can settle YES.

    Returns an opportunity if net profit > min_profit_cents, else None.
    """
    # Filter out legs with no valid quotes
    valid_legs = [l for l in legs if l.yes_ask_cents > 0]
    if len(valid_legs) < 3:
        return None

    # VALIDATION: Skip events with cumulative threshold titles.
    # These are NOT mutually exclusive (e.g. "10+ points" and "15+ points"
    # can both settle YES), so Dutch book logic does not apply.
    if _event_has_cumulative_markets(valid_legs):
        return None

    # Build event title from first leg
    event_title = event_ticker

    # --- Strategy 1: Buy all YES at ask prices ---
    sum_ask = sum(l.yes_ask_cents for l in valid_legs)
    if 0 < sum_ask < 100:
        gross_profit = 100 - sum_ask
        total_fees = sum(kalshi_fee_for_leg(l.yes_ask_cents) for l in valid_legs)
        net_profit = gross_profit - total_fees

        if net_profit >= min_profit_cents:
            return DutchBookOpportunity(
                event_ticker=event_ticker,
                event_title=event_title,
                arb_type="buy_all",
                markets=[
                    {
                        "ticker": l.ticker,
                        "title": l.title,
                        "yes_ask": l.yes_ask_cents,
                        "yes_bid": l.yes_bid_cents,
                        "volume": l.volume,
                    }
                    for l in valid_legs
                ],
                total_cost_cents=sum_ask,
                payout_cents=100,
                gross_profit_cents=gross_profit,
                total_fees_cents=total_fees,
                net_profit_cents=net_profit,
                num_legs=len(valid_legs),
                series="",
            )

    # --- Strategy 2: Sell all YES at bid prices ---
    bid_legs = [l for l in legs if l.yes_bid_cents > 0]
    if len(bid_legs) >= 3:
        sum_bid = sum(l.yes_bid_cents for l in bid_legs)
        if sum_bid > 100:
            gross_profit = sum_bid - 100
            total_fees = sum(kalshi_fee_for_leg(l.yes_bid_cents) for l in bid_legs)
            net_profit = gross_profit - total_fees

            if net_profit >= min_profit_cents:
                return DutchBookOpportunity(
                    event_ticker=event_ticker,
                    event_title=event_title,
                    arb_type="sell_all",
                    markets=[
                        {
                            "ticker": l.ticker,
                            "title": l.title,
                            "yes_ask": l.yes_ask_cents,
                            "yes_bid": l.yes_bid_cents,
                            "volume": l.volume,
                        }
                        for l in bid_legs
                    ],
                    total_cost_cents=100,  # obligation
                    payout_cents=sum_bid,  # revenue from selling
                    gross_profit_cents=gross_profit,
                    total_fees_cents=total_fees,
                    net_profit_cents=net_profit,
                    num_legs=len(bid_legs),
                    series="",
                )

    return None


def scan_all_dutch_books(
    client: KalshiClient,
    series_list: list[str] = None,
    min_profit_cents: float = 2.0,
) -> list[DutchBookOpportunity]:
    """
    Scan across all target series for Dutch book opportunities.

    For each series:
      1. Fetch all open markets
      2. Group by event_ticker
      3. Check each event for Dutch book

    Returns sorted by net profit descending.
    """
    if series_list is None:
        series_list = MULTI_OUTCOME_SERIES

    all_opportunities: list[DutchBookOpportunity] = []
    events_checked = 0

    for series in series_list:
        console.print(f"  [dim]Scanning {series}...[/dim]", end="")

        try:
            event_markets = fetch_event_markets(client, series)
        except Exception as e:
            console.print(f" [red]error: {e}[/red]")
            continue

        series_opps = 0
        for event_ticker, legs in event_markets.items():
            if len(legs) < 3:
                continue  # need 3+ outcomes for multi-outcome event

            events_checked += 1
            opp = check_dutch_book(event_ticker, legs, min_profit_cents)
            if opp:
                opp.series = series
                all_opportunities.append(opp)
                series_opps += 1

        n_events = len([l for l in event_markets.values() if len(l) >= 3])
        if series_opps > 0:
            console.print(f" {n_events} events, [green]{series_opps} opportunities[/green]")
        else:
            console.print(f" {n_events} events, no arb")

        time.sleep(0.3)  # rate limit between series

    all_opportunities.sort(key=lambda o: o.net_profit_cents, reverse=True)
    return all_opportunities


def display_opportunities(opportunities: list[DutchBookOpportunity]):
    """Rich table output of Dutch book opportunities."""
    if not opportunities:
        console.print(Panel(
            "[dim]No Dutch book opportunities found above fee threshold.\n"
            "Multi-outcome event prices are currently well-calibrated.[/dim]",
            title="Scan Result",
            border_style="yellow",
        ))
        return

    # Summary table
    table = Table(
        title=f"Dutch Book Opportunities ({len(opportunities)} found)",
        show_lines=True,
    )
    table.add_column("Event", style="cyan", width=30, no_wrap=True)
    table.add_column("Series", style="dim", width=12)
    table.add_column("Type", style="white", width=10)
    table.add_column("Legs", justify="right", width=5)
    table.add_column("Sum", justify="right", width=8)
    table.add_column("Gross", justify="right", style="green", width=8)
    table.add_column("Fees", justify="right", style="red", width=8)
    table.add_column("Net", justify="right", style="bold green", width=8)

    for opp in opportunities:
        sum_label = f"{opp.total_cost_cents:.1f}c" if opp.arb_type == "buy_all" else f"{opp.payout_cents:.1f}c"
        table.add_row(
            opp.event_ticker[:30],
            opp.series,
            opp.arb_type.upper().replace("_", " "),
            str(opp.num_legs),
            sum_label,
            f"{opp.gross_profit_cents:.1f}c",
            f"{opp.total_fees_cents:.1f}c",
            f"{opp.net_profit_cents:.1f}c",
        )

    console.print(table)

    # Detail breakdown for top opportunities
    for opp in opportunities[:3]:
        detail = Table(
            title=f"Legs: {opp.event_ticker} ({opp.arb_type.upper()})",
            show_lines=False,
            padding=(0, 1),
        )
        detail.add_column("Ticker", style="cyan", width=35, no_wrap=True)
        detail.add_column("Title", style="dim", width=40)
        detail.add_column("Bid", justify="right", width=6)
        detail.add_column("Ask", justify="right", width=6)
        detail.add_column("Vol", justify="right", style="yellow", width=7)
        detail.add_column("Fee", justify="right", style="red", width=6)

        for m in opp.markets:
            price = m["yes_ask"] if opp.arb_type == "buy_all" else m["yes_bid"]
            fee = kalshi_fee_for_leg(price)
            detail.add_row(
                m["ticker"][:35],
                m["title"][:40],
                f"{m['yes_bid']:.0f}c",
                f"{m['yes_ask']:.0f}c",
                f"{m['volume']:,}" if m["volume"] > 0 else "-",
                f"{fee:.1f}c",
            )

        console.print(detail)
        console.print()

    # Summary stats
    total_net = sum(o.net_profit_cents for o in opportunities)
    console.print(
        f"  Total extractable profit: [bold green]{total_net:.1f}c "
        f"(${total_net / 100:.2f})[/bold green] across {len(opportunities)} opportunities"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Dutch Book Arbitrage Scanner -- multi-outcome event mispricing detector"
    )
    parser.add_argument(
        "--series", type=str, default=None,
        help="Scan single series (e.g., KXNBAGAME, KXHIGHNY)",
    )
    parser.add_argument(
        "--min-profit", type=float, default=2.0,
        help="Minimum net profit threshold in cents (default: 2.0)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Save results as JSON to output/dutch_book.json",
    )
    args = parser.parse_args()

    console.print(Panel(
        "[bold]Dutch Book Arbitrage Scanner[/bold]\n"
        "Scanning multi-outcome events where sum(YES prices) != $1.00\n"
        f"Min profit threshold: {args.min_profit}c per contract",
        border_style="cyan",
    ))

    # Initialize client
    try:
        client = KalshiClient()
        balance = client.get_balance()
        bal_str = f"${float(balance.get('balance', 0)) / 100:.2f}" if balance else "unknown"
        console.print(f"  [dim]Connected to {config.KALSHI_ENV} | Balance: {bal_str}[/dim]\n")
    except Exception as e:
        console.print(f"[red]Failed to initialize Kalshi client: {e}[/red]")
        console.print("[yellow]Check .env for KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH[/yellow]")
        sys.exit(1)

    # Determine series to scan
    series_list = [args.series.upper()] if args.series else MULTI_OUTCOME_SERIES

    console.print(f"[dim]Scanning {len(series_list)} series for Dutch book opportunities...[/dim]\n")

    # Run scan
    opportunities = scan_all_dutch_books(
        client,
        series_list=series_list,
        min_profit_cents=args.min_profit,
    )

    # Display
    console.print()
    display_opportunities(opportunities)

    # JSON output
    if args.json:
        output = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "min_profit_cents": args.min_profit,
            "series_scanned": series_list,
            "opportunities": [
                {
                    "event_ticker": o.event_ticker,
                    "series": o.series,
                    "arb_type": o.arb_type,
                    "num_legs": o.num_legs,
                    "total_cost_cents": o.total_cost_cents,
                    "payout_cents": o.payout_cents,
                    "gross_profit_cents": o.gross_profit_cents,
                    "total_fees_cents": o.total_fees_cents,
                    "net_profit_cents": o.net_profit_cents,
                    "markets": o.markets,
                }
                for o in opportunities
            ],
        }
        out_path = config.OUTPUT_DIR / "dutch_book.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        console.print(f"\n[dim]JSON saved to {out_path}[/dim]")

    console.print(f"\n[bold]Scan complete.[/bold]")


if __name__ == "__main__":
    main()
