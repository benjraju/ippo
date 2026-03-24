"""
weather_tail_strategy.py -- Focused weather tail edge strategy with dutch book math.

STRATEGY: Buy NO on weather longshot markets (YES <= 15c) where our
forecast model says the probability is near zero.

WHY THIS WORKS:
  - Weather longshot markets price at 5-15c YES
  - Historical data shows 1-2% actual YES rate vs 7-12% implied (z>2.1)
  - Buying NO at 85-95c yields 5-15c per contract on >98% of trades
  - The dutch book total (sum of all YES asks for an event) tells us
    exactly how mispriced each bucket is
  - Risk: when tail DOES hit (~1-2% of the time), loss is large (85-95c/contract)
  - Kelly criterion + tail risk budgeting manages the bankroll impact

EDGE SOURCES:
  1. Model edge: NWS forecast uncertainty model puts near-zero probability on tails
  2. Dutch book edge: sum(YES asks) often exceeds 100c, meaning some buckets are overpriced
  3. Fee edge: maker orders at 1.75% fee rate vs taker 7% -- limit orders are 4x cheaper

INTEGRATION WITH auto_trade.py:
  This module returns trade dicts compatible with auto_trade.py's TradeDecision format.
  Import find_weather_tail_trades() from auto_trade.py or run standalone.

Usage:
    python weather_tail_strategy.py             # scan and display opportunities
    python weather_tail_strategy.py --json      # output JSON
    python weather_tail_strategy.py --dry-run   # show what would trade
"""

import json
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

import config

try:
    sys.path.insert(0, str(Path(__file__).parent / "autoresearch"))
    from candidate_strategy import (
        WEATHER_TAIL_MAX_YES,
        WEATHER_TAIL_MIN_NO_PROB,
        WEATHER_TAIL_MIN_EDGE,
        WEATHER_TAIL_MAX_CONTRACTS,
    )
except ImportError:
    WEATHER_TAIL_MAX_YES = 15
    WEATHER_TAIL_MIN_NO_PROB = 0.90
    WEATHER_TAIL_MIN_EDGE = 0.5
    WEATHER_TAIL_MAX_CONTRACTS = 3
from kalshi_client import KalshiClient
from dutch_book import (
    fetch_event_markets,
    kalshi_fee_for_leg,
    to_cents,
    MarketLeg,
    MULTI_OUTCOME_SERIES,
)
from weather_strategy import (
    NWS_GRID_POINTS,
    fetch_nws_forecast,
    calc_bucket_probability,
    calc_above_probability,
    calc_below_probability,
    parse_market_type,
    normal_cdf,
    FORECAST_STDEV,
)

console = Console()

# Weather-only series for tail scanning
WEATHER_SERIES = [s for s in MULTI_OUTCOME_SERIES if s.startswith("KXHIGH")]

# Tail trade thresholds (values imported from candidate_strategy, with fallback defaults above)
MAX_YES_PRICE_CENTS = WEATHER_TAIL_MAX_YES
MIN_NO_PROBABILITY = WEATHER_TAIL_MIN_NO_PROB
MIN_NET_EDGE_CENTS = WEATHER_TAIL_MIN_EDGE
MAX_CONTRACTS_PER_TRADE = WEATHER_TAIL_MAX_CONTRACTS
DEFAULT_BANKROLL = 75.0      # Default bankroll for risk budgeting


# ---------------------------------------------------------------------------
# Kelly Criterion for Tail Trades
# ---------------------------------------------------------------------------

def compute_tail_kelly(
    yes_price_cents: float,
    no_win_rate: float,
    fee_rate: float = config.KALSHI_MAKER_FEE_RATE,
) -> dict:
    """
    Kelly criterion for weather tail trades.

    Buy NO at (100 - yes_price) cents.
    Win: collect yes_price cents minus fee.
    Lose: lose (100 - yes_price) cents plus fee.

    Args:
        yes_price_cents: Current YES ask price in cents (e.g., 3)
        no_win_rate: Our model's estimated P(NO) (e.g., 0.99)
        fee_rate: Fee rate (default: maker rate 0.0175)

    Returns:
        {
            "kelly_fraction": float,    # Full Kelly fraction of bankroll
            "quarter_kelly": float,     # Conservative quarter Kelly
            "ev_per_dollar": float,     # Expected value per dollar risked
            "max_loss_per_contract": float,  # Max loss in dollars if YES wins
        }
    """
    no_price_cents = 100 - yes_price_cents
    p_no = no_win_rate
    p_yes = 1.0 - no_win_rate

    # Win payoff: yes_price_cents minus fee (in dollars per contract)
    # Fee on winning: ceil(rate * 1 * price * (1-price)) in cents
    no_price_frac = no_price_cents / 100.0
    fee_cents = math.ceil(fee_rate * 1 * no_price_frac * (1 - no_price_frac) * 100) / 100
    win_payoff_cents = yes_price_cents - fee_cents
    win_payoff = win_payoff_cents / 100.0

    # Loss payoff: lose the NO price (in dollars per contract)
    loss_payoff = no_price_cents / 100.0

    # EV per contract
    ev_per_contract = (p_no * win_payoff) - (p_yes * loss_payoff)
    ev_per_dollar = ev_per_contract / loss_payoff if loss_payoff > 0 else 0.0

    # Kelly: f* = (p * b - q) / b where b = win/loss ratio
    b = win_payoff / loss_payoff if loss_payoff > 0 else 0
    if b > 0:
        kelly = (p_no * b - p_yes) / b
    else:
        kelly = 0.0

    kelly = max(0.0, kelly)  # Never negative (don't bet the other side)

    return {
        "kelly_fraction": round(kelly, 6),
        "quarter_kelly": round(kelly * 0.25, 6),
        "ev_per_dollar": round(ev_per_dollar, 6),
        "max_loss_per_contract": round(loss_payoff, 4),
    }


# ---------------------------------------------------------------------------
# Tail Risk Budget
# ---------------------------------------------------------------------------

def tail_risk_budget(
    bankroll: float,
    max_single_loss_pct: float = 0.10,
) -> dict:
    """
    Compute maximum position size for tail trades.

    Key constraint: when a tail event DOES happen (0.14% of the time for
    extreme tails, up to 2% for near-tails), the loss is large.

    At 10 contracts buying NO at 97c, loss = $9.70 = 13% of $75 bankroll.
    We need to size so that a single tail event loss is survivable.

    Args:
        bankroll: Current account balance in dollars
        max_single_loss_pct: Max loss on any single tail trade as fraction of bankroll

    Returns:
        {
            "max_risk_dollars": float,       # Max dollars at risk per tail trade
            "example_97c_contracts": int,    # Max contracts if NO costs 97c
            "example_95c_contracts": int,    # Max contracts if NO costs 95c
            "expected_loss_frequency": str,  # How often tail losses happen
        }
    """
    max_risk = bankroll * max_single_loss_pct

    # Example sizing for common NO prices
    contracts_97 = int(max_risk / 0.97) if max_risk >= 0.97 else 0
    contracts_95 = int(max_risk / 0.95) if max_risk >= 0.95 else 0

    return {
        "max_risk_dollars": round(max_risk, 2),
        "example_97c_contracts": contracts_97,
        "example_95c_contracts": contracts_95,
        "expected_loss_frequency": (
            "Extreme tails (YES <= 5c): ~0.5-2% of events. "
            "Longshots (YES 5-15c): ~1-3% of events. "
            "Budget for 2-4 tail losses per month across all cities."
        ),
    }


# ---------------------------------------------------------------------------
# Core Strategy: Find Weather Tail Trades
# ---------------------------------------------------------------------------

def find_weather_tail_trades(client: KalshiClient = None) -> list[dict]:
    """
    Scan all weather markets for tail trades.

    For each city-date event:
    1. Compute sum(YES asks) -- the dutch book total
    2. Identify YES markets priced <= 15c (longshot markets)
    3. For each longshot market, compute:
       - Expected NO probability from our model (should be ~90%+)
       - Dutch book implied probability (from sum of all buckets)
       - Net edge after fees
    4. Only return trades where:
       - YES price <= 15c
       - Our model says P(NO) > 90%
       - Net edge after maker fees > 0.5c per contract
       - Market has any liquidity (yes_ask > 0)

    Returns list of trade dicts compatible with auto_trade.py's TradeDecision.
    """
    if client is None:
        client = KalshiClient()

    trades = []

    for series_ticker in WEATHER_SERIES:
        city = {
            "KXHIGHNY": "NYC",
            "KXHIGHCHI": "Chicago",
            "KXHIGHMIA": "Miami",
            "KXHIGHLA": "LA",
            "KXHIGHDC": "DC",
            "KXHIGHDEN": "Denver",
        }.get(series_ticker, series_ticker)

        # Fetch NWS forecast for model probabilities
        forecast_by_date = fetch_nws_forecast(series_ticker)
        if not forecast_by_date:
            continue

        # Fetch all markets grouped by event
        try:
            event_markets = fetch_event_markets(client, series_ticker)
        except Exception as e:
            console.print(f"[yellow]Failed to fetch {series_ticker}: {e}[/yellow]")
            continue

        # Process each event (city-date combination)
        for event_ticker, legs in event_markets.items():
            if len(legs) < 3:
                continue

            # 1. Compute dutch book total for this event
            sum_yes_asks = sum(l.yes_ask_cents for l in legs if l.yes_ask_cents > 0)

            # 2. Find tail markets (YES <= MAX_YES_PRICE_CENTS)
            for leg in legs:
                if leg.yes_ask_cents <= 0 or leg.yes_ask_cents > MAX_YES_PRICE_CENTS:
                    continue

                ticker = leg.ticker
                title = leg.title
                yes_price = leg.yes_ask_cents

                # Parse market type to compute our model's probability
                mtype = parse_market_type(ticker, title)
                if not mtype:
                    continue

                # Extract the settlement date from ticker
                market_date = _extract_date(ticker)
                if not market_date or market_date not in forecast_by_date:
                    continue

                forecast_temp = forecast_by_date[market_date]["temperature"]

                # Calculate days out for stdev
                now = datetime.now(timezone.utc)
                try:
                    market_dt = datetime.strptime(market_date, "%Y-%m-%d").replace(
                        tzinfo=timezone.utc
                    )
                    days_out = max(0, (market_dt.date() - now.date()).days)
                except Exception:
                    days_out = 1

                stdev = FORECAST_STDEV.get(min(days_out, 3), 4.0)

                # 3. Compute our model's probability for this bucket
                if mtype["type"] == "bucket":
                    our_yes_prob = calc_bucket_probability(
                        forecast_temp, mtype["low"], mtype["high"], stdev
                    )
                elif mtype["type"] == "above":
                    our_yes_prob = calc_above_probability(
                        forecast_temp, mtype["threshold"], stdev
                    )
                elif mtype["type"] == "below":
                    our_yes_prob = calc_below_probability(
                        forecast_temp, mtype["threshold"], stdev
                    )
                else:
                    continue

                our_no_prob = 1.0 - our_yes_prob

                # 4. Apply filters
                if our_no_prob < MIN_NO_PROBABILITY:
                    continue

                # Dutch book implied probability for this leg
                dutch_book_implied = (
                    yes_price / sum_yes_asks if sum_yes_asks > 0 else yes_price / 100.0
                )

                # Compute edge after maker fees
                no_price_cents = 100 - yes_price
                fee_cents = kalshi_fee_for_leg(no_price_cents, is_maker=True)
                win_cents = yes_price - fee_cents  # profit on NO win
                net_edge = win_cents * our_no_prob - no_price_cents * (1 - our_no_prob)

                if net_edge < MIN_NET_EDGE_CENTS:
                    continue

                # Kelly sizing
                kelly = compute_tail_kelly(
                    yes_price_cents=yes_price,
                    no_win_rate=our_no_prob,
                )

                # Build trade dict
                trade = {
                    "ticker": ticker,
                    "title": title,
                    "series": series_ticker,
                    "city": city,
                    "event_ticker": event_ticker,
                    "market_date": market_date,
                    "side": "no",
                    "action": "buy",
                    "strategy": "weather_tail",
                    # Prices
                    "yes_price_cents": yes_price,
                    "no_price_cents": int(no_price_cents),
                    "fee_cents": round(fee_cents, 2),
                    # Probabilities
                    "our_yes_prob": round(our_yes_prob, 6),
                    "our_no_prob": round(our_no_prob, 6),
                    "market_yes_prob": round(yes_price / 100.0, 4),
                    "dutch_book_total": round(sum_yes_asks, 1),
                    "dutch_book_implied": round(dutch_book_implied, 6),
                    # Edge
                    "net_edge_cents": round(net_edge, 2),
                    "ev_per_dollar": kelly["ev_per_dollar"],
                    "kelly_fraction": kelly["kelly_fraction"],
                    "quarter_kelly": kelly["quarter_kelly"],
                    # Forecast
                    "forecast_temp": forecast_temp,
                    "forecast_stdev": stdev,
                    "days_out": days_out,
                    # Volume
                    "volume": leg.volume,
                    # Metadata
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                trades.append(trade)

        time.sleep(0.5)  # Rate limit between series

    # Sort by net edge descending
    trades.sort(key=lambda t: t["net_edge_cents"], reverse=True)
    return trades


def _extract_date(ticker: str) -> Optional[str]:
    """Extract settlement date from ticker like KXHIGHCHI-26MAR21-T63 -> 2026-03-21."""
    month_map = {
        "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
        "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
        "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
    }
    match = re.search(r"(\d{2})([A-Z]{3})(\d{2})", ticker)
    if not match:
        return None
    year_short, month_str, day_str = match.groups()
    month_num = month_map.get(month_str)
    if not month_num:
        return None
    return f"20{year_short}-{month_num}-{day_str}"


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_tail_trades(trades: list[dict]):
    """Pretty-print detected weather tail trade opportunities."""
    if not trades:
        console.print(Panel(
            "[dim]No weather tail trades found above thresholds.\n"
            f"Filters: YES <= {MAX_YES_PRICE_CENTS}c, P(NO) > {MIN_NO_PROBABILITY:.0%}, "
            f"edge > {MIN_NET_EDGE_CENTS}c[/dim]",
            title="Weather Tail Strategy",
            border_style="yellow",
        ))
        return

    table = Table(
        title=f"Weather Tail Trades ({len(trades)} found)",
        show_lines=False,
    )
    table.add_column("City", style="cyan", width=8)
    table.add_column("Ticker", style="white", width=30, no_wrap=True)
    table.add_column("YES$", justify="right", width=5)
    table.add_column("NO$", justify="right", width=5)
    table.add_column("P(NO)", justify="right", style="green", width=8)
    table.add_column("Edge", justify="right", style="bold green", width=7)
    table.add_column("EV/$", justify="right", style="yellow", width=7)
    table.add_column("Kelly", justify="right", style="dim", width=7)
    table.add_column("DB Tot", justify="right", style="dim", width=7)
    table.add_column("Fcst", justify="right", width=5)

    for t in trades:
        table.add_row(
            t["city"],
            t["ticker"][:30],
            f"{t['yes_price_cents']:.0f}c",
            f"{t['no_price_cents']}c",
            f"{t['our_no_prob']:.1%}",
            f"+{t['net_edge_cents']:.1f}c",
            f"{t['ev_per_dollar']:.3f}",
            f"{t['quarter_kelly']:.3f}",
            f"{t['dutch_book_total']:.0f}c",
            f"{t['forecast_temp']:.0f}F",
        )

    console.print(table)

    # Summary stats
    total_edge = sum(t["net_edge_cents"] for t in trades)
    avg_no_prob = sum(t["our_no_prob"] for t in trades) / len(trades)
    console.print(
        f"\n  Total edge: [bold green]+{total_edge:.1f}c[/bold green] "
        f"across {len(trades)} opportunities"
    )
    console.print(f"  Avg P(NO): {avg_no_prob:.1%}")

    # Risk budget
    budget = tail_risk_budget(DEFAULT_BANKROLL)
    console.print(
        f"\n  Risk budget (${DEFAULT_BANKROLL:.0f} bankroll, "
        f"{budget['max_risk_dollars']:.0f}% max loss):"
    )
    console.print(
        f"    Max risk/trade: ${budget['max_risk_dollars']:.2f}"
    )
    console.print(
        f"    Max contracts @ 97c NO: {budget['example_97c_contracts']}"
    )
    console.print(
        f"    Max contracts @ 95c NO: {budget['example_95c_contracts']}"
    )
    console.print(f"\n  [dim]{budget['expected_loss_frequency']}[/dim]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Weather Tail Strategy -- buy NO on extreme tail weather markets"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON to output/weather_tail_trades.json",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what trades would be placed (no actual orders)",
    )
    parser.add_argument(
        "--max-yes", type=float, default=MAX_YES_PRICE_CENTS,
        help=f"Max YES price in cents (default: {MAX_YES_PRICE_CENTS})",
    )
    parser.add_argument(
        "--min-edge", type=float, default=MIN_NET_EDGE_CENTS,
        help=f"Min net edge in cents (default: {MIN_NET_EDGE_CENTS})",
    )
    parser.add_argument(
        "--bankroll", type=float, default=DEFAULT_BANKROLL,
        help=f"Bankroll for risk sizing (default: ${DEFAULT_BANKROLL})",
    )
    args = parser.parse_args()

    # Use CLI overrides for thresholds
    max_yes = args.max_yes
    min_edge = args.min_edge
    bankroll = args.bankroll

    console.print(Panel(
        "[bold]Weather Tail Strategy Scanner[/bold]\n"
        f"Buy NO on weather tails: YES <= {max_yes}c, "
        f"P(NO) > {MIN_NO_PROBABILITY:.0%}, edge > {min_edge}c\n"
        "Uses NWS forecast + dutch book math to identify mispriced tails",
        border_style="cyan",
    ))

    # Initialize client
    try:
        client = KalshiClient()
        balance = client.get_balance()
        bal_str = (
            f"${float(balance.get('balance', 0)) / 100:.2f}"
            if balance else "unknown"
        )
        console.print(f"  [dim]Connected to {config.KALSHI_ENV} | Balance: {bal_str}[/dim]\n")
    except Exception as e:
        console.print(f"[red]Failed to initialize Kalshi client: {e}[/red]")
        sys.exit(1)

    # Scan
    console.print("[dim]Scanning weather series for tail opportunities...[/dim]\n")
    trades = find_weather_tail_trades(client)

    # Display
    display_tail_trades(trades)

    # JSON output
    if args.json and trades:
        output = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "params": {
                "max_yes_cents": max_yes,
                "min_no_prob": MIN_NO_PROBABILITY,
                "min_edge_cents": min_edge,
            },
            "n_trades": len(trades),
            "trades": trades,
            "risk_budget": tail_risk_budget(bankroll),
        }
        out_path = config.OUTPUT_DIR / "weather_tail_trades.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        console.print(f"\n[dim]JSON saved to {out_path}[/dim]")

    console.print(f"\n[bold]Scan complete. {len(trades)} tail opportunities found.[/bold]")


if __name__ == "__main__":
    main()
