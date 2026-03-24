"""
nba_underdog_strategy.py -- Focused NBA underdog YES strategy.

ONLY trades KXNBAGAME game winner markets. NO props, NO player points,
NO favorites. Just underdogs priced 10-30c where the favorite-longshot
bias gives us a +7.7pp edge.

Uses maker orders only. Quarter Kelly sizing.

STRATEGY THESIS:
  - NBA underdogs priced 10-30c on Kalshi win 29.7% of the time (n=64)
  - Implied probability from price is ~22%, giving +7.7pp edge
  - This is the well-documented "favorite-longshot bias" in sports betting
  - The 25-30c bucket is strongest: 42.3% win rate vs 27.5% implied (n=26)

EDGE SOURCES:
  1. Favorite-longshot bias: bettors overpay for favorites, underprice underdogs
  2. Maker fee advantage: 1.75% maker vs 7% taker = 4x cheaper execution
  3. Binary outcome: game winner is clean — either team wins, no ambiguity

RISK MANAGEMENT:
  - Quarter Kelly sizing (~$2 per trade on $75 bankroll)
  - Max 5 underdog bets per day (portfolio concentration limit)
  - Max $2 per individual trade
  - Maker orders only (post_only=True) to minimize fee drag

INTEGRATION:
  Import find_nba_underdogs() and nba_risk_budget() from auto_trade.py.
  This module returns trade dicts compatible with TradeDecision format.

Usage:
    python nba_underdog_strategy.py             # scan and display opportunities
    python nba_underdog_strategy.py --json      # output JSON
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import config
from kalshi_client import KalshiClient

try:
    sys.path.insert(0, str(Path(__file__).parent / "autoresearch"))
    from candidate_strategy import (
        UNDERDOG_MAX_PRICE,
        UNDERDOG_MIN_PRICE,
        UNDERDOG_MAX_BET_DOLLARS,
        UNDERDOG_MAX_CONTRACTS,
        UNDERDOG_WINNER_ONLY,
    )
except ImportError:
    UNDERDOG_MAX_PRICE = 25
    UNDERDOG_MIN_PRICE = 10
    UNDERDOG_MAX_BET_DOLLARS = 2
    UNDERDOG_MAX_CONTRACTS = 20
    UNDERDOG_WINNER_ONLY = 0

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SERIES_TICKER = "KXNBAGAME"
MIN_YES_PRICE_CENTS = UNDERDOG_MIN_PRICE
MAX_YES_PRICE_CENTS = UNDERDOG_MAX_PRICE
ESTIMATED_WIN_RATE = 0.297    # Historical win rate for 10-30c underdogs (not mutated by autoresearch)
DEFAULT_BANKROLL = 75.0
MAX_DAILY_BETS = 5
MAX_BET_DOLLARS = UNDERDOG_MAX_BET_DOLLARS


# ---------------------------------------------------------------------------
# Kelly Criterion for Underdog Bets
# ---------------------------------------------------------------------------

def nba_underdog_kelly(
    yes_price_cents: int,
    estimated_win_rate: float = ESTIMATED_WIN_RATE,
    fee_rate: float = config.KALSHI_MAKER_FEE_RATE,
) -> dict:
    """
    Kelly criterion for NBA underdog YES bets.

    Buy YES at yes_price_cents.
    Win: collect (100 - yes_price_cents) minus fee.
    Lose: lose yes_price_cents.

    Example at 20c:
      b = win/loss = (80 - fee) / 20
      Kelly = (b * p - q) / b
      = (3.54 * 0.297 - 0.703) / 3.54 = 9.8%
      Quarter Kelly = 2.5%

    Args:
        yes_price_cents: Current YES ask price in cents (10-30)
        estimated_win_rate: Our estimated probability the underdog wins
        fee_rate: Kalshi maker fee rate (default 0.0175)

    Returns:
        dict with kelly_fraction, quarter_kelly, ev_per_dollar, cost_per_contract
    """
    p = estimated_win_rate
    q = 1.0 - p

    # Cost to buy YES (in dollars)
    cost = yes_price_cents / 100.0

    # Gross win payoff (per contract, in dollars)
    gross_win = (100 - yes_price_cents) / 100.0

    # Fee on winning side: ceil(rate * contracts * price * (1-price)) cents
    # For 1 contract, fee = ceil(rate * price * (1-price) * 100) / 100 dollars
    price_frac = yes_price_cents / 100.0
    fee_cents = math.ceil(fee_rate * price_frac * (1 - price_frac) * 100)
    fee_dollars = fee_cents / 100.0

    # Net win payoff after fee
    net_win = gross_win - fee_dollars

    # b = win/loss ratio (net of fees)
    b = net_win / cost if cost > 0 else 0

    # Kelly fraction: f* = (b*p - q) / b
    if b > 0:
        kelly = (b * p - q) / b
    else:
        kelly = 0.0
    kelly = max(0.0, kelly)

    # EV per contract
    ev = p * net_win - q * cost

    return {
        "kelly_fraction": round(kelly, 6),
        "quarter_kelly": round(kelly * 0.25, 6),
        "ev_per_contract": round(ev, 4),
        "ev_per_dollar": round(ev / cost, 4) if cost > 0 else 0.0,
        "cost_per_contract": round(cost, 4),
        "net_win_per_contract": round(net_win, 4),
        "fee_per_contract": round(fee_dollars, 4),
        "implied_prob": round(price_frac, 4),
        "estimated_prob": round(p, 4),
        "edge_pp": round((p - price_frac) * 100, 1),
    }


# ---------------------------------------------------------------------------
# Risk Budget
# ---------------------------------------------------------------------------

def nba_risk_budget(
    bankroll: float = DEFAULT_BANKROLL,
    max_daily_bets: int = MAX_DAILY_BETS,
    max_bet_dollars: float = MAX_BET_DOLLARS,
) -> dict:
    """
    Compute risk budget for NBA underdog bets.

    Conservative sizing: max $2 per trade, max 5 per day.
    Total daily exposure capped at $10 (13% of $75 bankroll).

    Args:
        bankroll: Current account balance
        max_daily_bets: Maximum number of underdog bets per day
        max_bet_dollars: Maximum dollars per individual bet

    Returns:
        dict with max_bet_dollars, max_daily_bets, max_contracts_at_price, etc.
    """
    # Cap single bet at lesser of $2 or 3% of bankroll
    effective_max = min(max_bet_dollars, bankroll * 0.03)

    # Contracts at common price points
    contracts = {}
    for price_c in [10, 15, 20, 25, 30]:
        cost = price_c / 100.0
        contracts[f"contracts_at_{price_c}c"] = int(effective_max / cost) if cost > 0 else 0

    return {
        "max_bet_dollars": round(effective_max, 2),
        "max_daily_bets": max_daily_bets,
        "max_daily_exposure": round(effective_max * max_daily_bets, 2),
        "bankroll": round(bankroll, 2),
        "exposure_pct": round(effective_max * max_daily_bets / bankroll * 100, 1) if bankroll > 0 else 0,
        **contracts,
    }


# ---------------------------------------------------------------------------
# Core Strategy: Find NBA Underdogs
# ---------------------------------------------------------------------------

def find_nba_underdogs(client: KalshiClient = None) -> list[dict]:
    """
    Scan KXNBAGAME series for open markets with YES ask between 10-30c.

    For each qualifying market:
    1. Get the order book to find current YES ask price
    2. Verify price is in 10-30c range
    3. Compute Kelly sizing and EV
    4. Return sorted by EV (best first)

    Returns:
        List of trade dicts with ticker, teams, price, kelly, EV, etc.
    """
    if client is None:
        client = KalshiClient()

    trades = []

    # Fetch all open KXNBAGAME markets
    cursor = None
    all_markets = []
    for _ in range(10):  # max 10 pages
        try:
            resp = client.get_markets(
                series_ticker=SERIES_TICKER,
                status="open",
                limit=100,
                cursor=cursor,
            )
        except Exception as e:
            print(f"Error fetching markets: {e}")
            break

        batch = resp.get("markets", [])
        all_markets.extend(batch)

        cursor = resp.get("cursor", None)
        if not cursor or not batch:
            break
        time.sleep(0.3)  # rate limit

    # For each market, check if it's an underdog opportunity
    for mkt in all_markets:
        ticker = mkt.get("ticker", "")
        title = mkt.get("title", "")

        # Get order book for current prices
        try:
            ob = client.get_market_orderbook(ticker, depth=3)
            book = ob.get("orderbook", {})
        except Exception:
            continue

        # Find best YES ask (cheapest we can buy YES for)
        yes_asks = book.get("yes", [])
        if not yes_asks:
            continue

        # yes_asks is [[price, quantity], ...] sorted ascending
        best_yes_ask = yes_asks[0][0] if yes_asks else None
        if best_yes_ask is None:
            continue

        # Check if in underdog range
        if best_yes_ask < MIN_YES_PRICE_CENTS or best_yes_ask > MAX_YES_PRICE_CENTS:
            continue

        # Compute Kelly and EV
        kelly_info = nba_underdog_kelly(best_yes_ask)

        # Only trade if EV positive
        if kelly_info["ev_per_contract"] <= 0:
            continue

        # Parse teams from title (e.g., "Los Angeles C at Dallas Winner?")
        teams_match = re.match(r"(.+?)(?:\s+at\s+|\s+vs\.?\s+)(.+?)(?:\s+Winner\??)?$", title, re.I)
        away_team = teams_match.group(1).strip() if teams_match else "?"
        home_team = teams_match.group(2).strip() if teams_match else "?"

        # Determine which team this ticker represents (last segment)
        ticker_parts = ticker.split("-")
        team_code = ticker_parts[-1] if len(ticker_parts) >= 3 else "?"

        trades.append({
            "ticker": ticker,
            "title": title,
            "team_code": team_code,
            "away_team": away_team,
            "home_team": home_team,
            "yes_price_cents": best_yes_ask,
            "implied_prob": kelly_info["implied_prob"],
            "estimated_prob": kelly_info["estimated_prob"],
            "edge_pp": kelly_info["edge_pp"],
            "ev_per_contract": kelly_info["ev_per_contract"],
            "ev_per_dollar": kelly_info["ev_per_dollar"],
            "kelly_fraction": kelly_info["kelly_fraction"],
            "quarter_kelly": kelly_info["quarter_kelly"],
            "cost_per_contract": kelly_info["cost_per_contract"],
            "net_win_per_contract": kelly_info["net_win_per_contract"],
            "fee_per_contract": kelly_info["fee_per_contract"],
            "suggested_contracts": 1,  # conservative default
        })

        time.sleep(0.2)  # rate limit between orderbook calls

    # Sort by EV per dollar (best opportunities first)
    trades.sort(key=lambda t: t["ev_per_dollar"], reverse=True)

    return trades


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def main():
    """Scan and display NBA underdog opportunities."""
    output_json = "--json" in sys.argv

    client = KalshiClient()
    trades = find_nba_underdogs(client)

    if output_json:
        print(json.dumps(trades, indent=2))
        return

    print(f"\n{'='*70}")
    print(f"  NBA UNDERDOG SCANNER — {SERIES_TICKER}")
    print(f"  Range: {MIN_YES_PRICE_CENTS}-{MAX_YES_PRICE_CENTS}c YES | "
          f"Est win rate: {ESTIMATED_WIN_RATE*100:.1f}%")
    print(f"{'='*70}\n")

    if not trades:
        print("  No underdog opportunities found right now.\n")
        return

    for i, t in enumerate(trades, 1):
        print(f"  {i}. {t['ticker']}")
        print(f"     {t['title']}")
        print(f"     YES@{t['yes_price_cents']}c | "
              f"Implied={t['implied_prob']*100:.0f}% | "
              f"Est={t['estimated_prob']*100:.1f}% | "
              f"Edge={t['edge_pp']:+.1f}pp")
        print(f"     EV={t['ev_per_contract']*100:.1f}c/contract | "
              f"Kelly={t['kelly_fraction']*100:.1f}% | "
              f"QKelly={t['quarter_kelly']*100:.2f}%")
        print()

    # Risk budget
    budget = nba_risk_budget()
    print(f"  Risk Budget: ${budget['max_bet_dollars']}/trade, "
          f"{budget['max_daily_bets']} bets/day, "
          f"${budget['max_daily_exposure']} max exposure "
          f"({budget['exposure_pct']:.0f}% of ${budget['bankroll']})")
    print()


if __name__ == "__main__":
    main()
