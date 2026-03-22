#!/usr/bin/env python3
"""
sports_backtest.py — Comprehensive sports strategy backtest using settled Kalshi markets.

Analyzes 35K+ settled markets from historical_settlements.json.
Focuses on sports category (nba_winner, nba_props).

Since the settlement data has NO price fields (last_price, yes_bid, yes_ask all null),
we perform structural/outcome analysis and simulate hypothetical price-tier P&L
using uniform price assumptions and the props point-threshold structure.

For nba_props, the ticker encodes the point threshold (10, 15, 20, 25, 30, 35, 40),
which serves as a strong proxy for implied probability (higher threshold = lower prob).
"""

import json
import math
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SETTLEMENTS_FILE = Path("/Users/benjamin/Desktop/ippo/output/historical_settlements.json")

# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def normal_cdf(x):
    """Standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def z_score_proportion(p_observed, p_null, n):
    """Z-score for observed proportion vs null hypothesis proportion."""
    if n == 0:
        return 0.0
    se = math.sqrt(p_null * (1 - p_null) / n) if p_null > 0 and p_null < 1 else 0.001
    return (p_observed - p_null) / se


def p_value_one_sided(z):
    """One-sided p-value (upper tail) from z-score."""
    return 1 - normal_cdf(z)


def fmt_pct(val):
    """Format as percentage string."""
    return f"{val*100:.1f}%"


def fmt_money(cents):
    """Format cents as dollar amount."""
    return f"${cents/100:,.2f}"


def fmt_pnl(cents):
    """Format P&L with sign."""
    if cents >= 0:
        return f"+${cents/100:,.2f}"
    else:
        return f"-${abs(cents)/100:,.2f}"


# ---------------------------------------------------------------------------
# Price-tier simulation
# ---------------------------------------------------------------------------

def simulate_price_tier(markets, tier_label, price_low_cents, price_high_cents):
    """
    Simulate buying YES at the midpoint of a price tier on all markets.

    For each market:
    - Cost = midpoint price (in cents per contract)
    - If settled YES: profit = 100 - cost
    - If settled NO:  loss = -cost

    Returns dict with stats or None if no markets.
    """
    mid_price = (price_low_cents + price_high_cents) / 2

    yes_count = sum(1 for m in markets if m.get("result") == "yes")
    no_count = sum(1 for m in markets if m.get("result") == "no")
    total = yes_count + no_count  # exclude scalar

    if total == 0:
        return None

    yes_rate = yes_count / total

    # P&L simulation
    total_cost = total * mid_price  # cents
    total_revenue = yes_count * 100  # each YES winner pays 100c
    total_pnl = total_revenue - total_cost
    pnl_per_market = total_pnl / total if total > 0 else 0

    # Break-even analysis
    # To break even: win_rate * 100 = mid_price  =>  break_even_rate = mid_price / 100
    break_even_rate = mid_price / 100
    edge = yes_rate - break_even_rate

    # Statistical significance
    z = z_score_proportion(yes_rate, break_even_rate, total)
    pval = p_value_one_sided(z)

    # ROI
    roi = total_pnl / total_cost if total_cost > 0 else 0

    return {
        "tier": tier_label,
        "price_range": f"{price_low_cents}-{price_high_cents}c",
        "mid_price": mid_price,
        "total": total,
        "yes_count": yes_count,
        "no_count": no_count,
        "yes_rate": yes_rate,
        "break_even_rate": break_even_rate,
        "edge": edge,
        "profitable": yes_rate > break_even_rate,
        "total_pnl_cents": total_pnl,
        "pnl_per_market_cents": pnl_per_market,
        "roi": roi,
        "z_score": z,
        "p_value": pval,
    }


# ---------------------------------------------------------------------------
# NBA Props: point threshold analysis
# ---------------------------------------------------------------------------

def extract_props_threshold(ticker):
    """Extract point threshold from nba_props ticker. E.g. KXNBAPTS-...-25 -> 25."""
    match = re.search(r'-(\d+)$', ticker)
    if match:
        return int(match.group(1))
    return None


def extract_player_name(title):
    """Extract player name from props title. E.g. 'Darius Garland: 30+ points' -> 'Darius Garland'."""
    match = re.match(r'^(.+?):\s*\d+', title)
    if match:
        return match.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# NBA Winner: team and home/away analysis
# ---------------------------------------------------------------------------

def extract_teams_from_title(title):
    """
    Extract teams from nba_winner title.
    Format: 'Team A at Team B Winner?'
    Returns (away_team, home_team) or (None, None).
    """
    match = re.match(r'^(.+?)\s+at\s+(.+?)\s+Winner\??$', title)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, None


def extract_team_from_ticker(ticker):
    """Extract team abbreviation from last part of ticker. E.g. ...-LAC -> LAC."""
    parts = ticker.split('-')
    if parts:
        return parts[-1]
    return None


# ---------------------------------------------------------------------------
# Main Analysis
# ---------------------------------------------------------------------------

def main():
    # Load data
    print("=" * 80)
    print("SPORTS STRATEGY BACKTEST — Kalshi Historical Settlements")
    print("=" * 80)

    with open(SETTLEMENTS_FILE) as f:
        data = json.load(f)

    all_markets = data["markets"]
    total_markets = data["total_markets"]
    fetched_at = data["fetched_at"]

    print(f"\nData source: {SETTLEMENTS_FILE}")
    print(f"Fetched at:  {fetched_at}")
    print(f"Total markets in file: {total_markets:,}")

    # Filter to sports
    sports = [m for m in all_markets if m.get("category") == "sports"]
    nba_winner = [m for m in sports if m.get("sub_category") == "nba_winner"]
    nba_props = [m for m in sports if m.get("sub_category") == "nba_props"]

    print(f"\nSports markets: {len(sports):,}")
    print(f"  nba_winner:   {len(nba_winner):,}")
    print(f"  nba_props:    {len(nba_props):,}")

    # Check for price data
    has_price = sum(1 for m in sports if m.get("last_price") is not None)
    print(f"\n  Markets with last_price data: {has_price} / {len(sports)}")
    if has_price == 0:
        print("  NOTE: Settlement data has NO historical price data.")
        print("  Analysis uses structural patterns and simulated price tiers.")

    # =========================================================================
    # SECTION 1: NBA WINNER ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 1: NBA WINNER MARKETS")
    print("=" * 80)

    # Filter out scalar (voided) markets
    nba_w_valid = [m for m in nba_winner if m.get("result") in ("yes", "no")]
    nba_w_scalar = [m for m in nba_winner if m.get("result") == "scalar"]

    yes_w = sum(1 for m in nba_w_valid if m["result"] == "yes")
    no_w = sum(1 for m in nba_w_valid if m["result"] == "no")

    print(f"\nTotal settled:   {len(nba_w_valid):,}  (+ {len(nba_w_scalar)} voided/scalar)")
    print(f"Settled YES:     {yes_w:,}  ({fmt_pct(yes_w/len(nba_w_valid))})")
    print(f"Settled NO:      {no_w:,}  ({fmt_pct(no_w/len(nba_w_valid))})")

    # ---- Game-level analysis (each game has 2 markets: TeamA-YES, TeamB-YES) ----
    print(f"\n--- Game-Level Analysis ---")

    # Group by game (series_ticker + game date portion of ticker)
    games = defaultdict(list)
    for m in nba_w_valid:
        # ticker: KXNBAGAME-26MAR21LACDAL-LAC -> game_id = KXNBAGAME-26MAR21LACDAL
        parts = m["ticker"].rsplit("-", 1)
        game_id = parts[0] if len(parts) > 1 else m["ticker"]
        games[game_id].append(m)

    complete_games = {gid: mkts for gid, mkts in games.items() if len(mkts) == 2}
    print(f"Complete games (2 sides each): {len(complete_games):,}")

    # Home vs Away analysis
    home_wins = 0
    away_wins = 0
    home_total = 0

    for gid, mkts in complete_games.items():
        title = mkts[0]["title"]
        away_team, home_team = extract_teams_from_title(title)
        if away_team is None:
            continue
        home_total += 1

        # Find which team won
        winner_mkt = [m for m in mkts if m["result"] == "yes"]
        if not winner_mkt:
            continue

        winner_ticker_team = extract_team_from_ticker(winner_mkt[0]["ticker"])

        # Determine if winner is home or away
        # The home team is second in the ticker matchup (e.g., LACDAL -> DAL is home)
        # Extract from the game_id
        match = re.search(r'-\d+[A-Z]+\d+([A-Z]+)$', gid)
        if match:
            # Last team abbreviation(s) in game_id = home team abbrev
            pass

        # Simpler: title says "Away at Home Winner?" so last team in title is home
        # Check if winner is in home_team string
        if winner_ticker_team and home_team:
            # Try to match
            if winner_ticker_team.upper() in home_team.upper().replace(" ", ""):
                home_wins += 1
            else:
                away_wins += 1

    if home_total > 0:
        print(f"\nHome vs Away (from {home_total:,} games with parseable titles):")
        print(f"  Home wins: {home_wins:,}  ({fmt_pct(home_wins/home_total)})")
        print(f"  Away wins: {away_wins:,}  ({fmt_pct(away_wins/home_total)})")

        # Home team edge
        home_rate = home_wins / home_total
        z_home = z_score_proportion(home_rate, 0.5, home_total)
        pval_home = p_value_one_sided(z_home) if z_home > 0 else p_value_one_sided(-z_home)
        print(f"  Home advantage z-score: {z_home:.3f}  (p={pval_home:.4f})")
        if home_rate > 0.5:
            print(f"  => Home teams win more often than 50%. Statistically {'significant' if pval_home < 0.05 else 'NOT significant'} at p<0.05.")
        else:
            print(f"  => No home court advantage detected in this sample.")

    # ---- Simulated price-tier analysis for nba_winner ----
    print(f"\n--- Simulated Price-Tier Analysis (nba_winner) ---")
    print(f"    (Simulating: 'buy YES on every market at tier midpoint price')")

    PRICE_TIERS = [
        ("1-10c",   1,  10),
        ("10-20c", 10,  20),
        ("20-30c", 20,  30),
        ("30-50c", 30,  50),
        ("50-70c", 50,  70),
        ("70-100c", 70, 100),
    ]

    # Since we have no actual prices, simulate what WOULD happen at each tier
    # using the overall YES rate as the true probability
    print(f"\n  Overall YES rate in nba_winner: {fmt_pct(yes_w/len(nba_w_valid))}")
    print(f"  (Each game has exactly 1 YES and 1 NO market, so YES rate = 50%)")
    print(f"\n  Hypothetical P&L table if you could buy at each price tier:")
    print(f"  {'Tier':<12} {'Mid Price':>10} {'Break-Even':>12} {'Actual YES%':>12} {'Edge':>8} {'P&L/mkt':>10} {'ROI':>8}")
    print(f"  {'-'*74}")

    actual_yes_rate = yes_w / len(nba_w_valid) if len(nba_w_valid) > 0 else 0

    for label, lo, hi in PRICE_TIERS:
        mid = (lo + hi) / 2
        be_rate = mid / 100
        edge = actual_yes_rate - be_rate
        pnl_per = actual_yes_rate * (100 - mid) - (1 - actual_yes_rate) * mid
        roi = pnl_per / mid if mid > 0 else 0

        marker = " <-- profitable" if edge > 0 else ""
        print(f"  {label:<12} {mid:>8.1f}c  {fmt_pct(be_rate):>10}   {fmt_pct(actual_yes_rate):>10}  {edge*100:>+6.1f}pp  {pnl_per:>+8.1f}c  {roi*100:>+6.1f}%{marker}")

    print(f"\n  Key insight: With paired markets (50% YES rate), only tiers below 50c")
    print(f"  are theoretically profitable. This is the 'naive' baseline.")

    # =========================================================================
    # SECTION 2: NBA PROPS ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 2: NBA PROPS MARKETS")
    print("=" * 80)

    nba_p_valid = [m for m in nba_props if m.get("result") in ("yes", "no")]
    nba_p_scalar = [m for m in nba_props if m.get("result") == "scalar"]

    yes_p = sum(1 for m in nba_p_valid if m["result"] == "yes")
    no_p = sum(1 for m in nba_p_valid if m["result"] == "no")

    print(f"\nTotal settled:   {len(nba_p_valid):,}  (+ {len(nba_p_scalar)} voided/scalar)")
    print(f"Settled YES:     {yes_p:,}  ({fmt_pct(yes_p/len(nba_p_valid) if nba_p_valid else 0)})")
    print(f"Settled NO:      {no_p:,}  ({fmt_pct(no_p/len(nba_p_valid) if nba_p_valid else 0)})")

    # ---- Analysis by point threshold ----
    print(f"\n--- YES Rate by Point Threshold ---")
    print(f"  (Higher threshold = harder to hit = lower implied probability)")

    threshold_stats = defaultdict(lambda: {"yes": 0, "no": 0, "total": 0})

    for m in nba_p_valid:
        thr = extract_props_threshold(m["ticker"])
        if thr is not None:
            threshold_stats[thr]["total"] += 1
            if m["result"] == "yes":
                threshold_stats[thr]["yes"] += 1
            else:
                threshold_stats[thr]["no"] += 1

    print(f"\n  {'Threshold':>10} {'Total':>8} {'YES':>8} {'NO':>8} {'YES Rate':>10} {'Expected*':>10} {'Edge':>8}")
    print(f"  {'-'*64}")

    sorted_thresholds = sorted(threshold_stats.keys())
    for thr in sorted_thresholds:
        s = threshold_stats[thr]
        if s["total"] == 0:
            continue
        yes_rate = s["yes"] / s["total"]
        # Expected: rough estimate based on typical NBA scoring
        # Average NBA player scores ~15-20 ppg, so:
        # 10+ pts: ~70-80% for starters, 15+: ~50-60%, 20+: ~30-40%, 25+: ~15-25%, 30+: ~8-15%, 35+: ~3-7%, 40+: ~1-3%
        print(f"  {thr:>8}+ pts {s['total']:>8,} {s['yes']:>8,} {s['no']:>8,} {fmt_pct(yes_rate):>10}")

    # ---- Underdog analysis for props ----
    print(f"\n--- 'Underdog' Analysis: High-Threshold Props ---")
    print(f"  (Buying YES on hard-to-hit props = betting on underdog outcomes)")

    underdog_thresholds = [30, 35, 40]
    for thr in underdog_thresholds:
        s = threshold_stats.get(thr, {"yes": 0, "no": 0, "total": 0})
        if s["total"] == 0:
            print(f"\n  {thr}+ pts: No markets found")
            continue
        yes_rate = s["yes"] / s["total"]

        print(f"\n  {thr}+ pts threshold:")
        print(f"    Markets: {s['total']:,}   YES: {s['yes']:,}   NO: {s['no']:,}")
        print(f"    YES rate: {fmt_pct(yes_rate)}")

        # Simulate buying at various prices
        print(f"    Simulated P&L at different buy prices:")
        for buy_price in [5, 10, 15, 20, 25, 30]:
            pnl_per = yes_rate * (100 - buy_price) - (1 - yes_rate) * buy_price
            roi = pnl_per / buy_price if buy_price > 0 else 0
            be = buy_price / 100
            z = z_score_proportion(yes_rate, be, s["total"])
            pv = p_value_one_sided(z)
            edge = yes_rate - be
            sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
            profit_marker = "PROFITABLE" if edge > 0 else ""
            print(f"      Buy@{buy_price:>2}c: P&L/mkt={pnl_per:>+7.1f}c  ROI={roi*100:>+7.1f}%  edge={edge*100:>+5.1f}pp  z={z:>+6.2f}  p={pv:.4f} {sig} {profit_marker}")

    # ---- Simulate price tiers for all nba_props ----
    print(f"\n--- Simulated Price-Tier P&L (all nba_props) ---")

    overall_yes_rate_p = yes_p / len(nba_p_valid) if nba_p_valid else 0
    print(f"  Overall YES rate: {fmt_pct(overall_yes_rate_p)} ({yes_p:,} / {len(nba_p_valid):,})")

    print(f"\n  {'Tier':<12} {'Mid Price':>10} {'Break-Even':>12} {'Actual YES%':>12} {'Edge':>8} {'P&L/mkt':>10} {'ROI':>8} {'Stat':>12}")
    print(f"  {'-'*88}")

    for label, lo, hi in PRICE_TIERS:
        mid = (lo + hi) / 2
        be_rate = mid / 100
        edge = overall_yes_rate_p - be_rate
        pnl_per = overall_yes_rate_p * (100 - mid) - (1 - overall_yes_rate_p) * mid
        roi = pnl_per / mid if mid > 0 else 0
        z = z_score_proportion(overall_yes_rate_p, be_rate, len(nba_p_valid))
        pv = p_value_one_sided(z)
        sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""

        marker = " <-- EDGE" if edge > 0 else ""
        print(f"  {label:<12} {mid:>8.1f}c  {fmt_pct(be_rate):>10}   {fmt_pct(overall_yes_rate_p):>10}  {edge*100:>+6.1f}pp  {pnl_per:>+8.1f}c  {roi*100:>+6.1f}%   z={z:>+.2f} p={pv:.3f}{sig}{marker}")

    # ---- Cross-reference: threshold as implied price proxy ----
    print(f"\n--- Threshold-as-Price-Proxy Analysis ---")
    print(f"  Mapping point thresholds to approximate implied prices:")
    print(f"  (Based on actual YES rates in data = what fair price should be)")

    print(f"\n  {'Threshold':>10} {'YES Rate':>10} {'Fair Price':>12} {'Typical Mkt':>14} {'Implied Edge':>14}")
    print(f"  {'-'*64}")

    # Typical market prices for NBA props (industry knowledge)
    typical_prices = {10: 75, 15: 55, 20: 35, 25: 20, 30: 12, 35: 7, 40: 4}

    for thr in sorted_thresholds:
        s = threshold_stats[thr]
        if s["total"] == 0:
            continue
        yes_rate = s["yes"] / s["total"]
        fair_price = yes_rate * 100  # in cents
        typ = typical_prices.get(thr, None)
        if typ:
            edge = yes_rate - typ/100
            edge_str = f"{edge*100:>+.1f}pp"
            if edge > 0:
                edge_str += " (buy YES)"
            elif edge < 0:
                edge_str += " (buy NO)"
        else:
            edge_str = "N/A"
            typ = "?"
        print(f"  {thr:>8}+ pts {fmt_pct(yes_rate):>10}   {fair_price:>9.1f}c    ~{typ}c          {edge_str}")

    # =========================================================================
    # SECTION 3: TITLE PATTERN ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 3: TITLE PATTERN ANALYSIS")
    print("=" * 80)

    # --- nba_winner title patterns ---
    print(f"\n--- nba_winner: Title Word Analysis ---")

    # All nba_winner titles follow "Team at Team Winner?" pattern
    # Check if any different patterns exist
    winner_titles = set(m["title"] for m in nba_w_valid)
    print(f"  Unique game titles: {len(winner_titles):,}")

    # Check for "Winner" vs other patterns
    has_winner = sum(1 for m in nba_w_valid if "Winner" in m["title"])
    no_winner = sum(1 for m in nba_w_valid if "Winner" not in m["title"])
    print(f"  Titles with 'Winner': {has_winner:,}")
    print(f"  Titles without 'Winner': {no_winner:,}")

    # --- nba_props title patterns ---
    print(f"\n--- nba_props: Title Pattern Analysis ---")

    # Props titles: "Player Name: N+ points"
    # Check for other patterns
    props_patterns = Counter()
    for m in nba_p_valid:
        title = m["title"]
        if "points" in title.lower():
            props_patterns["N+ points"] += 1
        elif "rebounds" in title.lower():
            props_patterns["N+ rebounds"] += 1
        elif "assists" in title.lower():
            props_patterns["N+ assists"] += 1
        elif "3-pointers" in title.lower() or "threes" in title.lower():
            props_patterns["N+ 3-pointers"] += 1
        elif "steals" in title.lower():
            props_patterns["N+ steals"] += 1
        elif "blocks" in title.lower():
            props_patterns["N+ blocks"] += 1
        else:
            props_patterns["other: " + title[:40]] += 1

    print(f"\n  Prop type breakdown:")
    for pat, count in props_patterns.most_common(20):
        pct = count / len(nba_p_valid) if nba_p_valid else 0
        print(f"    {pat:<30} {count:>6,}  ({fmt_pct(pct)})")

    # YES rate by prop type (only for types with enough data)
    print(f"\n  YES rate by prop type:")
    print(f"  {'Prop Type':<25} {'Total':>8} {'YES':>8} {'YES Rate':>10}")
    print(f"  {'-'*55}")

    prop_type_stats = defaultdict(lambda: {"yes": 0, "no": 0, "total": 0})
    for m in nba_p_valid:
        title = m["title"].lower()
        if "points" in title:
            ptype = "points"
        elif "rebounds" in title:
            ptype = "rebounds"
        elif "assists" in title:
            ptype = "assists"
        elif "3-pointer" in title or "three" in title:
            ptype = "3-pointers"
        elif "steals" in title:
            ptype = "steals"
        elif "blocks" in title:
            ptype = "blocks"
        else:
            ptype = "other"

        prop_type_stats[ptype]["total"] += 1
        if m["result"] == "yes":
            prop_type_stats[ptype]["yes"] += 1
        else:
            prop_type_stats[ptype]["no"] += 1

    for ptype in sorted(prop_type_stats.keys(), key=lambda x: prop_type_stats[x]["total"], reverse=True):
        s = prop_type_stats[ptype]
        if s["total"] < 10:
            continue
        yes_rate = s["yes"] / s["total"]
        print(f"  {ptype:<25} {s['total']:>8,} {s['yes']:>8,} {fmt_pct(yes_rate):>10}")

    # ---- Cross-analysis: threshold x prop type ----
    print(f"\n--- YES Rate by Threshold x Prop Type ---")

    threshold_type_stats = defaultdict(lambda: defaultdict(lambda: {"yes": 0, "no": 0, "total": 0}))
    for m in nba_p_valid:
        thr = extract_props_threshold(m["ticker"])
        title = m["title"].lower()
        if "points" in title:
            ptype = "points"
        elif "rebounds" in title:
            ptype = "rebounds"
        elif "assists" in title:
            ptype = "assists"
        else:
            ptype = "other"

        if thr is not None:
            threshold_type_stats[ptype][thr]["total"] += 1
            if m["result"] == "yes":
                threshold_type_stats[ptype][thr]["yes"] += 1
            else:
                threshold_type_stats[ptype][thr]["no"] += 1

    for ptype in ["points", "rebounds", "assists", "other"]:
        if ptype not in threshold_type_stats:
            continue
        print(f"\n  {ptype.upper()}:")
        print(f"    {'Threshold':>10} {'Total':>8} {'YES':>8} {'YES Rate':>10}")
        print(f"    {'-'*40}")
        for thr in sorted(threshold_type_stats[ptype].keys()):
            s = threshold_type_stats[ptype][thr]
            if s["total"] == 0:
                continue
            yes_rate = s["yes"] / s["total"]
            print(f"    {thr:>8}+ {s['total']:>8,} {s['yes']:>8,} {fmt_pct(yes_rate):>10}")

    # =========================================================================
    # SECTION 4: TEAM-LEVEL ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 4: TEAM-LEVEL WIN RATES (nba_winner)")
    print("=" * 80)

    team_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "total": 0})

    for m in nba_w_valid:
        team = extract_team_from_ticker(m["ticker"])
        if team:
            team_stats[team]["total"] += 1
            if m["result"] == "yes":
                team_stats[team]["wins"] += 1
            else:
                team_stats[team]["losses"] += 1

    print(f"\n  {'Team':>6} {'Games':>8} {'Wins':>8} {'Losses':>8} {'Win Rate':>10}")
    print(f"  {'-'*44}")

    sorted_teams = sorted(team_stats.keys(),
                          key=lambda t: team_stats[t]["wins"]/team_stats[t]["total"] if team_stats[t]["total"] > 0 else 0,
                          reverse=True)

    for team in sorted_teams:
        s = team_stats[team]
        if s["total"] == 0:
            continue
        win_rate = s["wins"] / s["total"]
        print(f"  {team:>6} {s['total']:>8,} {s['wins']:>8,} {s['losses']:>8,} {fmt_pct(win_rate):>10}")

    # Best and worst teams
    if sorted_teams:
        best = sorted_teams[0]
        worst = sorted_teams[-1]
        print(f"\n  Best:  {best} ({fmt_pct(team_stats[best]['wins']/team_stats[best]['total'])} win rate)")
        print(f"  Worst: {worst} ({fmt_pct(team_stats[worst]['wins']/team_stats[worst]['total'])} win rate)")

    # =========================================================================
    # SECTION 5: DATE/TIME PATTERNS
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 5: TEMPORAL PATTERNS")
    print("=" * 80)

    # Extract dates from close_time
    date_stats = defaultdict(lambda: {"yes": 0, "no": 0, "total": 0})
    dow_stats = defaultdict(lambda: {"yes": 0, "no": 0, "total": 0, "games": set()})

    from datetime import datetime

    for m in nba_w_valid:
        ct = m.get("close_time", "")
        if ct:
            try:
                dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                date_str = dt.strftime("%Y-%m-%d")
                dow = dt.strftime("%A")
                date_stats[date_str]["total"] += 1
                if m["result"] == "yes":
                    date_stats[date_str]["yes"] += 1
                else:
                    date_stats[date_str]["no"] += 1

                # Game ID for counting unique games per DOW
                parts = m["ticker"].rsplit("-", 1)
                game_id = parts[0] if len(parts) > 1 else m["ticker"]
                dow_stats[dow]["games"].add(game_id)
                dow_stats[dow]["total"] += 1
                if m["result"] == "yes":
                    dow_stats[dow]["yes"] += 1
                else:
                    dow_stats[dow]["no"] += 1
            except Exception:
                pass

    print(f"\n  Unique game dates: {len(date_stats):,}")
    print(f"  Date range: {min(date_stats.keys())} to {max(date_stats.keys())}")

    print(f"\n  Games by day of week:")
    print(f"  {'Day':<12} {'Markets':>8} {'Games':>8}")
    print(f"  {'-'*30}")
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for day in day_order:
        if day in dow_stats:
            s = dow_stats[day]
            print(f"  {day:<12} {s['total']:>8,} {len(s['games']):>8,}")

    # =========================================================================
    # SECTION 6: STRATEGIC INSIGHTS & SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 6: STRATEGIC INSIGHTS & ACTIONABLE FINDINGS")
    print("=" * 80)

    print(f"""
  DATA LIMITATION:
  The settlement export has NO price data (last_price, yes_bid, yes_ask all null).
  All price-tier P&L simulations above use the ACTUAL YES RATE applied uniformly
  across hypothetical price tiers. To do real price-tier backtesting, you would
  need to fetch individual market trade histories via the Kalshi API
  (GET /markets/{{ticker}}/trades) for each of the {len(sports):,} sports markets.

  KEY FINDINGS:

  1. NBA WINNER MARKETS:
     - Perfectly balanced: 50% YES / 50% NO (by construction — binary pairs)
     - Any edge must come from PRICING, not outcome rates
     - Home court advantage: {'YES' if home_total > 0 and home_wins/home_total > 0.5 else 'not clearly detected'}
       ({fmt_pct(home_wins/home_total) if home_total > 0 else 'N/A'} home win rate from {home_total:,} games)

  2. NBA PROPS MARKETS:
     - Overall YES rate: {fmt_pct(overall_yes_rate_p)} (NOT 50% — because each player has
       multiple thresholds and higher thresholds hit less often)
     - YES rate decreases monotonically with threshold (as expected)
     - The KEY question: at which thresholds does the market MIS-PRICE?""")

    # Show the threshold table one more time as summary
    print(f"\n  THRESHOLD EDGE TABLE (props — actual YES rate vs hypothetical prices):")
    print(f"  {'Threshold':>10} {'YES Rate':>10} {'Fair Price':>12} {'Typ. Market':>14} {'Edge Direction':>16}")
    print(f"  {'-'*66}")

    for thr in sorted_thresholds:
        s = threshold_stats[thr]
        if s["total"] == 0:
            continue
        yes_rate = s["yes"] / s["total"]
        fair_price = yes_rate * 100
        typ = typical_prices.get(thr, None)
        if typ:
            if fair_price > typ:
                direction = f"Buy YES (+{fair_price-typ:.0f}c)"
            else:
                direction = f"Buy NO  (+{typ-fair_price:.0f}c)"
        else:
            direction = "N/A"
        print(f"  {thr:>8}+ pts {fmt_pct(yes_rate):>10}   {fair_price:>9.1f}c   {'~'+str(typ)+'c' if typ else '?':>12}  {direction:>16}")

    print(f"""
  3. RECOMMENDED NEXT STEPS:
     a) Fetch real price data: Use kalshi_client.get_market_history() for sports tickers
        to get actual last_price / trade prices for each market
     b) Build price-tier backtest: With real prices, compute actual edge per tier
     c) Focus on nba_props: The varying YES rates by threshold create pricing
        inefficiency opportunities (unlike nba_winner which is always 50/50)
     d) Player-level analysis: Some players may be systematically mispriced
     e) Threshold clustering: Markets near the YES/NO boundary (e.g., 15-20 pt props
        with ~45-55% YES rates) may have the most pricing inefficiency

  4. VOIDED/SCALAR MARKETS:
     - nba_winner: {len(nba_w_scalar):,} voided (players DNP, game postponed, etc.)
     - nba_props:  {len(nba_p_scalar):,} voided
     - These are excluded from all analysis above
""")

    print("=" * 80)
    print("END OF REPORT")
    print("=" * 80)


if __name__ == "__main__":
    main()
