#!/usr/bin/env python3
"""
Weather Strategy Backtest
=========================
Analyzes 35K+ settled Kalshi weather markets to find structural edges.

Since no price data exists (all last_price=null, volume=0), we pivot to:
1. Settlement pattern analysis per city
2. Bucket position analysis (where does the actual temp land?)
3. Tail bucket analysis (extreme above/below buckets settle YES less often = potential fade targets)
4. Simulated strategies based on structural patterns
5. "Fair value" estimation from historical settlement rates per bucket position
"""

import json
import sys
from collections import defaultdict
from datetime import datetime

DATA_PATH = "/Users/benjamin/Desktop/ippo/output/historical_settlements.json"


def load_data():
    with open(DATA_PATH) as f:
        data = json.load(f)
    return data["markets"]


def filter_weather(markets):
    return [m for m in markets if m.get("category") == "weather"]


def extract_date(ticker):
    """Extract date string from ticker like KXHIGHNY-26MAR21-T60 -> 26MAR21"""
    parts = ticker.split("-")
    if len(parts) >= 2:
        return parts[1]
    return None


def extract_series_date(ticker):
    """Extract series+date from ticker: KXHIGHNY-26MAR21-T60 -> KXHIGHNY-26MAR21"""
    parts = ticker.split("-")
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    return None


def analyze_city(city, markets):
    """Comprehensive analysis for a single city."""
    total = len(markets)
    yes_count = sum(1 for m in markets if m["result"] == "yes")
    no_count = total - yes_count

    # Group by series_date to get individual "events" (one day's set of buckets)
    events = defaultdict(list)
    for m in markets:
        sd = extract_series_date(m["ticker"])
        if sd:
            events[sd].append(m)

    # Bucket type analysis
    bucket_types = defaultdict(lambda: {"total": 0, "yes": 0})
    for m in markets:
        bt = m["bucket"]["type"]
        bucket_types[bt]["total"] += 1
        if m["result"] == "yes":
            bucket_types[bt]["yes"] += 1

    # Bucket position analysis: for "bucket" type, analyze by midpoint
    # Group buckets by their midpoint relative to the range of the day
    bucket_midpoints_yes = []
    bucket_midpoints_no = []
    for m in markets:
        if m["bucket"]["type"] == "bucket":
            mid = (m["bucket"]["low"] + m["bucket"]["high"]) / 2
            if m["result"] == "yes":
                bucket_midpoints_yes.append(mid)
            else:
                bucket_midpoints_no.append(mid)

    # For each event, find the winning bucket position (rank within the day)
    # This tells us: does the actual temp tend to land in the middle or edges?
    winning_positions = []
    total_buckets_per_event = []
    winning_bucket_details = []

    for sd, event_markets in events.items():
        sorted_buckets = sorted(event_markets, key=lambda m: m["bucket"].get("low", 0))
        n = len(sorted_buckets)
        total_buckets_per_event.append(n)
        for i, m in enumerate(sorted_buckets):
            if m["result"] == "yes":
                position_pct = i / (n - 1) if n > 1 else 0.5  # 0=lowest, 1=highest
                winning_positions.append(position_pct)
                winning_bucket_details.append({
                    "series_date": sd,
                    "position": i,
                    "total": n,
                    "position_pct": position_pct,
                    "bucket_type": m["bucket"]["type"],
                    "bucket_low": m["bucket"].get("low"),
                    "bucket_high": m["bucket"].get("high"),
                })

    # Tail analysis: how often do extreme buckets (above/below) win?
    above_markets = [m for m in markets if m["bucket"]["type"] == "above"]
    below_markets = [m for m in markets if m["bucket"]["type"] == "below"]
    above_yes = sum(1 for m in above_markets if m["result"] == "yes")
    below_yes = sum(1 for m in below_markets if m["result"] == "yes")

    # Position quartile analysis
    q1 = [p for p in winning_positions if p <= 0.25]
    q2 = [p for p in winning_positions if 0.25 < p <= 0.5]
    q3 = [p for p in winning_positions if 0.5 < p <= 0.75]
    q4 = [p for p in winning_positions if p > 0.75]

    # Simulated strategy: "Fade the tails"
    # Buy NO on above bucket + Buy NO on below bucket for every event
    # Cost: assume we buy at fair value based on historical YES rate
    fade_above_pnl = 0
    fade_below_pnl = 0
    for m in above_markets:
        # If we buy NO (i.e., we bet it won't be above the top threshold)
        # Fair price of YES = historical above_yes_rate
        # We pay (1 - fair_yes_rate) for NO contract... but since no prices,
        # let's simulate buying NO at 50c (naive) and see raw win rate
        if m["result"] == "no":
            fade_above_pnl += 50  # win $0.50 (paid 50c, get $1)
        else:
            fade_above_pnl -= 50  # lose our 50c

    for m in below_markets:
        if m["result"] == "no":
            fade_below_pnl += 50
        else:
            fade_below_pnl -= 50

    # Simulated strategy: "Buy the middle"
    # For each event, buy YES on the 2 middle buckets
    middle_buy_pnl = 0
    middle_buy_trades = 0
    middle_buy_wins = 0
    for sd, event_markets in events.items():
        sorted_b = sorted(event_markets, key=lambda m: m["bucket"].get("low", 0))
        n = len(sorted_b)
        if n < 3:
            continue
        # Middle indices
        mid_start = n // 2 - 1
        mid_end = n // 2 + 1
        for i in range(max(0, mid_start), min(n, mid_end)):
            middle_buy_trades += 1
            if sorted_b[i]["result"] == "yes":
                middle_buy_wins += 1
                middle_buy_pnl += 50  # paid 50c, get $1
            else:
                middle_buy_pnl -= 50  # lose 50c

    # Simulated strategy: "Bucket position edge"
    # For each bucket, compute empirical YES rate by its position rank
    position_stats = defaultdict(lambda: {"total": 0, "yes": 0})
    for sd, event_markets in events.items():
        sorted_b = sorted(event_markets, key=lambda m: m["bucket"].get("low", 0))
        n = len(sorted_b)
        for i, m in enumerate(sorted_b):
            pos_key = f"{i}/{n}"  # e.g., "0/6" = lowest of 6
            position_stats[pos_key]["total"] += 1
            if m["result"] == "yes":
                position_stats[pos_key]["yes"] += 1

    # Temperature range analysis: what's the typical winning temp range?
    winning_temps = []
    for detail in winning_bucket_details:
        if detail["bucket_type"] == "bucket":
            mid = (detail["bucket_low"] + detail["bucket_high"]) / 2
            winning_temps.append(mid)

    return {
        "total": total,
        "yes_count": yes_count,
        "no_count": no_count,
        "yes_rate": yes_count / total if total > 0 else 0,
        "num_events": len(events),
        "avg_buckets_per_event": sum(total_buckets_per_event) / len(total_buckets_per_event) if total_buckets_per_event else 0,
        "bucket_types": dict(bucket_types),
        "above_total": len(above_markets),
        "above_yes": above_yes,
        "above_yes_rate": above_yes / len(above_markets) if above_markets else 0,
        "below_total": len(below_markets),
        "below_yes": below_yes,
        "below_yes_rate": below_yes / len(below_markets) if below_markets else 0,
        "winning_positions": winning_positions,
        "position_quartiles": {
            "Q1 (0-25%)": len(q1),
            "Q2 (25-50%)": len(q2),
            "Q3 (50-75%)": len(q3),
            "Q4 (75-100%)": len(q4),
        },
        "position_stats": dict(position_stats),
        "fade_above_pnl_cents": fade_above_pnl,
        "fade_below_pnl_cents": fade_below_pnl,
        "middle_buy_trades": middle_buy_trades,
        "middle_buy_wins": middle_buy_wins,
        "middle_buy_win_rate": middle_buy_wins / middle_buy_trades if middle_buy_trades > 0 else 0,
        "middle_buy_pnl_cents": middle_buy_pnl,
        "winning_temps": winning_temps,
    }


def print_report(city, stats):
    """Print formatted report for a city."""
    print(f"\n{'='*80}")
    print(f"  CITY: {city}")
    print(f"{'='*80}")

    print(f"\n--- Overview ---")
    print(f"  Total settled markets:     {stats['total']:,}")
    print(f"  Unique events (days):      {stats['num_events']:,}")
    print(f"  Avg buckets per event:     {stats['avg_buckets_per_event']:.1f}")
    print(f"  YES settlements:           {stats['yes_count']:,} ({stats['yes_rate']:.1%})")
    print(f"  NO settlements:            {stats['no_count']:,} ({1 - stats['yes_rate']:.1%})")

    print(f"\n--- Tail Bucket Analysis ---")
    print(f"  ABOVE bucket (temp exceeds top threshold):")
    print(f"    Total: {stats['above_total']:,}  |  YES: {stats['above_yes']:,}  |  YES rate: {stats['above_yes_rate']:.1%}")
    if stats['above_total'] > 0:
        implied_no_edge = (1 - stats['above_yes_rate']) - 0.50
        print(f"    -> If you always buy NO at 50c: edge = {implied_no_edge:+.1%}")

    print(f"  BELOW bucket (temp below bottom threshold):")
    print(f"    Total: {stats['below_total']:,}  |  YES: {stats['below_yes']:,}  |  YES rate: {stats['below_yes_rate']:.1%}")
    if stats['below_total'] > 0:
        implied_no_edge = (1 - stats['below_yes_rate']) - 0.50
        print(f"    -> If you always buy NO at 50c: edge = {implied_no_edge:+.1%}")

    print(f"\n--- Winning Position Distribution ---")
    print(f"  Where does the actual temp land in the bucket lineup?")
    print(f"  (0% = lowest bucket, 100% = highest bucket)")
    for label, count in stats["position_quartiles"].items():
        total_events = stats["num_events"]
        pct = count / total_events if total_events > 0 else 0
        bar = "#" * int(pct * 40)
        print(f"    {label:15s}: {count:4d} events ({pct:.1%}) {bar}")

    if stats["winning_positions"]:
        avg_pos = sum(stats["winning_positions"]) / len(stats["winning_positions"])
        print(f"    Average winning position: {avg_pos:.1%}")

    print(f"\n--- Bucket Position Fair Values ---")
    print(f"  Empirical YES rate by position (= fair value for YES contract):")
    # Sort position stats
    sorted_positions = sorted(
        stats["position_stats"].items(),
        key=lambda x: (int(x[0].split("/")[1]), int(x[0].split("/")[0]))
    )
    # Group by bucket count
    current_n = None
    for pos_key, pos_data in sorted_positions:
        pos_i, pos_n = pos_key.split("/")
        if pos_n != current_n:
            current_n = pos_n
            print(f"\n    {pos_n}-bucket events:")
        yes_rate = pos_data["yes"] / pos_data["total"] if pos_data["total"] > 0 else 0
        bar = "#" * int(yes_rate * 40)
        label = "BELOW" if int(pos_i) == 0 else ("ABOVE" if int(pos_i) == int(pos_n) - 1 else f"  MID-{pos_i}")
        print(f"      Pos {pos_i} ({label:8s}): {yes_rate:5.1%} YES  ({pos_data['yes']:3d}/{pos_data['total']:3d})  {bar}")

    print(f"\n--- Simulated Strategies (at naive 50c entry) ---")

    print(f"\n  Strategy 1: FADE THE ABOVE TAIL (buy NO on 'above' bucket every event)")
    win_rate_above = 1 - stats['above_yes_rate'] if stats['above_total'] > 0 else 0
    print(f"    Trades: {stats['above_total']:,}  |  Win rate: {win_rate_above:.1%}")
    print(f"    P&L at 50c entry: ${stats['fade_above_pnl_cents'] / 100:,.2f}")

    print(f"\n  Strategy 2: FADE THE BELOW TAIL (buy NO on 'below' bucket every event)")
    win_rate_below = 1 - stats['below_yes_rate'] if stats['below_total'] > 0 else 0
    print(f"    Trades: {stats['below_total']:,}  |  Win rate: {win_rate_below:.1%}")
    print(f"    P&L at 50c entry: ${stats['fade_below_pnl_cents'] / 100:,.2f}")

    print(f"\n  Strategy 3: BUY THE MIDDLE (buy YES on 2 middle buckets each event)")
    print(f"    Trades: {stats['middle_buy_trades']:,}  |  Wins: {stats['middle_buy_wins']:,}  |  Win rate: {stats['middle_buy_win_rate']:.1%}")
    print(f"    P&L at 50c entry: ${stats['middle_buy_pnl_cents'] / 100:,.2f}")

    # Breakeven analysis
    if stats['above_total'] > 0:
        print(f"\n--- Breakeven Prices ---")
        above_fair = stats['above_yes_rate']
        below_fair = stats['below_yes_rate']
        print(f"  ABOVE bucket fair YES price: {above_fair:.0%} ({above_fair*100:.1f}c)")
        print(f"    -> Buy NO if NO price < {1-above_fair:.0%} ({(1-above_fair)*100:.1f}c) = EDGE")
        print(f"    -> Buy YES only if YES price < {above_fair:.0%} ({above_fair*100:.1f}c)")
        print(f"  BELOW bucket fair YES price: {below_fair:.0%} ({below_fair*100:.1f}c)")
        print(f"    -> Buy NO if NO price < {1-below_fair:.0%} ({(1-below_fair)*100:.1f}c) = EDGE")
        print(f"    -> Buy YES only if YES price < {below_fair:.0%} ({below_fair*100:.1f}c)")

    # Winning temperature stats
    if stats["winning_temps"]:
        wt = stats["winning_temps"]
        print(f"\n--- Winning Temperature Stats ---")
        print(f"  Median winning temp (midpoint): {sorted(wt)[len(wt)//2]:.1f}F")
        print(f"  Mean winning temp:              {sum(wt)/len(wt):.1f}F")
        print(f"  Min winning temp:               {min(wt):.1f}F")
        print(f"  Max winning temp:               {max(wt):.1f}F")
        print(f"  Std dev:                        {(sum((t - sum(wt)/len(wt))**2 for t in wt) / len(wt))**0.5:.1f}F")


def print_edge_summary(all_stats):
    """Print combined edge opportunities across all cities."""
    print(f"\n{'='*80}")
    print(f"  COMBINED EDGE SUMMARY")
    print(f"{'='*80}")

    print(f"\n{'City':>6s} | {'Above YES%':>10s} | {'Below YES%':>10s} | {'Fade Above P&L':>14s} | {'Fade Below P&L':>14s} | {'Mid Buy P&L':>12s}")
    print(f"{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*14}-+-{'-'*14}-+-{'-'*12}")

    total_fade_above = 0
    total_fade_below = 0
    total_mid = 0

    for city in sorted(all_stats.keys()):
        s = all_stats[city]
        fa_pnl = s['fade_above_pnl_cents'] / 100
        fb_pnl = s['fade_below_pnl_cents'] / 100
        mb_pnl = s['middle_buy_pnl_cents'] / 100
        total_fade_above += fa_pnl
        total_fade_below += fb_pnl
        total_mid += mb_pnl
        print(f"{city:>6s} | {s['above_yes_rate']:>9.1%} | {s['below_yes_rate']:>9.1%} | ${fa_pnl:>12,.2f} | ${fb_pnl:>12,.2f} | ${mb_pnl:>10,.2f}")

    print(f"{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*14}-+-{'-'*14}-+-{'-'*12}")
    print(f"{'TOTAL':>6s} | {'':>10s} | {'':>10s} | ${total_fade_above:>12,.2f} | ${total_fade_below:>12,.2f} | ${total_mid:>10,.2f}")

    print(f"\n--- Key Takeaways ---")
    print(f"  * 'Fade Above' = buy NO on the top tail bucket every day")
    print(f"  * 'Fade Below' = buy NO on the bottom tail bucket every day")
    print(f"  * 'Mid Buy'    = buy YES on the 2 middle buckets every day")
    print(f"  * All P&L assumes naive 50c entry (no market price)")
    print()

    # Best strategies
    best_city_above = min(all_stats.items(), key=lambda x: x[1]['above_yes_rate'])
    best_city_below = min(all_stats.items(), key=lambda x: x[1]['below_yes_rate'])
    print(f"  BEST tail-fade city (ABOVE): {best_city_above[0]} "
          f"(above YES rate = {best_city_above[1]['above_yes_rate']:.1%}, "
          f"meaning NO wins {1-best_city_above[1]['above_yes_rate']:.1%} of the time)")
    print(f"  BEST tail-fade city (BELOW): {best_city_below[0]} "
          f"(below YES rate = {best_city_below[1]['below_yes_rate']:.1%}, "
          f"meaning NO wins {1-best_city_below[1]['below_yes_rate']:.1%} of the time)")

    print(f"\n--- Actionable Strategy ---")
    for city in sorted(all_stats.keys()):
        s = all_stats[city]
        above_fair = s['above_yes_rate']
        below_fair = s['below_yes_rate']
        print(f"\n  {city}:")
        print(f"    ABOVE tail: Fair YES = {above_fair*100:.0f}c  |  Buy NO if priced below {(1-above_fair)*100:.0f}c")
        print(f"    BELOW tail: Fair YES = {below_fair*100:.0f}c  |  Buy NO if priced below {(1-below_fair)*100:.0f}c")

        # Find best middle bucket position
        best_pos = None
        best_rate = 0
        for pos_key, pos_data in s["position_stats"].items():
            rate = pos_data["yes"] / pos_data["total"] if pos_data["total"] > 0 else 0
            if rate > best_rate:
                best_rate = rate
                best_pos = pos_key
        if best_pos:
            print(f"    Best bucket position: {best_pos} (YES rate = {best_rate:.1%})  |  Buy YES if priced below {best_rate*100:.0f}c")


def main():
    print("Loading historical settlements...")
    markets = load_data()
    weather = filter_weather(markets)
    print(f"Total markets: {len(markets):,}")
    print(f"Weather markets: {len(weather):,}")

    if not weather:
        print("No weather markets found!")
        sys.exit(1)

    # Check for price data
    has_price = sum(1 for m in weather if m.get("last_price") is not None)
    print(f"Markets with last_price data: {has_price:,}")
    if has_price == 0:
        print("\nNOTE: No price data in dataset. All last_price values are null.")
        print("Adapting analysis to use settlement patterns & structural edges instead.")
        print("Strategies use naive 50c entry for P&L simulation.")

    # Group by city
    cities = defaultdict(list)
    for m in weather:
        cities[m["sub_category"]].append(m)

    print(f"Cities: {', '.join(sorted(cities.keys()))}")

    # Analyze each city
    all_stats = {}
    for city in sorted(cities.keys()):
        stats = analyze_city(city, cities[city])
        all_stats[city] = stats
        print_report(city, stats)

    # Combined summary
    print_edge_summary(all_stats)


if __name__ == "__main__":
    main()
