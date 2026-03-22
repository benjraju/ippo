#!/usr/bin/env python3
"""
Crypto Settlement Pattern Backtest
===================================
Analyzes 35,306 settled Kalshi markets to find structural mispricing patterns
in crypto (BTC/ETH) price bucket markets.

KEY INSIGHT: Since last_price is null for all settled markets, we can't do
traditional "buy at X cents" analysis. Instead we analyze:

1. Event structure: each hourly event has ~100-188 buckets, exactly 1 settles YES
2. Positional analysis: WHERE in the range does the winner land?
3. Edge vs center bias: do prices settle near edges or center of the offered range?
4. Bucket count impact: does the number of buckets affect predictability?
5. "Above" bucket analysis: how often does the "above" catch-all bucket win?
6. Naive strategies: uniform random buying vs positional strategies
7. Implied vs actual probability analysis
"""

import json
import math
from collections import defaultdict
from pathlib import Path


def load_data():
    path = Path("/Users/benjamin/Desktop/ippo/output/historical_settlements.json")
    with open(path) as f:
        data = json.load(f)
    return data["markets"]


def group_events(markets):
    """Group markets into events (same close_time + sub_category = one event)."""
    events = defaultdict(list)
    for m in markets:
        key = (m["sub_category"], m["close_time"])
        events[key].append(m)
    # Sort each event's markets by bucket low value
    for key in events:
        events[key].sort(key=lambda x: x["bucket"].get("low", 0))
    return events


def z_score(observed_rate, expected_rate, n):
    """Calculate z-score for a proportion test."""
    if n == 0 or expected_rate == 0 or expected_rate == 1:
        return 0.0
    se = math.sqrt(expected_rate * (1 - expected_rate) / n)
    if se == 0:
        return 0.0
    return (observed_rate - expected_rate) / se


def p_value_from_z(z):
    """Approximate two-tailed p-value from z-score."""
    # Using complementary error function approximation
    return math.erfc(abs(z) / math.sqrt(2))


def print_separator(char="=", width=90):
    print(char * width)


def print_header(title, width=90):
    print()
    print_separator("=", width)
    print(f"  {title}")
    print_separator("=", width)


def analyze_asset(asset_name, events):
    """Run full analysis for one asset (BTC or ETH)."""

    print_header(f"{asset_name} ANALYSIS ({len(events)} hourly events)")

    total_markets = sum(len(e) for e in events.values())
    total_yes = sum(1 for e in events.values() for m in e if m["result"] == "yes")
    total_no = total_markets - total_yes

    print(f"\n  Total settled markets:  {total_markets:,}")
    print(f"  Total events (hours):  {len(events):,}")
    print(f"  Avg buckets per event: {total_markets / len(events):.1f}")
    print(f"  Total YES settlements: {total_yes:,} ({100*total_yes/total_markets:.2f}%)")
    print(f"  Total NO settlements:  {total_no:,} ({100*total_no/total_markets:.2f}%)")

    # ── 1. Bucket count distribution ──
    print(f"\n  --- Bucket Count Distribution ---")
    bucket_counts = defaultdict(int)
    for event_markets in events.values():
        n = len(event_markets)
        # Group into ranges
        if n <= 50:
            bucket_counts["1-50"] += 1
        elif n <= 100:
            bucket_counts["51-100"] += 1
        elif n <= 150:
            bucket_counts["101-150"] += 1
        elif n <= 200:
            bucket_counts["151-200"] += 1
        else:
            bucket_counts["200+"] += 1
    for rng in ["1-50", "51-100", "101-150", "151-200", "200+"]:
        if rng in bucket_counts:
            print(f"    {rng:>8} buckets: {bucket_counts[rng]:>4} events")

    # ── 2. "Above" bucket analysis ──
    print(f"\n  --- 'Above' (Catch-All) Bucket Analysis ---")
    above_total = 0
    above_yes = 0
    for event_markets in events.values():
        for m in event_markets:
            if m["bucket"]["type"] == "above":
                above_total += 1
                if m["result"] == "yes":
                    above_yes += 1
    above_events_with = sum(1 for e in events.values()
                           if any(m["bucket"]["type"] == "above" for m in e))
    above_wins = sum(1 for e in events.values()
                     if any(m["bucket"]["type"] == "above" and m["result"] == "yes" for m in e))

    print(f"    Events with 'above' bucket: {above_events_with}/{len(events)}")
    print(f"    'Above' buckets total:      {above_total}")
    if above_events_with > 0:
        print(f"    'Above' bucket won:         {above_wins}/{above_events_with} ({100*above_wins/above_events_with:.1f}%)")
        # If you bought YES on every "above" bucket
        if above_total > 0:
            # Implied fair price = 1/num_buckets per event (uniform assumption)
            avg_buckets = total_markets / len(events)
            implied_prob = 1.0 / avg_buckets
            actual_prob = above_yes / above_total if above_total > 0 else 0
            print(f"    Implied prob (uniform):     {100*implied_prob:.2f}%")
            print(f"    Actual win rate:            {100*actual_prob:.2f}%")
            edge = actual_prob - implied_prob
            print(f"    Edge vs uniform:            {100*edge:+.2f}%")

    # ── 3. Positional analysis: where in the range does the winner land? ──
    print(f"\n  --- Positional Analysis (Winner Location in Range) ---")
    # For each event, find where the YES bucket sits as a percentile of the range
    positions = []  # 0.0 = bottom of range, 1.0 = top
    quintile_counts = defaultdict(int)  # 5 quintiles
    decile_counts = defaultdict(int)    # 10 deciles

    for event_markets in events.values():
        # Only look at "bucket" type markets (exclude "above")
        bucket_markets = [m for m in event_markets if m["bucket"]["type"] == "bucket"]
        if len(bucket_markets) < 2:
            continue

        # Find the winner's index
        winner_idx = None
        for i, m in enumerate(bucket_markets):
            if m["result"] == "yes":
                winner_idx = i
                break

        if winner_idx is None:
            continue  # Winner was the "above" bucket

        n = len(bucket_markets)
        position = winner_idx / (n - 1) if n > 1 else 0.5
        positions.append(position)

        # Quintile (0-20%, 20-40%, etc.)
        q = min(int(position * 5), 4)
        quintile_counts[q] += 1

        # Decile
        d = min(int(position * 10), 9)
        decile_counts[d] += 1

    if positions:
        avg_pos = sum(positions) / len(positions)
        print(f"    Events analyzed: {len(positions)}")
        print(f"    Avg winner position: {avg_pos:.3f} (0.5 = center)")
        print(f"    Median position:     {sorted(positions)[len(positions)//2]:.3f}")

        # Quintile distribution
        print(f"\n    Quintile distribution (expect 20% each if uniform):")
        quintile_labels = ["Bottom 0-20%", "Low 20-40%", "Mid-Low 40-60%",
                          "Mid-High 60-80%", "Top 80-100%"]
        for q in range(5):
            count = quintile_counts.get(q, 0)
            pct = 100 * count / len(positions) if positions else 0
            expected = 20.0
            z = z_score(count/len(positions), 0.2, len(positions))
            sig = "**" if abs(z) > 2.58 else "*" if abs(z) > 1.96 else ""
            bar = "#" * int(pct)
            print(f"      {quintile_labels[q]:>17}: {count:>4} ({pct:5.1f}%) z={z:+.2f} {sig}  {bar}")

        # Decile distribution
        print(f"\n    Decile distribution (expect 10% each if uniform):")
        for d in range(10):
            count = decile_counts.get(d, 0)
            pct = 100 * count / len(positions) if positions else 0
            z = z_score(count/len(positions), 0.1, len(positions))
            sig = "**" if abs(z) > 2.58 else "*" if abs(z) > 1.96 else ""
            bar = "#" * int(pct * 2)
            print(f"      Decile {d}: {count:>4} ({pct:5.1f}%) z={z:+.2f} {sig}  {bar}")

    # ── 4. Edge vs Center bias ──
    print(f"\n  --- Edge vs Center Bias ---")
    if positions:
        edge_threshold = 0.15  # Bottom 15% or top 15%
        edge_wins = sum(1 for p in positions if p < edge_threshold or p > (1 - edge_threshold))
        center_wins = sum(1 for p in positions if edge_threshold <= p <= (1 - edge_threshold))
        edge_pct = 100 * edge_wins / len(positions)
        center_pct = 100 * center_wins / len(positions)
        expected_edge_pct = 30.0  # 15% on each side
        expected_center_pct = 70.0
        z_edge = z_score(edge_wins/len(positions), 0.30, len(positions))

        print(f"    Edge (outer 30%):   {edge_wins:>4} ({edge_pct:.1f}%, expected 30.0%) z={z_edge:+.2f}")
        print(f"    Center (inner 70%): {center_wins:>4} ({center_pct:.1f}%, expected 70.0%)")

    # ── 5. Strategy simulations ──
    print(f"\n  --- Strategy Simulations ---")
    print(f"    (All strategies assume buying YES at the implied fair price = 1/N buckets)")
    print(f"    (P&L = payout if win ($1) minus cost, summed across all trades)")

    strategies = {}

    # Strategy A: Buy EVERY bucket in every event (baseline - always loses to fees)
    strat_a_trades = 0
    strat_a_wins = 0
    strat_a_pnl = 0.0
    for event_markets in events.values():
        n = len(event_markets)
        cost_per = 1.0 / n  # fair price
        for m in event_markets:
            strat_a_trades += 1
            strat_a_pnl -= cost_per
            if m["result"] == "yes":
                strat_a_wins += 1
                strat_a_pnl += 1.0
    strategies["A: Buy all (baseline)"] = (strat_a_trades, strat_a_wins, strat_a_pnl)

    # Strategy B: Buy only the MIDDLE bucket(s) in each event
    strat_b_trades = 0
    strat_b_wins = 0
    strat_b_pnl = 0.0
    for event_markets in events.values():
        bucket_markets = [m for m in event_markets if m["bucket"]["type"] == "bucket"]
        n_total = len(event_markets)
        n_bucket = len(bucket_markets)
        if n_bucket == 0:
            continue
        mid_idx = n_bucket // 2
        # Buy the 3 middle buckets
        for offset in [-1, 0, 1]:
            idx = mid_idx + offset
            if 0 <= idx < n_bucket:
                m = bucket_markets[idx]
                cost = 1.0 / n_total
                strat_b_trades += 1
                strat_b_pnl -= cost
                if m["result"] == "yes":
                    strat_b_wins += 1
                    strat_b_pnl += 1.0
    strategies["B: Buy 3 middle buckets"] = (strat_b_trades, strat_b_wins, strat_b_pnl)

    # Strategy C: Buy only edge buckets (bottom 10% + top 10% of range)
    strat_c_trades = 0
    strat_c_wins = 0
    strat_c_pnl = 0.0
    for event_markets in events.values():
        bucket_markets = [m for m in event_markets if m["bucket"]["type"] == "bucket"]
        n_total = len(event_markets)
        n_bucket = len(bucket_markets)
        if n_bucket < 10:
            continue
        edge_count = max(1, n_bucket // 10)
        edges = bucket_markets[:edge_count] + bucket_markets[-edge_count:]
        for m in edges:
            cost = 1.0 / n_total
            strat_c_trades += 1
            strat_c_pnl -= cost
            if m["result"] == "yes":
                strat_c_wins += 1
                strat_c_pnl += 1.0
    strategies["C: Buy edge buckets (outer 20%)"] = (strat_c_trades, strat_c_wins, strat_c_pnl)

    # Strategy D: Buy the "above" bucket only
    strat_d_trades = 0
    strat_d_wins = 0
    strat_d_pnl = 0.0
    for event_markets in events.values():
        n_total = len(event_markets)
        for m in event_markets:
            if m["bucket"]["type"] == "above":
                cost = 1.0 / n_total
                strat_d_trades += 1
                strat_d_pnl -= cost
                if m["result"] == "yes":
                    strat_d_wins += 1
                    strat_d_pnl += 1.0
    strategies["D: Buy 'above' bucket only"] = (strat_d_trades, strat_d_wins, strat_d_pnl)

    # Strategy E: Buy bottom quintile only
    strat_e_trades = 0
    strat_e_wins = 0
    strat_e_pnl = 0.0
    for event_markets in events.values():
        bucket_markets = [m for m in event_markets if m["bucket"]["type"] == "bucket"]
        n_total = len(event_markets)
        n_bucket = len(bucket_markets)
        if n_bucket < 5:
            continue
        q_size = n_bucket // 5
        bottom_q = bucket_markets[:q_size]
        for m in bottom_q:
            cost = 1.0 / n_total
            strat_e_trades += 1
            strat_e_pnl -= cost
            if m["result"] == "yes":
                strat_e_wins += 1
                strat_e_pnl += 1.0
    strategies["E: Buy bottom quintile"] = (strat_e_trades, strat_e_wins, strat_e_pnl)

    # Strategy F: Buy top quintile only
    strat_f_trades = 0
    strat_f_wins = 0
    strat_f_pnl = 0.0
    for event_markets in events.values():
        bucket_markets = [m for m in event_markets if m["bucket"]["type"] == "bucket"]
        n_total = len(event_markets)
        n_bucket = len(bucket_markets)
        if n_bucket < 5:
            continue
        q_size = n_bucket // 5
        top_q = bucket_markets[-q_size:]
        for m in top_q:
            cost = 1.0 / n_total
            strat_f_trades += 1
            strat_f_pnl -= cost
            if m["result"] == "yes":
                strat_f_wins += 1
                strat_f_pnl += 1.0
    strategies["F: Buy top quintile"] = (strat_f_trades, strat_f_wins, strat_f_pnl)

    # Strategy G: Buy 2nd quintile (20-40%)
    strat_g_trades = 0
    strat_g_wins = 0
    strat_g_pnl = 0.0
    for event_markets in events.values():
        bucket_markets = [m for m in event_markets if m["bucket"]["type"] == "bucket"]
        n_total = len(event_markets)
        n_bucket = len(bucket_markets)
        if n_bucket < 5:
            continue
        q_size = n_bucket // 5
        second_q = bucket_markets[q_size:2*q_size]
        for m in second_q:
            cost = 1.0 / n_total
            strat_g_trades += 1
            strat_g_pnl -= cost
            if m["result"] == "yes":
                strat_g_wins += 1
                strat_g_pnl += 1.0
    strategies["G: Buy 2nd quintile (20-40%)"] = (strat_g_trades, strat_g_wins, strat_g_pnl)

    # Strategy H: Buy 4th quintile (60-80%)
    strat_h_trades = 0
    strat_h_wins = 0
    strat_h_pnl = 0.0
    for event_markets in events.values():
        bucket_markets = [m for m in event_markets if m["bucket"]["type"] == "bucket"]
        n_total = len(event_markets)
        n_bucket = len(bucket_markets)
        if n_bucket < 5:
            continue
        q_size = n_bucket // 5
        fourth_q = bucket_markets[3*q_size:4*q_size]
        for m in fourth_q:
            cost = 1.0 / n_total
            strat_h_trades += 1
            strat_h_pnl -= cost
            if m["result"] == "yes":
                strat_h_wins += 1
                strat_h_pnl += 1.0
    strategies["H: Buy 4th quintile (60-80%)"] = (strat_h_trades, strat_h_wins, strat_h_pnl)

    print(f"\n    {'Strategy':<35} {'Trades':>7} {'Wins':>6} {'WinRate':>8} {'P&L':>10} {'P&L/Trade':>10} {'Edge':>8}")
    print(f"    {'-'*35} {'-'*7} {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    for name, (trades, wins, pnl) in strategies.items():
        if trades > 0:
            wr = 100 * wins / trades
            pnl_per = pnl / trades
            # Expected win rate if uniform
            # For most strategies, each trade has 1/N chance
            # Edge = actual_wr - expected_wr
            avg_n = total_markets / len(events)
            expected_wr = 100 / avg_n
            edge = wr - expected_wr
            print(f"    {name:<35} {trades:>7,} {wins:>6} {wr:>7.2f}% ${pnl:>9.2f} ${pnl_per:>9.4f} {edge:>+7.2f}%")
        else:
            print(f"    {name:<35} {trades:>7} {'N/A':>6}")

    # ── 6. Bucket distance from center analysis ──
    print(f"\n  --- Settlement Distance from Range Center ---")
    distances = []  # absolute distance from center (0=center, 0.5=edge)
    for event_markets in events.values():
        bucket_markets = [m for m in event_markets if m["bucket"]["type"] == "bucket"]
        if len(bucket_markets) < 2:
            continue
        winner_idx = None
        for i, m in enumerate(bucket_markets):
            if m["result"] == "yes":
                winner_idx = i
                break
        if winner_idx is None:
            continue
        n = len(bucket_markets)
        position = winner_idx / (n - 1) if n > 1 else 0.5
        distance = abs(position - 0.5)
        distances.append(distance)

    if distances:
        avg_dist = sum(distances) / len(distances)
        # Expected average distance from center for uniform distribution = 0.25
        expected_avg_dist = 0.25
        print(f"    Avg distance from center: {avg_dist:.4f} (expected uniform: {expected_avg_dist:.4f})")
        if avg_dist < expected_avg_dist:
            print(f"    --> Center bias detected: settlements cluster toward center ({100*(expected_avg_dist-avg_dist)/expected_avg_dist:.1f}% closer)")
        else:
            print(f"    --> Edge bias detected: settlements cluster toward edges ({100*(avg_dist-expected_avg_dist)/expected_avg_dist:.1f}% further)")

    # ── 7. Hourly pattern analysis ──
    print(f"\n  --- Hourly Settlement Patterns ---")
    hourly_positions = defaultdict(list)
    for (sub, close_time), event_markets in events.items():
        if sub != asset_name.split()[0]:  # Skip if wrong asset
            # This function is called per-asset, so all events are the same asset
            pass
        hour = close_time.split("T")[1][:2]
        bucket_markets = [m for m in event_markets if m["bucket"]["type"] == "bucket"]
        if len(bucket_markets) < 2:
            continue
        winner_idx = None
        for i, m in enumerate(bucket_markets):
            if m["result"] == "yes":
                winner_idx = i
                break
        if winner_idx is None:
            continue
        n = len(bucket_markets)
        position = winner_idx / (n - 1) if n > 1 else 0.5
        hourly_positions[hour].append(position)

    if hourly_positions:
        print(f"    {'Hour':>6} {'Events':>7} {'AvgPos':>8} {'MedianPos':>10} {'StdDev':>8}")
        print(f"    {'-'*6} {'-'*7} {'-'*8} {'-'*10} {'-'*8}")
        for hour in sorted(hourly_positions.keys()):
            pos_list = hourly_positions[hour]
            avg = sum(pos_list) / len(pos_list)
            sorted_pos = sorted(pos_list)
            median = sorted_pos[len(sorted_pos) // 2]
            variance = sum((p - avg) ** 2 for p in pos_list) / len(pos_list) if len(pos_list) > 1 else 0
            std = math.sqrt(variance)
            print(f"    {hour:>6} {len(pos_list):>7} {avg:>8.3f} {median:>10.3f} {std:>8.3f}")

    # ── 8. Consecutive direction analysis (momentum) ──
    print(f"\n  --- Momentum Analysis (Consecutive Settlement Direction) ---")
    sorted_events = sorted(events.items(), key=lambda x: x[0][1])  # Sort by close_time

    winner_positions_seq = []
    for (sub, close_time), event_markets in sorted_events:
        bucket_markets = [m for m in event_markets if m["bucket"]["type"] == "bucket"]
        if len(bucket_markets) < 2:
            winner_positions_seq.append(None)
            continue
        winner_idx = None
        for i, m in enumerate(bucket_markets):
            if m["result"] == "yes":
                winner_idx = i
                break
        if winner_idx is None:
            winner_positions_seq.append(None)
            continue
        n = len(bucket_markets)
        position = winner_idx / (n - 1) if n > 1 else 0.5
        winner_positions_seq.append(position)

    # Check if "above center" or "below center" tends to repeat
    above_center = [p > 0.5 if p is not None else None for p in winner_positions_seq]
    same_direction_count = 0
    diff_direction_count = 0
    for i in range(1, len(above_center)):
        if above_center[i] is not None and above_center[i-1] is not None:
            if above_center[i] == above_center[i-1]:
                same_direction_count += 1
            else:
                diff_direction_count += 1

    total_pairs = same_direction_count + diff_direction_count
    if total_pairs > 0:
        same_pct = 100 * same_direction_count / total_pairs
        z_mom = z_score(same_direction_count / total_pairs, 0.5, total_pairs)
        print(f"    Consecutive same side: {same_direction_count}/{total_pairs} ({same_pct:.1f}%, expected 50%)")
        print(f"    z-score: {z_mom:+.2f} ({'significant' if abs(z_mom) > 1.96 else 'not significant'})")
        if same_pct > 50:
            print(f"    --> Momentum detected: price tends to stay on same side")
        else:
            print(f"    --> Mean-reversion detected: price tends to alternate sides")

    # ── 9. Range width vs settlement position ──
    print(f"\n  --- Range Width Analysis ---")
    range_data = []
    for event_markets in events.values():
        bucket_markets = [m for m in event_markets if m["bucket"]["type"] == "bucket"]
        if len(bucket_markets) < 2:
            continue
        low_bound = bucket_markets[0]["bucket"]["low"]
        high_bound = bucket_markets[-1]["bucket"]["high"]
        range_width = high_bound - low_bound

        winner_idx = None
        for i, m in enumerate(bucket_markets):
            if m["result"] == "yes":
                winner_idx = i
                break
        if winner_idx is None:
            continue

        n = len(bucket_markets)
        position = winner_idx / (n - 1) if n > 1 else 0.5
        range_data.append((range_width, position, n))

    if range_data:
        # Split into narrow vs wide ranges
        range_data.sort(key=lambda x: x[0])
        mid = len(range_data) // 2
        narrow = range_data[:mid]
        wide = range_data[mid:]

        narrow_avg_pos = sum(r[1] for r in narrow) / len(narrow)
        wide_avg_pos = sum(r[1] for r in wide) / len(wide)
        narrow_center_bias = sum(1 for r in narrow if 0.3 <= r[1] <= 0.7) / len(narrow)
        wide_center_bias = sum(1 for r in wide if 0.3 <= r[1] <= 0.7) / len(wide)

        print(f"    Narrow ranges (bottom 50%): avg_pos={narrow_avg_pos:.3f}, center_rate={100*narrow_center_bias:.1f}%")
        print(f"    Wide ranges (top 50%):      avg_pos={wide_avg_pos:.3f}, center_rate={100*wide_center_bias:.1f}%")
        print(f"    (Center = positions 0.3-0.7, expected 40% if uniform)")

    return strategies, positions


def combined_analysis(all_events):
    """Cross-asset analysis."""
    print_header("CROSS-ASSET COMBINED ANALYSIS")

    # Compare BTC vs ETH settlement patterns
    for asset in ["BTC", "ETH"]:
        asset_events = {k: v for k, v in all_events.items() if k[0] == asset}
        positions = []
        for event_markets in asset_events.values():
            bucket_markets = [m for m in event_markets if m["bucket"]["type"] == "bucket"]
            if len(bucket_markets) < 2:
                continue
            winner_idx = None
            for i, m in enumerate(bucket_markets):
                if m["result"] == "yes":
                    winner_idx = i
                    break
            if winner_idx is None:
                continue
            n = len(bucket_markets)
            position = winner_idx / (n - 1) if n > 1 else 0.5
            positions.append(position)

        if positions:
            avg = sum(positions) / len(positions)
            std = math.sqrt(sum((p - avg)**2 for p in positions) / len(positions))
            print(f"\n  {asset}: avg_position={avg:.4f}, std={std:.4f}, n={len(positions)}")


def real_world_strategies(all_events):
    """Strategies that could actually be traded."""
    print_header("ACTIONABLE TRADING STRATEGIES")

    print("""
  CONTEXT: On Kalshi, each hourly crypto event has ~100-190 buckets.
  Each bucket trades at some price (1c-99c). Without historical prices,
  we analyze WHERE the winner lands to identify structural biases.

  If buckets are priced uniformly (each at ~0.5c-1c for 100-190 buckets),
  any positional bias creates an exploitable edge.
  """)

    for asset in ["BTC", "ETH"]:
        asset_events = {k: v for k, v in all_events.items() if k[0] == asset}
        print(f"\n  --- {asset} Actionable Insights ---")

        # Find the quintile with the highest win rate
        quintile_wins = defaultdict(int)
        quintile_total = defaultdict(int)

        for event_markets in asset_events.values():
            bucket_markets = [m for m in event_markets if m["bucket"]["type"] == "bucket"]
            n = len(bucket_markets)
            if n < 5:
                continue
            q_size = n // 5

            for i, m in enumerate(bucket_markets):
                q = min(i // q_size, 4)
                quintile_total[q] += 1
                if m["result"] == "yes":
                    quintile_wins[q] += 1

        print(f"\n    Quintile win rates (buying at uniform price = 1/N):")
        best_q = None
        best_edge = -999
        for q in range(5):
            if quintile_total[q] > 0:
                actual = quintile_wins[q] / quintile_total[q]
                # Expected: each quintile has 20% of buckets, so ~20% of winners
                # But win rate per bucket = 1/N, and each quintile has N/5 buckets
                # So expected wins in quintile = (N/5) * (1/N) = 1/5 per event
                # Win rate per bucket in quintile = wins / total_buckets_in_quintile
                expected = sum(1 for e in asset_events.values() for _ in [1]) / quintile_total[q] if quintile_total[q] > 0 else 0
                # Simpler: across all events, each quintile should get 20% of the wins
                total_events_wins = sum(quintile_wins[qq] for qq in range(5))
                expected_wins = total_events_wins / 5
                actual_wins = quintile_wins[q]
                ratio = actual_wins / expected_wins if expected_wins > 0 else 0
                z = z_score(actual_wins / total_events_wins, 0.2, total_events_wins) if total_events_wins > 0 else 0
                p = p_value_from_z(z)

                edge = ratio - 1.0
                if edge > best_edge:
                    best_edge = edge
                    best_q = q

                labels = ["Bottom (0-20%)", "Low (20-40%)", "Middle (40-60%)",
                         "High (60-80%)", "Top (80-100%)"]
                sig_str = f"p={p:.4f}" if p < 0.05 else f"p={p:.2f}"
                print(f"      {labels[q]:>18}: {actual_wins:>4} wins ({100*actual_wins/total_events_wins:.1f}%) "
                      f"ratio={ratio:.2f}x z={z:+.2f} {sig_str}")

        if best_q is not None:
            labels = ["Bottom (0-20%)", "Low (20-40%)", "Middle (40-60%)",
                     "High (60-80%)", "Top (80-100%)"]
            print(f"\n    Best quintile: {labels[best_q]} ({100*best_edge:+.1f}% edge over uniform)")

        # "Above" bucket strategy
        above_wins = 0
        above_total = 0
        total_event_count = 0
        for event_markets in asset_events.values():
            total_event_count += 1
            n = len(event_markets)
            for m in event_markets:
                if m["bucket"]["type"] == "above":
                    above_total += 1
                    if m["result"] == "yes":
                        above_wins += 1

        if above_total > 0:
            above_wr = above_wins / above_total
            # The "above" bucket should win 1/N of the time if fairly priced
            # But there's usually just 1-2 "above" buckets per event
            above_events = sum(1 for e in asset_events.values()
                              if any(m["bucket"]["type"] == "above" for m in e))
            avg_n = sum(len(e) for e in asset_events.values()) / len(asset_events)
            expected_wr = 1.0 / avg_n
            z_above = z_score(above_wr, expected_wr, above_total)

            print(f"\n    'Above' bucket: {above_wins}/{above_total} wins ({100*above_wr:.1f}%)")
            print(f"    Expected (uniform): {100*expected_wr:.2f}%")
            print(f"    z-score: {z_above:+.2f}")
            if above_wr > expected_wr:
                print(f"    --> 'Above' is UNDERPRICED: wins {above_wr/expected_wr:.1f}x more than expected")
            else:
                print(f"    --> 'Above' is OVERPRICED: wins {above_wr/expected_wr:.1f}x less than expected")


def summary_and_recommendations(all_events):
    """Final summary with concrete recommendations."""
    print_header("SUMMARY & RECOMMENDATIONS")

    for asset in ["BTC", "ETH"]:
        asset_events = {k: v for k, v in all_events.items() if k[0] == asset}

        # Gather key metrics
        positions = []
        above_wins = 0
        above_total = 0
        quintile_wins = defaultdict(int)
        total_winners = 0

        for event_markets in asset_events.values():
            bucket_markets = [m for m in event_markets if m["bucket"]["type"] == "bucket"]
            n = len(bucket_markets)

            for m in event_markets:
                if m["bucket"]["type"] == "above":
                    above_total += 1
                    if m["result"] == "yes":
                        above_wins += 1

            if n < 5:
                continue

            winner_idx = None
            for i, m in enumerate(bucket_markets):
                if m["result"] == "yes":
                    winner_idx = i
                    break
            if winner_idx is None:
                continue

            position = winner_idx / (n - 1) if n > 1 else 0.5
            positions.append(position)
            total_winners += 1

            q_size = n // 5
            q = min(winner_idx // q_size, 4)
            quintile_wins[q] += 1

        print(f"\n  {asset} Key Findings ({len(positions)} events):")

        if positions:
            avg_pos = sum(positions) / len(positions)
            center_bias = sum(1 for p in positions if 0.3 <= p <= 0.7) / len(positions)

            bias_direction = "CENTER" if avg_pos > 0.45 and avg_pos < 0.55 else (
                "LOWER" if avg_pos < 0.45 else "UPPER")

            print(f"    Avg settlement position: {avg_pos:.4f} (0.5=center)")
            print(f"    Settlement bias: {bias_direction}")
            print(f"    Center rate (30-70%): {100*center_bias:.1f}% (uniform=40%)")

            # Most profitable quintile
            if total_winners > 0:
                best_q = max(range(5), key=lambda q: quintile_wins.get(q, 0))
                best_pct = 100 * quintile_wins[best_q] / total_winners
                labels = ["Bottom", "Low", "Middle", "High", "Top"]
                print(f"    Hottest quintile: {labels[best_q]} ({best_pct:.1f}% of wins, expected 20%)")

            if above_total > 0:
                print(f"    'Above' win rate: {100*above_wins/above_total:.1f}%")

    print(f"""
  IMPORTANT CAVEATS:
  ==================
  1. This data covers only ~2 days of hourly events (27 BTC + 27 ETH events).
     Statistical power is LIMITED. Most z-scores will be small.

  2. All strategies assume buying at the UNIFORM fair price (1/N per bucket).
     Real market prices reflect supply/demand and may already price in any bias.

  3. Without actual last_price data, we cannot determine if the market already
     correctly prices these patterns. The edge exists only if the market
     prices buckets uniformly when settlement is non-uniform.

  4. To validate these patterns, you need:
     a) More historical data (weeks/months of settlements)
     b) Historical order book / last_price data to compare implied vs actual
     c) Live paper trading to test in real-time

  5. Transaction costs (Kalshi fees) will eat into any small edge.

  NEXT STEPS:
  ===========
  - Fetch more historical data (the API should have months of settlements)
  - Capture last_price BEFORE settlement (need to poll markets while open)
  - Build a live monitor that tracks price vs positional probability
  - Focus on events where the market prices buckets uniformly but settlement
    shows consistent positional bias
""")


def main():
    print("\n" + "=" * 90)
    print("  CRYPTO SETTLEMENT PATTERN BACKTEST")
    print("  Analyzing settled Kalshi crypto markets for structural mispricing")
    print("=" * 90)

    markets = load_data()
    crypto = [m for m in markets if m.get("category") == "crypto"]

    print(f"\n  Total markets in dataset: {len(markets):,}")
    print(f"  Crypto markets:          {len(crypto):,}")

    # Group into events
    all_events = group_events(crypto)

    btc_events = {k: v for k, v in all_events.items() if k[0] == "BTC"}
    eth_events = {k: v for k, v in all_events.items() if k[0] == "ETH"}

    print(f"  BTC events:              {len(btc_events):,}")
    print(f"  ETH events:              {len(eth_events):,}")

    # Per-asset analysis
    analyze_asset("BTC", btc_events)
    analyze_asset("ETH", eth_events)

    # Cross-asset
    combined_analysis(all_events)

    # Actionable strategies
    real_world_strategies(all_events)

    # Summary
    summary_and_recommendations(all_events)


if __name__ == "__main__":
    main()
