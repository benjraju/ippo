#!/usr/bin/env python3
"""
autoresearch/calibration_analyzer.py — Mispricing & calibration curve analyzer.

DO NOT MODIFY. This is a fixed analysis tool, like backtest_harness.py.

Loads all settled Kalshi markets and computes:
1. Calibration curves: actual YES rate vs implied probability per series
2. Mispricing delta (δ): where actual settlement diverges from market price
3. Edge opportunities: price buckets where |δ| is largest
4. Return-on-$1 analysis: what you get back per dollar at each price level

Based on methodology from Jon Becker's research on 72M prediction market trades.

Usage:
    python3 autoresearch/calibration_analyzer.py
    python3 autoresearch/calibration_analyzer.py --series KXHIGHNY
    python3 autoresearch/calibration_analyzer.py --buckets 20
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SETTLEMENTS_FILE = Path(__file__).parent.parent / "output" / "historical_settlements_with_prices.json"
DEFAULT_BUCKET_SIZE = 5  # cents per bucket


def load_markets(series_filter: str = None) -> list[dict]:
    """Load settled markets with price data."""
    if not SETTLEMENTS_FILE.exists():
        print(f"ERROR: {SETTLEMENTS_FILE} not found.")
        print("Run: python3 autoresearch/backtest_harness.py --refresh")
        sys.exit(1)

    with open(SETTLEMENTS_FILE) as f:
        data = json.load(f)

    markets = []
    for m in data.get("markets", []):
        if not m.get("result"):
            continue

        price = float(m.get("previous_price", 0) or 0)
        if price <= 0:
            continue

        ticker = m.get("ticker", "")
        series = ticker.rsplit("-", 2)[0] if "-" in ticker else ticker

        if series_filter and not series.startswith(series_filter):
            continue

        # Normalize price to cents
        yes_cents = price * 100 if price < 1.5 else price

        if yes_cents <= 0 or yes_cents >= 100:
            continue

        markets.append({
            "ticker": ticker,
            "series": series,
            "yes_cents": yes_cents,
            "settled_yes": m["result"] == "yes",
            "volume": float(m.get("volume", 0) or 0),
            "close_time": m.get("close_time", ""),
        })

    return markets


def compute_calibration(markets: list[dict], bucket_size: int = DEFAULT_BUCKET_SIZE) -> dict:
    """
    Compute calibration curves and mispricing per price bucket per series.

    Returns nested dict: {series: {bucket: {metrics}}}
    """
    # Group by series and price bucket
    by_series_bucket = defaultdict(lambda: defaultdict(list))
    all_buckets = defaultdict(list)

    for m in markets:
        bucket = int(m["yes_cents"] // bucket_size) * bucket_size
        by_series_bucket[m["series"]][bucket].append(m)
        all_buckets[bucket].append(m)

    results = {}

    # Per-series analysis
    for series in sorted(by_series_bucket.keys()):
        series_data = {}
        for bucket in sorted(by_series_bucket[series].keys()):
            bucket_markets = by_series_bucket[series][bucket]
            series_data[bucket] = _analyze_bucket(bucket, bucket_markets, bucket_size)
        results[series] = series_data

    # Aggregate across all series
    agg_data = {}
    for bucket in sorted(all_buckets.keys()):
        agg_data[bucket] = _analyze_bucket(bucket, all_buckets[bucket], bucket_size)
    results["ALL"] = agg_data

    return results


def _analyze_bucket(bucket_start: int, markets: list[dict], bucket_size: int) -> dict:
    """Analyze a single price bucket."""
    n = len(markets)
    if n == 0:
        return {}

    yes_count = sum(1 for m in markets if m["settled_yes"])
    actual_rate = yes_count / n

    # Implied probability = midpoint of bucket
    implied_prob = (bucket_start + bucket_size / 2) / 100.0
    implied_prob = min(max(implied_prob, 0.01), 0.99)

    # Mispricing delta: actual - implied
    delta = actual_rate - implied_prob
    relative_delta = delta / implied_prob if implied_prob > 0 else 0

    # Return on $1 for YES buyer (taker style, for comparison to Becker data)
    # If you buy YES at implied_prob price and hold to settlement:
    # Win: get $1, paid implied_prob → profit = 1 - implied_prob
    # Lose: get $0, paid implied_prob → loss = implied_prob
    # EV = actual_rate × (1 - implied_prob) - (1 - actual_rate) × implied_prob
    ev_per_dollar = actual_rate * (1 - implied_prob) - (1 - actual_rate) * implied_prob
    return_on_dollar = 1 + (ev_per_dollar / implied_prob) if implied_prob > 0 else 0

    # Return on $1 for NO buyer
    no_implied = 1 - implied_prob
    no_actual = 1 - actual_rate
    ev_no = no_actual * implied_prob - actual_rate * no_implied
    return_on_dollar_no = 1 + (ev_no / no_implied) if no_implied > 0 else 0

    # Z-score for the deviation from implied probability
    if n > 0 and 0 < implied_prob < 1:
        se = np.sqrt(implied_prob * (1 - implied_prob) / n)
        z_score = delta / se if se > 0 else 0
    else:
        z_score = 0

    # Maker-adjusted EV (subtract 1.75% maker fee)
    fee_rate = 0.0175
    maker_fee = fee_rate * implied_prob * (1 - implied_prob)
    ev_after_fee_yes = ev_per_dollar - maker_fee
    ev_after_fee_no = ev_no - maker_fee

    return {
        "bucket": f"{bucket_start}-{bucket_start + bucket_size}c",
        "n": n,
        "yes_count": yes_count,
        "actual_rate": round(actual_rate * 100, 2),
        "implied_prob": round(implied_prob * 100, 2),
        "delta_pp": round(delta * 100, 2),          # percentage points
        "relative_delta_pct": round(relative_delta * 100, 1),
        "return_on_$1_yes": round(return_on_dollar, 3),
        "return_on_$1_no": round(return_on_dollar_no, 3),
        "z_score": round(z_score, 2),
        "ev_after_fee_yes": round(ev_after_fee_yes * 100, 2),  # cents per contract
        "ev_after_fee_no": round(ev_after_fee_no * 100, 2),
        "edge_direction": "BUY_NO" if delta < -0.02 else ("BUY_YES" if delta > 0.02 else "NEUTRAL"),
    }


def print_report(results: dict, series_filter: str = None):
    """Print human-readable calibration report."""
    print("=" * 90)
    print("  CALIBRATION & MISPRICING ANALYSIS")
    print("  Based on Becker (2026) methodology — 'Return on $1' at each price level")
    print("=" * 90)

    series_list = [series_filter] if series_filter and series_filter in results else sorted(results.keys())

    for series in series_list:
        data = results[series]
        if not data:
            continue

        total_n = sum(d.get("n", 0) for d in data.values())
        print(f"\n{'─' * 90}")
        print(f"  {series}  ({total_n:,} markets)")
        print(f"{'─' * 90}")
        print(f"  {'Bucket':<10} {'N':>6} {'Actual%':>8} {'Implied%':>9} {'δ (pp)':>8} "
              f"{'Rel δ%':>8} {'$/YES':>7} {'$/NO':>7} {'z':>6} {'Signal':>10}")
        print(f"  {'─'*10} {'─'*6} {'─'*8} {'─'*9} {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*6} {'─'*10}")

        for bucket in sorted(data.keys()):
            d = data[bucket]
            if not d or d.get("n", 0) == 0:
                continue

            # Color-code the signal strength
            signal = d.get("edge_direction", "NEUTRAL")
            z = d.get("z_score", 0)

            # Flag strong signals
            flag = ""
            if abs(z) >= 2.0 and d["n"] >= 30:
                flag = " ★"

            print(f"  {d['bucket']:<10} {d['n']:>6} {d['actual_rate']:>7.1f}% "
                  f"{d['implied_prob']:>8.1f}% {d['delta_pp']:>+7.2f} "
                  f"{d['relative_delta_pct']:>+7.1f}% "
                  f"${d['return_on_$1_yes']:>5.3f} ${d['return_on_$1_no']:>5.3f} "
                  f"{z:>+5.1f} {signal:>8}{flag}")

    # Summary: top opportunities
    print(f"\n{'=' * 90}")
    print("  TOP EDGE OPPORTUNITIES (|δ| > 2pp, n >= 30, |z| >= 1.5)")
    print(f"{'=' * 90}")
    print(f"  {'Series':<14} {'Bucket':<10} {'N':>5} {'δ (pp)':>8} {'z':>6} {'Signal':>10} "
          f"{'EV/contract':>12}")

    opportunities = []
    for series, data in results.items():
        if series == "ALL":
            continue
        for bucket, d in data.items():
            if not d or d.get("n", 0) < 30:
                continue
            if abs(d.get("delta_pp", 0)) > 2.0 and abs(d.get("z_score", 0)) >= 1.5:
                ev = d["ev_after_fee_no"] if d["edge_direction"] == "BUY_NO" else d["ev_after_fee_yes"]
                opportunities.append({
                    "series": series,
                    "bucket": d["bucket"],
                    "n": d["n"],
                    "delta_pp": d["delta_pp"],
                    "z_score": d["z_score"],
                    "signal": d["edge_direction"],
                    "ev_cents": ev,
                })

    opportunities.sort(key=lambda x: abs(x["z_score"]), reverse=True)

    for opp in opportunities[:25]:
        print(f"  {opp['series']:<14} {opp['bucket']:<10} {opp['n']:>5} "
              f"{opp['delta_pp']:>+7.2f} {opp['z_score']:>+5.1f} {opp['signal']:>10} "
              f"{opp['ev_cents']:>+10.2f}c")

    if not opportunities:
        print("  (no significant opportunities found)")

    # Per-series summary
    print(f"\n{'=' * 90}")
    print("  PER-SERIES SUMMARY")
    print(f"{'=' * 90}")

    for series in sorted(results.keys()):
        if series == "ALL":
            continue
        data = results[series]
        total_n = sum(d.get("n", 0) for d in data.values())
        if total_n < 50:
            continue

        # Find best NO edge and best YES edge
        best_no = min(data.values(), key=lambda d: d.get("delta_pp", 0) if d else 0, default={})
        best_yes = max(data.values(), key=lambda d: d.get("delta_pp", 0) if d else 0, default={})

        print(f"\n  {series} ({total_n:,} markets):")
        if best_no and best_no.get("delta_pp", 0) < -2:
            print(f"    Best NO edge: {best_no.get('bucket', '?')} → δ={best_no['delta_pp']:+.2f}pp, "
                  f"z={best_no.get('z_score', 0):+.1f}, n={best_no.get('n', 0)}")
        if best_yes and best_yes.get("delta_pp", 0) > 2:
            print(f"    Best YES edge: {best_yes.get('bucket', '?')} → δ={best_yes['delta_pp']:+.2f}pp, "
                  f"z={best_yes.get('z_score', 0):+.1f}, n={best_yes.get('n', 0)}")


def main():
    parser = argparse.ArgumentParser(description="Calibration & mispricing analyzer")
    parser.add_argument("--series", type=str, default=None, help="Filter to specific series")
    parser.add_argument("--buckets", type=int, default=5, help="Price bucket size in cents (default: 5)")
    args = parser.parse_args()

    markets = load_markets(args.series)
    print(f"Loaded {len(markets):,} settled markets")

    if len(markets) < 100:
        print("ERROR: Not enough data")
        sys.exit(1)

    results = compute_calibration(markets, bucket_size=args.buckets)
    print_report(results, args.series)


if __name__ == "__main__":
    main()
