"""
brier_tracker.py -- Calibration measurement system for probability estimates.

Tracks every probability estimate the bot makes, records actual outcomes after
settlement, and computes rolling Brier scores to measure whether our model has
alpha vs the market.

Brier score = mean((forecast - outcome)^2)
  Lower is better. 0 = perfect. 0.25 = random coin flip.

Skill score = 1 - (our_brier / market_brier)
  >0 means we're beating the market. <0 means the market is better.

The market's historical Brier score on weather markets is ~0.0089 (96.4% of
the way from random to perfect). We need to beat that.

Log format: JSONL (one JSON object per line) for append-friendly writes.
Each line is a prediction record:
  {
    "ticker": "KXHIGHNY-26MAR21-B57.5",
    "market_title": "Will the high temp in NYC be 57-58 on Mar 21, 2026?",
    "series": "KXHIGHNY",
    "our_probability": 0.73,
    "market_price": 0.72,
    "strategy": "weather",
    "timestamp": "2026-03-21T14:00:00Z",
    "outcome": 0.0,           # filled in after settlement
    "outcome_time": "..."     # filled in after settlement
  }

Usage:
    python brier_tracker.py report                # show calibration report
    python brier_tracker.py report --strategy weather  # weather-only
    python brier_tracker.py report --last 500     # last 500 scored predictions
    python brier_tracker.py backfill              # compute market baseline from historical data
"""

import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config


class BrierTracker:
    """Track probability estimates vs outcomes for calibration measurement."""

    def __init__(self, log_file: str = "output/brier_log.jsonl"):
        self.log_path = Path(log_file)
        if not self.log_path.is_absolute():
            self.log_path = config.PROJECT_ROOT / log_file
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_estimate(
        self,
        ticker: str,
        market_title: str,
        series: str,
        our_probability: float,
        market_price: float,
        strategy: str,
        timestamp: str = None,
    ):
        """
        Log a probability estimate BEFORE the outcome is known.
        Called every time the bot evaluates a market.

        Args:
            ticker: Market ticker (e.g. "KXHIGHNY-26MAR21-B57.5")
            market_title: Human-readable market title
            series: Series ticker (e.g. "KXHIGHNY")
            our_probability: Our model's estimated probability [0, 1]
            market_price: Market's YES price as probability [0, 1]
            strategy: Strategy name (weather, crypto, sports, tail_fade, etc.)
            timestamp: ISO timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        record = {
            "ticker": ticker,
            "market_title": market_title,
            "series": series,
            "our_probability": round(float(our_probability), 6),
            "market_price": round(float(market_price), 6),
            "strategy": strategy,
            "timestamp": timestamp,
            "outcome": None,
            "outcome_time": None,
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def record_outcome(self, ticker: str, outcome: float):
        """
        Log the actual outcome (1.0 for YES, 0.0 for NO) after settlement.
        Updates ALL matching entries in brier_log.jsonl (there may be multiple
        estimates for the same ticker from different sessions).

        Args:
            ticker: Market ticker
            outcome: 1.0 if YES settled, 0.0 if NO settled
        """
        if not self.log_path.exists():
            return

        outcome_time = datetime.now(timezone.utc).isoformat()
        updated_lines = []
        changed = False

        with open(self.log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    updated_lines.append(line)
                    continue

                if record.get("ticker") == ticker and record.get("outcome") is None:
                    record["outcome"] = float(outcome)
                    record["outcome_time"] = outcome_time
                    changed = True

                updated_lines.append(json.dumps(record))

        if changed:
            with open(self.log_path, "w") as f:
                for line in updated_lines:
                    f.write(line + "\n")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_records(self) -> list[dict]:
        """Load all records from the JSONL log."""
        records = []
        if not self.log_path.exists():
            return records
        with open(self.log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def _scored_records(
        self,
        strategy: str = None,
        last_n: int = None,
        series: str = None,
    ) -> list[dict]:
        """Return records that have both a probability estimate and an outcome."""
        records = self._load_records()
        scored = [
            r for r in records
            if r.get("outcome") is not None
            and r.get("our_probability") is not None
            and r.get("market_price") is not None
        ]

        if strategy:
            scored = [r for r in scored if r.get("strategy") == strategy]
        if series:
            scored = [r for r in scored if r.get("series") == series]

        # Sort by timestamp (oldest first) before applying last_n
        scored.sort(key=lambda r: r.get("timestamp", ""))

        if last_n is not None and last_n > 0:
            scored = scored[-last_n:]

        return scored

    # ------------------------------------------------------------------
    # Brier Score Computation
    # ------------------------------------------------------------------

    def compute_brier_scores(
        self,
        strategy: str = None,
        last_n: int = None,
        series: str = None,
    ) -> dict:
        """
        Compute Brier scores for our model vs the market.

        Brier score = mean((forecast - outcome)^2)
        Lower is better. 0 = perfect, 0.25 = random.

        Returns:
          {
            "our_brier": float,
            "market_brier": float,
            "skill_score": float,  # 1 - (our_brier / market_brier). >0 = we're better
            "n_scored": int,
            "our_calibration": dict,  # bucketed calibration
            "market_calibration": dict,
          }
        """
        scored = self._scored_records(strategy=strategy, last_n=last_n, series=series)

        if not scored:
            return {
                "our_brier": None,
                "market_brier": None,
                "skill_score": None,
                "n_scored": 0,
                "our_calibration": {},
                "market_calibration": {},
            }

        # Brier scores
        our_sq_errors = []
        market_sq_errors = []

        for r in scored:
            outcome = float(r["outcome"])
            our_p = float(r["our_probability"])
            mkt_p = float(r["market_price"])

            our_sq_errors.append((our_p - outcome) ** 2)
            market_sq_errors.append((mkt_p - outcome) ** 2)

        our_brier = sum(our_sq_errors) / len(our_sq_errors)
        market_brier = sum(market_sq_errors) / len(market_sq_errors)

        # Skill score: 1 - (our / market). Positive = we beat the market.
        if market_brier > 0:
            skill_score = 1.0 - (our_brier / market_brier)
        else:
            skill_score = 0.0

        # Calibration: bucket predictions into deciles and compute actual rates
        our_calibration = self._compute_calibration(scored, "our_probability")
        market_calibration = self._compute_calibration(scored, "market_price")

        return {
            "our_brier": round(our_brier, 6),
            "market_brier": round(market_brier, 6),
            "skill_score": round(skill_score, 4),
            "n_scored": len(scored),
            "our_calibration": our_calibration,
            "market_calibration": market_calibration,
        }

    def _compute_calibration(self, records: list[dict], prob_key: str) -> dict:
        """
        Compute calibration table: for each probability bucket,
        what fraction of outcomes actually occurred?

        Buckets: [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
        """
        buckets = defaultdict(lambda: {"count": 0, "sum_outcomes": 0.0, "sum_probs": 0.0})

        for r in records:
            p = float(r[prob_key])
            outcome = float(r["outcome"])

            # Bucket index: 0.0-0.1 -> "0.00-0.10", etc.
            bucket_idx = min(int(p * 10), 9)  # 0-9
            bucket_low = bucket_idx / 10.0
            bucket_high = (bucket_idx + 1) / 10.0
            bucket_label = f"{bucket_low:.2f}-{bucket_high:.2f}"

            buckets[bucket_label]["count"] += 1
            buckets[bucket_label]["sum_outcomes"] += outcome
            buckets[bucket_label]["sum_probs"] += p

        calibration = {}
        for label in sorted(buckets.keys()):
            b = buckets[label]
            if b["count"] > 0:
                calibration[label] = {
                    "predicted_avg": round(b["sum_probs"] / b["count"], 4),
                    "actual_rate": round(b["sum_outcomes"] / b["count"], 4),
                    "count": b["count"],
                    "deviation": round(
                        (b["sum_outcomes"] / b["count"]) - (b["sum_probs"] / b["count"]),
                        4,
                    ),
                }

        return calibration

    # ------------------------------------------------------------------
    # Calibration Report
    # ------------------------------------------------------------------

    def calibration_report(self, strategy: str = None, last_n: int = None) -> str:
        """
        Human-readable calibration report showing:
        - Our Brier vs Market Brier
        - Skill score (are we beating the market?)
        - Calibration table: for each probability bucket, what actually happened?
        - Recommendation: which price ranges should we trade vs avoid?
        """
        scores = self.compute_brier_scores(strategy=strategy, last_n=last_n)

        lines = []
        lines.append("=" * 70)
        lines.append("CALIBRATION & BRIER SCORE REPORT")
        if strategy:
            lines.append(f"Strategy: {strategy}")
        if last_n:
            lines.append(f"Window: last {last_n} scored predictions")
        lines.append("=" * 70)
        lines.append("")

        n = scores["n_scored"]
        if n == 0:
            lines.append("No scored predictions found.")
            lines.append("Predictions are scored after settlement outcomes are recorded.")
            lines.append("")
            lines.append("To backfill market baseline from historical data:")
            lines.append("  python brier_tracker.py backfill")
            return "\n".join(lines)

        our_brier = scores["our_brier"]
        mkt_brier = scores["market_brier"]
        skill = scores["skill_score"]

        lines.append(f"Scored predictions: {n}")
        lines.append("")
        lines.append(f"  Our Brier score:    {our_brier:.6f}")
        lines.append(f"  Market Brier score: {mkt_brier:.6f}")
        lines.append(f"  Skill score:        {skill:+.4f}")
        lines.append("")

        if skill > 0:
            lines.append(f"  >>> WE ARE BEATING THE MARKET by {skill:.1%} <<<")
        elif skill < 0:
            lines.append(f"  >>> MARKET IS BETTER by {abs(skill):.1%} <<<")
        else:
            lines.append("  >>> TIED WITH MARKET <<<")

        lines.append("")

        # Brier score context
        lines.append("Brier Score Context:")
        lines.append("  0.0000 = perfect    (always right)")
        lines.append("  0.0089 = market avg (weather, historical)")
        lines.append("  0.2500 = random     (always predict 50%)")
        lines.append("")

        # Our calibration table
        lines.append("-" * 70)
        lines.append("OUR MODEL CALIBRATION")
        lines.append(f"{'Bucket':>12s}  {'Predicted':>10s}  {'Actual':>8s}  {'Deviation':>10s}  {'Count':>6s}")
        lines.append("-" * 70)
        for label, cal in sorted(scores["our_calibration"].items()):
            dev = cal["deviation"]
            dev_str = f"{dev:+.4f}"
            flag = " ***" if abs(dev) > 0.10 else " *" if abs(dev) > 0.05 else ""
            lines.append(
                f"{label:>12s}  {cal['predicted_avg']:>10.4f}  {cal['actual_rate']:>8.4f}  "
                f"{dev_str:>10s}  {cal['count']:>6d}{flag}"
            )

        lines.append("")

        # Market calibration table
        lines.append("-" * 70)
        lines.append("MARKET CALIBRATION")
        lines.append(f"{'Bucket':>12s}  {'Predicted':>10s}  {'Actual':>8s}  {'Deviation':>10s}  {'Count':>6s}")
        lines.append("-" * 70)
        for label, cal in sorted(scores["market_calibration"].items()):
            dev = cal["deviation"]
            dev_str = f"{dev:+.4f}"
            flag = " ***" if abs(dev) > 0.10 else " *" if abs(dev) > 0.05 else ""
            lines.append(
                f"{label:>12s}  {cal['predicted_avg']:>10.4f}  {cal['actual_rate']:>8.4f}  "
                f"{dev_str:>10s}  {cal['count']:>6d}{flag}"
            )

        lines.append("")

        # Recommendations
        lines.append("-" * 70)
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 70)

        our_cal = scores["our_calibration"]
        good_ranges = []
        bad_ranges = []

        for label, cal in sorted(our_cal.items()):
            if cal["count"] < 10:
                continue
            dev = cal["deviation"]
            if abs(dev) < 0.05:
                good_ranges.append(label)
            elif abs(dev) > 0.10:
                bad_ranges.append((label, dev))

        if good_ranges:
            lines.append(f"  Well-calibrated ranges: {', '.join(good_ranges)}")
        if bad_ranges:
            lines.append("  Poorly calibrated ranges (avoid or recalibrate):")
            for label, dev in bad_ranges:
                direction = "overconfident" if dev < 0 else "underconfident"
                lines.append(f"    {label}: {direction} by {abs(dev):.1%}")

        if not good_ranges and not bad_ranges:
            lines.append("  Need more data (10+ predictions per bucket) for recommendations.")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Backfill: compute market baseline from historical settlements
    # ------------------------------------------------------------------

    def backfill_market_baseline(self, settlements_path: str = None) -> dict:
        """
        Use historical_settlements_with_prices.json to compute the market's
        Brier score as a baseline.

        The market's "prediction" is the previous YES price (prev_yes_ask or
        previous_price). The outcome is the settlement result (yes=1, no=0).

        Returns summary dict with market Brier score and calibration.
        """
        if settlements_path is None:
            settlements_path = str(
                config.OUTPUT_DIR / "historical_settlements_with_prices.json"
            )

        path = Path(settlements_path)
        if not path.exists():
            print(f"ERROR: {path} not found.")
            print("Run data_collector.py to fetch historical settlements first.")
            return {}

        with open(path, "r") as f:
            data = json.load(f)

        markets = data.get("markets", [])
        if not markets:
            print("No markets found in historical settlements.")
            return {}

        # Filter to settled markets with valid prices
        scored = []
        for m in markets:
            result = m.get("result", "")
            if result not in ("yes", "no"):
                continue

            # Market's probability estimate = previous YES ask price
            # (this is the price before settlement, representing the market's prediction)
            price_str = m.get("prev_yes_ask") or m.get("previous_price") or m.get("last_price")
            if not price_str:
                continue

            try:
                market_prob = float(price_str)
            except (ValueError, TypeError):
                continue

            # Skip 0 and 1 prices (already settled, no predictive content)
            if market_prob <= 0.0 or market_prob >= 1.0:
                continue

            outcome = 1.0 if result == "yes" else 0.0
            scored.append({
                "ticker": m.get("ticker", ""),
                "series": m.get("series", ""),
                "market_prob": market_prob,
                "outcome": outcome,
            })

        if not scored:
            print("No scorable markets found (markets with valid prices and outcomes).")
            return {}

        # Compute market Brier score
        sq_errors = [(s["market_prob"] - s["outcome"]) ** 2 for s in scored]
        market_brier = sum(sq_errors) / len(sq_errors)

        # Compute by series
        series_scores = defaultdict(list)
        for s in scored:
            series = s.get("series", "unknown")
            series_scores[series].append((s["market_prob"] - s["outcome"]) ** 2)

        series_brier = {}
        for series, errors in sorted(series_scores.items()):
            series_brier[series] = {
                "brier": round(sum(errors) / len(errors), 6),
                "n": len(errors),
            }

        # Calibration
        buckets = defaultdict(lambda: {"count": 0, "sum_outcomes": 0.0, "sum_probs": 0.0})
        for s in scored:
            p = s["market_prob"]
            bucket_idx = min(int(p * 10), 9)
            bucket_low = bucket_idx / 10.0
            bucket_high = (bucket_idx + 1) / 10.0
            label = f"{bucket_low:.2f}-{bucket_high:.2f}"
            buckets[label]["count"] += 1
            buckets[label]["sum_outcomes"] += s["outcome"]
            buckets[label]["sum_probs"] += p

        calibration = {}
        for label in sorted(buckets.keys()):
            b = buckets[label]
            if b["count"] > 0:
                calibration[label] = {
                    "predicted_avg": round(b["sum_probs"] / b["count"], 4),
                    "actual_rate": round(b["sum_outcomes"] / b["count"], 4),
                    "count": b["count"],
                }

        result = {
            "market_brier_overall": round(market_brier, 6),
            "n_scored": len(scored),
            "total_markets_in_file": len(markets),
            "series_brier": series_brier,
            "calibration": calibration,
        }

        # Print report
        print("=" * 70)
        print("MARKET BASELINE BRIER SCORE (from historical settlements)")
        print("=" * 70)
        print(f"\nTotal settled markets: {len(markets)}")
        print(f"Scorable markets (valid price + outcome): {len(scored)}")
        print(f"\nMarket Brier score (overall): {market_brier:.6f}")
        pct_to_perfect = (1 - market_brier / 0.25) * 100
        print(f"  ({pct_to_perfect:.1f}% of the way from random to perfect)")
        print()

        # By series
        print("-" * 70)
        print("BY SERIES")
        print(f"{'Series':>15s}  {'Brier':>10s}  {'N':>7s}  {'Quality':>10s}")
        print("-" * 70)
        for series, sb in sorted(series_brier.items(), key=lambda x: x[1]["brier"]):
            pct = (1 - sb["brier"] / 0.25) * 100
            print(f"{series:>15s}  {sb['brier']:>10.6f}  {sb['n']:>7d}  {pct:>9.1f}%")

        print()
        print("-" * 70)
        print("MARKET CALIBRATION")
        print(f"{'Bucket':>12s}  {'Predicted':>10s}  {'Actual':>8s}  {'Deviation':>10s}  {'Count':>6s}")
        print("-" * 70)
        for label, cal in sorted(calibration.items()):
            dev = cal["actual_rate"] - cal["predicted_avg"]
            dev_str = f"{dev:+.4f}"
            flag = " ***" if abs(dev) > 0.10 else " *" if abs(dev) > 0.05 else ""
            print(
                f"{label:>12s}  {cal['predicted_avg']:>10.4f}  {cal['actual_rate']:>8.4f}  "
                f"{dev_str:>10s}  {cal['count']:>6d}{flag}"
            )

        print()
        print("=" * 70)
        print(f"TARGET: Our model must achieve Brier < {market_brier:.6f} to have alpha.")
        print("=" * 70)

        # Save baseline to JSON for reference
        baseline_path = config.OUTPUT_DIR / "market_brier_baseline.json"
        with open(baseline_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nBaseline saved to: {baseline_path}")

        return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Brier Score Tracker -- calibration measurement for probability estimates"
    )
    subparsers = parser.add_subparsers(dest="command")

    # report subcommand
    report_parser = subparsers.add_parser("report", help="Show calibration report")
    report_parser.add_argument(
        "--strategy", type=str, default=None,
        help="Filter by strategy (weather, crypto, sports, tail_fade, etc.)",
    )
    report_parser.add_argument(
        "--last", type=int, default=None,
        help="Only score the last N predictions",
    )
    report_parser.add_argument(
        "--log-file", type=str, default="output/brier_log.jsonl",
        help="Path to Brier log file",
    )

    # backfill subcommand
    backfill_parser = subparsers.add_parser(
        "backfill",
        help="Compute market Brier baseline from historical settlements",
    )
    backfill_parser.add_argument(
        "--settlements-file", type=str, default=None,
        help="Path to historical_settlements_with_prices.json",
    )
    backfill_parser.add_argument(
        "--log-file", type=str, default="output/brier_log.jsonl",
        help="Path to Brier log file",
    )

    # stats subcommand (quick summary)
    stats_parser = subparsers.add_parser("stats", help="Quick Brier score summary")
    stats_parser.add_argument(
        "--log-file", type=str, default="output/brier_log.jsonl",
        help="Path to Brier log file",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    log_file = getattr(args, "log_file", "output/brier_log.jsonl")
    tracker = BrierTracker(log_file=log_file)

    if args.command == "report":
        report = tracker.calibration_report(
            strategy=args.strategy,
            last_n=args.last,
        )
        print(report)

    elif args.command == "backfill":
        tracker.backfill_market_baseline(
            settlements_path=args.settlements_file,
        )

    elif args.command == "stats":
        records = tracker._load_records()
        scored = [r for r in records if r.get("outcome") is not None]
        unscored = [r for r in records if r.get("outcome") is None]

        print(f"Total records:   {len(records)}")
        print(f"Scored:          {len(scored)}")
        print(f"Pending outcome: {len(unscored)}")

        if scored:
            scores = tracker.compute_brier_scores()
            print(f"\nOur Brier:       {scores['our_brier']:.6f}")
            print(f"Market Brier:    {scores['market_brier']:.6f}")
            print(f"Skill score:     {scores['skill_score']:+.4f}")

            if scores["skill_score"] > 0:
                print(f"\n>>> Beating the market by {scores['skill_score']:.1%}")
            else:
                print(f"\n>>> Market is better by {abs(scores['skill_score']):.1%}")

        # Show strategy breakdown
        strategies = defaultdict(lambda: {"total": 0, "scored": 0})
        for r in records:
            s = r.get("strategy", "unknown")
            strategies[s]["total"] += 1
            if r.get("outcome") is not None:
                strategies[s]["scored"] += 1

        if strategies:
            print(f"\nBy strategy:")
            for s, counts in sorted(strategies.items()):
                print(f"  {s:15s}  total={counts['total']:>5d}  scored={counts['scored']:>5d}")


if __name__ == "__main__":
    main()
