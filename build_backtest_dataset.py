"""
build_backtest_dataset.py — Joins price snapshots with settlement outcomes.

Creates a clean backtest dataset where each row is:
  "Market X was priced at Yc (pre-settlement) and settled YES/NO"

This is the GOLD STANDARD for backtesting because:
  - Prices are REAL pre-settlement trading prices (from price_capture.py)
  - Outcomes are REAL settlement results (from Kalshi API)
  - No post-settlement price contamination

Runs daily via cron. AutoResearch loads the output automatically.

Usage:
    python build_backtest_dataset.py
"""

import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

import config
from kalshi_client import KalshiClient

SNAPSHOTS_FILE = config.OUTPUT_DIR / "price_snapshots.jsonl"
DATASET_FILE = config.OUTPUT_DIR / "backtest_dataset.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_dataset")


def load_snapshots() -> dict:
    """Load all price snapshots. Returns {ticker: [(timestamp, yes_ask, yes_bid, volume), ...]}."""
    if not SNAPSHOTS_FILE.exists():
        return {}

    ticker_prices = defaultdict(list)
    for line in SNAPSHOTS_FILE.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            batch = json.loads(line)
            ts = batch.get("timestamp", "")
            for m in batch.get("markets", []):
                ticker = m.get("ticker", "")
                ya = m.get("yes_ask")
                yb = m.get("yes_bid")
                vol = m.get("volume", 0)
                if ticker and ya is not None:
                    ticker_prices[ticker].append({
                        "timestamp": ts,
                        "yes_ask": ya,
                        "yes_bid": yb,
                        "volume": vol,
                    })
        except json.JSONDecodeError:
            continue

    return dict(ticker_prices)


def fetch_settled_markets(client: KalshiClient) -> list[dict]:
    """Fetch recently settled markets."""
    settled = []
    for series in ["KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHDEN",
                    "KXBTC", "KXETH", "KXSOL", "KXNBAGAME", "KXNBAPTS"]:
        cursor = None
        for page in range(5):
            try:
                resp = client.get_markets(
                    series_ticker=series, status="settled", limit=100, cursor=cursor,
                )
            except Exception:
                break
            markets = resp.get("markets", [])
            if not markets:
                break
            for m in markets:
                settled.append({
                    "ticker": m.get("ticker", ""),
                    "title": m.get("title", ""),
                    "series": series,
                    "result": m.get("result", ""),
                    "close_time": m.get("close_time", ""),
                })
            cursor = resp.get("cursor")
            if not cursor:
                break
            time.sleep(0.2)

    return settled


def build_dataset():
    """Join snapshots with settlements to create the backtest dataset."""
    logger.info("Loading price snapshots...")
    snapshots = load_snapshots()
    logger.info(f"  {len(snapshots)} tickers with price history")

    logger.info("Fetching settled markets from Kalshi...")
    client = KalshiClient()
    settled = fetch_settled_markets(client)
    logger.info(f"  {len(settled)} settled markets")

    # Join: for each settled market, find the LAST price snapshot BEFORE settlement
    dataset = []
    matched = 0

    for mkt in settled:
        ticker = mkt["ticker"]
        result = mkt["result"]
        close_time = mkt["close_time"]

        if ticker not in snapshots or not result:
            continue

        prices = snapshots[ticker]

        # Find the last snapshot before settlement
        pre_settlement = [
            p for p in prices
            if p["timestamp"] < close_time
        ]

        if not pre_settlement:
            continue

        # Use the latest pre-settlement snapshot
        last = max(pre_settlement, key=lambda p: p["timestamp"])

        # Calculate hours before settlement
        try:
            snap_dt = datetime.fromisoformat(last["timestamp"].replace("Z", "+00:00"))
            close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            hours_before = (close_dt - snap_dt).total_seconds() / 3600
        except (ValueError, TypeError):
            hours_before = None

        dataset.append({
            "ticker": ticker,
            "title": mkt["title"],
            "series": mkt["series"],
            "result": result,
            "close_time": close_time,
            "snapshot_time": last["timestamp"],
            "hours_before_settlement": round(hours_before, 2) if hours_before else None,
            "yes_ask": last["yes_ask"],
            "yes_bid": last["yes_bid"],
            "volume": last["volume"],
            "num_snapshots": len(prices),
        })
        matched += 1

    # Save
    output = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "total_snapshots_tickers": len(snapshots),
        "total_settled": len(settled),
        "matched": matched,
        "dataset": dataset,
    }

    with open(DATASET_FILE, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Dataset built: {matched} markets with pre-settlement prices + outcomes")
    logger.info(f"Saved to {DATASET_FILE}")

    # Summary by category
    cats = defaultdict(int)
    for d in dataset:
        s = d["series"]
        if "HIGH" in s:
            cats["weather"] += 1
        elif s in ("KXBTC", "KXETH", "KXSOL"):
            cats["crypto"] += 1
        elif s == "KXNBAGAME":
            cats["nba_winner"] += 1
        elif s == "KXNBAPTS":
            cats["nba_props"] += 1

    for cat, count in sorted(cats.items()):
        logger.info(f"  {cat}: {count}")

    return dataset


if __name__ == "__main__":
    build_dataset()
