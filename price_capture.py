"""
price_capture.py -- Continuous orderbook snapshot daemon.

Records bid/ask/volume data every 5 minutes for crypto and weather Kalshi markets,
plus BTC/ETH/SOL spot prices from Coinbase. Designed for momentum and latency arb
edge detection.

Usage:
    python price_capture.py           # Run continuous loop (5-min intervals)
    python price_capture.py --once    # Single capture, then exit
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

import config
from kalshi_client import KalshiClient

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CAPTURE_SERIES = [
    "KXBTC", "KXETH", "KXSOL",
    "KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHDEN",
    "KXNBAGAME",
]

COINBASE_PAIRS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
}

INTERVAL_SECONDS = 5 * 60  # 5 minutes

OUTPUT_FILE = config.OUTPUT_DIR / "price_snapshots.jsonl"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("price_capture")

# ---------------------------------------------------------------------------
# Spot prices
# ---------------------------------------------------------------------------


def fetch_spot_prices() -> dict:
    """Fetch current BTC/ETH/SOL spot prices from Coinbase public API."""
    prices = {}
    for asset, pair in COINBASE_PAIRS.items():
        try:
            resp = requests.get(
                f"https://api.coinbase.com/v2/prices/{pair}/spot",
                timeout=10,
            )
            resp.raise_for_status()
            prices[asset] = float(resp.json()["data"]["amount"])
        except Exception as e:
            log.warning("Coinbase %s fetch failed: %s", asset, e)
    return prices


# ---------------------------------------------------------------------------
# Market snapshots
# ---------------------------------------------------------------------------


def safe_float(val) -> float:
    """Convert a value to float, returning 0.0 on failure."""
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def fetch_all_markets(client: KalshiClient) -> list[dict]:
    """Fetch open markets for all target series and extract price fields."""
    snapshots = []

    for series in CAPTURE_SERIES:
        try:
            cursor = None
            while True:
                resp = client.get_markets(
                    series_ticker=series,
                    status="open",
                    limit=100,
                    cursor=cursor,
                )
                markets = resp.get("markets", [])
                if not markets:
                    break

                for m in markets:
                    snapshots.append({
                        "ticker": m.get("ticker", ""),
                        "series": series,
                        "yes_ask": safe_float(m.get("yes_ask_dollars", 0)),
                        "yes_bid": safe_float(m.get("yes_bid_dollars", 0)),
                        "no_ask": safe_float(m.get("no_ask_dollars", 0)),
                        "no_bid": safe_float(m.get("no_bid_dollars", 0)),
                        "volume": int(safe_float(m.get("volume", 0))),
                        "last_price": safe_float(m.get("last_price", 0)),
                    })

                cursor = resp.get("cursor")
                if not cursor:
                    break

        except Exception as e:
            log.warning("Failed to fetch %s markets: %s", series, e)

    return snapshots


# ---------------------------------------------------------------------------
# Capture loop
# ---------------------------------------------------------------------------


def capture_once(client: KalshiClient) -> dict:
    """Run a single capture: spot prices + all market snapshots."""
    ts = datetime.now(timezone.utc).isoformat()

    spot = fetch_spot_prices()
    markets = fetch_all_markets(client)

    batch = {
        "timestamp": ts,
        "spot_prices": spot,
        "markets": markets,
    }

    # Append to JSONL
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "a") as f:
        f.write(json.dumps(batch) + "\n")

    log.info(
        "Captured %d markets | spot: %s | file: %s",
        len(markets),
        ", ".join(f"{k}=${v:,.0f}" for k, v in spot.items()),
        OUTPUT_FILE,
    )
    return batch


def run_loop(client: KalshiClient):
    """Run continuous capture loop with 5-minute intervals."""
    log.info("Starting price capture daemon (interval=%ds)", INTERVAL_SECONDS)
    log.info("Output: %s", OUTPUT_FILE)
    log.info("Series: %s", ", ".join(CAPTURE_SERIES))

    while True:
        try:
            capture_once(client)
        except Exception as e:
            log.error("Capture failed: %s", e, exc_info=True)

        time.sleep(INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Kalshi price capture daemon")
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single capture and exit",
    )
    args = parser.parse_args()

    client = KalshiClient()

    if args.once:
        batch = capture_once(client)
        print(json.dumps(batch, indent=2))
    else:
        run_loop(client)


if __name__ == "__main__":
    main()
