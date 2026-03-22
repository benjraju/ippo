"""
data_collector.py — Real Kalshi historical data collection for AutoResearch backtesting.

Maintains a local SQLite database of:
  - market_snapshots: price snapshots of open markets at collection time
  - settled_markets:  final outcomes with pre-close prices and actual temps
  - calibration log:  predicted vs actual outcomes for model validation

Usage:
    python cli.py collect-data             # Snapshot open + collect settled
    python cli.py collect-data --settled   # Settled markets only (no snapshots)
    python cli.py collect-data --report    # Print calibration report
"""

import json
import math
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HISTORICAL_DATA_DIR = config.OUTPUT_DIR / "historical_data"
DB_PATH = HISTORICAL_DATA_DIR / "kalshi_history.db"
CALIBRATION_LOG_PATH = config.OUTPUT_DIR / "calibration_log.jsonl"

# Series to collect, grouped by strategy
WEATHER_SERIES = [
    "KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA",
    "KXHIGHLA",  "KXHIGHDC",  "KXHIGHDEN",
]
CRYPTO_SERIES = ["KXBTC", "KXETH", "KXSOL"]
SPORTS_SERIES = ["KXNBA", "KXNCAAB", "KXNHL", "KXMLB"]

WEATHER_SERIES_TO_CITY = {
    "KXHIGHNY":  "NYC",
    "KXHIGHCHI": "Chicago",
    "KXHIGHMIA": "Miami",
    "KXHIGHLA":  "LA",
    "KXHIGHDC":  "DC",
    "KXHIGHDEN": "Denver",
}

ALL_SERIES = WEATHER_SERIES + CRYPTO_SERIES + SPORTS_SERIES


# ---------------------------------------------------------------------------
# Database init
# ---------------------------------------------------------------------------

def init_db() -> sqlite3.Connection:
    """Create tables if they don't exist; return open connection."""
    HISTORICAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Raw price snapshots of open markets
    c.execute("""
        CREATE TABLE IF NOT EXISTS market_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_time   TEXT NOT NULL,
            ticker          TEXT NOT NULL,
            series          TEXT,
            title           TEXT,
            yes_bid_cents   REAL,
            yes_ask_cents   REAL,
            last_price_cents REAL,
            volume          INTEGER,
            status          TEXT DEFAULT 'open',
            close_time      TEXT,
            UNIQUE(snapshot_time, ticker)
        )
    """)

    # Settled market records — the ground truth for backtesting
    c.execute("""
        CREATE TABLE IF NOT EXISTS settled_markets (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker              TEXT NOT NULL UNIQUE,
            series              TEXT,
            title               TEXT,
            result              TEXT NOT NULL,        -- 'yes' or 'no'
            close_time          TEXT,
            -- Pre-close market prices (last known before settlement)
            pre_close_yes_bid   REAL,
            pre_close_yes_ask   REAL,
            pre_close_volume    INTEGER,
            -- Strategy type
            strategy            TEXT,                 -- 'weather', 'crypto', 'sports'
            -- Weather-specific
            market_type         TEXT,                 -- 'bucket', 'above', 'below'
            bucket_low          REAL,
            bucket_high         REAL,
            threshold_val       REAL,
            city                TEXT,
            actual_temp         REAL,                 -- NWS observation (may be NULL)
            forecast_temp       REAL,                 -- derived from market structure
            -- Metadata
            collected_time      TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    # Calibration: predicted vs actual for every real-data backtest trade
    c.execute("""
        CREATE TABLE IF NOT EXISTS calibration (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            log_time             TEXT NOT NULL,
            ticker               TEXT NOT NULL,
            strategy             TEXT,
            predicted_edge       REAL,
            predicted_probability REAL,
            entry_price_cents    REAL,
            side                 TEXT,
            contracts            INTEGER,
            actual_outcome       INTEGER,             -- 1 = win, 0 = loss
            pnl                  REAL,
            params_snapshot      TEXT                 -- JSON of key params
        )
    """)

    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Market-type parsing helpers
# ---------------------------------------------------------------------------

def _parse_market_type(ticker: str) -> tuple[str, float, float, float]:
    """
    Parse ticker to (market_type, bucket_low, bucket_high, threshold).

    Ticker conventions:
      KXHIGHNY-26MAR21-B57.5   → bucket  [57, 58]
      KXHIGHNY-26MAR21-T63     → above   threshold=63
      KXHIGHNY-26MAR21-LT63    → below   threshold=63
    """
    market_type = "bucket"
    bucket_low = 0.0
    bucket_high = 999.0
    threshold = 0.0

    if "-LT" in ticker:
        market_type = "below"
        try:
            threshold = float(ticker.split("-LT")[-1])
        except (ValueError, IndexError):
            pass
        bucket_low = 0.0
        bucket_high = threshold
    elif "-T" in ticker:
        market_type = "above"
        try:
            threshold = float(ticker.split("-T")[-1])
        except (ValueError, IndexError):
            pass
        bucket_low = threshold
        bucket_high = 999.0
    elif "-B" in ticker:
        market_type = "bucket"
        try:
            center = float(ticker.split("-B")[-1])
            bucket_low = math.floor(center)
            bucket_high = math.ceil(center)
            if bucket_low == bucket_high:
                bucket_high = bucket_low + 1
        except (ValueError, IndexError):
            pass

    return market_type, bucket_low, bucket_high, threshold


def _synthetic_temp_for_result(
    market_type: str,
    bucket_low: float,
    bucket_high: float,
    threshold: float,
    result: str,
) -> float:
    """
    When NWS actual_temp is unavailable, synthesize a temperature that
    produces the correct settlement result so settle_trades() works correctly.
    """
    yes_happened = result == "yes"

    if market_type == "bucket":
        if yes_happened:
            return (bucket_low + bucket_high) / 2.0
        else:
            return bucket_high + 5.0          # clearly outside the bucket

    if market_type == "above":
        return threshold + 2.0 if yes_happened else threshold - 2.0

    if market_type == "below":
        return threshold - 2.0 if yes_happened else threshold + 2.0

    return 72.0  # fallback


# ---------------------------------------------------------------------------
# Collection: settled markets
# ---------------------------------------------------------------------------

def collect_settled_markets(
    conn: sqlite3.Connection,
    verbose: bool = True,
) -> int:
    """
    Fetch recently settled markets from Kalshi API and store in DB.
    Attempts NWS temperature lookup for weather markets.
    Returns number of new records inserted.
    """
    try:
        from kalshi_client import KalshiClient
        client = KalshiClient()
    except Exception as e:
        if verbose:
            print(f"  Cannot connect to Kalshi: {e}")
        return 0

    # Optional NWS integration
    fetch_actual_high_temp = None
    NWS_OBSERVATION_STATIONS: dict = {}
    try:
        from settlement_tracker import (
            fetch_actual_high_temp,
            NWS_OBSERVATION_STATIONS,
        )
    except ImportError:
        pass

    c = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    new_count = 0

    for series in ALL_SERIES:
        try:
            resp = client.get_markets(
                series_ticker=series, status="settled", limit=200
            )
            markets_data = resp.get("markets", [])
        except Exception as e:
            if verbose:
                print(f"  Skipping {series}: {e}")
            continue

        for m in markets_data:
            ticker = m.get("ticker", "")
            result = m.get("result", "")

            if not ticker or result not in ("yes", "no"):
                continue

            # Already stored?
            c.execute("SELECT id FROM settled_markets WHERE ticker = ?", (ticker,))
            if c.fetchone():
                continue

            # Prices — Kalshi returns dollars; convert to cents
            yes_bid = float(m.get("yes_bid_dollars") or 0) * 100
            yes_ask = float(m.get("yes_ask_dollars") or 0) * 100
            last_price = float(m.get("last_price") or 0) * 100

            # Use last_price as fallback when bid/ask are 0
            if yes_bid <= 0 and yes_ask <= 0 and last_price > 0:
                yes_bid = max(1.0, last_price - 2.0)
                yes_ask = min(99.0, last_price + 2.0)
            if yes_ask <= 0 and yes_bid > 0:
                yes_ask = min(99.0, yes_bid + 2.0)
            if yes_bid <= 0 and yes_ask > 0:
                yes_bid = max(1.0, yes_ask - 2.0)

            title = m.get("title", "")
            close_time = m.get("close_time", "")
            volume = int(m.get("volume") or 0)

            # Classify strategy
            if series in WEATHER_SERIES:
                strategy = "weather"
            elif series in CRYPTO_SERIES:
                strategy = "crypto"
            else:
                strategy = "sports"

            # Weather-specific parsing
            market_type = forecast_temp = actual_temp = city = None
            bucket_low = bucket_high = threshold_val = None

            if strategy == "weather":
                city = WEATHER_SERIES_TO_CITY.get(series, "")
                market_type, bucket_low, bucket_high, threshold_val = (
                    _parse_market_type(ticker)
                )

                # Derive rough forecast from market structure
                if market_type == "bucket" and bucket_low is not None:
                    forecast_temp = (bucket_low + bucket_high) / 2.0
                elif market_type in ("above", "below"):
                    forecast_temp = threshold_val

                # Attempt NWS lookup for actual temperature
                if (
                    fetch_actual_high_temp
                    and NWS_OBSERVATION_STATIONS
                    and close_time
                ):
                    try:
                        from datetime import datetime as _dt
                        ct = _dt.fromisoformat(close_time.replace("Z", "+00:00"))
                        date_str = ct.strftime("%Y-%m-%d")
                        station = NWS_OBSERVATION_STATIONS.get(series)
                        if station:
                            actual_temp = fetch_actual_high_temp(station, date_str)
                            time.sleep(0.25)  # rate-limit NWS
                    except Exception:
                        pass

            try:
                c.execute(
                    """
                    INSERT OR IGNORE INTO settled_markets
                        (ticker, series, title, result, close_time,
                         pre_close_yes_bid, pre_close_yes_ask, pre_close_volume,
                         strategy, market_type, bucket_low, bucket_high,
                         threshold_val, city, actual_temp, forecast_temp,
                         collected_time)
                    VALUES
                        (?, ?, ?, ?, ?,
                         ?, ?, ?,
                         ?, ?, ?, ?,
                         ?, ?, ?, ?,
                         ?)
                    """,
                    (
                        ticker, series, title, result, close_time,
                        round(yes_bid, 1), round(yes_ask, 1), volume,
                        strategy,
                        market_type,
                        round(bucket_low, 2) if bucket_low is not None else None,
                        round(bucket_high, 2) if bucket_high is not None else None,
                        round(threshold_val, 2) if threshold_val is not None else None,
                        city,
                        round(actual_temp, 1) if actual_temp is not None else None,
                        round(forecast_temp, 1) if forecast_temp is not None else None,
                        now,
                    ),
                )
                if c.rowcount > 0:
                    new_count += 1
            except sqlite3.Error:
                pass

        time.sleep(0.3)  # between series to avoid rate-limiting

    conn.commit()
    return new_count


# ---------------------------------------------------------------------------
# Collection: open market snapshots
# ---------------------------------------------------------------------------

def collect_open_snapshots(
    conn: sqlite3.Connection,
    verbose: bool = True,
) -> int:
    """
    Snapshot current open market prices for future backtesting.
    Returns number of new snapshot rows stored.
    """
    try:
        from kalshi_client import KalshiClient
        client = KalshiClient()
    except Exception as e:
        if verbose:
            print(f"  Cannot connect to Kalshi: {e}")
        return 0

    c = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    count = 0

    for series in ALL_SERIES:
        try:
            resp = client.get_markets(
                series_ticker=series, status="open", limit=200
            )
            markets_data = resp.get("markets", [])
        except Exception:
            continue

        for m in markets_data:
            ticker = m.get("ticker", "")
            if not ticker:
                continue

            yes_bid = float(m.get("yes_bid_dollars") or 0) * 100
            yes_ask = float(m.get("yes_ask_dollars") or 0) * 100
            last_price = float(m.get("last_price") or 0) * 100
            volume = int(m.get("volume") or 0)

            try:
                c.execute(
                    """
                    INSERT OR IGNORE INTO market_snapshots
                        (snapshot_time, ticker, series, title,
                         yes_bid_cents, yes_ask_cents, last_price_cents,
                         volume, status, close_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)
                    """,
                    (
                        now, ticker, series, m.get("title", ""),
                        round(yes_bid, 1), round(yes_ask, 1),
                        round(last_price, 1), volume,
                        m.get("close_time", ""),
                    ),
                )
                if c.rowcount > 0:
                    count += 1
            except sqlite3.Error:
                pass

        time.sleep(0.2)

    conn.commit()
    return count


# ---------------------------------------------------------------------------
# Scenario loading for research_loop.py
# ---------------------------------------------------------------------------

def load_real_weather_scenarios(min_records: int = 30) -> list[dict]:
    """
    Load real settled weather markets from SQLite, sorted oldest-first.

    Each returned dict has the fields needed by research_loop._db_record_to_scenario():
      ticker, result, pre_close_yes_bid, pre_close_yes_ask, market_type,
      bucket_low, bucket_high, threshold_val, city, actual_temp, forecast_temp

    Returns [] if DB doesn't exist or has fewer than min_records.
    """
    if not DB_PATH.exists():
        return []

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute(
        """
        SELECT ticker, series, title, result, close_time,
               pre_close_yes_bid, pre_close_yes_ask, pre_close_volume,
               market_type, bucket_low, bucket_high, threshold_val,
               city, actual_temp, forecast_temp
        FROM settled_markets
        WHERE strategy = 'weather'
          AND result IN ('yes', 'no')
          AND pre_close_yes_ask > 0
        ORDER BY close_time ASC
        """
    )
    rows = c.fetchall()
    conn.close()

    if len(rows) < min_records:
        return []

    return [dict(row) for row in rows]


def get_db_stats() -> dict:
    """Return summary stats about the local database."""
    if not DB_PATH.exists():
        return {"db_exists": False}

    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    def _count(query, *args):
        c.execute(query, args)
        row = c.fetchone()
        return row[0] if row else 0

    stats = {
        "db_exists": True,
        "db_path": str(DB_PATH),
        "total_settled":   _count("SELECT COUNT(*) FROM settled_markets"),
        "weather_settled": _count(
            "SELECT COUNT(*) FROM settled_markets WHERE strategy = 'weather'"
        ),
        "crypto_settled": _count(
            "SELECT COUNT(*) FROM settled_markets WHERE strategy = 'crypto'"
        ),
        "sports_settled": _count(
            "SELECT COUNT(*) FROM settled_markets WHERE strategy = 'sports'"
        ),
        "with_actual_temp": _count(
            "SELECT COUNT(*) FROM settled_markets "
            "WHERE strategy = 'weather' AND actual_temp IS NOT NULL"
        ),
        "total_snapshots": _count("SELECT COUNT(*) FROM market_snapshots"),
    }

    c.execute(
        "SELECT MIN(close_time), MAX(close_time) FROM settled_markets"
    )
    row = c.fetchone()
    stats["oldest_record"] = row[0]
    stats["newest_record"] = row[1]

    conn.close()
    return stats


# ---------------------------------------------------------------------------
# Calibration logging and reporting
# ---------------------------------------------------------------------------

def log_calibration_entry(entry: dict) -> None:
    """
    Append a calibration record to the JSONL log.

    Required keys in entry:
      ticker, strategy, predicted_edge, predicted_probability,
      entry_price_cents, side, contracts, actual_outcome (1/0), pnl
    """
    entry = dict(entry)
    entry["log_time"] = datetime.now(timezone.utc).isoformat()
    CALIBRATION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def generate_calibration_report() -> dict:
    """
    Compute calibration statistics from the JSONL log.

    Returns a dict with:
      total_predictions, win_rate, total_pnl, brier_score, calibration_curve
    """
    if not CALIBRATION_LOG_PATH.exists():
        return {"error": "No calibration data — run collect-data and research first"}

    entries = []
    with open(CALIBRATION_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not entries:
        return {"error": "Calibration log is empty"}

    # 0.1-wide probability buckets
    buckets: dict = {}
    for e in entries:
        pred_prob = float(e.get("predicted_probability") or 0.5)
        outcome = int(e.get("actual_outcome") or 0)
        bucket = round(round(pred_prob * 10) / 10, 1)  # 0.0 … 1.0

        if bucket not in buckets:
            buckets[bucket] = {"count": 0, "wins": 0, "pnl": 0.0}
        buckets[bucket]["count"] += 1
        buckets[bucket]["wins"] += outcome
        buckets[bucket]["pnl"] += float(e.get("pnl") or 0)

    calibration_curve = []
    for pred_bucket, data in sorted(buckets.items()):
        n = data["count"]
        actual_rate = data["wins"] / n if n > 0 else 0.0
        calibration_curve.append(
            {
                "predicted": pred_bucket,
                "actual": round(actual_rate, 3),
                "count": n,
                "pnl": round(data["pnl"], 2),
                "deviation": round(abs(pred_bucket - actual_rate), 3),
            }
        )

    total = len(entries)
    wins = sum(1 for e in entries if int(e.get("actual_outcome") or 0) == 1)
    total_pnl = sum(float(e.get("pnl") or 0) for e in entries)

    # Brier score (mean squared error of probability predictions)
    brier = (
        sum(
            (float(e.get("predicted_probability") or 0.5) - int(e.get("actual_outcome") or 0)) ** 2
            for e in entries
        )
        / total
    )

    # Break down by strategy
    by_strategy: dict = {}
    for e in entries:
        s = e.get("strategy", "unknown")
        if s not in by_strategy:
            by_strategy[s] = {"count": 0, "wins": 0, "pnl": 0.0}
        by_strategy[s]["count"] += 1
        by_strategy[s]["wins"] += int(e.get("actual_outcome") or 0)
        by_strategy[s]["pnl"] += float(e.get("pnl") or 0)

    strategy_summary = {
        s: {
            "count": d["count"],
            "win_rate": round(d["wins"] / d["count"] * 100, 1) if d["count"] else 0,
            "pnl": round(d["pnl"], 2),
        }
        for s, d in by_strategy.items()
    }

    return {
        "total_predictions": total,
        "win_rate": round(wins / total * 100, 1) if total else 0,
        "total_pnl": round(total_pnl, 2),
        "brier_score": round(brier, 4),
        "calibration_curve": calibration_curve,
        "by_strategy": strategy_summary,
    }
