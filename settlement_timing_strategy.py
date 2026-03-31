"""
settlement_timing_strategy.py — Near-risk-free weather settlement timing strategy.

Buy YES/NO on weather markets after the daily high is already locked in.
Uses ACTUAL NWS station observations (not forecasts) after 2 PM local time.

If NYC's observed high is 72F and there's a market "Will NYC high be above 70F?",
that market should be YES=99c but might still trade at 85-95c. We buy for near-certain profit.
"""

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

import config
from kalshi_client import KalshiClient
from weather_strategy import parse_market_type

log = logging.getLogger("settlement_timing")

# NWS station IDs and timezone offsets (hours from UTC)
NWS_STATIONS = {
    "KXHIGHNY": "KNYC", "KXHIGHCHI": "KMDW", "KXHIGHMIA": "KMIA",
    "KXHIGHLA": "KLAX", "KXHIGHDC": "KDCA", "KXHIGHDEN": "KDEN",
}
CITY_TZ = {
    "KXHIGHNY": (-5, -4), "KXHIGHCHI": (-6, -5), "KXHIGHMIA": (-5, -4),
    "KXHIGHLA": (-8, -7), "KXHIGHDC": (-5, -4), "KXHIGHDEN": (-7, -6),
}
CITY_NAMES = {
    "KXHIGHNY": "NYC", "KXHIGHCHI": "Chicago", "KXHIGHMIA": "Miami",
    "KXHIGHLA": "LA", "KXHIGHDC": "DC", "KXHIGHDEN": "Denver",
}
MONTH_MAP = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
    "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}

# Parameters
MIN_BUFFER_EARLY_F = 2.0   # Before 4 PM: need 2F clearance from threshold
MIN_BUFFER_LATE_F = 3.0    # After 4 PM: can be more aggressive with 3F+ clearance
EARLIEST_HOUR = 14          # Don't trade before 2 PM local
MAX_PRICE_CENTS = 95        # Don't buy above 95c (not enough upside)
MAX_CONTRACTS = 5           # Max contracts per market
MIN_VOLUME = 50             # Skip markets with low volume (meaningless prices, won't fill)
NWS_USER_AGENT = "(ippo-bot, contact@example.com)"

# In-memory cache of today's observed high per station
_daily_highs: dict[str, dict] = {}


def _celsius_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def _is_dst(now_utc: datetime) -> bool:
    """US DST: second Sunday of March to first Sunday of November."""
    y = now_utc.year
    mar1 = datetime(y, 3, 1, tzinfo=timezone.utc)
    dst_start = mar1 + timedelta(days=(6 - mar1.weekday()) % 7 + 7)
    nov1 = datetime(y, 11, 1, tzinfo=timezone.utc)
    dst_end = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)
    return dst_start <= now_utc < dst_end


def get_local_hour(series: str, now_utc: datetime = None) -> int:
    now_utc = now_utc or datetime.now(timezone.utc)
    tz = CITY_TZ.get(series)
    if not tz:
        return -1
    offset = tz[1] if _is_dst(now_utc) else tz[0]
    return (now_utc + timedelta(hours=offset)).hour


def _extract_date(ticker: str) -> Optional[str]:
    m = re.search(r"(\d{2})([A-Z]{3})(\d{2})", ticker)
    if not m:
        return None
    mn = MONTH_MAP.get(m.group(2))
    return f"20{m.group(1)}-{mn}-{m.group(3)}" if mn else None


def fetch_observation(station_id: str) -> Optional[dict]:
    """Fetch latest observation from NWS station. Returns temp_f and max_temp_24h_f."""
    url = f"https://api.weather.gov/stations/{station_id}/observations/latest"
    try:
        resp = requests.get(url, headers={"User-Agent": NWS_USER_AGENT}, timeout=10)
        resp.raise_for_status()
        props = resp.json().get("properties", {})
        temp_c = props.get("temperature", {}).get("value")
        if temp_c is None:
            return None
        max_c = props.get("maxTemperatureLast24Hours", {}).get("value")
        return {
            "temp_f": round(_celsius_to_f(temp_c), 1),
            "max_temp_24h_f": round(_celsius_to_f(max_c), 1) if max_c is not None else None,
            "timestamp": props.get("timestamp", ""),
        }
    except Exception as e:
        log.warning(f"NWS observation failed for {station_id}: {e}")
        return None


def get_observed_daily_high(station_id: str) -> Optional[float]:
    """Get running daily high for a station (tracks max across calls)."""
    obs = fetch_observation(station_id)
    if obs is None:
        return None
    key = f"{station_id}_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    prev_high = _daily_highs.get(key, {}).get("high_f", -999)
    high = max(obs["temp_f"], prev_high)
    if obs.get("max_temp_24h_f") is not None:
        high = max(high, obs["max_temp_24h_f"])
    _daily_highs[key] = {"high_f": high, "last_obs_f": obs["temp_f"], "updated": obs["timestamp"]}
    return high


@dataclass
class SettlementTrade:
    """A settlement timing trade opportunity."""
    ticker: str
    title: str
    series: str
    city: str
    side: str               # "yes" or "no"
    observed_high_f: float
    threshold_f: float
    buffer_f: float
    yes_price_cents: int
    no_price_cents: int
    suggested_contracts: int
    expected_profit_cents: float
    local_hour: int
    confidence: str         # "high" or "medium"


def _make_trade(
    ticker, title, series, city, side, observed_high, threshold, buffer,
    yes_ask, yes_bid, local_hour, is_late,
) -> Optional[SettlementTrade]:
    """Build a SettlementTrade if the price is attractive enough."""
    if side == "yes":
        price = yes_ask
        if not (0 < price <= MAX_PRICE_CENTS):
            return None
        profit = 100 - price
        no_price = 100 - price
    else:
        no_price = 100 - yes_bid if yes_bid > 0 else 100 - yes_ask
        if not (0 < no_price <= MAX_PRICE_CENTS):
            return None
        price = no_price
        profit = 100 - no_price

    contracts = min(MAX_CONTRACTS, max(1, int(config.MAX_BET_DOLLARS / (price / 100.0))))
    confidence = "high" if (is_late and buffer >= 3) else "medium"

    return SettlementTrade(
        ticker=ticker, title=title[:60], series=series, city=city, side=side,
        observed_high_f=observed_high, threshold_f=threshold, buffer_f=round(buffer, 1),
        yes_price_cents=yes_ask if yes_ask > 0 else yes_bid,
        no_price_cents=no_price,
        suggested_contracts=contracts, expected_profit_cents=profit,
        local_hour=local_hour, confidence=confidence,
    )


def find_settlement_timing_trades(client: KalshiClient = None) -> list[SettlementTrade]:
    """
    Scan weather markets for settlement timing opportunities.
    Only activates after 2 PM local when the daily high is likely locked in.
    """
    client = client or KalshiClient()
    now_utc = datetime.now(timezone.utc)
    today_str = now_utc.strftime("%Y-%m-%d")
    trades: list[SettlementTrade] = []

    for series_ticker, station_id in NWS_STATIONS.items():
        city = CITY_NAMES.get(series_ticker, series_ticker)
        local_hour = get_local_hour(series_ticker, now_utc)
        if local_hour < EARLIEST_HOUR:
            log.debug(f"{city}: {local_hour}:00 local, too early")
            continue

        is_late = local_hour >= 16
        min_buf = MIN_BUFFER_LATE_F if is_late else MIN_BUFFER_EARLY_F

        observed_high = get_observed_daily_high(station_id)
        if observed_high is None:
            log.warning(f"{city}: no observation from {station_id}")
            continue

        log.info(f"{city}: obs_high={observed_high:.1f}F @{local_hour}:00 local ({'late' if is_late else 'early'})")

        try:
            markets = client.get_markets(series_ticker=series_ticker, limit=50, status="open").get("markets", [])
        except Exception as e:
            log.warning(f"{city}: market fetch failed: {e}")
            continue

        for mkt in markets:
            ticker = mkt.get("ticker", "")
            title = mkt.get("title", "")
            if _extract_date(ticker) != today_str:
                continue

            vol = int(mkt.get("volume", 0) or 0)
            if vol < MIN_VOLUME:
                continue  # No liquidity = order won't fill

            mtype = parse_market_type(ticker, title)
            if not mtype:
                continue

            yes_ask = int(float(mkt.get("yes_ask_dollars", 0) or 0) * 100)
            yes_bid = int(float(mkt.get("yes_bid_dollars", 0) or 0) * 100)
            if yes_ask <= 0 and yes_bid <= 0:
                continue

            trade = None
            if mtype["type"] == "above":
                buf = observed_high - mtype["threshold"]
                if buf >= min_buf:
                    trade = _make_trade(ticker, title, series_ticker, city, "yes",
                                        observed_high, mtype["threshold"], buf, yes_ask, yes_bid, local_hour, is_late)
                elif buf <= -min_buf:
                    trade = _make_trade(ticker, title, series_ticker, city, "no",
                                        observed_high, mtype["threshold"], abs(buf), yes_ask, yes_bid, local_hour, is_late)

            elif mtype["type"] == "below":
                buf = mtype["threshold"] - observed_high
                if buf >= min_buf:
                    trade = _make_trade(ticker, title, series_ticker, city, "yes",
                                        observed_high, mtype["threshold"], buf, yes_ask, yes_bid, local_hour, is_late)
                elif buf <= -min_buf:
                    trade = _make_trade(ticker, title, series_ticker, city, "no",
                                        observed_high, mtype["threshold"], abs(buf), yes_ask, yes_bid, local_hour, is_late)

            elif mtype["type"] == "bucket":
                low, high = mtype["low"], mtype["high"]
                if observed_high > high + min_buf or observed_high < low - min_buf:
                    buf = min(abs(observed_high - high), abs(observed_high - low))
                    trade = _make_trade(ticker, title, series_ticker, city, "no",
                                        observed_high, (low + high) / 2, buf, yes_ask, yes_bid, local_hour, is_late)

            if trade is not None:
                trades.append(trade)

        time.sleep(1.0)  # Rate limit between cities

    trades.sort(key=lambda t: (t.confidence == "high", t.expected_profit_cents), reverse=True)
    return trades


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_trades(trades: list[SettlementTrade]):
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()
    if not trades:
        console.print(Panel("[dim]No settlement timing opportunities found.[/dim]",
                            title="Settlement Timing", border_style="yellow"))
        return

    table = Table(title=f"Settlement Timing ({len(trades)} found)", show_lines=False)
    for col, kw in [("City", {"style": "cyan", "width": 8}), ("Market", {"width": 30}),
                     ("Side", {"width": 5}), ("Obs", {"justify": "right", "width": 7}),
                     ("Thresh", {"justify": "right", "width": 7}), ("Buf", {"justify": "right", "width": 6}),
                     ("Price", {"justify": "right", "width": 6}), ("Profit", {"justify": "right", "width": 7}),
                     ("Ctrs", {"justify": "right", "width": 5}), ("Conf", {"width": 6})]:
        table.add_column(col, **kw)

    for t in trades:
        p = t.yes_price_cents if t.side == "yes" else t.no_price_cents
        table.add_row(t.city, t.title[:30], t.side.upper(), f"{t.observed_high_f:.0f}F",
                      f"{t.threshold_f:.0f}F", f"+{t.buffer_f:.0f}F", f"{p}c",
                      f"+{t.expected_profit_cents:.0f}c", str(t.suggested_contracts), t.confidence)
    console.print(table)
    total = sum(t.expected_profit_cents * t.suggested_contracts for t in trades)
    console.print(f"\n  Expected profit: +{total:.0f}c ({total/100:.2f} USD) | Source: NWS observations")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)-7s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    parser = argparse.ArgumentParser(description="Settlement timing strategy")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    trades = find_settlement_timing_trades(KalshiClient())
    display_trades(trades)
