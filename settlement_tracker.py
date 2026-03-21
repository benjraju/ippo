"""
settlement_tracker.py -- Tracks trade outcomes, calculates real P&L,
and feeds settlement data back to AutoResearch.

Pulls positions, fills, and settled markets from the Kalshi API.
For weather markets, also fetches actual observed temperatures from
the NWS observations API to measure forecast accuracy over time.

Outputs:
    output/trade_history.csv     -- Full trade log with P&L
    output/forecast_accuracy.csv -- Weather forecast vs. actual temps
    output/daily_summary.csv     -- Daily/weekly/monthly P&L summaries
    autoresearch/real_outcomes.json -- Settlement data for AutoResearch loop

Usage:
    python settlement_tracker.py                # Full run
    python cli.py check-settlements             # Check settled trades
    python cli.py daily-summary                 # P&L report
"""

import csv
import json
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import requests
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

import config
from kalshi_client import KalshiClient

console = Console()

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """A single completed trade with settlement outcome."""
    date: str
    ticker: str
    title: str
    strategy: str            # weather / btc / sports / other
    side: str                # yes / no
    contracts: int
    entry_price: float       # cents
    settlement_result: str   # yes / no / open
    pnl: float               # dollars
    cumulative_pnl: float
    forecast_temp: Optional[float] = None
    actual_temp: Optional[float] = None


@dataclass
class ForecastAccuracy:
    """Weather forecast accuracy record."""
    date: str
    city: str
    forecast_temp: float
    ensemble_mean: Optional[float]
    ensemble_stdev: Optional[float]
    actual_temp: Optional[float]
    error: Optional[float]


# ---------------------------------------------------------------------------
# NWS Observation Fetching
# ---------------------------------------------------------------------------

# ICAO station IDs matching Kalshi's settlement locations
NWS_OBSERVATION_STATIONS = {
    "KXHIGHNY":  "KNYC",    # NYC Central Park
    "KXHIGHCHI": "KMDW",    # Chicago Midway
    "KXHIGHMIA": "KMIA",    # Miami International
    "KXHIGHLA":  "KLAX",    # LAX
    "KXHIGHDC":  "KDCA",    # Washington DC Reagan National
    "KXHIGHDEN": "KDEN",    # Denver International
}

SERIES_TO_CITY = {
    "KXHIGHNY":  "NYC",
    "KXHIGHCHI": "Chicago",
    "KXHIGHMIA": "Miami",
    "KXHIGHLA":  "LA",
    "KXHIGHDC":  "DC",
    "KXHIGHDEN": "Denver",
}


def fetch_actual_high_temp(station_id: str, date_str: str) -> Optional[float]:
    """
    Fetch the actual recorded high temperature from NWS observations.
    Uses the observations API for a given station and date.

    Args:
        station_id: ICAO station ID (e.g. "KNYC")
        date_str: Date in YYYY-MM-DD format

    Returns:
        High temperature in Fahrenheit, or None if unavailable.
    """
    # Observations endpoint: get observations for the station on that date
    start = f"{date_str}T00:00:00Z"
    end_date = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    end = f"{end_date}T00:00:00Z"

    url = f"https://api.weather.gov/stations/{station_id}/observations"
    params = {"start": start, "end": end}
    headers = {"User-Agent": "kalshi-settlement-tracker/1.0"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        features = data.get("features", [])
        if not features:
            return None

        # Find the maximum temperature from all observations that day
        max_temp_c = None
        for obs in features:
            props = obs.get("properties", {})
            temp = props.get("maxTemperatureLast24Hours", {})
            if temp and temp.get("value") is not None:
                val = temp["value"]
                if max_temp_c is None or val > max_temp_c:
                    max_temp_c = val

        # Fallback: use the max of all individual temperature readings
        if max_temp_c is None:
            temps = []
            for obs in features:
                props = obs.get("properties", {})
                temp = props.get("temperature", {})
                if temp and temp.get("value") is not None:
                    temps.append(temp["value"])
            if temps:
                max_temp_c = max(temps)

        if max_temp_c is not None:
            # Convert Celsius to Fahrenheit
            return round(max_temp_c * 9 / 5 + 32, 1)

        return None

    except Exception as e:
        console.print(f"[dim]NWS observation fetch failed for {station_id} on {date_str}: {e}[/dim]")
        return None


# ---------------------------------------------------------------------------
# Ticker / Market Parsing
# ---------------------------------------------------------------------------

def classify_strategy(ticker: str) -> str:
    """Classify a ticker into a strategy category."""
    ticker_upper = ticker.upper()
    if any(ticker_upper.startswith(s) for s in ["KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHLA", "KXHIGHDC", "KXHIGHDEN"]):
        return "weather"
    elif any(ticker_upper.startswith(s) for s in ["KXBTC", "KXETH", "KXSOL"]):
        return "crypto"
    elif any(ticker_upper.startswith(s) for s in ["KXNBA", "KXNHL", "KXMLB", "KXNCAA", "KXMARMAD"]):
        return "sports"
    return "other"


def extract_series_from_ticker(ticker: str) -> Optional[str]:
    """Extract the series ticker from a market ticker.
    e.g. KXHIGHCHI-26MAR21-T63 -> KXHIGHCHI
    """
    for series in config.TARGET_MARKET_SERIES:
        if ticker.upper().startswith(series):
            return series
    # Fallback: take everything before the first dash
    parts = ticker.split("-")
    return parts[0] if parts else None


def extract_date_from_ticker(ticker: str) -> Optional[str]:
    """Extract settlement date from ticker like KXHIGHCHI-26MAR21-T63 -> 2026-03-21."""
    month_map = {
        "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
        "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
        "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
    }
    match = re.search(r'(\d{2})([A-Z]{3})(\d{2})', ticker)
    if not match:
        return None
    year_short, month_str, day_str = match.groups()
    month_num = month_map.get(month_str)
    if not month_num:
        return None
    return f"20{year_short}-{month_num}-{day_str}"


# ---------------------------------------------------------------------------
# Settlement Tracker
# ---------------------------------------------------------------------------

class SettlementTracker:
    """Tracks trade outcomes and settlement P&L from Kalshi API data."""

    def __init__(self, client: KalshiClient = None):
        self.client = client or KalshiClient()
        self.trade_history_path = config.OUTPUT_DIR / "trade_history.csv"
        self.forecast_accuracy_path = config.OUTPUT_DIR / "forecast_accuracy.csv"
        self.daily_summary_path = config.OUTPUT_DIR / "daily_summary.csv"
        self.real_outcomes_path = config.AUTORESEARCH_DIR / "real_outcomes.json"

    # -------------------------------------------------------------------
    # Data Collection from Kalshi
    # -------------------------------------------------------------------

    def get_all_fills(self, limit: int = 1000) -> list[dict]:
        """Fetch all fills, paginating if necessary."""
        all_fills = []
        try:
            resp = self.client.get_fills(limit=min(limit, 100))
            fills = resp.get("fills", [])
            all_fills.extend(fills)
            # Kalshi paginates with cursors
            cursor = resp.get("cursor")
            while cursor and len(all_fills) < limit:
                resp = self.client.get_fills(limit=100, cursor=cursor)
                new_fills = resp.get("fills", [])
                if not new_fills:
                    break
                all_fills.extend(new_fills)
                cursor = resp.get("cursor")
                time.sleep(0.3)
        except Exception as e:
            console.print(f"[red]Error fetching fills: {e}[/red]")
        return all_fills

    def get_all_positions(self) -> list[dict]:
        """Fetch all current positions."""
        try:
            resp = self.client.get_positions()
            return resp.get("market_positions", [])
        except Exception as e:
            console.print(f"[red]Error fetching positions: {e}[/red]")
            return []

    def get_market_details(self, ticker: str) -> Optional[dict]:
        """Fetch market details, return None on error."""
        try:
            resp = self.client.get_market(ticker)
            return resp.get("market", {})
        except Exception:
            return None

    # -------------------------------------------------------------------
    # Settlement Analysis
    # -------------------------------------------------------------------

    def analyze_fills(self) -> list[TradeRecord]:
        """
        Pull fills from the API, match with market settlement status,
        and compute P&L for each trade.
        """
        fills = self.get_all_fills()
        if not fills:
            console.print("[yellow]No fills found in account.[/yellow]")
            return []

        # Group fills by ticker
        fills_by_ticker: dict[str, list[dict]] = defaultdict(list)
        for fill in fills:
            ticker = fill.get("ticker", "")
            if ticker:
                fills_by_ticker[ticker].append(fill)

        records = []
        cumulative_pnl = 0.0

        # Load existing history to carry forward cumulative_pnl
        existing = self._load_existing_history()
        if existing:
            cumulative_pnl = existing[-1].cumulative_pnl

        # Skip tickers that are already settled in history.
        # Re-process "open" tickers to check if they've settled since last run.
        settled_tickers = {r.ticker for r in existing if r.settlement_result in ("yes", "no")}

        for ticker, ticker_fills in sorted(fills_by_ticker.items()):
            if ticker in settled_tickers:
                continue

            # Get market details for settlement info
            market = self.get_market_details(ticker)
            if not market:
                continue

            title = market.get("title", ticker)
            status = market.get("status", "unknown")
            result = market.get("result", "")  # "yes" or "no" for settled

            strategy = classify_strategy(ticker)

            # Compute net position from fills
            net_yes = 0
            total_cost = 0.0
            for fill in ticker_fills:
                side = fill.get("side", "")
                action = fill.get("action", "")
                # API returns count_fp as string (e.g. "1.00"), fall back to count
                count_raw = fill.get("count_fp") or fill.get("count", "0")
                count = int(float(count_raw))
                # API returns yes_price_dollars as string (e.g. "0.12" = 12 cents)
                price_dollars = float(fill.get("yes_price_dollars") or fill.get("yes_price", "0"))
                price_cents = price_dollars * 100

                if action == "buy" and side == "yes":
                    net_yes += count
                    total_cost += count * price_cents / 100.0
                elif action == "sell" and side == "yes":
                    net_yes -= count
                    total_cost -= count * price_cents / 100.0
                elif action == "buy" and side == "no":
                    net_yes -= count
                    total_cost += count * (100 - price_cents) / 100.0
                elif action == "sell" and side == "no":
                    net_yes += count
                    total_cost -= count * (100 - price_cents) / 100.0

            # Determine trade side
            side_label = "yes" if net_yes > 0 else "no" if net_yes < 0 else "flat"
            abs_contracts = abs(net_yes)

            if abs_contracts == 0:
                continue

            # Average entry price
            entry_price = abs(total_cost / abs_contracts * 100) if abs_contracts else 0

            # P&L calculation
            pnl = 0.0
            settlement_result = "open"

            if status == "settled" or status == "finalized":
                settlement_result = result if result else "unknown"
                if result == "yes":
                    # YES holders get $1 per contract
                    if net_yes > 0:
                        pnl = abs_contracts * (100 - entry_price) / 100.0
                    else:
                        pnl = -abs_contracts * (100 - entry_price) / 100.0
                elif result == "no":
                    # NO holders get $1 per contract
                    if net_yes > 0:
                        pnl = -abs_contracts * entry_price / 100.0
                    else:
                        pnl = abs_contracts * entry_price / 100.0

            cumulative_pnl += pnl

            # Extract weather data if applicable
            forecast_temp = None
            actual_temp = None
            if strategy == "weather":
                # Always look up forecast (capture it while JSON logs are fresh)
                forecast_temp = self._lookup_forecast_temp(ticker)
                # Only fetch actual temp once market has settled
                if settlement_result in ("yes", "no"):
                    actual_temp = self._fetch_weather_actual(ticker)

            # Get fill date
            fill_date = ""
            if ticker_fills:
                ts = ticker_fills[0].get("created_time", "")
                if ts:
                    fill_date = ts[:10]
                else:
                    fill_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            record = TradeRecord(
                date=fill_date,
                ticker=ticker,
                title=title[:80],
                strategy=strategy,
                side=side_label,
                contracts=abs_contracts,
                entry_price=round(entry_price, 1),
                settlement_result=settlement_result,
                pnl=round(pnl, 4),
                cumulative_pnl=round(cumulative_pnl, 4),
                forecast_temp=forecast_temp,
                actual_temp=actual_temp,
            )
            records.append(record)
            time.sleep(0.3)  # Rate limiting

        return records

    def _lookup_forecast_temp(self, ticker: str) -> Optional[float]:
        """Look up the forecast temp used when this trade was placed.
        Searches auto_trade JSON logs for the ticker's forecast_temp field,
        falling back to parsing the reason string.
        """
        # Search recent auto_trade JSON logs
        try:
            json_logs = sorted(
                config.OUTPUT_DIR.glob("auto_trade_*.json"),
                key=lambda p: p.name,
                reverse=True,
            )
            for log_path in json_logs[:14]:  # Check last 14 days
                try:
                    with open(log_path) as f:
                        data = json.load(f)
                    for d in data.get("decisions", []):
                        if d.get("ticker") == ticker:
                            # Prefer structured field
                            if d.get("forecast_temp") is not None:
                                return float(d["forecast_temp"])
                            # Fall back to parsing reason string
                            reason = d.get("reason", "")
                            if "forecast=" in reason:
                                try:
                                    part = reason.split("forecast=")[1]
                                    return float(part.split("F")[0])
                                except (IndexError, ValueError):
                                    pass
                except (json.JSONDecodeError, IOError):
                    continue
        except Exception:
            pass
        return None

    def _fetch_weather_actual(self, ticker: str) -> Optional[float]:
        """Fetch actual observed high temp for a weather market ticker."""
        series = extract_series_from_ticker(ticker)
        date_str = extract_date_from_ticker(ticker)
        if not series or not date_str:
            return None

        station_id = NWS_OBSERVATION_STATIONS.get(series)
        if not station_id:
            return None

        return fetch_actual_high_temp(station_id, date_str)

    # -------------------------------------------------------------------
    # Forecast Accuracy Tracking
    # -------------------------------------------------------------------

    def build_forecast_accuracy(self, records: list[TradeRecord]) -> list[ForecastAccuracy]:
        """
        For settled weather trades, compare forecast to actual temperature.
        """
        accuracy_records = []

        for rec in records:
            if rec.strategy != "weather":
                continue
            if rec.settlement_result not in ("yes", "no"):
                continue

            series = extract_series_from_ticker(rec.ticker)
            city = SERIES_TO_CITY.get(series, "Unknown")
            date_str = extract_date_from_ticker(rec.ticker)

            actual_temp = rec.actual_temp
            if actual_temp is None and date_str:
                station = NWS_OBSERVATION_STATIONS.get(series)
                if station:
                    actual_temp = fetch_actual_high_temp(station, date_str)
                    time.sleep(0.5)

            error = None
            if rec.forecast_temp is not None and actual_temp is not None:
                error = round(actual_temp - rec.forecast_temp, 1)

            accuracy_records.append(ForecastAccuracy(
                date=date_str or rec.date,
                city=city,
                forecast_temp=rec.forecast_temp or 0.0,
                ensemble_mean=None,
                ensemble_stdev=None,
                actual_temp=actual_temp,
                error=error,
            ))

        return accuracy_records

    # -------------------------------------------------------------------
    # P&L Summaries
    # -------------------------------------------------------------------

    def compute_daily_summary(self, records: list[TradeRecord]) -> dict:
        """
        Compute daily, weekly, monthly P&L and key metrics from trade records.
        """
        if not records:
            return self._empty_summary()

        settled = [r for r in records if r.settlement_result in ("yes", "no")]
        if not settled:
            return self._empty_summary()

        # Basic P&L
        total_pnl = sum(r.pnl for r in settled)
        wins = [r for r in settled if r.pnl > 0]
        losses = [r for r in settled if r.pnl < 0]
        win_rate = len(wins) / len(settled) * 100 if settled else 0

        # Profit factor
        gross_profit = sum(r.pnl for r in wins)
        gross_loss = abs(sum(r.pnl for r in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Daily P&L grouped by date
        daily_pnl = defaultdict(float)
        for r in settled:
            daily_pnl[r.date] += r.pnl

        daily_returns = list(daily_pnl.values())

        # Sharpe ratio (annualized, assuming 365 trading days for prediction markets)
        if len(daily_returns) >= 2:
            mean_daily = sum(daily_returns) / len(daily_returns)
            var = sum((x - mean_daily) ** 2 for x in daily_returns) / (len(daily_returns) - 1)
            std_daily = math.sqrt(var) if var > 0 else 0.001
            sharpe = (mean_daily / std_daily) * math.sqrt(365) if std_daily > 0 else 0
        else:
            sharpe = 0
            mean_daily = total_pnl

        # Max drawdown
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for r in settled:
            cum += r.pnl
            if cum > peak:
                peak = cum
            dd = peak - cum
            if dd > max_dd:
                max_dd = dd

        # Period summaries
        now = datetime.now(timezone.utc).date()
        today_str = now.strftime("%Y-%m-%d")
        week_ago = (now - timedelta(days=7)).strftime("%Y-%m-%d")
        month_ago = (now - timedelta(days=30)).strftime("%Y-%m-%d")

        today_pnl = sum(r.pnl for r in settled if r.date == today_str)
        week_pnl = sum(r.pnl for r in settled if r.date >= week_ago)
        month_pnl = sum(r.pnl for r in settled if r.date >= month_ago)

        # Strategy breakdown
        strategy_pnl = defaultdict(lambda: {"pnl": 0.0, "trades": 0, "wins": 0})
        for r in settled:
            strategy_pnl[r.strategy]["pnl"] += r.pnl
            strategy_pnl[r.strategy]["trades"] += 1
            if r.pnl > 0:
                strategy_pnl[r.strategy]["wins"] += 1

        return {
            "total_pnl": round(total_pnl, 2),
            "today_pnl": round(today_pnl, 2),
            "week_pnl": round(week_pnl, 2),
            "month_pnl": round(month_pnl, 2),
            "total_trades": len(settled),
            "open_trades": len([r for r in records if r.settlement_result == "open"]),
            "win_rate": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "sharpe": round(sharpe, 2),
            "max_drawdown": round(max_dd, 2),
            "avg_win": round(gross_profit / len(wins), 4) if wins else 0,
            "avg_loss": round(gross_loss / len(losses), 4) if losses else 0,
            "best_trade": round(max(r.pnl for r in settled), 4) if settled else 0,
            "worst_trade": round(min(r.pnl for r in settled), 4) if settled else 0,
            "strategy_breakdown": dict(strategy_pnl),
            "daily_returns": dict(daily_pnl),
        }

    def _empty_summary(self) -> dict:
        return {
            "total_pnl": 0, "today_pnl": 0, "week_pnl": 0, "month_pnl": 0,
            "total_trades": 0, "open_trades": 0, "win_rate": 0,
            "profit_factor": 0, "sharpe": 0, "max_drawdown": 0,
            "avg_win": 0, "avg_loss": 0, "best_trade": 0, "worst_trade": 0,
            "strategy_breakdown": {}, "daily_returns": {},
        }

    # -------------------------------------------------------------------
    # File I/O
    # -------------------------------------------------------------------

    def _load_existing_history(self) -> list[TradeRecord]:
        """Load existing trade_history.csv into TradeRecord list."""
        if not self.trade_history_path.exists():
            return []

        records = []
        try:
            with open(self.trade_history_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(TradeRecord(
                        date=row.get("date", ""),
                        ticker=row.get("ticker", ""),
                        title=row.get("title", ""),
                        strategy=row.get("strategy", ""),
                        side=row.get("side", ""),
                        contracts=int(row.get("contracts", 0)),
                        entry_price=float(row.get("entry_price", 0)),
                        settlement_result=row.get("settlement_result", "open"),
                        pnl=float(row.get("pnl", 0)),
                        cumulative_pnl=float(row.get("cumulative_pnl", 0)),
                        forecast_temp=float(row["forecast_temp"]) if row.get("forecast_temp") else None,
                        actual_temp=float(row["actual_temp"]) if row.get("actual_temp") else None,
                    ))
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load existing history: {e}[/yellow]")
        return records

    def save_trade_history(self, records: list[TradeRecord]):
        """Save trade history to CSV."""
        config.OUTPUT_DIR.mkdir(exist_ok=True)
        fieldnames = [
            "date", "ticker", "title", "strategy", "side", "contracts",
            "entry_price", "settlement_result", "pnl", "cumulative_pnl",
            "forecast_temp", "actual_temp",
        ]
        with open(self.trade_history_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow({
                    "date": r.date,
                    "ticker": r.ticker,
                    "title": r.title,
                    "strategy": r.strategy,
                    "side": r.side,
                    "contracts": r.contracts,
                    "entry_price": r.entry_price,
                    "settlement_result": r.settlement_result,
                    "pnl": r.pnl,
                    "cumulative_pnl": r.cumulative_pnl,
                    "forecast_temp": r.forecast_temp if r.forecast_temp is not None else "",
                    "actual_temp": r.actual_temp if r.actual_temp is not None else "",
                })
        console.print(f"[green]Saved trade history: {self.trade_history_path}[/green]")

    def save_forecast_accuracy(self, accuracy: list[ForecastAccuracy]):
        """Save forecast accuracy to CSV."""
        config.OUTPUT_DIR.mkdir(exist_ok=True)
        fieldnames = [
            "date", "city", "forecast_temp", "ensemble_mean",
            "ensemble_stdev", "actual_temp", "error",
        ]
        with open(self.forecast_accuracy_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for a in accuracy:
                writer.writerow({
                    "date": a.date,
                    "city": a.city,
                    "forecast_temp": a.forecast_temp,
                    "ensemble_mean": a.ensemble_mean if a.ensemble_mean is not None else "",
                    "ensemble_stdev": a.ensemble_stdev if a.ensemble_stdev is not None else "",
                    "actual_temp": a.actual_temp if a.actual_temp is not None else "",
                    "error": a.error if a.error is not None else "",
                })
        console.print(f"[green]Saved forecast accuracy: {self.forecast_accuracy_path}[/green]")

    def save_daily_summary(self, summary: dict):
        """Append today's summary to daily_summary.csv."""
        config.OUTPUT_DIR.mkdir(exist_ok=True)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        fieldnames = [
            "date", "total_pnl", "today_pnl", "week_pnl", "month_pnl",
            "total_trades", "open_trades", "win_rate", "profit_factor",
            "sharpe", "max_drawdown", "avg_win", "avg_loss",
            "best_trade", "worst_trade",
        ]

        file_exists = self.daily_summary_path.exists()
        with open(self.daily_summary_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "date": today,
                "total_pnl": summary["total_pnl"],
                "today_pnl": summary["today_pnl"],
                "week_pnl": summary["week_pnl"],
                "month_pnl": summary["month_pnl"],
                "total_trades": summary["total_trades"],
                "open_trades": summary["open_trades"],
                "win_rate": summary["win_rate"],
                "profit_factor": summary["profit_factor"],
                "sharpe": summary["sharpe"],
                "max_drawdown": summary["max_drawdown"],
                "avg_win": summary["avg_win"],
                "avg_loss": summary["avg_loss"],
                "best_trade": summary["best_trade"],
                "worst_trade": summary["worst_trade"],
            })
        console.print(f"[green]Appended daily summary: {self.daily_summary_path}[/green]")

    def save_real_outcomes(self, records: list[TradeRecord], summary: dict):
        """
        Write real_outcomes.json for AutoResearch to consume.
        This lets the research loop validate simulated results against reality.
        """
        config.AUTORESEARCH_DIR.mkdir(exist_ok=True)

        settled = [r for r in records if r.settlement_result in ("yes", "no")]
        outcomes = []
        for r in settled:
            entry = {
                "date": r.date,
                "ticker": r.ticker,
                "strategy": r.strategy,
                "side": r.side,
                "contracts": r.contracts,
                "entry_price": r.entry_price,
                "settlement_result": r.settlement_result,
                "pnl": r.pnl,
            }
            if r.forecast_temp is not None:
                entry["forecast_temp"] = r.forecast_temp
            if r.actual_temp is not None:
                entry["actual_temp"] = r.actual_temp
            outcomes.append(entry)

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_settled_trades": len(settled),
            "total_pnl": summary["total_pnl"],
            "win_rate": summary["win_rate"],
            "sharpe": summary["sharpe"],
            "profit_factor": summary["profit_factor"],
            "max_drawdown": summary["max_drawdown"],
            "strategy_breakdown": summary["strategy_breakdown"],
            "outcomes": outcomes,
        }

        with open(self.real_outcomes_path, "w") as f:
            json.dump(payload, f, indent=2)
        console.print(f"[green]Saved real outcomes for AutoResearch: {self.real_outcomes_path}[/green]")

    # -------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------

    def display_settlements(self, records: list[TradeRecord]):
        """Show settled trades in a rich table."""
        settled = [r for r in records if r.settlement_result in ("yes", "no")]
        open_trades = [r for r in records if r.settlement_result == "open"]

        if not settled and not open_trades:
            console.print("[yellow]No trades found.[/yellow]")
            return

        if settled:
            table = Table(title="Settled Trades", show_lines=False)
            table.add_column("Date", style="dim", width=10)
            table.add_column("Ticker", style="cyan", width=28)
            table.add_column("Strategy", width=8)
            table.add_column("Side", width=5)
            table.add_column("Qty", justify="right", width=4)
            table.add_column("Entry", justify="right", width=7)
            table.add_column("Result", width=6)
            table.add_column("P&L", justify="right", width=9)
            table.add_column("Cumul.", justify="right", width=9)

            for r in settled:
                pnl_color = "green" if r.pnl >= 0 else "red"
                result_color = "green" if (r.settlement_result == "yes" and r.side == "yes") or \
                                          (r.settlement_result == "no" and r.side == "no") else "red"
                table.add_row(
                    r.date,
                    r.ticker[:28],
                    r.strategy,
                    r.side,
                    str(r.contracts),
                    f"{r.entry_price:.0f}c",
                    f"[{result_color}]{r.settlement_result.upper()}[/{result_color}]",
                    f"[{pnl_color}]${r.pnl:+.2f}[/{pnl_color}]",
                    f"${r.cumulative_pnl:+.2f}",
                )
            console.print(table)

        if open_trades:
            console.print(f"\n[dim]Open positions: {len(open_trades)}[/dim]")
            for r in open_trades:
                console.print(f"  [cyan]{r.ticker}[/cyan] — {r.side} x{r.contracts} @ {r.entry_price:.0f}c")

    def display_daily_summary(self, summary: dict):
        """Print a rich terminal P&L summary."""
        pnl_color = "green" if summary["total_pnl"] >= 0 else "red"
        today_color = "green" if summary["today_pnl"] >= 0 else "red"
        week_color = "green" if summary["week_pnl"] >= 0 else "red"
        month_color = "green" if summary["month_pnl"] >= 0 else "red"

        panel_content = (
            f"[bold]P&L Overview[/bold]\n"
            f"  Total:   [{pnl_color}]${summary['total_pnl']:+.2f}[/{pnl_color}]\n"
            f"  Today:   [{today_color}]${summary['today_pnl']:+.2f}[/{today_color}]\n"
            f"  7-day:   [{week_color}]${summary['week_pnl']:+.2f}[/{week_color}]\n"
            f"  30-day:  [{month_color}]${summary['month_pnl']:+.2f}[/{month_color}]\n"
            f"\n[bold]Performance Metrics[/bold]\n"
            f"  Win Rate:      {summary['win_rate']:.1f}%\n"
            f"  Profit Factor: {summary['profit_factor']:.2f}\n"
            f"  Sharpe Ratio:  {summary['sharpe']:.2f}\n"
            f"  Max Drawdown:  ${summary['max_drawdown']:.2f}\n"
            f"  Avg Win:       ${summary['avg_win']:.4f}\n"
            f"  Avg Loss:      ${summary['avg_loss']:.4f}\n"
            f"\n[bold]Trade Counts[/bold]\n"
            f"  Settled: {summary['total_trades']}  |  Open: {summary['open_trades']}\n"
            f"  Best:  ${summary['best_trade']:+.4f}  |  Worst: ${summary['worst_trade']:+.4f}"
        )

        console.print(Panel(panel_content, title="Daily P&L Summary", border_style="cyan"))

        # Strategy breakdown
        breakdown = summary.get("strategy_breakdown", {})
        if breakdown:
            table = Table(title="Strategy Breakdown", show_lines=False)
            table.add_column("Strategy", style="cyan")
            table.add_column("Trades", justify="right")
            table.add_column("Wins", justify="right")
            table.add_column("Win %", justify="right")
            table.add_column("P&L", justify="right")

            for strat, data in sorted(breakdown.items()):
                pnl = data["pnl"]
                trades = data["trades"]
                wins = data["wins"]
                wr = wins / trades * 100 if trades > 0 else 0
                sc = "green" if pnl >= 0 else "red"
                table.add_row(
                    strat,
                    str(trades),
                    str(wins),
                    f"{wr:.0f}%",
                    f"[{sc}]${pnl:+.2f}[/{sc}]",
                )
            console.print(table)

    def display_forecast_accuracy(self, accuracy: list[ForecastAccuracy]):
        """Show forecast accuracy table."""
        valid = [a for a in accuracy if a.actual_temp is not None and a.error is not None]
        if not valid:
            console.print("[dim]No forecast accuracy data available yet.[/dim]")
            return

        table = Table(title="Forecast Accuracy (Weather)", show_lines=False)
        table.add_column("Date", style="dim", width=10)
        table.add_column("City", style="cyan", width=10)
        table.add_column("Forecast", justify="right", width=9)
        table.add_column("Actual", justify="right", width=9)
        table.add_column("Error", justify="right", width=9)

        for a in valid:
            err_color = "green" if abs(a.error) <= 2 else "yellow" if abs(a.error) <= 4 else "red"
            table.add_row(
                a.date,
                a.city,
                f"{a.forecast_temp:.0f}F",
                f"{a.actual_temp:.0f}F",
                f"[{err_color}]{a.error:+.1f}F[/{err_color}]",
            )

        console.print(table)

        # Summary stats
        errors = [abs(a.error) for a in valid]
        mae = sum(errors) / len(errors)
        rmse = math.sqrt(sum(e ** 2 for e in errors) / len(errors))
        console.print(f"\n[dim]MAE: {mae:.1f}F | RMSE: {rmse:.1f}F | Samples: {len(valid)}[/dim]")

    # -------------------------------------------------------------------
    # Main Entry Points
    # -------------------------------------------------------------------

    def check_settlements(self):
        """Check what has settled, compute P&L, and save results."""
        console.print("[cyan]Fetching fills and positions from Kalshi...[/cyan]\n")

        # Analyze fills and compute P&L
        records = self.analyze_fills()

        # Merge with existing history
        existing = self._load_existing_history()
        existing_tickers = {r.ticker for r in existing}

        # Update existing open trades that may have settled
        updated_existing = []
        for r in existing:
            if r.settlement_result == "open":
                market = self.get_market_details(r.ticker)
                if market and market.get("status") in ("settled", "finalized"):
                    result = market.get("result", "")
                    r.settlement_result = result
                    # Recalculate P&L
                    if result == "yes":
                        if r.side == "yes":
                            r.pnl = round(r.contracts * (100 - r.entry_price) / 100.0, 4)
                        else:
                            r.pnl = round(-r.contracts * (100 - r.entry_price) / 100.0, 4)
                    elif result == "no":
                        if r.side == "no":
                            r.pnl = round(r.contracts * r.entry_price / 100.0, 4)
                        else:
                            r.pnl = round(-r.contracts * r.entry_price / 100.0, 4)
                    # Fetch weather data for autoresearch feedback
                    if r.strategy == "weather":
                        if r.actual_temp is None:
                            r.actual_temp = self._fetch_weather_actual(r.ticker)
                        if r.forecast_temp is None:
                            r.forecast_temp = self._lookup_forecast_temp(r.ticker)
                    time.sleep(0.3)
            updated_existing.append(r)

        # Merge: existing (updated) + new
        new_records = [r for r in records if r.ticker not in existing_tickers]
        all_records = updated_existing + new_records

        # Recompute cumulative P&L
        cum = 0.0
        for r in all_records:
            cum += r.pnl
            r.cumulative_pnl = round(cum, 4)

        # Save trade history
        self.save_trade_history(all_records)

        # Display
        self.display_settlements(all_records)

        return all_records

    def daily_summary(self):
        """Full daily P&L report with all outputs."""
        console.print("[cyan]Generating daily P&L report...[/cyan]\n")

        records = self.check_settlements()

        # Compute summary
        summary = self.compute_daily_summary(records)
        self.display_daily_summary(summary)

        # Forecast accuracy
        accuracy = self.build_forecast_accuracy(records)
        if accuracy:
            self.save_forecast_accuracy(accuracy)
            self.display_forecast_accuracy(accuracy)

        # Save daily summary
        self.save_daily_summary(summary)

        # Feed back to AutoResearch
        self.save_real_outcomes(records, summary)

        console.print(f"\n[green]All outputs saved to: {config.OUTPUT_DIR}[/green]")
        console.print(f"[green]AutoResearch data: {self.real_outcomes_path}[/green]")

        # Trigger a quick research burst if new settlements were detected.
        # This runs *after* save_real_outcomes() so the loop trains on the
        # freshest data.  The trigger is a no-op if research is already running.
        self._trigger_post_settlement_research(records)

        return summary

    def _trigger_post_settlement_research(self, records: list) -> None:
        """
        Spawn a short AutoResearch run when new settlements have accumulated
        since the last trigger.

        Uses a persistent state file (autoresearch/settlement_trigger_state.json)
        to track how many trades were settled at the time of the last trigger.
        If the current count is higher, new settlements occurred and research
        is kicked off.  If research is already running the call is a no-op.
        """
        try:
            from autoresearch.post_settlement_research import (
                get_last_trigger_settled_count,
                trigger_post_settlement_research,
            )

            settled_total = sum(
                1 for r in records if r.settlement_result != "open"
            )
            last_count = get_last_trigger_settled_count()
            new_count = settled_total - last_count

            if new_count <= 0:
                return

            # Pick the research strategy based on what settled.
            # Weather is the only loop implemented today; sports/btc fall
            # through to the 'other' stub and are logged but not run.
            any_weather = any(
                r.strategy == "weather" and r.settlement_result != "open"
                for r in records
            )
            strategy = "weather" if any_weather else "other"

            triggered = trigger_post_settlement_research(
                new_settlement_count=new_count,
                settled_total=settled_total,
                strategy=strategy,
            )
            if triggered:
                console.print(
                    f"[cyan]Post-settlement research started: "
                    f"{new_count} new settlement(s) → running 15 iterations "
                    f"with real data...[/cyan]"
                )

        except Exception as exc:
            # Never let a trigger failure break the settlement report.
            console.print(
                f"[yellow]Post-settlement research trigger skipped: {exc}[/yellow]"
            )


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tracker = SettlementTracker()
    tracker.daily_summary()
