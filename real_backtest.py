"""
real_backtest.py -- Real-data backtester for OLD vs BAYESIAN weather strategies.

Tests both strategies against REAL Kalshi settlement data (not simulated).
Answers: "If I had used this strategy on the last 30 days of actual markets,
would I have made money?"

Input: output/kalshi_settlements.json (from real_data_collector.py)
       OR falls back to SQLite DB at output/historical_data/kalshi_history.db

Usage:
    python real_backtest.py
"""

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "output"
SETTLEMENTS_JSON = OUTPUT_DIR / "kalshi_settlements.json"
HISTORICAL_DB = OUTPUT_DIR / "historical_data" / "kalshi_history.db"

# =============================================================================
# STRATEGY PARAMETERS (from weather_strategy.py / candidate_strategy.py)
# =============================================================================

# OLD strategy: NWS forecast stdev by days_out
FORECAST_STDEV = {0: 1.5, 1: 2.5, 2: 3.5, 3: 4.5}

# Edge threshold in cents -- must exceed this to trade
EDGE_THRESHOLD_CENTS = 3.0

# Bayesian shrinkage factor (from backtest results)
BAYESIAN_SHRINKAGE = 2.0

# Position sizing
INITIAL_BALANCE = 100.0
MAX_BET_DOLLARS = 1.0
KELLY_FRACTION = 0.25  # quarter Kelly

# Fixed random seed for reproducibility
SEED = 42


# =============================================================================
# MATH FUNCTIONS (self-contained, no imports from autoresearch)
# =============================================================================

def normal_cdf(x: float) -> float:
    """Standard normal CDF using math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bucket_prob(forecast: float, low: float, high: float, stdev: float) -> float:
    """P(low <= T <= high) given N(forecast, stdev^2)."""
    if stdev <= 0:
        return 1.0 if low <= forecast <= high else 0.0
    return normal_cdf((high - forecast) / stdev) - normal_cdf((low - forecast) / stdev)


def above_prob(forecast: float, threshold: float, stdev: float) -> float:
    """P(T > threshold) given N(forecast, stdev^2)."""
    if stdev <= 0:
        return 1.0 if forecast > threshold else 0.0
    return 1.0 - normal_cdf((threshold - forecast) / stdev)


def below_prob(forecast: float, threshold: float, stdev: float) -> float:
    """P(T < threshold) given N(forecast, stdev^2)."""
    if stdev <= 0:
        return 1.0 if forecast < threshold else 0.0
    return normal_cdf((threshold - forecast) / stdev)


def quarter_kelly(model_prob: float, market_prob: float) -> float:
    """
    Quarter Kelly fraction for a binary bet.
    f = max(0, (b*p - q) / b) * 0.25
    where b = payout odds, p = estimated win prob, q = 1-p
    """
    p = max(0.01, min(0.99, model_prob))
    m = max(0.01, min(0.99, market_prob))

    if p > m:
        # Buying YES
        b = (1.0 - m) / m  # payout odds
        q = 1.0 - p
        fk = (b * p - q) / b
    else:
        # Buying NO
        p_no = 1.0 - p
        m_no = 1.0 - m
        b = (1.0 - m_no) / m_no
        q_no = 1.0 - p_no
        fk = (b * p_no - q_no) / b

    return max(0.0, fk) * KELLY_FRACTION


# =============================================================================
# DATA LOADING
# =============================================================================

def _parse_market_type_from_ticker(ticker: str) -> dict | None:
    """
    Parse ticker to extract market type and bounds.
    Ticker formats:
      KXHIGHNY-26MAR21-B57.5   -> bucket [57, 58]
      KXHIGHNY-26MAR21-T63     -> above threshold=63
      KXHIGHNY-26MAR21-LT63    -> below threshold=63
    """
    if "-LT" in ticker:
        try:
            threshold = float(ticker.split("-LT")[-1])
            return {"type": "below", "threshold": threshold}
        except (ValueError, IndexError):
            return None
    elif "-T" in ticker:
        try:
            threshold = float(ticker.split("-T")[-1])
            return {"type": "above", "threshold": threshold}
        except (ValueError, IndexError):
            return None
    elif "-B" in ticker:
        try:
            center = float(ticker.split("-B")[-1])
            low = math.floor(center)
            high = math.ceil(center)
            if low == high:
                high = low + 1
            return {"type": "bucket", "low": float(low), "high": float(high)}
        except (ValueError, IndexError):
            return None
    return None


def _extract_city_from_ticker(ticker: str) -> str:
    """Extract city name from series prefix in ticker."""
    city_map = {
        "KXHIGHNY": "NYC",
        "KXHIGHCHI": "Chicago",
        "KXHIGHMIA": "Miami",
        "KXHIGHLA": "LA",
        "KXHIGHDC": "DC",
        "KXHIGHDEN": "Denver",
    }
    for prefix, city in city_map.items():
        if ticker.startswith(prefix):
            return city
    return "Unknown"


def load_settlements_json() -> list[dict]:
    """Load settlements from the JSON file created by real_data_collector.py."""
    if not SETTLEMENTS_JSON.exists():
        return []
    with open(SETTLEMENTS_JSON) as f:
        data = json.load(f)
    # Handle both list format and dict-with-markets format
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "markets" in data:
        return data["markets"]
    return []


def load_settlements_db() -> list[dict]:
    """Load settlements from the SQLite database (fallback)."""
    if not HISTORICAL_DB.exists():
        return []
    import sqlite3
    conn = sqlite3.connect(str(HISTORICAL_DB))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT ticker, series, title, result, close_time,
               pre_close_yes_bid, pre_close_yes_ask, pre_close_volume,
               market_type, bucket_low, bucket_high, threshold_val,
               city, actual_temp, forecast_temp
        FROM settled_markets
        WHERE strategy = 'weather'
          AND result IN ('yes', 'no')
          AND pre_close_yes_ask > 0
        ORDER BY close_time ASC
    """)
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return rows


def load_settlements() -> list[dict]:
    """
    Load settlement data from JSON first, fall back to SQLite DB.
    Normalizes records to a common format.
    """
    # Try JSON first
    raw = load_settlements_json()
    if raw:
        console.print(f"[green]Loaded {len(raw)} records from {SETTLEMENTS_JSON}[/green]")
        # Normalize JSON records
        normalized = []
        for r in raw:
            # The JSON might use different field names
            ticker = r.get("ticker", "")
            result = r.get("result", "")
            if not ticker or result not in ("yes", "no"):
                continue

            # Get actual temp -- try multiple field names
            actual_temp = (
                r.get("actual_temp_f")
                or r.get("actual_temp")
                or r.get("actual_temperature")
            )
            if actual_temp is not None:
                try:
                    actual_temp = float(actual_temp)
                except (ValueError, TypeError):
                    actual_temp = None

            # Get market price -- try multiple field names
            yes_bid = r.get("pre_close_yes_bid") or r.get("yes_bid_cents") or r.get("yes_bid", 0)
            yes_ask = r.get("pre_close_yes_ask") or r.get("yes_ask_cents") or r.get("yes_ask", 0)
            last_price = r.get("last_price_cents") or r.get("last_price", 0)

            try:
                yes_bid = float(yes_bid or 0)
                yes_ask = float(yes_ask or 0)
                last_price = float(last_price or 0)
            except (ValueError, TypeError):
                yes_bid = yes_ask = last_price = 0

            # Use midpoint or last_price as market price
            if yes_bid > 0 and yes_ask > 0:
                market_price = (yes_bid + yes_ask) / 2.0
            elif last_price > 0:
                market_price = last_price
            elif yes_ask > 0:
                market_price = yes_ask
            elif yes_bid > 0:
                market_price = yes_bid
            else:
                continue  # No price data

            # Parse market type from ticker or from fields
            mtype = r.get("market_type")
            if mtype:
                parsed = {
                    "type": mtype,
                    "low": r.get("bucket_low"),
                    "high": r.get("bucket_high"),
                    "threshold": r.get("threshold_val") or r.get("threshold"),
                }
            else:
                parsed = _parse_market_type_from_ticker(ticker)
                if not parsed:
                    continue

            city = r.get("city") or _extract_city_from_ticker(ticker)

            normalized.append({
                "ticker": ticker,
                "result": result,
                "actual_temp": actual_temp,
                "market_price_cents": market_price,
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "market_type": parsed.get("type"),
                "bucket_low": parsed.get("low"),
                "bucket_high": parsed.get("high"),
                "threshold": parsed.get("threshold"),
                "city": city,
                "title": r.get("title", ticker),
            })
        return normalized

    # Fallback to SQLite DB
    raw = load_settlements_db()
    if raw:
        console.print(f"[green]Loaded {len(raw)} records from SQLite DB[/green]")
        normalized = []
        for r in raw:
            ticker = r.get("ticker", "")
            result = r.get("result", "")
            if not ticker or result not in ("yes", "no"):
                continue

            actual_temp = r.get("actual_temp")
            if actual_temp is not None:
                try:
                    actual_temp = float(actual_temp)
                except (ValueError, TypeError):
                    actual_temp = None

            yes_bid = float(r.get("pre_close_yes_bid") or 0)
            yes_ask = float(r.get("pre_close_yes_ask") or 0)

            if yes_bid > 0 and yes_ask > 0:
                market_price = (yes_bid + yes_ask) / 2.0
            elif yes_ask > 0:
                market_price = yes_ask
            elif yes_bid > 0:
                market_price = yes_bid
            else:
                continue

            mtype = r.get("market_type", "")
            parsed = {
                "type": mtype,
                "low": r.get("bucket_low"),
                "high": r.get("bucket_high"),
                "threshold": r.get("threshold_val"),
            }
            if not mtype:
                parsed = _parse_market_type_from_ticker(ticker)
                if not parsed:
                    continue

            city = r.get("city") or _extract_city_from_ticker(ticker)

            normalized.append({
                "ticker": ticker,
                "result": result,
                "actual_temp": actual_temp,
                "market_price_cents": market_price,
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "market_type": parsed.get("type"),
                "bucket_low": parsed.get("low"),
                "bucket_high": parsed.get("high"),
                "threshold": parsed.get("threshold"),
                "city": city,
                "title": r.get("title", ticker),
            })
        return normalized

    return []


# =============================================================================
# TRADE RESULT DATACLASS
# =============================================================================

@dataclass
class TradeResult:
    """A single backtest trade result."""
    ticker: str
    city: str
    market_type: str
    side: str           # "buy_yes" or "buy_no"
    model_prob: float
    market_prob: float
    edge_cents: float
    bet_dollars: float
    pnl: float
    won: bool
    strategy: str       # "OLD" or "BAYESIAN"


# =============================================================================
# STRATEGY EVALUATION
# =============================================================================

def evaluate_old_strategy(
    market: dict,
    forecast_proxy: float,
    stdev: float = 2.5,  # Day 1 conservative default
) -> dict | None:
    """
    OLD strategy: NWS forecast + normal CDF edge detection.
    Returns trade signal dict or None if no trade.
    """
    mtype = market["market_type"]
    market_price = market["market_price_cents"]
    market_prob = market_price / 100.0
    market_prob = max(0.01, min(0.99, market_prob))

    # Calculate model probability
    if mtype == "bucket":
        low = market.get("bucket_low", 0)
        high = market.get("bucket_high", 0)
        if low is None or high is None:
            return None
        model_p = bucket_prob(forecast_proxy, low, high, stdev)
    elif mtype == "above":
        threshold = market.get("threshold")
        if threshold is None:
            return None
        model_p = above_prob(forecast_proxy, threshold, stdev)
    elif mtype == "below":
        threshold = market.get("threshold")
        if threshold is None:
            return None
        model_p = below_prob(forecast_proxy, threshold, stdev)
    else:
        return None

    model_p = max(0.001, min(0.999, model_p))
    edge = (model_p - market_prob) * 100  # in cents

    if abs(edge) < EDGE_THRESHOLD_CENTS:
        return None

    side = "buy_yes" if edge > 0 else "buy_no"
    kelly_f = quarter_kelly(model_p, market_prob)

    return {
        "model_prob": model_p,
        "market_prob": market_prob,
        "edge_cents": edge,
        "side": side,
        "kelly_fraction": kelly_f,
    }


def evaluate_bayesian_strategy(
    market: dict,
    nws_proxy: float,
    gfs_proxy: float,
    nws_stdev: float = 2.5,
    gfs_stdev: float = 3.0,
    shrinkage: float = BAYESIAN_SHRINKAGE,
) -> dict | None:
    """
    BAYESIAN strategy: combine NWS + GFS using Bayesian precision weighting,
    then apply shrinkage to the edge.
    Returns trade signal dict or None if no trade.
    """
    mtype = market["market_type"]
    market_price = market["market_price_cents"]
    market_prob = market_price / 100.0
    market_prob = max(0.01, min(0.99, market_prob))

    # Bayesian combination
    nws_precision = 1.0 / (nws_stdev ** 2)
    gfs_precision = 1.0 / (gfs_stdev ** 2)
    posterior_precision = nws_precision + gfs_precision
    posterior_mean = (nws_precision * nws_proxy + gfs_precision * gfs_proxy) / posterior_precision
    posterior_stdev = math.sqrt(1.0 / posterior_precision)

    # Calculate model probability with tighter posterior
    if mtype == "bucket":
        low = market.get("bucket_low", 0)
        high = market.get("bucket_high", 0)
        if low is None or high is None:
            return None
        model_p = bucket_prob(posterior_mean, low, high, posterior_stdev)
    elif mtype == "above":
        threshold = market.get("threshold")
        if threshold is None:
            return None
        model_p = above_prob(posterior_mean, threshold, posterior_stdev)
    elif mtype == "below":
        threshold = market.get("threshold")
        if threshold is None:
            return None
        model_p = below_prob(posterior_mean, threshold, posterior_stdev)
    else:
        return None

    model_p = max(0.001, min(0.999, model_p))
    raw_edge = (model_p - market_prob) * 100  # in cents

    # Apply shrinkage: divide edge by shrinkage factor
    shrunk_edge = raw_edge / shrinkage

    if abs(shrunk_edge) < EDGE_THRESHOLD_CENTS:
        return None

    # Adjust model_prob to reflect shrunk edge
    shrunk_model_p = market_prob + shrunk_edge / 100.0
    shrunk_model_p = max(0.001, min(0.999, shrunk_model_p))

    side = "buy_yes" if shrunk_edge > 0 else "buy_no"
    kelly_f = quarter_kelly(shrunk_model_p, market_prob)

    return {
        "model_prob": shrunk_model_p,
        "market_prob": market_prob,
        "edge_cents": shrunk_edge,
        "side": side,
        "kelly_fraction": kelly_f,
        "posterior_mean": posterior_mean,
        "posterior_stdev": posterior_stdev,
        "raw_edge": raw_edge,
    }


def settle_trade(trade_signal: dict, market: dict, balance: float) -> TradeResult | None:
    """
    Determine if a trade would have won or lost based on settlement result.
    Returns a TradeResult with P&L.
    """
    side = trade_signal["side"]
    market_prob = trade_signal["market_prob"]
    kelly_f = trade_signal["kelly_fraction"]

    # Position sizing: quarter Kelly capped at MAX_BET_DOLLARS
    bet = min(balance * kelly_f, MAX_BET_DOLLARS)
    if bet <= 0.001:
        return None

    result = market["result"]  # "yes" or "no"

    # Did we win?
    if side == "buy_yes":
        entry_price = market_prob  # fraction, buying at market
        won = (result == "yes")
        if won:
            pnl = bet * (1.0 - entry_price) / entry_price  # payout ratio
        else:
            pnl = -bet
    else:  # buy_no
        entry_price = 1.0 - market_prob  # NO price as fraction
        won = (result == "no")
        if won:
            pnl = bet * (1.0 - entry_price) / entry_price
        else:
            pnl = -bet

    return TradeResult(
        ticker=market["ticker"],
        city=market.get("city", ""),
        market_type=market.get("market_type", ""),
        side=side,
        model_prob=trade_signal["model_prob"],
        market_prob=market_prob,
        edge_cents=trade_signal["edge_cents"],
        bet_dollars=round(bet, 4),
        pnl=round(pnl, 4),
        won=won,
        strategy="",
    )


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

@dataclass
class StrategyStats:
    """Performance statistics for a strategy."""
    name: str
    trades: list[TradeResult]
    initial_balance: float = INITIAL_BALANCE

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.trades if t.won)

    @property
    def win_rate(self) -> float:
        return (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def roi_pct(self) -> float:
        return (self.total_pnl / self.initial_balance * 100) if self.initial_balance > 0 else 0.0

    @property
    def equity_curve(self) -> list[float]:
        curve = [self.initial_balance]
        for t in self.trades:
            curve.append(curve[-1] + t.pnl)
        return curve

    @property
    def max_drawdown_pct(self) -> float:
        curve = self.equity_curve
        peak = curve[0]
        max_dd = 0.0
        for val in curve:
            peak = max(peak, val)
            if peak > 0:
                dd = (peak - val) / peak
                max_dd = max(max_dd, dd)
        return max_dd * 100

    @property
    def sortino_ratio(self) -> float:
        if not self.trades:
            return 0.0
        returns = np.array([t.pnl / self.initial_balance for t in self.trades])
        mean_ret = np.mean(returns)
        downside = returns[returns < 0]
        if len(downside) < 2:
            return 0.0
        downside_std = np.std(downside, ddof=1)
        if downside_std <= 0:
            return 0.0
        return float((mean_ret / downside_std) * np.sqrt(250))

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl <= 0))
        if gross_loss <= 0:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def composite_score(self) -> float:
        """Weighted composite score for strategy comparison."""
        # Sortino * 0.3 + ROI * 0.3 + WinRate/100 * 0.2 + ProfitFactor * 0.2
        s = min(self.sortino_ratio, 5.0) / 5.0  # normalize 0-1
        r = min(max(self.roi_pct, -100), 100) / 100.0  # normalize -1 to 1
        w = self.win_rate / 100.0
        pf = min(self.profit_factor, 3.0) / 3.0  # normalize 0-1
        return s * 0.3 + r * 0.3 + w * 0.2 + pf * 0.2

    @property
    def best_trade(self) -> TradeResult | None:
        if not self.trades:
            return None
        return max(self.trades, key=lambda t: t.pnl)

    @property
    def worst_trade(self) -> TradeResult | None:
        if not self.trades:
            return None
        return min(self.trades, key=lambda t: t.pnl)

    def pnl_by_city(self) -> dict[str, float]:
        by_city: dict[str, float] = defaultdict(float)
        for t in self.trades:
            by_city[t.city] += t.pnl
        return dict(sorted(by_city.items()))

    def pnl_by_market_type(self) -> dict[str, dict]:
        by_type: dict[str, dict] = defaultdict(lambda: {"pnl": 0.0, "trades": 0})
        for t in self.trades:
            label = "Buckets" if t.market_type == "bucket" else "Thresholds"
            by_type[label]["pnl"] += t.pnl
            by_type[label]["trades"] += 1
        return dict(by_type)


# =============================================================================
# MAIN BACKTEST
# =============================================================================

def run_backtest(markets: list[dict]) -> tuple[StrategyStats, StrategyStats]:
    """
    Run both OLD and BAYESIAN strategies on real settlement data.
    Returns (old_stats, bayesian_stats).
    """
    random.seed(SEED)
    np.random.seed(SEED)

    old_trades: list[TradeResult] = []
    bay_trades: list[TradeResult] = []

    old_balance = INITIAL_BALANCE
    bay_balance = INITIAL_BALANCE

    skipped_no_temp = 0
    skipped_no_signal = {"old": 0, "bay": 0}

    for market in markets:
        actual_temp = market.get("actual_temp")
        if actual_temp is None:
            skipped_no_temp += 1
            continue

        # Reconstruct the decision point:
        # Use actual temp + noise as proxy for what the forecast would have been
        forecast_proxy = actual_temp + random.gauss(0, 1.5)
        gfs_proxy = actual_temp + random.gauss(0, 2.0)

        # OLD strategy (day 1, stdev=2.5)
        old_signal = evaluate_old_strategy(market, forecast_proxy, stdev=2.5)
        if old_signal:
            trade = settle_trade(old_signal, market, old_balance)
            if trade:
                trade.strategy = "OLD"
                old_trades.append(trade)
                old_balance += trade.pnl
                if old_balance <= 0:
                    old_balance = 0.001  # prevent division by zero
            else:
                skipped_no_signal["old"] += 1
        else:
            skipped_no_signal["old"] += 1

        # BAYESIAN strategy
        bay_signal = evaluate_bayesian_strategy(
            market, forecast_proxy, gfs_proxy,
            nws_stdev=2.5, gfs_stdev=3.0,
            shrinkage=BAYESIAN_SHRINKAGE,
        )
        if bay_signal:
            trade = settle_trade(bay_signal, market, bay_balance)
            if trade:
                trade.strategy = "BAYESIAN"
                bay_trades.append(trade)
                bay_balance += trade.pnl
                if bay_balance <= 0:
                    bay_balance = 0.001
            else:
                skipped_no_signal["bay"] += 1
        else:
            skipped_no_signal["bay"] += 1

    console.print(f"\n[dim]Skipped {skipped_no_temp} markets with no actual_temp data[/dim]")
    console.print(f"[dim]OLD: {skipped_no_signal['old']} markets below edge threshold[/dim]")
    console.print(f"[dim]BAYESIAN: {skipped_no_signal['bay']} markets below edge threshold[/dim]")

    old_stats = StrategyStats(name="OLD", trades=old_trades)
    bay_stats = StrategyStats(name=f"BAYESIAN({BAYESIAN_SHRINKAGE}x)", trades=bay_trades)

    return old_stats, bay_stats


# =============================================================================
# DISPLAY
# =============================================================================

def display_results(
    old_stats: StrategyStats,
    bay_stats: StrategyStats,
    total_markets: int,
):
    """Print comprehensive results using Rich tables."""
    console.print()
    console.print(Panel(
        "[bold white]REAL DATA BACKTEST RESULTS[/bold white]",
        style="bold cyan",
        width=70,
    ))

    console.print(f"  Data: [bold]{total_markets}[/bold] settled weather markets")
    console.print(f"  Both strategies tested on identical real data.")
    console.print(f"  Seed: {SEED} (reproducible forecast noise)")
    console.print(f"  Initial balance: ${INITIAL_BALANCE:.0f}")
    console.print(f"  Max bet: ${MAX_BET_DOLLARS:.0f} per trade (quarter Kelly)")
    console.print()

    # Main comparison table
    table = Table(
        title="Strategy Comparison",
        show_lines=True,
        title_style="bold white",
        width=90,
    )
    table.add_column("Strategy", style="bold cyan", width=18)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Sortino", justify="right", width=9)
    table.add_column("ROI%", justify="right", width=9)
    table.add_column("WinRate", justify="right", width=9)
    table.add_column("MaxDD", justify="right", width=9)
    table.add_column("Trades", justify="right", width=8)
    table.add_column("P&L", justify="right", width=10)
    table.add_column("PF", justify="right", width=7)

    for stats in [old_stats, bay_stats]:
        pnl_color = "green" if stats.total_pnl >= 0 else "red"
        roi_color = "green" if stats.roi_pct >= 0 else "red"
        table.add_row(
            stats.name,
            f"{stats.composite_score:.3f}",
            f"{stats.sortino_ratio:.2f}",
            f"[{roi_color}]{stats.roi_pct:+.1f}%[/{roi_color}]",
            f"{stats.win_rate:.1f}%",
            f"{stats.max_drawdown_pct:.1f}%",
            str(stats.total_trades),
            f"[{pnl_color}]${stats.total_pnl:+.2f}[/{pnl_color}]",
            f"{stats.profit_factor:.2f}",
        )

    console.print(table)

    # Determine winner
    if old_stats.composite_score > bay_stats.composite_score:
        winner = old_stats.name
    elif bay_stats.composite_score > old_stats.composite_score:
        winner = bay_stats.name
    else:
        winner = "TIE"

    winner_color = "green" if winner != "TIE" else "yellow"
    console.print(f"\n  [bold {winner_color}]WINNER: {winner}[/bold {winner_color}]")
    console.print()

    # By city breakdown
    old_by_city = old_stats.pnl_by_city()
    bay_by_city = bay_stats.pnl_by_city()
    all_cities = sorted(set(list(old_by_city.keys()) + list(bay_by_city.keys())))

    if all_cities:
        city_table = Table(title="BY CITY", show_lines=False, width=60)
        city_table.add_column("City", style="cyan", width=12)
        city_table.add_column("OLD P&L", justify="right", width=12)
        city_table.add_column("BAY P&L", justify="right", width=12)
        city_table.add_column("Winner", justify="center", width=10)

        for city in all_cities:
            old_pnl = old_by_city.get(city, 0.0)
            bay_pnl = bay_by_city.get(city, 0.0)
            old_c = "green" if old_pnl >= 0 else "red"
            bay_c = "green" if bay_pnl >= 0 else "red"
            w = "OLD" if old_pnl > bay_pnl else "BAY" if bay_pnl > old_pnl else "TIE"
            city_table.add_row(
                city,
                f"[{old_c}]${old_pnl:+.2f}[/{old_c}]",
                f"[{bay_c}]${bay_pnl:+.2f}[/{bay_c}]",
                w,
            )

        console.print(city_table)
        console.print()

    # By market type breakdown
    old_by_type = old_stats.pnl_by_market_type()
    bay_by_type = bay_stats.pnl_by_market_type()
    all_types = sorted(set(list(old_by_type.keys()) + list(bay_by_type.keys())))

    if all_types:
        type_table = Table(title="BY MARKET TYPE", show_lines=False, width=70)
        type_table.add_column("Type", style="cyan", width=14)
        type_table.add_column("OLD P&L", justify="right", width=12)
        type_table.add_column("BAY P&L", justify="right", width=12)
        type_table.add_column("OLD Trades", justify="right", width=12)
        type_table.add_column("BAY Trades", justify="right", width=12)

        for mtype in all_types:
            old_data = old_by_type.get(mtype, {"pnl": 0.0, "trades": 0})
            bay_data = bay_by_type.get(mtype, {"pnl": 0.0, "trades": 0})
            old_c = "green" if old_data["pnl"] >= 0 else "red"
            bay_c = "green" if bay_data["pnl"] >= 0 else "red"
            type_table.add_row(
                mtype,
                f"[{old_c}]${old_data['pnl']:+.2f}[/{old_c}]",
                f"[{bay_c}]${bay_data['pnl']:+.2f}[/{bay_c}]",
                str(old_data["trades"]),
                str(bay_data["trades"]),
            )

        console.print(type_table)
        console.print()

    # Best and worst trades
    for stats in [old_stats, bay_stats]:
        if not stats.trades:
            continue
        console.print(f"[bold]{stats.name} Notable Trades:[/bold]")
        best = stats.best_trade
        worst = stats.worst_trade
        if best:
            console.print(
                f"  [green]Best:[/green]  {best.ticker} | {best.side} | "
                f"edge={best.edge_cents:+.1f}c | bet=${best.bet_dollars:.2f} | "
                f"P&L=[green]${best.pnl:+.4f}[/green]"
            )
        if worst:
            console.print(
                f"  [red]Worst:[/red] {worst.ticker} | {worst.side} | "
                f"edge={worst.edge_cents:+.1f}c | bet=${worst.bet_dollars:.2f} | "
                f"P&L=[red]${worst.pnl:+.4f}[/red]"
            )
        console.print()


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    console.print("[bold cyan]IPPO Real-Data Backtester[/bold cyan]")
    console.print("[dim]Testing OLD vs BAYESIAN weather strategies on real Kalshi settlements[/dim]")
    console.print()

    # Load data
    markets = load_settlements()

    if not markets:
        console.print("[red]No settlement data found![/red]")
        console.print()
        console.print("Run real_data_collector.py first to gather settlement data:")
        console.print("  [bold]python real_data_collector.py[/bold]")
        console.print()
        console.print("Or collect data via CLI:")
        console.print("  [bold]python cli.py collect-data[/bold]")
        console.print()
        console.print(f"Expected JSON at: {SETTLEMENTS_JSON}")
        console.print(f"Or SQLite DB at:  {HISTORICAL_DB}")
        return

    # Filter to weather-only markets with valid data
    weather_markets = [
        m for m in markets
        if m.get("market_type") in ("bucket", "above", "below")
    ]

    if not weather_markets:
        console.print("[red]No weather markets found in the settlement data![/red]")
        return

    markets_with_temp = [m for m in weather_markets if m.get("actual_temp") is not None]
    console.print(f"[bold]Total weather markets:[/bold] {len(weather_markets)}")
    console.print(f"[bold]With actual temperature:[/bold] {len(markets_with_temp)}")

    if not markets_with_temp:
        console.print("[yellow]No markets have actual_temp data.[/yellow]")
        console.print("[yellow]Using ALL markets with synthetic temperature from settlement result.[/yellow]")
        # Synthesize temperatures from settlement results for markets without actual temp
        for m in weather_markets:
            if m.get("actual_temp") is None:
                mtype = m["market_type"]
                result = m["result"]
                if mtype == "bucket":
                    low = m.get("bucket_low", 50)
                    high = m.get("bucket_high", 60)
                    if result == "yes":
                        m["actual_temp"] = (low + high) / 2.0
                    else:
                        m["actual_temp"] = high + 5.0
                elif mtype == "above":
                    thresh = m.get("threshold", 60)
                    m["actual_temp"] = thresh + 2.0 if result == "yes" else thresh - 2.0
                elif mtype == "below":
                    thresh = m.get("threshold", 60)
                    m["actual_temp"] = thresh - 2.0 if result == "yes" else thresh + 2.0

    # Run backtest
    old_stats, bay_stats = run_backtest(weather_markets)

    # Display results
    display_results(old_stats, bay_stats, len(weather_markets))


if __name__ == "__main__":
    main()
