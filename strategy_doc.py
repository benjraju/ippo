"""
strategy_doc.py -- Generate a living STRATEGY.md document.

Reads current strategy parameters, AutoResearch experiment results,
trade history, and real outcomes to produce a plain-English strategy
document that anyone can understand.
"""

import json
import csv
from pathlib import Path
from datetime import datetime

import config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STRATEGY_FILE = config.AUTORESEARCH_DIR / "candidate_strategy.py"
RESULTS_LOG = config.AUTORESEARCH_DIR / "results.log"
TRADE_HISTORY = config.OUTPUT_DIR / "trade_history.csv"
REAL_OUTCOMES = config.AUTORESEARCH_DIR / "real_outcomes.json"
OUTPUT_MD = config.PROJECT_ROOT / "STRATEGY.md"

# ---------------------------------------------------------------------------
# Plain-English parameter names (from alerts.py _PARAM_EXPLAIN)
# ---------------------------------------------------------------------------
_PARAM_EXPLAIN = {
    "FORECAST_STDEV_0": {
        "name": "Same-day forecast confidence",
        "meaning": "How uncertain we think today's weather forecast is (in degrees F). Lower means we trust the forecast more and trade more aggressively.",
    },
    "FORECAST_STDEV_1": {
        "name": "Tomorrow forecast confidence",
        "meaning": "Uncertainty for tomorrow's forecast. Lower means tighter model, more trades on tomorrow's markets.",
    },
    "FORECAST_STDEV_2": {
        "name": "2-day forecast confidence",
        "meaning": "Uncertainty for day-after-tomorrow. Higher means we're more cautious on these markets.",
    },
    "FORECAST_STDEV_3": {
        "name": "3-day forecast confidence",
        "meaning": "Uncertainty for 3-day-out forecasts. Higher means fewer trades on longer-dated markets.",
    },
    "EDGE_THRESHOLD_CENTS": {
        "name": "Minimum edge to trade",
        "meaning": "The smallest advantage (in cents) we need before placing a bet. Higher = fewer but higher-quality trades.",
    },
    "CONTRACTS_PER_TRADE": {
        "name": "Trade size (contracts)",
        "meaning": "How many contracts we buy per trade. More contracts = bigger bets.",
    },
    "NWS_OFFICIAL_WEIGHT": {
        "name": "Weather source blend",
        "meaning": "How much we trust the official NWS forecast vs. the GFS ensemble model. 1.0 = all NWS, 0.0 = all ensemble.",
    },
    "CITY_WEIGHT_NYC": {
        "name": "NYC trading weight",
        "meaning": "How aggressively we trade NYC weather markets relative to baseline.",
    },
    "CITY_WEIGHT_CHI": {
        "name": "Chicago trading weight",
        "meaning": "How aggressively we trade Chicago weather markets relative to baseline.",
    },
    "CITY_WEIGHT_MIA": {
        "name": "Miami trading weight",
        "meaning": "How aggressively we trade Miami weather markets relative to baseline.",
    },
    "CITY_WEIGHT_LA": {
        "name": "LA trading weight",
        "meaning": "How aggressively we trade LA weather markets relative to baseline.",
    },
    "CITY_WEIGHT_DC": {
        "name": "DC trading weight",
        "meaning": "How aggressively we trade DC weather markets relative to baseline.",
    },
    "CITY_WEIGHT_DEN": {
        "name": "Denver trading weight",
        "meaning": "How aggressively we trade Denver weather markets relative to baseline.",
    },
    "BUCKET_MULTIPLIER": {
        "name": "Bucket market preference",
        "meaning": "How much we favor 'temperature falls in X-Y range' markets. Above 1.0 = trade more of these.",
    },
    "THRESHOLD_MULTIPLIER": {
        "name": "Threshold market preference",
        "meaning": "How much we favor 'temperature above/below X' markets. Above 1.0 = trade more of these.",
    },
    "HIGH_CONFIDENCE_EDGE": {
        "name": "High confidence threshold",
        "meaning": "Edge (in cents) needed to flag a trade as 'high confidence' and size up. Lower = more aggressive sizing.",
    },
    "MEDIUM_CONFIDENCE_EDGE": {
        "name": "Medium confidence threshold",
        "meaning": "Edge (in cents) needed to flag a trade as 'medium confidence'. Lower = we take more borderline trades.",
    },
    "MAX_POSITION_DOLLARS": {
        "name": "Max position per market ($)",
        "meaning": "The most dollars we will put into any single market.",
    },
    "MIN_VOLUME": {
        "name": "Minimum market volume",
        "meaning": "We skip markets with fewer contracts traded than this (too illiquid).",
    },
    "TIGHT_ENSEMBLE_THRESHOLD": {
        "name": "Ensemble agreement threshold",
        "meaning": "When forecast models agree within this many degrees, we consider the forecast extra reliable.",
    },
    "TIGHT_ENSEMBLE_MULTIPLIER": {
        "name": "Ensemble agreement bet boost",
        "meaning": "When models agree, multiply our bet size by this amount. Higher = bigger bets on consensus forecasts.",
    },
    "COPY_MIN_CONFIDENCE": {
        "name": "Copy-trade confidence filter",
        "meaning": "Minimum confidence to auto-copy a whale trade. Higher = only copy the best signals.",
    },
    "COPY_DELAY_SECONDS": {
        "name": "Copy-trade reaction speed",
        "meaning": "How many seconds we wait after seeing a whale trade before copying. Lower = faster reaction.",
    },
    "COPY_SIZE_FRACTION": {
        "name": "Copy-trade position fraction",
        "meaning": "What fraction of the whale's position we copy. 0.01 = 1% of their size.",
    },
    "COPY_MAX_TRADES_PER_DAY": {
        "name": "Max copy trades per day",
        "meaning": "Daily cap on how many whale trades we copy.",
    },
    "COPY_MIN_WHALE_SIZE": {
        "name": "Min whale trade size ($)",
        "meaning": "We only copy trades above this dollar amount.",
    },
    "COPY_TRADER_WEIGHT_0x8dxd": {
        "name": "Whale trader trust weight",
        "meaning": "How much we trust signals from this specific tracked trader. 1.0 = full trust.",
    },
    "COPY_CRYPTO_MULT": {
        "name": "Copy crypto multiplier",
        "meaning": "Size multiplier when copying crypto trades. 0 = skip, 1 = normal, 2 = double.",
    },
    "COPY_POLITICS_MULT": {
        "name": "Copy politics multiplier",
        "meaning": "Size multiplier when copying politics trades. 0 = skip entirely.",
    },
    "COPY_SPORTS_MULT": {
        "name": "Copy sports multiplier",
        "meaning": "Size multiplier when copying sports trades. 0.5 = half size.",
    },
}

# Which group each param belongs to
_PARAM_GROUPS = {
    "Forecast Model": [
        "FORECAST_STDEV_0", "FORECAST_STDEV_1", "FORECAST_STDEV_2",
        "FORECAST_STDEV_3", "NWS_OFFICIAL_WEIGHT",
        "TIGHT_ENSEMBLE_THRESHOLD", "TIGHT_ENSEMBLE_MULTIPLIER",
    ],
    "Trading Rules": [
        "EDGE_THRESHOLD_CENTS", "CONTRACTS_PER_TRADE",
        "HIGH_CONFIDENCE_EDGE", "MEDIUM_CONFIDENCE_EDGE",
        "MAX_POSITION_DOLLARS", "MIN_VOLUME",
    ],
    "City Weights": [
        "CITY_WEIGHT_NYC", "CITY_WEIGHT_CHI", "CITY_WEIGHT_MIA",
        "CITY_WEIGHT_LA", "CITY_WEIGHT_DC", "CITY_WEIGHT_DEN",
    ],
    "Market Preferences": [
        "BUCKET_MULTIPLIER", "THRESHOLD_MULTIPLIER",
    ],
    "Copy-Trading": [
        "COPY_MIN_CONFIDENCE", "COPY_DELAY_SECONDS", "COPY_SIZE_FRACTION",
        "COPY_MAX_TRADES_PER_DAY", "COPY_MIN_WHALE_SIZE",
        "COPY_TRADER_WEIGHT_0x8dxd", "COPY_CRYPTO_MULT",
        "COPY_POLITICS_MULT", "COPY_SPORTS_MULT",
    ],
}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _read_strategy_params() -> dict:
    """Parse candidate_strategy.py and return {PARAM_NAME: value}."""
    params = {}
    if not STRATEGY_FILE.exists():
        return params
    for line in STRATEGY_FILE.read_text().splitlines():
        stripped = line.strip()
        if (
            "=" in stripped
            and not stripped.startswith("#")
            and not stripped.startswith("def ")
            and not stripped.startswith("return")
            and not stripped.startswith("\"")
            and not stripped.startswith("'")
        ):
            parts = stripped.split("=", 1)
            name = parts[0].strip()
            raw = parts[1].strip().split("#")[0].strip()
            try:
                params[name] = float(raw)
            except (ValueError, IndexError):
                pass
    return params


def _read_results_log() -> list[dict]:
    """Read results.log (JSON lines) into a list of dicts."""
    experiments = []
    if not RESULTS_LOG.exists():
        return experiments
    for line in RESULTS_LOG.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            experiments.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return experiments


def _read_trade_history() -> list[dict]:
    """Read trade_history.csv into a list of dicts."""
    trades = []
    if not TRADE_HISTORY.exists():
        return trades
    with open(TRADE_HISTORY, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append(row)
    return trades


def _read_real_outcomes() -> dict:
    """Read real_outcomes.json."""
    if not REAL_OUTCOMES.exists():
        return {}
    try:
        return json.loads(REAL_OUTCOMES.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _section_how_it_works() -> str:
    return """## How Ippo Makes Money

Ippo is a weather forecast arbitrage bot. Here's how it works in plain English:

1. **We get weather forecasts** from the National Weather Service (NWS) and GFS
   ensemble models for cities like NYC, Chicago, Miami, LA, DC, and Denver.

2. **We check prediction markets** on Kalshi, where people bet on questions like
   "Will the high temperature in Miami be above 80 degrees tomorrow?"

3. **We compare our forecast to the market price.** If our weather model says
   there's a 70% chance of hitting 80 degrees but the market is only pricing it
   at 55%, that's a 15-cent edge -- the market is underpricing the outcome.

4. **When we spot a big enough disagreement, we place a bet.** We buy the
   underpriced side and wait for the weather to actually happen.

5. **The market settles based on real weather data.** If we were right, we profit.
   Over many trades, our better forecasts should translate into consistent gains.

We also copy-trade large "whale" traders on other Kalshi markets (crypto, sports)
when their signals meet our confidence threshold.
"""


def _section_current_settings(params: dict) -> str:
    lines = ["## Current Strategy Settings\n"]
    lines.append(f"*Last updated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}*\n")

    for group_name, param_keys in _PARAM_GROUPS.items():
        lines.append(f"### {group_name}\n")
        lines.append("| Setting | Current Value | What It Means |")
        lines.append("|---------|:------------:|---------------|")
        for key in param_keys:
            val = params.get(key)
            if val is None:
                continue
            info = _PARAM_EXPLAIN.get(key, {})
            nice_name = info.get("name", key)
            meaning = info.get("meaning", "")
            # Format the value nicely
            if val == int(val):
                val_str = str(int(val))
            else:
                val_str = f"{val:.2f}".rstrip("0").rstrip(".")
            lines.append(f"| {nice_name} | {val_str} | {meaning} |")
        lines.append("")

    return "\n".join(lines)


def _section_research(experiments: list[dict]) -> str:
    lines = ["## What AutoResearch Has Learned\n"]

    if not experiments:
        lines.append("No experiments have been run yet.\n")
        return "\n".join(lines)

    total = len(experiments)
    kept = [e for e in experiments if e.get("kept")]
    hit_rate = (len(kept) / total * 100) if total else 0

    lines.append(f"AutoResearch runs experiments overnight, tweaking one parameter at a")
    lines.append(f"time and backtesting to see if the change improves performance.\n")
    lines.append(f"- **Total experiments run:** {total:,}")
    lines.append(f"- **Improvements found:** {len(kept)}")
    lines.append(f"- **Hit rate:** {hit_rate:.1f}% of experiments improved the strategy\n")

    # Find top 5 most impactful changes (kept=true, sorted by score descending)
    kept_sorted = sorted(kept, key=lambda e: e.get("metrics", {}).get("score", -999), reverse=True)
    top5 = kept_sorted[:5]

    if top5:
        lines.append("### Top Changes That Improved the Strategy\n")
        for i, exp in enumerate(top5, 1):
            param = exp.get("parameter", "?")
            old = exp.get("old_value", "?")
            new = exp.get("new_value", "?")
            metrics = exp.get("metrics", {})
            score = metrics.get("score", 0)
            roi = metrics.get("roi_pct", 0)
            win_rate = metrics.get("win_rate", 0)
            max_dd = metrics.get("max_dd_pct", 0)

            info = _PARAM_EXPLAIN.get(param, {})
            nice_name = info.get("name", param)

            # Determine direction for explanation
            try:
                direction = "up" if float(new) > float(old) else "down"
            except (ValueError, TypeError):
                direction = "up"

            # Build a plain-English explanation of the change
            if direction == "up":
                arrow = "increased"
            else:
                arrow = "decreased"

            lines.append(
                f"**{i}. {nice_name}** -- {arrow} from {old} to {new}  "
            )
            lines.append(
                f"   Backtest result: {roi:.0f}% return, {win_rate:.1f}% win rate, "
                f"{max_dd:.1f}% max drawdown (score: {score:.2f})"
            )
            # Add a short plain-English note
            meaning = info.get("meaning", "")
            if meaning:
                lines.append(f"   *{meaning}*\n")
            else:
                lines.append("")

    # Show parameters that were tested but rejected
    rejected = [e for e in experiments if not e.get("kept")]
    if rejected:
        # Count which params were tested most
        param_counts = {}
        for e in experiments:
            p = e.get("parameter", "?")
            param_counts[p] = param_counts.get(p, 0) + 1
        top_tested = sorted(param_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        lines.append("### Most-Tested Parameters\n")
        lines.append("| Parameter | Times Tested |")
        lines.append("|-----------|:------------:|")
        for p, count in top_tested:
            nice = _PARAM_EXPLAIN.get(p, {}).get("name", p)
            lines.append(f"| {nice} | {count} |")
        lines.append("")

    return "\n".join(lines)


def _section_performance(trades: list[dict], outcomes: dict) -> str:
    lines = ["## Real Performance\n"]

    if not trades:
        lines.append("No trades recorded yet.\n")
        return "\n".join(lines)

    # Separate settled vs open trades
    settled = [t for t in trades if t.get("settlement_result", "open") != "open"]
    open_trades = [t for t in trades if t.get("settlement_result", "open") == "open"]

    # Use real_outcomes.json summary if available
    if outcomes and "total_settled_trades" in outcomes:
        total_pnl = outcomes.get("total_pnl", 0)
        total_settled = outcomes.get("total_settled_trades", 0)
        win_rate = outcomes.get("win_rate", 0)
        profit_factor = outcomes.get("profit_factor", 0)
        max_dd = outcomes.get("max_drawdown", 0)

        sign = "+" if total_pnl >= 0 else ""
        lines.append(f"### Overall (from real_outcomes.json)\n")
        lines.append(f"- **Total settled trades:** {total_settled}")
        lines.append(f"- **Total P&L:** {sign}${total_pnl:,.2f}")
        lines.append(f"- **Win rate:** {win_rate:.1f}%")
        lines.append(f"- **Profit factor:** {profit_factor:.2f}")
        lines.append(f"- **Worst drawdown:** ${max_dd:,.2f}\n")

        # Strategy breakdown
        breakdown = outcomes.get("strategy_breakdown", {})
        if breakdown:
            lines.append("### By Strategy\n")
            lines.append("| Strategy | Trades | Wins | P&L |")
            lines.append("|----------|:------:|:----:|----:|")
            for strat, data in sorted(breakdown.items()):
                s_pnl = data.get("pnl", 0)
                s_trades = data.get("trades", 0)
                s_wins = data.get("wins", 0)
                s_sign = "+" if s_pnl >= 0 else ""
                s_wr = (s_wins / s_trades * 100) if s_trades else 0
                lines.append(
                    f"| {strat.title()} | {s_trades} | {s_wins} ({s_wr:.0f}%) | {s_sign}${s_pnl:,.2f} |"
                )
            lines.append("")
    else:
        # Fall back to trade_history.csv data
        total_pnl = 0
        wins = 0
        for t in settled:
            try:
                pnl = float(t.get("pnl", 0))
            except (ValueError, TypeError):
                pnl = 0
            total_pnl += pnl
            if pnl > 0:
                wins += 1
        win_rate = (wins / len(settled) * 100) if settled else 0
        sign = "+" if total_pnl >= 0 else ""
        lines.append(f"- **Settled trades:** {len(settled)}")
        lines.append(f"- **Total P&L:** {sign}${total_pnl:,.2f}")
        lines.append(f"- **Win rate:** {win_rate:.1f}%\n")

    # Check for weather-specific performance
    weather_trades = [t for t in settled if t.get("strategy") == "weather"]
    if weather_trades:
        w_pnl = sum(float(t.get("pnl", 0)) for t in weather_trades)
        w_wins = sum(1 for t in weather_trades if float(t.get("pnl", 0)) > 0)
        w_wr = (w_wins / len(weather_trades) * 100) if weather_trades else 0
        w_sign = "+" if w_pnl >= 0 else ""
        lines.append(f"### Weather Strategy Specifically\n")
        lines.append(f"- **Settled weather trades:** {len(weather_trades)}")
        lines.append(f"- **Weather P&L:** {w_sign}${w_pnl:,.2f}")
        lines.append(f"- **Weather win rate:** {w_wr:.1f}%\n")

        # Check for forecast accuracy
        has_forecast = any(t.get("forecast_temp") and t.get("actual_temp") for t in weather_trades)
        if has_forecast:
            errors = []
            for t in weather_trades:
                try:
                    fc = float(t["forecast_temp"])
                    ac = float(t["actual_temp"])
                    errors.append(abs(fc - ac))
                except (ValueError, TypeError, KeyError):
                    pass
            if errors:
                avg_err = sum(errors) / len(errors)
                lines.append(f"- **Average forecast error:** {avg_err:.1f} degrees F")
                lines.append(f"- **Trades with forecast data:** {len(errors)}\n")
        else:
            lines.append("*No weather trades have settled with actual temperature data yet.  ")
            lines.append("Backtest vs. real comparison will appear once weather trades settle.*\n")
    else:
        lines.append("### Weather Strategy\n")
        lines.append("No weather trades have settled yet. Weather trades are currently open ")
        lines.append("and will settle when the actual temperature is recorded.\n")

    return "\n".join(lines)


def _section_open_positions(trades: list[dict]) -> str:
    lines = ["## What's Trading Next\n"]

    open_trades = [t for t in trades if t.get("settlement_result", "open") == "open"]

    if not open_trades:
        lines.append("No open positions right now.\n")
        return "\n".join(lines)

    lines.append(f"We have **{len(open_trades)} open positions** waiting to settle:\n")

    # Group by strategy
    by_strat = {}
    for t in open_trades:
        strat = t.get("strategy", "unknown")
        by_strat.setdefault(strat, []).append(t)

    for strat, strat_trades in sorted(by_strat.items()):
        lines.append(f"### {strat.title()} Markets\n")
        lines.append("| Market | Our Bet | Contracts | Entry Price | What Needs to Happen |")
        lines.append("|--------|:-------:|:---------:|:-----------:|----------------------|")
        for t in strat_trades:
            title = t.get("title", t.get("ticker", "?"))
            # Truncate long titles
            if len(title) > 60:
                title = title[:57] + "..."
            side = t.get("side", "?")
            contracts = t.get("contracts", "?")
            entry = t.get("entry_price", "?")
            try:
                entry_f = float(entry)
                entry_str = f"{entry_f:.0f}c"
            except (ValueError, TypeError):
                entry_str = str(entry)

            # Build the "what needs to happen" text
            if side == "yes":
                outcome = "This needs to happen (settle YES) for us to win"
            else:
                outcome = "This must NOT happen (settle NO) for us to win"

            lines.append(f"| {title} | {side.upper()} | {contracts} | {entry_str} | {outcome} |")
        lines.append("")

    return "\n".join(lines)


def _section_risk_controls() -> str:
    return f"""## Risk Controls

These safety limits protect the account from big losses:

- **Maximum bet per trade:** ${config.MAX_BET_DOLLARS:.0f} -- no single trade can risk
  more than this, no matter how good the edge looks.

- **Daily loss cap:** {config.MAX_DAILY_LOSS_PCT * 100:.0f}% of the account -- if we lose
  this much in one day, all trading stops until tomorrow. On a $100 account,
  that's ${config.ACCOUNT_BALANCE * config.MAX_DAILY_LOSS_PCT:.0f}.

- **Quarter Kelly sizing:** We use the Kelly Criterion (a math formula for
  optimal bet sizing) but only bet 25% of what Kelly suggests. This is very
  conservative -- it means slower growth but much lower risk of ruin.

- **Position limit:** Maximum {config.MAX_OPEN_POSITIONS} open positions at once.

- **Minimum edge:** We need at least a {config.MIN_EDGE_THRESHOLD * 100:.0f}-cent edge
  before placing any trade. No edge = no trade.

- **Minimum volume:** We only trade markets with at least {config.MIN_MARKET_VOLUME}
  contracts traded, so we can always get in and out.
"""


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_strategy_doc():
    """Generate STRATEGY.md from current data."""
    params = _read_strategy_params()
    experiments = _read_results_log()
    trades = _read_trade_history()
    outcomes = _read_real_outcomes()

    now = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    sections = [
        f"# Ippo Strategy Document\n",
        f"*Auto-generated on {now}. This document updates every time "
        f"`strategy_doc.py` runs.*\n",
        "---\n",
        _section_how_it_works(),
        "---\n",
        _section_current_settings(params),
        "---\n",
        _section_research(experiments),
        "---\n",
        _section_performance(trades, outcomes),
        "---\n",
        _section_open_positions(trades),
        "---\n",
        _section_risk_controls(),
    ]

    md = "\n".join(sections)
    OUTPUT_MD.write_text(md)
    print(f"Generated {OUTPUT_MD} ({len(md):,} bytes)")


if __name__ == "__main__":
    generate_strategy_doc()
