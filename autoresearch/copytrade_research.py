"""
autoresearch/copytrade_research.py -- Copy-Trade Parameter optimization loop.

Same Karpathy AutoResearch pattern as research_loop.py (weather), but for
copy-trading parameters. It:

1. Reads the current candidate_strategy.py (copy-trade parameters)
2. Simulates realistic whale-trade scenarios:
   - Random whale trades (category, direction, size, timing)
   - Market outcomes based on whale historical win rates per category
   - Our P&L based on delay, sizing, confidence filter, category multiplier
3. Scores based on: Sortino ratio, ROI, max drawdown, win rate
4. Mutates one parameter at a time, keeps winners, reverts losers
5. Logs everything to autoresearch/copytrade_results.log

Usage:
    python -m autoresearch.copytrade_research --iterations 50
    python -m autoresearch.copytrade_research --iterations 100 --quiet
"""

import os
import sys
import json
import math
import time
import random
import subprocess
import importlib
import copy

import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent directory to path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

try:
    from alerts import alert_research_improvement
except ImportError:
    alert_research_improvement = None

console = Console()

STRATEGY_FILE = Path(__file__).parent / "candidate_strategy.py"
RESULTS_LOG = Path(__file__).parent / "copytrade_results.log"
REAL_OUTCOMES_FILE = Path(__file__).parent.parent / "output" / "copytrade_log.json"

# =============================================================================
# COPY-TRADE MUTATION SPACE
# =============================================================================

MUTATION_SPACE = {
    "COPY_MIN_CONFIDENCE": [0.5, 0.6, 0.7, 0.8, 0.9],
    "COPY_DELAY_SECONDS": [10, 20, 30, 45, 60, 90],
    "COPY_SIZE_FRACTION": [0.005, 0.01, 0.02, 0.03, 0.05],
    "COPY_MAX_TRADES_PER_DAY": [5, 8, 10, 15, 20],
    "COPY_MIN_WHALE_SIZE": [50, 100, 200, 500, 1000],
    "COPY_TRADER_WEIGHT_0x8dxd": [0.5, 0.8, 1.0, 1.2, 1.5],
    "COPY_CRYPTO_MULT": [0.5, 1.0, 1.5, 2.0],
    "COPY_SPORTS_MULT": [0.0, 0.3, 0.5, 0.8, 1.0],
}


# =============================================================================
# WHALE TRADE SCENARIO SIMULATOR
# =============================================================================

# Category configurations: name, whale historical win rate, typical market odds,
# avg whale trade size (USD), volatility of outcomes
CATEGORIES = [
    {
        "name": "crypto",
        "whale_win_rate": 0.98,   # Whales dominate crypto markets
        "typical_odds_range": (0.40, 0.85),  # implied prob range
        "avg_whale_size": 500.0,
        "outcome_volatility": 0.05,
        "param_key": "COPY_CRYPTO_MULT",
    },
    {
        "name": "politics",
        "whale_win_rate": 0.55,   # Politics is noisy, even whales struggle
        "typical_odds_range": (0.30, 0.70),
        "avg_whale_size": 300.0,
        "outcome_volatility": 0.15,
        "param_key": "COPY_POLITICS_MULT",
    },
    {
        "name": "sports",
        "whale_win_rate": 0.62,
        "typical_odds_range": (0.35, 0.75),
        "avg_whale_size": 400.0,
        "outcome_volatility": 0.10,
        "param_key": "COPY_SPORTS_MULT",
    },
]


@dataclass
class WhaleTrade:
    """A simulated whale trade we might copy."""
    category: str
    whale_size_usd: float       # How much the whale traded
    whale_confidence: float     # Whale's implied confidence (0-1)
    market_price_cents: float   # Current market yes price (cents)
    side: str                   # "yes" or "no"
    outcome: bool               # Did this trade win?
    delay_penalty: float        # Price slippage from delay (cents)
    trader_id: str              # Which tracked trader


@dataclass
class CopyTradeResult:
    """Result of copying a single whale trade."""
    whale_trade: WhaleTrade
    copied: bool                # Did we copy it?
    skip_reason: str            # Why we skipped (if applicable)
    our_size_usd: float         # Our position size
    our_entry_cents: float      # Our entry price (after delay slippage)
    pnl: float                  # Our profit/loss in dollars
    won: bool                   # Did we win?


def generate_whale_scenario(
    rng: np.random.Generator,
    n_trades: int = 30,
) -> list[WhaleTrade]:
    """
    Generate a day's worth of whale trades across categories.

    Models realistic whale trading patterns:
    - Whales trade bigger in categories they're confident about
    - Market prices are noisy around fair value
    - Whale confidence correlates with actual outcome
    - Different categories have different win rates
    """
    trades = []

    for _ in range(n_trades):
        cat = rng.choice(CATEGORIES)
        cat_name = cat["name"]
        base_win_rate = cat["whale_win_rate"]

        # Whale trade size (log-normal distribution -- most trades are medium,
        # a few are very large)
        whale_size = float(
            cat["avg_whale_size"] * np.exp(rng.normal(0, 0.6))
        )

        # Whale's confidence: correlated with whether they'll actually win.
        # Higher base_win_rate categories show tighter confidence clustering.
        noise = rng.normal(0, cat["outcome_volatility"])
        whale_confidence = float(np.clip(base_win_rate + noise, 0.1, 0.99))

        # Market price: noisy version of fair value, typically below whale's
        # confidence (that's why the whale is trading -- they see edge)
        odds_low, odds_high = cat["typical_odds_range"]
        market_fair = rng.uniform(odds_low, odds_high)
        market_price_cents = float(
            np.clip(market_fair * 100 + rng.normal(0, 5), 5, 95)
        )

        # Side: whale buys YES if their confidence > market price, else NO
        if whale_confidence > market_fair:
            side = "yes"
        else:
            side = "no"

        # Outcome: based on whale's win rate + some randomness
        # Whales with higher confidence on this specific trade win more often
        outcome_prob = base_win_rate * (0.7 + 0.3 * whale_confidence)
        outcome_prob = min(outcome_prob, 0.995)  # never guaranteed
        outcome = bool(rng.random() < outcome_prob)

        # Delay penalty: how much price moves against us while we wait
        # More volatile categories have worse slippage, bigger trades
        # cause more market impact
        base_slippage = rng.exponential(1.5)  # cents
        size_impact = (whale_size / cat["avg_whale_size"]) * 0.5
        delay_penalty = float(base_slippage + size_impact)

        trades.append(WhaleTrade(
            category=cat_name,
            whale_size_usd=round(whale_size, 2),
            whale_confidence=round(whale_confidence, 4),
            market_price_cents=round(market_price_cents, 1),
            side=side,
            outcome=outcome,
            delay_penalty=round(delay_penalty, 2),
            trader_id="0x8dxd",  # single tracked trader for now
        ))

    return trades


def load_real_copytrade_outcomes() -> list[dict]:
    """
    Load real copy-trade outcomes from output/copytrade_log.json.

    Returns empty list if file doesn't exist or has no valid entries.
    Each entry should have: category, whale_size_usd, side, outcome,
    our_entry_cents, our_size_usd, pnl.
    """
    if not REAL_OUTCOMES_FILE.exists():
        return []

    try:
        with open(REAL_OUTCOMES_FILE, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

    if isinstance(data, list):
        outcomes = data
    elif isinstance(data, dict):
        outcomes = data.get("outcomes", data.get("trades", []))
    else:
        return []

    # Filter to entries with enough data to reconstruct a scenario
    valid = [
        o for o in outcomes
        if o.get("category") is not None
        and o.get("whale_size_usd") is not None
        and o.get("outcome") is not None
    ]

    return valid


def real_outcome_to_whale_trade(outcome: dict) -> WhaleTrade:
    """Convert a real copytrade_log entry into a WhaleTrade for backtesting."""
    return WhaleTrade(
        category=outcome.get("category", "crypto"),
        whale_size_usd=float(outcome.get("whale_size_usd", 100)),
        whale_confidence=float(outcome.get("whale_confidence", 0.8)),
        market_price_cents=float(outcome.get("market_price_cents", 50)),
        side=outcome.get("side", "yes"),
        outcome=bool(outcome.get("outcome", False)),
        delay_penalty=float(outcome.get("delay_penalty", 2.0)),
        trader_id=outcome.get("trader_id", "0x8dxd"),
    )


# =============================================================================
# COPY-TRADE DECISION ENGINE
# =============================================================================

def load_copytrade_params() -> dict:
    """Import candidate_strategy fresh and return copy-trade params."""
    # Force reimport
    for mod_name in list(sys.modules.keys()):
        if "candidate_strategy" in mod_name:
            del sys.modules[mod_name]

    sys.path.insert(0, str(STRATEGY_FILE.parent))
    try:
        import candidate_strategy as strat
        importlib.reload(strat)
        return {
            "copy_min_confidence": strat.COPY_MIN_CONFIDENCE,
            "copy_delay_seconds": strat.COPY_DELAY_SECONDS,
            "copy_size_fraction": strat.COPY_SIZE_FRACTION,
            "copy_max_trades_per_day": strat.COPY_MAX_TRADES_PER_DAY,
            "copy_min_whale_size": strat.COPY_MIN_WHALE_SIZE,
            "copy_trader_weight_0x8dxd": strat.COPY_TRADER_WEIGHT_0x8dxd,
            "copy_crypto_mult": strat.COPY_CRYPTO_MULT,
            "copy_politics_mult": getattr(strat, "COPY_POLITICS_MULT", 0.0),
            "copy_sports_mult": strat.COPY_SPORTS_MULT,
        }
    except Exception as e:
        return {"error": str(e)}


def simulate_copy_decisions(
    whale_trades: list[WhaleTrade],
    params: dict,
    balance: float = 100.0,
) -> list[CopyTradeResult]:
    """
    Decide which whale trades to copy based on current parameters,
    then calculate P&L for each.

    Models:
    - Confidence filter: skip trades below COPY_MIN_CONFIDENCE
    - Whale size filter: skip small whale trades
    - Category multiplier: scale position by category preference (0 = skip)
    - Delay slippage: worse entry price based on COPY_DELAY_SECONDS
    - Position sizing: fraction of whale's size, capped by balance
    - Daily trade limit: max trades per day
    """
    if "error" in params:
        return []

    min_conf = params["copy_min_confidence"]
    delay_sec = params["copy_delay_seconds"]
    size_frac = params["copy_size_fraction"]
    max_trades = params["copy_max_trades_per_day"]
    min_whale_size = params["copy_min_whale_size"]
    trader_weight = params["copy_trader_weight_0x8dxd"]

    # Category multiplier lookup
    cat_mult = {
        "crypto": params["copy_crypto_mult"],
        "politics": params["copy_politics_mult"],
        "sports": params["copy_sports_mult"],
    }

    results = []
    trades_today = 0

    for wt in whale_trades:
        # --- Filter checks ---

        # Category multiplier: 0 means skip entirely
        cm = cat_mult.get(wt.category, 1.0)
        if cm <= 0.0:
            results.append(CopyTradeResult(
                whale_trade=wt, copied=False, skip_reason="category_disabled",
                our_size_usd=0, our_entry_cents=0, pnl=0, won=False,
            ))
            continue

        # Confidence filter
        if wt.whale_confidence < min_conf:
            results.append(CopyTradeResult(
                whale_trade=wt, copied=False, skip_reason="low_confidence",
                our_size_usd=0, our_entry_cents=0, pnl=0, won=False,
            ))
            continue

        # Whale size filter
        if wt.whale_size_usd < min_whale_size:
            results.append(CopyTradeResult(
                whale_trade=wt, copied=False, skip_reason="whale_too_small",
                our_size_usd=0, our_entry_cents=0, pnl=0, won=False,
            ))
            continue

        # Daily trade limit
        if trades_today >= max_trades:
            results.append(CopyTradeResult(
                whale_trade=wt, copied=False, skip_reason="daily_limit",
                our_size_usd=0, our_entry_cents=0, pnl=0, won=False,
            ))
            continue

        # --- Position sizing ---

        # Base size: fraction of whale's trade, scaled by category mult and
        # trader weight
        our_size = wt.whale_size_usd * size_frac * cm * trader_weight

        # Cap at percentage of balance (risk management)
        max_single = balance * 0.05  # max 5% of balance per trade
        our_size = min(our_size, max_single)

        if our_size < 0.01:
            results.append(CopyTradeResult(
                whale_trade=wt, copied=False, skip_reason="size_too_small",
                our_size_usd=0, our_entry_cents=0, pnl=0, won=False,
            ))
            continue

        # --- Entry price with delay slippage ---
        # Longer delay = more price movement against us.
        # Model: slippage scales with sqrt(delay_seconds/30) * base_slippage
        delay_factor = math.sqrt(delay_sec / 30.0)
        slippage_cents = wt.delay_penalty * delay_factor

        if wt.side == "yes":
            our_entry = min(95.0, wt.market_price_cents + slippage_cents)
        else:
            # Buying NO: the no_price goes up (worse for us)
            no_price = 100.0 - wt.market_price_cents
            our_entry = min(95.0, no_price + slippage_cents)

        # --- Calculate P&L ---
        # Binary outcome: we pay our_entry cents per contract, and if we win
        # we get $1 (100 cents) per contract.
        # Number of contracts = our_size / (our_entry / 100)
        cost_per_contract = our_entry / 100.0
        if cost_per_contract <= 0:
            continue

        n_contracts = our_size / cost_per_contract
        total_cost = n_contracts * cost_per_contract

        if wt.side == "yes":
            won = wt.outcome
        else:
            won = not wt.outcome

        if won:
            payout = n_contracts * 1.0  # $1 per contract
            pnl = payout - total_cost
        else:
            pnl = -total_cost

        results.append(CopyTradeResult(
            whale_trade=wt,
            copied=True,
            skip_reason="",
            our_size_usd=round(our_size, 4),
            our_entry_cents=round(our_entry, 2),
            pnl=round(pnl, 4),
            won=won,
        ))
        trades_today += 1

    return results


# =============================================================================
# SCORING (same composite as weather research)
# =============================================================================

def score_simulation(
    all_results: list[CopyTradeResult],
    initial_balance: float = 100.0,
) -> dict:
    """
    Score a full copy-trade simulation run for strategy quality.

    Returns dict with: sortino, roi_pct, max_dd_pct, win_rate, total_trades,
                       profit_factor, total_pnl, score (composite).
    """
    # Only look at trades we actually copied
    copied = [r for r in all_results if r.copied]

    if not copied:
        return {
            "sortino": 0, "roi_pct": 0, "max_dd_pct": 0, "win_rate": 0,
            "total_trades": 0, "profit_factor": 0, "total_pnl": 0,
            "trades_skipped": len(all_results),
            "score": -999.0,
        }

    pnls = [r.pnl for r in copied]
    total_pnl = sum(pnls)
    n_trades = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    # Win rate
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0

    # ROI
    roi_pct = (total_pnl / initial_balance) * 100

    # Equity curve for drawdown
    balance = initial_balance
    peak = balance
    max_dd = 0
    for p in pnls:
        balance += p
        peak = max(peak, balance)
        dd = (peak - balance) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
    max_dd_pct = max_dd * 100

    # Sortino ratio (annualized, ~250 trading days)
    if n_trades > 1:
        returns = np.array(pnls) / initial_balance
        mean_ret = np.mean(returns)
        downside = returns[returns < 0]
        down_std = np.std(downside, ddof=1) if len(downside) > 1 else 0.0001
        sortino = (mean_ret / down_std) * np.sqrt(250)
    else:
        sortino = 0

    # Profit factor
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.0001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Composite score
    # Copy-trading should have high win rates (riding whale alpha) and
    # moderate returns. Penalize heavy drawdown and reward consistency.
    if max_dd_pct > 60:
        score = -999.0  # Hard constraint: reject catastrophic drawdown
    else:
        score = (
            sortino * 0.30 +
            min(roi_pct, 500) * 0.01 * 0.20 +  # cap ROI contribution
            (win_rate / 100.0) * 10.0 * 0.15 +
            min(profit_factor, 3.0) * 3.0 * 0.15 -
            max_dd_pct * 0.15                    # penalize drawdown
        )

    trades_skipped = len([r for r in all_results if not r.copied])

    return {
        "sortino": round(float(sortino), 3),
        "roi_pct": round(float(roi_pct), 2),
        "max_dd_pct": round(float(max_dd_pct), 2),
        "win_rate": round(float(win_rate), 1),
        "total_trades": n_trades,
        "profit_factor": round(float(profit_factor), 2),
        "total_pnl": round(float(total_pnl), 2),
        "trades_skipped": trades_skipped,
        "score": round(float(score), 4),
    }


# =============================================================================
# FULL BACKTEST: MANY SCENARIOS
# =============================================================================

def run_copytrade_backtest(
    seed: int = 42,
    n_scenarios: int = 200,
    use_real_data: bool = False,
) -> dict:
    """
    Run a full copy-trade strategy backtest:
    1. Load candidate_strategy copy-trade params
    2. Generate n_scenarios days of whale trades
    3. For each day, decide which to copy and calculate P&L
    4. Track balance across days
    5. Score the result

    If use_real_data=True and copytrade_log.json exists, the FIRST N scenarios
    use real outcome data. Remaining scenarios are synthetic.
    """
    params = load_copytrade_params()
    if "error" in params:
        return {"error": params["error"], "score": -999.0}

    initial_balance = 100.0
    rng = np.random.default_rng(seed)
    all_results = []
    balance = initial_balance

    # Load real outcomes if requested
    real_days = []
    if use_real_data:
        real_outcomes = load_real_copytrade_outcomes()
        if real_outcomes:
            # Group real outcomes into "days" of ~30 trades each
            chunk_size = 30
            for i in range(0, len(real_outcomes), chunk_size):
                chunk = real_outcomes[i:i + chunk_size]
                day_trades = [real_outcome_to_whale_trade(o) for o in chunk]
                real_days.append(day_trades)

    n_real = len(real_days)
    n_synthetic = max(0, n_scenarios - n_real)

    for scenario_idx in range(n_real + n_synthetic):
        if balance <= 1.0:
            break  # Account blown

        # Use real scenario data first, then synthetic
        if scenario_idx < n_real:
            whale_trades = real_days[scenario_idx]
        else:
            # Each "day" has a random number of whale trades (15-45)
            n_trades = int(rng.integers(15, 46))
            whale_trades = generate_whale_scenario(rng, n_trades=n_trades)

        day_results = simulate_copy_decisions(whale_trades, params, balance)

        # Update balance
        for r in day_results:
            if r.copied:
                balance += r.pnl

        all_results.extend(day_results)

    metrics = score_simulation(all_results, initial_balance=initial_balance)
    return metrics


# =============================================================================
# PARAMETER MUTATION (reuse the same file-editing approach)
# =============================================================================

def read_strategy_file() -> str:
    """Read current strategy source code."""
    return STRATEGY_FILE.read_text()


def write_strategy_file(content: str):
    """Write modified strategy back."""
    STRATEGY_FILE.write_text(content)


def mutate_parameter(source: str, param_name: str, new_value) -> str:
    """
    Replace a parameter value in the strategy source code.
    Handles int, float, and bool values.
    """
    lines = source.split("\n")
    new_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{param_name} =") or stripped.startswith(f"{param_name}="):
            parts = line.split("#")
            comment = f"  # {parts[1].strip()}" if len(parts) > 1 else ""
            indent = len(line) - len(line.lstrip())
            new_line = f"{' ' * indent}{param_name} = {repr(new_value)}{comment}"
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


def get_current_value(source: str, param_name: str):
    """Extract current value of a parameter from source."""
    for line in source.split("\n"):
        stripped = line.strip()
        if stripped.startswith(f"{param_name} =") or stripped.startswith(f"{param_name}="):
            val_part = stripped.split("=", 1)[1].split("#")[0].strip()
            try:
                return eval(val_part)
            except Exception:
                return val_part
    return None


def random_mutation(source: str) -> tuple[str, object, str]:
    """Pick a random copy-trade parameter and a random value for it."""
    param = random.choice(list(MUTATION_SPACE.keys()))
    new_val = random.choice(MUTATION_SPACE[param])
    return param, new_val, "Random exploration"


# =============================================================================
# GIT HELPERS
# =============================================================================

def git_commit(message: str):
    """Commit current strategy state."""
    try:
        subprocess.run(
            ["git", "add", str(STRATEGY_FILE)],
            cwd=config.PROJECT_ROOT,
            capture_output=True,
            timeout=10,
        )
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=config.PROJECT_ROOT,
            capture_output=True,
            timeout=10,
        )
    except Exception:
        pass


def git_revert():
    """Revert strategy to last committed version."""
    try:
        subprocess.run(
            ["git", "checkout", "--", str(STRATEGY_FILE)],
            cwd=config.PROJECT_ROOT,
            capture_output=True,
            timeout=10,
        )
    except Exception:
        pass


# =============================================================================
# LOGGING
# =============================================================================

def log_result(iteration: int, param: str, old_val, new_val, metrics: dict, kept: bool):
    """Append result to copytrade research log."""
    entry = {
        "iteration": iteration,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameter": str(param),
        "old_value": float(old_val) if isinstance(old_val, (int, float)) else str(old_val),
        "new_value": float(new_val) if isinstance(new_val, (int, float)) else str(new_val),
        "metrics": {
            k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else str(v)
            for k, v in metrics.items()
        },
        "kept": bool(kept),
    }
    with open(RESULTS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


# =============================================================================
# MAIN RESEARCH LOOP
# =============================================================================

def run_research(
    max_iterations: int = None,
    n_scenarios: int = 200,
    verbose: bool = True,
    use_real_data: bool = False,
):
    """
    Run the Copy-Trade AutoResearch loop.

    For each iteration:
    1. Read current candidate_strategy.py
    2. Pick a random copy-trade parameter mutation
    3. Apply mutation, run copy-trade backtest
    4. If score improves: keep and git commit
    5. If score worsens: revert
    6. Log everything

    Args:
        max_iterations: Number of experiments to run
        n_scenarios: Number of trading days to simulate per backtest
        verbose: Print progress to terminal
        use_real_data: If True, incorporate real outcomes from
                       output/copytrade_log.json for the first N scenarios
    """
    max_iterations = max_iterations or config.AUTORESEARCH_MAX_ITERATIONS

    console.print("\n[bold cyan]================================================================[/bold cyan]")
    console.print("[bold cyan]  COPYTRADE AUTORESEARCH: Copy-Trading Parameter Optimization  [/bold cyan]")
    console.print("[bold cyan]================================================================[/bold cyan]\n")

    # Report real data status
    if use_real_data:
        real_outcomes = load_real_copytrade_outcomes()
        if real_outcomes:
            console.print(
                f"[green]Real data mode: {len(real_outcomes)} real outcomes loaded "
                f"from {REAL_OUTCOMES_FILE.name}[/green]"
            )
        else:
            console.print(
                "[yellow]Real data mode requested but no copytrade outcomes found -- "
                "falling back to 100% synthetic[/yellow]"
            )

    # Baseline: score current strategy
    console.print("[dim]Running baseline copy-trade backtest...[/dim]")
    baseline = run_copytrade_backtest(
        seed=42, n_scenarios=n_scenarios, use_real_data=use_real_data
    )
    if "error" in baseline:
        console.print(f"[red]Baseline failed: {baseline['error']}[/red]")
        return

    console.print(f"[green]Baseline score: {baseline['score']:.4f}[/green]")
    console.print(
        f"  Sortino: {baseline['sortino']:.3f} | "
        f"ROI: {baseline['roi_pct']:.2f}% | "
        f"Win: {baseline['win_rate']:.1f}% | "
        f"MaxDD: {baseline['max_dd_pct']:.2f}% | "
        f"Trades: {baseline['total_trades']} | "
        f"PF: {baseline['profit_factor']:.2f} | "
        f"Skipped: {baseline.get('trades_skipped', 0)}\n"
    )

    best_score = baseline["score"]
    improvements = 0
    history = []

    # Load existing history
    if RESULTS_LOG.exists():
        for line in RESULTS_LOG.read_text().strip().split("\n"):
            if line.strip():
                try:
                    history.append(json.loads(line))
                except Exception:
                    pass

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Copy-trade research loop", total=max_iterations)

        for i in range(1, max_iterations + 1):
            source = read_strategy_file()

            # Pick mutation
            param, new_val, reason = random_mutation(source)
            old_val = get_current_value(source, param)

            # Skip if same value
            if old_val == new_val:
                progress.update(task, advance=1)
                continue

            progress.update(task, description=f"Iter {i}/{max_iterations}: {param}={new_val}")

            # Apply mutation
            new_source = mutate_parameter(source, param, new_val)
            write_strategy_file(new_source)

            # Run copy-trade backtest with slightly different seed per iteration
            metrics = run_copytrade_backtest(
                seed=42 + i, n_scenarios=n_scenarios, use_real_data=use_real_data
            )
            if "error" in metrics:
                write_strategy_file(source)  # revert on error
                progress.update(task, advance=1)
                continue

            new_score = metrics["score"]

            # Decision: keep or revert
            kept = new_score > best_score

            if kept:
                best_score = new_score
                improvements += 1
                git_commit(
                    f"CopyTradeResearch iter {i}: {param}={new_val} "
                    f"score={new_score:.4f} sortino={metrics['sortino']:.3f}"
                )
                if alert_research_improvement is not None:
                    try:
                        alert_research_improvement(param, old_val, new_val, new_score)
                    except Exception:
                        pass
                if verbose:
                    console.print(
                        f"  [green]+ Iter {i}: {param} {old_val}->{new_val} "
                        f"score={new_score:.4f} sortino={metrics['sortino']:.3f} "
                        f"roi={metrics['roi_pct']:.1f}% win={metrics['win_rate']:.0f}% KEPT[/green]"
                    )
            else:
                write_strategy_file(source)  # revert to pre-mutation content
                if verbose and i % 5 == 0:
                    console.print(
                        f"  [dim]- Iter {i}: {param} {old_val}->{new_val} "
                        f"score={new_score:.4f} (vs {best_score:.4f}) REVERTED[/dim]"
                    )

            # Log
            log_result(i, param, old_val, new_val, metrics, kept)
            history.append({
                "iteration": i,
                "parameter": param,
                "old_value": old_val,
                "new_value": new_val,
                "score": new_score,
                "kept": kept,
            })

            progress.update(task, advance=1)
            time.sleep(0.05)  # brief pause

    # Final summary
    console.print(f"\n[bold cyan]{'=' * 65}[/bold cyan]")
    console.print(f"[bold]Copy-Trade Research Complete: {max_iterations} iterations[/bold]")
    console.print(f"  Improvements found: {improvements}")
    console.print(f"  Best score: {best_score:.4f} (baseline was {baseline['score']:.4f})")

    final = run_copytrade_backtest(
        seed=42, n_scenarios=n_scenarios, use_real_data=use_real_data
    )
    if "error" in final:
        console.print(f"[red]Final backtest failed: {final['error']}[/red]")
    else:
        console.print(f"\n[bold]Final Strategy Performance:[/bold]")
        console.print(f"  Sortino:       {final['sortino']:.3f}")
        console.print(f"  ROI:           {final['roi_pct']:.2f}%")
        console.print(f"  Win Rate:      {final['win_rate']:.1f}%")
        console.print(f"  Max Drawdown:  {final['max_dd_pct']:.2f}%")
        console.print(f"  Profit Factor: {final['profit_factor']:.2f}")
        console.print(f"  Total Trades:  {final['total_trades']}")
        console.print(f"  Skipped:       {final.get('trades_skipped', 0)}")
        console.print(f"  Total P&L:     ${final['total_pnl']:.2f}")
    console.print(f"\n  Results log: {RESULTS_LOG}")

    # Show current best params
    console.print(f"\n[bold]Optimized Parameters:[/bold]")
    params = load_copytrade_params()
    if "error" not in params:
        console.print(f"  MIN_CONFIDENCE:    {params['copy_min_confidence']}")
        console.print(f"  DELAY_SECONDS:     {params['copy_delay_seconds']}")
        console.print(f"  SIZE_FRACTION:     {params['copy_size_fraction']}")
        console.print(f"  MAX_TRADES/DAY:    {params['copy_max_trades_per_day']}")
        console.print(f"  MIN_WHALE_SIZE:    ${params['copy_min_whale_size']}")
        console.print(f"  TRADER_WEIGHT:     {params['copy_trader_weight_0x8dxd']}")
        console.print(f"  CRYPTO_MULT:       {params['copy_crypto_mult']}")
        console.print(f"  POLITICS_MULT:     {params['copy_politics_mult']}")
        console.print(f"  SPORTS_MULT:       {params['copy_sports_mult']}")

    console.print(f"[bold cyan]{'=' * 65}[/bold cyan]\n")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Copy-Trade AutoResearch Loop")
    parser.add_argument("--iterations", type=int, default=50, help="Number of experiments")
    parser.add_argument("--scenarios", type=int, default=200, help="Trading days per backtest")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    parser.add_argument(
        "--use-real-data", action="store_true",
        help="Incorporate real outcomes from output/copytrade_log.json"
    )
    args = parser.parse_args()

    run_research(
        max_iterations=args.iterations,
        n_scenarios=args.scenarios,
        verbose=not args.quiet,
        use_real_data=args.use_real_data,
    )
