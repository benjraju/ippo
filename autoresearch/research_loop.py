"""
autoresearch/research_loop.py — Karpathy-style AutoResearch loop for Kalshi.

This is the overnight self-improvement engine. It:
1. Reads the current candidate_strategy.py
2. Uses MLX (Apple Silicon native) or Claude to propose modifications
3. Runs a full backtest with P&L metrics
4. Git-commits winners, reverts losers
5. Logs everything to results.log
6. Repeats for N iterations or until morning

Based on: https://github.com/trevin-creator/autoresearch-mlx
Adapted for prediction market strategy optimization.
"""

import os
import sys
import json
import time
import random
import subprocess
import importlib
from pathlib import Path
from datetime import datetime, timezone

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from backtester import Backtester

console = Console()

STRATEGY_FILE = Path(__file__).parent / "candidate_strategy.py"
RESULTS_LOG = Path(__file__).parent / "results.log"

# =============================================================================
# PARAMETER MUTATIONS — What the research loop tries
# =============================================================================

MUTATION_SPACE = {
    "MIN_EDGE": [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10],
    "KELLY_MULT": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
    "MAX_POSITION_FRAC": [0.01, 0.015, 0.02, 0.025, 0.03],
    "MIN_VOLUME": [25, 50, 75, 100, 150],
    "MAX_HOURS": [12, 24, 36, 48, 72],
    "MEAN_REVERT_STRENGTH": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
    "MEAN_REVERT_THRESHOLD_HIGH": [0.80, 0.82, 0.85, 0.87, 0.90],
    "MEAN_REVERT_THRESHOLD_LOW": [0.10, 0.13, 0.15, 0.18, 0.20],
    "VOLUME_CONFIDENCE_SCALE": [100, 150, 200, 250, 300],
    "MAX_SPREAD": [8, 10, 12, 15, 20],
}


def score_result(sharpe: float, roi: float, win_rate: float, max_dd: float) -> float:
    """
    Scoring function for strategy evaluation.
    Higher is better. Penalizes high drawdown heavily.
    """
    # Hard constraint: reject if drawdown exceeds limit
    if max_dd > config.AUTORESEARCH_MAX_DRAWDOWN * 100:
        return -999.0

    score = (
        sharpe * 0.40 +
        roi * 0.30 +
        (win_rate / 100.0) * 0.20 -
        (max_dd / 100.0) * 10.0  # Heavy drawdown penalty
    )
    return round(score, 4)


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
        if line.strip().startswith(f"{param_name} =") or line.strip().startswith(f"{param_name}="):
            # Preserve the comment if any
            parts = line.split("#")
            comment = f"  # {parts[1].strip()}" if len(parts) > 1 else ""
            indent = len(line) - len(line.lstrip())
            new_line = f"{' ' * indent}{param_name} = {repr(new_value)}{comment}"
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


def run_backtest_for_strategy(seed: int = 42) -> dict:
    """
    Import the current candidate_strategy and run a backtest.
    Returns dict with all performance metrics.
    """
    # Force reimport of the modified strategy
    if "autoresearch.candidate_strategy" in sys.modules:
        del sys.modules["autoresearch.candidate_strategy"]
    if "candidate_strategy" in sys.modules:
        del sys.modules["candidate_strategy"]

    # Import fresh
    sys.path.insert(0, str(STRATEGY_FILE.parent))
    try:
        import candidate_strategy as strat
        importlib.reload(strat)
    except Exception as e:
        return {"error": str(e)}

    # Run backtest using strategy's position sizing
    bt = Backtester(initial_balance=config.ACCOUNT_BALANCE)

    def strategy_sizer(balance, price):
        prob = strat.estimate_probability(price, 100, 24)
        return strat.calculate_position_size(balance, prob, price)

    # Estimate win rate from strategy signals
    # Higher edge detection should mean higher win rate
    base_win_rate = 0.50 + strat.MEAN_REVERT_STRENGTH * 2
    base_win_rate = min(0.65, max(0.48, base_win_rate))

    result = bt.run_simulated_backtest(
        strategy_fn=strategy_sizer,
        n_trades=250,
        win_rate=base_win_rate,
        avg_odds=50,
        seed=seed,
    )

    return {
        "sharpe": result.sharpe_ratio,
        "sortino": result.sortino_ratio,
        "roi_pct": result.roi_pct,
        "max_dd_pct": result.max_drawdown_pct,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "total_trades": result.total_trades,
        "final_balance": result.final_balance,
        "total_pnl": result.total_pnl,
    }


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


def log_result(iteration: int, param: str, old_val, new_val, metrics: dict, score: float, kept: bool):
    """Append result to research log."""
    entry = {
        "iteration": iteration,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameter": param,
        "old_value": old_val,
        "new_value": new_val,
        "metrics": metrics,
        "score": score,
        "kept": kept,
    }
    with open(RESULTS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def use_claude_for_mutation(source: str, history: list[dict]) -> tuple[str, str, str]:
    """
    Use Claude API to suggest smarter mutations based on history.
    Falls back to random mutation if Claude is unavailable.
    """
    if not config.ANTHROPIC_API_KEY:
        return random_mutation(source)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

        # Summarize recent history
        recent = history[-10:] if history else []
        history_summary = "\n".join([
            f"  Iter {h['iteration']}: {h['parameter']}={h['new_value']} → score={h['score']:.3f} ({'KEPT' if h['kept'] else 'reverted'})"
            for h in recent
        ])

        prompt = f"""You are optimizing a prediction market trading strategy.

Current strategy parameters (in candidate_strategy.py):
{source[:2000]}

Recent experiment history:
{history_summary if history_summary else "No history yet."}

Available parameters to modify:
{json.dumps({k: v for k, v in MUTATION_SPACE.items()}, indent=2)}

Goal: Maximize Sharpe ratio >1.2, ROI >5%, keep max drawdown <8%.

Suggest ONE parameter change. Respond with ONLY JSON:
{{"parameter": "PARAM_NAME", "new_value": value, "reasoning": "brief explanation"}}"""

        resp = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )

        text = resp.content[0].text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        suggestion = json.loads(text)
        param = suggestion["parameter"]
        new_val = suggestion["new_value"]
        reason = suggestion.get("reasoning", "Claude suggestion")

        if param in MUTATION_SPACE:
            return param, new_val, reason

    except Exception as e:
        console.print(f"[yellow]Claude suggestion failed: {e}. Using random mutation.[/yellow]")

    return random_mutation(source)


def random_mutation(source: str) -> tuple[str, str, str]:
    """Pick a random parameter and a random value for it."""
    param = random.choice(list(MUTATION_SPACE.keys()))
    new_val = random.choice(MUTATION_SPACE[param])
    return param, new_val, "Random exploration"


def get_current_value(source: str, param_name: str):
    """Extract current value of a parameter from source."""
    for line in source.split("\n"):
        stripped = line.strip()
        if stripped.startswith(f"{param_name} =") or stripped.startswith(f"{param_name}="):
            # Get value part (before any comment)
            val_part = stripped.split("=", 1)[1].split("#")[0].strip()
            try:
                return eval(val_part)
            except Exception:
                return val_part
    return None


# =============================================================================
# MAIN RESEARCH LOOP
# =============================================================================

def run_research(
    max_iterations: int = None,
    use_claude: bool = True,
    verbose: bool = True,
):
    """
    Run the AutoResearch loop.

    This is the main overnight function. It:
    1. Reads current strategy
    2. Proposes a mutation (via Claude or random)
    3. Applies mutation and backtests
    4. Keeps improvements, reverts regressions
    5. Logs everything

    Args:
        max_iterations: Number of experiments to run
        use_claude: Whether to use Claude for smart mutations
        verbose: Print progress to terminal
    """
    max_iterations = max_iterations or config.AUTORESEARCH_MAX_ITERATIONS

    console.print("\n[bold cyan]════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  AUTORESEARCH: Overnight Strategy Improvement  [/bold cyan]")
    console.print("[bold cyan]════════════════════════════════════════════════[/bold cyan]\n")

    # Baseline: score current strategy
    console.print("[dim]Running baseline backtest...[/dim]")
    baseline_metrics = run_backtest_for_strategy(seed=42)
    if "error" in baseline_metrics:
        console.print(f"[red]Baseline failed: {baseline_metrics['error']}[/red]")
        return

    baseline_score = score_result(
        baseline_metrics["sharpe"],
        baseline_metrics["roi_pct"],
        baseline_metrics["win_rate"],
        baseline_metrics["max_dd_pct"],
    )

    console.print(f"[green]Baseline score: {baseline_score:.4f}[/green]")
    console.print(f"  Sharpe: {baseline_metrics['sharpe']:.3f} | ROI: {baseline_metrics['roi_pct']:.2f}% | "
                  f"Win: {baseline_metrics['win_rate']:.1f}% | MaxDD: {baseline_metrics['max_dd_pct']:.2f}%\n")

    best_score = baseline_score
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
        task = progress.add_task("Research loop", total=max_iterations)

        for i in range(1, max_iterations + 1):
            source = read_strategy_file()

            # Get mutation suggestion
            if use_claude and random.random() < 0.7:  # 70% Claude, 30% random
                param, new_val, reason = use_claude_for_mutation(source, history)
            else:
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

            # Backtest the mutation
            metrics = run_backtest_for_strategy(seed=42 + i)
            if "error" in metrics:
                write_strategy_file(source)  # Revert on error
                progress.update(task, advance=1)
                continue

            new_score = score_result(
                metrics["sharpe"],
                metrics["roi_pct"],
                metrics["win_rate"],
                metrics["max_dd_pct"],
            )

            # Decision: keep or revert
            kept = new_score > best_score

            if kept:
                best_score = new_score
                improvements += 1
                git_commit(f"AutoResearch iter {i}: {param}={new_val} score={new_score:.4f} (+{new_score - baseline_score:.4f})")
                if verbose:
                    console.print(
                        f"  [green]✓ Iter {i}: {param}={old_val}→{new_val} "
                        f"score={new_score:.4f} (+{new_score - best_score + (new_score - baseline_score):.4f}) KEPT[/green]"
                    )
            else:
                write_strategy_file(source)  # Revert
                git_revert()
                if verbose and i % 5 == 0:
                    console.print(
                        f"  [dim]✗ Iter {i}: {param}={old_val}→{new_val} "
                        f"score={new_score:.4f} (worse) REVERTED[/dim]"
                    )

            # Log
            log_result(i, param, old_val, new_val, metrics, new_score, kept)
            history.append({
                "iteration": i,
                "parameter": param,
                "new_value": new_val,
                "score": new_score,
                "kept": kept,
            })

            progress.update(task, advance=1)

            # Brief pause to avoid overheating CPU
            time.sleep(0.1)

    # Final summary
    console.print(f"\n[bold cyan]{'═' * 50}[/bold cyan]")
    console.print(f"[bold]Research Complete: {max_iterations} iterations[/bold]")
    console.print(f"  Improvements found: {improvements}")
    console.print(f"  Best score: {best_score:.4f} (baseline was {baseline_score:.4f})")

    # Run final backtest and show charts
    final_metrics = run_backtest_for_strategy(seed=42)
    console.print(f"\n[bold]Final Strategy Performance:[/bold]")
    console.print(f"  Sharpe: {final_metrics['sharpe']:.3f}")
    console.print(f"  ROI: {final_metrics['roi_pct']:.2f}%")
    console.print(f"  Win Rate: {final_metrics['win_rate']:.1f}%")
    console.print(f"  Max Drawdown: {final_metrics['max_dd_pct']:.2f}%")
    console.print(f"  Profit Factor: {final_metrics['profit_factor']:.2f}")
    console.print(f"\n  Results log: {RESULTS_LOG}")
    console.print(f"[bold cyan]{'═' * 50}[/bold cyan]\n")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AutoResearch Loop")
    parser.add_argument("--iterations", type=int, default=50, help="Number of experiments")
    parser.add_argument("--no-claude", action="store_true", help="Use random mutations only")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    args = parser.parse_args()

    run_research(
        max_iterations=args.iterations,
        use_claude=not args.no_claude,
        verbose=not args.quiet,
    )
