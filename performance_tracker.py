"""
performance_tracker.py -- Tracks whether AutoResearch improvements translate
to real trading performance.

Answers the question: "Is the bot actually getting smarter, or is it just
overfitting to fake data?"

Usage:
    python performance_tracker.py
"""

import csv
import json
import math
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
RESULTS_LOG = PROJECT_ROOT / "autoresearch" / "results.log"
TRADE_HISTORY = PROJECT_ROOT / "output" / "trade_history.csv"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_experiments() -> list[dict]:
    """Load AutoResearch experiment results from JSONL log."""
    if not RESULTS_LOG.exists():
        return []
    experiments = []
    with open(RESULTS_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                experiments.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return experiments


def _load_trades() -> list[dict]:
    """Load trade history from CSV."""
    if not TRADE_HISTORY.exists():
        return []
    try:
        with open(TRADE_HISTORY) as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Simulated metrics (AutoResearch)
# ---------------------------------------------------------------------------

def _simulated_metrics(experiments: list[dict]) -> dict:
    """Compute metrics from AutoResearch experiment log."""
    if not experiments:
        return {}

    total = len(experiments)
    kept = [e for e in experiments if e.get("kept")]
    kept_count = len(kept)

    # Filter out inf/nan and absurdly large scores (overflow artifacts)
    MAX_SANE_SCORE = 1e6
    kept_scores = [
        e["metrics"]["score"]
        for e in kept
        if math.isfinite(e["metrics"].get("score", 0))
        and abs(e["metrics"]["score"]) < MAX_SANE_SCORE
    ]

    best_score = max(kept_scores) if kept_scores else 0.0

    # Simulated win rate: average win_rate from kept experiments (sane only)
    kept_win_rates = [
        e["metrics"]["win_rate"]
        for e in kept
        if "win_rate" in e["metrics"]
        and math.isfinite(e["metrics"]["win_rate"])
        and abs(e["metrics"]["win_rate"]) < MAX_SANE_SCORE
    ]
    sim_win_rate = np.mean(kept_win_rates) if kept_win_rates else 0.0

    # Simulated Sortino: average from kept experiments (sane only)
    MAX_SANE_SORTINO = 100  # Sortino > 100 is nonsensical
    kept_sortinos = [
        e["metrics"]["sortino"]
        for e in kept
        if math.isfinite(e["metrics"].get("sortino", 0))
        and abs(e["metrics"]["sortino"]) < MAX_SANE_SORTINO
    ]
    sim_sortino = np.mean(kept_sortinos) if kept_sortinos else 0.0

    # Score trend: compare rolling avg of last N kept vs first N kept
    trend = 0.0
    window = min(10, len(kept_scores))
    if len(kept_scores) >= 4:
        half = len(kept_scores) // 2
        early_avg = np.mean(kept_scores[:half])
        late_avg = np.mean(kept_scores[half:])
        trend = late_avg - early_avg

    return {
        "total_experiments": total,
        "kept_count": kept_count,
        "best_score": best_score,
        "score_trend": trend,
        "trend_window": total,
        "sim_win_rate": sim_win_rate,
        "sim_sortino": sim_sortino,
    }


# ---------------------------------------------------------------------------
# Real trading metrics
# ---------------------------------------------------------------------------

def _settled_trades(trades: list[dict]) -> list[dict]:
    """Return only trades that have settled (yes/no result)."""
    return [
        t for t in trades
        if t.get("settlement_result") in ("yes", "no")
    ]


def _real_win_rate(settled: list[dict]) -> float:
    """Win rate from settled trades (a win is pnl > 0)."""
    if not settled:
        return 0.0
    wins = sum(1 for t in settled if float(t.get("pnl", 0)) > 0)
    return (wins / len(settled)) * 100


def _real_sortino(settled: list[dict], target_return: float = 0.0) -> float:
    """Sortino ratio from the P&L sequence of settled trades."""
    if len(settled) < 2:
        return 0.0

    pnls = np.array([float(t.get("pnl", 0)) for t in settled])
    mean_return = np.mean(pnls)
    downside = pnls[pnls < target_return] - target_return

    if len(downside) == 0:
        return float("inf") if mean_return > 0 else 0.0

    downside_std = np.sqrt(np.mean(downside ** 2))
    if downside_std == 0:
        return float("inf") if mean_return > 0 else 0.0

    return float(mean_return / downside_std)


def _real_metrics(trades: list[dict]) -> dict:
    """Compute real trading performance metrics."""
    if not trades:
        return {}

    settled = _settled_trades(trades)
    total_trades = len(trades)
    settled_count = len(settled)
    open_count = total_trades - settled_count

    win_rate = _real_win_rate(settled)
    sortino = _real_sortino(settled)

    settled_pnls = [float(t.get("pnl", 0)) for t in settled]
    total_pnl = sum(settled_pnls)

    return {
        "total_trades": total_trades,
        "settled_count": settled_count,
        "open_count": open_count,
        "win_rate": win_rate,
        "sortino": sortino,
        "total_pnl": total_pnl,
    }


# ---------------------------------------------------------------------------
# Strategy breakdown
# ---------------------------------------------------------------------------

def _strategy_breakdown(trades: list[dict]) -> list[dict]:
    """P&L and win rate per strategy, sorted by total P&L descending."""
    strategies: dict[str, list[dict]] = {}
    for t in trades:
        strat = t.get("strategy", "unknown")
        strategies.setdefault(strat, []).append(t)

    breakdown = []
    for name, strat_trades in strategies.items():
        settled = _settled_trades(strat_trades)
        total_pnl = sum(float(t.get("pnl", 0)) for t in settled)
        win_rate = _real_win_rate(settled) if settled else 0.0
        breakdown.append({
            "strategy": name,
            "total_pnl": total_pnl,
            "trade_count": len(strat_trades),
            "settled_count": len(settled),
            "win_rate": win_rate,
        })

    breakdown.sort(key=lambda x: x["total_pnl"], reverse=True)
    return breakdown


# ---------------------------------------------------------------------------
# Reality gap
# ---------------------------------------------------------------------------

def _reality_gap(sim: dict, real: dict) -> dict:
    """Compute gap between simulated and real metrics."""
    if not sim or not real:
        return {}

    win_gap = sim.get("sim_win_rate", 0) - real.get("win_rate", 0)
    sortino_gap = sim.get("sim_sortino", 0) - real.get("sortino", 0)

    # Verdict based on absolute win rate gap
    abs_gap = abs(win_gap)
    if abs_gap < 3:
        verdict = "SMALL GAP - on track"
    elif abs_gap < 10:
        verdict = "MODERATE GAP - monitor"
    elif abs_gap < 20:
        verdict = "LARGE GAP - likely overfitting"
    else:
        verdict = "CRITICAL GAP - overfitting confirmed"

    return {
        "win_rate_gap": win_gap,
        "sortino_gap": sortino_gap,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report() -> dict:
    """Generate a full performance comparison report as a dict."""
    experiments = _load_experiments()
    trades = _load_trades()

    sim = _simulated_metrics(experiments)
    real = _real_metrics(trades)
    gap = _reality_gap(sim, real)
    breakdown = _strategy_breakdown(trades)

    return {
        "simulated": sim,
        "real": real,
        "reality_gap": gap,
        "strategy_breakdown": breakdown,
    }


def print_report() -> None:
    """Pretty-print the performance tracker report using Rich."""
    console = Console()
    report = generate_report()

    sim = report["simulated"]
    real = report["real"]
    gap = report["reality_gap"]
    breakdown = report["strategy_breakdown"]

    sections: list[str] = []

    # --- Simulated section ---
    if sim:
        lines = [
            "[bold cyan]SIMULATED (AutoResearch)[/bold cyan]",
            f"  Best score:          {sim['best_score']:.2f}",
            f"  Improvements found:  {sim['kept_count']}/{sim['total_experiments']}",
        ]
        trend = sim["score_trend"]
        sign = "+" if trend >= 0 else ""
        lines.append(
            f"  Score trend:         {sign}{trend:.2f} "
            f"(over {sim['trend_window']} iter)"
        )
        lines.append(f"  Sim win rate:        {sim['sim_win_rate']:.1f}%")
        lines.append(f"  Sim Sortino:         {sim['sim_sortino']:.2f}")
        sections.append("\n".join(lines))
    else:
        sections.append(
            "[bold cyan]SIMULATED (AutoResearch)[/bold cyan]\n"
            "  No data yet -- run autoresearch first"
        )

    # --- Real trading section ---
    if real:
        sign = "+" if real["total_pnl"] >= 0 else ""
        lines = [
            "[bold green]REAL TRADING[/bold green]",
            f"  Total trades:  {real['total_trades']}  "
            f"({real['settled_count']} settled, {real['open_count']} open)",
            f"  Win rate:      {real['win_rate']:.1f}%",
            f"  Sortino:       {real['sortino']:.2f}",
            f"  Total P&L:     {sign}${real['total_pnl']:.2f}",
        ]
        sections.append("\n".join(lines))
    else:
        sections.append(
            "[bold green]REAL TRADING[/bold green]\n"
            "  No data yet -- waiting for trades"
        )

    # --- Reality gap section ---
    if gap:
        win_gap = gap["win_rate_gap"]
        win_dir = "sim higher" if win_gap > 0 else "real higher"
        sort_gap = gap["sortino_gap"]
        sort_dir = "sim higher" if sort_gap > 0 else "real higher"

        lines = [
            "[bold yellow]REALITY GAP[/bold yellow]",
            f"  Win rate gap:  {win_gap:+.1f}% ({win_dir})",
            f"  Sortino gap:   {sort_gap:+.2f} ({sort_dir})",
            f"  Verdict:       {gap['verdict']}",
        ]
        sections.append("\n".join(lines))

    # --- Strategy breakdown section ---
    if breakdown:
        lines = ["[bold magenta]BY STRATEGY[/bold magenta]"]
        for s in breakdown:
            sign = "+" if s["total_pnl"] >= 0 else ""
            pnl_str = f"{sign}${s['total_pnl']:.2f}"
            settled_label = (
                f"{s['trade_count']} trades"
                if s["settled_count"] == s["trade_count"]
                else f"{s['settled_count']}/{s['trade_count']} settled"
            )
            win_str = (
                f"{s['win_rate']:.0f}% win"
                if s["settled_count"] > 0
                else "no settlements"
            )
            lines.append(
                f"  {s['strategy']:<10} {pnl_str:>12}  "
                f"({settled_label}, {win_str})"
            )
        sections.append("\n".join(lines))

    body = "\n\n".join(sections)
    panel = Panel(
        body,
        title="[bold white]IPPO PERFORMANCE TRACKER[/bold white]",
        border_style="blue",
        padding=(1, 2),
    )
    console.print(panel)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_report()
