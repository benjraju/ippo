"""
backtester.py — Backtesting engine with Plotly charts, CSV export, and Monte Carlo.

Produces:
- Equity curve (HTML interactive + PNG)
- Trade-by-trade CSV
- Summary stats: ROI, Sharpe, Sortino, max drawdown, win rate
- Monte Carlo simulation of strategy variants
"""

import csv
import math
import random
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.console import Console
from rich.table import Table

import config

console = Console()


@dataclass
class BacktestTrade:
    """A single simulated trade."""
    timestamp: str
    ticker: str
    side: str
    price_cents: int
    count: int
    cost: float
    pnl: float
    cumulative_pnl: float
    result: str  # "win", "loss"
    reason: str = ""


@dataclass
class BacktestResult:
    """Complete backtest results."""
    trades: list[BacktestTrade]
    initial_balance: float
    final_balance: float
    total_pnl: float
    roi_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    equity_curve: list[float]
    timestamps: list[str]


class Backtester:
    """
    Historical backtesting engine.

    Since Kalshi doesn't provide extensive historical data via API,
    this backtester works in two modes:
    1. Simulated: Generate realistic trades based on strategy parameters
    2. Live replay: Replay actual fills from your trading history
    """

    def __init__(self, initial_balance: float = None):
        self.initial_balance = initial_balance or config.ACCOUNT_BALANCE

    def run_simulated_backtest(
        self,
        strategy_fn,
        n_trades: int = 200,
        win_rate: float = 0.55,
        avg_odds: float = 50,
        seed: int = None,
    ) -> BacktestResult:
        """
        Run a simulated backtest with configurable parameters.

        Args:
            strategy_fn: Callable that returns (side, count, price) for sizing
            n_trades: Number of simulated trades
            win_rate: Base win rate of the strategy
            avg_odds: Average price in cents
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        balance = self.initial_balance
        peak_balance = balance
        trades = []
        equity_curve = [balance]
        timestamps = [datetime.now(timezone.utc).isoformat()]

        for i in range(n_trades):
            # Simulate market price around avg_odds with noise
            price = max(5, min(95, int(avg_odds + np.random.normal(0, 15))))

            # Strategy determines position size
            count = strategy_fn(balance, price)
            if count <= 0:
                continue

            cost = count * price / 100.0

            # Can we afford it?
            if cost > balance * config.MAX_POSITION_PCT or cost > config.MAX_BET_DOLLARS:
                cost = min(balance * config.MAX_POSITION_PCT, config.MAX_BET_DOLLARS)
                count = max(1, int(cost / (price / 100.0)))
                cost = count * price / 100.0

            if cost <= 0 or cost > balance:
                continue

            # Simulate outcome
            is_win = random.random() < win_rate

            if is_win:
                pnl = count * (100 - price) / 100.0  # Profit from correct prediction
            else:
                pnl = -cost  # Lose the cost

            balance += pnl
            peak_balance = max(peak_balance, balance)

            ts = f"2025-01-{(i % 28) + 1:02d}T{(i * 7) % 24:02d}:00:00Z"
            trade = BacktestTrade(
                timestamp=ts,
                ticker=f"SIM-{i:04d}",
                side="yes",
                price_cents=price,
                count=count,
                cost=cost,
                pnl=round(pnl, 4),
                cumulative_pnl=round(balance - self.initial_balance, 4),
                result="win" if is_win else "loss",
            )
            trades.append(trade)
            equity_curve.append(balance)
            timestamps.append(ts)

            # Stop if account blown
            if balance <= 0:
                console.print("[red]Account blown during backtest.[/red]")
                break

        return self._compute_stats(trades, equity_curve, timestamps)

    def run_from_trades(self, trade_data: list[dict]) -> BacktestResult:
        """
        Backtest from actual trade data (list of dicts with price, pnl, etc.)
        """
        balance = self.initial_balance
        trades = []
        equity_curve = [balance]
        timestamps = [datetime.now(timezone.utc).isoformat()]

        for td in trade_data:
            pnl = td.get("pnl", 0)
            balance += pnl

            trade = BacktestTrade(
                timestamp=td.get("timestamp", ""),
                ticker=td.get("ticker", ""),
                side=td.get("side", "yes"),
                price_cents=td.get("price_cents", 50),
                count=td.get("count", 1),
                cost=td.get("cost", 0),
                pnl=round(pnl, 4),
                cumulative_pnl=round(balance - self.initial_balance, 4),
                result="win" if pnl > 0 else "loss",
                reason=td.get("reason", ""),
            )
            trades.append(trade)
            equity_curve.append(balance)
            timestamps.append(td.get("timestamp", ""))

        return self._compute_stats(trades, equity_curve, timestamps)

    def _compute_stats(
        self,
        trades: list[BacktestTrade],
        equity_curve: list[float],
        timestamps: list[str],
    ) -> BacktestResult:
        """Compute all performance statistics."""
        if not trades:
            return BacktestResult(
                trades=[], initial_balance=self.initial_balance,
                final_balance=self.initial_balance, total_pnl=0,
                roi_pct=0, sharpe_ratio=0, sortino_ratio=0,
                max_drawdown_pct=0, win_rate=0, avg_win=0, avg_loss=0,
                profit_factor=0, total_trades=0, winning_trades=0,
                losing_trades=0, equity_curve=equity_curve, timestamps=timestamps,
            )

        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_pnl = sum(pnls)
        final_balance = self.initial_balance + total_pnl

        # Sharpe ratio (annualized, assuming ~250 trading days)
        if len(pnls) > 1:
            returns = np.array(pnls) / self.initial_balance
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
            sharpe = (mean_return / std_return) * np.sqrt(250) if std_return > 0 else 0

            # Sortino (only downside deviation)
            downside = returns[returns < 0]
            downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 0
            sortino = (mean_return / downside_std) * np.sqrt(250) if downside_std > 0 else 0
        else:
            sharpe = 0
            sortino = 0

        # Max drawdown
        peak = equity_curve[0]
        max_dd = 0
        for val in equity_curve:
            peak = max(peak, val)
            dd = (peak - val) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return BacktestResult(
            trades=trades,
            initial_balance=self.initial_balance,
            final_balance=round(final_balance, 2),
            total_pnl=round(total_pnl, 2),
            roi_pct=round((total_pnl / self.initial_balance) * 100, 2),
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            max_drawdown_pct=round(max_dd * 100, 2),
            win_rate=round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
            avg_win=round(np.mean(wins), 4) if wins else 0,
            avg_loss=round(np.mean(losses), 4) if losses else 0,
            profit_factor=round(profit_factor, 2),
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            equity_curve=equity_curve,
            timestamps=timestamps,
        )

    # =========================================================================
    # CHARTS & VISUALIZATION
    # =========================================================================

    def plot_equity_curve(self, result: BacktestResult, save: bool = True) -> str:
        """Generate interactive Plotly equity curve + drawdown chart."""
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
            subplot_titles=("Equity Curve", "Drawdown"),
            vertical_spacing=0.08,
        )

        # Equity curve
        x_vals = list(range(len(result.equity_curve)))
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=result.equity_curve,
                mode="lines",
                name="Equity",
                line=dict(color="#00d4aa", width=2),
                fill="tozeroy",
                fillcolor="rgba(0,212,170,0.1)",
            ),
            row=1, col=1,
        )

        # Starting balance line
        fig.add_hline(
            y=result.initial_balance,
            line_dash="dash",
            line_color="rgba(255,255,255,0.3)",
            row=1, col=1,
        )

        # Drawdown
        peak = result.equity_curve[0]
        drawdowns = []
        for val in result.equity_curve:
            peak = max(peak, val)
            dd = ((peak - val) / peak * 100) if peak > 0 else 0
            drawdowns.append(-dd)

        fig.add_trace(
            go.Scatter(
                x=x_vals, y=drawdowns,
                mode="lines",
                name="Drawdown %",
                line=dict(color="#ff6b6b", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(255,107,107,0.2)",
            ),
            row=2, col=1,
        )

        # Layout
        fig.update_layout(
            template="plotly_dark",
            title=dict(
                text=f"Backtest Results | ROI: {result.roi_pct}% | Sharpe: {result.sharpe_ratio} | Win Rate: {result.win_rate}%",
                font=dict(size=16),
            ),
            height=700,
            showlegend=False,
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#16213e",
        )

        fig.update_xaxes(title_text="Trade #", row=2, col=1)
        fig.update_yaxes(title_text="Balance ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown %", row=2, col=1)

        if save:
            html_path = config.OUTPUT_DIR / "equity_curve.html"
            png_path = config.OUTPUT_DIR / "equity_curve.png"
            fig.write_html(str(html_path))
            try:
                fig.write_image(str(png_path), width=1200, height=700)
                console.print(f"  Chart saved: {png_path}")
            except Exception:
                pass  # kaleido may not be installed
            console.print(f"  Interactive chart: {html_path}")
            return str(html_path)

        fig.show()
        return ""

    def plot_trade_distribution(self, result: BacktestResult, save: bool = True) -> str:
        """Plot P&L distribution histogram."""
        pnls = [t.pnl for t in result.trades]
        if not pnls:
            return ""

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=pnls,
            nbinsx=30,
            marker_color=["#00d4aa" if p > 0 else "#ff6b6b" for p in sorted(pnls)],
            name="Trade P&L",
        ))

        fig.add_vline(x=0, line_dash="dash", line_color="white")
        fig.add_vline(x=np.mean(pnls), line_dash="dot", line_color="#ffd93d",
                      annotation_text=f"Mean: ${np.mean(pnls):.2f}")

        fig.update_layout(
            template="plotly_dark",
            title="Trade P&L Distribution",
            xaxis_title="P&L ($)",
            yaxis_title="Count",
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#16213e",
        )

        if save:
            path = config.OUTPUT_DIR / "pnl_distribution.html"
            fig.write_html(str(path))
            console.print(f"  Distribution chart: {path}")
            return str(path)

        fig.show()
        return ""

    def export_csv(self, result: BacktestResult) -> str:
        """Export trade-by-trade CSV."""
        csv_path = config.OUTPUT_DIR / "trades.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "ticker", "side", "price_cents", "count",
                "cost", "pnl", "cumulative_pnl", "result", "reason",
            ])
            for t in result.trades:
                writer.writerow([
                    t.timestamp, t.ticker, t.side, t.price_cents, t.count,
                    t.cost, t.pnl, t.cumulative_pnl, t.result, t.reason,
                ])

        console.print(f"  Trade log: {csv_path}")
        return str(csv_path)

    def display_summary(self, result: BacktestResult):
        """Print rich summary stats table."""
        table = Table(title="Backtest Summary", show_lines=True, width=60)
        table.add_column("Metric", style="cyan", width=30)
        table.add_column("Value", justify="right", style="white", width=25)

        pnl_color = "green" if result.total_pnl >= 0 else "red"
        roi_color = "green" if result.roi_pct >= 0 else "red"

        rows = [
            ("Initial Balance", f"${result.initial_balance:.2f}"),
            ("Final Balance", f"${result.final_balance:.2f}"),
            ("Total P&L", f"[{pnl_color}]${result.total_pnl:+.2f}[/{pnl_color}]"),
            ("ROI", f"[{roi_color}]{result.roi_pct:+.2f}%[/{roi_color}]"),
            ("", ""),
            ("Sharpe Ratio", f"{result.sharpe_ratio:.3f}"),
            ("Sortino Ratio", f"{result.sortino_ratio:.3f}"),
            ("Max Drawdown", f"[red]{result.max_drawdown_pct:.2f}%[/red]"),
            ("Profit Factor", f"{result.profit_factor:.2f}"),
            ("", ""),
            ("Total Trades", str(result.total_trades)),
            ("Win Rate", f"{result.win_rate:.1f}%"),
            ("Winning Trades", str(result.winning_trades)),
            ("Losing Trades", str(result.losing_trades)),
            ("Avg Win", f"${result.avg_win:.4f}"),
            ("Avg Loss", f"${result.avg_loss:.4f}"),
        ]

        for metric, value in rows:
            if metric == "":
                table.add_row("─" * 20, "─" * 15)
            else:
                table.add_row(metric, value)

        console.print(table)

    # =========================================================================
    # MONTE CARLO SIMULATION
    # =========================================================================

    def monte_carlo(
        self,
        result: BacktestResult,
        n_simulations: int = 500,
        n_trades: int = None,
    ) -> dict:
        """
        Monte Carlo simulation: resample trades to estimate outcome distribution.
        """
        if not result.trades:
            return {}

        pnls = [t.pnl for t in result.trades]
        n_trades = n_trades or len(pnls)

        final_balances = []
        max_drawdowns = []

        for _ in range(n_simulations):
            sampled = random.choices(pnls, k=n_trades)
            balance = self.initial_balance
            peak = balance
            max_dd = 0

            for p in sampled:
                balance += p
                peak = max(peak, balance)
                dd = (peak - balance) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

            final_balances.append(balance)
            max_drawdowns.append(max_dd * 100)

        fb = np.array(final_balances)
        mc_result = {
            "median_balance": round(float(np.median(fb)), 2),
            "mean_balance": round(float(np.mean(fb)), 2),
            "p5_balance": round(float(np.percentile(fb, 5)), 2),
            "p95_balance": round(float(np.percentile(fb, 95)), 2),
            "prob_profit": round(float(np.mean(fb > self.initial_balance)) * 100, 1),
            "prob_ruin": round(float(np.mean(fb <= 0)) * 100, 1),
            "median_max_dd": round(float(np.median(max_drawdowns)), 2),
        }

        # Plot Monte Carlo
        fig = go.Figure()
        for i in range(min(100, n_simulations)):
            sampled = random.choices(pnls, k=n_trades)
            curve = [self.initial_balance]
            for p in sampled:
                curve.append(curve[-1] + p)
            fig.add_trace(go.Scatter(
                y=curve, mode="lines",
                line=dict(width=0.5, color="rgba(0,212,170,0.15)"),
                showlegend=False,
            ))

        fig.add_hline(y=self.initial_balance, line_dash="dash", line_color="white")

        fig.update_layout(
            template="plotly_dark",
            title=f"Monte Carlo ({n_simulations} sims) | P(profit): {mc_result['prob_profit']}% | Median: ${mc_result['median_balance']}",
            xaxis_title="Trade #",
            yaxis_title="Balance ($)",
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#16213e",
        )

        mc_path = config.OUTPUT_DIR / "monte_carlo.html"
        fig.write_html(str(mc_path))
        console.print(f"  Monte Carlo chart: {mc_path}")

        # Print summary
        table = Table(title="Monte Carlo Summary", show_lines=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        for k, v in mc_result.items():
            table.add_row(k.replace("_", " ").title(), str(v))
        console.print(table)

        return mc_result
