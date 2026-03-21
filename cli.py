"""
cli.py — Simple command-line interface for the Kalshi trading bot.

Commands:
    python cli.py scan-markets          # See active high-volume markets
    python cli.py arb-scan              # Find arbitrage opportunities
    python cli.py backtest              # Run backtest with charts
    python cli.py backtest --plot       # Same, opens chart in browser
    python cli.py research --overnight  # Start AutoResearch (overnight)
    python cli.py research --quick      # Quick 10-iteration research
    python cli.py status                # Show account status
    python cli.py analyze TICKER        # Analyze a specific market
"""

import sys
import webbrowser

import click
from rich.console import Console

import config

console = Console()


@click.group()
def cli():
    """Kalshi Self-Improving Trading Bot CLI"""
    pass


@cli.command("scan-markets")
@click.option("--top", default=20, help="Number of markets to show")
def scan_markets(top):
    """Scan for active high-volume markets."""
    from kalshi_client import KalshiClient
    from market_scanner import MarketScanner

    console.print("[cyan]Scanning Kalshi for high-volume markets...[/cyan]\n")
    try:
        client = KalshiClient()
        scanner = MarketScanner(client)
        df = scanner.scan_all_target_series()
        scanner.display_markets(df, top_n=top)
    except FileNotFoundError:
        console.print("[red]Private key not found. Check KALSHI_PRIVATE_KEY_PATH in .env[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Make sure your .env is configured correctly.[/yellow]")


@cli.command("arb-scan")
@click.option("--top", default=15, help="Number of opportunities to show")
def arb_scan(top):
    """Scan for arbitrage and mispricing opportunities."""
    from kalshi_client import KalshiClient
    from market_scanner import MarketScanner
    from arb_scanner import ArbScanner

    console.print("[cyan]Scanning for arbitrage opportunities...[/cyan]\n")
    try:
        client = KalshiClient()
        scanner = MarketScanner(client)
        arb = ArbScanner(client)
        markets_df = scanner.scan_all_target_series()
        opps = arb.full_scan(markets_df)
        arb.display_opportunities(opps, top_n=top)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command("backtest")
@click.option("--trades", default=200, help="Number of simulated trades")
@click.option("--win-rate", default=0.55, help="Assumed win rate (0.0-1.0)")
@click.option("--plot", is_flag=True, help="Open charts in browser")
@click.option("--monte-carlo", is_flag=True, help="Run Monte Carlo simulation")
@click.option("--seed", default=42, help="Random seed for reproducibility")
def backtest(trades, win_rate, plot, monte_carlo, seed):
    """Run a backtest with full P&L visualization."""
    from backtester import Backtester

    sys.path.insert(0, str(config.AUTORESEARCH_DIR))
    try:
        from autoresearch.candidate_strategy import calculate_position_size, estimate_probability
    except ImportError:
        calculate_position_size = None

    console.print("[cyan]Running backtest...[/cyan]\n")

    bt = Backtester(initial_balance=config.ACCOUNT_BALANCE)

    # Use current strategy for sizing if available
    if calculate_position_size:
        def strategy_sizer(balance, price):
            prob = estimate_probability(price, 100, 24)
            return calculate_position_size(balance, prob, price)
    else:
        def strategy_sizer(balance, price):
            max_bet = min(balance * config.MAX_POSITION_PCT, config.MAX_BET_DOLLARS)
            cost_per = price / 100.0
            return max(1, int(max_bet / cost_per)) if cost_per > 0 else 0

    result = bt.run_simulated_backtest(
        strategy_fn=strategy_sizer,
        n_trades=trades,
        win_rate=win_rate,
        seed=seed,
    )

    # Display results
    bt.display_summary(result)

    # Export CSV
    csv_path = bt.export_csv(result)

    # Generate charts
    html_path = bt.plot_equity_curve(result)
    bt.plot_trade_distribution(result)

    if monte_carlo:
        console.print("\n[cyan]Running Monte Carlo simulation...[/cyan]")
        bt.monte_carlo(result, n_simulations=500)

    if plot and html_path:
        webbrowser.open(f"file://{html_path}")

    console.print(f"\n[green]All outputs saved to: {config.OUTPUT_DIR}[/green]")


@cli.command("research")
@click.option("--overnight", is_flag=True, help="Run full overnight research (50 iterations)")
@click.option("--quick", is_flag=True, help="Quick 10-iteration test")
@click.option("--iterations", default=None, type=int, help="Custom iteration count")
@click.option("--no-claude", is_flag=True, help="Use random mutations only (no API cost)")
def research(overnight, quick, iterations, no_claude):
    """Start AutoResearch strategy improvement loop."""
    from autoresearch.research_loop import run_research

    if overnight:
        n = iterations or 50
        console.print(f"[cyan]Starting overnight AutoResearch ({n} iterations)...[/cyan]")
        console.print("[dim]This will run for a while. Leave your Terminal open.[/dim]")
        console.print("[dim]Press Ctrl+C to stop at any time.[/dim]\n")
    elif quick:
        n = iterations or 10
        console.print(f"[cyan]Quick research run ({n} iterations)...[/cyan]\n")
    else:
        n = iterations or 25
        console.print(f"[cyan]Research run ({n} iterations)...[/cyan]\n")

    run_research(
        max_iterations=n,
        use_claude=not no_claude,
        verbose=True,
    )

    # Show updated backtest after research
    console.print("\n[cyan]Running backtest with improved strategy...[/cyan]")
    from backtester import Backtester
    sys.path.insert(0, str(config.AUTORESEARCH_DIR))

    # Force reimport
    if "autoresearch.candidate_strategy" in sys.modules:
        del sys.modules["autoresearch.candidate_strategy"]

    from autoresearch.candidate_strategy import calculate_position_size, estimate_probability

    bt = Backtester(initial_balance=config.ACCOUNT_BALANCE)

    def strategy_sizer(balance, price):
        prob = estimate_probability(price, 100, 24)
        return calculate_position_size(balance, prob, price)

    result = bt.run_simulated_backtest(strategy_fn=strategy_sizer, n_trades=250, seed=42)
    bt.display_summary(result)
    bt.plot_equity_curve(result)
    bt.export_csv(result)
    console.print(f"\n[green]Charts and data saved to: {config.OUTPUT_DIR}[/green]")


@cli.command("status")
def status():
    """Show account balance and positions."""
    from kalshi_client import KalshiClient

    console.print("[cyan]Fetching account status...[/cyan]\n")
    try:
        client = KalshiClient()
        balance = client.get_balance()
        positions = client.get_positions()

        console.print(f"  Balance: ${balance.get('balance', 0) / 100:.2f}")
        console.print(f"  Environment: {config.KALSHI_ENV}")

        pos_list = positions.get("market_positions", [])
        if pos_list:
            console.print(f"\n  Open positions ({len(pos_list)}):")
            for p in pos_list[:10]:
                console.print(f"    {p.get('ticker', '?')}: {p.get('position', 0)} contracts")
        else:
            console.print("  No open positions.")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command("analyze")
@click.argument("ticker")
def analyze(ticker):
    """Analyze a specific market with Claude."""
    from kalshi_client import KalshiClient
    from claude_agent import ClaudeAgent

    console.print(f"[cyan]Analyzing market: {ticker}[/cyan]\n")
    try:
        client = KalshiClient()
        market = client.get_market(ticker)
        m = market.get("market", {})

        console.print(f"  Title: {m.get('title', '?')}")
        console.print(f"  Yes: {m.get('yes_bid', '?')}c | No: {m.get('no_bid', '?')}c")
        console.print(f"  Volume: {m.get('volume', 0)} | OI: {m.get('open_interest', 0)}")

        agent = ClaudeAgent()
        result = agent.estimate_probability(
            market_title=m.get("title", ticker),
            yes_price=m.get("yes_bid", 50),
            no_price=m.get("no_bid", 50),
            volume=m.get("volume", 0),
            hours_to_settlement=24,
        )

        if result:
            console.print(f"\n  [bold]Claude Analysis:[/bold]")
            console.print(f"    Estimated probability: {result['estimated_probability']:.1%}")
            console.print(f"    Edge vs market: {result['edge_vs_market']:+.1%}")
            console.print(f"    Confidence: {result['confidence']}")
            console.print(f"    Reasoning: {result['reasoning']}")
            console.print(f"    Recommendation: {result['recommended_action']} {result['recommended_side']}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    cli()
