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
    python cli.py btc-edges             # Find BTC daily range edges
    python cli.py check-settlements     # Check settled trades and P&L
    python cli.py daily-summary         # Full daily P&L report
"""

import sys
import webbrowser

import click
from rich.console import Console

import config


# "hajime" = begin (Japanese) — opens the daily briefing
def _auto_briefing():
    """Run briefing if no command or 'hajime' specified."""
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] == "hajime"):
        try:
            from briefing import run_briefing
            run_briefing()
            sys.exit(0)
        except Exception:
            pass


_auto_briefing()

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
    except (ImportError, OSError):
        calculate_position_size = None

    console.print("[cyan]Running backtest...[/cyan]\n")

    bt = Backtester(initial_balance=config.ACCOUNT_BALANCE)

    # For simulation, use a simple sizer that respects risk limits
    # (The real strategy edge filter doesn't apply to random simulated prices)
    if calculate_position_size:
        def strategy_sizer(balance, price):
            # Use strategy's Kelly sizing but with a guaranteed edge for simulation
            frac = min(config.MAX_POSITION_PCT, 0.02)
            max_bet = min(balance * frac, config.MAX_BET_DOLLARS)
            cost_per = price / 100.0
            return max(1, int(max_bet / cost_per)) if cost_per > 0 else 0
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
@click.option(
    "--strategy",
    type=click.Choice(["weather", "crypto", "sports", "all"]),
    default="weather",
    show_default=True,
    help="Which strategy to optimize",
)
def research(overnight, quick, iterations, no_claude, strategy):
    """Start AutoResearch strategy improvement loop."""
    from autoresearch.research_loop import (
        run_research, run_crypto_research, run_sports_research,
    )

    if overnight:
        n = iterations or 50
        console.print(f"[cyan]Starting overnight AutoResearch ({n} iterations, strategy={strategy})...[/cyan]")
        console.print("[dim]This will run for a while. Leave your Terminal open.[/dim]")
        console.print("[dim]Press Ctrl+C to stop at any time.[/dim]\n")
    elif quick:
        n = iterations or 10
        console.print(f"[cyan]Quick research run ({n} iterations, strategy={strategy})...[/cyan]\n")
    else:
        n = iterations or 25
        console.print(f"[cyan]Research run ({n} iterations, strategy={strategy})...[/cyan]\n")

    if strategy in ("weather", "all"):
        run_research(max_iterations=n, verbose=True)
    if strategy in ("crypto", "all"):
        run_crypto_research(max_iterations=n, verbose=True)
    if strategy in ("sports", "all"):
        run_sports_research(max_iterations=n, verbose=True)

    # Show updated backtest after research
    console.print("\n[cyan]Running backtest with improved strategy...[/cyan]")
    from backtester import Backtester
    sys.path.insert(0, str(config.AUTORESEARCH_DIR))

    # Force reimport
    if "autoresearch.candidate_strategy" in sys.modules:
        del sys.modules["autoresearch.candidate_strategy"]

    from autoresearch.candidate_strategy import MAX_POSITION_DOLLARS

    bt = Backtester(initial_balance=config.ACCOUNT_BALANCE)

    def strategy_sizer(balance, price):
        max_bet = min(balance * 0.03, MAX_POSITION_DOLLARS, 2.0)
        cost_per = price / 100.0
        return max(1, int(max_bet / cost_per)) if cost_per > 0 else 0

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


@cli.command("weather-edges")
def weather_edges():
    """Find weather forecast vs. market price edges (main strategy)."""
    from weather_strategy import find_weather_edges, display_edges

    console.print("[cyan]Fetching NWS forecasts and comparing to Kalshi prices...[/cyan]\n")
    edges = find_weather_edges()
    display_edges(edges)

    if edges:
        top = edges[0]
        console.print(f"\n[bold green]Best edge: {top.city} — {top.side.replace('_', ' ').upper()} "
                      f"@ {top.market_yes_ask if top.side == 'buy_yes' else 100-top.market_yes_price:.0f}c "
                      f"(fair: {top.fair_value:.0f}c, edge: +{top.edge:.0f}c)[/bold green]")
        console.print(f"  Forecast: {top.forecast_temp:.0f}°F | Ticker: {top.ticker}")


@cli.command("btc-edges")
def btc_edges():
    """Find BTC daily range model vs. market edges (crypto strategy)."""
    from crypto_strategy import find_btc_edges, display_edges

    console.print("[cyan]Fetching BTC price, volatility, and comparing to Kalshi KXBTC markets...[/cyan]\n")
    edges = find_btc_edges()
    display_edges(edges)

    if edges:
        top = edges[0]
        action = "BUY YES" if top.side == "buy_yes" else "BUY NO"
        if top.bucket_high >= top.current_price * 4:
            bucket_str = f">= ${top.bucket_low:,.0f}"
        elif top.bucket_low <= 0:
            bucket_str = f"<= ${top.bucket_high:,.0f}"
        else:
            bucket_str = f"${top.bucket_low:,.0f}-${top.bucket_high:,.0f}"
        console.print(
            f"\n[bold green]Best edge: {action} on {bucket_str} "
            f"@ {top.entry_price:.0f}c (fair: {top.fair_value:.0f}c, "
            f"edge: +{top.edge:.0f}c)[/bold green]"
        )
        console.print(
            f"  BTC: ${top.current_price:,.0f} | Vol: {top.vol_used:.0%} | "
            f"Ticker: {top.ticker}"
        )


@cli.command("nba-edges")
@click.option("--backtest", is_flag=True, help="Run backtest against recent games")
@click.option("--days", default=14, help="Number of days to backtest (default: 14)")
def nba_edges(backtest, days):
    """Find NBA/March Madness model vs. market edges."""
    from sports_strategy import (
        find_nba_edges, display_edges,
        backtest_model, display_backtest,
    )

    console.print("[cyan]Fetching ESPN data and comparing to Kalshi NBA prices...[/cyan]\n")

    if backtest:
        console.print("[cyan]Running model backtest...[/cyan]\n")
        result = backtest_model(n_days=days)
        display_backtest(result)
    else:
        edges = find_nba_edges()
        display_edges(edges)

        if edges:
            top = edges[0]
            console.print(
                f"\n[bold green]Best edge: {top.team_a} vs {top.team_b} — "
                f"{top.side.replace('_', ' ').upper()} "
                f"@ {top.market_price:.0f}c "
                f"(fair: {top.fair_value:.0f}c, edge: +{top.edge:.0f}c)[/bold green]"
            )
            console.print(f"  Model spread: {top.model_spread:+.1f} | Ticker: {top.ticker}")


@cli.command("briefing")
def briefing():
    """Quick terminal status briefing."""
    from briefing import run_briefing
    run_briefing()


@cli.command("dashboard")
def dashboard():
    """Open live HTML dashboard with KPIs, positions, and bot health."""
    from dashboard import run_dashboard
    run_dashboard()


@cli.command("check-settlements")
def check_settlements():
    """Check settled trades and calculate real P&L."""
    from settlement_tracker import SettlementTracker

    try:
        tracker = SettlementTracker()
        tracker.check_settlements()
    except FileNotFoundError:
        console.print("[red]Private key not found. Check KALSHI_PRIVATE_KEY_PATH in .env[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command("daily-summary")
def daily_summary():
    """Full daily P&L report with performance metrics."""
    from settlement_tracker import SettlementTracker

    try:
        tracker = SettlementTracker()
        tracker.daily_summary()
    except FileNotFoundError:
        console.print("[red]Private key not found. Check KALSHI_PRIVATE_KEY_PATH in .env[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command("hourly-update")
def hourly_update():
    """Send hourly Telegram status update (balance, positions, P&L, new trades)."""
    from hourly_update import send_hourly_update
    send_hourly_update()
    console.print("[green]Hourly update sent via Telegram.[/green]")


@cli.command("auto-trade")
@click.option("--live", is_flag=True, help="Place real orders (DANGER)")
@click.option("--dry-run", is_flag=True, default=True, help="Show what would happen (default)")
@click.option("--weather-only", is_flag=True, help="Run only weather session")
@click.option("--btc-only", is_flag=True, help="Run only BTC session")
@click.option("--sports-only", is_flag=True, help="Run only sports/NBA session")
@click.option("--arb-only", is_flag=True, help="Run only arbitrage session")
@click.option("--copy-only", is_flag=True, help="Run only copy-trade session")
def auto_trade(live, dry_run, weather_only, btc_only, sports_only, arb_only, copy_only):
    """Autonomous daily trading: weather + crypto + sports + arb + copy-trade."""
    from auto_trade import run_auto_trade

    is_live = live
    any_only = any([weather_only, btc_only, sports_only, arb_only, copy_only])
    do_weather = weather_only if any_only else True
    do_btc = btc_only if any_only else True
    do_sports = sports_only if any_only else True
    do_arb = arb_only if any_only else True
    do_copy = copy_only if any_only else True

    if is_live:
        console.print("\n[bold red]LIVE MODE -- Real orders will be placed[/bold red]")
        console.print(f"  Environment: {config.KALSHI_ENV}")
        console.print(f"  Max per trade: $2")
        console.print(f"  Daily loss cap: 8% of account\n")
        confirmation = click.prompt(
            "Type 'YES' to continue", default="no"
        )
        if confirmation != "YES":
            console.print("[yellow]Aborted. Use --dry-run to preview trades.[/yellow]")
            return

    mode_label = "[red]LIVE[/red]" if is_live else "[green]DRY-RUN[/green]"
    console.print(f"\n[cyan]Starting autonomous trading session ({mode_label})...[/cyan]\n")

    decisions = run_auto_trade(
        dry_run=not is_live,
        weather=do_weather,
        btc=do_btc,
        sports=do_sports,
        arb=do_arb,
        copy=do_copy,
    )

    if decisions:
        placed = [d for d in decisions if d.placed or (not is_live and d.contracts > 0)]
        console.print(f"\n[bold]Result: {len(placed)} trades, {len(decisions)} total decisions[/bold]")
        console.print(f"[dim]Full log: output/auto_trade_{__import__('datetime').date.today()}.log[/dim]")
    else:
        console.print("[yellow]No trades found this session.[/yellow]")


if __name__ == "__main__":
    cli()
