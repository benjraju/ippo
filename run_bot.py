"""
run_bot.py — Main trading bot runner.
Scans markets, evaluates edges, and places trades (paper or live).

Usage:
    python run_bot.py --paper     # Paper/demo trading (DEFAULT and SAFE)
    python run_bot.py --live      # Real money (DANGER — read warnings first)
"""

import sys
import time
import argparse
from datetime import datetime, timezone

from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.table import Table

import config
from kalshi_client import KalshiClient
from market_scanner import MarketScanner
from arb_scanner import ArbScanner
from claude_agent import ClaudeAgent
from risk_manager import RiskManager, TradeProposal
from backtester import Backtester

console = Console()


LIVE_WARNING = """
╔══════════════════════════════════════════════════════════════╗
║                    ⚠️  LIVE TRADING MODE  ⚠️                  ║
║                                                              ║
║  You are about to trade with REAL MONEY on Kalshi.           ║
║                                                              ║
║  Risk parameters:                                            ║
║  • Max $2 per trade                                          ║
║  • Max $8 daily loss (8% of $100)                            ║
║  • Quarter Kelly sizing                                      ║
║                                                              ║
║  This bot is EXPERIMENTAL. You WILL lose money sometimes.    ║
║  Only trade with money you can afford to lose completely.    ║
║                                                              ║
║  Type 'YES I UNDERSTAND THE RISKS' to continue:             ║
╚══════════════════════════════════════════════════════════════╝
"""


def validate_env():
    """Check that required environment variables are set."""
    errors = []
    if not config.KALSHI_API_KEY_ID or config.KALSHI_API_KEY_ID == "your_kalshi_api_key_id_here":
        errors.append("KALSHI_API_KEY_ID not set in .env")
    if not config.KALSHI_PRIVATE_KEY_PATH:
        errors.append("KALSHI_PRIVATE_KEY_PATH not set in .env")
    if errors:
        console.print("[red]Configuration errors:[/red]")
        for e in errors:
            console.print(f"  [red]• {e}[/red]")
        console.print("\n[yellow]Fix your .env file and try again. See README.md for help.[/yellow]")
        sys.exit(1)


def run_trading_loop(mode: str = "paper", interval: int = 60):
    """
    Main trading loop.

    1. Scan markets for high-volume opportunities
    2. Run Claude analysis on top candidates
    3. Check arb scanner
    4. Evaluate risk on promising trades
    5. Place approved trades
    6. Sleep and repeat
    """
    validate_env()

    is_live = mode == "live"

    if is_live:
        console.print(LIVE_WARNING)
        confirmation = input("> ").strip()
        if confirmation != "YES I UNDERSTAND THE RISKS":
            console.print("[yellow]Live trading cancelled. Running in paper mode instead.[/yellow]")
            is_live = False

    env_label = "LIVE 💰" if is_live else "PAPER 📝"
    api_env = "PROD" if is_live else "DEMO"

    # Override env for this session
    if is_live:
        config.KALSHI_ENV = "PROD"
    else:
        config.KALSHI_ENV = "DEMO"

    console.print(Panel(
        f"[bold]Kalshi Trading Bot — {env_label}[/bold]\n"
        f"Account: ${config.ACCOUNT_BALANCE} | Max/trade: ${config.MAX_BET_DOLLARS} | "
        f"Daily cap: {config.MAX_DAILY_LOSS_PCT*100}%\n"
        f"API: {config.get_base_url()[:40]}...",
        title="Bot Started",
        border_style="cyan",
    ))

    # Initialize components
    try:
        client = KalshiClient()
        scanner = MarketScanner(client)
        arb = ArbScanner(client)
        agent = ClaudeAgent()
        risk = RiskManager()
    except FileNotFoundError as e:
        console.print(f"[red]Setup error: {e}[/red]")
        console.print("[yellow]Make sure your private key .pem file is in the project directory.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Initialization error: {e}[/red]")
        sys.exit(1)

    cycle = 0
    while True:
        cycle += 1
        now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        console.print(f"\n[cyan]━━━ Cycle {cycle} | {now} ━━━[/cyan]")

        try:
            # Step 1: Scan markets
            console.print("[dim]Scanning markets...[/dim]")
            markets_df = scanner.scan_all_target_series()

            if markets_df.empty:
                console.print("[yellow]No active markets found. Waiting...[/yellow]")
                time.sleep(interval)
                continue

            console.print(f"  Found {len(markets_df)} active markets")

            # Show top markets
            scanner.display_markets(markets_df, top_n=10)

            # Step 2: Check for arb opportunities
            console.print("[dim]Scanning for arbitrage...[/dim]")
            arb_opps = arb.full_scan(markets_df)
            if arb_opps:
                console.print(f"  [green]Found {len(arb_opps)} opportunities![/green]")
                arb.display_opportunities(arb_opps, top_n=5)

            # Step 3: Claude analysis on top markets
            console.print("[dim]Analyzing top markets with Claude...[/dim]")
            analyses = agent.analyze_batch(markets_df, top_n=5)

            # Step 4: Generate trade proposals
            proposals = []
            for analysis in analyses:
                edge = analysis.get("edge_vs_market", 0)
                if abs(edge) >= config.MIN_EDGE_THRESHOLD:
                    proposals.append(TradeProposal(
                        ticker=analysis["ticker"],
                        side=analysis.get("recommended_side", "yes"),
                        action=analysis.get("recommended_action", "buy"),
                        estimated_prob=analysis["estimated_probability"],
                        market_price=analysis["market_price"],
                        edge=edge,
                        reason=analysis.get("reasoning", ""),
                    ))

            if not proposals:
                console.print("  [dim]No trades meet edge threshold this cycle.[/dim]")
            else:
                console.print(f"  [green]{len(proposals)} trade proposals generated[/green]")

            # Step 5: Risk check and execute
            for proposal in proposals:
                approved = risk.evaluate_trade(proposal)
                if approved:
                    console.print(
                        f"  [bold green]TRADE: {approved.side.upper()} {approved.ticker} "
                        f"x{approved.count} @ {approved.price_cents}c "
                        f"(EV: ${approved.expected_value:+.4f})[/bold green]"
                    )
                    console.print(f"    Reason: {approved.reason}")

                    # Place the order
                    try:
                        result = client.place_order(
                            ticker=approved.ticker,
                            side=approved.side,
                            action=approved.action,
                            count=approved.count,
                            type="limit",
                            yes_price=approved.price_cents if approved.side == "yes" else None,
                            no_price=approved.price_cents if approved.side == "no" else None,
                        )
                        console.print(f"    [green]Order placed: {result.get('order', {}).get('order_id', 'OK')}[/green]")
                    except Exception as e:
                        console.print(f"    [red]Order failed: {e}[/red]")

            # Step 6: Show risk status
            status = risk.get_status()
            console.print(f"\n  [dim]Daily P&L: ${status['daily_pnl']:+.2f} | "
                         f"Budget left: ${status['daily_budget_remaining']:.2f} | "
                         f"Positions: {status['open_positions']}/{config.MAX_OPEN_POSITIONS}[/dim]")

            if status["is_blocked"]:
                console.print("  [red]⚠ Daily loss cap reached. No more trades today.[/red]")

        except KeyboardInterrupt:
            console.print("\n[yellow]Bot stopped by user.[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error in cycle {cycle}: {e}[/red]")

        console.print(f"[dim]Next scan in {interval}s... (Ctrl+C to stop)[/dim]")
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Kalshi Self-Improving Trading Bot")
    parser.add_argument("--paper", action="store_true", default=True, help="Paper/demo trading (DEFAULT)")
    parser.add_argument("--live", action="store_true", help="Live trading with real money (DANGER)")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between scan cycles (default: 60)")
    args = parser.parse_args()

    mode = "live" if args.live else "paper"
    run_trading_loop(mode=mode, interval=args.interval)


if __name__ == "__main__":
    main()
