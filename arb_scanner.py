"""
arb_scanner.py — Finds arbitrage and mispricing opportunities across Kalshi markets.
Scans for: cross-market arb, yes/no mispricing, stale orderbooks.
"""

from dataclasses import dataclass

import pandas as pd
from rich.console import Console
from rich.table import Table

from kalshi_client import KalshiClient
from market_scanner import MarketScanner

console = Console()


@dataclass
class ArbOpportunity:
    """A detected arbitrage or mispricing opportunity."""
    type: str          # "yes_no_arb", "cross_event_arb", "stale_price"
    ticker: str
    description: str
    edge_cents: float  # Edge in cents
    confidence: float  # 0-1
    details: dict


class ArbScanner:
    """
    Scans for arbitrage opportunities on Kalshi.

    Types of opportunities:
    1. Yes/No mispricing: yes_price + no_price != 100 (rare but happens)
    2. Cross-event arb: Same underlying, different events mispriced
    3. Stale orderbook: Prices that haven't moved with new information
    """

    def __init__(self, client: KalshiClient = None):
        self.client = client or KalshiClient()
        self.scanner = MarketScanner(self.client)

    def scan_yes_no_arb(self, markets_df: pd.DataFrame) -> list[ArbOpportunity]:
        """
        Check if yes_price + no_price != 100 for any market.
        On Kalshi, yes and no should sum to ~100 cents.
        Any deviation is a potential arb.
        """
        opps = []

        for _, row in markets_df.iterrows():
            yes_p = row.get("yes_price", 0) or 0
            no_p = row.get("no_price", 0) or 0

            if yes_p <= 0 or no_p <= 0:
                continue

            total = yes_p + no_p
            # If total < 100, you can buy both sides for less than $1 and guarantee $1 payout
            if total < 97:  # 3-cent threshold to account for spread
                edge = 100 - total
                opps.append(ArbOpportunity(
                    type="yes_no_arb",
                    ticker=row["ticker"],
                    description=f"Yes({yes_p}c) + No({no_p}c) = {total}c < 100c → {edge}c edge",
                    edge_cents=edge,
                    confidence=0.9,
                    details={"yes_price": yes_p, "no_price": no_p, "total": total},
                ))

        return sorted(opps, key=lambda x: x.edge_cents, reverse=True)

    def scan_cross_event_arb(self, markets_df: pd.DataFrame) -> list[ArbOpportunity]:
        """
        Look for mispricings within the same event.
        E.g., if an event has mutually exclusive outcomes that should sum to ~100%.
        """
        opps = []

        # Group markets by event
        if "event_ticker" not in markets_df.columns:
            return opps

        for event, group in markets_df.groupby("event_ticker"):
            if len(group) < 2:
                continue

            # Sum of all yes prices in mutually exclusive event
            total_yes = group["yes_price"].sum()

            # For mutually exclusive events, total should be ~100
            if total_yes > 0 and abs(total_yes - 100) > 5:
                edge = abs(total_yes - 100)
                direction = "overpriced" if total_yes > 100 else "underpriced"
                opps.append(ArbOpportunity(
                    type="cross_event_arb",
                    ticker=str(event),
                    description=f"Event {event}: {len(group)} markets sum to {total_yes}c ({direction})",
                    edge_cents=edge,
                    confidence=0.7,
                    details={
                        "event": event,
                        "market_count": len(group),
                        "total_yes": total_yes,
                        "markets": group["ticker"].tolist(),
                    },
                ))

        return sorted(opps, key=lambda x: x.edge_cents, reverse=True)

    def scan_spread_opportunities(self, markets_df: pd.DataFrame) -> list[ArbOpportunity]:
        """Find markets with unusually wide spreads (potential stale prices)."""
        opps = []

        for _, row in markets_df.iterrows():
            spread = row.get("spread", 99)
            volume = row.get("volume", 0)

            # High volume + wide spread = potential opportunity
            if volume > 100 and spread > 8:
                opps.append(ArbOpportunity(
                    type="wide_spread",
                    ticker=row["ticker"],
                    description=f"Spread={spread}c with Vol={volume} — possible stale quotes",
                    edge_cents=spread / 2,  # Approximate edge as half the spread
                    confidence=0.5,
                    details={"spread": spread, "volume": volume},
                ))

        return sorted(opps, key=lambda x: x.edge_cents, reverse=True)

    def full_scan(self, markets_df: pd.DataFrame = None) -> list[ArbOpportunity]:
        """Run all arb scanners and return consolidated results."""
        if markets_df is None:
            markets_df = self.scanner.scan_all_target_series()

        all_opps = []
        all_opps.extend(self.scan_yes_no_arb(markets_df))
        all_opps.extend(self.scan_cross_event_arb(markets_df))
        all_opps.extend(self.scan_spread_opportunities(markets_df))

        return sorted(all_opps, key=lambda x: (x.confidence, x.edge_cents), reverse=True)

    def display_opportunities(self, opps: list[ArbOpportunity], top_n: int = 15):
        """Pretty-print arb opportunities."""
        if not opps:
            console.print("[yellow]No arbitrage opportunities found right now.[/yellow]")
            return

        table = Table(title="Arbitrage & Mispricing Scanner", show_lines=True)
        table.add_column("Type", style="cyan", width=15)
        table.add_column("Ticker", style="white", width=25)
        table.add_column("Description", style="white", width=50)
        table.add_column("Edge", justify="right", style="green")
        table.add_column("Conf", justify="right", style="magenta")

        for opp in opps[:top_n]:
            conf_color = "green" if opp.confidence > 0.7 else "yellow" if opp.confidence > 0.5 else "red"
            table.add_row(
                opp.type,
                opp.ticker,
                opp.description,
                f"{opp.edge_cents:.1f}c",
                f"[{conf_color}]{opp.confidence:.0%}[/{conf_color}]",
            )

        console.print(table)
