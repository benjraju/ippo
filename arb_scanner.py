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
from config import KALSHI_FEE_PER_SIDE_CENTS

KALSHI_FEE_CENTS_PER_SIDE = KALSHI_FEE_PER_SIDE_CENTS  # alias for existing references

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
        Check if best_ask(YES) + best_ask(NO) < 100c after fees.

        IMPORTANT: We use the ACTUAL ASK prices from the orderbook, not the
        market data's yes_price/no_price (which are mid/last prices). Using
        mid prices gave false signals -- the market showed YES@45 + NO@50 = 95c
        but the actual asks were YES@48 + NO@53 = 101c, making the arb illusory.

        For a true risk-free arb, both sides must be fillable IMMEDIATELY at
        their ask prices. We place limit orders AT the ask (not above) to get
        maker fees, but we only signal when the ask-based math works.

        In practice, Kalshi spreads are wide enough that
        ask(YES) + ask(NO) >= 100c after fees on virtually every market.
        This scanner has found 0 true arbs in 5,500+ scan cycles.
        """
        opps = []

        for _, row in markets_df.iterrows():
            ticker = row.get("ticker", "")
            if not ticker:
                continue

            # Fetch the actual orderbook to get real ask prices
            try:
                ob = self.client.get_market_orderbook(ticker, depth=3)
            except Exception:
                continue

            book = ob.get("orderbook", {})
            yes_asks = book.get("yes", [])  # [[price, qty], ...]
            no_asks = book.get("no", [])

            if not yes_asks or not no_asks:
                continue

            # Best (cheapest) ask on each side
            yes_ask = min(yes_asks, key=lambda x: x[0])
            no_ask = min(no_asks, key=lambda x: x[0])

            yes_price = yes_ask[0]  # cents
            yes_depth = yes_ask[1]
            no_price = no_ask[0]    # cents
            no_depth = no_ask[1]

            total = yes_price + no_price

            # Calculate actual maker fees for both sides
            # fee = ceil(rate * contracts * price * (1-price)) per side
            MAKER_RATE = 0.0175
            yes_fee = MAKER_RATE * yes_price * (100 - yes_price) / 100.0
            no_fee = MAKER_RATE * no_price * (100 - no_price) / 100.0
            total_fees = yes_fee + no_fee

            edge_after_fees = 100 - total - total_fees

            if edge_after_fees <= 0:
                continue

            opps.append(ArbOpportunity(
                type="yes_no_arb",
                ticker=ticker,
                description=(
                    f"ASK: Yes({yes_price}c x{yes_depth}) + No({no_price}c x{no_depth}) "
                    f"= {total}c | fees={total_fees:.1f}c | net edge={edge_after_fees:.1f}c"
                ),
                edge_cents=edge_after_fees,
                confidence=0.95,
                details={
                    "yes_price": yes_price, "no_price": no_price,
                    "yes_depth": yes_depth, "no_depth": no_depth,
                    "total": total, "fees": round(total_fees, 2),
                },
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
