"""
market_scanner.py — Scans Kalshi for high-volume, fast-settling markets.
Focuses on: weather (KXHIGH/KXLOW), college basketball (NCAA), crypto buckets.
"""

import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd
from rich.console import Console
from rich.table import Table

import config
from kalshi_client import KalshiClient

console = Console()


class MarketScanner:
    """Finds tradeable high-volume markets on Kalshi."""

    def __init__(self, client: Optional[KalshiClient] = None):
        self.client = client or KalshiClient()

    def scan_all_target_series(self) -> pd.DataFrame:
        """
        Scan all target market series for active, high-volume markets.
        Returns a DataFrame with market details sorted by volume.
        """
        all_markets = []

        for series in config.TARGET_MARKET_SERIES:
            try:
                markets = self._scan_series(series)
                all_markets.extend(markets)
            except Exception as e:
                console.print(f"  [yellow]Warning: Could not scan {series}: {e}[/yellow]")

        if not all_markets:
            console.print("[red]No markets found. Check your API keys and connection.[/red]")
            return pd.DataFrame()

        df = pd.DataFrame(all_markets)

        # Filter by volume and time to settlement
        if "volume" in df.columns:
            df = df[df["volume"] >= config.MIN_MARKET_VOLUME]
        if "hours_to_settlement" in df.columns:
            df = df[df["hours_to_settlement"] <= config.MAX_HOURS_TO_SETTLEMENT]

        df = df.sort_values("volume", ascending=False).reset_index(drop=True)
        return df

    def _scan_series(self, series_ticker: str) -> list[dict]:
        """Scan a single series for open markets."""
        markets = []
        cursor = None

        while True:
            resp = self.client.get_markets(
                series_ticker=series_ticker,
                status="open",
                limit=100,
                cursor=cursor,
            )
            for m in resp.get("markets", []):
                parsed = self._parse_market(m)
                if parsed:
                    markets.append(parsed)

            cursor = resp.get("cursor")
            if not cursor:
                break
            time.sleep(0.2)  # Rate limiting

        return markets

    def _parse_market(self, raw: dict) -> Optional[dict]:
        """Parse raw market data into a clean dict."""
        try:
            now = datetime.now(timezone.utc)
            close_time = datetime.fromisoformat(
                raw.get("close_time", "").replace("Z", "+00:00")
            )
            hours_to_settlement = (close_time - now).total_seconds() / 3600

            # Skip markets that are about to close (< 30 min) or too far out
            if hours_to_settlement < 0.5:
                return None

            yes_price = raw.get("yes_bid", 0) or 0
            no_price = raw.get("no_bid", 0) or 0
            yes_ask = raw.get("yes_ask", 0) or 0
            last_price = raw.get("last_price", 0) or 0
            volume = raw.get("volume", 0) or 0

            # Calculate spread
            spread = (yes_ask - yes_price) if (yes_ask and yes_price) else 99

            return {
                "ticker": raw.get("ticker", ""),
                "title": raw.get("title", "")[:60],
                "series": raw.get("series_ticker", ""),
                "event_ticker": raw.get("event_ticker", ""),
                "yes_price": yes_price,
                "no_price": no_price,
                "yes_ask": yes_ask,
                "last_price": last_price,
                "spread": spread,
                "volume": volume,
                "open_interest": raw.get("open_interest", 0) or 0,
                "hours_to_settlement": round(hours_to_settlement, 1),
                "close_time": close_time.isoformat(),
                "result": raw.get("result", ""),
                "status": raw.get("status", ""),
            }
        except Exception:
            return None

    def get_orderbook_snapshot(self, ticker: str) -> dict:
        """Get current orderbook for a market."""
        try:
            book = self.client.get_market_orderbook(ticker, depth=5)
            return {
                "ticker": ticker,
                "yes_bids": book.get("yes", []),
                "no_bids": book.get("no", []),
            }
        except Exception as e:
            return {"ticker": ticker, "error": str(e)}

    def display_markets(self, df: pd.DataFrame, top_n: int = 20):
        """Pretty-print market scan results to terminal."""
        if df.empty:
            console.print("[red]No markets to display.[/red]")
            return

        table = Table(title="Active High-Volume Markets", show_lines=True)
        table.add_column("Ticker", style="cyan", width=25)
        table.add_column("Title", style="white", width=40)
        table.add_column("Yes$", justify="right", style="green")
        table.add_column("Spread", justify="right")
        table.add_column("Vol", justify="right", style="yellow")
        table.add_column("Hrs", justify="right", style="magenta")

        for _, row in df.head(top_n).iterrows():
            spread_color = "green" if row["spread"] <= 5 else "yellow" if row["spread"] <= 10 else "red"
            table.add_row(
                str(row["ticker"]),
                str(row["title"]),
                f"{row['yes_price']}c",
                f"[{spread_color}]{row['spread']}c[/{spread_color}]",
                str(row["volume"]),
                str(row["hours_to_settlement"]),
            )

        console.print(table)
        console.print(f"\n[dim]Showing top {min(top_n, len(df))} of {len(df)} markets[/dim]")
