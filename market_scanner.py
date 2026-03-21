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

        # First do a broad scan (no series filter) to catch all active markets
        try:
            markets = self._scan_series(None)
            all_markets.extend(markets)
        except Exception as e:
            console.print(f"  [yellow]Warning: Broad scan failed: {e}[/yellow]")

        # Also scan specific target series
        for series in config.TARGET_MARKET_SERIES:
            try:
                markets = self._scan_series(series)
                # Deduplicate by ticker
                existing = {m["ticker"] for m in all_markets}
                all_markets.extend(m for m in markets if m["ticker"] not in existing)
            except Exception as e:
                pass  # Series may not exist on demo

        if not all_markets:
            console.print("[red]No markets found. Check your API keys and connection.[/red]")
            return pd.DataFrame()

        df = pd.DataFrame(all_markets)

        # Filter by volume and time to settlement
        # Filter by minimum volume (lowered for demo mode)
        min_vol = 0 if config.KALSHI_ENV == "DEMO" else max(1, config.MIN_MARKET_VOLUME // 10)
        if "volume" in df.columns and min_vol > 0:
            df = df[df["volume"] >= min_vol]
        if "hours_to_settlement" in df.columns:
            df = df[df["hours_to_settlement"] <= config.MAX_HOURS_TO_SETTLEMENT]

        df = df.sort_values("volume", ascending=False).reset_index(drop=True)
        return df

    def _scan_series(self, series_ticker: str = None) -> list[dict]:
        """Scan markets, optionally filtered by series. Pass None for broad scan."""
        markets = []
        cursor = None
        max_pages = 5  # Limit pages to avoid rate limits

        for _ in range(max_pages):
            try:
                resp = self.client.get_markets(
                    series_ticker=series_ticker,
                    status="open",
                    limit=100,
                    cursor=cursor,
                )
            except Exception:
                break

            for m in resp.get("markets", []):
                parsed = self._parse_market(m)
                if parsed:
                    markets.append(parsed)

            cursor = resp.get("cursor")
            if not cursor:
                break
            time.sleep(1.0)  # Rate limiting — Kalshi enforces strict limits

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

            # API v2 returns prices as dollar strings (e.g., "0.5000")
            # Convert to cents (integer 0-99) for internal use
            def to_cents(val):
                if val is None:
                    return 0
                try:
                    return int(float(val) * 100)
                except (ValueError, TypeError):
                    return 0

            yes_price = to_cents(raw.get("yes_bid_dollars") or raw.get("yes_bid") or 0)
            no_price = to_cents(raw.get("no_bid_dollars") or raw.get("no_bid") or 0)
            yes_ask = to_cents(raw.get("yes_ask_dollars") or raw.get("yes_ask") or 0)
            last_price = to_cents(raw.get("last_price_dollars") or raw.get("last_price") or 0)

            # Volume: try float string first, then int
            vol_raw = raw.get("volume_fp") or raw.get("volume_24h_fp") or raw.get("volume") or 0
            try:
                volume = int(float(vol_raw))
            except (ValueError, TypeError):
                volume = 0

            # Calculate spread in cents
            spread = (yes_ask - yes_price) if (yes_ask and yes_price) else 99

            # Open interest
            oi_raw = raw.get("open_interest_fp") or raw.get("open_interest") or 0
            try:
                open_interest = int(float(oi_raw))
            except (ValueError, TypeError):
                open_interest = 0

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
                "open_interest": open_interest,
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

        table = Table(title="Active High-Volume Markets", show_lines=False, padding=(0, 1))
        table.add_column("Ticker", style="cyan", max_width=35, no_wrap=True)
        table.add_column("Title", style="white", max_width=45)
        table.add_column("Yes", justify="right", style="green", min_width=5)
        table.add_column("Sprd", justify="right", min_width=5)
        table.add_column("Volume", justify="right", style="yellow", min_width=7)
        table.add_column("Hrs", justify="right", style="magenta", min_width=5)

        for _, row in df.head(top_n).iterrows():
            spread_color = "green" if row["spread"] <= 5 else "yellow" if row["spread"] <= 10 else "red"
            vol_str = f"{row['volume']:,}" if row['volume'] > 0 else "-"
            table.add_row(
                str(row["ticker"])[:35],
                str(row["title"])[:45],
                f"{row['yes_price']}c",
                f"[{spread_color}]{row['spread']}c[/{spread_color}]",
                vol_str,
                f"{row['hours_to_settlement']:.0f}",
            )

        console.print(table)
        console.print(f"\n[dim]Showing top {min(top_n, len(df))} of {len(df)} markets[/dim]")
