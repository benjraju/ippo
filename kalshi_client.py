"""
kalshi_client.py — Authenticated Kalshi API client.
Handles RSA-signed requests to demo or production API.
"""

import time
import json
import base64
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, utils

import config


class KalshiClient:
    """Authenticated REST client for Kalshi Trade API v2."""

    def __init__(self):
        self.base_url = config.get_base_url()
        self.api_key_id = config.KALSHI_API_KEY_ID
        self.private_key = self._load_private_key()
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _load_private_key(self):
        """Load RSA private key from PEM file."""
        key_path = Path(config.KALSHI_PRIVATE_KEY_PATH)
        if not key_path.exists():
            raise FileNotFoundError(
                f"Private key not found at {key_path}. "
                "Download it from your Kalshi account settings."
            )
        with open(key_path, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password=None)

    def _sign_request(self, method: str, path: str, timestamp_ms: int) -> str:
        """Create RSA signature for API authentication."""
        # Kalshi v2 signature: timestamp + method + path
        message = f"{timestamp_ms}{method}{path}"
        signature = self.private_key.sign(
            message.encode("utf-8"),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _request(self, method: str, path: str, params: dict = None, data: dict = None):
        """Make authenticated request to Kalshi API."""
        url = f"{self.base_url}{path}"
        timestamp_ms = int(time.time() * 1000)
        signature = self._sign_request(method.upper(), path, timestamp_ms)

        headers = {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
        }

        resp = self.session.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=data,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    # =========================================================================
    # MARKET DATA
    # =========================================================================

    def get_markets(
        self,
        limit: int = 100,
        cursor: str = None,
        event_ticker: str = None,
        series_ticker: str = None,
        status: str = "open",
    ) -> dict:
        """Get list of markets with optional filters."""
        params = {"limit": limit, "status": status}
        if cursor:
            params["cursor"] = cursor
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        return self._request("GET", "/markets", params=params)

    def get_market(self, ticker: str) -> dict:
        """Get single market details."""
        return self._request("GET", f"/markets/{ticker}")

    def get_market_orderbook(self, ticker: str, depth: int = 10) -> dict:
        """Get market order book."""
        return self._request("GET", f"/markets/{ticker}/orderbook", params={"depth": depth})

    def get_market_history(self, ticker: str, limit: int = 1000) -> dict:
        """Get trade history for a market."""
        return self._request("GET", f"/markets/{ticker}/trades", params={"limit": limit})

    def get_series(self, series_ticker: str) -> dict:
        """Get series info."""
        return self._request("GET", f"/series/{series_ticker}")

    def get_events(self, series_ticker: str = None, status: str = None, limit: int = 100) -> dict:
        """Get events, optionally filtered by series."""
        params = {"limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status
        return self._request("GET", "/events", params=params)

    # =========================================================================
    # TRADING
    # =========================================================================

    def place_order(
        self,
        ticker: str,
        side: str,  # "yes" or "no"
        action: str,  # "buy" or "sell"
        count: int,  # number of contracts
        type: str = "limit",
        yes_price: int = None,  # price in cents (1-99)
        no_price: int = None,
        expiration_ts: int = None,
    ) -> dict:
        """
        Place an order on a market.

        ⚠️  This sends a REAL order if KALSHI_ENV=PROD.
        In DEMO mode, orders are simulated on Kalshi's demo platform.
        """
        order = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": type,
        }
        if yes_price is not None:
            order["yes_price"] = yes_price
        if no_price is not None:
            order["no_price"] = no_price
        if expiration_ts is not None:
            order["expiration_ts"] = expiration_ts

        return self._request("POST", "/portfolio/orders", data=order)

    def cancel_order(self, order_id: str) -> dict:
        """Cancel an open order."""
        return self._request("DELETE", f"/portfolio/orders/{order_id}")

    def get_positions(self) -> dict:
        """Get current portfolio positions."""
        return self._request("GET", "/portfolio/positions")

    def get_balance(self) -> dict:
        """Get account balance."""
        return self._request("GET", "/portfolio/balance")

    def get_orders(self, ticker: str = None, status: str = None) -> dict:
        """Get order history."""
        params = {}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status
        return self._request("GET", "/portfolio/orders", params=params)

    def get_fills(self, ticker: str = None, limit: int = 100) -> dict:
        """Get fill history."""
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        return self._request("GET", "/portfolio/fills", params=params)
