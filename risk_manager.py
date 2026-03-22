"""
risk_manager.py — Position sizing, Kelly criterion, and loss caps.
ALL DEFAULTS ARE EXTREMELY CONSERVATIVE for a $100 account.
"""

import json
import math
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path

import config


@dataclass
class TradeProposal:
    """A proposed trade before risk checks."""
    ticker: str
    side: str          # "yes" or "no"
    action: str        # "buy" or "sell"
    estimated_prob: float   # Our model's probability estimate (0-1)
    market_price: float     # Current market price in cents (1-99)
    edge: float             # estimated_prob - (market_price/100)
    reason: str = ""


@dataclass
class ApprovedTrade:
    """A trade that passed all risk checks."""
    ticker: str
    side: str
    action: str
    count: int              # Number of contracts
    price_cents: int        # Limit price in cents
    max_loss: float         # Maximum possible loss in dollars
    expected_value: float   # Expected value of the trade
    kelly_size: float       # Raw Kelly fraction
    reason: str = ""


@dataclass
class DailyRiskState:
    """Tracks daily P&L for loss cap enforcement."""
    date: str = ""
    realized_pnl: float = 0.0
    open_positions: int = 0
    trades_today: int = 0
    blocked: bool = False


class RiskManager:
    """
    Conservative risk management for prediction market trading.

    Rules:
    - Quarter Kelly sizing (very conservative)
    - Max $2 per trade on $100 account
    - Max 8% daily loss cap ($8)
    - Max 5 concurrent positions
    - Minimum 5-cent edge required
    """

    def __init__(self, account_balance: float = None):
        self.balance = account_balance or config.ACCOUNT_BALANCE
        self.daily_state = self._load_state()
        self.trade_log: list[dict] = []

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    @staticmethod
    def _state_file_path() -> Path:
        return config.OUTPUT_DIR / "daily_risk_state.json"

    def _load_state(self) -> DailyRiskState:
        """Load daily risk state from disk, or start fresh."""
        path = self._state_file_path()
        today = self._today()
        try:
            if path.exists():
                data = json.loads(path.read_text())
                if data.get("date") == today:
                    return DailyRiskState(
                        date=today,
                        realized_pnl=float(data.get("realized_pnl", 0.0)),
                        open_positions=int(data.get("open_positions", 0)),
                        trades_today=int(data.get("trades_today", 0)),
                        blocked=bool(data.get("blocked", False)),
                    )
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            pass  # Corrupted or unreadable — start fresh
        return DailyRiskState(date=today)

    def _save_state(self):
        """Persist daily risk state to disk."""
        path = self._state_file_path()
        data = {
            "date": self.daily_state.date,
            "realized_pnl": self.daily_state.realized_pnl,
            "trades_today": self.daily_state.trades_today,
            "open_positions": self.daily_state.open_positions,
            "blocked": self.daily_state.blocked,
        }
        try:
            path.write_text(json.dumps(data, indent=2) + "\n")
        except OSError:
            pass  # Non-fatal — state will reload next time

    def _reset_daily_if_needed(self):
        """Reset daily counters if it's a new day."""
        today = self._today()
        if self.daily_state.date != today:
            self.daily_state = DailyRiskState(date=today)
            self._save_state()

    def evaluate_trade(self, proposal: TradeProposal) -> ApprovedTrade | None:
        """
        Run all risk checks on a proposed trade.
        Returns ApprovedTrade if it passes, None if rejected.
        """
        self._reset_daily_if_needed()

        # --- Check 1: Daily loss cap ---
        max_daily_loss = self.balance * config.MAX_DAILY_LOSS_PCT
        if self.daily_state.realized_pnl <= -max_daily_loss:
            self.daily_state.blocked = True
            self._save_state()
            return None  # Daily loss cap hit

        # --- Check 2: Maximum open positions ---
        if self.daily_state.open_positions >= config.MAX_OPEN_POSITIONS:
            return None  # Too many open positions

        # --- Check 3: Minimum edge ---
        if abs(proposal.edge) < config.MIN_EDGE_THRESHOLD:
            return None  # Edge too small

        # --- Check 4: Kelly sizing ---
        kelly_size = self._quarter_kelly(proposal.estimated_prob, proposal.market_price)
        if kelly_size <= 0:
            return None  # Kelly says don't bet

        # --- Check 5: Position sizing ---
        max_dollars = min(
            kelly_size * self.balance,                      # Kelly-sized amount
            self.balance * config.MAX_POSITION_PCT,         # Max position %
            config.MAX_BET_DOLLARS,                          # Hard dollar cap
            max_daily_loss + self.daily_state.realized_pnl,  # Remaining daily budget
        )

        if max_dollars < 0.01:
            return None  # Can't afford any position

        # Calculate contract count based on price
        price_cents = int(proposal.market_price)
        if price_cents <= 0 or price_cents >= 100:
            return None

        # Cost per contract = price in cents / 100 (in dollars)
        cost_per_contract = price_cents / 100.0
        count = max(1, int(max_dollars / cost_per_contract))

        # Ensure we don't exceed max dollars
        actual_cost = count * cost_per_contract
        if actual_cost > max_dollars:
            count = max(1, int(max_dollars / cost_per_contract))
            actual_cost = count * cost_per_contract

        # Expected value calculation
        if proposal.side == "yes":
            ev = (proposal.estimated_prob * (100 - price_cents) / 100.0 -
                  (1 - proposal.estimated_prob) * price_cents / 100.0) * count
        else:
            ev = ((1 - proposal.estimated_prob) * (100 - (100 - price_cents)) / 100.0 -
                  proposal.estimated_prob * (100 - price_cents) / 100.0) * count

        return ApprovedTrade(
            ticker=proposal.ticker,
            side=proposal.side,
            action=proposal.action,
            count=count,
            price_cents=price_cents,
            max_loss=actual_cost,
            expected_value=round(ev, 4),
            kelly_size=round(kelly_size, 4),
            reason=proposal.reason,
        )

    def _quarter_kelly(self, prob: float, market_price_cents: float) -> float:
        """
        Quarter Kelly criterion for binary outcomes.

        Full Kelly: f* = (b*p - q) / b
        where b = odds, p = win prob, q = 1-p
        Quarter Kelly: f*/4 (much more conservative)
        """
        if prob <= 0 or prob >= 1:
            return 0.0

        market_prob = market_price_cents / 100.0
        q = 1 - prob

        # Decimal odds from market price
        if market_prob <= 0 or market_prob >= 1:
            return 0.0

        b = (1 - market_prob) / market_prob  # odds offered

        full_kelly = (b * prob - q) / b
        if full_kelly <= 0:
            return 0.0

        return full_kelly * config.KELLY_FRACTION

    def record_fill(self, ticker: str, pnl: float):
        """Record a completed trade for daily tracking."""
        self._reset_daily_if_needed()
        self.daily_state.realized_pnl += pnl
        self.daily_state.trades_today += 1
        self._save_state()
        self.trade_log.append({
            "ticker": ticker,
            "pnl": pnl,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cumulative_pnl": self.daily_state.realized_pnl,
        })

    def update_positions(self, count: int):
        """Update open position count."""
        self.daily_state.open_positions = count
        self._save_state()

    def get_status(self) -> dict:
        """Get current risk status summary."""
        self._reset_daily_if_needed()
        max_daily = self.balance * config.MAX_DAILY_LOSS_PCT
        return {
            "balance": self.balance,
            "daily_pnl": round(self.daily_state.realized_pnl, 2),
            "daily_loss_cap": round(max_daily, 2),
            "daily_budget_remaining": round(max_daily + self.daily_state.realized_pnl, 2),
            "open_positions": self.daily_state.open_positions,
            "trades_today": self.daily_state.trades_today,
            "is_blocked": self.daily_state.blocked,
            "max_per_trade": config.MAX_BET_DOLLARS,
            "kelly_fraction": config.KELLY_FRACTION,
        }
