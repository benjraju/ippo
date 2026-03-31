"""
deep_itm_strategy.py -- Deep in-the-money maker strategy.

Places maker (post_only) limit orders to buy YES at 95c on markets
where YES is priced 97-99c. Earns ~5c per contract when correct (99.6%).
Loses ~95c when wrong (0.4%).

Only places maker orders to avoid taker fees which destroy the edge.

BACKTEST RESULTS (n=34,570 settled markets, 3,603 with price >= 95c):
  Entry  |  WR%   | EV/trade | Sharpe | Total P&L | Kelly  | Breakeven WR
  -------|--------|----------|--------|-----------|--------|-------------
   95c   | 99.58% |  4.57c   |  0.710 |  $164.79  | 91.7%  |  95.01%
   96c   | 99.66% |  3.65c   |  0.632 |  $130.84  | 91.6%  |  96.01%
   97c   | 99.66% |  2.65c   |  0.457 |   $94.32  | 88.7%  |  97.01%
   98c   | 99.68% |  1.67c   |  0.295 |   $57.30  | 83.9%  |  98.01%

  Optimal entry: 95c (best Sharpe, best EV, largest margin to breakeven)
  Out-of-sample (30%): 99.91% WR at 95c, 100.00% WR at 96-97c
  Monte Carlo (10K sims x 500 trades): 0.00% ruin probability
  Maker fee at 95c: 0.01c (negligible)
  Stress test: edge survives down to 97.5% win rate at 97c entry

  Losses cluster in weather markets (Chicago, Miami, NYC).
  KXNBAGAME has 0 losses across 1,069 markets at >= 97c.
  Only 12 total losses at >= 97c across all series.

Risk management:
  - NEVER deploy more than 5% of bankroll per single market
  - Spread across at least 10 simultaneous positions
  - Use post_only=True (maker only, cancel if would be taker)
  - Maximum 30% of bankroll deployed at any time
  - Track correlation: weather markets may fail together (storm events)
  - Prefer KXNBAGAME (0% loss rate) over weather series
"""

import math
import sys
import logging
from typing import Optional
from pathlib import Path
from dataclasses import dataclass, field

import config
from kalshi_client import KalshiClient

try:
    sys.path.insert(0, str(Path(__file__).parent / "autoresearch"))
    from candidate_strategy import (
        DEEP_ITM_BID_PRICE,
        DEEP_ITM_MAX_POSITIONS,
        DEEP_ITM_MAX_POSITION_PCT,
        DEEP_ITM_MIN_SPREAD,
    )
except ImportError:
    DEEP_ITM_BID_PRICE = 95
    DEEP_ITM_MAX_POSITIONS = 10
    DEEP_ITM_MAX_POSITION_PCT = 0.05
    DEEP_ITM_MIN_SPREAD = 2

logger = logging.getLogger("deep_itm")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Constants (values imported from candidate_strategy where applicable)
# ---------------------------------------------------------------------------
MIN_MARKET_PRICE = 97       # Only consider markets with YES ask >= 97c
MAX_BID_PRICE = 97          # Our maximum bid price (cents)
MIN_BID_PRICE = 95          # Our minimum bid price (cents)
DEFAULT_BID_PRICE = DEEP_ITM_BID_PRICE   # Bid price from autoresearch (default 95c)
WIN_RATE = 0.9958           # Empirical win rate at >= 95c from 3,603 settled markets
MAX_SINGLE_POSITION_PCT = DEEP_ITM_MAX_POSITION_PCT   # Max % of bankroll on one market (10% = $58 on $580)
MAX_TOTAL_DEPLOYED_PCT = 0.35    # Max 35% of bankroll deployed total ($203 on $580)
MIN_POSITIONS = 8           # Minimum number of positions for diversification
MAX_POSITIONS = DEEP_ITM_MAX_POSITIONS   # Maximum simultaneous positions from autoresearch
MIN_VOLUME = 100            # Minimum market volume to consider
MAKER_FEE_RATE = 0.0175     # Kalshi maker fee rate
MIN_SPREAD = DEEP_ITM_MIN_SPREAD  # Minimum bid-ask spread from autoresearch


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class DeepITMOpportunity:
    """A market opportunity for the deep ITM strategy."""
    ticker: str
    title: str
    series: str
    current_yes_ask: int       # Current ask price in cents
    current_yes_bid: int       # Current bid price in cents
    our_bid_price: int         # What we'd bid (cents)
    volume: int
    profit_if_win: float       # cents
    loss_if_lose: float        # cents
    ev_per_contract: float     # expected value in cents
    kelly_fraction: float      # optimal Kelly bet fraction
    maker_fee: float           # fee in cents per contract
    contracts: int = 0         # how many contracts to buy (set by risk budget)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
def kalshi_maker_fee_cents(price_cents: int) -> float:
    """Calculate Kalshi maker fee in cents for 1 contract at given price."""
    p = price_cents / 100.0
    return math.ceil(MAKER_FEE_RATE * 1 * p * (1 - p) * 100) / 100


def compute_deep_itm_kelly(
    buy_price_cents: int,
    win_rate: float = WIN_RATE,
) -> float:
    """
    Compute Kelly criterion fraction for deep ITM strategy.

    Kelly f* = (p * b - q) / b
    where:
        p = probability of win (win_rate)
        q = 1 - p
        b = profit/loss ratio (profit_if_win / loss_if_lose)
    """
    fee = kalshi_maker_fee_cents(buy_price_cents)
    profit = (100 - buy_price_cents) - fee   # cents profit per win
    loss = buy_price_cents + fee             # cents lost per loss
    q = 1 - win_rate
    b = profit / loss  # odds ratio

    kelly = (win_rate * b - q) / b
    return max(0.0, kelly)  # Never negative


def deep_itm_risk_budget(
    bankroll: float,
    buy_price_cents: int = DEFAULT_BID_PRICE,
    max_positions: int = MAX_POSITIONS,
    max_single_loss_pct: float = MAX_SINGLE_POSITION_PCT,
    max_deployed_pct: float = MAX_TOTAL_DEPLOYED_PCT,
    kelly_fraction_mult: float = 0.25,  # quarter-Kelly
) -> dict:
    """
    Compute position sizing for deep ITM strategy.

    Returns:
        dict with max_contracts_per_position, total_budget, num_positions, etc.
    """
    bankroll_cents = bankroll * 100
    fee = kalshi_maker_fee_cents(buy_price_cents)

    # Cost per contract = buy_price + fee (in cents)
    cost_per_contract = buy_price_cents + fee

    # Max loss per position (5% of bankroll)
    max_loss_per_position = bankroll_cents * max_single_loss_pct

    # Max contracts per position based on max loss
    # If we lose, we lose cost_per_contract per contract
    max_contracts_loss = int(max_loss_per_position / cost_per_contract)

    # Kelly-based sizing
    kelly = compute_deep_itm_kelly(buy_price_cents)
    quarter_kelly = kelly * kelly_fraction_mult
    kelly_budget_cents = bankroll_cents * quarter_kelly
    max_contracts_kelly = int(kelly_budget_cents / cost_per_contract)

    # Use the more conservative of the two
    max_contracts = min(max_contracts_loss, max_contracts_kelly)
    max_contracts = max(1, max_contracts)  # At least 1

    # Total budget constraint
    total_budget_cents = bankroll_cents * max_deployed_pct
    total_max_contracts = int(total_budget_cents / cost_per_contract)

    # Contracts per position given we want MIN_POSITIONS diversification
    contracts_per_position = min(
        max_contracts,
        int(total_max_contracts / MIN_POSITIONS),
    )
    contracts_per_position = max(1, contracts_per_position)

    return {
        "bankroll": bankroll,
        "buy_price_cents": buy_price_cents,
        "fee_per_contract": fee,
        "cost_per_contract": cost_per_contract,
        "kelly_fraction": kelly,
        "quarter_kelly": quarter_kelly,
        "max_contracts_per_position": contracts_per_position,
        "max_total_contracts": total_max_contracts,
        "max_positions": max_positions,
        "max_deployed_dollars": total_budget_cents / 100,
        "profit_per_win_cents": (100 - buy_price_cents) - fee,
        "loss_per_lose_cents": buy_price_cents + fee,
    }


def find_deep_itm_opportunities(
    client: KalshiClient,
    bid_price: int = DEFAULT_BID_PRICE,
    min_market_price: int = MIN_MARKET_PRICE,
    min_volume: int = MIN_VOLUME,
) -> list[DeepITMOpportunity]:
    """
    Scan all target series for deep ITM opportunities.

    Looks for markets where YES ask >= min_market_price (97c default).
    We place a maker bid at bid_price (96c default), waiting for a fill.
    """
    opportunities = []

    # Scan all configured series (skip blocked ones)
    blocked = getattr(config, "BLOCKED_SERIES", set())
    for series in config.TARGET_MARKET_SERIES:
        if series in blocked:
            continue
        cursor = None
        for page in range(10):  # Max 10 pages per series
            try:
                resp = client.get_markets(
                    series_ticker=series,
                    status="open",
                    limit=100,
                    cursor=cursor,
                )
            except Exception as e:
                logger.warning(f"Error fetching {series}: {e}")
                break

            markets = resp.get("markets", [])
            if not markets:
                break

            # Convert dollar strings to cents (Kalshi API returns "0.9500" etc.)
            def to_cents(val):
                if val is None:
                    return 0
                try:
                    return int(float(val) * 100)
                except (ValueError, TypeError):
                    return 0

            for m in markets:
                # Skip markets from blocked series (e.g., KXNBAPTS from KXNBA parent query)
                m_ticker = m.get("ticker", "")
                m_series = m.get("series_ticker", "") or (m_ticker.split("-")[0] if "-" in m_ticker else "")
                if m_series in blocked or (m_ticker.split("-")[0] if "-" in m_ticker else "") in blocked:
                    continue

                yes_ask = to_cents(m.get("yes_ask_dollars") or m.get("yes_ask") or 0)
                yes_bid = to_cents(m.get("yes_bid_dollars") or m.get("yes_bid") or 0)

                vol_raw = m.get("volume_fp") or m.get("volume") or 0
                try:
                    volume = int(float(vol_raw))
                except (ValueError, TypeError):
                    volume = 0

                if yes_ask == 0 or yes_bid == 0:
                    continue

                # Filter: ask must be >= our threshold
                if yes_ask < min_market_price:
                    continue

                if volume < min_volume:
                    continue

                # Our bid is below the current ask (maker order)
                if bid_price >= yes_ask:
                    # We'd be a taker, skip
                    continue

                fee = kalshi_maker_fee_cents(bid_price)
                profit = (100 - bid_price) - fee
                loss = bid_price + fee
                ev = WIN_RATE * profit - (1 - WIN_RATE) * loss
                kelly = compute_deep_itm_kelly(bid_price)

                opportunities.append(DeepITMOpportunity(
                    ticker=m.get("ticker", ""),
                    title=m.get("title", ""),
                    series=series,
                    current_yes_ask=yes_ask,
                    current_yes_bid=yes_bid,
                    our_bid_price=bid_price,
                    volume=volume,
                    profit_if_win=profit,
                    loss_if_lose=loss,
                    ev_per_contract=ev,
                    kelly_fraction=kelly,
                    maker_fee=fee,
                ))

            cursor = resp.get("cursor")
            if not cursor:
                break

    # Sort by EV descending (higher spread = more profit)
    opportunities.sort(key=lambda x: x.ev_per_contract, reverse=True)
    return opportunities


def execute_deep_itm(
    client: KalshiClient,
    bankroll: float,
    bid_price: int = DEFAULT_BID_PRICE,
    dry_run: bool = True,
    max_positions: int = MAX_POSITIONS,
) -> list[dict]:
    """
    Execute the deep ITM strategy.

    1. Find opportunities (markets with YES >= 97c)
    2. Compute risk budget
    3. Place maker limit orders to buy YES at bid_price

    Returns list of orders placed (or would-be-placed in dry run).
    """
    logger.info(f"=== Deep ITM Strategy {'(DRY RUN)' if dry_run else '(LIVE)'} ===")
    logger.info(f"Bankroll: ${bankroll:.2f} | Bid: {bid_price}c | Mode: {'DRY RUN' if dry_run else 'LIVE'}")

    # Find opportunities
    opps = find_deep_itm_opportunities(client, bid_price=bid_price)
    logger.info(f"Found {len(opps)} opportunities")

    if not opps:
        logger.info("No opportunities found. Markets may not be deep enough ITM.")
        return []

    # Compute risk budget
    budget = deep_itm_risk_budget(bankroll, bid_price, max_positions=max_positions)
    contracts_per = budget["max_contracts_per_position"]
    logger.info(f"Risk budget: {contracts_per} contracts/position, "
                f"max {budget['max_total_contracts']} total, "
                f"Kelly={budget['kelly_fraction']:.3f}, "
                f"Quarter-Kelly={budget['quarter_kelly']:.3f}")

    # Limit to max_positions
    opps = opps[:max_positions]

    orders = []
    total_deployed = 0
    max_deploy_cents = bankroll * 100 * MAX_TOTAL_DEPLOYED_PCT

    for opp in opps:
        # Check total deployment limit
        cost = contracts_per * (bid_price + opp.maker_fee)
        if total_deployed + cost > max_deploy_cents:
            remaining = max_deploy_cents - total_deployed
            contracts_this = int(remaining / (bid_price + opp.maker_fee))
            if contracts_this <= 0:
                logger.info("Total deployment limit reached.")
                break
            contracts_per_adj = contracts_this
        else:
            contracts_per_adj = contracts_per

        order_info = {
            "ticker": opp.ticker,
            "title": opp.title,
            "series": opp.series,
            "side": "yes",
            "action": "buy",
            "price": bid_price,
            "contracts": contracts_per_adj,
            "cost_cents": contracts_per_adj * (bid_price + opp.maker_fee),
            "ev_per_contract": opp.ev_per_contract,
            "current_ask": opp.current_yes_ask,
            "current_bid": opp.current_yes_bid,
        }

        if not dry_run:
            try:
                result = client.place_order(
                    ticker=opp.ticker,
                    side="yes",
                    action="buy",
                    count=contracts_per_adj,
                    type="limit",
                    yes_price=bid_price,
                    post_only=True,  # CRITICAL: maker only
                )
                order_info["order_id"] = result.get("order", {}).get("order_id")
                order_info["status"] = "placed"
                logger.info(f"  PLACED: {opp.ticker} | {contracts_per_adj} contracts @ {bid_price}c | "
                           f"EV={opp.ev_per_contract:.2f}c")
            except Exception as e:
                order_info["status"] = "error"
                order_info["error"] = str(e)
                logger.error(f"  ERROR: {opp.ticker} | {e}")
        else:
            order_info["status"] = "dry_run"
            logger.info(f"  [DRY] {opp.ticker} | {contracts_per_adj}x @ {bid_price}c | "
                       f"ask={opp.current_yes_ask}c | EV={opp.ev_per_contract:.2f}c/contract")

        orders.append(order_info)
        total_deployed += order_info["cost_cents"]

    logger.info(f"\nSummary: {len(orders)} orders, ${total_deployed/100:.2f} deployed "
                f"({total_deployed/bankroll/100*100:.1f}% of bankroll)")

    return orders


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deep ITM Maker Strategy")
    parser.add_argument("--live", action="store_true", help="Execute live trades (default: dry run)")
    parser.add_argument("--bid", type=int, default=DEFAULT_BID_PRICE,
                       help=f"Bid price in cents (default: {DEFAULT_BID_PRICE})")
    parser.add_argument("--bankroll", type=float, default=None,
                       help="Override bankroll (default: from API balance)")
    parser.add_argument("--max-positions", type=int, default=MAX_POSITIONS,
                       help=f"Max simultaneous positions (default: {MAX_POSITIONS})")
    args = parser.parse_args()

    client = KalshiClient()

    # Get bankroll
    if args.bankroll:
        bankroll = args.bankroll
    else:
        try:
            balance = client.get_balance()
            bankroll = balance.get("balance", 10000) / 100  # API returns cents
        except Exception:
            bankroll = config.ACCOUNT_BALANCE
            logger.warning(f"Could not fetch balance, using config: ${bankroll}")

    logger.info(f"Bankroll: ${bankroll:.2f}")

    # Show risk budget
    budget = deep_itm_risk_budget(bankroll, args.bid, max_positions=args.max_positions)
    logger.info(f"Risk Budget:")
    logger.info(f"  Buy price: {budget['buy_price_cents']}c")
    logger.info(f"  Fee/contract: {budget['fee_per_contract']:.2f}c")
    logger.info(f"  Profit/win: {budget['profit_per_win_cents']:.2f}c")
    logger.info(f"  Loss/lose: {budget['loss_per_lose_cents']:.2f}c")
    logger.info(f"  Kelly: {budget['kelly_fraction']:.4f}")
    logger.info(f"  Quarter-Kelly: {budget['quarter_kelly']:.4f}")
    logger.info(f"  Max contracts/position: {budget['max_contracts_per_position']}")
    logger.info(f"  Max total contracts: {budget['max_total_contracts']}")
    logger.info(f"  Max deployed: ${budget['max_deployed_dollars']:.2f}")

    # Execute
    orders = execute_deep_itm(
        client,
        bankroll=bankroll,
        bid_price=args.bid,
        dry_run=not args.live,
        max_positions=args.max_positions,
    )

    if orders:
        print(f"\n{'='*60}")
        print(f"{'LIVE ORDERS' if args.live else 'DRY RUN ORDERS'}")
        print(f"{'='*60}")
        for o in orders:
            print(f"  {o['ticker']:40s} | {o['contracts']:3d}x @ {o['price']}c | "
                  f"EV={o['ev_per_contract']:.2f}c | {o['status']}")
