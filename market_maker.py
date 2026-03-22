"""
market_maker.py — Avellaneda-Stoikov market maker for Kalshi binary markets.

Captures the bid-ask spread by quoting both sides. No prediction needed —
just mathematical inventory-aware pricing.

Core math (Avellaneda-Stoikov adapted for binary markets):

    Reservation price:  r = mid - q * gamma * sigma^2 * tau
    Optimal spread:     delta = gamma * sigma^2 * tau + (2/gamma) * ln(1 + gamma/kappa)
    Bid:                r - delta/2
    Ask:                r + delta/2

Where:
    mid   = (best_yes_bid + best_yes_ask) / 2
    q     = net inventory (positive = long YES)
    gamma = risk aversion parameter (higher = wider spreads, less inventory risk)
    sigma = volatility in logit space: std(log(mid / (1 - mid)))
    tau   = time remaining as fraction [0, 1]
    kappa = order arrival rate sensitivity

Usage:
    # Dry run — shows what it would do
    python market_maker.py

    # Live trading
    python market_maker.py --live

    # Specify tickers
    python market_maker.py --tickers KXBTCD-26MAR22-T50000 KXBTCD-26MAR22-T55000
"""

import logging
import math
import time
import argparse
from datetime import datetime, timezone
from typing import Optional

import config
from kalshi_client import KalshiClient

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("market_maker")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_GAMMA = 0.1       # Risk aversion
DEFAULT_KAPPA = 1.5       # Order arrival rate sensitivity
MAX_POSITION = 20         # Max contracts per market (long or short)
MIN_SPREAD_CENTS = 2      # Never quote tighter than 2c
MAX_SPREAD_CENTS = 15     # Never quote wider than 15c
MIN_PRICE_CENTS = 1       # Floor
MAX_PRICE_CENTS = 99      # Ceiling
TAU_WITHDRAW = 0.01       # Withdraw quotes below this tau
POLL_INTERVAL_SEC = 30    # Seconds between quote refreshes
PRICE_HISTORY_LEN = 50    # Number of recent mid prices for sigma estimation
MIN_VOLUME_FOR_MM = 100   # Minimum 24h volume to consider a market
MIN_SPREAD_FOR_MM = 3     # Minimum existing spread (cents) to make MM viable
MAX_SPREAD_FOR_MM = 30    # Maximum spread — too wide means no liquidity


# ---------------------------------------------------------------------------
# Sigma estimation
# ---------------------------------------------------------------------------
def estimate_sigma(price_history: list[float], min_obs: int = 5) -> float:
    """
    Estimate volatility in logit space from recent mid prices (in cents).

    Transforms prices to logit space: log(p / (1 - p)) where p = price/100,
    then computes the standard deviation of differences.

    Returns 0.0 if insufficient data, which signals "don't quote."
    """
    if len(price_history) < min_obs:
        return 0.0

    logits = []
    for price_cents in price_history:
        p = max(0.01, min(0.99, price_cents / 100.0))
        logits.append(math.log(p / (1.0 - p)))

    if len(logits) < 2:
        return 0.0

    # Standard deviation of logit differences (returns)
    diffs = [logits[i] - logits[i - 1] for i in range(1, len(logits))]
    n = len(diffs)
    mean = sum(diffs) / n
    variance = sum((d - mean) ** 2 for d in diffs) / max(n - 1, 1)
    sigma = math.sqrt(variance)

    # Floor: never let sigma go below a small positive value
    # (prevents zero spread which would guarantee losses on fees)
    return max(sigma, 0.05)


# ---------------------------------------------------------------------------
# Avellaneda-Stoikov core
# ---------------------------------------------------------------------------
def calc_reservation_price(
    mid: float,
    position: int,
    gamma: float,
    sigma: float,
    tau: float,
) -> float:
    """
    Avellaneda-Stoikov reservation price (in cents).

    r = mid - q * gamma * sigma^2 * tau

    Positive position (long YES) pushes reservation DOWN (willing to sell cheaper).
    Negative position (short YES / long NO) pushes reservation UP.
    """
    return mid - position * gamma * (sigma ** 2) * tau


def calc_optimal_spread(
    gamma: float,
    sigma: float,
    tau: float,
    kappa: float,
) -> float:
    """
    Avellaneda-Stoikov optimal spread (in cents).

    delta = gamma * sigma^2 * tau + (2/gamma) * ln(1 + gamma/kappa)

    The first term captures volatility risk.
    The second term captures adverse selection from informed traders.
    """
    volatility_component = gamma * (sigma ** 2) * tau
    adverse_selection = (2.0 / gamma) * math.log(1.0 + gamma / kappa)
    return volatility_component + adverse_selection


def generate_quotes(
    mid: float,
    position: int,
    gamma: float,
    sigma: float,
    tau: float,
    kappa: float,
    max_position: int = MAX_POSITION,
) -> dict:
    """
    Generate bid/ask quotes using A-S model.

    Returns:
        {
            "bid_price": int (cents),
            "ask_price": int (cents),
            "bid_size": int (contracts),
            "ask_size": int (contracts),
            "reservation": float,
            "spread": float,
            "skipped": bool,
            "skip_reason": str,
        }
    """
    result = {
        "bid_price": 0,
        "ask_price": 0,
        "bid_size": 0,
        "ask_size": 0,
        "reservation": mid,
        "spread": 0.0,
        "skipped": False,
        "skip_reason": "",
    }

    # Safety: don't quote near expiry
    if tau < TAU_WITHDRAW:
        result["skipped"] = True
        result["skip_reason"] = f"tau={tau:.4f} < {TAU_WITHDRAW} (near expiry)"
        return result

    # Safety: need valid sigma
    if sigma <= 0:
        result["skipped"] = True
        result["skip_reason"] = "sigma=0 (insufficient price history)"
        return result

    reservation = calc_reservation_price(mid, position, gamma, sigma, tau)
    delta = calc_optimal_spread(gamma, sigma, tau, kappa)

    # Clamp spread to [MIN_SPREAD, MAX_SPREAD]
    delta = max(delta, MIN_SPREAD_CENTS)
    delta = min(delta, MAX_SPREAD_CENTS)

    bid_raw = reservation - delta / 2.0
    ask_raw = reservation + delta / 2.0

    # Clamp to valid price range [1, 99]
    bid_price = max(MIN_PRICE_CENTS, min(MAX_PRICE_CENTS, int(math.floor(bid_raw))))
    ask_price = max(MIN_PRICE_CENTS, min(MAX_PRICE_CENTS, int(math.ceil(ask_raw))))

    # Ensure bid < ask (at least 1c apart)
    if bid_price >= ask_price:
        # Center around reservation and force minimum gap
        center = int(round(reservation))
        bid_price = max(MIN_PRICE_CENTS, center - 1)
        ask_price = min(MAX_PRICE_CENTS, center + 1)
        if bid_price >= ask_price:
            result["skipped"] = True
            result["skip_reason"] = "price at boundary, cannot create valid spread"
            return result

    # Inventory-aware sizing:
    # Reduce bid size when long (don't want more), reduce ask size when short
    base_size = 5
    if position >= 0:
        bid_size = max(1, base_size - position)     # Less buying when long
        ask_size = min(base_size + position, max_position - position) if position < max_position else base_size
    else:
        abs_pos = abs(position)
        bid_size = min(base_size + abs_pos, max_position - abs_pos) if abs_pos < max_position else base_size
        ask_size = max(1, base_size - abs_pos)      # Less selling when short

    # Hard cap sizes at remaining position room
    bid_size = max(0, min(bid_size, max_position - position))
    ask_size = max(0, min(ask_size, max_position + position))

    # If at max position on one side, only quote the other
    if bid_size <= 0 and ask_size <= 0:
        result["skipped"] = True
        result["skip_reason"] = f"at max position ({position}), cannot quote"
        return result

    result["bid_price"] = bid_price
    result["ask_price"] = ask_price
    result["bid_size"] = max(bid_size, 0)
    result["ask_size"] = max(ask_size, 0)
    result["reservation"] = reservation
    result["spread"] = delta

    return result


# ---------------------------------------------------------------------------
# Market scanning
# ---------------------------------------------------------------------------
def scan_mm_opportunities(client: KalshiClient) -> list[dict]:
    """
    Scan open markets for viable market-making opportunities.

    Criteria:
        - Sufficient 24h volume (>= MIN_VOLUME_FOR_MM)
        - Spread between MIN_SPREAD_FOR_MM and MAX_SPREAD_FOR_MM cents
        - Not too close to expiry (>1 hour)
        - Valid orderbook with bids and asks on both sides

    Returns list of dicts with market info + orderbook snapshot.
    """
    opportunities = []
    cursor = None

    for page in range(5):  # Max 5 pages
        try:
            resp = client.get_markets(limit=100, cursor=cursor, status="open")
        except Exception as e:
            logger.error(f"Failed to fetch markets page {page}: {e}")
            break

        for m in resp.get("markets", []):
            opp = _evaluate_mm_candidate(client, m)
            if opp:
                opportunities.append(opp)

        cursor = resp.get("cursor")
        if not cursor:
            break
        time.sleep(1.0)  # Rate limiting

    # Sort by volume descending (most liquid first)
    opportunities.sort(key=lambda x: x["volume"], reverse=True)
    logger.info(f"Found {len(opportunities)} viable MM opportunities")
    return opportunities


def _evaluate_mm_candidate(client: KalshiClient, market: dict) -> Optional[dict]:
    """Evaluate a single market for MM viability. Returns dict or None."""
    ticker = market.get("ticker", "")
    if not ticker:
        return None

    # Parse close time
    close_time_str = market.get("close_time", "")
    if not close_time_str:
        return None
    try:
        close_time = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None

    now = datetime.now(timezone.utc)
    hours_left = (close_time - now).total_seconds() / 3600.0
    if hours_left < 1.0:
        return None  # Too close to expiry

    # Volume check
    vol_raw = market.get("volume_fp") or market.get("volume_24h_fp") or market.get("volume") or 0
    try:
        volume = int(float(vol_raw))
    except (ValueError, TypeError):
        volume = 0
    if volume < MIN_VOLUME_FOR_MM:
        return None

    # Get best bid/ask from market-level data
    def to_cents(val):
        if val is None:
            return 0
        try:
            return int(float(val) * 100)
        except (ValueError, TypeError):
            return 0

    yes_bid = to_cents(market.get("yes_bid_dollars") or market.get("yes_bid") or 0)
    yes_ask = to_cents(market.get("yes_ask_dollars") or market.get("yes_ask") or 0)

    if yes_bid <= 0 or yes_ask <= 0:
        return None

    spread = yes_ask - yes_bid
    if spread < MIN_SPREAD_FOR_MM or spread > MAX_SPREAD_FOR_MM:
        return None

    mid = (yes_bid + yes_ask) / 2.0

    # Total time from now to close (for tau calculation later)
    total_seconds = (close_time - now).total_seconds()

    return {
        "ticker": ticker,
        "title": market.get("title", "")[:60],
        "series": market.get("series_ticker", ""),
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "mid": mid,
        "spread": spread,
        "volume": volume,
        "hours_left": round(hours_left, 1),
        "close_time": close_time,
        "total_seconds": total_seconds,
    }


# ---------------------------------------------------------------------------
# Price history tracking (in-memory, per-ticker)
# ---------------------------------------------------------------------------
class PriceTracker:
    """Maintains rolling price history for sigma estimation."""

    def __init__(self, max_len: int = PRICE_HISTORY_LEN):
        self._history: dict[str, list[float]] = {}
        self._max_len = max_len

    def add(self, ticker: str, mid_cents: float):
        if ticker not in self._history:
            self._history[ticker] = []
        self._history[ticker].append(mid_cents)
        if len(self._history[ticker]) > self._max_len:
            self._history[ticker] = self._history[ticker][-self._max_len:]

    def get(self, ticker: str) -> list[float]:
        return self._history.get(ticker, [])


# ---------------------------------------------------------------------------
# Order management
# ---------------------------------------------------------------------------
def _cancel_existing_orders(client: KalshiClient, ticker: str) -> int:
    """Cancel all open orders for a ticker. Returns count cancelled."""
    cancelled = 0
    try:
        orders = client.get_orders(ticker=ticker, status="resting")
        for order in orders.get("orders", []):
            try:
                client.cancel_order(order["order_id"])
                cancelled += 1
            except Exception as e:
                logger.warning(f"Failed to cancel order {order.get('order_id')}: {e}")
    except Exception as e:
        logger.warning(f"Failed to fetch orders for {ticker}: {e}")
    return cancelled


def _get_position(client: KalshiClient, ticker: str) -> int:
    """Get net YES position for a ticker. Positive = long YES, negative = short."""
    try:
        positions = client.get_positions()
        for pos in positions.get("market_positions", []):
            if pos.get("ticker") == ticker:
                yes_count = int(pos.get("position", 0))
                return yes_count
        return 0
    except Exception:
        return 0


def _place_quote(
    client: KalshiClient,
    ticker: str,
    side: str,
    price_cents: int,
    size: int,
    dry_run: bool,
) -> Optional[dict]:
    """Place a single maker order. Returns order response or None."""
    if size <= 0 or price_cents < MIN_PRICE_CENTS or price_cents > MAX_PRICE_CENTS:
        return None

    if dry_run:
        logger.info(
            f"  [DRY RUN] {side.upper()} {size}x @ {price_cents}c on {ticker}"
        )
        return {"dry_run": True, "side": side, "price": price_cents, "size": size}

    try:
        if side == "bid":
            # Bid = buy YES at this price
            resp = client.place_order(
                ticker=ticker,
                side="yes",
                action="buy",
                count=size,
                type="limit",
                yes_price=price_cents,
                post_only=True,
            )
        else:
            # Ask = sell YES at this price (equivalent to buying NO at 100-price)
            resp = client.place_order(
                ticker=ticker,
                side="no",
                action="buy",
                count=size,
                type="limit",
                no_price=100 - price_cents,
                post_only=True,
            )
        logger.info(
            f"  PLACED {side.upper()} {size}x @ {price_cents}c on {ticker} "
            f"-> order_id={resp.get('order', {}).get('order_id', 'N/A')}"
        )
        return resp
    except Exception as e:
        logger.error(f"  Failed to place {side} on {ticker}: {e}")
        return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_market_maker(
    client: KalshiClient,
    tickers: list[str],
    dry_run: bool = True,
    gamma: float = DEFAULT_GAMMA,
    kappa: float = DEFAULT_KAPPA,
    max_cycles: int = 0,
):
    """
    Main market-making loop.

    For each ticker:
        1. Fetch orderbook -> compute mid
        2. Get current position
        3. Estimate sigma from price history
        4. Compute A-S quotes
        5. Cancel old orders, place new quotes
        6. Sleep and repeat

    Args:
        client: Authenticated KalshiClient
        tickers: List of market tickers to make on
        dry_run: If True, log actions but don't place real orders
        gamma: Risk aversion parameter
        kappa: Order arrival sensitivity
        max_cycles: Stop after N cycles (0 = run forever)
    """
    mode = "DRY RUN" if dry_run else "LIVE"
    logger.info(f"Starting Avellaneda-Stoikov market maker [{mode}]")
    logger.info(f"  Tickers: {tickers}")
    logger.info(f"  gamma={gamma}, kappa={kappa}, max_position={MAX_POSITION}")
    logger.info(f"  poll_interval={POLL_INTERVAL_SEC}s, spread_bounds=[{MIN_SPREAD_CENTS}, {MAX_SPREAD_CENTS}]c")

    tracker = PriceTracker()
    cycle = 0

    try:
        while True:
            cycle += 1
            if max_cycles > 0 and cycle > max_cycles:
                logger.info(f"Reached max cycles ({max_cycles}), stopping.")
                break

            logger.info(f"\n--- Cycle {cycle} @ {datetime.now(timezone.utc).strftime('%H:%M:%S')} ---")

            for ticker in tickers:
                try:
                    _process_ticker(
                        client, ticker, tracker, gamma, kappa, dry_run
                    )
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {e}")

            logger.info(f"Sleeping {POLL_INTERVAL_SEC}s...")
            time.sleep(POLL_INTERVAL_SEC)

    except KeyboardInterrupt:
        logger.info("Shutting down market maker (KeyboardInterrupt).")
        if not dry_run:
            logger.info("Cancelling all open orders...")
            for ticker in tickers:
                n = _cancel_existing_orders(client, ticker)
                logger.info(f"  Cancelled {n} orders on {ticker}")


def _process_ticker(
    client: KalshiClient,
    ticker: str,
    tracker: PriceTracker,
    gamma: float,
    kappa: float,
    dry_run: bool,
):
    """Process a single ticker: fetch data, compute quotes, place orders."""

    # 1. Fetch orderbook
    try:
        ob = client.get_market_orderbook(ticker, depth=5)
    except Exception as e:
        logger.warning(f"  {ticker}: orderbook fetch failed: {e}")
        return

    book = ob.get("orderbook", {})
    yes_levels = book.get("yes", [])  # [[price_cents, qty], ...]
    no_levels = book.get("no", [])

    if not yes_levels and not no_levels:
        logger.debug(f"  {ticker}: empty orderbook, skipping")
        return

    # Derive best bid/ask from orderbook
    # YES side: these are resting YES orders (asks from our perspective as buyer)
    # NO side: these are resting NO orders; best NO ask at X implies YES bid at 100-X
    best_yes_ask = min(yes_levels, key=lambda x: x[0])[0] if yes_levels else None
    best_no_ask = min(no_levels, key=lambda x: x[0])[0] if no_levels else None

    # YES bid = 100 - best NO ask
    best_yes_bid = (100 - best_no_ask) if best_no_ask else None

    if best_yes_bid is None or best_yes_ask is None:
        logger.debug(f"  {ticker}: incomplete book (bid={best_yes_bid}, ask={best_yes_ask})")
        return

    if best_yes_bid >= best_yes_ask:
        # Crossed book — unusual, skip
        logger.warning(f"  {ticker}: crossed book bid={best_yes_bid} >= ask={best_yes_ask}")
        return

    mid = (best_yes_bid + best_yes_ask) / 2.0
    tracker.add(ticker, mid)

    # 2. Get market info for tau
    try:
        market_info = client.get_market(ticker)
        market_data = market_info.get("market", market_info)
        close_time_str = market_data.get("close_time", "")
        close_time = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        total_seconds = (close_time - now).total_seconds()
        if total_seconds <= 0:
            logger.info(f"  {ticker}: market expired, skipping")
            return
        # tau: normalize to [0, 1] using 24 hours as the reference period
        # (most Kalshi markets are daily)
        tau = min(total_seconds / 86400.0, 1.0)
    except Exception as e:
        logger.warning(f"  {ticker}: could not determine tau: {e}")
        tau = 0.5  # Default: assume mid-life

    # 3. Get position
    position = _get_position(client, ticker)

    # 4. Estimate sigma
    history = tracker.get(ticker)
    sigma = estimate_sigma(history)
    if sigma <= 0:
        # Not enough history yet — use a conservative default
        # based on the current spread as a proxy
        spread_now = best_yes_ask - best_yes_bid
        sigma = max(spread_now / 10.0, 0.1)
        logger.debug(f"  {ticker}: using fallback sigma={sigma:.3f}")

    # 5. Generate quotes
    quotes = generate_quotes(mid, position, gamma, sigma, tau, kappa, MAX_POSITION)

    logger.info(
        f"  {ticker}: mid={mid:.1f}c bid/ask={best_yes_bid}/{best_yes_ask} "
        f"pos={position} sigma={sigma:.3f} tau={tau:.3f} "
        f"r={quotes['reservation']:.1f} spread={quotes['spread']:.1f}"
    )

    if quotes["skipped"]:
        logger.info(f"  {ticker}: SKIP — {quotes['skip_reason']}")
        # Cancel existing orders if we're withdrawing
        if not dry_run:
            _cancel_existing_orders(client, ticker)
        return

    bid_p = quotes["bid_price"]
    ask_p = quotes["ask_price"]
    bid_s = quotes["bid_size"]
    ask_s = quotes["ask_size"]

    logger.info(
        f"  {ticker}: QUOTE bid={bid_p}c x{bid_s} / ask={ask_p}c x{ask_s}"
    )

    # 6. Cancel and replace
    if not dry_run:
        _cancel_existing_orders(client, ticker)
        time.sleep(0.5)  # Brief pause after cancels

    if bid_s > 0:
        _place_quote(client, ticker, "bid", bid_p, bid_s, dry_run)
    if ask_s > 0:
        _place_quote(client, ticker, "ask", ask_p, ask_s, dry_run)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Avellaneda-Stoikov market maker for Kalshi"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Place real orders (default: dry run)"
    )
    parser.add_argument(
        "--tickers", nargs="+", default=[],
        help="Specific tickers to market-make on"
    )
    parser.add_argument(
        "--scan", action="store_true",
        help="Scan for opportunities and auto-select tickers"
    )
    parser.add_argument(
        "--gamma", type=float, default=DEFAULT_GAMMA,
        help=f"Risk aversion (default: {DEFAULT_GAMMA})"
    )
    parser.add_argument(
        "--kappa", type=float, default=DEFAULT_KAPPA,
        help=f"Order arrival sensitivity (default: {DEFAULT_KAPPA})"
    )
    parser.add_argument(
        "--max-cycles", type=int, default=0,
        help="Stop after N cycles (default: run forever)"
    )
    args = parser.parse_args()

    dry_run = not args.live
    client = KalshiClient()

    # Verify connection
    try:
        balance = client.get_balance()
        bal_cents = balance.get("balance", 0)
        logger.info(f"Connected to Kalshi ({config.KALSHI_ENV}). Balance: ${bal_cents / 100:.2f}")
    except Exception as e:
        logger.error(f"Failed to connect to Kalshi: {e}")
        return

    tickers = args.tickers
    if not tickers or args.scan:
        logger.info("Scanning for MM opportunities...")
        opportunities = scan_mm_opportunities(client)
        if not opportunities:
            logger.warning("No viable MM opportunities found.")
            return

        # Display opportunities
        logger.info(f"\n{'Ticker':<35} {'Mid':>5} {'Sprd':>5} {'Vol':>8} {'Hrs':>6}")
        logger.info("-" * 65)
        for opp in opportunities[:10]:
            logger.info(
                f"{opp['ticker']:<35} {opp['mid']:>5.1f} {opp['spread']:>5d} "
                f"{opp['volume']:>8,d} {opp['hours_left']:>6.1f}"
            )

        if not tickers:
            # Auto-select top 3 by volume
            tickers = [opp["ticker"] for opp in opportunities[:3]]
            logger.info(f"\nAuto-selected tickers: {tickers}")

    if not tickers:
        logger.warning("No tickers to market-make on.")
        return

    run_market_maker(
        client,
        tickers,
        dry_run=dry_run,
        gamma=args.gamma,
        kappa=args.kappa,
        max_cycles=args.max_cycles,
    )


if __name__ == "__main__":
    main()
