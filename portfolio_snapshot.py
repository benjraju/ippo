#!/usr/bin/env python3
"""
portfolio_snapshot.py — Quick mark-to-market account valuation.
Starting capital: $98
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from kalshi_client import KalshiClient

STARTING_CAPITAL = 98.00

def main():
    client = KalshiClient()

    # 1. Cash balance
    balance_resp = client.get_balance()
    balance_cents = balance_resp.get("balance", 0)
    cash = balance_cents / 100.0

    # 2. Positions
    pos_resp = client.get_positions()
    market_positions = pos_resp.get("market_positions", [])

    # Filter to positions with non-zero holdings
    open_positions = [p for p in market_positions if float(p.get("position_fp", 0)) != 0]

    print(f"\n{'='*70}")
    print(f"  IPPO PORTFOLIO SNAPSHOT  (starting capital: ${STARTING_CAPITAL:.2f})")
    print(f"{'='*70}")
    print(f"  Available cash:      ${cash:.2f}")
    print(f"  Open positions:      {len(open_positions)}")
    print()

    if not open_positions:
        total_value = cash
        pnl = total_value - STARTING_CAPITAL
        pnl_pct = (pnl / STARTING_CAPITAL) * 100
        print(f"  No open positions detected.")
        print()
        print(f"{'='*70}")
        print(f"  TOTAL ACCOUNT VALUE: ${total_value:.2f}")
        print(f"  OVERALL P&L:         ${pnl:+.2f}  ({pnl_pct:+.1f}%)")
        print(f"{'='*70}\n")
        return

    # 3. Print each open position with current market value
    print(f"  {'TICKER':<45} {'CONTRACTS':>10} {'COST $':>8} {'MKT VAL':>8} {'UNREAL':>8}")
    print(f"  {'-'*45} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

    total_cost = 0.0
    total_mtm = 0.0
    total_unrealized = 0.0

    for p in sorted(open_positions, key=lambda x: x.get("ticker", "")):
        ticker = p.get("ticker", "")
        contracts = float(p.get("position_fp", 0))
        # market_exposure_dollars = current mark-to-market value of the position
        mtm = float(p.get("market_exposure_dollars", 0))
        cost = float(p.get("total_traded_dollars", 0))
        realized = float(p.get("realized_pnl_dollars", 0))
        unrealized = mtm - cost

        total_cost += cost
        total_mtm += mtm
        total_unrealized += unrealized

        ticker_short = ticker[:45]
        side_sign = "+" if contracts > 0 else "-"
        print(f"  {ticker_short:<45} {side_sign}{abs(contracts):>9.0f} ${cost:>7.2f} ${mtm:>7.2f} ${unrealized:>+7.2f}")

    print(f"  {'─'*90}")
    print(f"  {'TOTALS':<45} {len(open_positions):>10}  ${total_cost:>7.2f} ${total_mtm:>7.2f} ${total_unrealized:>+7.2f}")
    print()

    total_value = cash + total_mtm
    pnl = total_value - STARTING_CAPITAL
    pnl_pct = (pnl / STARTING_CAPITAL) * 100

    print(f"{'='*70}")
    print(f"  Cash balance:           ${cash:.2f}")
    print(f"  Open positions (MTM):   ${total_mtm:.2f}  (cost basis: ${total_cost:.2f})")
    print(f"  Unrealized P&L:         ${total_unrealized:+.2f}")
    print(f"  ─────────────────────────────────")
    print(f"  TOTAL ACCOUNT VALUE:    ${total_value:.2f}")
    print(f"  Starting capital:       ${STARTING_CAPITAL:.2f}")
    print(f"  OVERALL P&L:            ${pnl:+.2f}  ({pnl_pct:+.1f}%)")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
