"""
Backtest NBA hedging strategies against 2,414 real game winner markets.

Strategy 1: PURE ARB — Buy NO on both teams when sum < $1
Strategy 2: LEVERAGED UNDERDOG — YES underdog + NO favorite (double exposure)
Strategy 3: HEDGED UNDERDOG — YES underdog + partial NO favorite (reduce downside)
Strategy 4: RATIO HEDGE — Tune contracts per side for optimal risk/reward
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

SETTLEMENTS = Path(__file__).parent.parent / "output" / "historical_settlements_with_prices.json"
MAKER_FEE = 0.0175

def maker_fee(price_cents):
    p = price_cents / 100
    return MAKER_FEE * p * (1 - p) * 100  # in cents

def load_nba_games():
    """Load NBA games, group by event (both teams' markets together)."""
    with open(SETTLEMENTS) as f:
        data = json.load(f)
    
    markets = data["markets"]
    
    # Group by game (everything before last dash)
    games = defaultdict(list)
    for m in markets:
        ticker = m.get("ticker", "")
        if not ticker.startswith("KXNBAGAME"):
            continue
        if not m.get("result"):
            continue
        
        game_id = ticker.rsplit("-", 1)[0]
        team = ticker.rsplit("-", 1)[1]
        
        yes_price = float(m.get("previous_price", 0) or 0)
        yes_cents = yes_price * 100 if yes_price < 1.5 else yes_price
        ask_cents = float(m.get("prev_yes_ask", 0) or 0)
        ask_cents = ask_cents * 100 if ask_cents < 1.5 else ask_cents
        bid_cents = float(m.get("prev_yes_bid", 0) or 0)
        bid_cents = bid_cents * 100 if bid_cents < 1.5 else bid_cents
        
        settled_yes = m["result"] == "yes"
        volume = float(m.get("volume", 0) or 0)
        close_time = m.get("close_time", "")
        
        games[game_id].append({
            "ticker": ticker, "team": team,
            "yes_cents": yes_cents, "ask_cents": ask_cents, "bid_cents": bid_cents,
            "settled_yes": settled_yes, "volume": volume, "close_time": close_time,
        })
    
    # Only keep games with exactly 2 teams
    complete = {k: v for k, v in games.items() if len(v) == 2}
    
    # Sort by close_time
    sorted_games = sorted(complete.items(), key=lambda x: x[1][0].get("close_time", ""))
    return sorted_games


def run_strategy(games, strategy_fn, label):
    """Run a strategy function against all games, with train/test split."""
    # 70/30 chronological split
    split = int(len(games) * 0.7)
    train_games = games[:split]
    test_games = games[split:]
    
    train_trades = []
    test_trades = []
    
    for game_id, legs in train_games:
        trades = strategy_fn(game_id, legs)
        train_trades.extend(trades)
    
    for game_id, legs in test_games:
        trades = strategy_fn(game_id, legs)
        test_trades.extend(trades)
    
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    
    for period, trades in [("TRAIN", train_trades), ("TEST", test_trades)]:
        if not trades:
            print(f"  {period}: 0 trades")
            continue
        
        pnls = [t["pnl"] for t in trades]
        total = sum(pnls)
        wins = sum(1 for p in pnls if p > 0)
        avg = np.mean(pnls)
        std = np.std(pnls, ddof=1) if len(pnls) > 1 else 1
        z = avg / (std / math.sqrt(len(pnls))) if std > 0 else 0
        
        # Consistency: what % of games are profitable?
        game_pnls = defaultdict(float)
        for t in trades:
            game_pnls[t["game_id"]] += t["pnl"]
        games_won = sum(1 for p in game_pnls.values() if p > 0)
        
        # Max drawdown
        cumsum = np.cumsum(pnls)
        max_dd = np.max(np.maximum.accumulate(cumsum) - cumsum) if len(cumsum) > 0 else 0
        
        print(f"  {period}: {len(trades)} trades across {len(game_pnls)} games")
        print(f"    P&L: ${total/100:.2f} | Win rate: {wins/len(trades)*100:.0f}% | Game win: {games_won/len(game_pnls)*100:.0f}%")
        print(f"    Avg: {avg:.1f}c/trade | z-score: {z:.2f} | Max DD: {max_dd:.0f}c")
        
        # Show P&L distribution
        big_wins = sum(1 for p in pnls if p > 50)
        big_losses = sum(1 for p in pnls if p < -50)
        small = sum(1 for p in pnls if -10 <= p <= 10)
        print(f"    Distribution: {big_wins} big wins (>50c), {big_losses} big losses, {small} small (<10c)")


# ============================================================================
# STRATEGIES
# ============================================================================

def strategy_naked_underdog(game_id, legs):
    """Current strategy: buy YES on underdog at 18-30c."""
    trades = []
    for leg in legs:
        if 18 <= leg["yes_cents"] <= 30 and leg["volume"] >= 50:
            entry = leg["ask_cents"] if leg["ask_cents"] > 0 else leg["yes_cents"]
            fee = maker_fee(entry)
            if leg["settled_yes"]:
                pnl = 100 - entry - fee
            else:
                pnl = -(entry + fee)
            trades.append({"game_id": game_id, "pnl": pnl, "type": "underdog_yes"})
    return trades


def strategy_pure_arb(game_id, legs):
    """Buy NO on both teams when sum(NO prices) < $1."""
    if len(legs) != 2:
        return []
    
    a, b = legs
    # NO price = 100 - YES price
    no_a = 100 - a["yes_cents"]
    no_b = 100 - b["yes_cents"]
    total_no = no_a + no_b
    
    fee_a = maker_fee(no_a)
    fee_b = maker_fee(no_b)
    total_cost = no_a + fee_a + no_b + fee_b
    
    # One NO always wins ($1 payout), one always loses
    if total_cost < 100:  # guaranteed profit after fees
        profit = 100 - total_cost
        return [{"game_id": game_id, "pnl": profit, "type": "arb"}]
    return []


def strategy_leveraged_underdog(game_id, legs):
    """YES underdog + NO favorite = double exposure on underdog winning."""
    if len(legs) != 2:
        return []
    
    # Find underdog (lower YES price)
    a, b = sorted(legs, key=lambda x: x["yes_cents"])
    underdog, favorite = a, b
    
    if not (18 <= underdog["yes_cents"] <= 30) or underdog["volume"] < 50:
        return []
    
    # Buy YES underdog + NO favorite
    yes_entry = underdog["ask_cents"] if underdog["ask_cents"] > 0 else underdog["yes_cents"]
    no_entry = 100 - favorite["yes_cents"]
    
    yes_fee = maker_fee(yes_entry)
    no_fee = maker_fee(no_entry)
    
    total_cost = yes_entry + yes_fee + no_entry + no_fee
    
    if underdog["settled_yes"]:
        # Underdog wins: both pay $1 each
        pnl = 200 - total_cost
    else:
        # Favorite wins: both worthless
        pnl = -total_cost
    
    return [{"game_id": game_id, "pnl": pnl, "type": "leveraged"}]


def strategy_hedged_underdog(game_id, legs):
    """YES underdog (3x) + YES favorite (1x) = reduced downside."""
    if len(legs) != 2:
        return []
    
    a, b = sorted(legs, key=lambda x: x["yes_cents"])
    underdog, favorite = a, b
    
    if not (18 <= underdog["yes_cents"] <= 30) or underdog["volume"] < 50:
        return []
    
    # Buy 3x YES underdog + 1x YES favorite (hedge)
    ud_entry = underdog["ask_cents"] if underdog["ask_cents"] > 0 else underdog["yes_cents"]
    fav_entry = favorite["ask_cents"] if favorite["ask_cents"] > 0 else favorite["yes_cents"]
    
    ud_fee = maker_fee(ud_entry)
    fav_fee = maker_fee(fav_entry)
    
    ud_contracts = 3
    fav_contracts = 1
    
    total_cost = (ud_entry + ud_fee) * ud_contracts + (fav_entry + fav_fee) * fav_contracts
    
    if underdog["settled_yes"]:
        # Underdog wins: 3x $1 from underdog YES, fav YES worthless
        pnl = (100 * ud_contracts) - total_cost
    else:
        # Favorite wins: underdog YES worthless, 1x $1 from favorite YES
        pnl = (100 * fav_contracts) - total_cost
    
    return [{"game_id": game_id, "pnl": pnl, "type": "hedged"}]


def strategy_hedged_no(game_id, legs):
    """NO favorite (3x) + NO underdog (1x) = consistent returns."""
    if len(legs) != 2:
        return []
    
    a, b = sorted(legs, key=lambda x: x["yes_cents"])
    underdog, favorite = a, b
    
    # Only trade when favorite is 60-85c (moderate to strong favorite)
    if not (60 <= favorite["yes_cents"] <= 85) or favorite["volume"] < 50:
        return []
    
    # Buy 3x NO favorite (cheap, pays when underdog wins) + 1x NO underdog (expensive, pays when fav wins)
    no_fav = 100 - favorite["yes_cents"]  # cheap NO (15-40c)
    no_ud = 100 - underdog["yes_cents"]   # expensive NO (70-82c)
    
    no_fav_fee = maker_fee(no_fav)
    no_ud_fee = maker_fee(no_ud)
    
    fav_no_contracts = 3
    ud_no_contracts = 1
    
    total_cost = (no_fav + no_fav_fee) * fav_no_contracts + (no_ud + no_ud_fee) * ud_no_contracts
    
    if underdog["settled_yes"]:
        # Underdog wins: fav NO pays (3x $1), ud NO worthless
        pnl = (100 * fav_no_contracts) - total_cost
    else:
        # Favorite wins: fav NO worthless, ud NO pays (1x $1)
        pnl = (100 * ud_no_contracts) - total_cost
    
    return [{"game_id": game_id, "pnl": pnl, "type": "hedged_no"}]


def strategy_ratio_search(game_id, legs):
    """Dynamic ratio: size based on price to equalize outcomes."""
    if len(legs) != 2:
        return []
    
    a, b = sorted(legs, key=lambda x: x["yes_cents"])
    underdog, favorite = a, b
    
    if not (15 <= underdog["yes_cents"] <= 35) or underdog["volume"] < 50:
        return []
    
    # Buy YES underdog and YES favorite in a ratio that minimizes worst-case
    # Key insight: size inversely to price for balanced risk
    ud_price = underdog["ask_cents"] if underdog["ask_cents"] > 0 else underdog["yes_cents"]
    fav_price = favorite["ask_cents"] if favorite["ask_cents"] > 0 else favorite["yes_cents"]
    
    # Target: equal dollar payout in both scenarios
    # If underdog wins: ud_contracts * $1 should be similar to
    # If favorite wins: fav_contracts * $1
    # So: ud_contracts ≈ fav_contracts
    # But cost differs: ud cheap, fav expensive
    # Budget: $2 max total
    budget_cents = 200
    
    # Allocate 70% to underdog, 30% to favorite
    ud_budget = budget_cents * 0.7
    fav_budget = budget_cents * 0.3
    
    ud_contracts = max(1, int(ud_budget / (ud_price + maker_fee(ud_price))))
    fav_contracts = max(1, int(fav_budget / (fav_price + maker_fee(fav_price))))
    
    total_cost = (ud_price + maker_fee(ud_price)) * ud_contracts + \
                 (fav_price + maker_fee(fav_price)) * fav_contracts
    
    if total_cost > budget_cents:
        return []  # over budget
    
    if underdog["settled_yes"]:
        pnl = (100 * ud_contracts) - total_cost
    else:
        pnl = (100 * fav_contracts) - total_cost
    
    return [{"game_id": game_id, "pnl": pnl, "type": "ratio"}]


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    games = load_nba_games()
    print(f"Loaded {len(games)} complete NBA games (2 teams each)")
    print(f"Date range: {games[0][1][0]['close_time'][:10]} to {games[-1][1][0]['close_time'][:10]}")
    
    run_strategy(games, strategy_naked_underdog, "STRATEGY 1: Naked Underdog YES (current)")
    run_strategy(games, strategy_pure_arb, "STRATEGY 2: Pure Arb (NO both sides)")
    run_strategy(games, strategy_leveraged_underdog, "STRATEGY 3: Leveraged Underdog (YES ud + NO fav)")
    run_strategy(games, strategy_hedged_underdog, "STRATEGY 4: Hedged Underdog (3x YES ud + 1x YES fav)")
    run_strategy(games, strategy_hedged_no, "STRATEGY 5: Hedged NO (3x NO fav + 1x NO ud)")
    run_strategy(games, strategy_ratio_search, "STRATEGY 6: Ratio Hedge (70/30 budget split)")
