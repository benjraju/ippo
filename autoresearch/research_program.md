# AutoResearch Program: Kalshi Prediction Market Strategy Optimization

## Research Goal
Maximize Sharpe ratio >1.2 and ROI on fast, high-volume Kalshi markets while keeping:
- Max drawdown <8%
- Position size ≤2% of $100 account
- Win rate >52%

## Target Markets
- **Weather**: KXHIGH/KXLOW daily temperature markets (settle every day)
- **Sports**: NCAA Men's College Basketball (high volume during season)
- **Crypto**: BTC/ETH hourly and daily range buckets

## Strategy File
`candidate_strategy.py` — the AI modifies this file each iteration.

## Experiment Parameters to Vary

### Position Sizing
- `KELLY_MULT`: Try 0.1, 0.15, 0.2, 0.25, 0.3, 0.35
- `MAX_POSITION_FRAC`: Try 0.01, 0.015, 0.02, 0.025, 0.03

### Edge Detection
- `MIN_EDGE`: Try 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10
- `MEAN_REVERT_STRENGTH`: Try 0.01, 0.02, 0.03, 0.04, 0.05
- `MEAN_REVERT_THRESHOLD_HIGH`: Try 0.80, 0.82, 0.85, 0.87, 0.90
- `MEAN_REVERT_THRESHOLD_LOW`: Try 0.10, 0.13, 0.15, 0.18, 0.20

### Market Filters
- `MIN_VOLUME`: Try 25, 50, 75, 100, 150
- `MAX_HOURS`: Try 12, 24, 36, 48, 72
- `MAX_SPREAD`: Try 8, 10, 12, 15, 20

### Signal Combinations
- Try adding momentum signal (price change over last N trades)
- Try adding orderbook imbalance signal
- Try time-of-day effect (markets may be less efficient at certain hours)
- Try different weighting of signals

## Scoring Function
```
score = (sharpe_ratio * 0.4) + (roi_pct * 0.3) + (win_rate * 0.2) - (max_drawdown * 0.1)
```
Higher is better. Negative score = reject iteration.

## Constraints (HARD LIMITS — do not violate)
- Max drawdown must stay below 8%
- Position size must stay at or below 2% of account
- Must have at least 50 trades in backtest for statistical significance
- Kelly multiplier must stay between 0.05 and 0.5

## Iteration Protocol
1. Read current `candidate_strategy.py`
2. Modify ONE parameter or function at a time
3. Run full backtest (200+ simulated trades)
4. Score the result
5. If score improved: `git commit` the change
6. If score worsened: `git revert` to previous version
7. Log results to `results.log`
8. Repeat

## Success Criteria
- Sharpe >1.2
- ROI >5% per backtest period
- Max drawdown <8%
- Win rate >52%
- Profit factor >1.3
