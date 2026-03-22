# Ippo Testing Plan - CTO Assessment

**Date**: 2026-03-22
**Perspective**: Quant fund CTO - brutal honesty, math rigor, real-money readiness

---

## 1. Honest Assessment: Where We Actually Stand

### The Good News
- The **weather strategy thesis is real**. NWS forecasts are free, accurate, updated hourly, and most Kalshi retail traders don't check them. This is a genuine information asymmetry.
- The **math is correct**. Normal CDF for bucket probabilities, Kelly criterion, Sortino ratio - all properly implemented.
- The **risk management is excellent**. Quarter Kelly + $2 cap + 8% daily loss limit means you can survive hundreds of bad trades while learning.
- The **AutoResearch architecture is clever**. Self-improvement via parameter evolution is the right approach. With the LLM-guided mutations we just added, it should converge faster.

### The Bad News (Must Fix)

**Problem 1: The simulation is circular.**

This is the biggest issue. Look at `research_loop.py` lines 227-239:

```python
# "Dumb retail" uses stdev * 0.7 (underestimates uncertainty)
dumb_prob = calc_bucket_probability(forecast_temp, bucket_low, bucket_high, true_stdev * 0.7)
# Market = 40% smart + 60% dumb
market_prob = 0.4 * smart_fair + 0.6 * dumb_prob
```

You're literally telling the simulation "retail traders are dumb in exactly this way" then optimizing to exploit that exact dumbness. **Of course it's profitable - you designed the opponent to lose.**

A real quant fund would say: "Prove to me that actual Kalshi market prices exhibit this specific bias. Show me the data."

**Problem 2: No real historical backtest.**

The system has never been tested against actual historical Kalshi prices matched to actual NWS forecasts matched to actual temperature outcomes. Everything we "know" about strategy performance comes from simulated data.

**Problem 3: No statistical significance testing.**

When AutoResearch finds that changing EDGE_THRESHOLD from 3.0 to 5.0 improves the score from 1.82 to 1.91, is that real? Or is it noise? With 200 simulated trades, the confidence interval on the score is wide. We need p-values.

**Problem 4: The `auto_trade.py` hardcodes parameters that should come from `candidate_strategy.py`.**

```python
WEATHER_EDGE_THRESHOLD_CENTS = 3.0  # hardcoded in auto_trade.py line 141
```

But `candidate_strategy.py` has `EDGE_THRESHOLD_CENTS = 3.0` which AutoResearch evolves. If AutoResearch changes it to 5.0, `auto_trade.py` still uses 3.0. **The brain and the body aren't connected properly.**

---

## 2. Math Review

### 2.1 Bucket Probability (CORRECT)

```
P(low <= temp < high) = normalCDF((high - forecast) / stdev) - normalCDF((low - forecast) / stdev)
```

This is textbook. The `math.erf` based CDF is accurate to ~15 decimal places.

### 2.2 Kelly Criterion (CORRECT but watch the edge cases)

```
Full Kelly: f* = (b*p - q) / b
Quarter Kelly: f*/4
```

One subtle issue: Kelly assumes you know the true probability `p`. You don't. You have an *estimate* of `p` from a model. Overconfident estimates lead to overbetting even at quarter Kelly. The right adjustment is:

```
Adjusted Kelly: f* = (b*p_adjusted - q) / b
where p_adjusted = p * confidence_shrinkage_factor
```

This isn't implemented. For now, quarter Kelly provides enough buffer, but as you scale up, add confidence shrinkage.

### 2.3 Sortino Ratio (CORRECT)

```
Sortino = mean(returns) / stdev(negative_returns) * sqrt(250)
```

Correctly uses only downside deviation. Good choice over Sharpe for asymmetric returns.

### 2.4 Scoring Composite (QUESTIONABLE)

```
score = Sortino*0.30 + ROI*0.002 + WinRate*1.5 + ProfitFactor*0.45 - MaxDrawdown*0.15
```

**Issues:**
- These weights are arbitrary. Different weights produce different "optimal" strategies.
- Mixing metrics with different scales (Sortino can be 0-10, ROI can be 0-500%) creates implicit weighting.
- Win rate is rewarded, but weather bucket strategies have inherently low win rates (~30-40%) with high payoffs. The scoring penalizes the very thing the strategy does.

**Recommendation:** Use Sortino ratio alone. It already captures risk-adjusted returns. If you must use a composite, weight it 80% Sortino + 20% drawdown penalty. Win rate and profit factor are just different views of the same thing.

### 2.5 Normal Distribution Assumption (APPROXIMATELY CORRECT, with caveats)

NWS forecast errors ARE approximately normal for 1-2 day horizons, but:

| Caveat | Impact | Mitigation |
|--------|--------|-----------|
| Errors are slightly skewed by season | Small | Could use skew-normal distribution |
| Stdev varies by weather pattern (convective days are wilder) | Medium | Use ensemble stdev instead of fixed stdev |
| Errors are correlated across nearby cities | Medium | Don't treat cities as independent bets |
| Occasional large busts (fat tails) | Small | Quarter Kelly absorbs this |
| Errors are non-stationary (stdev changes) | Medium | Use rolling stdev calibration |

The `auto_trade.py` already uses GFS ensemble stdev instead of fixed values. This is better than the `research_loop.py` simulation which uses fixed `REALITY_STDEV`. **The trading code is smarter than the research code.**

---

## 3. Strategy-by-Strategy Viability Assessment

### Weather Forecast Arb: VIABLE (7/10)

**Why it can work:**
- Real information asymmetry (NWS updates hourly, retail traders are slow)
- You're using 3 forecast sources (NWS + GFS ensemble + HRRR) which is more sophisticated than most participants
- Weather markets settle daily with clear outcomes - no ambiguity
- High volume markets (KXHIGHNY, KXHIGHCHI) provide liquidity

**Risks:**
- Other quants are doing this same strategy
- Edge may decay as markets get more efficient
- Need to measure actual edge decay over time

**Estimated realistic edge:** 2-4 cents per trade after fees

### Crypto Buckets: MARGINAL (4/10)

**Why it's harder:**
- Crypto markets are heavily traded by sophisticated participants
- Log-normal model is commodity knowledge - everyone uses it
- Your 30-day realized vol estimate is backward-looking; implied vol markets are forward-looking
- BTC can move 5-10% in hours, making bucket probabilities very unstable

**Estimated realistic edge:** 0-2 cents (likely noise)

### Sports (NBA): UNLIKELY (2/10)

**Why it probably doesn't work:**
- Sports betting markets are THE most efficient prediction markets
- ESPN public stats are already priced in
- You'd need proprietary models (player tracking, injury intelligence) to have edge
- The Kalshi sports markets are low volume, so even a real edge is hard to monetize

**Estimated realistic edge:** 0 cents (no edge with public data)

### Structural Arb (YES/NO): REAL BUT RARE (6/10)

**Why it works in theory:**
- If YES + NO < $1.00, buying both is risk-free profit
- This is pure arbitrage - no model needed

**Why it's hard in practice:**
- These opportunities last milliseconds on active markets
- 30-second scan cycle is WAY too slow
- After Kalshi fees (~7 cents round-trip), you need >7 cent mispricing
- With our new fee-adjusted threshold (93c), this is more realistic

**Estimated realistic edge:** 0-1 trade per day, ~$0.10-0.50 profit each

### Latency Arb: POSSIBLE BUT HARD (5/10)

**Needs sub-second execution which you don't have yet.**

### Copy-Trade: INTERESTING (5/10)

**Adverse selection risk:** By the time you detect and copy a whale's trade, the edge may already be gone. But if the whale has a long time horizon (days), there could be residual edge.

---

## 4. The Testing Protocol

### Phase 0: Calibration Test (Do This First, This Week)

**Question:** "Are NWS forecast errors actually normal with the stdev we assume?"

```
Test:
1. Pull 90 days of NWS forecast data for all 6 cities
2. Pull 90 days of actual high temps from NWS observations
3. Calculate forecast_error = actual - forecast for each day
4. Plot the distribution
5. Calculate actual stdev per city per days_out
6. Run Shapiro-Wilk normality test
7. Check for skewness and kurtosis

Pass criteria:
- Shapiro-Wilk p > 0.05 (errors are approximately normal)
- Actual stdev within 30% of our assumed FORECAST_STDEV values
- Skewness < 0.5 (not heavily skewed)
```

If this fails, our entire model is built on a wrong assumption.

### Phase 1: Edge Existence Test (Week 2-3)

**Question:** "Do actual Kalshi weather market prices have exploitable mispricing?"

```
Test:
1. For 30 days, at market open:
   a. Fetch NWS forecast for each city
   b. Calculate our model's fair value for each bucket
   c. Record the Kalshi market price for each bucket
   d. Record edge = fair_value - market_price
2. At settlement, record whether our "edge" actually predicted correctly
3. Calculate:
   - Mean edge by city and days_out
   - Edge accuracy: what % of predicted edges were real?
   - Edge vs market efficiency: do larger edges predict better?

Pass criteria:
- Mean edge accuracy > 55% (statistically better than random)
- t-test p-value < 0.05 on edge accuracy
- Positive expectancy AFTER fees on the identified edges
```

**This is the most important test.** If edges don't exist in real markets, nothing else matters.

### Phase 2: Paper Trading Validation (Week 3-6)

**Question:** "Does the full system (forecast + model + risk + execution) actually make money?"

```
Test:
1. Paper trade for 30 days using auto_trade.py --dry-run
2. Log every decision: what edge was detected, what would have been traded
3. At settlement, calculate what the P&L WOULD have been
4. Track:
   - Simulated win rate
   - Simulated Sortino
   - Simulated total P&L
   - Fill probability (would our limit orders have filled?)
   - Spread cost (how much did the bid-ask spread eat?)

Pass criteria:
- Win rate > 50% (weather bucket strategy may be lower with higher payoffs)
- Positive P&L after simulated fees
- Sortino > 0.5 (even modest risk-adjusted returns are promising)
- > 3 trades per day (enough flow to be worth running)
```

### Phase 3: Live Micro-Test (Week 6-8)

**Question:** "Does paper trading P&L match real trading P&L?"

```
Test:
1. Trade with real money at $1 per trade (half the normal limit)
2. Run for 2 weeks
3. Compare:
   - Paper P&L vs real P&L (should be within 20%)
   - Fill rate on limit orders
   - Slippage (difference between intended and actual fill price)
   - Any API issues or execution failures

Pass criteria:
- Real P&L within 30% of paper P&L
- Fill rate > 60% on limit orders
- No execution errors
- No unexpected losses
```

### Phase 4: AutoResearch Validation (Ongoing)

**Question:** "Does AutoResearch actually make the strategy better over time?"

```
Test:
1. Run performance_tracker.py weekly
2. Track: simulated score vs real Sortino
3. If reality gap > 50%: AutoResearch is overfitting, increase multi-seed count
4. If reality gap < 20%: AutoResearch is well-calibrated, trust its improvements

Dashboard metric: "Reality Correlation" (correlation between simulated and real scores)
Target: > 0.6
```

---

## 5. The Build-Measure-Learn Loop

```
WEEK 1: Calibration (Phase 0)
  - Validate the normal distribution assumption
  - Calculate real forecast error stdevs
  - Update candidate_strategy.py with real stdevs
  - Build the calibration test script

WEEK 2-3: Edge hunting (Phase 1)
  - Log edges daily without trading
  - After 14 days, run statistical analysis
  - Decision: does edge exist? How big?

WEEK 3-6: Paper trading (Phase 2)
  - Full system running in paper mode
  - Settlement tracking active
  - AutoResearch running with real data feedback

WEEK 6-8: Micro-live (Phase 3)
  - $1 per trade, weather only
  - Compare to paper performance
  - Identify execution issues

WEEK 8+: Scale or Pivot
  - IF weather strategy is profitable: increase to $2/trade
  - IF weather strategy is marginal: focus AutoResearch on it
  - IF weather strategy is unprofitable: pivot to arb-only
```

---

## 6. Critical Code Fixes for Testing

### Fix 1: Connect auto_trade.py to candidate_strategy.py

auto_trade.py hardcodes edge thresholds that AutoResearch should control. These need to read from the living strategy file.

### Fix 2: Build the calibration test script

A script that fetches 90 days of NWS forecasts vs actuals and validates our model assumptions.

### Fix 3: Build the edge logger

A script that runs daily, records model fair values vs Kalshi prices, and tracks edge accuracy over time without trading.

### Fix 4: Fix the simulation's "dumb retail" bias

The synthetic market generator should be calibrated against REAL Kalshi market behavior, not an assumed model of retail stupidity.

---

## 7. What a Quant Fund Would Say

> "You have a plausible thesis (weather forecast information asymmetry), correct math
> (normal CDF, Kelly, Sortino), and good risk management. But you've never tested it
> against real data. Your entire evidence base is 'I simulated a dumb opponent and
> beat it.' That proves nothing.
>
> Before committing capital, you need:
> 1. Proof that the edge exists in real markets (Phase 1)
> 2. Proof that your execution captures the edge (Phase 2)
> 3. Proof that AutoResearch improvements are real, not overfitting (Phase 4)
>
> The $100 account size is actually perfect for this stage - you can learn cheaply.
> But don't scale until you have 30+ days of profitable real trading data."
