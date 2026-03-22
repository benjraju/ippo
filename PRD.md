# Ippo Trading Bot - Product Requirements Document

**Version**: 1.0
**Date**: 2026-03-22
**Author**: Benjamin (owner) + Claude (technical architect)
**Status**: Living document - updated as system evolves

---

## 1. Vision

Build an autonomous, self-improving trading bot for Kalshi prediction markets that:
1. Pulls real-world data (weather forecasts, crypto prices, sports stats)
2. Compares model-derived probabilities against Kalshi market prices to find edges
3. Executes trades with strict risk limits ($1-2 per trade during strategy discovery)
4. Uses AutoResearch (Karpathy's pattern) to evolve its own strategy parameters overnight
5. Feeds real settlement outcomes back into the research loop to get smarter over time

**Goal**: Find a mathematically profitable strategy through automated experimentation, then scale it.

---

## 2. System Architecture

### 2.1 Three-Layer Design

```
LAYER 3: AUTORESEARCH (the brain)
  - Runs 100+ experiments per cycle
  - Mutates strategy parameters one at a time
  - Backtests each mutation against simulated + real market data
  - Keeps improvements (git commit), reverts regressions (git revert)
  - Output: an ever-improving candidate_strategy.py
           |
           v
LAYER 2: STRATEGY (the playbook)
  - candidate_strategy.py = the "living file" AutoResearch edits
  - ~25 tunable parameters (edge thresholds, position sizing, city weights, etc.)
  - Read by the trading layer every cycle
           |
           v
LAYER 1: TRADING (the executor)
  - Scans Kalshi markets every 2 hours
  - Runs 6 strategy modules (weather, crypto, sports, arb, latency arb, copy-trade)
  - Applies risk management (Kelly sizing, daily loss caps)
  - Places limit orders via Kalshi API
  - Logs everything for settlement tracking
```

### 2.2 Component Map

| File | Role | Layer |
|------|------|-------|
| `autoresearch/research_loop.py` | Self-improvement engine | 3 |
| `autoresearch/candidate_strategy.py` | Living strategy parameters | 2 |
| `auto_trade.py` | Autonomous daily trading | 1 |
| `weather_strategy.py` | Weather forecast edge detection | 1 |
| `crypto_strategy.py` | BTC/ETH/SOL bucket pricing | 1 |
| `sports_strategy.py` | NBA win probability model | 1 |
| `arb_scanner.py` | YES/NO and cross-event arbitrage | 1 |
| `arb_runner.py` | Continuous 30-second arb scanner | 1 |
| `latency_arb.py` | Crypto exchange-to-Kalshi speed trading | 1 |
| `copy_trade.py` | Mirror Polymarket whale trades | 1 |
| `settlement_tracker.py` | Track outcomes, feed P&L to research | Feedback |
| `kalshi_client.py` | Authenticated Kalshi REST API client | Infra |
| `risk_manager.py` | Position sizing, Kelly criterion, loss caps | Infra |
| `config.py` | Central configuration (all defaults) | Infra |
| `telegram_bot.py` | Interactive monitoring bot | Monitoring |
| `health_check.py` | HTTP health endpoint (port 8787) | Monitoring |
| `daily_recap.py` | 11 PM UTC daily KPI summary | Monitoring |
| `alerts.py` | Telegram + file notifications | Monitoring |
| `strategy_doc.py` | Auto-generates STRATEGY.md | Docs |
| `dashboard.py` | Browser-based dashboard | Monitoring |

### 2.3 Data Flow

```
NWS Forecast API ----\
GFS Ensemble ---------\
HRRR Model ------------> weather_strategy.py --> edge detection
Kalshi Market Prices --/                              |
                                                      v
                                               risk_manager.py
                                                      |
Coinbase/CoinGecko ----> crypto_strategy.py --------->|
ESPN Stats ------------> sports_strategy.py --------->|
Kalshi Orderbook ------> arb_scanner.py ------------->|
Polymarket Whale ------> copy_trade.py -------------->|
                                                      v
                                               kalshi_client.py
                                                      |
                                               [LIMIT ORDERS]
                                                      |
                                                      v
                                            settlement_tracker.py
                                                      |
                                            autoresearch/real_outcomes.json
                                                      |
                                                      v
                                            research_loop.py (learn from results)
```

---

## 3. AutoResearch Deep-Dive

### 3.1 What It Is

AutoResearch is a loop inspired by Andrej Karpathy's `autoresearch` pattern. The idea:
**instead of you manually tuning parameters, let the computer run thousands of experiments and keep what works.**

### 3.2 How One Iteration Works (Step by Step)

```
STEP 1: Read candidate_strategy.py
        Current: EDGE_THRESHOLD_CENTS = 3.0

STEP 2: Pick a random parameter to mutate
        Chosen: EDGE_THRESHOLD_CENTS
        Pick from: [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        New value: 5.0

STEP 3: Write the mutation to candidate_strategy.py
        EDGE_THRESHOLD_CENTS = 5.0  (was 3.0)

STEP 4: Run a full backtest (200 simulated weather days)
        - Generate 200 fake weather scenarios per city
        - Each scenario: pick a forecast, add noise for true temp
        - Create fake Kalshi markets with noisy prices
        - Run edge detection with the new parameters
        - Settle each trade against the true temperature
        - Calculate: Sortino, ROI, win rate, max drawdown

STEP 5: Score the result
        composite = Sortino*0.30 + ROI*0.20 + WinRate*0.15
                    + ProfitFactor*0.15 - MaxDrawdown*0.15

STEP 6: Decision
        IF new_score > best_score:
            git commit "WeatherResearch: EDGE_THRESHOLD=5.0 score=2.14"
            Keep the change. New best = 2.14
        ELSE:
            git revert candidate_strategy.py
            Throw away the change, go back to 3.0

STEP 7: Log to results.log, repeat from Step 1
```

### 3.3 The Mutation Space

These are the parameters AutoResearch can tweak, and the values it tries:

| Parameter | Description | Range Tested |
|-----------|-------------|-------------|
| `FORECAST_STDEV_0` | Forecast uncertainty today (degrees F) | 1.0 - 2.3 |
| `FORECAST_STDEV_1` | Forecast uncertainty tomorrow | 1.8 - 3.3 |
| `FORECAST_STDEV_2` | Forecast uncertainty day+2 | 2.5 - 4.5 |
| `FORECAST_STDEV_3` | Forecast uncertainty day+3 | 3.5 - 6.0 |
| `EDGE_THRESHOLD_CENTS` | Minimum edge to trade | 2.0 - 8.0 |
| `CONTRACTS_PER_TRADE` | Base contracts per trade | 5 - 25 |
| `NWS_OFFICIAL_WEIGHT` | Trust NWS vs ensemble (0=ensemble, 1=NWS) | 0.20 - 0.70 |
| `CITY_WEIGHT_*` | Position size multiplier per city | 0.5 - 1.5 |
| `BUCKET_MULTIPLIER` | Preference for bucket markets | 0.5 - 2.0 |
| `THRESHOLD_MULTIPLIER` | Preference for above/below markets | 0.5 - 2.0 |
| `HIGH_CONFIDENCE_EDGE` | Edge threshold for "high confidence" | 7.0 - 15.0 |
| `MAX_POSITION_DOLLARS` | Max dollars per single market | 3.0 - 10.0 |
| `MIN_VOLUME` | Minimum market volume filter | 5 - 50 |
| `TIGHT_ENSEMBLE_THRESHOLD` | Ensemble stdev below which to bet bigger | 1.5 - 3.0 |
| `TIGHT_ENSEMBLE_MULTIPLIER` | Position boost for tight ensembles | 1.0 - 2.0 |

### 3.4 The Scoring Function (How "Good" Is Measured)

The composite score that determines whether a mutation is kept:

```
score = Sortino * 0.30          (risk-adjusted returns - rewards consistency)
      + min(ROI, 500%) * 0.002  (total profit - capped to prevent overfitting)
      + WinRate * 0.15          (percentage of winning trades)
      + min(ProfitFactor, 3.0) * 0.45  (gross wins / gross losses)
      - MaxDrawdown * 0.15      (penalty for large losses)

Hard rejection: if MaxDrawdown > 60%, score = -999 (auto-reject)
```

### 3.5 Data Sources for Backtesting

The research loop uses three data sources (in priority order):

1. **Real outcomes** (`real_outcomes.json`) - actual trades with known forecast/actual temp
2. **Kalshi settled markets** (API fetch, cached 4 hours) - real market prices + NWS actual temps, weighted 3x
3. **Synthetic scenarios** - randomly generated weather days with realistic noise

---

## 4. The Math (Plain English)

### 4.1 Edge Detection (The Core)

**Question**: "Should I buy YES on 'NYC high temp = 72-73F tomorrow'?"

**Step 1**: Get the NWS forecast: "NYC high tomorrow = 71F"

**Step 2**: Model the uncertainty. The forecast isn't perfect, so we model it as a bell curve:
```
True temp ~ Normal(forecast=71, stdev=2.5)

This means:
  - 68% chance the true temp is within 71 +/- 2.5 (68.5 to 73.5)
  - 95% chance within 71 +/- 5.0 (66 to 76)
```

**Step 3**: Calculate the probability the temp lands in the 72-73 bucket:
```
P(72 <= temp < 73) = area under bell curve from 72 to 73
                   = normalCDF((73-71)/2.5) - normalCDF((72-71)/2.5)
                   = normalCDF(0.8) - normalCDF(0.4)
                   = 0.7881 - 0.6554
                   = 0.1327 (13.3%)

Fair value = 13.3 cents
```

**Step 4**: Compare to Kalshi's price:
```
Kalshi is selling YES at 8 cents (8% implied probability)
Our model says 13.3 cents
Edge = 13.3 - 8 = 5.3 cents

Edge > EDGE_THRESHOLD (3 cents) --> TRADE!
```

### 4.2 Kelly Criterion (Position Sizing)

Kelly tells you how much of your bankroll to bet based on your edge:

```
Full Kelly: f* = (b * p - q) / b
  where:
    p = probability of winning (our model's estimate)
    q = 1 - p (probability of losing)
    b = odds (what you win per dollar risked)

Example:
  We think p = 0.133 (13.3% chance this bucket hits)
  Kalshi price = 8 cents, so b = (100-8)/8 = 11.5x odds
  q = 0.867

  f* = (11.5 * 0.133 - 0.867) / 11.5 = 0.058 = 5.8% of bankroll

Ippo uses QUARTER Kelly: f*/4 = 1.45%
  On a $100 account: bet $1.45
```

Quarter Kelly is extremely conservative. Full Kelly maximizes long-term growth rate but has wild swings. Quarter Kelly trades growth for safety - exactly what you want when the model is still learning.

### 4.3 Sortino Ratio (The Primary Metric)

```
Sortino = (average return) / (downside deviation) * sqrt(250)

Unlike Sharpe (which penalizes ALL volatility), Sortino only penalizes
DOWNSIDE volatility. Upside variance is fine - we want big wins.

Sortino > 1.0 = decent
Sortino > 2.0 = very good
Sortino > 3.0 = exceptional
```

### 4.4 Monte Carlo Simulation

Resamples your actual trade P&L history to model what COULD happen:
```
1. Take your list of trade results: [+$0.50, -$0.30, +$1.20, -$0.80, ...]
2. Randomly pick N trades with replacement (like shuffling and dealing)
3. Compute the equity curve
4. Repeat 500 times
5. Result: 500 possible futures, showing probability of profit and ruin
```

---

## 5. Risk Management

### 5.1 Hard Limits

| Rule | Value | Why |
|------|-------|-----|
| Max per trade | $2 | Survive 50 consecutive losses |
| Daily loss cap | 8% ($8) | Stop bleeding on bad days |
| Max open positions | 5 | Avoid concentration risk |
| Min edge | 5 cents | Don't trade noise |
| Kelly fraction | 0.25x (quarter) | Ultra-conservative sizing |
| Max drawdown stop | 15% | Emergency circuit breaker |
| Order type | Limit only | Never pay the spread |
| Max daily deployment | 20% of account | Keep 80% as dry powder |

### 5.2 Risk Check Pipeline

Every trade proposal goes through this gauntlet:

```
Proposal: BUY YES KXHIGHNY-B72 @ 8c, 10 contracts

Check 1: Daily loss cap hit?         --> No ($0 P&L today) --> PASS
Check 2: Too many open positions?    --> No (2 of 5 max)   --> PASS
Check 3: Edge >= 5 cents?            --> Yes (5.3c edge)   --> PASS
Check 4: Kelly says bet?             --> Yes (1.45%)       --> PASS
Check 5: Position size within limits?
  kelly_amount = 1.45% * $100 = $1.45
  max_position = min($1.45, 2% * $100, $2, remaining_daily_budget)
  max_position = min($1.45, $2.00, $2.00, $8.00) = $1.45
  contracts = floor($1.45 / $0.08) = 18 --> cap at what Kelly says

Result: APPROVED - BUY 18 YES @ 8c ($1.44 risk)
```

---

## 6. Deployment Architecture

### 6.1 Infrastructure

| Component | Details |
|-----------|---------|
| VPS | DigitalOcean, Ubuntu 24.04, 2GB RAM, $12/mo |
| IP | 24.144.91.158 |
| Python | 3.14 |
| Git | Private repo `benjraju/ippo` |
| Process manager | systemd services + timers |

### 6.2 Running Processes

| Service | Schedule | What It Does |
|---------|----------|-------------|
| `ippo-autoresearch` | Every 2 hours | 3-phase: AutoResearch (100 iter) -> Auto-trade (live) -> Settlement check |
| `ippo-arb-runner` | Every 30 seconds | Continuous YES/NO arbitrage scanning |
| `ippo-telegram` | Always on | Interactive Telegram bot (@Ippotrading77_bot) |
| `ippo-health` | Always on | HTTP health check on port 8787 |

### 6.3 Deploy Workflow

```bash
# From MacBook:
bash deploy/deploy.sh root@24.144.91.158    # rsync code to VPS
ssh root@24.144.91.158 'systemctl restart ippo-arb-runner ippo-health ippo-autoresearch ippo-telegram'
```

### 6.4 Monitoring

- **Telegram bot**: /status /trades /pnl /strategy /research /health /pause /resume
- **Daily recap**: 11 PM UTC automatic KPI summary to Telegram
- **Health check**: External monitoring on port 8787
- **Logs**: `autoresearch/launchd.log`, `output/auto_trade_*.log`

---

## 7. Six Active Strategies

### 7.1 Weather Forecast Arbitrage (Primary)

**Edge source**: NWS forecasts update hourly; Kalshi retail traders don't check them.

**Data sources**: NWS API (free), GFS 31-member ensemble (Open-Meteo), HRRR 3km model

**Markets**: KXHIGHNY, KXHIGHCHI, KXHIGHMIA, KXHIGHLA, KXHIGHDC, KXHIGHDEN

**Model**: Normal distribution over forecast temp with day-dependent stdev

**Blending** (auto_trade.py uses multiple forecast sources):
- Day 0: 40% HRRR + 30% GFS + 30% NWS
- Day 1: 25% HRRR + 40% GFS + 35% NWS
- Day 2+: 60% GFS + 40% NWS

### 7.2 Crypto Bucket Pricing

**Edge source**: Log-normal model with realized volatility vs retail-priced buckets

**Data sources**: Coinbase + CoinGecko (cross-verified)

**Markets**: KXBTC, KXETH, KXSOL daily range buckets

**Model**: 30-day realized vol, log-normal distribution, asset-specific

### 7.3 NBA Sports Model

**Edge source**: Efficiency rating model vs Kalshi game prices

**Data sources**: ESPN free API (team stats, schedule)

**Markets**: KXNBA, KXNBAGAME, KXNBAPTS

### 7.4 Structural Arbitrage

**Edge source**: YES + NO prices sum to less than $1.00 (risk-free profit)

**How it works**: Buy both YES and NO, guarantee $1 payout for less than $1 cost

**Scanner**: Runs every 30 seconds via `arb_runner.py`

### 7.5 Latency Arbitrage

**Edge source**: Real-time crypto prices move before Kalshi reprices

**How it works**: Coinbase websocket detects price move -> Kalshi bucket is stale -> trade it

**Status**: Partially built (`latency_arb.py`, 855 lines)

### 7.6 Copy-Trading (Whale Mirror)

**Edge source**: Polymarket whale @k9Q2mX4L8A7ZP3R has $1.67M profit on 50K trades

**How it works**: Monitor whale's Polymarket positions, mirror on Kalshi

**Parameters**: Min confidence 0.8, 60s delay, 1% of whale's position size

---

## 8. Issues, Bugs, and Improvements

### 8.1 CRITICAL - Fix Before Trusting Live Trading

#### C1: Local codebase is incomplete
**Problem**: Your MacBook only has 6 of ~30 Python files. The complete codebase exists in `.claude/worktrees/` and on the VPS, but your local working directory is missing `config.py`, `cli.py`, `kalshi_client.py`, `market_scanner.py`, `auto_trade.py`, `weather_strategy.py`, `settlement_tracker.py`, and all other core files.

**Impact**: You can't run or test anything locally. If the VPS dies, you don't have a working local copy.

**Fix**: Sync the full codebase from VPS or from a worktree:
```bash
# Option A: from VPS
scp -r root@24.144.91.158:/opt/ippo/*.py ~/Desktop/ippo/
scp -r root@24.144.91.158:/opt/ippo/autoresearch/*.py ~/Desktop/ippo/autoresearch/
scp -r root@24.144.91.158:/opt/ippo/deploy/ ~/Desktop/ippo/deploy/

# Option B: from a worktree (already on your Mac)
cp ~/Desktop/ippo/.claude/worktrees/infallible-gagarin/*.py ~/Desktop/ippo/
cp ~/Desktop/ippo/.claude/worktrees/infallible-gagarin/autoresearch/*.py ~/Desktop/ippo/autoresearch/
```

#### C2: AutoResearch backtests synthetic data, not real markets
**Problem**: The research loop generates fake weather scenarios where "dumb retail" consistently misprices vs "smart money." This creates an artificially exploitable edge that may not exist in real Kalshi markets. The model can overfit to beating fake dumb traders.

**Impact**: Strategies that score well in simulation may not work in production.

**Fix**: The real-data pipeline exists (`--use-real-data`, Kalshi settlement fetching) but needs more data. Priority actions:
1. Ensure `settlement_tracker.py` is writing to `real_outcomes.json` on every settlement
2. Accumulate 30+ days of real settlement data before trusting AutoResearch results
3. Consider adding a "reality gap" metric that compares simulated vs real performance

#### C3: `eval()` in parameter parsing is a security risk
**Problem**: `research_loop.py:1165` uses `eval(val_part)` to parse parameter values from the strategy file. If the strategy file were corrupted or injected with malicious code, this would execute it.

**Impact**: Low risk in practice (the file is only written by the bot itself), but bad practice.

**Fix**: Replace with `ast.literal_eval()` which only evaluates literals (numbers, strings, bools):
```python
import ast
return ast.literal_eval(val_part)
```

#### C4: No validation that AutoResearch improvements generalize
**Problem**: Each backtest uses a different random seed (`seed=42+i`), but the baseline always uses `seed=42`. A mutation could score well on one seed but poorly on others (overfitting to noise).

**Impact**: "Improvements" may be random noise rather than real improvements.

**Fix**: Run each candidate against multiple seeds and require improvement on the average:
```python
scores = [run_weather_backtest(seed=42+s, ...) for s in range(5)]
avg_score = mean([s["score"] for s in scores])
kept = avg_score > best_score
```

### 8.2 HIGH - Significant Improvements

#### H1: AutoResearch mutations are unintelligent
**Problem**: Parameters are chosen randomly from a fixed list. No learning from which mutations worked before. The original Karpathy approach uses an LLM to propose intelligent mutations based on the research log.

**Impact**: Slow convergence. Most iterations are wasted trying random values.

**Fix**: Add an LLM-guided mutation mode (you already have the Anthropic API key):
```
1. Read the last 20 entries from results.log
2. Ask Claude: "Given these results, which parameter should I change next and why?"
3. Use Claude's suggestion instead of random choice
4. Fall back to random 30% of the time (exploration)
```

#### H2: No walk-forward validation
**Problem**: `research_program.md` says "always use walk-forward validation" but the code doesn't implement it. All backtests use the same data pool. The strategy could overfit to the simulation distribution.

**Impact**: Strategies may not generalize to future market conditions.

**Fix**: Split scenarios into train (first 70%) and test (last 30%). Only keep mutations that improve BOTH train and test scores.

#### H3: Scoring function weights are arbitrary
**Problem**: The composite score weights (Sortino 30%, ROI 20%, etc.) were chosen by intuition. Different weights could produce very different optimal strategies.

**Impact**: May be optimizing for the wrong objective.

**Fix**: Consider making the weights themselves part of the mutation space, or simplify to a single metric (Sortino alone is often sufficient for trading strategies).

#### H4: Copy-trade strategy has no backtesting
**Problem**: Copy-trade parameters are in `candidate_strategy.py` but the research loop only backtests weather strategy. Copy-trade, crypto, and sports strategies are not part of the self-improvement loop.

**Impact**: Only 1 of 6 strategies benefits from AutoResearch.

**Fix**: Build separate research loops for each strategy, or a unified loop that rotates between them.

#### H5: No out-of-sample tracking
**Problem**: There's no dashboard or metric tracking whether AutoResearch improvements actually translate to real trading performance.

**Impact**: You can't tell if the system is actually getting smarter or just overfitting.

**Fix**: Track two metrics side by side:
- "Simulated score" (from AutoResearch)
- "Real score" (from settlement_tracker.py)
If they diverge, the simulation is unreliable.

### 8.3 MEDIUM - Quality Improvements

#### M1: Worktree proliferation
**Problem**: There are 20+ worktree directories under `.claude/worktrees/`, each containing a full copy of the codebase. This wastes disk space and creates confusion about which version is canonical.

**Fix**: Clean up old worktrees:
```bash
cd ~/Desktop/ippo
git worktree list
git worktree prune
rm -rf .claude/worktrees/*/  # after confirming none have unique changes
```

#### M2: `run_bot.py` duplicates `auto_trade.py` logic
**Problem**: Two separate entry points (`run_bot.py` and `auto_trade.py`) do similar things but with different code paths. `run_bot.py` uses Claude for probability estimation while `auto_trade.py` uses mathematical models. This creates confusion about which one is "the real bot."

**Fix**: Deprecate `run_bot.py` in favor of `auto_trade.py`, which is more mature and runs in production.

#### M3: Backtester win rate is passed as an argument
**Problem**: `backtester.py:run_simulated_backtest()` takes `win_rate` as a parameter and uses `random.random() < win_rate` to decide outcomes. This doesn't test your actual strategy - it tests a coin flip with your assumed win rate.

**Impact**: Backtest results are only as good as your win rate assumption.

**Fix**: This backtester is effectively unused since `research_loop.py` has its own weather-specific backtester that tests actual strategy logic. Consider removing or clearly labeling `backtester.py` as a generic tool, not the strategy tester.

#### M4: No rate limiting on NWS API calls
**Problem**: `weather_strategy.py` fetches from the NWS API without rate limiting. NWS has a soft rate limit of ~5 requests/second. If multiple strategies or research iterations hit it simultaneously, you'll get 503 errors.

**Fix**: Add a simple rate limiter or use the `time.sleep(0.3)` pattern already used in `research_loop.py`.

#### M5: Config has stale Claude model reference
**Problem**: `config.py:41` references `claude-sonnet-4-20250514`. This model ID is from May 2025 and isn't the latest.

**Fix**: Update to the latest model, e.g. `claude-sonnet-4-6` for speed/cost efficiency on probability estimates.

#### M6: The 3x weighting of Kalshi scenarios is arbitrary
**Problem**: `research_loop.py:1063` weights real Kalshi settlements 3x (`kalshi_scenarios * 3`). This tripling is a magic number with no justification.

**Fix**: Either justify the weight empirically (e.g., "real data should be 75% of the training set") or make it a tunable parameter.

### 8.4 LOW - Nice to Have

#### L1: No automated alerting on strategy degradation
If the strategy's real-world win rate drops below expectations, nobody is notified.

#### L2: No A/B testing infrastructure
Can't run two strategy variants simultaneously and compare real performance.

#### L3: Weather-only AutoResearch
Crypto, sports, and copy-trade strategies don't benefit from the self-improvement loop yet.

#### L4: No position tracking across restarts
`risk_manager.py` tracks positions in memory. If the process restarts, daily P&L tracking resets.

#### L5: Arb scanner doesn't account for Kalshi fees
`arb_scanner.py` checks `yes + no < 97` (3-cent threshold) but doesn't factor in actual Kalshi fee structure, which may eat the edge.

---

## 9. Recommended Priority Roadmap

### Phase 1: Foundation (This Week)
1. **Sync local codebase** (C1) - get all files on your MacBook
2. **Fix `eval()` security issue** (C3) - one-line change
3. **Clean up worktrees** (M1) - reclaim disk space
4. **Update Claude model** (M5) - one-line change

### Phase 2: Research Integrity (Next 2 Weeks)
5. **Multi-seed validation** (C4) - prevent overfitting to random noise
6. **Walk-forward validation** (H2) - train/test split
7. **Track simulated vs real performance** (H5) - the reality check
8. **Accumulate 30+ days of real settlement data** (C2)

### Phase 3: Intelligence (Month 2)
9. **LLM-guided mutations** (H1) - make AutoResearch 5-10x faster
10. **Expand AutoResearch to crypto strategy** (H4/L3)
11. **Simplify scoring function** (H3) - maybe just use Sortino
12. **Add degradation alerts** (L1)

### Phase 4: Scale (Month 3+)
13. **Build latency arb strategy** (already partially coded)
14. **A/B testing infrastructure** (L2)
15. **Persistent position tracking** (L4)
16. **Increase position sizes once strategy is proven**

---

## 10. Key Metrics to Track

| Metric | Target | Current |
|--------|--------|---------|
| Sortino ratio | > 1.8 | Unknown (need real data) |
| Monthly ROI | > 8% | Unknown |
| Max drawdown | < 8% | Unknown |
| Win rate | > 68% (weather) | Unknown |
| Trades per day | 5-20 | Unknown |
| AutoResearch improvements per 100 iterations | 5-15 | Unknown |
| Real vs simulated score correlation | > 0.7 | Not tracked |
| Account balance | Growing | ~$85 |

---

## 11. Glossary

| Term | Meaning |
|------|---------|
| **Edge** | Difference between your model's probability and the market price. A 5-cent edge means you think the fair price is 5 cents higher than what you can buy it for. |
| **Sortino ratio** | Return per unit of downside risk. Like Sharpe but only penalizes losses, not upside variance. |
| **Kelly criterion** | Formula for optimal bet sizing. Quarter Kelly = very conservative version. |
| **Bucket market** | "Will NYC high be 72-73F?" - a specific temperature range. |
| **Threshold market** | "Will NYC high be above 70F?" - above or below a cutoff. |
| **Walk-forward** | Testing a strategy on data it hasn't seen. Train on old data, test on new. |
| **Monte Carlo** | Run thousands of random simulations to estimate probability distributions. |
| **Mutation** | Changing one strategy parameter to a new value (the core of AutoResearch). |
| **Settlement** | When a market resolves (YES or NO) and you know if you won or lost. |
| **Drawdown** | How much the account has dropped from its peak. Max drawdown = worst drop ever. |
| **Profit factor** | Gross profits / gross losses. Above 1.0 = profitable overall. |
| **NWS** | National Weather Service - free US government weather forecasts. |
| **GFS** | Global Forecast System - NOAA's primary weather model with 31 ensemble members. |
| **HRRR** | High-Resolution Rapid Refresh - 3km resolution model, very accurate for day-0/1. |
