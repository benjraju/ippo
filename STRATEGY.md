# Ippo Strategy Document

*Auto-generated on March 21, 2026 at 10:45 AM. This document updates every time `strategy_doc.py` runs.*

---

## How Ippo Makes Money

Ippo is a weather forecast arbitrage bot. Here's how it works in plain English:

1. **We get weather forecasts** from the National Weather Service (NWS) and GFS
   ensemble models for cities like NYC, Chicago, Miami, LA, DC, and Denver.

2. **We check prediction markets** on Kalshi, where people bet on questions like
   "Will the high temperature in Miami be above 80 degrees tomorrow?"

3. **We compare our forecast to the market price.** If our weather model says
   there's a 70% chance of hitting 80 degrees but the market is only pricing it
   at 55%, that's a 15-cent edge -- the market is underpricing the outcome.

4. **When we spot a big enough disagreement, we place a bet.** We buy the
   underpriced side and wait for the weather to actually happen.

5. **The market settles based on real weather data.** If we were right, we profit.
   Over many trades, our better forecasts should translate into consistent gains.

We also copy-trade large "whale" traders on other Kalshi markets (crypto, sports)
when their signals meet our confidence threshold.

---

## Current Strategy Settings

*Last updated: March 21, 2026 at 10:45 AM*

### Forecast Model

| Setting | Current Value | What It Means |
|---------|:------------:|---------------|
| Same-day forecast confidence | 1.5 | How uncertain we think today's weather forecast is (in degrees F). Lower means we trust the forecast more and trade more aggressively. |
| Tomorrow forecast confidence | 2.5 | Uncertainty for tomorrow's forecast. Lower means tighter model, more trades on tomorrow's markets. |
| 2-day forecast confidence | 3.5 | Uncertainty for day-after-tomorrow. Higher means we're more cautious on these markets. |
| 3-day forecast confidence | 4.5 | Uncertainty for 3-day-out forecasts. Higher means fewer trades on longer-dated markets. |
| Weather source blend | 0.4 | How much we trust the official NWS forecast vs. the GFS ensemble model. 1.0 = all NWS, 0.0 = all ensemble. |
| Ensemble agreement threshold | 2 | When forecast models agree within this many degrees, we consider the forecast extra reliable. |
| Ensemble agreement bet boost | 1.5 | When models agree, multiply our bet size by this amount. Higher = bigger bets on consensus forecasts. |

### Trading Rules

| Setting | Current Value | What It Means |
|---------|:------------:|---------------|
| Minimum edge to trade | 3 | The smallest advantage (in cents) we need before placing a bet. Higher = fewer but higher-quality trades. |
| Trade size (contracts) | 10 | How many contracts we buy per trade. More contracts = bigger bets. |
| High confidence threshold | 7 | Edge (in cents) needed to flag a trade as 'high confidence' and size up. Lower = more aggressive sizing. |
| Medium confidence threshold | 5 | Edge (in cents) needed to flag a trade as 'medium confidence'. Lower = we take more borderline trades. |
| Max position per market ($) | 5 | The most dollars we will put into any single market. |
| Minimum market volume | 10 | We skip markets with fewer contracts traded than this (too illiquid). |

### City Weights

| Setting | Current Value | What It Means |
|---------|:------------:|---------------|
| NYC trading weight | 1 | How aggressively we trade NYC weather markets relative to baseline. |
| Chicago trading weight | 0.8 | How aggressively we trade Chicago weather markets relative to baseline. |
| Miami trading weight | 1 | How aggressively we trade Miami weather markets relative to baseline. |
| LA trading weight | 1 | How aggressively we trade LA weather markets relative to baseline. |
| DC trading weight | 1 | How aggressively we trade DC weather markets relative to baseline. |
| Denver trading weight | 1 | How aggressively we trade Denver weather markets relative to baseline. |

### Market Preferences

| Setting | Current Value | What It Means |
|---------|:------------:|---------------|
| Bucket market preference | 1 | How much we favor 'temperature falls in X-Y range' markets. Above 1.0 = trade more of these. |
| Threshold market preference | 1 | How much we favor 'temperature above/below X' markets. Above 1.0 = trade more of these. |

### Copy-Trading

| Setting | Current Value | What It Means |
|---------|:------------:|---------------|
| Copy-trade confidence filter | 0.8 | Minimum confidence to auto-copy a whale trade. Higher = only copy the best signals. |
| Copy-trade reaction speed | 60 | How many seconds we wait after seeing a whale trade before copying. Lower = faster reaction. |
| Copy-trade position fraction | 0.01 | What fraction of the whale's position we copy. 0.01 = 1% of their size. |
| Max copy trades per day | 10 | Daily cap on how many whale trades we copy. |
| Min whale trade size ($) | 100 | We only copy trades above this dollar amount. |
| Whale trader trust weight | 1 | How much we trust signals from this specific tracked trader. 1.0 = full trust. |
| Copy crypto multiplier | 1.5 | Size multiplier when copying crypto trades. 0 = skip, 1 = normal, 2 = double. |
| Copy politics multiplier | 0 | Size multiplier when copying politics trades. 0 = skip entirely. |
| Copy sports multiplier | 0.5 | Size multiplier when copying sports trades. 0.5 = half size. |

---

## What AutoResearch Has Learned

AutoResearch runs experiments overnight, tweaking one parameter at a
time and backtesting to see if the change improves performance.

- **Total experiments run:** 37
- **Improvements found:** 4
- **Hit rate:** 10.8% of experiments improved the strategy

### Top Changes That Improved the Strategy

**1. Chicago trading weight** -- decreased from 1.0 to 0.8  
   Backtest result: 1287% return, 33.9% win rate, 15.4% max drawdown (score: 1.36)
   *How aggressively we trade Chicago weather markets relative to baseline.*

**2. High confidence threshold** -- decreased from 10.0 to 7.0  
   Backtest result: 3630% return, 33.7% win rate, 15.4% max drawdown (score: 1.24)
   *Edge (in cents) needed to flag a trade as 'high confidence' and size up. Lower = more aggressive sizing.*

**3. Miami trading weight** -- increased from 1.0 to 1.5  
   Backtest result: 691% return, 33.0% win rate, 27.6% max drawdown (score: -0.06)
   *How aggressively we trade Miami weather markets relative to baseline.*

**4. Miami trading weight** -- decreased from 1.5 to 1.0  
   Backtest result: 1295% return, 33.1% win rate, 24.6% max drawdown (score: -0.16)
   *How aggressively we trade Miami weather markets relative to baseline.*

### Most-Tested Parameters

| Parameter | Times Tested |
|-----------|:------------:|
| Miami trading weight | 6 |
| High confidence threshold | 4 |
| Medium confidence threshold | 3 |
| Chicago trading weight | 3 |
| Minimum edge to trade | 3 |

---

## Real Performance

### Overall (from real_outcomes.json)

- **Total settled trades:** 133
- **Total P&L:** +$56,864.07
- **Win rate:** 44.4%
- **Profit factor:** 2.79
- **Worst drawdown:** $8,864.83

### By Strategy

| Strategy | Trades | Wins | P&L |
|----------|:------:|:----:|----:|
| Other | 37 | 13 (35%) | $-172.78 |
| Sports | 96 | 46 (48%) | +$57,036.85 |

### Weather Strategy

No weather trades have settled yet. Weather trades are currently open 
and will settle when the actual temperature is recorded.

---

## What's Trading Next

We have **16 open positions** waiting to settle:

### Crypto Markets

| Market | Our Bet | Contracts | Entry Price | What Needs to Happen |
|--------|:-------:|:---------:|:-----------:|----------------------|
| Bitcoin price range  on Mar 21, 2026? | NO | 1 | 86c | This must NOT happen (settle NO) for us to win |
| Bitcoin price range  on Mar 21, 2026? | NO | 1 | 87c | This must NOT happen (settle NO) for us to win |
| Bitcoin price range  on Mar 21, 2026? | NO | 1 | 88c | This must NOT happen (settle NO) for us to win |

### Other Markets

| Market | Our Bet | Contracts | Entry Price | What Needs to Happen |
|--------|:-------:|:---------:|:-----------:|----------------------|
| Will the **high temp in Denver** be 90-91° on Mar 21, 2026? | NO | 1 | 50c | This must NOT happen (settle NO) for us to win |

### Weather Markets

| Market | Our Bet | Contracts | Entry Price | What Needs to Happen |
|--------|:-------:|:---------:|:-----------:|----------------------|
| Will the high temp in Chicago be 67-68° on Mar 21, 2026? | YES | 25 | 4c | This needs to happen (settle YES) for us to win |
| Will the high temp in Chicago be 69-70° on Mar 21, 2026? | YES | 11 | 9c | This needs to happen (settle YES) for us to win |
| Will the high temp in Chicago be >70° on Mar 21, 2026? | NO | 5 | 19c | This must NOT happen (settle NO) for us to win |
| Will the **high temp in Miami** be 75-76° on Mar 21, 2026? | YES | 20 | 5c | This needs to happen (settle YES) for us to win |
| Will the **high temp in Miami** be 77-78° on Mar 21, 2026? | YES | 16 | 6c | This needs to happen (settle YES) for us to win |
| Will the **high temp in Miami** be 79-80° on Mar 21, 2026? | NO | 1 | 54c | This must NOT happen (settle NO) for us to win |
| Will the **high temp in Miami** be 81-82° on Mar 21, 2026? | NO | 1 | 58c | This must NOT happen (settle NO) for us to win |
| Will the **high temp in Miami** be <75° on Mar 21, 2026? | YES | 50 | 2c | This needs to happen (settle YES) for us to win |
| Will the **high temp in Miami** be >82° on Mar 21, 2026? | NO | 1 | 92c | This must NOT happen (settle NO) for us to win |
| Will the **high temp in NYC** be 53-54° on Mar 21, 2026? | NO | 1 | 90c | This must NOT happen (settle NO) for us to win |
| Will the **high temp in NYC** be 57-58° on Mar 21, 2026? | NO | 1 | 58c | This must NOT happen (settle NO) for us to win |
| Will the **high temp in NYC** be 59-60° on Mar 21, 2026? | NO | 1 | 82c | This must NOT happen (settle NO) for us to win |

---

## Risk Controls

These safety limits protect the account from big losses:

- **Maximum bet per trade:** $2 -- no single trade can risk
  more than this, no matter how good the edge looks.

- **Daily loss cap:** 8% of the account -- if we lose
  this much in one day, all trading stops until tomorrow. On a $100 account,
  that's $8.

- **Quarter Kelly sizing:** We use the Kelly Criterion (a math formula for
  optimal bet sizing) but only bet 25% of what Kelly suggests. This is very
  conservative -- it means slower growth but much lower risk of ruin.

- **Position limit:** Maximum 5 open positions at once.

- **Minimum edge:** We need at least a 5-cent edge
  before placing any trade. No edge = no trade.

- **Minimum volume:** We only trade markets with at least 50
  contracts traded, so we can always get in and out.
