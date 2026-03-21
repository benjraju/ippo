# Kalshi Weather Forecast Arbitrage Strategy — Full AutoResearch Goals (March 2026)

Primary objective:
Maximize Sortino ratio + net ROI on Kalshi KXHIGH series (daily high temperature buckets) using real forecast arbitrage while keeping risk tiny for a $100 account.

Target markets ONLY:
- KXHIGHNY, KXHIGHCHI, KXHIGHMIA, KXHIGHLAX, KXHIGHDC, KXHIGHDEN (and any new high-volume cities)
- Only markets resolving in 24–48 hours with >$75k 24h volume

Core strategy (the one that works):
Compare ensemble forecasts (GFS 31-member from Open-Meteo + ECMWF + NWS official) against Kalshi bucket prices.
- Calculate mean forecast temp + stdev from ensemble
- Build probability distribution for each temperature bucket
- Edge = |forecast probability - Kalshi mid price| minus fees/slippage
- Trade YES or NO only when edge survives all filters

Parameters AutoResearch must optimize every night:
1. Edge threshold (start at 0.08 / 8% — test 0.04 to 0.15)
2. Stdev multiplier for confidence bands (1.0σ to 2.5σ)
3. NWS station weighting vs model ensemble (0–100%)
4. City weighting (higher liquidity cities like NYC get bigger positions)
5. Position sizing / Kelly fraction (0.1× to 0.5×, max $2 per trade on $100 account)
6. Entry timing (how many hours before close)
7. Dynamic exit rules (probability drift >5%, or 70% resolution)
8. Minimum liquidity & open interest filters

Hard safety constraints (never break these):
- Max position = $2 (2% of $100 bankroll)
- Daily loss cap = 8%
- Overall drawdown breaker = 15%
- After realistic fees (Kalshi + slippage) every strategy must show positive expectancy
- Only trade markets with resolution <48 hours

Metrics to beat every experiment:
- Sortino ratio > 1.8
- Net ROI > 8% per month
- Max drawdown < 8%
- Win rate > 68%
- At least 200 trades in walk-forward backtest

Experiment rules:
- Always use walk-forward validation (train on older data, test on most recent 30 days)
- Run Monte Carlo (1,000 simulations) on every candidate
- Use real historical Kalshi settlements + actual NWS outcomes
- Log equity curve PNG + trade CSV after every winner

Start every loop by reading current best performance from results.log and beating it. Prioritize weather arbitrage above all other strategies.

Goal: Turn this into the most profitable, safest $100 Kalshi weather bot possible.
