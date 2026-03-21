# Ippo Cheat Sheet

## Daily Check Commands

```bash
cd ~/Desktop/ippo && source .venv/bin/activate

# Dashboard (opens in browser — KPIs, positions, health)
python dashboard.py

# Check if your trades won or lost
python cli.py check-settlements

# Daily P&L report
python cli.py daily-summary

# See account balance + open positions
python cli.py status
```

## Trading Commands

```bash
# Run all strategies (dry-run, safe)
python cli.py auto-trade

# Run all strategies LIVE
python cli.py auto-trade --live

# Run specific strategy only
python cli.py auto-trade --weather-only
python cli.py auto-trade --btc-only
python cli.py auto-trade --sports-only
python cli.py auto-trade --arb-only
python cli.py auto-trade --copy-only
```

## See What Edges Exist Right Now

```bash
python cli.py weather-edges        # Weather forecast vs Kalshi prices
python cli.py btc-edges             # BTC bucket mispricings
python cli.py nba-edges             # Sports model edges
python cli.py arb-scan              # YES/NO arbitrage
python arb_runner.py --scan-once    # Orderbook arb scan (deeper)
python copy_trade.py --signals      # What whales are trading
```

## Strategy Discovery (Claude-powered)

```bash
python strategy_discovery.py --scan      # Scan all Kalshi markets by category
python strategy_discovery.py --analyze   # Claude estimates on top 10 markets
python strategy_discovery.py --find-edges  # Markets with >10c edge
python strategy_discovery.py --lab-status  # Accuracy per category
python strategy_discovery.py --settle    # Check outcomes vs predictions
```

## Copy-Trade (Watch Polymarket Whales)

```bash
python copy_trade.py --watch        # Watch 0x8dxd live (polls every 60s)
python copy_trade.py --positions    # Their current open positions
python copy_trade.py --signals      # Signals to mirror on Kalshi
python copy_trade.py --stats        # Copy-trade accuracy stats
python copy_trade.py --add-wallet ADDRESS NAME  # Track another whale
```

## AutoResearch (Overnight Optimization)

```bash
# Weather strategy optimization
python cli.py research --quick         # 10 iterations (test)
python cli.py research --overnight     # 50 iterations

# Copy-trade parameter optimization
python -m autoresearch.copytrade_research --iterations 50

# Run research loop directly
python -m autoresearch.research_loop --iterations 100 --no-claude
python -m autoresearch.research_loop --iterations 100 --use-real-data
```

## Monitoring Background Processes

```bash
# Check arb runner (scans every 30s)
tail -f ~/Desktop/ippo/output/arb_runner_launchd.log

# Check autoresearch logs
cat ~/Desktop/ippo/autoresearch/launchd.log

# See if background jobs are running
launchctl print gui/$(id -u)/com.kalshi.arb-runner
launchctl print gui/$(id -u)/com.kalshi.autoresearch

# Restart arb runner
launchctl kickstart -k gui/$(id -u)/com.kalshi.arb-runner

# Stop arb runner
launchctl bootout gui/$(id -u)/com.kalshi.arb-runner

# Restart autoresearch
launchctl kickstart -k gui/$(id -u)/com.kalshi.autoresearch
```

## Latency Arb (Crypto Speed Trading)

```bash
python latency_arb.py --dry-run     # Watch for crypto latency edges
python latency_arb.py --live        # Execute latency arbs
```

## Output Files

```
output/
  dashboard.html              # Browser dashboard
  auto_trade_YYYY-MM-DD.log   # Daily trade log
  auto_trade_YYYY-MM-DD.json  # Daily trade decisions (JSON)
  trade_history.csv            # All settled trades with P&L
  daily_summary.csv            # Daily/weekly P&L summaries
  forecast_accuracy.csv        # Weather forecast vs actual temps
  equity_curve.html            # Backtest equity curve
  trades.csv                   # Backtest trade log
  strategy_lab.json            # Strategy discovery predictions
  copytrade_log.json           # Copy-trade signal history
  arb_runner_YYYY-MM-DD.json   # Arb execution log
  arb_runner_launchd.log       # Arb runner background log
  alerts.log                   # All macOS notifications sent
```

## What Runs Automatically

| Process | Schedule | What It Does |
|---------|----------|-------------|
| `com.kalshi.autoresearch` | Every 2 hours | AutoResearch + trade + settle |
| `com.kalshi.arb-runner` | Every 30 seconds | YES/NO arb scanning |

## Go Live Checklist

1. Paper trade for a few days, check results with `check-settlements`
2. Switch arb runner: edit plist, change `--dry-run` to `--live`
3. For main bot: launchd already runs `--live --no-confirm`
4. Monitor daily with `python dashboard.py`

## Risk Limits

- $2 max per trade
- 8% daily loss cap
- 5 max concurrent positions per strategy
- Quarter Kelly sizing
- Limit orders only (never market)
- 15% hard drawdown stop

## 6 Active Strategies

1. **Weather** — Forecast arb (NWS + GFS + HRRR vs Kalshi)
2. **Crypto** — BTC/ETH/SOL bucket pricing (lognormal model)
3. **Sports** — NBA win probability model
4. **Structural Arb** — YES + NO < $1.00 (risk-free)
5. **Latency Arb** — Exchange prices move before Kalshi reprices
6. **Copy-Trade** — Mirror Polymarket whale 0x8dxd
