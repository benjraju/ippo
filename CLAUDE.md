# Ippo Trading Bot

Autonomous Kalshi prediction market trading bot. ~$580 portfolio, 5 active strategies, deployed on a DigitalOcean VPS running 24/7.

## Architecture

- **Local**: `/Users/benjamin/Desktop/ippo`
- **VPS**: `root@24.144.91.158:/opt/ippo` (Ubuntu 24.04, 2GB RAM, Python 3.14)
- **GitHub**: `github.com/benjraju/ippo` (private)

### Systemd Services (on VPS)

| Service | Schedule | What it does |
|---------|----------|--------------|
| `ippo-auto-trade` | Every 2h (timer) | Main trading loop — runs all 5 strategies |
| `ippo-weather-tail` | Every 30m (timer) | Weather tail NO strategy runner |
| `ippo-arb-runner` | Always on | Scans for YES+NO arb every 30s |
| `ippo-autoresearch` | Always on | Karpathy-style experiment loop (Claude agent) |
| `ippo-telegram` | Always on | Telegram bot + 30-min portfolio updates |
| `ippo-health` | Always on | Health check on port 8787 |
| `ippo-settlement` | Every 4h (timer) | Settlement tracking, feeds P&L data |
| `ippo-watchdog` | Timer | Monitors other services |

## Key Files

| File | Purpose |
|------|---------|
| `auto_trade.py` | Main trading loop — 5 strategy sessions, balance checks, limit orders only |
| `config.py` | Risk limits: $25 max bet, 8% daily loss cap, half Kelly, 20 max positions |
| `kalshi_client.py` | RSA-signed Kalshi API client |
| `risk_manager.py` | Kelly criterion sizing, position limits |
| `weather_tail_strategy.py` | Weather tail NO logic + dutch book math |
| `weather_tail_runner.py` | Standalone weather tail runner (30m cycle) |
| `weather_strategy.py` | NWS/GFS forecast model for weather markets |
| `nba_underdog_strategy.py` | NBA underdog YES logic |
| `arb_scanner.py` | Scans for YES+NO mispricing |
| `arb_runner.py` | Continuous arb scanning loop |
| `dutch_book.py` | Dutch book (sum of YES bids > 102c) |
| `deep_itm_strategy.py` | Buy YES at 95c on near-certain markets |
| `settlement_timing_strategy.py` | Near-risk-free: buy YES/NO on known outcomes after 2 PM local |
| `ss.py` | Status dashboard — run `python ss.py` for full overview |
| `telegram_bot.py` | Telegram bot (@Ippotrading77_bot): /status /trades /pnl /strategy /pause |
| `watchdog.py` | Service health monitor |
| `alerts.py` | Telegram send helper |
| `settlement_tracker.py` | Tracks settlements, feeds P&L |
| `brier_tracker.py` | Calibration measurement |
| `autoresearch/program.md` | AutoResearch agent system prompt |
| `autoresearch/candidate_strategy.py` | The ONE file the autoresearch agent modifies |
| `autoresearch/backtest_harness.py` | Ground truth evaluator (40K+ markets, train/test split). DO NOT MODIFY. |
| `deploy/sync-to-vps.sh` | Full deploy script with smoke tests |

## Active Strategies

1. **Weather Tail NO** -- Buy NO on weather longshots (YES <= 15c). z=25.0, 12K samples. 100% win rate live.
2. **Weather Forecast** -- NWS/GFS model-based weather trading. ~58% win rate.
3. **NBA Underdog YES** -- Buy YES on game winners priced 8-20c. +7.7pp edge, z=7.04. ~25% WR but wins pay 4-12x.
4. **Arb Pairs** -- DISABLED by default (config.ARB_STRATEGY_ENABLED=False). 0 true arbs in 5,500+ scans. Kalshi spreads too wide. Set flag to True to re-enable.
5. **AutoResearch Bridge** -- Strategies discovered by the autoresearch agent that pass z >= 2.0 out-of-sample.
6. **Settlement Timing** -- Near-risk-free: buy YES/NO on weather markets after 2 PM local when observed high already exceeds threshold. Uses NWS station observations, not forecasts. Requires 2F buffer (3F after 4 PM). Max 5 contracts/market.

## Disabled Strategies -- PERMANENT KILL LIST (2026-03-28)

All losing strategies permanently disabled at 3 layers: removed from TARGET_MARKET_SERIES, added to BLOCKED_SERIES, and hardcoded skip in evaluate_market(). Do NOT re-enable any of these.

- **NBA Extreme NO** (-$38.75, -86% ROI, 20% WR): Pennies in front of steamrollers. Backtest 0% YES but live 3/3 YES at 7-8c.
- **NBA Player Props / KXNBAPTS** (-$18.96, -68% ROI): No player-level data = pure information disadvantage.
- **Crypto directional / KXBTC+KXETH+KXSOL** (-$6.95, -70% ROI, 15% WR): No real-time price feeds = no edge.
- **Other sports / KXNBA+KXNHL+KXMLB+KXNCAAB+KXMARMAD**: No proven edge. Removed to reduce noise.
- **Crypto tail fade**: TAIL_FADE_CRYPTO_ENABLED=0, params zeroed out.
- **NBA tail fade**: TAIL_FADE_NBA_ENABLED=0.

## Deploy Workflow

```bash
# Full deploy with smoke tests (preferred)
bash deploy/sync-to-vps.sh root@24.144.91.158

# Quick manual sync
rsync -avz --exclude '.venv/' --exclude '.git/' --exclude '__pycache__/' \
  --exclude '.claude/' --exclude 'output/' --exclude '.env' \
  --exclude 'kalshi_private_key.pem' \
  /Users/benjamin/Desktop/ippo/ root@24.144.91.158:/opt/ippo/
ssh root@24.144.91.158 'systemctl restart ippo-auto-trade ippo-arb-runner ippo-health ippo-telegram'
```

## Commands

```bash
# Status dashboard (on VPS)
ssh root@24.144.91.158 'cd /opt/ippo && .venv/bin/python ss.py'

# SSH to VPS
ssh root@24.144.91.158

# View service logs
ssh root@24.144.91.158 'journalctl -u ippo-auto-trade -n 50'
ssh root@24.144.91.158 'journalctl -u ippo-autoresearch -f'

# Dry run trading locally
python auto_trade.py --dry-run
```

## Critical Rules

1. **NEVER create `numpy.py`, `dotenv.py`, or `pandas.py`** in the project root. These shadow installed packages and break the entire bot. The deploy script auto-detects and removes them.
2. **Kalshi balance is the source of truth** -- not CSV files. trade_history.csv had 107 inflated trades with impossible contract sizes. Always query the API for real balance.
3. **Test before deploying.** Run `--dry-run` locally, verify with backtest, then deploy.
4. **Never modify `backtest_harness.py`.** It is the ground truth evaluator for autoresearch.
5. **Max $15 per trade.** This is a small account. Survival first.
6. **Limit orders only** -- maker fee 1.75%, taker fee 7%. Always use maker.
7. **AutoResearch agent rules:** It may only edit `candidate_strategy.py`. No new file creation, no Write tool, 500 line limit. Every strategy needs z >= 2.0 out-of-sample with 30+ trades.
8. **Keep contract counts sane** -- integers 1-200. Past autoresearch runs produced astronomical numbers (10^300+) that overflow in live trading.

## Known Issues

- **Rogue file protection**: AutoResearch agent has previously created `numpy.py` in the project root, shadowing the real numpy package. The watchdog and deploy script check for this.
- **Inflated CSV history**: Dec 2025 entries in trade_history.csv showed fake $56K P&L with impossible contract sizes. Cleaned 2026-03-24; backup at `output/trade_history_backup_with_inflated.csv`.
- **AutoResearch oscillation**: Old param mutation system was oscillating with 0 improvements. Replaced with Karpathy-style agent that runs hypothesis-driven experiments.
