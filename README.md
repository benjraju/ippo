# Kalshi Self-Improving Trading Bot

A prediction market trading bot for [Kalshi](https://kalshi.com) that uses AI (Claude) for probability estimation and an AutoResearch loop to improve its strategy overnight.

**Paper mode only by default. No real money at risk until you explicitly opt in.**

---

## Quick Start (3 commands)

### 1. Setup (one command)

```bash
cd ~/Desktop/kalshi-self-improving-bot && bash setup.sh
```

### 2. Add your API keys

```bash
nano .env
```

Fill in:
- `KALSHI_API_KEY_ID` — from https://demo.kalshi.co (Settings > API Keys)
- `KALSHI_PRIVATE_KEY_PATH` — path to your downloaded .pem file
- `ANTHROPIC_API_KEY` — from https://console.anthropic.com (optional but recommended)

### 3. Run

```bash
source .venv/bin/activate

# Scan markets
python cli.py scan-markets

# Run backtest with charts
python cli.py backtest --plot

# Start paper trading
python run_bot.py --paper

# Start overnight AutoResearch
python cli.py research --overnight
```

---

## How It Works

### Trading Loop
1. Scans Kalshi for high-volume, fast-settling markets (weather, sports, crypto)
2. Uses Claude AI to estimate true probabilities and find edges
3. Runs arb scanner for mispricings
4. Applies strict risk management (max $2/trade, 8% daily loss cap)
5. Places trades on Kalshi demo platform

### AutoResearch (Overnight Self-Improvement)
Based on [Karpathy's AutoResearch pattern](https://github.com/trevin-creator/autoresearch-mlx):
1. Reads `autoresearch/candidate_strategy.py`
2. Uses Claude (or random mutations) to propose parameter changes
3. Runs full backtest with P&L metrics
4. Git-commits improvements, reverts regressions
5. Repeats overnight — wake up to a smarter bot

### Risk Management
- **Quarter Kelly sizing** — extremely conservative position sizing
- **$2 max per trade** on a $100 account
- **8% daily loss cap** — bot stops trading after $8 daily loss
- **5-cent minimum edge** — only trades with statistical advantage
- **Max 5 concurrent positions**

---

## Commands

| Command | What it does |
|---------|-------------|
| `python cli.py scan-markets` | Show active high-volume markets |
| `python cli.py arb-scan` | Find arbitrage opportunities |
| `python cli.py backtest --plot` | Run backtest, open charts |
| `python cli.py backtest --monte-carlo` | Backtest + Monte Carlo simulation |
| `python cli.py research --overnight` | 50-iteration overnight research |
| `python cli.py research --quick` | Quick 10-iteration test |
| `python cli.py status` | Show account balance & positions |
| `python cli.py analyze TICKER` | Deep analysis of one market |
| `python run_bot.py --paper` | Start paper trading loop |
| `python run_bot.py --live` | Live trading (requires confirmation) |

---

## Output Files

All in the `output/` directory:
- `equity_curve.html` — Interactive equity curve (open in browser)
- `equity_curve.png` — Static chart image
- `pnl_distribution.html` — P&L histogram
- `monte_carlo.html` — Monte Carlo simulation paths
- `trades.csv` — Trade-by-trade log

---

## Project Structure

```
kalshi-self-improving-bot/
├── setup.sh                  # One-command setup
├── .env.template             # API key template
├── run_bot.py                # Main trading loop
├── cli.py                    # CLI commands
├── config.py                 # All configuration
├── kalshi_client.py          # Kalshi API client
├── market_scanner.py         # High-volume market finder
├── arb_scanner.py            # Arbitrage detector
├── claude_agent.py           # Claude probability estimator
├── risk_manager.py           # Position sizing & loss caps
├── backtester.py             # Backtest engine + charts
├── autoresearch/
│   ├── research_loop.py      # AutoResearch engine
│   ├── candidate_strategy.py # Strategy file (AI edits this)
│   └── research_program.md   # Research goals
├── output/                   # Charts, CSVs, logs
└── requirements.txt
```

---

## Live Trading (DANGER)

Only after you've:
1. Run backtests and seen positive results
2. Paper traded for several days
3. Verified AutoResearch has improved the strategy

```bash
# Edit .env: change KALSHI_ENV=PROD and use production API keys
python run_bot.py --live
```

You will be asked to type "YES I UNDERSTAND THE RISKS" before any real orders are placed. Max risk: $2 per trade, $8 per day.
