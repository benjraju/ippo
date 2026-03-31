# Crypto Latency Arbitrage Strategy -- Implementation Plan

**Date**: 2026-03-28
**Status**: Research & Planning (no code yet)
**Inspired by**: Polymarket whale @k9Q2mX4L8A7ZP3R / 0x8dxd

---

## 1. The Core Thesis

Kalshi crypto markets (KXBTC, KXETH, KXSOL) offer bucket-style contracts like
"Will BTC close above $87,000 today?" that settle based on end-of-day prices.
These markets are updated by human market makers and retail traders who react
slowly to real-time crypto price moves. Meanwhile, centralized exchanges
(Binance, Coinbase) reflect price changes within milliseconds.

**The edge**: When BTC moves from $86,500 to $87,800 in minutes, the Kalshi
market for "BTC above $87,000" should jump from ~50c to ~85c. But Kalshi
market makers are slow -- the price might still sit at 55c for seconds to
minutes. We buy at 55c what is now worth 85c.

This is the same pattern as the Polymarket whale strategy: exploit the latency
between price discovery on liquid crypto exchanges and illiquid prediction
market order books.

### Why Kalshi Crypto Markets Are Vulnerable

1. **Low liquidity**: KXBTC/KXETH/KXSOL have thin books (often 1-5 contracts at each level)
2. **Slow updates**: Market makers don't use automated pricing tied to spot feeds
3. **Bucket structure**: "Above $X" and "Range $X-$Y" contracts have sharp probability transitions near strike prices -- a $500 BTC move can flip a 20% outcome to 80%
4. **Settlement lag**: Markets settle at end of day, but prices are observable in real time
5. **Fee advantage**: We can use limit orders (1.75% maker) to further improve entry

### What the Polymarket Whale Does (Reverse-Engineered)

@k9Q2mX4L8A7ZP3R on Polymarket:
- Monitors BTC/ETH spot prices via exchange websockets
- Watches prediction market order books for stale quotes
- When spot price crosses a strike level, instantly sweeps the stale side
- High win rate (>70%) because they're trading on information the market hasn't priced yet
- Position sizing is aggressive -- single trades up to $1,000+ on Polymarket

We adapt this for Kalshi's smaller markets and our $580 bankroll.

---

## 2. Kalshi Crypto Market Structure

### Series and Tickers

From `config.py`, we target three series:

| Series | Asset | Ticker Format | Example |
|--------|-------|---------------|---------|
| KXBTC  | Bitcoin  | `KXBTC-26MAR28-B87000` | "BTC above $87,000 on March 28" |
| KXETH  | Ethereum | `KXETH-26MAR28-B3200`  | "ETH above $3,200 on March 28" |
| KXSOL  | Solana   | `KXSOL-26MAR28-B145`   | "SOL above $145 on March 28" |

### Market Types

Kalshi crypto markets typically offer:
- **"Above $X"**: YES = price closes above X, NO = below
- **"Range $X-$Y"**: YES = price closes in range, NO = outside
- **Individual bucket**: "BTC closes between $86,000-$87,000"

Each event has multiple buckets covering the full price range. The buckets sum
to ~100c (with vig). This is the same structure as weather markets.

### Settlement

- Settles based on the **closing price** at a specific time (typically 4:00 PM ET or midnight UTC)
- Settlement source: CoinGecko or similar reference price
- Key insight: settlement is deterministic once we know the reference price at settlement time

### Current Kalshi Crypto Performance (from memory)

- **Crypto Tail NO**: PAUSED after -$5.36 with 32% live win rate (vs 100% backtest)
- The tail NO strategy was overfitting -- it buys NO on cheap YES but doesn't use real-time price data
- Latency arb is fundamentally different: it uses real-time price feeds, not statistical patterns

---

## 3. Edge Calculation Model

### Implied Probability from Kalshi Price

If a Kalshi market for "BTC above $87,000" trades at `P_kalshi` cents:
```
implied_prob_above = P_kalshi / 100
```

### True Probability from Spot Price + Volatility

Given:
- `S` = current BTC spot price (from Binance/Coinbase)
- `K` = strike price on Kalshi market
- `sigma` = annualized volatility of BTC
- `T` = time to settlement (in years)
- `mu` = drift (assume 0 for short timeframes)

Using a log-normal model (similar to Black-Scholes):
```
d = (ln(S/K) + (mu - sigma^2/2) * T) / (sigma * sqrt(T))
P_true_above = Phi(d)   # standard normal CDF
```

For intraday/next-day horizons (T < 1 day), this simplifies dramatically:
```
# For a 1-hour horizon with 50% annualized vol:
sigma_hourly = 0.50 / sqrt(365 * 24) = 0.50 / 93.6 = 0.00534

# BTC at $87,500, strike $87,000:
d = ln(87500/87000) / (0.00534 * sqrt(1/8760))
  = 0.00574 / 0.0000571
  = 100.5

P_true = Phi(100.5) = ~1.0   (virtually certain to be above)
```

This extreme example shows how even small price moves create near-certainty
once you're meaningfully above/below the strike, because intraday crypto vol
is tiny relative to the current price level.

### Edge = True Probability - Implied Probability

```
edge = P_true - P_kalshi
```

We only trade when:
1. `|edge| > MIN_EDGE_THRESHOLD` (e.g., 10+ cents)
2. The spot price move is significant relative to hourly volatility (>2 sigma)
3. The Kalshi price hasn't caught up yet (stale quote detection)

### Stale Quote Detection

A Kalshi quote is "stale" when:
1. The spot price has moved more than 1 standard deviation since the last Kalshi trade
2. The Kalshi order book hasn't updated in >60 seconds (no new trades/quotes)
3. The edge exceeds a threshold that's implausible for an efficient market

### Volatility Estimation

We need realized volatility, not implied. Calculate rolling:
- **1-hour realized vol**: from 1-minute candles (last 60 data points)
- **Daily realized vol**: from hourly candles (last 24 data points)
- Scale to annualized: `sigma_annual = sigma_period * sqrt(periods_per_year)`

For a 1-hour realized vol:
```
sigma_1h = stdev(log_returns of last 60 minutes)
sigma_annual = sigma_1h * sqrt(365 * 24)
```

---

## 4. Real-Time Crypto Price APIs

### Primary: Binance REST API (Free, No Key Required)

```python
# Current price
GET https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT
# Response: {"symbol": "BTCUSDT", "price": "87523.45"}

# Klines (candles) for volatility calculation
GET https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=60
# Returns: [[open_time, open, high, low, close, volume, ...], ...]
```

**Rate limits**: 1,200 requests/minute (20/second) -- more than enough for our 5-10 second polling.

### Secondary: Coinbase REST API (Free, No Key Required)

```python
# Current price
GET https://api.coinbase.com/v2/prices/BTC-USD/spot
# Response: {"data": {"base": "BTC", "currency": "USD", "amount": "87523.45"}}
```

**Rate limits**: 10 requests/second -- adequate for backup.

### WebSocket Option (Future Enhancement)

Binance WebSocket for real-time streaming:
```python
wss://stream.binance.com:9443/ws/btcusdt@ticker
# Pushes price updates in real-time (~100ms latency)
```

The `websockets` package is already in `requirements.txt` (websockets>=12.0).
Initial implementation uses REST polling (simpler, sufficient for our
30-second edge window). WebSocket upgrade is Phase 2.

### Symbols Mapping

| Kalshi Series | Binance Symbol | Coinbase Symbol |
|---------------|----------------|-----------------|
| KXBTC         | BTCUSDT        | BTC-USD         |
| KXETH         | ETHUSDT        | ETH-USD         |
| KXSOL         | SOLUSDT        | SOL-USD         |

---

## 5. Strategy Architecture

### How It Fits Into the Existing System

The crypto latency arb runs as a **standalone continuous runner** (like
`arb_runner.py` and `weather_tail_runner.py`), not inside the 2-hour
`auto_trade.py` cycle. The 2-hour cycle is too slow -- latency arb
opportunities exist for seconds to minutes.

```
Existing Architecture:
  auto_trade.py (2h timer)  -- 5 strategy sessions, slow cycle
  arb_runner.py (30s loop)  -- YES/NO mispricing scanner
  weather_tail_runner.py (30m loop)  -- weather tail NO

New Addition:
  crypto_latency_runner.py (5s loop)  -- latency arb scanner + executor
```

### Proposed File Structure

```
ippo/
  crypto_latency_arb.py       # Core strategy logic (edge calc, stale detection)
  crypto_price_feed.py         # Binance/Coinbase price fetching + vol estimation
  crypto_latency_runner.py     # Standalone runner (5-second loop, systemd service)
  deploy/
    ippo-crypto-latency.service   # systemd unit file
    ippo-crypto-latency.timer     # (not needed -- service runs continuously)
```

### Data Flow

```
[Binance API] -----> crypto_price_feed.py -----> current spot price
                                                  + realized volatility
                                                  + price change magnitude

[Kalshi API]  -----> crypto_latency_arb.py -----> current Kalshi prices
                                                  + order book depth
                                                  + last trade timestamp

crypto_latency_arb.py:
  1. For each open KXBTC/KXETH/KXSOL market:
     a. Get strike price from ticker (parse "B87000" -> $87,000)
     b. Get current spot price from price feed
     c. Calculate P_true using log-normal model + realized vol
     d. Get P_kalshi from current bid/ask
     e. Calculate edge = P_true - P_kalshi
     f. Check stale quote indicators
  2. If edge > threshold AND stale quote detected:
     a. Calculate position size (Kelly or fixed)
     b. Place limit order at favorable price
     c. Log trade with full rationale

crypto_latency_runner.py:
  - Runs every 5 seconds
  - Manages crypto_price_feed (caches, rate limits)
  - Calls crypto_latency_arb for each scan
  - Risk management integration (daily loss cap, max exposure)
  - Telegram alerts for trades placed
```

### Integration Points

| Component | Integration |
|-----------|-------------|
| `config.py` | New constants: `CRYPTO_LATENCY_*` prefix |
| `kalshi_client.py` | No changes -- already has all needed methods |
| `risk_manager.py` | Use existing `RiskManager` for daily loss tracking |
| `alerts.py` | Use existing `alert_big_edge` and `alert_trade_settled` |
| `telegram_bot.py` | Add `/crypto` command for latency arb status |
| `candidate_strategy.py` | New section: `CRYPTO_LATENCY_*` parameters for AutoResearch |
| `auto_trade.py` | No changes -- latency arb runs independently |
| `ss.py` | Add crypto latency stats to dashboard |

---

## 6. Risk Parameters

### Position Sizing

Given our $580 bankroll, we need conservative sizing:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max per trade | $5 | ~1% of bankroll, limits single-trade blowup |
| Max daily exposure | $25 | ~4.3% of bankroll across all latency arb trades |
| Max concurrent positions | 5 | Across all crypto series |
| Min edge threshold | 10 cents | Only trade large, clear mispricings |
| Stale quote threshold | 60 seconds | Kalshi market must be inactive for >1 min |
| Kelly fraction | 0.25 | Quarter Kelly -- very conservative |
| Max daily loss | $15 | ~2.6% of bankroll, auto-shutdown |

### Entry Criteria (ALL must be met)

1. **Edge >= 10 cents** (true probability vs Kalshi implied)
2. **Spot price move >= 2 sigma** (relative to 1-hour realized vol)
3. **Kalshi book stale >= 60 seconds** (no trades or quote updates)
4. **Sufficient liquidity** (at least 1 contract available at target price)
5. **Within daily exposure limit** (not already maxed out)
6. **Time to settlement > 2 hours** (avoid settlement-related vol)
7. **Not already holding a position on this market**

### Exit Criteria

1. **Edge reversal**: If edge drops below 3 cents, cancel any unfilled orders
2. **Time-based**: Cancel unfilled limit orders after 5 minutes
3. **Hold to settlement**: If filled, hold to settlement (don't try to exit early -- Kalshi spreads are wide)
4. **Stop loss**: If spot price reverses past the strike, the position becomes a loser -- but we sized small enough (max $5) that this is acceptable

### Risk Scenarios

| Scenario | Probability | Loss | Mitigation |
|----------|-------------|------|------------|
| Spot reverses after we buy | ~25% | $1-5 per trade | Quarter Kelly sizing |
| Flash crash during position | <1% | Up to $5 per trade | Max position cap |
| API rate limiting | ~5% | Missed opportunity (no loss) | Fallback to Coinbase |
| Kalshi API down | <1% | No trading (no loss) | Graceful degradation |
| All daily trades lose | ~2% | Up to $15 | Daily loss cap shutdown |
| Strategy not profitable | Unknown | Up to $25/day | 2-week paper trading first |

---

## 7. Volatility Model Details

### Realized Volatility Calculation

```python
def calc_realized_vol(candles_1min: list, window: int = 60) -> float:
    """
    Calculate 1-hour realized volatility from 1-minute candles.
    Returns annualized volatility.
    """
    log_returns = [
        math.log(candles[i+1].close / candles[i].close)
        for i in range(len(candles) - 1)
    ]
    if len(log_returns) < 10:
        return DEFAULT_ANNUAL_VOL  # fallback: 60% annualized

    sigma_1min = statistics.stdev(log_returns)
    sigma_annual = sigma_1min * math.sqrt(365 * 24 * 60)  # annualize
    return sigma_annual
```

### True Probability Calculation

```python
def calc_true_prob_above(spot: float, strike: float, sigma_annual: float,
                          hours_to_settlement: float) -> float:
    """
    Log-normal probability that price at settlement > strike.
    Assumes zero drift for short horizons.
    """
    T = hours_to_settlement / (365 * 24)
    if T <= 0 or sigma_annual <= 0:
        return 1.0 if spot > strike else 0.0

    sigma_T = sigma_annual * math.sqrt(T)
    d = math.log(spot / strike) / sigma_T
    return normal_cdf(d)  # reuse from weather_strategy.py
```

### Dynamic Threshold Adjustment

The edge threshold should scale with:
1. **Time to settlement**: Larger threshold far from settlement (more uncertainty)
2. **Current volatility**: Larger threshold in high-vol regimes
3. **Market liquidity**: Larger threshold in thin markets (wider effective spread)

```python
def dynamic_edge_threshold(base_threshold: float, hours_to_settlement: float,
                           realized_vol: float, book_depth: int) -> float:
    vol_factor = min(realized_vol / 0.50, 2.0)  # scale by vol relative to 50%
    time_factor = math.sqrt(hours_to_settlement / 24)  # sqrt scaling
    liquidity_penalty = max(0, (5 - book_depth) * 0.02)  # 2c per missing level
    return base_threshold * vol_factor * time_factor + liquidity_penalty * 100
```

---

## 8. Ticker Parsing

Kalshi crypto tickers encode the strike price. We need to parse them:

```python
def parse_crypto_strike(ticker: str) -> tuple[str, float, str]:
    """
    Parse KXBTC-26MAR28-B87000 -> ("KXBTC", 87000.0, "above")
    Parse KXETH-26MAR28-T3200 -> ("KXETH", 3200.0, "above")
    Parse KXBTC-26MAR28-86000-87000 -> ("KXBTC", (86000, 87000), "range")
    """
    # Implementation will need to handle:
    # - B prefix (above/below threshold)
    # - Range format (two prices)
    # - Date parsing for settlement time
```

This requires checking actual Kalshi ticker formats by querying the API for
KXBTC markets and examining the response structure. The exact format may differ
from the examples above.

**Action item**: Before implementation, query Kalshi for 5-10 live KXBTC markets
and document the exact ticker format, subtitle field, and any metadata that
encodes the strike price.

---

## 9. Implementation Phases

### Phase 1: Price Feed + Edge Scanner (3-4 hours)

**Goal**: Real-time crypto prices + edge calculation against Kalshi markets.
No trading yet -- just logging opportunities.

Files:
- `crypto_price_feed.py` -- Binance/Coinbase price fetching, vol calculation
- `crypto_latency_arb.py` -- Edge calculation, stale detection, opportunity scoring

Testing:
- Run for 24 hours, log all opportunities with edge > 5 cents
- Analyze: how many opportunities per day? Average edge? How long do they persist?
- Compare spot price vs Kalshi implied -- verify our model is calibrated

### Phase 2: Paper Trading Runner (2-3 hours)

**Goal**: Full execution pipeline but dry-run only.

Files:
- `crypto_latency_runner.py` -- 5-second loop, dry-run mode
- Config additions in `config.py` and `candidate_strategy.py`

Testing:
- Run alongside live bot for 1 week in dry-run mode
- Track simulated P&L: would we have been profitable?
- Tune entry/exit thresholds based on observed data
- Verify no interference with existing strategies

### Phase 3: Live Trading (1-2 hours)

**Goal**: Enable real order placement.

Changes:
- Toggle dry-run flag off
- Add systemd service on VPS
- Telegram alerts for crypto latency trades
- Dashboard integration in ss.py

Testing:
- Start with $1 max per trade (paper size)
- Gradually increase to $5 over 1 week if profitable
- Monitor daily via /crypto Telegram command

### Phase 4: WebSocket Upgrade (2-3 hours, future)

**Goal**: Replace REST polling with WebSocket streaming for lower latency.

Changes:
- WebSocket connection to Binance for real-time price stream
- Reduce reaction time from 5 seconds (REST poll) to <1 second
- Handle reconnection, heartbeats, message parsing

This is only worth doing if Phase 3 shows that the 5-second polling misses
opportunities that a faster feed would catch.

---

## 10. Dependencies

### Existing (no changes)

| Package | Version | Used For |
|---------|---------|----------|
| `requests` | >=2.31.0 | REST API calls to Binance/Coinbase |
| `websockets` | >=12.0 | Future WebSocket upgrade |
| `numpy` | >=1.26.0 | Volatility calculations |

### New Packages Required

**None.** The strategy uses only standard library + existing dependencies.
Binance and Coinbase REST APIs are free, no-auth, and use simple JSON
over HTTPS. The `requests` library handles everything.

---

## 11. Expected Performance

### Optimistic Scenario (Edge Exists and Is Exploitable)

- **Opportunities per day**: 5-15 (based on Polymarket data showing ~10 exploitable mispricings/day)
- **Average edge**: 12 cents per contract
- **Win rate**: 70% (edge detected correctly, Kalshi price catches up)
- **Average position**: 3 contracts at $0.50 = $1.50 deployed
- **Daily P&L**: ~$0.50-$1.50 per day (after fees)
- **Monthly**: ~$15-$45
- **Sharpe**: >2.0 (frequent small wins, rare small losses)

### Realistic Scenario (Edge Is Small)

- **Opportunities per day**: 2-5
- **Average edge**: 8 cents
- **Win rate**: 60%
- **Daily P&L**: ~$0.10-$0.50 per day
- **Monthly**: ~$3-$15
- **Sharpe**: ~1.2-1.5

### Pessimistic Scenario (No Edge)

- **Finding**: Kalshi books update fast enough that REST polling can't catch them
- **Action**: Kill the strategy after 2 weeks of paper trading
- **Cost**: Engineering time only (no capital at risk during paper phase)
- **Lesson**: Need WebSocket-level speed or the edge doesn't exist on Kalshi

### Key Unknown

**Does the edge actually exist on Kalshi?** Polymarket has confirmed latency arb
opportunities, but Kalshi's crypto markets are much smaller (lower liquidity =
wider spreads but also fewer sophisticated participants). The edge might be:
- Larger than Polymarket (less efficient market)
- Smaller (not enough liquidity to trade)
- Non-existent (Kalshi may use automated market makers that update instantly)

**Phase 1 (scanner) answers this question before we risk any capital.**

---

## 12. Monitoring and Observability

### Logging

Every scan cycle logs:
```
[SCAN] BTC=$87,523 | KXBTC-26MAR28-B87000: kalshi=55c true=91c edge=36c stale=120s -> TRADE
[SCAN] ETH=$3,245  | KXETH-26MAR28-B3200: kalshi=60c true=62c edge=2c stale=5s -> SKIP (edge < 10c)
[SCAN] SOL=$148.50 | KXSOL-26MAR28-B145: kalshi=72c true=74c edge=2c stale=30s -> SKIP (not stale)
```

### Metrics to Track

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Scan latency | <500ms | >2s |
| Opportunities/day | >2 | 0 for 6 hours |
| Win rate | >60% | <40% over 20 trades |
| Daily P&L | >$0 | <-$5 (daily loss cap) |
| Binance API errors | 0 | >5/hour |
| Kalshi API errors | <2/hour | >10/hour |
| Realized vol calculation | Valid | NaN or negative |

### Telegram Alerts

- Trade placed: "CRYPTO ARB: Bought YES KXBTC-B87000 x3 @ 55c (spot=$87,523, true=91c, edge=36c)"
- Daily summary: Integrated with existing daily summary from auto_trade.py
- Error: "CRYPTO ARB: Binance API unreachable for 5 minutes"

---

## 13. Safety Considerations

### What Can Go Wrong

1. **Model miscalibration**: Our vol estimate is wrong, leading to bad P_true
   - Mitigation: Use realized vol (data-driven), not assumed vol
   - Mitigation: Phase 1 paper trading validates calibration

2. **Ticker parsing error**: We misread the strike price from the ticker
   - Mitigation: Validate parsed strike against market subtitle/description
   - Mitigation: Unit tests for ticker parser

3. **Race condition**: We place an order, spot price reverses, we're stuck
   - Mitigation: Limit orders only (post_only=True), cancel after 5 min
   - Mitigation: Max $5 per trade, $15 daily loss cap

4. **API rate limiting**: We poll too fast and get blocked
   - Mitigation: Binance allows 20 req/s, we do 1-2/5s (well under limit)
   - Mitigation: Fallback to Coinbase if Binance is down

5. **Interference with existing strategies**: Arb runner also trades crypto
   - Mitigation: Dedup against existing positions (same pattern as auto_trade.py)
   - Mitigation: Shared position tracker across runners

6. **Double-spending**: Multiple runners place orders that exceed bankroll
   - Mitigation: Balance check before every order (existing pattern)
   - Mitigation: Dedicated crypto latency budget separate from other strategies

### Non-Negotiable Safety Rules

1. **Paper trade for 2 weeks before going live**
2. **Max $5 per trade, $15 daily loss cap**
3. **Limit orders only** (post_only=True, 1.75% maker fee)
4. **Balance check before every order**
5. **Kill switch**: `CRYPTO_LATENCY_ENABLED = False` in config to instantly disable
6. **No modifications to existing strategy files** -- this is entirely additive

---

## 14. Estimated Development Effort

| Phase | Hours | Deliverable |
|-------|-------|-------------|
| 1. Price Feed + Edge Scanner | 3-4h | `crypto_price_feed.py`, `crypto_latency_arb.py` |
| 2. Paper Trading Runner | 2-3h | `crypto_latency_runner.py`, config additions |
| 3. Live Trading | 1-2h | Systemd service, Telegram integration |
| 4. WebSocket Upgrade | 2-3h | Replace REST with Binance WebSocket |
| **Total** | **8-12h** | Full latency arb system |

### Priority Order

1. Phase 1 is the most important -- it answers whether the edge exists
2. Phase 2 validates without risking capital
3. Phase 3 only after 2 weeks of positive paper P&L
4. Phase 4 only if Phase 3 shows missed opportunities due to polling speed

---

## 15. Open Questions to Resolve Before Implementation

1. **Kalshi crypto ticker format**: Need to query the API for live KXBTC/KXETH/KXSOL
   markets and document the exact ticker structure, especially how strike prices
   are encoded. The examples above are hypothetical.

2. **Settlement time**: What exact time do KXBTC markets settle? Is it midnight UTC,
   4 PM ET, or market-specific? This affects our time-to-settlement calculation.

3. **Kalshi order book update frequency**: How often does the Kalshi order book
   change for crypto markets? If it updates every few seconds, our 5-second
   polling might be too slow. If it updates every few minutes, we have ample time.

4. **Existing position dedup**: How should the latency arb runner coordinate with
   `arb_runner.py` which also scans KXBTC/KXETH/KXSOL? Options:
   - Shared position file that both runners read/write
   - Latency arb replaces the crypto portion of arb_runner
   - Both run independently with separate budgets

5. **Binance vs. CoinGecko pricing**: Kalshi may use CoinGecko for settlement.
   If there's a systematic difference between Binance spot and CoinGecko settlement
   price, we need to account for that basis risk.

6. **VPS latency to Binance API**: The VPS is in NYC (DigitalOcean). Binance API
   servers are globally distributed. Need to measure actual round-trip latency
   to ensure we can poll fast enough.

---

## 16. Next Steps

1. **Query Kalshi API** for 5-10 live KXBTC markets to document exact ticker format
   and market structure
2. **Build Phase 1** (price feed + scanner) and run for 24-48 hours
3. **Analyze scan results** to determine if exploitable edges exist
4. **Decision point**: Continue to Phase 2 or kill the strategy based on data
