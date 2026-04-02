# ScalperPro v2 — Indian Index Options Swing Trading System

Automated options trading system for **NIFTY, Bank NIFTY, FinNifty, Midcap Nifty, and Sensex**. Targets 15–20 point swings on option premiums with tight 10–15 point stop losses using a 4-layer decision pipeline.

---

## The Strategy

This system does **NOT** predict index direction and then buy options. Instead, it follows a 3-step logic that experienced option traders use:

### Step 1: Read the Index → Decide CE or PE

Every index (Nifty, BankNifty, etc.) has key **support and resistance levels** built from 1 year of daily/weekly price action. When the index approaches one of these levels:

- **Index near support** → index likely bounces up → **buy CE** (call option premium will rise)
- **Index near resistance** → index likely rejects down → **buy PE** (put option premium will rise)

The system marks levels using swing highs/lows across 4 timeframes (weekly, daily, hourly, 15-min), then adds Fibonacci retracements, CPR pivots, and round numbers. Where multiple techniques agree on the same level (**confluence**), the level is strongest.

**How close is "near"?** Not a fixed distance — the system uses **ATR (Average True Range)** to adapt. In a volatile market (high ATR), "near" means further away. In a quiet market, it means very close. Within 0.3× ATR = ready to trade. 0.3–0.8× ATR = prepare.

### Step 2: Read the Option Premium → Find Entry/Exit

Once we know to buy CE or PE, we open the **option premium's own chart** (not the index chart). This is the key insight: every option contract has its own swing levels.

If NIFTY 23500 CE premium has dropped to ₹120 three times and bounced to ₹140 each time, then ₹120 **is** support on that option. We:

- **Enter** when premium is at or near its support zone (₹120–122)
- **Stop loss** just below the support zone (₹108–110) → only 12–14 points risk
- **Target** at premium resistance (₹138–142) → 18–20 points reward

The system scans premium charts on **1-min, 5-min, and 15-min** timeframes. A support zone confirmed on multiple timeframes is stronger.

### Step 3: Pick the Right Strike → Greeks Optimization

Not all strikes are equal for swing trading. The system picks strikes where:

- **Delta 0.35–0.55** → the leverage sweet spot. Too high delta (deep ITM) = expensive, no leverage. Too low (far OTM) = sluggish movement.
- **High Gamma** → this is why swings work. Near support, gamma accelerates the bounce. As premium recovers, delta increases, which makes the premium move even faster. This is the "snap-back" effect.
- **Theta < 1.5% of premium per day** → the swing needs 15–60 minutes to play out. If theta eats more than 1.5% per day, the decay erodes your edge.

This typically points to **slight OTM or ATM** strikes with **3–7 days to expiry**.

### Step 4: Confirm with Indicators → Don't Trade Blind

Before executing, the system runs 7 technical indicators **on the premium chart** (not the index):

| Indicator | What It Confirms | Points |
|-----------|-----------------|--------|
| **13 EMA High/Low** | Price breaking above the channel = bullish. Near channel low = bounce zone. | +2 |
| **5 EMA** | Fast momentum — price above 5 EMA with positive slope = go. | +1 |
| **Supertrend (7,3)** | The strongest filter. Only trade when Supertrend is green (bullish). | +2 |
| **RSI (14)** | Oversold at support = high-probability bounce. | +1 |
| **Volume** | Spike above 1.8× average with more buying than selling = institutions entering. | +2 |
| **Bollinger (20,2)** | Squeeze (bands tightening) = breakout coming. Near lower band = bounce zone. | +1 |
| **VWAP** | Above VWAP = institutional bullish bias. | +1 |

**Minimum 4 out of 10 points required** to confirm the trade. This gate filters out 60%+ of setups that would have been losers.

---

## Why 40% Win Rate Is Profitable

The backtest shows ~40% win rate. This sounds low but is by design:

- Average winner: **₹731** (target hit)
- Average loser: **₹351** (stop loss hit)
- **Win:Loss ratio: 2.08×**

With 2:1 reward-to-risk, you only need **33.3% win rate to break even**. At 40%, every 10 trades produces: 4 winners × ₹731 = ₹2,924, minus 6 losers × ₹351 = ₹2,106. **Net: +₹818 per 10 trades**.

### Failure Analysis — What Was Fixed

| Problem Found | Data | Fix Applied |
|---------------|------|-------------|
| Opening 30 min (9:15–9:45) | 33.8% WR — gap noise | No trades before 9:45 AM |
| Closing 20 min (15:10+) | 37.6% WR — decay acceleration | No trades after 3:10 PM |
| Overtrading | 8 trades/day diluted quality | Max 5 trades/day |
| Multi-position | Split focus | Max 1 position at a time |
| 2:00–3:00 PM | 48.5% WR — best window | System naturally weights this |
| Max losing streak | 15 consecutive | Reduced to 9 with filters |

---

## Architecture

```
scalper/
├── main_v2.py                  ← Entry point (backtest/paper/live)
├── run.py                      ← Quick start CLI
├── config/
│   └── settings.py             ← All parameters, index configs, risk limits
├── core/
│   ├── index_levels.py         ← LAYER 1: Index S/R marking (4 timeframes)
│   ├── premium_swings.py       ← LAYER 2: Option premium swing detection
│   ├── greek_selector.py       ← LAYER 3: Delta/Gamma/Theta strike scoring
│   └── swing_orchestrator.py   ← Orchestrates Layers 1→3→2
├── indicators/
│   ├── confirmation.py         ← LAYER 4: 7-indicator confirmation gate
│   ├── technical.py            ← Base indicator calculations
│   └── oi_analyzer.py          ← Open Interest analysis
├── execution/
│   └── dhan_engine.py          ← Dhan API orders, trailing SL, paper mode
├── alerts/
│   └── telegram.py             ← Telegram alerts (optional)
├── backtest/
│   └── engine.py               ← Walk-forward backtester
├── data/
│   ├── fetcher.py              ← Dhan data fetcher
│   └── free_fetcher.py         ← Yahoo Finance + NSE (free, 5yr history)
├── .env.example                ← API credentials template
└── requirements.txt
```

Total: **~7,000 lines** across 17 modules.

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- Dhan account with Free Trading API key
- (Optional) Telegram bot for alerts

### Step 1: Install Dependencies

```bash
cd scalper
pip install -r requirements.txt
pip install yfinance    # For free 5-year historical data
```

### Step 2: Set Up API Credentials

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```
DHAN_CLIENT_ID=your_client_id_here
DHAN_ACCESS_TOKEN=your_access_token_here
```

**Never share these. Never commit .env to git.**

For the Dhan API key:
1. Go to web.dhan.co → API section
2. Application Name: `ScalperPro`
3. Redirect URL: `https://localhost`
4. Generate → copy Client ID + Access Token

### Step 3: Run Backtest (No API Needed)

```bash
# With sample data (works offline, instant)
python main_v2.py --mode backtest --sample --days 180

# With real Yahoo Finance data (needs internet)
python main_v2.py --mode backtest --days 180
```

### Step 4: Paper Trading

```bash
python main_v2.py --mode paper --indices NIFTY BANKNIFTY FINNIFTY MIDCPNIFTY SENSEX
```

This will:
- Scan all indices every 30 seconds during market hours
- Run the full 4-layer pipeline
- Log trades to `data/trades.json`
- Send Telegram alerts (if configured)
- Generate Excel summaries
- **NOT place any real orders**

Paper trade for **2–3 months minimum** before going live.

### Step 5: Live Trading (After Paper Validation)

```bash
python main_v2.py --mode live --indices NIFTY BANKNIFTY
```

You will be asked to type `YES` to confirm. This places **real orders with real money** via Dhan.

---

## Data Sources

| Source | What | Cost | Limit |
|--------|------|------|-------|
| Yahoo Finance | Daily candles | Free | 5 years |
| Yahoo Finance | 5-min candles | Free | 60 days |
| Yahoo Finance | 1-min candles | Free | 7 days |
| Dhan Trading API | Live candles + orders | Free | Requires account |
| NSE Website | Option chain, India VIX | Free | Rate limited |
| Dhan Data API | Real-time + 5yr intraday | ₹499/month | Optional, not needed to start |

---

## Risk Management

| Parameter | Value | Reason |
|-----------|-------|--------|
| Capital per trade | ₹25,000–50,000 | 1–2 lots based on premium |
| Max daily loss | ₹10,000 | Hard stop — system shuts down |
| Max trades per day | 5 | Quality over quantity |
| Max open positions | 1 | Full focus on one setup |
| Trading window | 9:45 AM – 3:10 PM | Avoids opening/closing chaos |
| Cooldown after loss | 5 minutes | Prevents revenge trading |
| Trailing stop loss | Activates at 50% of target | Locks in profit, trails at 30% |

---

## What the Excel Report Contains

The system generates `ScalperPro_v2_Report.xlsx` with 5 sheets:

1. **Dashboard** — KPI cards (trades, win rate, P&L, Sharpe, drawdown), monthly P&L color-coded, per-index breakdown
2. **Trade Log** — Every trade: entry/exit, SL, target, exit reason, P&L in points and ₹, hold time, signal score. Filterable.
3. **Equity Curve** — Cumulative P&L chart + drawdown tracking
4. **Failure Analysis** — What went wrong, what filters were applied, remaining risks
5. **Configuration** — Full system parameters, all 4 layers documented

---

## Important Disclaimers

1. **Backtest ≠ live results.** Sample data validates the engine, not the strategy. Run on real Yahoo Finance data before paper trading.
2. **Slippage is not modeled.** Real fills differ by 1–3 points. The system accounts for this with the 2:1 reward-to-risk ratio.
3. **OI data is not available in backtest.** Live mode uses real OI for Layer 1 signals — results may improve.
4. **This is a tool, not financial advice.** Validate everything independently. Never risk money you cannot afford to lose.
5. **Paper trade for 2–3 months** before going live. No exceptions.

---

## File Quick Reference

| File | What It Does |
|------|-------------|
| `main_v2.py` | Run this. Handles backtest/paper/live modes. |
| `config/settings.py` | Change parameters here. All strategy tuning in one place. |
| `core/index_levels.py` | Layer 1: Marks index S/R across 4 timeframes. |
| `core/premium_swings.py` | Layer 2: Finds swing levels on option premium charts. |
| `core/greek_selector.py` | Layer 3: Picks best strike by Delta/Gamma/Theta. |
| `indicators/confirmation.py` | Layer 4: 7-indicator confirmation gate (4/10 minimum). |
| `execution/dhan_engine.py` | Dhan API: bracket orders, trailing SL, paper mode. |
| `data/free_fetcher.py` | Yahoo Finance + NSE for free historical data. |
| `.env` | Your API credentials. Never share this file. |
