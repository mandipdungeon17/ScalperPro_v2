"""
=============================================================================
SCALPER PRO - Configuration Settings
=============================================================================
Indian Index Options Scalping & Swing System
Supports: NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, SENSEX
=============================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import json
import os

# Load .env from the scalper/ directory (parent of this config/ folder)
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(dotenv_path=os.path.abspath(_env_path), override=True)
except ImportError:
    pass  # dotenv optional — fall back to OS env vars


# ─────────────────────────────────────────────────────────────────────────────
# API CREDENTIALS (Set via environment variables or .env file)
# ─────────────────────────────────────────────────────────────────────────────

DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "YOUR_DHAN_CLIENT_ID")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "YOUR_DHAN_ACCESS_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID")


# ─────────────────────────────────────────────────────────────────────────────
# TRADING MODE
# ─────────────────────────────────────────────────────────────────────────────

class TradingMode(Enum):
    BACKTEST = "backtest"
    PAPER = "paper"          # Sends alerts, no real orders
    LIVE = "live"            # Real orders via Dhan

CURRENT_MODE = TradingMode.PAPER   # START WITH PAPER ALWAYS


# ─────────────────────────────────────────────────────────────────────────────
# INDEX CONFIGURATIONS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IndexConfig:
    name: str
    symbol: str
    dhan_security_id: str       # Dhan's security ID for the index
    exchange_segment: str       # NSE_FNO or BSE_FNO
    lot_size: int
    tick_size: float
    strike_interval: int        # Gap between strikes (50 for Nifty, 100 for BankNifty)
    max_lots: int               # Based on ₹25k-50k capital
    scalp_target: int           # Points target for scalping
    scalp_stoploss: int         # Points SL for scalping
    swing_target: int           # Points target for 100+ pt swing
    swing_stoploss: int         # Points SL for swing
    typical_spread: float       # Typical bid-ask spread in points
    expiry_day: str             # Weekly expiry day

INDEX_CONFIGS: Dict[str, IndexConfig] = {
    "NIFTY": IndexConfig(
        name="NIFTY 50",
        symbol="NIFTY",
        dhan_security_id="13",
        exchange_segment="NSE_FNO",
        lot_size=75,
        tick_size=0.05,
        strike_interval=50,
        max_lots=2,
        scalp_target=25,
        scalp_stoploss=12,
        swing_target=120,
        swing_stoploss=40,
        typical_spread=1.0,
        expiry_day="Thursday",
    ),
    "BANKNIFTY": IndexConfig(
        name="BANK NIFTY",
        symbol="BANKNIFTY",
        dhan_security_id="25",
        exchange_segment="NSE_FNO",
        lot_size=30,
        tick_size=0.05,
        strike_interval=100,
        max_lots=2,
        scalp_target=30,
        scalp_stoploss=15,
        swing_target=150,
        swing_stoploss=50,
        typical_spread=2.0,
        expiry_day="Wednesday",
    ),
    "FINNIFTY": IndexConfig(
        name="FIN NIFTY",
        symbol="FINNIFTY",
        dhan_security_id="27",
        exchange_segment="NSE_FNO",
        lot_size=40,
        tick_size=0.05,
        strike_interval=50,
        max_lots=2,
        scalp_target=25,
        scalp_stoploss=12,
        swing_target=100,
        swing_stoploss=35,
        typical_spread=1.5,
        expiry_day="Tuesday",
    ),
    "MIDCPNIFTY": IndexConfig(
        name="MIDCAP NIFTY",
        symbol="MIDCPNIFTY",
        dhan_security_id="442",
        exchange_segment="NSE_FNO",
        lot_size=50,
        tick_size=0.05,
        strike_interval=25,
        max_lots=1,              # Lower due to liquidity
        scalp_target=20,
        scalp_stoploss=10,
        swing_target=100,
        swing_stoploss=35,
        typical_spread=2.5,
        expiry_day="Monday",
    ),
    "SENSEX": IndexConfig(
        name="SENSEX",
        symbol="SENSEX",
        dhan_security_id="51",
        exchange_segment="BSE_FNO",
        lot_size=20,
        tick_size=0.05,
        strike_interval=100,
        max_lots=2,
        scalp_target=30,
        scalp_stoploss=15,
        swing_target=150,
        swing_stoploss=50,
        typical_spread=3.0,
        expiry_day="Friday",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScalpParameters:
    """Parameters for the 20-30 point scalping strategy."""

    # EMA Settings
    ema_fast: int = 5
    ema_mid: int = 9
    ema_slow: int = 20

    # RSI
    rsi_period: int = 7
    rsi_long_threshold: float = 60.0
    rsi_short_threshold: float = 40.0
    rsi_deadzone_low: float = 45.0
    rsi_deadzone_high: float = 55.0

    # Supertrend
    supertrend_period: int = 7
    supertrend_multiplier: float = 3.0

    # MACD (fast settings for scalping)
    macd_fast: int = 5
    macd_slow: int = 13
    macd_signal: int = 1

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # ATR
    atr_period: int = 14

    # VWAP bands
    vwap_std_1: float = 1.0
    vwap_std_2: float = 2.0

    # Stochastic RSI
    stoch_rsi_period: int = 14
    stoch_rsi_k: int = 3
    stoch_rsi_d: int = 3

    # Volume confirmation
    volume_spike_multiplier: float = 2.0  # Volume > 2x avg = spike
    volume_avg_period: int = 20

    # VIX filter
    vix_min: float = 12.0
    vix_max: float = 22.0
    vix_sweet_low: float = 14.0
    vix_sweet_high: float = 20.0

    # OI filters
    pcr_bullish_threshold: float = 1.2
    pcr_bearish_threshold: float = 0.7

    # Signal scoring - minimum score to trigger trade
    # Analysis showed score=7 has 41% WR, score=8 drops to 33% (noise)
    # Raising bar to 8 AND requiring volume confirmation
    min_scalp_score: int = 8       # Out of ~12 possible signals
    min_swing_score: int = 9       # Higher bar for swing trades

    # Timeframes (in minutes)
    trend_timeframe: int = 15      # Overall direction
    setup_timeframe: int = 5       # Setup identification
    entry_timeframe: int = 3       # Entry timing


@dataclass
class SwingParameters:
    """Parameters for the 100+ point swing strategy using 1-year levels."""

    # Lookback for swing levels
    yearly_lookback_days: int = 365
    monthly_lookback_days: int = 30

    # Support/Resistance detection
    sr_touch_count: int = 3         # Min touches to confirm a level
    sr_zone_width_pct: float = 0.2  # % width of S/R zone

    # Fibonacci levels to watch
    fib_levels: List[float] = field(
        default_factory=lambda: [0.236, 0.382, 0.5, 0.618, 0.786]
    )

    # Minimum distance from key level to enter (points)
    min_proximity_pct: float = 0.3   # Within 0.3% of key level

    # Target: Risk ratio for swing trades
    min_rr_ratio: float = 2.5


@dataclass
class StrikeSelectionParams:
    """Parameters for optimal strike selection to minimize decay."""

    # Delta targeting
    min_delta: float = 0.65
    ideal_delta_low: float = 0.70
    ideal_delta_high: float = 0.80

    # Theta constraints
    max_theta_pct_of_target: float = 0.10  # Theta cost < 10% of target

    # Days to expiry logic
    dte_switch_to_next_week: int = 1  # On expiry day, use next week
    dte_max_theta_day: int = 2        # Extra ITM on last 2 days

    # Liquidity
    min_oi: int = 5000                # Minimum OI on the strike
    min_bid_ask_ratio: float = 0.95   # Tight spread required

    # IV filter
    max_iv_percentile: float = 80.0   # Avoid strikes with IV > 80th percentile


# ─────────────────────────────────────────────────────────────────────────────
# RISK MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskConfig:
    max_capital_per_trade: float = 50000.0    # ₹50,000 max
    min_capital_per_trade: float = 25000.0    # ₹25,000 min
    max_daily_loss: float = 10000.0           # Stop trading if daily loss > ₹10k
    max_daily_trades: int = 5                 # Quality over quantity — fewer, better trades
    max_concurrent_positions: int = 1          # Focus on one trade at a time
    trailing_sl_activation: float = 0.5        # Activate trailing SL at 50% of target
    trailing_sl_distance: float = 0.3          # Trail at 30% of target
    cooldown_after_loss_seconds: int = 300     # 5-min cooldown after a loss
    no_trade_before: str = "09:45"             # Avoid first 30 min — analysis shows 33.8% WR
    no_trade_after: str = "15:10"              # No new trades in last 20 min
    no_trade_window_events: int = 30           # Minutes before major events


# ─────────────────────────────────────────────────────────────────────────────
# MARKET HOURS (IST)
# ─────────────────────────────────────────────────────────────────────────────

MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"
PRE_MARKET_START = "09:00"
PRE_MARKET_END = "09:15"


# ─────────────────────────────────────────────────────────────────────────────
# DATA PATHS
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "historical")
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
TRADES_DB = os.path.join(os.path.dirname(__file__), "..", "data", "trades.json")
BACKTEST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "backtest_results")
