"""
=============================================================================
SCALPER PRO v2 — NIFTY Walk-Forward Backtest Engine
=============================================================================
Simulates the full 3-layer algorithm on real Dhan historical data.

Algorithm per bar:
  1. Mark S/R levels from prior daily history (no lookahead)
  2. Check NIFTY proximity to any level (Layer 1)
  3. If AT_LEVEL or APPROACHING → generate CE/PE signal
  4. Apply Greek-based strike selection (Layer 3, simplified)
  5. Simulate option trade on subsequent bars
  6. P&L = index_move × delta × lot_size (delta 0.70 for ITM swing)

Runs 4 periods: today (15min), 1-month (15min), 6-month (daily), 1-year (daily)
=============================================================================
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — NIFTY ONLY
# ─────────────────────────────────────────────────────────────────────────────
LOT_SIZE          = 75     # NIFTY lot size
DELTA             = 0.70   # ITM option delta approximation for swing
SWING_TARGET      = 120    # Daily: index points target (RR 1:3)
SWING_SL          = 40     # Daily: index points SL
INTRADAY_TARGET   = 30     # 15-min backtest: index points target
INTRADAY_SL       = 15     # 15-min backtest: index points SL
MAX_DAILY_BARS    = 7      # Max bars to hold on daily chart
MAX_INTRA_BARS    = 12     # Max 15-min bars to hold (3 hours)
NO_TRADE_BEFORE   = "09:45"
NO_TRADE_AFTER    = "15:00"
MIN_LEVEL_TOUCHES = 2      # Minimum touches (2 = valid cluster)
MAX_LEVEL_TOUCHES = 4      # Levels tested 4+ times tend to break, not bounce
MIN_LEVEL_STRENGTH = 0.30  # Skip very weak levels
EMA_PERIOD        = 20     # For trend filter
LEVEL_LOOKBACK    = 252    # Trading days of history
MIN_HISTORY_BARS  = 60


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    trade_id:        str
    period_label:    str
    entry_date:      str
    entry_time:      str
    entry_bar_close: float   # spot at signal bar close
    entry_spot:      float   # spot at entry (next bar open)
    direction:       str     # CE or PE
    level_type:      str     # SUPPORT or RESISTANCE
    level_price:     float   # the S/R level traded against
    level_strength:  float
    level_touches:   int
    level_tf:        str     # WEEKLY/DAILY/HOURLY/15MIN
    proximity_zone:  str     # AT_LEVEL / APPROACHING
    distance_atr:    float
    target_spot:     float   # spot target
    sl_spot:         float   # spot SL
    exit_spot:       float   = 0.0
    exit_date:       str     = ""
    exit_time:       str     = ""
    exit_reason:     str     = ""   # TARGET_HIT / SL_HIT / EOD / MAX_HOLD
    holding_bars:    int     = 0
    index_pnl_pts:   float   = 0.0  # index move points
    option_pnl_pts:  float   = 0.0  # index_pnl × delta
    pnl_rupees:      float   = 0.0  # option_pnl × lot_size
    is_winner:       bool    = False
    setup_quality:   str     = ""   # A+ / A / B / C
    failure_reason:  str     = ""   # analysis of why trade failed
    # Indicator scores at entry
    conf_score:      int     = 0    # total indicator score (0-7)
    conf_breakdown:  str     = ""   # e.g. "5/7: [5EMA, 13EMA, ST, RSI(45), VOL(1.2x), BB(0.28)]"
    rsi_at_entry:    float   = 0.0
    supertrend_bull: bool    = False
    bb_pct_b:        float   = 0.5


@dataclass
class PeriodResult:
    label:            str     # "Today", "1-Month", "6-Month", "1-Year"
    start_date:       str
    end_date:         str
    timeframe:        str     # "15min" or "Daily"
    total_bars:       int     = 0
    signal_bars:      int     = 0
    total_trades:     int     = 0
    winners:          int     = 0
    losers:           int     = 0
    breakeven:        int     = 0
    win_rate:         float   = 0.0
    gross_profit:     float   = 0.0
    gross_loss:       float   = 0.0
    total_pnl:        float   = 0.0
    avg_win:          float   = 0.0
    avg_loss:         float   = 0.0
    profit_factor:    float   = 0.0
    sharpe_ratio:     float   = 0.0
    max_drawdown:     float   = 0.0
    max_win_streak:   int     = 0
    max_loss_streak:  int     = 0
    avg_hold_bars:    float   = 0.0
    best_trade_pnl:   float   = 0.0
    worst_trade_pnl:  float   = 0.0
    monthly_pnl:      Dict[str, float]    = field(default_factory=dict)
    equity_curve:     List[float]         = field(default_factory=list)
    trades:           List[BacktestTrade] = field(default_factory=list)
    failure_analysis: Dict[str, int]      = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class NiftyBacktest:
    """
    Walk-forward NIFTY backtest using real Dhan data.

    Usage:
        bt = NiftyBacktest(data_fetcher)
        results = bt.run_all()
        # results is List[PeriodResult] for [today, 1m, 6m, 1y]
    """

    def __init__(self, data_fetcher):
        self._fetcher = data_fetcher
        self._daily_df: Optional[pd.DataFrame] = None
        self._fifteen_min_df: Optional[pd.DataFrame] = None

    def fetch_all_data(self):
        """Fetch all required data once upfront.

        Data priority:
          Daily:   yfinance (2 years) — Dhan /charts/historical endpoint is
                   intraday-only under the current Data API subscription.
          15-min:  Dhan intraday (90 days) — works perfectly.
        """
        logger.info("[BACKTEST] Fetching NIFTY data...")

        # ── Daily: yfinance primary (Dhan historical requires separate sub) ──
        try:
            from scalper.data.free_fetcher import FreeDataFetcher
            free = FreeDataFetcher()
            self._daily_df = free.fetch_daily("NIFTY", period="2y")
            if self._daily_df is not None and "datetime" not in self._daily_df.columns:
                self._daily_df = self._daily_df.reset_index()
                if "Date" in self._daily_df.columns:
                    self._daily_df = self._daily_df.rename(columns={"Date": "datetime"})
        except Exception as e:
            logger.warning(f"[BACKTEST] yfinance daily failed: {e}")
            self._daily_df = None

        # Fallback: Dhan daily
        if self._daily_df is None or len(self._daily_df) < MIN_HISTORY_BARS:
            logger.info("[BACKTEST] Trying Dhan daily API...")
            self._daily_df = self._fetcher.fetch_daily_data("NIFTY", days_back=500)

        if self._daily_df is None or len(self._daily_df) < MIN_HISTORY_BARS:
            raise RuntimeError(
                f"Insufficient daily data ({len(self._daily_df) if self._daily_df is not None else 0} bars). "
                "Install yfinance: pip install yfinance"
            )

        self._daily_df["datetime"] = pd.to_datetime(self._daily_df["datetime"])
        logger.info(f"[BACKTEST] Daily data: {len(self._daily_df)} bars  "
                    f"({self._daily_df['datetime'].iloc[0].date()} → "
                    f"{self._daily_df['datetime'].iloc[-1].date()})")

        # ── 15-min: Dhan intraday (90 days) ──────────────────────────────
        self._fifteen_min_df = self._fetcher.fetch_index_data("NIFTY", interval="15", days_back=90)
        if self._fifteen_min_df is None or len(self._fifteen_min_df) < 50:
            logger.warning("[BACKTEST] Insufficient 15-min data — intraday periods may have few trades")
        else:
            self._fifteen_min_df["datetime"] = pd.to_datetime(self._fifteen_min_df["datetime"])
            logger.info(f"[BACKTEST] 15-min data: {len(self._fifteen_min_df)} bars  "
                        f"({self._fifteen_min_df['datetime'].iloc[0].date()} → "
                        f"{self._fifteen_min_df['datetime'].iloc[-1].date()})")

    def run_all(self) -> List[PeriodResult]:
        """Run all 4 backtest periods."""
        self.fetch_all_data()

        # Suppress verbose per-bar logging from index_levels during backtest
        logging.getLogger("scalper.core.index_levels").setLevel(logging.WARNING)
        logging.getLogger("scalper.core.premarket_analysis").setLevel(logging.WARNING)

        results = []

        today = datetime.now().date()

        # Today — 15-min bars
        logger.info("[BACKTEST] Period: TODAY (15-min)")
        r_today = self._run_intraday(
            label="Today",
            start_date=today,
            end_date=today,
        )
        results.append(r_today)

        # 1 Month — 15-min bars
        logger.info("[BACKTEST] Period: 1-MONTH (15-min)")
        r_1m = self._run_intraday(
            label="1-Month",
            start_date=today - timedelta(days=30),
            end_date=today,
        )
        results.append(r_1m)

        # 6 Months — daily bars
        logger.info("[BACKTEST] Period: 6-MONTHS (daily)")
        r_6m = self._run_daily(
            label="6-Month",
            start_date=today - timedelta(days=180),
            end_date=today,
        )
        results.append(r_6m)

        # 1 Year — daily bars
        logger.info("[BACKTEST] Period: 1-YEAR (daily)")
        r_1y = self._run_daily(
            label="1-Year",
            start_date=today - timedelta(days=365),
            end_date=today,
        )
        results.append(r_1y)

        # Analyze failures across all periods
        for r in results:
            r.failure_analysis = self._analyze_failures(r.trades)

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # INTRADAY (15-min) BACKTEST
    # ─────────────────────────────────────────────────────────────────────────

    def _run_intraday(self, label: str, start_date: date, end_date: date) -> PeriodResult:
        """Walk-forward backtest on 15-min bars."""
        from scalper.core.index_levels import IndexLevelMarker
        from scalper.core.indicators import score_signal, MIN_CONF_SCORE

        df15 = self._fifteen_min_df
        daily = self._daily_df

        if df15 is None or len(df15) < 50:
            return self._empty_result(label, str(start_date), str(end_date), "15min")

        # Filter 15-min data to date range
        mask = (df15["datetime"].dt.date >= start_date) & (df15["datetime"].dt.date <= end_date)
        df_period = df15[mask].reset_index(drop=True)

        if len(df_period) < 5:
            return self._empty_result(label, str(start_date), str(end_date), "15min")

        trades: List[BacktestTrade] = []
        trade_counter = 0
        signal_bars = 0
        open_trade: Optional[BacktestTrade] = None
        daily_trade_count = 0
        last_trade_date = None

        # Pre-mark levels from all available daily data (no rolling for intraday)
        marker = IndexLevelMarker()
        marker.mark_levels(
            daily_df=daily,
            index="NIFTY",
        )
        daily_atr = marker._daily_atr if marker._daily_atr > 0 else 200.0

        for i in range(5, len(df_period)):
            row = df_period.iloc[i]
            bar_dt = row["datetime"]
            bar_date = bar_dt.date()
            bar_time = bar_dt.strftime("%H:%M")

            # Reset daily trade count
            if bar_date != last_trade_date:
                daily_trade_count = 0
                last_trade_date = bar_date

            # Time filter
            if bar_time < NO_TRADE_BEFORE or bar_time > NO_TRADE_AFTER:
                # Still check exits
                if open_trade:
                    open_trade = self._check_intraday_exit(open_trade, row, bar_dt, bar_time)
                    if open_trade.exit_reason:
                        self._finalize_trade(open_trade)
                        trades.append(open_trade)
                        open_trade = None
                continue

            # Check exit for open trade
            if open_trade:
                open_trade.holding_bars += 1
                open_trade = self._check_intraday_exit(open_trade, row, bar_dt, bar_time)
                if open_trade.exit_reason:
                    self._finalize_trade(open_trade)
                    trades.append(open_trade)
                    open_trade = None

            if open_trade is not None:
                continue

            # Risk limit: 2 trades per day for intraday
            if daily_trade_count >= 2:
                continue

            # Level proximity check
            spot = float(row["close"])
            proximity = marker.check_proximity(spot, "NIFTY")

            if proximity.proximity_zone not in ("AT_LEVEL", "APPROACHING"):
                continue
            if proximity.nearest_level is None:
                continue

            lv = proximity.nearest_level
            if lv.touches < MIN_LEVEL_TOUCHES or lv.touches > MAX_LEVEL_TOUCHES:
                continue
            if lv.strength < MIN_LEVEL_STRENGTH:
                continue

            # EMA20 trend filter on daily data
            ema20_daily = float(daily["close"].ewm(span=EMA_PERIOD, adjust=False).mean().iloc[-1])
            trend_up = spot > ema20_daily

            direction = proximity.direction
            if direction not in ("CE", "PE"):
                continue
            if direction == "CE" and not trend_up:
                continue
            if direction == "PE" and trend_up:
                continue

            # ── Multi-indicator scoring ───────────────────────────────────────
            # Use all bars available up to and including current bar i
            # Map period bar index back to full df15 for indicator history
            global_i = df15[df15["datetime"] == row["datetime"]].index
            if len(global_i) == 0:
                continue
            global_idx = global_i[0]
            sig_score = score_signal(
                df=df15,
                bar_idx=global_idx,
                direction=direction,
                proximity_zone=proximity.proximity_zone,
                india_vix=0.0,   # not available in historical backtest
                oi_bull=None,    # not available in historical backtest
            )
            if not sig_score.trade_allowed:
                continue

            signal_bars += 1

            # Entry on next bar
            if i + 1 >= len(df_period):
                continue
            next_bar = df_period.iloc[i + 1]
            entry_spot = float(next_bar["open"])

            if direction == "CE":
                target_spot = entry_spot + INTRADAY_TARGET
                sl_spot     = entry_spot - INTRADAY_SL
            else:
                target_spot = entry_spot - INTRADAY_TARGET
                sl_spot     = entry_spot + INTRADAY_SL

            trade_counter += 1
            quality = self._grade_setup(proximity, sig_score)
            trade = BacktestTrade(
                trade_id        = f"IT-{trade_counter:04d}",
                period_label    = label,
                entry_date      = bar_date.strftime("%Y-%m-%d"),
                entry_time      = next_bar["datetime"].strftime("%H:%M"),
                entry_bar_close = spot,
                entry_spot      = entry_spot,
                direction       = direction,
                level_type      = proximity.nearest_level.level_type.value,
                level_price     = proximity.nearest_level.price,
                level_strength  = proximity.nearest_level.strength,
                level_touches   = proximity.nearest_level.touches,
                level_tf        = proximity.nearest_level.timeframe.value,
                proximity_zone  = proximity.proximity_zone,
                distance_atr    = proximity.distance_atr,
                target_spot     = target_spot,
                sl_spot         = sl_spot,
                setup_quality   = quality,
                conf_score      = sig_score.total_score,
                conf_breakdown  = sig_score.score_breakdown,
                rsi_at_entry    = sig_score.rsi,
                supertrend_bull = sig_score.supertrend_bull,
                bb_pct_b        = sig_score.bb_pct_b,
            )
            open_trade = trade
            daily_trade_count += 1

        # Close any remaining open trade
        if open_trade and not open_trade.exit_reason:
            last_row = df_period.iloc[-1]
            open_trade.exit_spot = float(last_row["close"])
            open_trade.exit_date = last_row["datetime"].strftime("%Y-%m-%d")
            open_trade.exit_time = last_row["datetime"].strftime("%H:%M")
            open_trade.exit_reason = "EOD"
            self._finalize_trade(open_trade)
            trades.append(open_trade)

        return self._compile_result(
            label, str(start_date), str(end_date), "15min",
            len(df_period), signal_bars, trades
        )

    def _check_intraday_exit(self, trade: BacktestTrade, row: pd.Series,
                              bar_dt: datetime, bar_time: str) -> BacktestTrade:
        """Check SL/target/EOD/max-hold for intraday trade."""
        if trade.exit_reason:
            return trade

        high  = float(row["high"])
        low   = float(row["low"])
        close = float(row["close"])
        dt_str = bar_dt.strftime("%Y-%m-%d")
        tm_str = bar_dt.strftime("%H:%M")

        if trade.direction == "CE":
            if low <= trade.sl_spot:
                trade.exit_spot = trade.sl_spot
                trade.exit_reason = "SL_HIT"
            elif high >= trade.target_spot:
                trade.exit_spot = trade.target_spot
                trade.exit_reason = "TARGET_HIT"
        else:
            if high >= trade.sl_spot:
                trade.exit_spot = trade.sl_spot
                trade.exit_reason = "SL_HIT"
            elif low <= trade.target_spot:
                trade.exit_spot = trade.target_spot
                trade.exit_reason = "TARGET_HIT"

        # EOD force-close
        if not trade.exit_reason and bar_time >= "15:25":
            trade.exit_spot = close
            trade.exit_reason = "EOD"

        # Max hold
        if not trade.exit_reason and trade.holding_bars >= MAX_INTRA_BARS:
            trade.exit_spot = close
            trade.exit_reason = "MAX_HOLD"

        if trade.exit_reason:
            trade.exit_date = dt_str
            trade.exit_time = tm_str

        return trade

    # ─────────────────────────────────────────────────────────────────────────
    # DAILY BACKTEST
    # ─────────────────────────────────────────────────────────────────────────

    def _run_daily(self, label: str, start_date: date, end_date: date) -> PeriodResult:
        """Walk-forward backtest on daily bars using rolling S/R window."""
        from scalper.core.index_levels import IndexLevelMarker

        daily = self._daily_df.copy().reset_index(drop=True)

        trades: List[BacktestTrade] = []
        trade_counter = 0
        signal_bars = 0
        open_trade: Optional[BacktestTrade] = None

        for i in range(LEVEL_LOOKBACK, len(daily)):
            bar_dt   = daily.iloc[i]["datetime"]
            bar_date = bar_dt.date()

            if bar_date < start_date:
                continue
            if bar_date > end_date:
                break

            # Check exit for open trade using this bar's high/low
            if open_trade:
                open_trade.holding_bars += 1
                open_trade = self._check_daily_exit(open_trade, daily.iloc[i])
                if open_trade.exit_reason:
                    self._finalize_trade(open_trade)
                    trades.append(open_trade)
                    open_trade = None

            if open_trade is not None:
                continue

            # Mark levels from rolling window ending at i-1 (no lookahead)
            hist = daily.iloc[max(0, i - LEVEL_LOOKBACK): i]
            if len(hist) < MIN_HISTORY_BARS:
                continue

            marker = IndexLevelMarker()
            marker.mark_levels(daily_df=hist, index="NIFTY")

            spot = float(daily.iloc[i]["close"])
            proximity = marker.check_proximity(spot, "NIFTY")

            if proximity.proximity_zone not in ("AT_LEVEL", "APPROACHING"):
                continue
            if proximity.nearest_level is None:
                continue

            lv = proximity.nearest_level

            # Level quality filters
            if lv.touches < MIN_LEVEL_TOUCHES:
                continue
            if lv.touches > MAX_LEVEL_TOUCHES:
                # 4+ touch levels tend to break through on next test
                continue
            if lv.strength < MIN_LEVEL_STRENGTH:
                continue

            # ── EMA20 TREND FILTER ──────────────────────────────────────
            # Only trade WITH the trend:
            #   Price > EMA20 (uptrend) → only buy CE at support
            #   Price < EMA20 (downtrend) → only buy PE at resistance
            ema20 = float(hist["close"].ewm(span=EMA_PERIOD, adjust=False).mean().iloc[-1])
            trend_up = spot > ema20

            direction = proximity.direction
            if direction not in ("CE", "PE"):
                continue

            # Skip counter-trend trades
            if direction == "CE" and not trend_up:
                continue   # Don't buy CE (bullish) when trend is down
            if direction == "PE" and trend_up:
                continue   # Don't buy PE (bearish) when trend is up

            # ── Rejection candle confirmation ─────────────────────────
            # For CE (at support): bar's low must be near the level
            #   AND close must be ABOVE the level (showing rejection/bounce)
            # For PE (at resistance): bar's high must be near the level
            #   AND close must be BELOW the level (showing rejection)
            bar_open  = float(daily.iloc[i]["open"])
            bar_high  = float(daily.iloc[i]["high"])
            bar_low   = float(daily.iloc[i]["low"])
            bar_close = float(daily.iloc[i]["close"])
            daily_atr = marker._daily_atr if marker._daily_atr > 0 else 200.0

            if direction == "CE":
                # Low wick near support, close above support
                low_near_level = abs(bar_low - lv.price) < daily_atr * 0.20
                close_above    = bar_close > lv.price
                if not (low_near_level and close_above):
                    continue
            else:
                # High wick near resistance, close below resistance
                high_near_level = abs(bar_high - lv.price) < daily_atr * 0.20
                close_below     = bar_close < lv.price
                if not (high_near_level and close_below):
                    continue

            signal_bars += 1

            # Entry: next bar's open
            if i + 1 >= len(daily):
                continue
            next_bar   = daily.iloc[i + 1]
            entry_spot = float(next_bar["open"])

            if direction == "CE":
                target_spot = entry_spot + SWING_TARGET
                sl_spot     = entry_spot - SWING_SL
            else:
                target_spot = entry_spot - SWING_TARGET
                sl_spot     = entry_spot + SWING_SL

            trade_counter += 1
            quality = self._grade_setup(proximity)
            trade = BacktestTrade(
                trade_id       = f"DY-{trade_counter:04d}",
                period_label   = label,
                entry_date     = bar_date.strftime("%Y-%m-%d"),
                entry_time     = "09:15",
                entry_bar_close= spot,
                entry_spot     = entry_spot,
                direction      = direction,
                level_type     = proximity.nearest_level.level_type.value,
                level_price    = proximity.nearest_level.price,
                level_strength = proximity.nearest_level.strength,
                level_touches  = proximity.nearest_level.touches,
                level_tf       = proximity.nearest_level.timeframe.value,
                proximity_zone = proximity.proximity_zone,
                distance_atr   = proximity.distance_atr,
                target_spot    = target_spot,
                sl_spot        = sl_spot,
                setup_quality  = quality,
            )
            open_trade = trade

        # Close any remaining
        if open_trade and not open_trade.exit_reason:
            last_bar = daily.iloc[-1]
            open_trade.exit_spot   = float(last_bar["close"])
            open_trade.exit_date   = last_bar["datetime"].strftime("%Y-%m-%d")
            open_trade.exit_time   = "15:30"
            open_trade.exit_reason = "EOD"
            self._finalize_trade(open_trade)
            trades.append(open_trade)

        return self._compile_result(
            label, str(start_date), str(end_date), "Daily",
            len(daily[daily["datetime"].dt.date.between(start_date, end_date)]),
            signal_bars, trades
        )

    def _check_daily_exit(self, trade: BacktestTrade, row: pd.Series) -> BacktestTrade:
        """Check SL/target/max-hold on a daily bar."""
        if trade.exit_reason:
            return trade

        high  = float(row["high"])
        low   = float(row["low"])
        close = float(row["close"])
        dt_str = row["datetime"].strftime("%Y-%m-%d")

        if trade.direction == "CE":
            if low <= trade.sl_spot:
                trade.exit_spot = trade.sl_spot
                trade.exit_reason = "SL_HIT"
            elif high >= trade.target_spot:
                trade.exit_spot = trade.target_spot
                trade.exit_reason = "TARGET_HIT"
        else:
            if high >= trade.sl_spot:
                trade.exit_spot = trade.sl_spot
                trade.exit_reason = "SL_HIT"
            elif low <= trade.target_spot:
                trade.exit_spot = trade.target_spot
                trade.exit_reason = "TARGET_HIT"

        if not trade.exit_reason and trade.holding_bars >= MAX_DAILY_BARS:
            trade.exit_spot   = close
            trade.exit_reason = "MAX_HOLD"

        if trade.exit_reason:
            trade.exit_date = dt_str
            trade.exit_time = "15:30"

        return trade

    # ─────────────────────────────────────────────────────────────────────────
    # P&L + GRADES
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _finalize_trade(trade: BacktestTrade):
        """Compute P&L and winner flag."""
        if trade.direction == "CE":
            trade.index_pnl_pts = trade.exit_spot - trade.entry_spot
        else:
            trade.index_pnl_pts = trade.entry_spot - trade.exit_spot

        trade.option_pnl_pts = round(trade.index_pnl_pts * DELTA, 2)
        trade.pnl_rupees     = round(trade.option_pnl_pts * LOT_SIZE, 2)
        trade.is_winner      = trade.pnl_rupees > 0

        if not trade.is_winner and trade.pnl_rupees <= 0:
            trade.failure_reason = NiftyBacktest._diagnose_failure(trade)

    @staticmethod
    def _diagnose_failure(trade: BacktestTrade) -> str:
        """Classify why a trade failed."""
        if trade.exit_reason == "SL_HIT":
            if trade.distance_atr > 0.6:
                return "Entered too far from level (APPROACHING zone, low probability)"
            elif trade.level_touches <= 2:
                return "Weak level (only 2 touches — insufficient confirmation)"
            elif trade.level_strength < 0.3:
                return "Low-strength level — below 15MIN timeframe"
            else:
                return "SL hit — valid setup but adverse price action"
        elif trade.exit_reason == "MAX_HOLD":
            return "Position held too long — no clear breakout in direction"
        elif trade.exit_reason in ("EOD", "EXPIRED"):
            return "Trade not resolved within session"
        return "Breakeven or minor loss"

    @staticmethod
    def _grade_setup(proximity, sig_score=None) -> str:
        """Grade setup quality A+ / A / B / C.
        Now incorporates multi-indicator confirmation score."""
        lv = proximity.nearest_level
        pts = 0
        if proximity.proximity_zone == "AT_LEVEL":
            pts += 3
        elif proximity.proximity_zone == "APPROACHING":
            pts += 1
        pts += min(lv.touches, 5)
        pts += int(lv.strength * 4)
        if lv.timeframe.value in ("DAILY", "WEEKLY"):
            pts += 2
        elif lv.timeframe.value == "HOURLY":
            pts += 1
        if lv.is_round_number:
            pts += 1
        if lv.fib_confluence:
            pts += 1
        if lv.pivot_confluence:
            pts += 1
        # Bonus from multi-indicator confirmation score
        if sig_score is not None:
            pts += sig_score.total_score  # 0-7 additional points

        if pts >= 16:
            return "A+"
        elif pts >= 12:
            return "A"
        elif pts >= 8:
            return "B"
        else:
            return "C"

    # ─────────────────────────────────────────────────────────────────────────
    # COMPILE RESULTS
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compile_result(
        label: str, start_date: str, end_date: str, timeframe: str,
        total_bars: int, signal_bars: int, trades: List[BacktestTrade],
    ) -> PeriodResult:
        r = PeriodResult(
            label=label, start_date=start_date, end_date=end_date,
            timeframe=timeframe, total_bars=total_bars,
            signal_bars=signal_bars, total_trades=len(trades),
            trades=trades,
        )

        if not trades:
            return r

        pnls = [t.pnl_rupees for t in trades]
        r.winners    = sum(1 for p in pnls if p > 0)
        r.losers     = sum(1 for p in pnls if p < 0)
        r.breakeven  = sum(1 for p in pnls if p == 0)
        r.win_rate   = round(r.winners / r.total_trades * 100, 1) if r.total_trades else 0

        wins  = [p for p in pnls if p > 0]
        losses= [p for p in pnls if p < 0]
        r.gross_profit = round(sum(wins), 2)
        r.gross_loss   = round(abs(sum(losses)), 2)
        r.total_pnl    = round(sum(pnls), 2)
        r.avg_win      = round(np.mean(wins), 2) if wins else 0
        r.avg_loss     = round(np.mean(losses), 2) if losses else 0
        r.profit_factor= round(r.gross_profit / r.gross_loss, 2) if r.gross_loss > 0 else 999.0
        r.best_trade_pnl  = max(pnls)
        r.worst_trade_pnl = min(pnls)

        # Equity curve & max drawdown
        equity = [0.0]
        for p in pnls:
            equity.append(equity[-1] + p)
        r.equity_curve = [round(e, 2) for e in equity]
        peak, max_dd = equity[0], 0.0
        for v in equity:
            if v > peak:
                peak = v
            dd = peak - v
            if dd > max_dd:
                max_dd = dd
        r.max_drawdown = round(max_dd, 2)

        # Sharpe
        arr = np.array(pnls)
        r.sharpe_ratio = round(
            (np.mean(arr) / np.std(arr)) * np.sqrt(252)
            if len(arr) > 1 and np.std(arr) > 0 else 0.0, 2
        )

        # Streaks
        cur, max_w, max_l = 0, 0, 0
        for p in pnls:
            if p > 0:
                cur = cur + 1 if cur > 0 else 1
                max_w = max(max_w, cur)
            elif p < 0:
                cur = cur - 1 if cur < 0 else -1
                max_l = max(max_l, abs(cur))
            else:
                cur = 0
        r.max_win_streak  = max_w
        r.max_loss_streak = max_l
        r.avg_hold_bars   = round(np.mean([t.holding_bars for t in trades]), 1)

        # Monthly P&L
        monthly: Dict[str, float] = {}
        for t in trades:
            key = t.entry_date[:7]
            monthly[key] = monthly.get(key, 0.0) + t.pnl_rupees
        r.monthly_pnl = {k: round(v, 2) for k, v in sorted(monthly.items())}

        return r

    @staticmethod
    def _empty_result(label, start, end, timeframe) -> PeriodResult:
        return PeriodResult(
            label=label, start_date=start, end_date=end,
            timeframe=timeframe, total_bars=0, signal_bars=0,
            total_trades=0, winners=0, losers=0, breakeven=0,
            win_rate=0, gross_profit=0, gross_loss=0, total_pnl=0,
            avg_win=0, avg_loss=0, profit_factor=0, sharpe_ratio=0,
            max_drawdown=0, max_win_streak=0, max_loss_streak=0,
            avg_hold_bars=0, best_trade_pnl=0, worst_trade_pnl=0,
        )

    @staticmethod
    def _analyze_failures(trades: List[BacktestTrade]) -> Dict[str, int]:
        """Group losing trades by failure reason."""
        failure_map: Dict[str, int] = {}
        for t in trades:
            if not t.is_winner and t.failure_reason:
                failure_map[t.failure_reason] = failure_map.get(t.failure_reason, 0) + 1
        return dict(sorted(failure_map.items(), key=lambda x: -x[1]))


# ─────────────────────────────────────────────────────────────────────────────
# S/R LEVEL ANALYSIS (2-4 year chart)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_long_term_levels(data_fetcher) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mark all NIFTY S/R levels on 2-4 year daily chart.
    Returns (levels_df, summary_df) for Excel export.
    """
    from scalper.core.index_levels import IndexLevelMarker

    logger.info("[LEVELS] Fetching 2-year NIFTY daily data for long-term level analysis...")
    daily = None
    try:
        from scalper.data.free_fetcher import FreeDataFetcher
        free = FreeDataFetcher()
        daily = free.fetch_daily("NIFTY", period="2y")
        if daily is not None and "datetime" not in daily.columns:
            daily = daily.reset_index()
            if "Date" in daily.columns:
                daily = daily.rename(columns={"Date": "datetime"})
    except Exception:
        pass
    if daily is None or len(daily) < 60:
        daily = data_fetcher.fetch_daily_data("NIFTY", days_back=500)
    if daily is None or len(daily) < 60:
        logger.warning("[LEVELS] Insufficient data for long-term analysis")
        return pd.DataFrame(), pd.DataFrame()

    daily["datetime"] = pd.to_datetime(daily["datetime"])

    # Resample to weekly for macro view
    weekly = (daily.set_index("datetime")
              .resample("W-FRI")
              .agg(open=("open","first"), high=("high","max"),
                   low=("low","min"), close=("close","last"), volume=("volume","sum"))
              .dropna(subset=["close"])
              .reset_index())

    marker = IndexLevelMarker()
    marker.mark_levels(
        daily_df=daily,
        weekly_df=weekly,
        index="NIFTY",
    )

    current_spot = float(daily["close"].iloc[-1])
    daily_atr    = marker._daily_atr

    rows = []
    for lv in marker.levels:
        dist_pts = current_spot - lv.price
        dist_atr = abs(dist_pts) / max(daily_atr, 1)
        proximity = "ABOVE" if dist_pts < 0 else "BELOW"
        tags = []
        if lv.is_round_number:   tags.append("Round#")
        if lv.fib_confluence:    tags.append("Fib")
        if lv.pivot_confluence:  tags.append("CPR/Pivot")
        rows.append({
            "Level":          round(lv.price, 2),
            "Type":           lv.level_type.value,
            "Timeframe":      lv.timeframe.value,
            "Touches":        lv.touches,
            "Strength":       round(lv.strength, 3),
            "Current Pos":    proximity,
            "Dist Points":    round(abs(dist_pts), 1),
            "Dist ATR":       round(dist_atr, 2),
            "First Touch":    lv.first_touch_date,
            "Last Touch":     lv.last_touch_date,
            "Tags":           " | ".join(tags),
            "Round Number":   lv.is_round_number,
            "Fib Level":      lv.fib_confluence,
            "CPR/Pivot":      lv.pivot_confluence,
        })

    levels_df = pd.DataFrame(rows).sort_values("Level", ascending=False)

    # Summary stats
    summary_rows = [
        {"Metric": "Current NIFTY",      "Value": f"{current_spot:,.2f}"},
        {"Metric": "Daily ATR",           "Value": f"{daily_atr:.1f} pts"},
        {"Metric": "Total Levels Marked", "Value": len(levels_df)},
        {"Metric": "Support Levels",      "Value": len(levels_df[levels_df["Type"]=="SUPPORT"])},
        {"Metric": "Resistance Levels",   "Value": len(levels_df[levels_df["Type"]=="RESISTANCE"])},
        {"Metric": "Weekly Levels",       "Value": len(levels_df[levels_df["Timeframe"]=="WEEKLY"])},
        {"Metric": "Daily Levels",        "Value": len(levels_df[levels_df["Timeframe"]=="DAILY"])},
        {"Metric": "High-Strength (>0.6)","Value": len(levels_df[levels_df["Strength"]>0.6])},
        {"Metric": "Round Numbers",       "Value": levels_df["Round Number"].sum()},
        {"Metric": "Fib Confluences",     "Value": levels_df["Fib Level"].sum()},
        {"Metric": "CPR Confluences",     "Value": levels_df["CPR/Pivot"].sum()},
        {"Metric": "Nearest Support",     "Value": f"{levels_df[levels_df['Type']=='SUPPORT']['Level'].max():,.0f}" if len(levels_df[levels_df['Type']=='SUPPORT']) else "N/A"},
        {"Metric": "Nearest Resistance",  "Value": f"{levels_df[levels_df['Type']=='RESISTANCE']['Level'].min():,.0f}" if len(levels_df[levels_df['Type']=='RESISTANCE']) else "N/A"},
        {"Metric": "Data Range",          "Value": f"{daily['datetime'].iloc[0].date()} to {daily['datetime'].iloc[-1].date()}"},
    ]
    summary_df = pd.DataFrame(summary_rows)

    return levels_df, summary_df
