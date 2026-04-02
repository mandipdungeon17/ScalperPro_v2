"""
=============================================================================
SCALPER PRO - Technical Indicators Engine
=============================================================================
All indicators computed on pandas DataFrames with OHLCV columns.
Optimized for 1-min to 15-min timeframes.
=============================================================================
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class IndicatorEngine:
    """Computes all technical indicators needed for the scalping/swing system."""

    # ── EMA ────────────────────────────────────────────────────────────────
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def ema_crossover(df: pd.DataFrame, fast: int = 5, mid: int = 9, slow: int = 20) -> pd.DataFrame:
        """Compute EMA crossovers. Returns df with ema_fast, ema_mid, ema_slow, and signals."""
        df = df.copy()
        df["ema_fast"] = IndicatorEngine.ema(df["close"], fast)
        df["ema_mid"] = IndicatorEngine.ema(df["close"], mid)
        df["ema_slow"] = IndicatorEngine.ema(df["close"], slow)

        # Crossover signals
        df["ema_bull_cross"] = (
            (df["ema_fast"] > df["ema_mid"]) &
            (df["ema_fast"].shift(1) <= df["ema_mid"].shift(1))
        )
        df["ema_bear_cross"] = (
            (df["ema_fast"] < df["ema_mid"]) &
            (df["ema_fast"].shift(1) >= df["ema_mid"].shift(1))
        )
        # Trend alignment: fast > mid > slow = strong bull
        df["ema_trend_bull"] = (df["ema_fast"] > df["ema_mid"]) & (df["ema_mid"] > df["ema_slow"])
        df["ema_trend_bear"] = (df["ema_fast"] < df["ema_mid"]) & (df["ema_mid"] < df["ema_slow"])

        return df

    # ── VWAP ───────────────────────────────────────────────────────────────
    @staticmethod
    def vwap(df: pd.DataFrame, std_1: float = 1.0, std_2: float = 2.0) -> pd.DataFrame:
        """
        Compute VWAP and bands. Resets daily.
        Expects: datetime index or 'datetime' column, and 'high','low','close','volume'.
        """
        df = df.copy()
        if "datetime" in df.columns:
            df["_date"] = pd.to_datetime(df["datetime"]).dt.date
        else:
            df["_date"] = df.index.date

        df["_tp"] = (df["high"] + df["low"] + df["close"]) / 3
        df["_tp_vol"] = df["_tp"] * df["volume"]

        # Cumulative within each day
        df["_cum_vol"] = df.groupby("_date")["volume"].cumsum()
        df["_cum_tp_vol"] = df.groupby("_date")["_tp_vol"].cumsum()

        df["vwap"] = df["_cum_tp_vol"] / df["_cum_vol"]

        # VWAP Standard Deviation bands
        df["_tp_diff_sq"] = (df["_tp"] - df["vwap"]) ** 2
        df["_cum_tp_diff_sq"] = df.groupby("_date")["_tp_diff_sq"].cumsum()
        df["_count"] = df.groupby("_date").cumcount() + 1
        df["_vwap_std"] = np.sqrt(df["_cum_tp_diff_sq"] / df["_count"])

        df["vwap_upper_1"] = df["vwap"] + std_1 * df["_vwap_std"]
        df["vwap_lower_1"] = df["vwap"] - std_1 * df["_vwap_std"]
        df["vwap_upper_2"] = df["vwap"] + std_2 * df["_vwap_std"]
        df["vwap_lower_2"] = df["vwap"] - std_2 * df["_vwap_std"]

        # Price relative to VWAP
        df["above_vwap"] = df["close"] > df["vwap"]

        # Cleanup temp columns
        df.drop(columns=[c for c in df.columns if c.startswith("_")], inplace=True)
        return df

    # ── RSI ─────────────────────────────────────────────────────────────────
    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 7) -> pd.DataFrame:
        df = df.copy()
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    # ── Stochastic RSI ─────────────────────────────────────────────────────
    @staticmethod
    def stochastic_rsi(df: pd.DataFrame, rsi_period: int = 14,
                       k_period: int = 3, d_period: int = 3) -> pd.DataFrame:
        df = df.copy()
        if "rsi" not in df.columns:
            df = IndicatorEngine.rsi(df, rsi_period)

        rsi_min = df["rsi"].rolling(window=rsi_period).min()
        rsi_max = df["rsi"].rolling(window=rsi_period).max()
        rsi_range = rsi_max - rsi_min

        df["stoch_rsi_k"] = ((df["rsi"] - rsi_min) / rsi_range.replace(0, np.nan)) * 100
        df["stoch_rsi_d"] = df["stoch_rsi_k"].rolling(window=d_period).mean()
        return df

    # ── MACD ────────────────────────────────────────────────────────────────
    @staticmethod
    def macd(df: pd.DataFrame, fast: int = 5, slow: int = 13,
             signal: int = 1) -> pd.DataFrame:
        df = df.copy()
        ema_fast = IndicatorEngine.ema(df["close"], fast)
        ema_slow = IndicatorEngine.ema(df["close"], slow)

        df["macd_line"] = ema_fast - ema_slow
        df["macd_signal"] = IndicatorEngine.ema(df["macd_line"], signal)
        df["macd_histogram"] = df["macd_line"] - df["macd_signal"]

        # Histogram flip
        df["macd_hist_bull"] = (
            (df["macd_histogram"] > 0) &
            (df["macd_histogram"].shift(1) <= 0)
        )
        df["macd_hist_bear"] = (
            (df["macd_histogram"] < 0) &
            (df["macd_histogram"].shift(1) >= 0)
        )
        return df

    # ── Supertrend ──────────────────────────────────────────────────────────
    @staticmethod
    def supertrend(df: pd.DataFrame, period: int = 7,
                   multiplier: float = 3.0) -> pd.DataFrame:
        df = df.copy()
        hl2 = (df["high"] + df["low"]) / 2
        atr = IndicatorEngine._atr_series(df, period)

        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr

        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)

        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = -1

        for i in range(1, len(df)):
            if df["close"].iloc[i] > upper_band.iloc[i - 1]:
                direction.iloc[i] = 1
            elif df["close"].iloc[i] < lower_band.iloc[i - 1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i - 1]

            if direction.iloc[i] == 1:
                lower_band.iloc[i] = max(lower_band.iloc[i], lower_band.iloc[i - 1])
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                upper_band.iloc[i] = min(upper_band.iloc[i], upper_band.iloc[i - 1])
                supertrend.iloc[i] = upper_band.iloc[i]

        df["supertrend"] = supertrend
        df["supertrend_dir"] = direction   # 1 = bullish, -1 = bearish
        return df

    # ── Bollinger Bands ─────────────────────────────────────────────────────
    @staticmethod
    def bollinger_bands(df: pd.DataFrame, period: int = 20,
                        std: float = 2.0) -> pd.DataFrame:
        df = df.copy()
        df["bb_mid"] = df["close"].rolling(window=period).mean()
        rolling_std = df["close"].rolling(window=period).std()
        df["bb_upper"] = df["bb_mid"] + std * rolling_std
        df["bb_lower"] = df["bb_mid"] - std * rolling_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

        # Squeeze detection (low bandwidth)
        df["bb_squeeze"] = df["bb_width"] < df["bb_width"].rolling(120).quantile(0.2)
        return df

    # ── ATR ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _atr_series(df: pd.DataFrame, period: int) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - close).abs(),
            (low - close).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        df = df.copy()
        df["atr"] = IndicatorEngine._atr_series(df, period)
        return df

    # ── Volume Analysis ─────────────────────────────────────────────────────
    @staticmethod
    def volume_analysis(df: pd.DataFrame, avg_period: int = 20,
                        spike_mult: float = 2.0) -> pd.DataFrame:
        df = df.copy()
        df["vol_avg"] = df["volume"].rolling(window=avg_period).mean()
        df["vol_ratio"] = df["volume"] / df["vol_avg"].replace(0, np.nan)
        df["vol_spike"] = df["vol_ratio"] > spike_mult

        # Cumulative Volume Delta (approximation using candle body)
        df["_body"] = df["close"] - df["open"]
        df["_range"] = df["high"] - df["low"]
        df["_buy_ratio"] = np.where(
            df["_range"] > 0,
            (df["_body"] / df["_range"] + 1) / 2,
            0.5
        )
        df["cvd"] = (df["volume"] * (2 * df["_buy_ratio"] - 1)).cumsum()
        df["cvd_slope"] = df["cvd"].diff(5)  # 5-bar CVD trend

        df.drop(columns=[c for c in df.columns if c.startswith("_")], inplace=True)
        return df

    # ── Pivot Points (CPR) ──────────────────────────────────────────────────
    @staticmethod
    def daily_cpr(daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Central Pivot Range from daily OHLC.
        Returns pivot, bc (bottom central), tc (top central).
        """
        df = daily_df.copy()
        df["pivot"] = (df["high"] + df["low"] + df["close"]) / 3
        df["bc"] = (df["high"] + df["low"]) / 2
        df["tc"] = (df["pivot"] - df["bc"]) + df["pivot"]

        # Standard pivot levels
        df["r1"] = 2 * df["pivot"] - df["low"]
        df["s1"] = 2 * df["pivot"] - df["high"]
        df["r2"] = df["pivot"] + (df["high"] - df["low"])
        df["s2"] = df["pivot"] - (df["high"] - df["low"])
        df["r3"] = df["high"] + 2 * (df["pivot"] - df["low"])
        df["s3"] = df["low"] - 2 * (df["high"] - df["pivot"])
        return df

    # ── Support / Resistance from 1-Year Data ──────────────────────────────
    @staticmethod
    def find_sr_levels(daily_df: pd.DataFrame, lookback_days: int = 365,
                       min_touches: int = 3, zone_width_pct: float = 0.2) -> list:
        """
        Find key support/resistance levels from 1-year daily data.
        Returns list of dicts: [{"level": float, "type": "support"|"resistance", "touches": int, "strength": float}]
        """
        df = daily_df.tail(lookback_days).copy()
        if len(df) < 30:
            return []

        # Detect swing highs and lows
        highs = []
        lows = []
        for i in range(2, len(df) - 2):
            if df["high"].iloc[i] >= df["high"].iloc[i-1] and df["high"].iloc[i] >= df["high"].iloc[i-2] and \
               df["high"].iloc[i] >= df["high"].iloc[i+1] and df["high"].iloc[i] >= df["high"].iloc[i+2]:
                highs.append(df["high"].iloc[i])
            if df["low"].iloc[i] <= df["low"].iloc[i-1] and df["low"].iloc[i] <= df["low"].iloc[i-2] and \
               df["low"].iloc[i] <= df["low"].iloc[i+1] and df["low"].iloc[i] <= df["low"].iloc[i+2]:
                lows.append(df["low"].iloc[i])

        all_levels = highs + lows
        if not all_levels:
            return []

        # Cluster nearby levels
        all_levels.sort()
        current_price = df["close"].iloc[-1]
        zone_width = current_price * zone_width_pct / 100

        clusters = []
        cluster = [all_levels[0]]
        for level in all_levels[1:]:
            if level - cluster[0] <= zone_width:
                cluster.append(level)
            else:
                if len(cluster) >= min_touches:
                    clusters.append(cluster)
                cluster = [level]
        if len(cluster) >= min_touches:
            clusters.append(cluster)

        sr_levels = []
        for cluster in clusters:
            avg_level = np.mean(cluster)
            level_type = "support" if avg_level < current_price else "resistance"
            sr_levels.append({
                "level": round(avg_level, 2),
                "type": level_type,
                "touches": len(cluster),
                "strength": len(cluster) / min_touches,
                "distance_pct": round(abs(current_price - avg_level) / current_price * 100, 2)
            })

        # Sort by proximity to current price
        sr_levels.sort(key=lambda x: x["distance_pct"])
        return sr_levels

    # ── Fibonacci Retracement ──────────────────────────────────────────────
    @staticmethod
    def fibonacci_levels(swing_high: float, swing_low: float,
                         levels: list = None) -> dict:
        if levels is None:
            levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

        diff = swing_high - swing_low
        return {
            f"fib_{l}": round(swing_high - l * diff, 2)
            for l in levels
        }

    # ── Compute All Indicators ──────────────────────────────────────────────
    @staticmethod
    def compute_all(df: pd.DataFrame, params=None) -> pd.DataFrame:
        """Apply all indicators to an OHLCV dataframe."""
        from scalper.config.settings import ScalpParameters
        if params is None:
            params = ScalpParameters()

        df = IndicatorEngine.ema_crossover(df, params.ema_fast, params.ema_mid, params.ema_slow)
        df = IndicatorEngine.vwap(df, params.vwap_std_1, params.vwap_std_2)
        df = IndicatorEngine.rsi(df, params.rsi_period)
        df = IndicatorEngine.stochastic_rsi(df, params.stoch_rsi_period, params.stoch_rsi_k, params.stoch_rsi_d)
        df = IndicatorEngine.macd(df, params.macd_fast, params.macd_slow, params.macd_signal)
        df = IndicatorEngine.supertrend(df, params.supertrend_period, params.supertrend_multiplier)
        df = IndicatorEngine.bollinger_bands(df, params.bb_period, params.bb_std)
        df = IndicatorEngine.atr(df, params.atr_period)
        df = IndicatorEngine.volume_analysis(df, params.volume_avg_period, params.volume_spike_multiplier)

        return df
