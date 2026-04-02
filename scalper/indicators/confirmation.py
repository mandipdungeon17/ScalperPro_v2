"""
=============================================================================
SCALPER PRO v2 — Technical Indicators for Trade Confirmation
=============================================================================
These indicators DON'T generate trades. They CONFIRM or REJECT the swing
setup found by Layers 1-2-3.

Indicators used for confirmation:
  1. 13 EMA of High + 13 EMA of Low → Dynamic channel (price inside = consolidation, 
     breakout above = bullish confirmation, below = bearish)
  2. 5 EMA → Fast trend for immediate momentum direction
  3. Supertrend (7,3) → Trend filter — only trade in Supertrend direction
  4. RSI (14) → Momentum + oversold/overbought at S/R for bounce confirmation
  5. Volume → Spike on bounce from support = institutional buying = confirm
  6. Bollinger Bands (20,2) → Squeeze = breakout coming, band walk = strong trend
  7. VWAP → Institutional reference — above VWAP = bullish bias

Each indicator returns a confirmation score: +1 (confirms), 0 (neutral), -1 (rejects)
Total score determines if we take the trade.
=============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConfirmationResult:
    """Result of technical confirmation for a swing trade."""
    total_score: int            # Sum of all indicator scores
    max_possible: int           # Maximum possible score
    percentage: float           # score / max as %
    confirmed: bool             # Score >= threshold
    details: Dict[str, Dict]    # Per-indicator breakdown
    summary: List[str]          # Human-readable reasons
    
    # Individual scores
    ema13_channel_score: int = 0
    ema5_score: int = 0
    supertrend_score: int = 0
    rsi_score: int = 0
    volume_score: int = 0
    bollinger_score: int = 0
    vwap_score: int = 0


class TechnicalConfirmation:
    """
    Confirms or rejects swing trade setups using technical indicators.
    
    This is the FINAL gate before execution:
    Layer 1 (Index S/R) → Layer 3 (Strike) → Layer 2 (Premium Swing) → THIS → Execute
    
    Usage:
        conf = TechnicalConfirmation()
        result = conf.confirm(
            df=option_premium_5min_ohlcv,
            direction="LONG",  # buying CE or PE (always buying, so always LONG on premium)
            entry_price=122.0,
            support_level=118.0
        )
        if result.confirmed:
            execute_trade()
    """

    def __init__(self, min_score: int = 4):
        """
        Args:
            min_score: Minimum confirmation score to approve trade (out of ~10)
        """
        self.min_score = min_score

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all confirmation indicators on OHLCV data."""
        df = df.copy()
        
        # ── 1. 13 EMA of High and Low (channel) ──────────────────
        df["ema13_high"] = df["high"].ewm(span=13, adjust=False).mean()
        df["ema13_low"] = df["low"].ewm(span=13, adjust=False).mean()
        df["ema13_mid"] = (df["ema13_high"] + df["ema13_low"]) / 2
        df["ema13_width"] = df["ema13_high"] - df["ema13_low"]
        
        # Price position relative to 13 EMA channel
        df["above_ema13_high"] = df["close"] > df["ema13_high"]
        df["below_ema13_low"] = df["close"] < df["ema13_low"]
        df["inside_ema13"] = (~df["above_ema13_high"]) & (~df["below_ema13_low"])
        
        # Channel breakout detection
        df["ema13_bull_breakout"] = (
            (df["close"] > df["ema13_high"]) & 
            (df["close"].shift(1) <= df["ema13_high"].shift(1))
        )
        df["ema13_bear_breakout"] = (
            (df["close"] < df["ema13_low"]) & 
            (df["close"].shift(1) >= df["ema13_low"].shift(1))
        )

        # ── 2. 5 EMA (fast momentum) ─────────────────────────────
        df["ema5"] = df["close"].ewm(span=5, adjust=False).mean()
        df["ema5_slope"] = df["ema5"].diff(3)  # 3-bar slope
        df["above_ema5"] = df["close"] > df["ema5"]
        
        # 5 EMA crossing 13 EMA mid
        df["ema5_above_ema13"] = df["ema5"] > df["ema13_mid"]
        df["ema5_cross_up"] = (
            (df["ema5"] > df["ema13_mid"]) & 
            (df["ema5"].shift(1) <= df["ema13_mid"].shift(1))
        )
        df["ema5_cross_down"] = (
            (df["ema5"] < df["ema13_mid"]) & 
            (df["ema5"].shift(1) >= df["ema13_mid"].shift(1))
        )

        # ── 3. Supertrend (7, 3) ─────────────────────────────────
        df = self._supertrend(df, period=7, multiplier=3.0)

        # ── 4. RSI (14) ──────────────────────────────────────────
        df = self._rsi(df, period=14)
        
        # RSI divergence detection (simplified)
        df["rsi_oversold"] = df["rsi"] < 30
        df["rsi_overbought"] = df["rsi"] > 70
        df["rsi_bullish_zone"] = (df["rsi"] > 40) & (df["rsi"] < 65)
        df["rsi_bearish_zone"] = (df["rsi"] > 35) & (df["rsi"] < 60)
        df["rsi_slope"] = df["rsi"].diff(3)

        # ── 5. Volume analysis ────────────────────────────────────
        df["vol_sma20"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_sma20"].replace(0, np.nan)
        df["vol_spike"] = df["vol_ratio"] > 1.8
        df["vol_climax"] = df["vol_ratio"] > 3.0
        
        # Volume on up vs down candles
        df["_up_candle"] = df["close"] > df["open"]
        df["vol_up"] = df["volume"] * df["_up_candle"].astype(int)
        df["vol_down"] = df["volume"] * (~df["_up_candle"]).astype(int)
        df["vol_up_avg"] = df["vol_up"].rolling(10).mean()
        df["vol_down_avg"] = df["vol_down"].rolling(10).mean()
        df["buy_pressure"] = df["vol_up_avg"] > df["vol_down_avg"] * 1.3

        # ── 6. Bollinger Bands (20, 2) ────────────────────────────
        df["bb_mid"] = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + 2 * bb_std
        df["bb_lower"] = df["bb_mid"] - 2 * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
        
        # Squeeze: bandwidth in bottom 20% of recent history
        df["bb_squeeze"] = df["bb_width"] < df["bb_width"].rolling(100).quantile(0.2)
        # Band walk: price hugging upper or lower band
        df["bb_walk_up"] = (df["close"] > df["bb_upper"] * 0.98).rolling(3).sum() >= 2
        df["bb_walk_down"] = (df["close"] < df["bb_lower"] * 1.02).rolling(3).sum() >= 2

        # ── 7. VWAP ──────────────────────────────────────────────
        df = self._vwap(df)

        # Cleanup
        df.drop(columns=[c for c in df.columns if c.startswith("_")], inplace=True, errors='ignore')
        
        return df

    def confirm(
        self,
        df: pd.DataFrame,
        direction: str = "LONG",    # "LONG" (buying CE/PE) — on premium chart it's always a buy
        entry_price: float = 0,
        support_level: float = 0,
        resistance_level: float = 0,
    ) -> ConfirmationResult:
        """
        Run all confirmation checks on the option premium chart.
        
        For swing trading, we're always BUYING the option (CE or PE),
        so on the premium chart the direction is always LONG.
        We want to confirm that premium is bouncing from support.
        """
        if len(df) < 25:
            return ConfirmationResult(
                total_score=0, max_possible=10, percentage=0,
                confirmed=False, details={}, summary=["Insufficient data"]
            )

        # Compute indicators if not already done
        if "ema13_high" not in df.columns:
            df = self.compute_indicators(df)

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        prev2 = df.iloc[-3] if len(df) > 2 else prev

        total_score = 0
        max_score = 10  # 7 indicators, some weighted higher
        details = {}
        summary = []

        # ── 1. 13 EMA Channel (+2 / 0 / -1) ─────────────────────
        ema13_score = 0
        if direction == "LONG":
            if latest.get("ema13_bull_breakout", False):
                ema13_score = 2
                summary.append("13 EMA: Bullish breakout above channel ✅✅")
            elif latest.get("above_ema13_high", False):
                ema13_score = 1
                summary.append("13 EMA: Above channel (strong)")
            elif latest.get("inside_ema13", False):
                # Inside channel near low = potential bounce
                close = latest["close"]
                ema_low = latest.get("ema13_low", close)
                ema_high = latest.get("ema13_high", close)
                position = (close - ema_low) / max(ema_high - ema_low, 0.01)
                if position < 0.3:
                    ema13_score = 1
                    summary.append("13 EMA: Inside channel near low (bounce setup)")
                else:
                    ema13_score = 0
                    summary.append("13 EMA: Inside channel (neutral)")
            elif latest.get("below_ema13_low", False):
                ema13_score = -1
                summary.append("13 EMA: Below channel ⚠️")
        
        total_score += ema13_score
        details["ema13_channel"] = {"score": ema13_score, "max": 2}

        # ── 2. 5 EMA (+1 / 0 / -1) ───────────────────────────────
        ema5_score = 0
        if direction == "LONG":
            slope = latest.get("ema5_slope", 0)
            if latest.get("above_ema5", False) and slope > 0:
                ema5_score = 1
                summary.append(f"5 EMA: Price above, slope positive (+{slope:.2f})")
            elif latest.get("ema5_cross_up", False):
                ema5_score = 1
                summary.append("5 EMA: Just crossed above 13 EMA mid ✅")
            elif not latest.get("above_ema5", True):
                # Below 5 EMA but at support = acceptable for bounce
                if support_level > 0 and abs(latest["close"] - support_level) < 5:
                    ema5_score = 0
                    summary.append("5 EMA: Below but at support (bounce pending)")
                else:
                    ema5_score = -1
                    summary.append("5 EMA: Below and falling ⚠️")
        
        total_score += ema5_score
        details["ema5"] = {"score": ema5_score, "max": 1}

        # ── 3. Supertrend (+2 / 0 / -1) ──────────────────────────
        st_score = 0
        st_dir = latest.get("supertrend_dir", 0)
        prev_st_dir = prev.get("supertrend_dir", 0)
        
        if direction == "LONG":
            if st_dir == 1:
                st_score = 2
                summary.append("Supertrend: BULLISH (green) ✅✅")
            elif st_dir == -1 and prev_st_dir == -1:
                # Bearish supertrend but we're buying at support — risky
                st_score = -1
                summary.append("Supertrend: BEARISH — counter-trend ⚠️")
            elif st_dir == 1 and prev_st_dir == -1:
                st_score = 2
                summary.append("Supertrend: JUST FLIPPED BULLISH ✅✅")
        
        total_score += st_score
        details["supertrend"] = {"score": st_score, "max": 2}

        # ── 4. RSI (+1 / 0 / -1) ─────────────────────────────────
        rsi_score = 0
        rsi_val = latest.get("rsi", 50)
        rsi_slope = latest.get("rsi_slope", 0)
        
        if direction == "LONG":
            if latest.get("rsi_oversold", False) or (rsi_val < 35):
                # Oversold at support = bounce setup
                rsi_score = 1
                summary.append(f"RSI: Oversold ({rsi_val:.0f}) at support — bounce likely ✅")
            elif latest.get("rsi_bullish_zone", False) and rsi_slope > 0:
                rsi_score = 1
                summary.append(f"RSI: Bullish zone ({rsi_val:.0f}), rising")
            elif rsi_val > 70:
                rsi_score = -1
                summary.append(f"RSI: Overbought ({rsi_val:.0f}) — late entry ⚠️")
            else:
                summary.append(f"RSI: Neutral ({rsi_val:.0f})")
        
        total_score += rsi_score
        details["rsi"] = {"score": rsi_score, "max": 1, "value": round(rsi_val, 1)}

        # ── 5. Volume (+2 / 0 / -1) ──────────────────────────────
        vol_score = 0
        vol_ratio = latest.get("vol_ratio", 1)
        
        if direction == "LONG":
            if latest.get("vol_spike", False) and latest.get("buy_pressure", False):
                vol_score = 2
                summary.append(f"Volume: Spike ({vol_ratio:.1f}x) with buy pressure ✅✅")
            elif latest.get("buy_pressure", False):
                vol_score = 1
                summary.append(f"Volume: Buy pressure detected ({vol_ratio:.1f}x)")
            elif vol_ratio < 0.5:
                vol_score = -1
                summary.append(f"Volume: Very low ({vol_ratio:.1f}x) — no conviction ⚠️")
            else:
                summary.append(f"Volume: Normal ({vol_ratio:.1f}x)")
        
        total_score += vol_score
        details["volume"] = {"score": vol_score, "max": 2, "ratio": round(vol_ratio, 2)}

        # ── 6. Bollinger Bands (+1 / 0 / -1) ─────────────────────
        bb_score = 0
        bb_pct = latest.get("bb_pct", 0.5)
        
        if direction == "LONG":
            if latest.get("bb_squeeze", False):
                bb_score = 1
                summary.append("Bollinger: SQUEEZE — breakout imminent ✅")
            elif bb_pct < 0.15:
                # Near lower band at support = bounce zone
                bb_score = 1
                summary.append(f"Bollinger: Near lower band ({bb_pct:.0%}) — bounce zone ✅")
            elif latest.get("bb_walk_up", False):
                bb_score = 1
                summary.append("Bollinger: Walking upper band (strong trend)")
            elif bb_pct > 0.95:
                bb_score = -1
                summary.append(f"Bollinger: At upper band ({bb_pct:.0%}) — overextended ⚠️")
            else:
                summary.append(f"Bollinger: Mid-range ({bb_pct:.0%})")
        
        total_score += bb_score
        details["bollinger"] = {"score": bb_score, "max": 1, "pct": round(bb_pct, 2)}

        # ── 7. VWAP (+1 / 0 / -1) ────────────────────────────────
        vwap_score = 0
        vwap = latest.get("vwap", 0)
        close = latest["close"]
        
        if direction == "LONG" and vwap > 0:
            if close > vwap:
                vwap_score = 1
                summary.append(f"VWAP: Above (₹{close:.1f} > ₹{vwap:.1f}) ✅")
            elif close > vwap * 0.99:
                vwap_score = 0
                summary.append(f"VWAP: Near (₹{close:.1f} ~ ₹{vwap:.1f})")
            else:
                vwap_score = -1
                summary.append(f"VWAP: Below (₹{close:.1f} < ₹{vwap:.1f}) ⚠️")
        
        total_score += vwap_score
        details["vwap"] = {"score": vwap_score, "max": 1}

        # ── FINAL VERDICT ─────────────────────────────────────────
        # Clamp score to 0 minimum
        total_score = max(total_score, 0)
        percentage = round(total_score / max_score * 100, 1)
        confirmed = total_score >= self.min_score

        if confirmed:
            summary.insert(0, f"✅ CONFIRMED ({total_score}/{max_score} = {percentage}%)")
        else:
            summary.insert(0, f"❌ REJECTED ({total_score}/{max_score} = {percentage}%)")

        return ConfirmationResult(
            total_score=total_score,
            max_possible=max_score,
            percentage=percentage,
            confirmed=confirmed,
            details=details,
            summary=summary,
            ema13_channel_score=ema13_score,
            ema5_score=ema5_score,
            supertrend_score=st_score,
            rsi_score=rsi_score,
            volume_score=vol_score,
            bollinger_score=bb_score,
            vwap_score=vwap_score,
        )

    # ══════════════════════════════════════════════════════════════
    # INDICATOR IMPLEMENTATIONS
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _supertrend(df: pd.DataFrame, period: int = 7, multiplier: float = 3.0) -> pd.DataFrame:
        hl2 = (df["high"] + df["low"]) / 2
        
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        
        upper = hl2 + multiplier * atr
        lower = hl2 - multiplier * atr
        
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        supertrend.iloc[0] = upper.iloc[0]
        direction.iloc[0] = -1
        
        for i in range(1, len(df)):
            if df["close"].iloc[i] > upper.iloc[i - 1]:
                direction.iloc[i] = 1
            elif df["close"].iloc[i] < lower.iloc[i - 1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i - 1]
            
            if direction.iloc[i] == 1:
                lower.iloc[i] = max(lower.iloc[i], lower.iloc[i - 1])
                supertrend.iloc[i] = lower.iloc[i]
            else:
                upper.iloc[i] = min(upper.iloc[i], upper.iloc[i - 1])
                supertrend.iloc[i] = upper.iloc[i]
        
        df["supertrend"] = supertrend
        df["supertrend_dir"] = direction
        return df

    @staticmethod
    def _rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def _vwap(df: pd.DataFrame) -> pd.DataFrame:
        if "datetime" in df.columns:
            dates = pd.to_datetime(df["datetime"]).dt.date
        elif hasattr(df.index, 'date'):
            dates = df.index.date
        else:
            dates = pd.Series([0] * len(df))
        
        tp = (df["high"] + df["low"] + df["close"]) / 3
        tp_vol = tp * df["volume"]
        
        cum_vol = df.groupby(dates)["volume"].cumsum()
        cum_tp_vol = tp_vol.groupby(dates).cumsum()
        
        df["vwap"] = cum_tp_vol / cum_vol.replace(0, np.nan)
        return df
