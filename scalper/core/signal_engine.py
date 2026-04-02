"""
=============================================================================
SCALPER PRO - Signal Engine
=============================================================================
Scores all indicators and generates trade signals.
Only fires when multiple conditions align (confluence-based approach).
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from scalper.indicators.technical import IndicatorEngine
from scalper.indicators.oi_analyzer import OISnapshot
from scalper.config.settings import ScalpParameters, SwingParameters, IndexConfig

logger = logging.getLogger(__name__)


class SignalType(Enum):
    SCALP_LONG = "SCALP_LONG"
    SCALP_SHORT = "SCALP_SHORT"
    SWING_LONG = "SWING_LONG"
    SWING_SHORT = "SWING_SHORT"
    NO_SIGNAL = "NO_SIGNAL"


@dataclass
class TradeSignal:
    """Complete trade signal with all context."""
    timestamp: datetime
    index: str
    signal_type: SignalType
    direction: str                # "LONG" or "SHORT"
    strategy: str                 # "scalp" or "swing"
    entry_price: float
    target_price: float
    stoploss_price: float
    score: int                    # Total score out of max
    max_score: int
    confidence: float             # score / max_score
    reasons: List[str]            # Why this signal triggered
    indicators: Dict              # Snapshot of all indicator values
    oi_data: Optional[Dict]       # OI snapshot if available
    strike_recommendation: Optional[Dict]  # Best strike to trade
    risk_reward: float
    atr_value: float
    vix_value: Optional[float]


class SignalEngine:
    """
    Multi-factor signal generator.

    Scalp signals require score >= min_scalp_score (default 7/12).
    Swing signals require score >= min_swing_score (default 8/12)
    plus proximity to a 1-year key level.
    """

    def __init__(self, scalp_params: ScalpParameters = None,
                 swing_params: SwingParameters = None):
        self.scalp_params = scalp_params or ScalpParameters()
        self.swing_params = swing_params or SwingParameters()

    def generate_signals(
        self,
        df: pd.DataFrame,
        index_config: IndexConfig,
        oi_snapshot: Optional[OISnapshot] = None,
        daily_df: Optional[pd.DataFrame] = None,
        vix: Optional[float] = None,
    ) -> List[TradeSignal]:
        """
        Main signal generation pipeline.
        df: OHLCV data with all indicators already computed (use IndicatorEngine.compute_all)
        daily_df: Daily OHLCV for swing level calculation
        """
        signals = []

        if len(df) < 50:
            return signals

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # ── Scalp Signals ─────────────────────────────────────────────────
        scalp_long_score, scalp_long_reasons = self._score_scalp(
            df, latest, prev, "LONG", oi_snapshot, vix
        )
        scalp_short_score, scalp_short_reasons = self._score_scalp(
            df, latest, prev, "SHORT", oi_snapshot, vix
        )

        max_scalp = 12

        if scalp_long_score >= self.scalp_params.min_scalp_score:
            signals.append(self._build_signal(
                index_config, latest, "LONG", "scalp",
                scalp_long_score, max_scalp, scalp_long_reasons,
                oi_snapshot, vix
            ))

        if scalp_short_score >= self.scalp_params.min_scalp_score:
            signals.append(self._build_signal(
                index_config, latest, "SHORT", "scalp",
                scalp_short_score, max_scalp, scalp_short_reasons,
                oi_snapshot, vix
            ))

        # ── Swing Signals (100+ pts, 1-year levels) ──────────────────────
        if daily_df is not None and len(daily_df) > 60:
            sr_levels = IndicatorEngine.find_sr_levels(
                daily_df,
                lookback_days=self.swing_params.yearly_lookback_days,
                min_touches=self.swing_params.sr_touch_count,
                zone_width_pct=self.swing_params.sr_zone_width_pct
            )

            swing_signals = self._check_swing_signals(
                df, latest, prev, index_config, sr_levels,
                oi_snapshot, vix, daily_df
            )
            signals.extend(swing_signals)

        return signals

    def _score_scalp(
        self,
        df: pd.DataFrame,
        latest: pd.Series,
        prev: pd.Series,
        direction: str,
        oi_snapshot: Optional[OISnapshot],
        vix: Optional[float],
    ) -> Tuple[int, List[str]]:
        """Score a potential scalp trade. Returns (score, reasons)."""
        score = 0
        reasons = []

        if direction == "LONG":
            # 1. EMA Trend Alignment (fast > mid > slow)
            if latest.get("ema_trend_bull", False):
                score += 2
                reasons.append("EMA trend aligned bullish")
            elif latest.get("ema_fast", 0) > latest.get("ema_mid", 0):
                score += 1
                reasons.append("EMA fast > mid")

            # 2. Price above VWAP
            if latest.get("above_vwap", False):
                score += 1
                reasons.append("Above VWAP")

            # 3. RSI momentum
            rsi = latest.get("rsi", 50)
            if rsi > self.scalp_params.rsi_long_threshold:
                score += 1
                reasons.append(f"RSI bullish ({rsi:.0f})")
            elif rsi < self.scalp_params.rsi_deadzone_low:
                score -= 1  # Against momentum

            # 4. MACD histogram positive & rising
            hist = latest.get("macd_histogram", 0)
            prev_hist = prev.get("macd_histogram", 0)
            if hist > 0 and hist > prev_hist:
                score += 1
                reasons.append("MACD histogram rising positive")
            elif latest.get("macd_hist_bull", False):
                score += 1
                reasons.append("MACD histogram flipped bullish")

            # 5. Supertrend bullish
            if latest.get("supertrend_dir", 0) == 1:
                score += 1
                reasons.append("Supertrend bullish")

            # 6. Bollinger Band context
            close = latest["close"]
            bb_mid = latest.get("bb_mid", close)
            bb_upper = latest.get("bb_upper", close + 100)
            if close > bb_mid and close < bb_upper:
                score += 1
                reasons.append("Price between BB mid and upper (trending)")
            if latest.get("bb_squeeze", False):
                score += 1
                reasons.append("BB squeeze — breakout imminent")

            # 7. Volume confirmation
            if latest.get("vol_spike", False):
                score += 1
                reasons.append(f"Volume spike ({latest.get('vol_ratio', 0):.1f}x)")
            elif latest.get("vol_ratio", 0) > 1.3:
                score += 0.5
                reasons.append("Above average volume")

            # 8. CVD positive
            if latest.get("cvd_slope", 0) > 0:
                score += 1
                reasons.append("CVD slope positive (net buying)")

            # 9. OI signal
            if oi_snapshot and oi_snapshot.signal == "BULLISH":
                score += 1
                reasons.append(f"OI bullish (PCR={oi_snapshot.pcr})")

            # 10. VIX in sweet spot
            if vix is not None:
                if self.scalp_params.vix_sweet_low <= vix <= self.scalp_params.vix_sweet_high:
                    score += 1
                    reasons.append(f"VIX in sweet zone ({vix:.1f})")
                elif vix < self.scalp_params.vix_min or vix > self.scalp_params.vix_max:
                    score -= 1
                    reasons.append(f"VIX unfavorable ({vix:.1f})")

        elif direction == "SHORT":
            # Mirror logic for shorts
            if latest.get("ema_trend_bear", False):
                score += 2
                reasons.append("EMA trend aligned bearish")
            elif latest.get("ema_fast", 0) < latest.get("ema_mid", 0):
                score += 1
                reasons.append("EMA fast < mid")

            if not latest.get("above_vwap", True):
                score += 1
                reasons.append("Below VWAP")

            rsi = latest.get("rsi", 50)
            if rsi < self.scalp_params.rsi_short_threshold:
                score += 1
                reasons.append(f"RSI bearish ({rsi:.0f})")

            hist = latest.get("macd_histogram", 0)
            prev_hist = prev.get("macd_histogram", 0)
            if hist < 0 and hist < prev_hist:
                score += 1
                reasons.append("MACD histogram falling negative")
            elif latest.get("macd_hist_bear", False):
                score += 1
                reasons.append("MACD histogram flipped bearish")

            if latest.get("supertrend_dir", 0) == -1:
                score += 1
                reasons.append("Supertrend bearish")

            close = latest["close"]
            bb_mid = latest.get("bb_mid", close)
            bb_lower = latest.get("bb_lower", close - 100)
            if close < bb_mid and close > bb_lower:
                score += 1
                reasons.append("Price between BB mid and lower (trending down)")
            if latest.get("bb_squeeze", False):
                score += 1
                reasons.append("BB squeeze — breakout imminent")

            if latest.get("vol_spike", False):
                score += 1
                reasons.append(f"Volume spike ({latest.get('vol_ratio', 0):.1f}x)")

            if latest.get("cvd_slope", 0) < 0:
                score += 1
                reasons.append("CVD slope negative (net selling)")

            if oi_snapshot and oi_snapshot.signal == "BEARISH":
                score += 1
                reasons.append(f"OI bearish (PCR={oi_snapshot.pcr})")

            if vix is not None:
                if self.scalp_params.vix_sweet_low <= vix <= self.scalp_params.vix_sweet_high:
                    score += 1
                    reasons.append(f"VIX in sweet zone ({vix:.1f})")

        return int(score), reasons

    def _check_swing_signals(
        self,
        df: pd.DataFrame,
        latest: pd.Series,
        prev: pd.Series,
        index_config: IndexConfig,
        sr_levels: List[Dict],
        oi_snapshot: Optional[OISnapshot],
        vix: Optional[float],
        daily_df: pd.DataFrame,
    ) -> List[TradeSignal]:
        """Check if price is near a 1-year key level for a swing trade."""
        signals = []
        close = latest["close"]

        # Find nearby support levels for LONG swings
        supports = [l for l in sr_levels
                    if l["type"] == "support"
                    and l["distance_pct"] <= self.swing_params.min_proximity_pct]

        for support in supports:
            score = 0
            reasons = [f"Near 1Y support {support['level']} ({support['touches']} touches)"]
            score += min(support["touches"], 5)  # Up to 5 points for strong level

            # Add technical confirmation
            if latest.get("supertrend_dir", 0) == 1:
                score += 1
                reasons.append("Supertrend bullish")
            if latest.get("above_vwap", False):
                score += 1
                reasons.append("Above VWAP")
            rsi = latest.get("rsi", 50)
            if 30 < rsi < 50:  # RSI pulling back but not collapsed
                score += 1
                reasons.append("RSI in bounce zone")
            if oi_snapshot and oi_snapshot.signal in ("BULLISH", "NEUTRAL"):
                score += 1
                reasons.append(f"OI supports ({oi_snapshot.signal})")

            # Fibonacci confirmation
            if len(daily_df) > 20:
                recent_high = daily_df["high"].tail(60).max()
                recent_low = daily_df["low"].tail(60).min()
                fibs = IndicatorEngine.fibonacci_levels(recent_high, recent_low)
                for fib_name, fib_level in fibs.items():
                    if abs(close - fib_level) / close < 0.003:  # Within 0.3%
                        score += 1
                        reasons.append(f"At {fib_name} level ({fib_level})")
                        break

            max_swing = 12
            if score >= self.swing_params.min_swing_score if hasattr(self.swing_params, 'min_swing_score') else 8:
                # Check risk-reward
                target = close + index_config.swing_target
                sl = close - index_config.swing_stoploss
                rr = index_config.swing_target / index_config.swing_stoploss

                if rr >= self.swing_params.min_rr_ratio:
                    signals.append(TradeSignal(
                        timestamp=datetime.now(),
                        index=index_config.symbol,
                        signal_type=SignalType.SWING_LONG,
                        direction="LONG",
                        strategy="swing",
                        entry_price=close,
                        target_price=target,
                        stoploss_price=sl,
                        score=score,
                        max_score=max_swing,
                        confidence=round(score / max_swing, 2),
                        reasons=reasons,
                        indicators=self._snapshot_indicators(latest),
                        oi_data=self._oi_to_dict(oi_snapshot),
                        strike_recommendation=None,
                        risk_reward=round(rr, 2),
                        atr_value=latest.get("atr", 0),
                        vix_value=vix,
                    ))

        # Find nearby resistance levels for SHORT swings
        resistances = [l for l in sr_levels
                       if l["type"] == "resistance"
                       and l["distance_pct"] <= self.swing_params.min_proximity_pct]

        for resistance in resistances:
            score = 0
            reasons = [f"Near 1Y resistance {resistance['level']} ({resistance['touches']} touches)"]
            score += min(resistance["touches"], 5)

            if latest.get("supertrend_dir", 0) == -1:
                score += 1
                reasons.append("Supertrend bearish")
            if not latest.get("above_vwap", True):
                score += 1
                reasons.append("Below VWAP")
            rsi = latest.get("rsi", 50)
            if 50 < rsi < 70:
                score += 1
                reasons.append("RSI in rejection zone")
            if oi_snapshot and oi_snapshot.signal in ("BEARISH", "NEUTRAL"):
                score += 1
                reasons.append(f"OI supports ({oi_snapshot.signal})")

            max_swing = 12
            if score >= 8:
                target = close - index_config.swing_target
                sl = close + index_config.swing_stoploss
                rr = index_config.swing_target / index_config.swing_stoploss

                if rr >= self.swing_params.min_rr_ratio:
                    signals.append(TradeSignal(
                        timestamp=datetime.now(),
                        index=index_config.symbol,
                        signal_type=SignalType.SWING_SHORT,
                        direction="SHORT",
                        strategy="swing",
                        entry_price=close,
                        target_price=target,
                        stoploss_price=sl,
                        score=score,
                        max_score=max_swing,
                        confidence=round(score / max_swing, 2),
                        reasons=reasons,
                        indicators=self._snapshot_indicators(latest),
                        oi_data=self._oi_to_dict(oi_snapshot),
                        strike_recommendation=None,
                        risk_reward=round(rr, 2),
                        atr_value=latest.get("atr", 0),
                        vix_value=vix,
                    ))

        return signals

    def _build_signal(
        self,
        config: IndexConfig,
        latest: pd.Series,
        direction: str,
        strategy: str,
        score: int,
        max_score: int,
        reasons: List[str],
        oi_snapshot: Optional[OISnapshot],
        vix: Optional[float],
    ) -> TradeSignal:
        close = latest["close"]
        if strategy == "scalp":
            target_pts = config.scalp_target
            sl_pts = config.scalp_stoploss
        else:
            target_pts = config.swing_target
            sl_pts = config.swing_stoploss

        if direction == "LONG":
            target = close + target_pts
            sl = close - sl_pts
            sig_type = SignalType.SCALP_LONG if strategy == "scalp" else SignalType.SWING_LONG
        else:
            target = close - target_pts
            sl = close + sl_pts
            sig_type = SignalType.SCALP_SHORT if strategy == "scalp" else SignalType.SWING_SHORT

        return TradeSignal(
            timestamp=datetime.now(),
            index=config.symbol,
            signal_type=sig_type,
            direction=direction,
            strategy=strategy,
            entry_price=close,
            target_price=target,
            stoploss_price=sl,
            score=score,
            max_score=max_score,
            confidence=round(score / max_score, 2),
            reasons=reasons,
            indicators=self._snapshot_indicators(latest),
            oi_data=self._oi_to_dict(oi_snapshot),
            strike_recommendation=None,
            risk_reward=round(target_pts / sl_pts, 2),
            atr_value=latest.get("atr", 0),
            vix_value=vix,
        )

    @staticmethod
    def _snapshot_indicators(latest: pd.Series) -> Dict:
        keys = [
            "close", "ema_fast", "ema_mid", "ema_slow", "vwap",
            "rsi", "macd_histogram", "supertrend_dir", "bb_mid",
            "bb_upper", "bb_lower", "atr", "vol_ratio", "cvd_slope"
        ]
        return {k: round(float(latest.get(k, 0)), 2) for k in keys}

    @staticmethod
    def _oi_to_dict(oi: Optional[OISnapshot]) -> Optional[Dict]:
        if oi is None:
            return None
        return {
            "pcr": oi.pcr,
            "max_pain": oi.max_pain,
            "signal": oi.signal,
            "signal_strength": oi.signal_strength,
            "straddle_premium": oi.straddle_premium,
        }
