"""
=============================================================================
INSTITUTIONAL MODULE — Opening Range Breakout (ORB) + Gap Analysis
=============================================================================
The first 15 minutes define the day's battleground.

ORB Logic:
  1. Capture the high/low of the first 15-min candle (09:15-09:30)
  2. Wait for a CLEAN breakout (close above/below the range)
  3. Confirm with volume absorption (breakout candle > 1.5x avg volume)
  4. Enter on breakout, SL at opposite end of opening range

Gap Analysis:
  - Gap Up + holds above previous close = bullish continuation
  - Gap Up + fails to hold = gap fill (bearish)
  - Gap Down + holds below previous close = bearish continuation
  - Gap Down + recovers = gap fill (bullish)
  
  The "Opening Auction" (09:00-09:08 pre-market) gives equilibrium
  price clues via GIFT Nifty and pre-open session.

Volume Absorption:
  - On the first few candles, watch if volume is getting absorbed
    at a level (high volume but no price movement = institutional
    absorption / accumulation)
=============================================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class OpeningRange:
    """The first-candle range (typically 09:15-09:30)."""
    date: str
    orb_high: float
    orb_low: float
    orb_range: float         # High - Low
    orb_range_pct: float     # As % of price
    orb_close: float         # Where the first candle closed
    orb_volume: int
    prev_day_close: float
    gap_points: float        # Open - Prev Close
    gap_pct: float
    gap_type: str            # "GAP_UP", "GAP_DOWN", "FLAT"
    orb_body_type: str       # "BULLISH", "BEARISH", "DOJI"


@dataclass
class ORBSignal:
    """Actionable signal from ORB analysis."""
    bar_index: int
    timestamp: str
    signal_type: str          # "ORB_BREAKOUT_UP", "ORB_BREAKOUT_DOWN", "ORB_FAIL", "GAP_FILL"
    direction: str            # "CE" or "PE"
    entry_price: float
    sl_price: float
    target_price: float
    orb_range: float
    volume_ratio: float       # Breakout candle volume vs first candle
    score: int                # 0-10
    reasons: List[str]
    gap_context: str          # How the gap is behaving


class ORBEngine:
    """
    Opening Range Breakout engine.
    
    Usage:
        engine = ORBEngine()
        orb = engine.compute_opening_range(df_15min, date)
        signal = engine.check_breakout(df_15min, bar_index, orb)
    """

    def __init__(self, first_candle_minutes: int = 15, min_rr: float = 2.0):
        self.first_candle_min = first_candle_minutes
        self.min_rr = min_rr

    def compute_opening_range(
        self,
        df: pd.DataFrame,
        target_date: str = None,
        prev_close: float = None,
    ) -> Optional[OpeningRange]:
        """Compute ORB from the first candle of the session."""
        df = df.copy()
        if "datetime" not in df.columns:
            return None

        df["datetime"] = pd.to_datetime(df["datetime"])

        if target_date:
            day_df = df[df["datetime"].dt.date == pd.Timestamp(target_date).date()]
        else:
            # Use the last trading day in the data
            dates = df["datetime"].dt.date.unique()
            if len(dates) == 0:
                return None
            day_df = df[df["datetime"].dt.date == dates[-1]]

        if len(day_df) < 2:
            return None

        first_bar = day_df.iloc[0]
        orb_high = float(first_bar["high"])
        orb_low = float(first_bar["low"])
        orb_close = float(first_bar["close"])
        orb_open = float(first_bar["open"])
        orb_vol = int(first_bar["volume"])
        orb_range = orb_high - orb_low

        # Gap from previous close
        if prev_close is None:
            # Try to find previous day's close
            all_dates = sorted(df["datetime"].dt.date.unique())
            target_d = day_df["datetime"].dt.date.iloc[0]
            prev_dates = [d for d in all_dates if d < target_d]
            if prev_dates:
                prev_day = df[df["datetime"].dt.date == prev_dates[-1]]
                prev_close = float(prev_day["close"].iloc[-1])
            else:
                prev_close = orb_open

        gap_pts = orb_open - prev_close
        gap_pct = gap_pts / prev_close * 100 if prev_close > 0 else 0

        if gap_pct > 0.3:
            gap_type = "GAP_UP"
        elif gap_pct < -0.3:
            gap_type = "GAP_DOWN"
        else:
            gap_type = "FLAT"

        body_type = "BULLISH" if orb_close > orb_open else "BEARISH" if orb_close < orb_open else "DOJI"

        return OpeningRange(
            date=str(day_df["datetime"].dt.date.iloc[0]),
            orb_high=round(orb_high, 2),
            orb_low=round(orb_low, 2),
            orb_range=round(orb_range, 2),
            orb_range_pct=round(orb_range / orb_close * 100, 3),
            orb_close=round(orb_close, 2),
            orb_volume=orb_vol,
            prev_day_close=round(prev_close, 2),
            gap_points=round(gap_pts, 2),
            gap_pct=round(gap_pct, 2),
            gap_type=gap_type,
            orb_body_type=body_type,
        )

    def check_breakout(
        self,
        df: pd.DataFrame,
        bar_index: int,
        orb: OpeningRange,
    ) -> Optional[ORBSignal]:
        """Check if current bar breaks the opening range."""
        if bar_index < 1 or bar_index >= len(df):
            return None

        row = df.iloc[bar_index]
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        opn = float(row["open"])
        vol = int(row["volume"])

        vol_ratio = vol / max(orb.orb_volume, 1)
        time_str = str(row.get("datetime", ""))

        reasons = []
        score = 0

        # ── ORB BREAKOUT UP ───────────────────────────────────────
        if close > orb.orb_high and opn <= orb.orb_high:
            direction = "CE"
            signal_type = "ORB_BREAKOUT_UP"
            entry = close
            sl = orb.orb_low
            target = entry + orb.orb_range * 2

            reasons.append(f"ORB breakout above {orb.orb_high:.0f}")

            # Score components
            if close > orb.orb_high + orb.orb_range * 0.1:
                score += 2; reasons.append("Clean breakout (>10% past range)")
            else:
                score += 1; reasons.append("Marginal breakout")

            if vol_ratio > 2.0:
                score += 2; reasons.append(f"Strong volume {vol_ratio:.1f}x ORB")
            elif vol_ratio > 1.3:
                score += 1; reasons.append(f"Volume {vol_ratio:.1f}x ORB")

            if orb.orb_body_type == "BULLISH":
                score += 1; reasons.append("First candle was bullish")

            if orb.gap_type == "GAP_UP":
                score += 1; reasons.append(f"Gap Up {orb.gap_pct:+.1f}% supports")
            elif orb.gap_type == "GAP_DOWN" and close > orb.prev_day_close:
                score += 2; reasons.append("Recovered gap down — strong buyers")

            if orb.orb_range_pct < 0.5:
                score += 1; reasons.append(f"Tight ORB {orb.orb_range_pct:.2f}% — explosive breakout likely")

            rr = (target - entry) / max(entry - sl, 0.01)
            if rr >= self.min_rr:
                score += 1

            return ORBSignal(
                bar_index=bar_index, timestamp=time_str,
                signal_type=signal_type, direction=direction,
                entry_price=round(entry, 2), sl_price=round(sl, 2),
                target_price=round(target, 2), orb_range=orb.orb_range,
                volume_ratio=round(vol_ratio, 2), score=min(score, 10),
                reasons=reasons, gap_context=f"{orb.gap_type} {orb.gap_pct:+.1f}%",
            )

        # ── ORB BREAKOUT DOWN ─────────────────────────────────────
        elif close < orb.orb_low and opn >= orb.orb_low:
            direction = "PE"
            signal_type = "ORB_BREAKOUT_DOWN"
            entry = close
            sl = orb.orb_high
            target = entry - orb.orb_range * 2

            reasons.append(f"ORB breakdown below {orb.orb_low:.0f}")

            if close < orb.orb_low - orb.orb_range * 0.1:
                score += 2; reasons.append("Clean breakdown")
            else:
                score += 1

            if vol_ratio > 2.0:
                score += 2; reasons.append(f"Strong volume {vol_ratio:.1f}x")
            elif vol_ratio > 1.3:
                score += 1

            if orb.orb_body_type == "BEARISH":
                score += 1; reasons.append("First candle was bearish")

            if orb.gap_type == "GAP_DOWN":
                score += 1; reasons.append(f"Gap Down {orb.gap_pct:+.1f}% confirms")
            elif orb.gap_type == "GAP_UP" and close < orb.prev_day_close:
                score += 2; reasons.append("Gap Up failed — trapped bulls")

            if orb.orb_range_pct < 0.5:
                score += 1; reasons.append("Tight ORB — explosive move")

            return ORBSignal(
                bar_index=bar_index, timestamp=time_str,
                signal_type=signal_type, direction=direction,
                entry_price=round(entry, 2), sl_price=round(sl, 2),
                target_price=round(target, 2), orb_range=orb.orb_range,
                volume_ratio=round(vol_ratio, 2), score=min(score, 10),
                reasons=reasons, gap_context=f"{orb.gap_type} {orb.gap_pct:+.1f}%",
            )

        # ── GAP FILL SIGNAL ───────────────────────────────────────
        elif orb.gap_type == "GAP_UP" and close < orb.prev_day_close:
            return ORBSignal(
                bar_index=bar_index, timestamp=time_str,
                signal_type="GAP_FILL", direction="PE",
                entry_price=round(close, 2),
                sl_price=round(orb.orb_high, 2),
                target_price=round(orb.prev_day_close - orb.gap_points, 2),
                orb_range=orb.orb_range,
                volume_ratio=round(vol_ratio, 2), score=5,
                reasons=[f"Gap Up {orb.gap_pct:+.1f}% FILLED — bearish"],
                gap_context=f"GAP_FILL from {orb.gap_type}",
            )

        elif orb.gap_type == "GAP_DOWN" and close > orb.prev_day_close:
            return ORBSignal(
                bar_index=bar_index, timestamp=time_str,
                signal_type="GAP_FILL", direction="CE",
                entry_price=round(close, 2),
                sl_price=round(orb.orb_low, 2),
                target_price=round(orb.prev_day_close + abs(orb.gap_points), 2),
                orb_range=orb.orb_range,
                volume_ratio=round(vol_ratio, 2), score=5,
                reasons=[f"Gap Down {orb.gap_pct:+.1f}% FILLED — bullish"],
                gap_context=f"GAP_FILL from {orb.gap_type}",
            )

        return None

    def detect_volume_absorption(
        self, df: pd.DataFrame, bar_index: int, window: int = 5
    ) -> Optional[str]:
        """
        Detect volume absorption: high volume but minimal price movement.
        This indicates institutional accumulation (bullish) or distribution (bearish).
        """
        if bar_index < window:
            return None

        recent = df.iloc[bar_index - window:bar_index + 1]
        price_range = float(recent["high"].max() - recent["low"].min())
        avg_volume = float(recent["volume"].mean())
        price_pct = price_range / float(recent["close"].iloc[-1]) * 100

        # High volume + low price movement = absorption
        vol_threshold = float(df["volume"].iloc[max(0, bar_index-30):bar_index].mean()) * 1.5

        if avg_volume > vol_threshold and price_pct < 0.3:
            close = float(recent["close"].iloc[-1])
            opn = float(recent["open"].iloc[0])
            if close > opn:
                return "ACCUMULATION"  # Bullish absorption
            elif close < opn:
                return "DISTRIBUTION"  # Bearish absorption

        return None
