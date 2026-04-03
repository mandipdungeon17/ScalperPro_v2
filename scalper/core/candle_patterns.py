"""
=============================================================================
INSTITUTIONAL MODULE — Candlestick Pattern Recognition
=============================================================================
Detects institutional-grade candle patterns for trade confirmation.

NOT generic textbook patterns. These are the patterns that MATTER at key levels:

REVERSAL (at S/R levels — high probability):
  - Engulfing (bullish/bearish)
  - Pin Bar / Hammer / Shooting Star (long wick rejection)
  - Morning Star / Evening Star (3-bar reversal)
  - Tweezer Top / Bottom (equal highs/lows → rejection)

CONTINUATION (in trend):
  - Inside Bar (consolidation before breakout)
  - Marubozu (full-body candle = strong momentum)
  - Three White Soldiers / Three Black Crows

TRAP CONFIRMATION:
  - Doji at resistance/support (indecision → trap setup)
  - Gravestone / Dragonfly Doji (extreme rejection)

Each pattern returns: name, direction bias, strength (0-3), bar index.
Strength matters: a pin bar at a 3-touch support with volume spike = 3.
Same pin bar in no-man's land = 1.
=============================================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class CandlePattern:
    bar_index: int
    pattern: str            # "BULLISH_ENGULFING", "PIN_BAR_BULL", etc.
    direction: str          # "BULLISH" or "BEARISH"
    strength: int           # 1 = weak, 2 = moderate, 3 = strong
    description: str
    body_pct: float         # Body as % of total range
    upper_wick_pct: float   # Upper wick as % of total range
    lower_wick_pct: float


class CandlePatternDetector:
    """
    Detects candle patterns on OHLC data.
    
    Usage:
        detector = CandlePatternDetector()
        patterns = detector.scan(df, bar_index)  # scan at specific bar
        patterns = detector.scan_all(df)          # scan entire df
    """

    def __init__(self, body_threshold: float = 0.3):
        self.body_thresh = body_threshold  # Min body % to be considered "real body"

    def scan(self, df: pd.DataFrame, i: int) -> List[CandlePattern]:
        """Detect all patterns at bar i."""
        if i < 2 or i >= len(df):
            return []

        patterns = []
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        prev2 = df.iloc[i - 2] if i >= 2 else prev

        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        po, ph, pl, pc = float(prev["open"]), float(prev["high"]), float(prev["low"]), float(prev["close"])
        p2o, p2h, p2l, p2c = float(prev2["open"]), float(prev2["high"]), float(prev2["low"]), float(prev2["close"])

        rng = h - l
        if rng < 0.01:
            return patterns

        body = abs(c - o)
        upper_wick = h - max(c, o)
        lower_wick = min(c, o) - l
        body_pct = body / rng
        uw_pct = upper_wick / rng
        lw_pct = lower_wick / rng
        is_bull = c > o
        is_bear = c < o

        prev_rng = ph - pl
        prev_body = abs(pc - po)
        prev_bull = pc > po

        # ── 1. BULLISH ENGULFING ──────────────────────────────────
        if (is_bull and not prev_bull and
            o <= min(pc, po) and c >= max(pc, po) and
            body > prev_body * 1.1):
            strength = 3 if body > prev_body * 1.5 else 2
            patterns.append(CandlePattern(
                i, "BULLISH_ENGULFING", "BULLISH", strength,
                "Current bull candle fully engulfs previous bear candle",
                body_pct, uw_pct, lw_pct
            ))

        # ── 2. BEARISH ENGULFING ──────────────────────────────────
        if (is_bear and prev_bull and
            o >= max(pc, po) and c <= min(pc, po) and
            body > prev_body * 1.1):
            strength = 3 if body > prev_body * 1.5 else 2
            patterns.append(CandlePattern(
                i, "BEARISH_ENGULFING", "BEARISH", strength,
                "Current bear candle fully engulfs previous bull candle",
                body_pct, uw_pct, lw_pct
            ))

        # ── 3. PIN BAR / HAMMER (bullish) ─────────────────────────
        # Long lower wick (>60% of range), small body at top, small upper wick
        if lw_pct > 0.60 and body_pct < 0.30 and uw_pct < 0.15:
            strength = 3 if lw_pct > 0.70 else 2
            patterns.append(CandlePattern(
                i, "PIN_BAR_BULL", "BULLISH", strength,
                f"Hammer/Pin Bar: {lw_pct:.0%} lower wick rejection",
                body_pct, uw_pct, lw_pct
            ))

        # ── 4. PIN BAR / SHOOTING STAR (bearish) ──────────────────
        if uw_pct > 0.60 and body_pct < 0.30 and lw_pct < 0.15:
            strength = 3 if uw_pct > 0.70 else 2
            patterns.append(CandlePattern(
                i, "PIN_BAR_BEAR", "BEARISH", strength,
                f"Shooting Star: {uw_pct:.0%} upper wick rejection",
                body_pct, uw_pct, lw_pct
            ))

        # ── 5. MORNING STAR (3-bar bullish reversal) ──────────────
        if i >= 2:
            p2_bear = p2c < p2o
            mid_small = prev_body < prev_rng * 0.3 if prev_rng > 0 else False
            cur_bull = is_bull and c > (p2o + p2c) / 2
            if p2_bear and mid_small and cur_bull:
                patterns.append(CandlePattern(
                    i, "MORNING_STAR", "BULLISH", 3,
                    "3-bar reversal: bear → indecision → bull",
                    body_pct, uw_pct, lw_pct
                ))

        # ── 6. EVENING STAR (3-bar bearish reversal) ──────────────
        if i >= 2:
            p2_bull = p2c > p2o
            mid_small = prev_body < prev_rng * 0.3 if prev_rng > 0 else False
            cur_bear = is_bear and c < (p2o + p2c) / 2
            if p2_bull and mid_small and cur_bear:
                patterns.append(CandlePattern(
                    i, "EVENING_STAR", "BEARISH", 3,
                    "3-bar reversal: bull → indecision → bear",
                    body_pct, uw_pct, lw_pct
                ))

        # ── 7. INSIDE BAR ─────────────────────────────────────────
        if h <= ph and l >= pl:
            patterns.append(CandlePattern(
                i, "INSIDE_BAR", "BULLISH" if is_bull else "BEARISH", 1,
                "Consolidation inside previous range — breakout pending",
                body_pct, uw_pct, lw_pct
            ))

        # ── 8. MARUBOZU (strong momentum) ─────────────────────────
        if body_pct > 0.85:
            direction = "BULLISH" if is_bull else "BEARISH"
            patterns.append(CandlePattern(
                i, "MARUBOZU", direction, 2,
                f"Full-body {direction.lower()} candle — strong momentum",
                body_pct, uw_pct, lw_pct
            ))

        # ── 9. DOJI (indecision) ──────────────────────────────────
        if body_pct < 0.08:
            if uw_pct > 0.5:
                patterns.append(CandlePattern(
                    i, "GRAVESTONE_DOJI", "BEARISH", 2,
                    "Gravestone Doji: strong upper wick rejection",
                    body_pct, uw_pct, lw_pct
                ))
            elif lw_pct > 0.5:
                patterns.append(CandlePattern(
                    i, "DRAGONFLY_DOJI", "BULLISH", 2,
                    "Dragonfly Doji: strong lower wick rejection (demand)",
                    body_pct, uw_pct, lw_pct
                ))
            else:
                patterns.append(CandlePattern(
                    i, "DOJI", "BULLISH" if c >= o else "BEARISH", 1,
                    "Standard Doji: indecision — wait for confirmation",
                    body_pct, uw_pct, lw_pct
                ))

        # ── 10. TWEEZER TOP ───────────────────────────────────────
        if abs(h - ph) / max(rng, 0.01) < 0.05 and is_bear and prev_bull:
            patterns.append(CandlePattern(
                i, "TWEEZER_TOP", "BEARISH", 2,
                f"Equal highs {h:.1f} ≈ {ph:.1f}: double rejection at top",
                body_pct, uw_pct, lw_pct
            ))

        # ── 11. TWEEZER BOTTOM ────────────────────────────────────
        if abs(l - pl) / max(rng, 0.01) < 0.05 and is_bull and not prev_bull:
            patterns.append(CandlePattern(
                i, "TWEEZER_BOTTOM", "BULLISH", 2,
                f"Equal lows {l:.1f} ≈ {pl:.1f}: double support hold",
                body_pct, uw_pct, lw_pct
            ))

        # ── 12. THREE WHITE SOLDIERS ──────────────────────────────
        if i >= 2:
            all_bull = is_bull and prev_bull and p2c > p2o
            progressive = c > pc > p2c and o > po > p2o
            decent_body = body_pct > 0.4 and prev_body / max(prev_rng, 0.01) > 0.4
            if all_bull and progressive and decent_body:
                patterns.append(CandlePattern(
                    i, "THREE_WHITE_SOLDIERS", "BULLISH", 3,
                    "3 consecutive bullish candles with higher closes",
                    body_pct, uw_pct, lw_pct
                ))

        # ── 13. THREE BLACK CROWS ─────────────────────────────────
        if i >= 2:
            all_bear = is_bear and not prev_bull and p2c < p2o
            progressive = c < pc < p2c and o < po < p2o
            decent_body = body_pct > 0.4 and prev_body / max(prev_rng, 0.01) > 0.4
            if all_bear and progressive and decent_body:
                patterns.append(CandlePattern(
                    i, "THREE_BLACK_CROWS", "BEARISH", 3,
                    "3 consecutive bearish candles with lower closes",
                    body_pct, uw_pct, lw_pct
                ))

        return patterns

    def scan_all(self, df: pd.DataFrame) -> List[CandlePattern]:
        """Scan entire DataFrame for patterns."""
        all_patterns = []
        for i in range(2, len(df)):
            all_patterns.extend(self.scan(df, i))
        return all_patterns

    def get_confirmation_score(
        self, df: pd.DataFrame, bar_index: int, direction: str
    ) -> int:
        """
        Score how well candle patterns confirm the proposed direction.
        Returns 0, 1, or 2 (used as bonus in signal scoring).
        """
        patterns = self.scan(df, bar_index)
        if not patterns:
            return 0

        aligned = [p for p in patterns if p.direction == direction]
        contrary = [p for p in patterns if p.direction != direction and p.direction != "NEUTRAL"]

        if not aligned:
            return -1 if contrary else 0

        max_strength = max(p.strength for p in aligned)
        if max_strength >= 3:
            return 2  # Strong reversal pattern at level = +2
        elif max_strength >= 2:
            return 1
        return 0
