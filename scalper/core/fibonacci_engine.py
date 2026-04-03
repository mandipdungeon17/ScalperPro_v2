"""
=============================================================================
INSTITUTIONAL MODULE — Fibonacci Retracement & Extension Engine
=============================================================================
Institutional Fib usage is different from retail:

RETAIL uses Fib as fixed levels. INSTITUTIONS use Fib for:
  1. OPTIMAL TRADE ENTRY (OTE) zone: 0.62-0.79 retracement in impulsive move
     This is where Smart Money enters — the "sweet spot" for pullback entries.
  2. Premium vs Discount zones:
     - Above 0.5 Fib = PREMIUM zone (sell/PE territory)
     - Below 0.5 Fib = DISCOUNT zone (buy/CE territory)
  3. Extension targets: -0.27, -0.618 for profit targets
  4. Confluence: Fib level + S/R + POC = very high probability

Auto-detection of major swing for Fib calculation:
  - Uses the MOST RECENT significant swing high → swing low (for uptrend Fib)
  - Or swing low → swing high (for downtrend Fib)
  - "Significant" = swing that spans at least 1.5x daily ATR
=============================================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# Standard + Institutional Fib levels
FIB_LEVELS = {
    0.0:   "Swing High/Low",
    0.236: "Shallow Retracement",
    0.382: "First Support/Resistance",
    0.5:   "Equilibrium (Premium/Discount boundary)",
    0.618: "Golden Ratio — OTE zone start",
    0.705: "Optimal Trade Entry (OTE mid)",
    0.786: "OTE zone end — deep retracement",
    1.0:   "Full Retracement",
}

FIB_EXTENSIONS = {
    -0.272: "First Extension Target",
    -0.618: "Golden Extension Target",
    -1.0:   "Full Extension (measured move)",
    -1.618: "Deep Extension",
}


@dataclass
class FibLevel:
    ratio: float
    price: float
    label: str
    is_ote_zone: bool        # True if in 0.618-0.786 range
    is_premium: bool         # True if above 0.5 (sell zone)
    is_discount: bool        # True if below 0.5 (buy zone)
    has_sr_confluence: bool = False  # True if S/R level is nearby
    has_poc_confluence: bool = False


@dataclass
class FibSetup:
    """Complete Fibonacci setup from a detected swing."""
    swing_type: str          # "UPSWING" (low→high) or "DOWNSWING" (high→low)
    swing_high: float
    swing_low: float
    swing_range: float
    swing_high_bar: int
    swing_low_bar: int
    retracement_levels: List[FibLevel]
    extension_levels: List[FibLevel]
    ote_zone_high: float     # 0.618 level
    ote_zone_low: float      # 0.786 level
    current_position: str    # "PREMIUM", "DISCOUNT", "AT_OTE", "ABOVE", "BELOW"
    trade_bias: str          # "BUY_CE_AT_OTE", "BUY_PE_AT_OTE", "WAIT"
    score: int               # 0-5


class FibonacciEngine:
    """
    Auto-detects swings and computes Fibonacci levels.
    
    Usage:
        engine = FibonacciEngine()
        setups = engine.analyze(df_15min, atr=200)
        
        # Check if current price is at OTE zone
        for setup in setups:
            if setup.current_position == "AT_OTE":
                # High probability entry zone
    """

    def __init__(self, min_swing_atr: float = 1.5, max_setups: int = 3):
        self.min_swing_atr = min_swing_atr
        self.max_setups = max_setups

    def analyze(
        self,
        df: pd.DataFrame,
        atr: float = None,
        sr_levels: list = None,
        poc_levels: list = None,
    ) -> List[FibSetup]:
        """
        Find significant swings and compute Fib levels.
        
        Args:
            df: OHLCV DataFrame
            atr: Daily ATR (for minimum swing size filtering)
            sr_levels: List of S/R prices for confluence checking
            poc_levels: List of POC prices for confluence checking
        """
        if len(df) < 20:
            return []

        if atr is None:
            atr = self._compute_atr(df)

        # Detect significant swings
        swings = self._detect_major_swings(df, atr)
        if not swings:
            return []

        current_price = float(df["close"].iloc[-1])
        setups = []

        for swing_type, sh, sl, sh_bar, sl_bar in swings[:self.max_setups]:
            setup = self._compute_fib_setup(
                swing_type, sh, sl, sh_bar, sl_bar,
                current_price, sr_levels, poc_levels
            )
            if setup:
                setups.append(setup)

        return setups

    def get_confirmation_score(
        self,
        df: pd.DataFrame,
        direction: str,
        atr: float = None,
    ) -> Tuple[int, str]:
        """
        Quick check: is the current price in an OTE zone aligned with direction?
        Returns (score: 0-2, reason: str)
        """
        setups = self.analyze(df, atr)
        if not setups:
            return 0, "No significant Fib setup detected"

        for setup in setups:
            if setup.current_position == "AT_OTE":
                if direction == "CE" and setup.swing_type == "UPSWING":
                    return 2, f"At OTE zone {setup.ote_zone_high:.0f}-{setup.ote_zone_low:.0f} in upswing — buy CE"
                elif direction == "PE" and setup.swing_type == "DOWNSWING":
                    return 2, f"At OTE zone {setup.ote_zone_high:.0f}-{setup.ote_zone_low:.0f} in downswing — buy PE"
                else:
                    return 1, f"At OTE but direction mismatch (setup={setup.swing_type})"

            if setup.current_position == "DISCOUNT" and direction == "CE":
                return 1, f"In discount zone (below 0.5 Fib) — favorable for CE"
            elif setup.current_position == "PREMIUM" and direction == "PE":
                return 1, f"In premium zone (above 0.5 Fib) — favorable for PE"

        return 0, "Price not at significant Fib level"

    def _detect_major_swings(self, df, atr) -> list:
        """Find the most recent significant swing high-low pairs."""
        highs_idx = []
        lows_idx = []
        lb = 5

        for i in range(lb, len(df) - lb):
            if all(df["high"].iloc[i] >= df["high"].iloc[i-j] and
                   df["high"].iloc[i] >= df["high"].iloc[i+j] for j in range(1, lb+1)):
                highs_idx.append(i)
            if all(df["low"].iloc[i] <= df["low"].iloc[i-j] and
                   df["low"].iloc[i] <= df["low"].iloc[i+j] for j in range(1, lb+1)):
                lows_idx.append(i)

        if not highs_idx or not lows_idx:
            return []

        min_range = atr * self.min_swing_atr
        swings = []

        # Most recent swing high → most recent swing low (upswing Fib)
        for hi_idx in reversed(highs_idx[-5:]):
            for lo_idx in reversed(lows_idx[-5:]):
                sh = float(df["high"].iloc[hi_idx])
                sl = float(df["low"].iloc[lo_idx])
                swing_range = sh - sl

                if swing_range < min_range:
                    continue

                if lo_idx < hi_idx:
                    swings.append(("UPSWING", sh, sl, hi_idx, lo_idx))
                else:
                    swings.append(("DOWNSWING", sh, sl, hi_idx, lo_idx))

                if len(swings) >= self.max_setups:
                    break
            if len(swings) >= self.max_setups:
                break

        return swings

    def _compute_fib_setup(
        self, swing_type, sh, sl, sh_bar, sl_bar,
        current_price, sr_levels, poc_levels
    ) -> Optional[FibSetup]:
        swing_range = sh - sl
        if swing_range <= 0:
            return None

        # Retracement levels
        ret_levels = []
        for ratio, label in FIB_LEVELS.items():
            if swing_type == "UPSWING":
                price = sh - ratio * swing_range  # Retrace from high
            else:
                price = sl + ratio * swing_range  # Retrace from low

            is_ote = 0.618 <= ratio <= 0.786
            is_premium = ratio < 0.5 if swing_type == "UPSWING" else ratio > 0.5
            is_discount = ratio > 0.5 if swing_type == "UPSWING" else ratio < 0.5

            fib = FibLevel(
                ratio=ratio, price=round(price, 2), label=label,
                is_ote_zone=is_ote, is_premium=is_premium, is_discount=is_discount,
            )

            # Check confluence
            if sr_levels:
                fib.has_sr_confluence = any(abs(price - sr) / price < 0.002 for sr in sr_levels)
            if poc_levels:
                fib.has_poc_confluence = any(abs(price - poc) / price < 0.002 for poc in poc_levels)

            ret_levels.append(fib)

        # Extension levels
        ext_levels = []
        for ratio, label in FIB_EXTENSIONS.items():
            if swing_type == "UPSWING":
                price = sh - ratio * swing_range  # Extensions above swing high
            else:
                price = sl + ratio * swing_range
            ext_levels.append(FibLevel(
                ratio=ratio, price=round(price, 2), label=label,
                is_ote_zone=False, is_premium=False, is_discount=False,
            ))

        # OTE zone prices
        if swing_type == "UPSWING":
            ote_high = sh - 0.618 * swing_range
            ote_low = sh - 0.786 * swing_range
        else:
            ote_high = sl + 0.618 * swing_range
            ote_low = sl + 0.786 * swing_range

        # Current price position
        if ote_low <= current_price <= ote_high or ote_high <= current_price <= ote_low:
            position = "AT_OTE"
        elif swing_type == "UPSWING":
            mid = sh - 0.5 * swing_range
            if current_price > mid:
                position = "PREMIUM"
            elif current_price < mid:
                position = "DISCOUNT"
            else:
                position = "AT_EQUILIBRIUM"
        else:
            mid = sl + 0.5 * swing_range
            if current_price > mid:
                position = "PREMIUM"
            else:
                position = "DISCOUNT"

        # Trade bias
        if position == "AT_OTE":
            if swing_type == "UPSWING":
                bias = "BUY_CE_AT_OTE"
            else:
                bias = "BUY_PE_AT_OTE"
        elif position == "DISCOUNT" and swing_type == "UPSWING":
            bias = "BUY_CE_DISCOUNT"
        elif position == "PREMIUM" and swing_type == "DOWNSWING":
            bias = "BUY_PE_PREMIUM"
        else:
            bias = "WAIT"

        # Score
        score = 0
        if position == "AT_OTE": score += 3
        elif position in ("DISCOUNT", "PREMIUM"): score += 1
        # Confluence bonuses
        ote_levels = [l for l in ret_levels if l.is_ote_zone]
        if any(l.has_sr_confluence for l in ote_levels): score += 1
        if any(l.has_poc_confluence for l in ote_levels): score += 1

        return FibSetup(
            swing_type=swing_type, swing_high=sh, swing_low=sl,
            swing_range=round(swing_range, 2),
            swing_high_bar=sh_bar, swing_low_bar=sl_bar,
            retracement_levels=ret_levels, extension_levels=ext_levels,
            ote_zone_high=round(ote_high, 2), ote_zone_low=round(ote_low, 2),
            current_position=position, trade_bias=bias, score=min(score, 5),
        )

    @staticmethod
    def _compute_atr(df, period=14):
        if len(df) < period + 1:
            return float(df["high"].iloc[-1] - df["low"].iloc[-1])
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs()
        ], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])
