"""
=============================================================================
SCALPER PRO v2 — Market Structure Analysis
=============================================================================
Identifies the internal structure of price action:

  Higher High / Higher Low  = UPTREND  (buy CE at HL)
  Lower High  / Lower Low   = DOWNTREND (buy PE at LH)

  Break of Structure (BOS)  = trend continues (trade with it)
  Change of Character (CHoCH) = potential reversal (watch for new setup)

  Liquidity Void     = price imbalance, fills fast (avoid entries inside)
  External Liquidity = stop clusters above HH or below LL (traps live here)
=============================================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class StructureTrend(Enum):
    UPTREND   = "UPTREND"
    DOWNTREND = "DOWNTREND"
    SIDEWAYS  = "SIDEWAYS"
    UNKNOWN   = "UNKNOWN"


class StructureEvent(Enum):
    BOS_UP    = "BOS_UP"     # Break of Structure bullish (HH broken → trend continues up)
    BOS_DOWN  = "BOS_DOWN"   # Break of Structure bearish (LL broken → trend continues down)
    CHOCH_UP  = "CHOCH_UP"   # Change of Character bullish (LH broken → reversal to up)
    CHOCH_DOWN= "CHOCH_DOWN" # Change of Character bearish (HL broken → reversal to down)
    NONE      = "NONE"


@dataclass
class SwingPoint:
    idx:       int
    price:     float
    date:      str
    is_high:   bool    # True = swing high, False = swing low
    confirmed: bool    # True once 3+ candles have followed


@dataclass
class StructureZone:
    """A significant swing point that defines structure."""
    price:     float
    date:      str
    zone_type: str    # "HH", "HL", "LH", "LL"
    is_broken: bool = False
    break_date: str = ""


@dataclass
class MarketStructure:
    """Full structure snapshot at a given bar."""
    trend:          StructureTrend
    last_event:     StructureEvent
    last_hh:        Optional[float]   # last Higher High
    last_hl:        Optional[float]   # last Higher Low (key support in uptrend)
    last_lh:        Optional[float]   # last Lower High (key resistance in downtrend)
    last_ll:        Optional[float]   # last Lower Low
    last_bos_price: Optional[float]   # price where last BOS occurred
    last_choch_price: Optional[float] # price where last CHoCH occurred
    swing_highs:    List[SwingPoint] = field(default_factory=list)
    swing_lows:     List[SwingPoint] = field(default_factory=list)
    structure_zones: List[StructureZone] = field(default_factory=list)
    trend_strength: float = 0.0  # 0-1: how clean the trend is
    # For entry logic:
    ideal_ce_entry: Optional[float] = None  # HL level (buy CE here in uptrend)
    ideal_pe_entry: Optional[float] = None  # LH level (buy PE here in downtrend)


def analyze_market_structure(
    df: pd.DataFrame,
    lookback: int = 5,
    min_swing_bars: int = 3,
) -> MarketStructure:
    """
    Analyze market structure from OHLCV DataFrame.

    lookback:       bars on each side to confirm a swing high/low
    min_swing_bars: minimum bars since last swing to create a new one

    Returns MarketStructure with trend, swing points, BOS/CHoCH events.
    """
    if len(df) < lookback * 2 + 10:
        return MarketStructure(
            trend=StructureTrend.UNKNOWN,
            last_event=StructureEvent.NONE,
            last_hh=None, last_hl=None,
            last_lh=None, last_ll=None,
            last_bos_price=None, last_choch_price=None,
        )

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    n = len(df)

    # ── Step 1: Detect swing highs and lows ──────────────────────────────────
    swing_highs: List[SwingPoint] = []
    swing_lows:  List[SwingPoint] = []

    for i in range(lookback, n - lookback):
        # Swing high: highest of (2*lookback + 1) window
        window_h = highs[i - lookback: i + lookback + 1]
        if highs[i] == max(window_h) and list(window_h).count(highs[i]) == 1:
            date = str(df.iloc[i].get("datetime", df.index[i]))[:10]
            swing_highs.append(SwingPoint(
                idx=i, price=highs[i], date=date,
                is_high=True, confirmed=True,
            ))

        # Swing low: lowest of window
        window_l = lows[i - lookback: i + lookback + 1]
        if lows[i] == min(window_l) and list(window_l).count(lows[i]) == 1:
            date = str(df.iloc[i].get("datetime", df.index[i]))[:10]
            swing_lows.append(SwingPoint(
                idx=i, price=lows[i], date=date,
                is_high=False, confirmed=True,
            ))

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return MarketStructure(
            trend=StructureTrend.UNKNOWN,
            last_event=StructureEvent.NONE,
            last_hh=None, last_hl=None,
            last_lh=None, last_ll=None,
            last_bos_price=None, last_choch_price=None,
            swing_highs=swing_highs, swing_lows=swing_lows,
        )

    # ── Step 2: Label HH/HL/LH/LL ────────────────────────────────────────────
    structure_zones: List[StructureZone] = []
    hh_prices, hl_prices = [], []
    lh_prices, ll_prices = [], []

    # Highs: compare consecutive swing highs
    for i in range(1, len(swing_highs)):
        prev = swing_highs[i - 1]
        curr = swing_highs[i]
        if curr.price > prev.price:
            zone_type = "HH"
            hh_prices.append(curr.price)
        else:
            zone_type = "LH"
            lh_prices.append(curr.price)
        structure_zones.append(StructureZone(
            price=curr.price, date=curr.date, zone_type=zone_type
        ))

    # Lows: compare consecutive swing lows
    for i in range(1, len(swing_lows)):
        prev = swing_lows[i - 1]
        curr = swing_lows[i]
        if curr.price > prev.price:
            zone_type = "HL"
            hl_prices.append(curr.price)
        else:
            zone_type = "LL"
            ll_prices.append(curr.price)
        structure_zones.append(StructureZone(
            price=curr.price, date=curr.date, zone_type=zone_type
        ))

    # ── Step 3: Determine trend ───────────────────────────────────────────────
    recent_highs = [sh.price for sh in swing_highs[-4:]]
    recent_lows  = [sl.price for sl in swing_lows[-4:]]

    hh_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] > recent_highs[i-1])
    hl_count = sum(1 for i in range(1, len(recent_lows))  if recent_lows[i]  > recent_lows[i-1])
    lh_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] < recent_highs[i-1])
    ll_count = sum(1 for i in range(1, len(recent_lows))  if recent_lows[i]  < recent_lows[i-1])

    if hh_count >= 2 and hl_count >= 2:
        trend = StructureTrend.UPTREND
        trend_strength = min((hh_count + hl_count) / 6, 1.0)
    elif lh_count >= 2 and ll_count >= 2:
        trend = StructureTrend.DOWNTREND
        trend_strength = min((lh_count + ll_count) / 6, 1.0)
    elif hh_count + hl_count > lh_count + ll_count:
        trend = StructureTrend.UPTREND
        trend_strength = 0.4
    elif lh_count + ll_count > hh_count + hl_count:
        trend = StructureTrend.DOWNTREND
        trend_strength = 0.4
    else:
        trend = StructureTrend.SIDEWAYS
        trend_strength = 0.2

    # ── Step 4: Detect BOS and CHoCH on the most recent bars ─────────────────
    last_event = StructureEvent.NONE
    last_bos   = None
    last_choch = None
    current_close = closes[-1]

    # Use recent swing levels for BOS/CHoCH detection
    recent_sh = swing_highs[-3:]
    recent_sl = swing_lows[-3:]

    if recent_sh and recent_sl:
        prev_hh = max(s.price for s in recent_sh[:-1]) if len(recent_sh) > 1 else None
        prev_ll = min(s.price for s in recent_sl[:-1]) if len(recent_sl) > 1 else None
        last_sh_price = recent_sh[-1].price
        last_sl_price = recent_sl[-1].price

        # BOS bullish: price closes above a previous HH → trend continues up
        if prev_hh and current_close > prev_hh and trend == StructureTrend.UPTREND:
            last_event = StructureEvent.BOS_UP
            last_bos   = prev_hh

        # BOS bearish: price closes below a previous LL → trend continues down
        elif prev_ll and current_close < prev_ll and trend == StructureTrend.DOWNTREND:
            last_event = StructureEvent.BOS_DOWN
            last_bos   = prev_ll

        # CHoCH bullish: in downtrend, price closes above a previous LH → reversal attempt
        elif trend == StructureTrend.DOWNTREND and lh_prices:
            last_lh = lh_prices[-1]
            if current_close > last_lh:
                last_event = StructureEvent.CHOCH_UP
                last_choch = last_lh

        # CHoCH bearish: in uptrend, price closes below a previous HL → reversal attempt
        elif trend == StructureTrend.UPTREND and hl_prices:
            last_hl = hl_prices[-1]
            if current_close < last_hl:
                last_event = StructureEvent.CHOCH_DOWN
                last_choch = last_hl

    # ── Step 5: Ideal entry zones ─────────────────────────────────────────────
    # CE: buy at the last HL (higher low) in uptrend — price pulls back here for bounce
    # PE: buy at the last LH (lower high) in downtrend — price bounces here for rejection
    ideal_ce = hl_prices[-1] if hl_prices else (swing_lows[-1].price if swing_lows else None)
    ideal_pe = lh_prices[-1] if lh_prices else (swing_highs[-1].price if swing_highs else None)

    return MarketStructure(
        trend=trend,
        last_event=last_event,
        last_hh=hh_prices[-1] if hh_prices else None,
        last_hl=hl_prices[-1] if hl_prices else None,
        last_lh=lh_prices[-1] if lh_prices else None,
        last_ll=ll_prices[-1] if ll_prices else None,
        last_bos_price=last_bos,
        last_choch_price=last_choch,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        structure_zones=structure_zones,
        trend_strength=trend_strength,
        ideal_ce_entry=ideal_ce,
        ideal_pe_entry=ideal_pe,
    )


def is_direction_aligned_with_structure(
    direction: str,
    structure: MarketStructure,
) -> Tuple[bool, str]:
    """
    Check if proposed trade direction aligns with market structure.

    Returns (aligned: bool, reason: str)
    """
    t = structure.trend

    if direction == "CE":
        if t == StructureTrend.UPTREND:
            return True, f"CE aligned: UPTREND (strength {structure.trend_strength:.1f})"
        elif t == StructureTrend.SIDEWAYS:
            return True, "CE marginal: SIDEWAYS — only if at strong demand zone"
        elif t == StructureTrend.DOWNTREND:
            # Allow CE only if CHoCH (reversal signal) just fired
            if structure.last_event == StructureEvent.CHOCH_UP:
                return True, f"CE allowed: CHoCH_UP at {structure.last_choch_price:.0f} — reversal setup"
            return False, f"CE blocked: DOWNTREND with no CHoCH (trend {structure.trend_strength:.1f})"
        else:
            return False, "CE blocked: unknown structure"

    elif direction == "PE":
        if t == StructureTrend.DOWNTREND:
            return True, f"PE aligned: DOWNTREND (strength {structure.trend_strength:.1f})"
        elif t == StructureTrend.SIDEWAYS:
            return True, "PE marginal: SIDEWAYS — only if at strong supply zone"
        elif t == StructureTrend.UPTREND:
            if structure.last_event == StructureEvent.CHOCH_DOWN:
                return True, f"PE allowed: CHoCH_DOWN at {structure.last_choch_price:.0f} — reversal setup"
            return False, f"PE blocked: UPTREND with no CHoCH (trend {structure.trend_strength:.1f})"
        else:
            return False, "PE blocked: unknown structure"

    return False, f"Unknown direction: {direction}"
