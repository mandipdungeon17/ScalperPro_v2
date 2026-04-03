"""
=============================================================================
INSTITUTIONAL MODULE — Volume Profile & Value Area
=============================================================================
Institutional traders don't use simple volume bars. They use VOLUME PROFILE:
  - Histogram of volume AT EACH PRICE LEVEL (not per candle)
  - POC = Point of Control = price with highest volume (fair value)
  - VAH = Value Area High = upper bound of 70% volume zone
  - VAL = Value Area Low = lower bound of 70% volume zone
  - Virgin POC = POC from a previous session that price hasn't revisited
                 → acts as a MAGNET, price tends to return to it

This is fundamentally different from looking at volume bars below candles.
Volume profile shows WHERE the volume happened (at which price), not WHEN.

For scalping:
  - Price above POC = bullish (buying CE territory)
  - Price below POC = bearish (buying PE territory)
  - Virgin POC below = price will likely pull back to it (target for PE)
  - Virgin POC above = price will likely rally to it (target for CE)
  - Price at VAH = potential resistance (unless strong trend)
  - Price at VAL = potential support (unless strong trend)
=============================================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class VolumeProfileResult:
    """Volume profile for a given session/period."""
    session_date: str
    poc: float                  # Point of Control (highest volume price)
    poc_volume: int             # Volume at POC
    vah: float                  # Value Area High
    val: float                  # Value Area Low
    total_volume: int
    price_bins: List[float]     # All price levels
    volume_at_price: List[int]  # Volume at each level
    is_virgin_poc: bool = False # True if price hasn't revisited this POC


@dataclass
class ValueAreaSignal:
    """Actionable signal from volume profile analysis."""
    current_price: float
    position: str               # "ABOVE_VAH", "IN_VALUE", "BELOW_VAL", "AT_POC"
    poc: float
    vah: float
    val: float
    bias: str                   # "BULLISH", "BEARISH", "NEUTRAL"
    virgin_pocs: List[float]    # Untested POCs (magnets)
    nearest_virgin_poc: Optional[float]
    virgin_poc_direction: str   # "ABOVE" or "BELOW" current price
    score: int                  # 0-3 confirmation score
    reason: str


class VolumeProfileEngine:
    """
    Computes volume profile from OHLCV data.
    
    Usage:
        engine = VolumeProfileEngine()
        profiles = engine.compute_daily_profiles(df_15min)
        signal = engine.get_signal(current_price, profiles)
    """

    def __init__(self, num_bins: int = 50, value_area_pct: float = 0.70):
        self.num_bins = num_bins
        self.va_pct = value_area_pct

    def compute_session_profile(
        self, df: pd.DataFrame, session_label: str = ""
    ) -> Optional[VolumeProfileResult]:
        """Compute volume profile for a single session (day)."""
        if len(df) < 5:
            return None

        price_low = float(df["low"].min())
        price_high = float(df["high"].max())
        if price_high <= price_low:
            return None

        bin_size = (price_high - price_low) / self.num_bins
        bins = [price_low + i * bin_size for i in range(self.num_bins + 1)]
        bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(self.num_bins)]
        vol_at_price = [0] * self.num_bins

        # Distribute each candle's volume across the price bins it spans
        for _, row in df.iterrows():
            bar_low = float(row["low"])
            bar_high = float(row["high"])
            bar_vol = int(row["volume"])

            for bi in range(self.num_bins):
                bl = bins[bi]
                bh = bins[bi + 1]
                # Overlap between bar range and bin
                overlap_low = max(bar_low, bl)
                overlap_high = min(bar_high, bh)
                if overlap_high > overlap_low:
                    bar_range = max(bar_high - bar_low, 0.01)
                    fraction = (overlap_high - overlap_low) / bar_range
                    vol_at_price[bi] += int(bar_vol * fraction)

        total_vol = sum(vol_at_price)
        if total_vol == 0:
            return None

        # POC = bin with max volume
        poc_idx = vol_at_price.index(max(vol_at_price))
        poc = bin_centers[poc_idx]
        poc_vol = vol_at_price[poc_idx]

        # Value Area: expand from POC until 70% of volume is covered
        va_target = total_vol * self.va_pct
        va_vol = vol_at_price[poc_idx]
        lo_idx, hi_idx = poc_idx, poc_idx

        while va_vol < va_target:
            expand_lo = vol_at_price[lo_idx - 1] if lo_idx > 0 else 0
            expand_hi = vol_at_price[hi_idx + 1] if hi_idx < self.num_bins - 1 else 0

            if expand_lo == 0 and expand_hi == 0:
                break

            if expand_lo >= expand_hi and lo_idx > 0:
                lo_idx -= 1
                va_vol += expand_lo
            elif hi_idx < self.num_bins - 1:
                hi_idx += 1
                va_vol += expand_hi
            elif lo_idx > 0:
                lo_idx -= 1
                va_vol += expand_lo
            else:
                break

        val = bin_centers[lo_idx]  # Value Area Low
        vah = bin_centers[hi_idx]  # Value Area High

        return VolumeProfileResult(
            session_date=session_label,
            poc=round(poc, 2),
            poc_volume=poc_vol,
            vah=round(vah, 2),
            val=round(val, 2),
            total_volume=total_vol,
            price_bins=bin_centers,
            volume_at_price=vol_at_price,
        )

    def compute_daily_profiles(self, df: pd.DataFrame) -> List[VolumeProfileResult]:
        """Compute volume profile for each trading day."""
        if "datetime" not in df.columns:
            return []

        df = df.copy()
        df["_date"] = pd.to_datetime(df["datetime"]).dt.date
        profiles = []

        for dt, group in df.groupby("_date", sort=True):
            profile = self.compute_session_profile(group, str(dt))
            if profile:
                profiles.append(profile)

        # Mark virgin POCs
        self._mark_virgin_pocs(profiles, df)
        return profiles

    def _mark_virgin_pocs(self, profiles: List[VolumeProfileResult], df: pd.DataFrame):
        """Mark POCs that haven't been revisited by subsequent price action."""
        if len(profiles) < 2:
            return

        for i, profile in enumerate(profiles[:-1]):
            poc = profile.poc
            # Check if any bar AFTER this session touched the POC
            subsequent = df[df["datetime"] > pd.Timestamp(profile.session_date)]
            if len(subsequent) == 0:
                profile.is_virgin_poc = True
                continue

            touched = ((subsequent["low"] <= poc) & (subsequent["high"] >= poc)).any()
            profile.is_virgin_poc = not touched

    def get_signal(
        self,
        current_price: float,
        profiles: List[VolumeProfileResult],
    ) -> Optional[ValueAreaSignal]:
        """Generate a signal based on volume profile analysis."""
        if not profiles:
            return None

        latest = profiles[-1]
        poc, vah, val = latest.poc, latest.vah, latest.val

        # Determine position relative to value area
        if current_price > vah:
            position = "ABOVE_VAH"
            bias = "BULLISH"
            reason = f"Price {current_price:.0f} above Value Area High {vah:.0f} — bullish acceptance"
        elif current_price < val:
            position = "BELOW_VAL"
            bias = "BEARISH"
            reason = f"Price {current_price:.0f} below Value Area Low {val:.0f} — bearish acceptance"
        elif abs(current_price - poc) / poc < 0.001:
            position = "AT_POC"
            bias = "NEUTRAL"
            reason = f"Price at POC {poc:.0f} — fair value, wait for direction"
        else:
            position = "IN_VALUE"
            bias = "BULLISH" if current_price > poc else "BEARISH"
            side = "above" if current_price > poc else "below"
            reason = f"Price {side} POC {poc:.0f}, inside value area"

        # Find virgin POCs (magnets)
        virgin_pocs = [p.poc for p in profiles if p.is_virgin_poc]
        nearest_vp = None
        vp_dir = ""
        if virgin_pocs:
            dists = [(vp, abs(current_price - vp)) for vp in virgin_pocs]
            nearest_vp, _ = min(dists, key=lambda x: x[1])
            vp_dir = "ABOVE" if nearest_vp > current_price else "BELOW"
            reason += f" | Virgin POC {nearest_vp:.0f} ({vp_dir})"

        # Score
        score = 0
        if position == "ABOVE_VAH" and bias == "BULLISH":
            score = 2
        elif position == "BELOW_VAL" and bias == "BEARISH":
            score = 2
        elif position == "IN_VALUE":
            score = 1
        if nearest_vp:
            score += 1

        return ValueAreaSignal(
            current_price=current_price,
            position=position, poc=poc, vah=vah, val=val,
            bias=bias, virgin_pocs=virgin_pocs,
            nearest_virgin_poc=nearest_vp,
            virgin_poc_direction=vp_dir,
            score=min(score, 3), reason=reason,
        )
