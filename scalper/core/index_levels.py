"""
=============================================================================
SCALPER PRO v2 — LAYER 1: Index Level Marker
=============================================================================
Marks Support/Resistance on INDEX charts (not options).
This is the COMPASS — it tells you WHAT to trade (CE or PE).

Flow:
  1. Daily/Weekly chart → Major S/R levels (positional)
  2. 15-min/1-hour chart → Minor S/R levels (intraday)
  3. ATR-based proximity check → Is index close enough to a level?
  4. Decision: Near support → look for CE | Near resistance → look for PE

The index level marker does NOT generate trades.
It outputs: "Index is near [SUPPORT/RESISTANCE] at [level], look for [CE/PE]"
=============================================================================
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LevelType(Enum):
    SUPPORT = "SUPPORT"
    RESISTANCE = "RESISTANCE"


class LevelTimeframe(Enum):
    WEEKLY = "WEEKLY"       # Strongest — positional
    DAILY = "DAILY"         # Strong — swing
    HOURLY = "HOURLY"       # Medium — intraday
    FIFTEEN_MIN = "15MIN"   # Intraday minor levels


@dataclass
class IndexLevel:
    """A single support or resistance level on the index."""
    price: float
    level_type: LevelType
    timeframe: LevelTimeframe
    touches: int                    # How many times price has reacted here
    first_touch_date: str           # When this level was first formed
    last_touch_date: str            # Most recent reaction
    strength: float                 # 0-1 composite score
    atr_at_level: float             # ATR when this level was active
    is_round_number: bool           # Nifty 23500, BankNifty 51000 etc.
    fib_confluence: bool            # Does this align with a Fibonacci level?
    pivot_confluence: bool          # Does this align with CPR pivot?
    notes: str = ""


@dataclass
class IndexProximitySignal:
    """
    Output of Layer 1: tells Layer 2 what to look for.
    This is the bridge between index analysis and option selection.
    """
    index: str
    current_price: float
    nearest_level: IndexLevel
    distance_points: float          # How far index is from the level
    distance_atr: float             # Distance expressed in ATR multiples
    proximity_zone: str             # "AT_LEVEL", "APPROACHING", "FAR"
    direction: str                  # "CE" or "PE" — what to trade
    confidence: float               # 0-1 based on level strength + proximity
    all_nearby_supports: List[IndexLevel]
    all_nearby_resistances: List[IndexLevel]
    trend_context: str              # "UPTREND", "DOWNTREND", "SIDEWAYS"
    action: str                     # "BUY_CE_AT_SUPPORT", "BUY_PE_AT_RESISTANCE", "WAIT"


class IndexLevelMarker:
    """
    Marks S/R levels on INDEX charts using multi-timeframe analysis.

    Usage:
        marker = IndexLevelMarker()

        # Feed it data at multiple timeframes
        marker.mark_levels(
            daily_df=daily_ohlcv,       # 1 year of daily candles
            weekly_df=weekly_ohlcv,      # 2 years of weekly candles (optional)
            hourly_df=hourly_ohlcv,      # 30 days of 1-hour candles
            fifteen_min_df=fifteen_ohlcv # 10 days of 15-min candles
        )

        # Check if index is near a level
        signal = marker.check_proximity(current_price=23480, index="NIFTY")
        # signal.direction = "CE"  (because near support)
        # signal.action = "BUY_CE_AT_SUPPORT"
    """

    # Levels with more than this many touches tend to break, not bounce
    MAX_TOUCH_LIMIT: int = 4

    def __init__(self):
        self.levels: List[IndexLevel] = []
        self._daily_atr: float = 0
        self._intraday_atr: float = 0
        self._daily_df: Optional[pd.DataFrame] = None   # stored for EMA20 filter

    def mark_levels(
        self,
        daily_df: pd.DataFrame,
        weekly_df: Optional[pd.DataFrame] = None,
        hourly_df: Optional[pd.DataFrame] = None,
        fifteen_min_df: Optional[pd.DataFrame] = None,
        index: str = "NIFTY",
    ) -> List[IndexLevel]:
        """
        Master method: Marks all S/R levels across timeframes.
        Each DataFrame needs: datetime, open, high, low, close, volume
        """
        self.levels = []
        self._daily_df = daily_df  # store for EMA20 trend filter

        # Calculate ATR for proximity measurement
        if len(daily_df) > 14:
            self._daily_atr = self._compute_atr(daily_df, 14)
        if hourly_df is not None and len(hourly_df) > 14:
            self._intraday_atr = self._compute_atr(hourly_df, 14)

        # ── LAYER 1A: Weekly S/R (strongest) ──────────────────────────
        if weekly_df is not None and len(weekly_df) > 20:
            weekly_levels = self._detect_swing_levels(
                weekly_df, LevelTimeframe.WEEKLY, lookback=3
            )
            self.levels.extend(weekly_levels)
            logger.info(f"Weekly levels: {len(weekly_levels)}")

        # ── LAYER 1B: Daily S/R (strong) ──────────────────────────────
        if len(daily_df) > 20:
            daily_levels = self._detect_swing_levels(
                daily_df, LevelTimeframe.DAILY, lookback=3
            )
            self.levels.extend(daily_levels)
            logger.info(f"Daily levels: {len(daily_levels)}")

        # ── LAYER 1C: Hourly S/R (medium) ─────────────────────────────
        if hourly_df is not None and len(hourly_df) > 30:
            hourly_levels = self._detect_swing_levels(
                hourly_df, LevelTimeframe.HOURLY, lookback=3
            )
            self.levels.extend(hourly_levels)
            logger.info(f"Hourly levels: {len(hourly_levels)}")

        # ── LAYER 1D: 15-min S/R (intraday minor) ─────────────────────
        if fifteen_min_df is not None and len(fifteen_min_df) > 30:
            intra_levels = self._detect_swing_levels(
                fifteen_min_df, LevelTimeframe.FIFTEEN_MIN, lookback=4
            )
            self.levels.extend(intra_levels)
            logger.info(f"15-min levels: {len(intra_levels)}")

        # ── Add round number levels ────────────────────────────────────
        self._add_round_number_levels(daily_df, index)

        # ── Add Fibonacci levels from major swings ─────────────────────
        self._add_fibonacci_levels(daily_df)

        # ── Add CPR/Pivot levels for today ─────────────────────────────
        self._add_pivot_levels(daily_df)

        # ── Cluster overlapping levels (confluence) ────────────────────
        self._merge_confluent_levels()

        # Sort by strength descending
        self.levels.sort(key=lambda l: l.strength, reverse=True)

        logger.info(f"Total index levels marked: {len(self.levels)}")
        return self.levels

    def check_proximity(
        self,
        current_price: float,
        index: str = "NIFTY",
    ) -> IndexProximitySignal:
        """
        Check if the index is near any marked S/R level.
        Uses ATR-based proximity (not fixed percentage).

        Returns an IndexProximitySignal telling Layer 2 what to do.
        """
        if not self.levels:
            return IndexProximitySignal(
                index=index, current_price=current_price,
                nearest_level=None, distance_points=999,
                distance_atr=999, proximity_zone="FAR",
                direction="WAIT", confidence=0,
                all_nearby_supports=[], all_nearby_resistances=[],
                trend_context="UNKNOWN", action="WAIT"
            )

        atr = self._daily_atr if self._daily_atr > 0 else current_price * 0.01

        # Find nearby supports and resistances
        nearby_supports = []
        nearby_resistances = []

        for level in self.levels:
            dist = abs(current_price - level.price)
            dist_atr = dist / atr

            # Consider levels within 2x ATR as "nearby"
            if dist_atr > 2.0:
                continue

            # Skip high-touch levels — 4+ touches tend to break, not bounce
            if level.touches > self.MAX_TOUCH_LIMIT:
                continue

            if level.price <= current_price:
                nearby_supports.append((level, dist, dist_atr))
            else:
                nearby_resistances.append((level, dist, dist_atr))

        # Sort by distance
        nearby_supports.sort(key=lambda x: x[1])
        nearby_resistances.sort(key=lambda x: x[1])

        # Determine nearest level and direction
        nearest_support = nearby_supports[0] if nearby_supports else None
        nearest_resistance = nearby_resistances[0] if nearby_resistances else None

        # Decision logic:
        # Near support → expect bounce → BUY CE
        # Near resistance → expect rejection → BUY PE
        # ATR-based proximity:
        #   < 0.3 ATR = AT_LEVEL (very close, ready to trade)
        #   0.3-0.8 ATR = APPROACHING (prepare, watch for entry)
        #   > 0.8 ATR = FAR (wait)

        direction = "WAIT"
        action = "WAIT"
        nearest = None
        dist_pts = 999
        dist_atr_val = 999
        proximity = "FAR"
        confidence = 0

        if nearest_support and (nearest_resistance is None or
                                nearest_support[1] < nearest_resistance[1]):
            # Closer to support
            nearest = nearest_support[0]
            dist_pts = nearest_support[1]
            dist_atr_val = nearest_support[2]

            if dist_atr_val < 0.3:
                proximity = "AT_LEVEL"
                direction = "CE"
                action = "BUY_CE_AT_SUPPORT"
                confidence = min(nearest.strength * 1.2, 1.0)
            elif dist_atr_val < 0.8:
                proximity = "APPROACHING"
                direction = "CE"
                action = "PREPARE_CE_NEAR_SUPPORT"
                confidence = nearest.strength * 0.8
            else:
                proximity = "FAR"

        elif nearest_resistance:
            nearest = nearest_resistance[0]
            dist_pts = nearest_resistance[1]
            dist_atr_val = nearest_resistance[2]

            if dist_atr_val < 0.3:
                proximity = "AT_LEVEL"
                direction = "PE"
                action = "BUY_PE_AT_RESISTANCE"
                confidence = min(nearest.strength * 1.2, 1.0)
            elif dist_atr_val < 0.8:
                proximity = "APPROACHING"
                direction = "PE"
                action = "PREPARE_PE_NEAR_RESISTANCE"
                confidence = nearest.strength * 0.8
            else:
                proximity = "FAR"

        # Determine trend context from daily data
        trend = self._determine_index_trend()

        # ── EMA20 trend filter ─────────────────────────────────────────────
        # CE signals only when price is above EMA20 (uptrend)
        # PE signals only when price is below EMA20 (downtrend)
        if direction in ("CE", "PE") and self._daily_df is not None and len(self._daily_df) >= 20:
            ema20 = float(self._daily_df["close"].ewm(span=20, adjust=False).mean().iloc[-1])
            trend_up = current_price > ema20
            if direction == "CE" and not trend_up:
                logger.info(
                    f"[EMA20] CE signal blocked — price {current_price:.0f} < EMA20 {ema20:.0f} (downtrend)"
                )
                direction = "WAIT"
                action = "WAIT"
                proximity = "FAR"
                confidence = 0
            elif direction == "PE" and trend_up:
                logger.info(
                    f"[EMA20] PE signal blocked — price {current_price:.0f} > EMA20 {ema20:.0f} (uptrend)"
                )
                direction = "WAIT"
                action = "WAIT"
                proximity = "FAR"
                confidence = 0

        return IndexProximitySignal(
            index=index,
            current_price=current_price,
            nearest_level=nearest,
            distance_points=round(dist_pts, 2),
            distance_atr=round(dist_atr_val, 3),
            proximity_zone=proximity,
            direction=direction,
            confidence=round(confidence, 3),
            all_nearby_supports=[s[0] for s in nearby_supports[:5]],
            all_nearby_resistances=[r[0] for r in nearby_resistances[:5]],
            trend_context=trend,
            action=action,
        )

    # ══════════════════════════════════════════════════════════════════════
    # INTERNAL METHODS
    # ══════════════════════════════════════════════════════════════════════

    def _detect_swing_levels(
        self,
        df: pd.DataFrame,
        timeframe: LevelTimeframe,
        lookback: int = 3,
    ) -> List[IndexLevel]:
        """
        Detect swing highs and lows, then cluster them into S/R zones.
        A swing high: bar's high > highs of `lookback` bars on both sides.
        A swing low: bar's low < lows of `lookback` bars on both sides.
        """
        levels = []
        n = len(df)

        swing_highs = []
        swing_lows = []

        for i in range(lookback, n - lookback):
            # Swing high check
            is_high = True
            for j in range(1, lookback + 1):
                if df["high"].iloc[i] < df["high"].iloc[i - j] or \
                   df["high"].iloc[i] < df["high"].iloc[i + j]:
                    is_high = False
                    break
            if is_high:
                swing_highs.append({
                    "price": df["high"].iloc[i],
                    "idx": i,
                    "date": str(df.iloc[i].get("datetime", df.index[i])),
                })

            # Swing low check
            is_low = True
            for j in range(1, lookback + 1):
                if df["low"].iloc[i] > df["low"].iloc[i - j] or \
                   df["low"].iloc[i] > df["low"].iloc[i + j]:
                    is_low = False
                    break
            if is_low:
                swing_lows.append({
                    "price": df["low"].iloc[i],
                    "idx": i,
                    "date": str(df.iloc[i].get("datetime", df.index[i])),
                })

        # Cluster swing highs into resistance zones
        current_price = df["close"].iloc[-1]
        atr = self._compute_atr(df, min(14, len(df) - 1)) if len(df) > 14 else current_price * 0.01
        cluster_width = atr * 0.3  # Levels within 0.3 ATR are the same zone

        for cluster_type, swings, level_type in [
            ("resistance", swing_highs, LevelType.RESISTANCE),
            ("support", swing_lows, LevelType.SUPPORT),
        ]:
            if not swings:
                continue

            swings_sorted = sorted(swings, key=lambda x: x["price"])
            clusters = []
            current_cluster = [swings_sorted[0]]

            for s in swings_sorted[1:]:
                if s["price"] - current_cluster[0]["price"] <= cluster_width:
                    current_cluster.append(s)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [s]
            clusters.append(current_cluster)

            for cluster in clusters:
                if len(cluster) < 2:
                    continue

                avg_price = np.mean([s["price"] for s in cluster])
                touches = len(cluster)
                dates = [s["date"] for s in cluster]

                # Strength scoring
                strength = 0
                # More touches = stronger
                strength += min(touches / 5, 0.4)  # Up to 0.4
                # Higher timeframe = stronger
                tf_weight = {
                    LevelTimeframe.WEEKLY: 0.3,
                    LevelTimeframe.DAILY: 0.2,
                    LevelTimeframe.HOURLY: 0.1,
                    LevelTimeframe.FIFTEEN_MIN: 0.05,
                }
                strength += tf_weight.get(timeframe, 0.1)
                # Recency bonus
                most_recent_idx = max(s["idx"] for s in cluster)
                recency = most_recent_idx / len(df)
                strength += recency * 0.2  # Up to 0.2
                # Proximity to round numbers
                round_50 = avg_price % 50
                if round_50 < 10 or round_50 > 40:
                    strength += 0.1

                strength = min(strength, 1.0)

                levels.append(IndexLevel(
                    price=round(avg_price, 2),
                    level_type=level_type,
                    timeframe=timeframe,
                    touches=touches,
                    first_touch_date=min(dates),
                    last_touch_date=max(dates),
                    strength=round(strength, 3),
                    atr_at_level=round(atr, 2),
                    is_round_number=(avg_price % 100 < 15 or avg_price % 100 > 85),
                    fib_confluence=False,  # Set later
                    pivot_confluence=False,  # Set later
                ))

        return levels

    def _add_round_number_levels(self, daily_df: pd.DataFrame, index: str):
        """Add psychological round number levels."""
        current = daily_df["close"].iloc[-1]
        atr = self._daily_atr if self._daily_atr > 0 else current * 0.01

        # Round number intervals by index
        intervals = {
            "NIFTY": [100, 500],
            "BANKNIFTY": [100, 500, 1000],
            "FINNIFTY": [100, 500],
            "MIDCPNIFTY": [50, 100],
            "SENSEX": [500, 1000],
        }
        rounds = intervals.get(index, [100, 500])

        for interval in rounds:
            base = (current // interval) * interval
            for offset in range(-3, 4):
                level_price = base + offset * interval
                dist = abs(current - level_price)
                if dist > atr * 3:
                    continue

                # Check if this round number already exists as a swing level
                existing = [l for l in self.levels if abs(l.price - level_price) < atr * 0.2]
                if existing:
                    # Boost existing level
                    for l in existing:
                        l.is_round_number = True
                        l.strength = min(l.strength + 0.1, 1.0)
                else:
                    level_type = LevelType.SUPPORT if level_price < current else LevelType.RESISTANCE
                    self.levels.append(IndexLevel(
                        price=level_price,
                        level_type=level_type,
                        timeframe=LevelTimeframe.DAILY,
                        touches=0,
                        first_touch_date="",
                        last_touch_date="",
                        strength=0.2,  # Weak on its own
                        atr_at_level=atr,
                        is_round_number=True,
                        fib_confluence=False,
                        pivot_confluence=False,
                        notes=f"Round number {interval}",
                    ))

    def _add_fibonacci_levels(self, daily_df: pd.DataFrame):
        """Add Fibonacci retracement levels from the major swing."""
        if len(daily_df) < 60:
            return

        # Find the major swing in last 6 months
        recent = daily_df.tail(130)  # ~6 months
        swing_high = recent["high"].max()
        swing_low = recent["low"].min()
        diff = swing_high - swing_low

        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        atr = self._daily_atr

        for ratio in fib_ratios:
            # Retracement from high
            fib_level = swing_high - ratio * diff

            # Check if this aligns with existing level
            existing = [l for l in self.levels if abs(l.price - fib_level) < atr * 0.3]
            if existing:
                for l in existing:
                    l.fib_confluence = True
                    l.strength = min(l.strength + 0.15, 1.0)
                    l.notes += f" | Fib {ratio}"
            else:
                current = daily_df["close"].iloc[-1]
                level_type = LevelType.SUPPORT if fib_level < current else LevelType.RESISTANCE
                self.levels.append(IndexLevel(
                    price=round(fib_level, 2),
                    level_type=level_type,
                    timeframe=LevelTimeframe.DAILY,
                    touches=0,
                    first_touch_date="",
                    last_touch_date="",
                    strength=0.25,
                    atr_at_level=atr,
                    is_round_number=False,
                    fib_confluence=True,
                    pivot_confluence=False,
                    notes=f"Fib {ratio} of {swing_low:.0f}-{swing_high:.0f}",
                ))

    def _add_pivot_levels(self, daily_df: pd.DataFrame):
        """Add today's CPR/Pivot from yesterday's candle."""
        if len(daily_df) < 2:
            return

        prev = daily_df.iloc[-1]  # Yesterday
        h, l, c = prev["high"], prev["low"], prev["close"]

        pivot = (h + l + c) / 3
        bc = (h + l) / 2
        tc = (pivot - bc) + pivot
        r1 = 2 * pivot - l
        s1 = 2 * pivot - h
        r2 = pivot + (h - l)
        s2 = pivot - (h - l)

        atr = self._daily_atr
        current = daily_df["close"].iloc[-1]

        pivot_levels = [
            (pivot, "Pivot"),
            (bc, "CPR Bottom"),
            (tc, "CPR Top"),
            (r1, "R1"),
            (s1, "S1"),
            (r2, "R2"),
            (s2, "S2"),
        ]

        for price, name in pivot_levels:
            existing = [l for l in self.levels if abs(l.price - price) < atr * 0.2]
            if existing:
                for l in existing:
                    l.pivot_confluence = True
                    l.strength = min(l.strength + 0.1, 1.0)
                    l.notes += f" | {name}"
            else:
                level_type = LevelType.SUPPORT if price < current else LevelType.RESISTANCE
                self.levels.append(IndexLevel(
                    price=round(price, 2),
                    level_type=level_type,
                    timeframe=LevelTimeframe.DAILY,
                    touches=0,
                    first_touch_date="today",
                    last_touch_date="today",
                    strength=0.15,
                    atr_at_level=atr,
                    is_round_number=False,
                    fib_confluence=False,
                    pivot_confluence=True,
                    notes=name,
                ))

    def _merge_confluent_levels(self):
        """
        Merge levels that are very close to each other.
        Confluence = multiple timeframes agreeing on the same level.
        """
        if len(self.levels) < 2:
            return

        atr = self._daily_atr if self._daily_atr > 0 else 100
        merge_distance = atr * 0.2

        self.levels.sort(key=lambda l: l.price)
        merged = []
        i = 0

        while i < len(self.levels):
            cluster = [self.levels[i]]
            j = i + 1
            while j < len(self.levels) and self.levels[j].price - cluster[0].price <= merge_distance:
                cluster.append(self.levels[j])
                j += 1

            if len(cluster) == 1:
                merged.append(cluster[0])
            else:
                # Merge into strongest level
                best = max(cluster, key=lambda l: l.strength)
                total_touches = sum(l.touches for l in cluster)
                timeframes = set(l.timeframe for l in cluster)

                best.touches = total_touches
                best.price = round(np.mean([l.price for l in cluster]), 2)
                best.fib_confluence = any(l.fib_confluence for l in cluster)
                best.pivot_confluence = any(l.pivot_confluence for l in cluster)
                best.is_round_number = any(l.is_round_number for l in cluster)

                # Confluence bonus: multiple timeframes = stronger
                confluence_bonus = len(timeframes) * 0.1
                best.strength = min(best.strength + confluence_bonus, 1.0)
                best.notes += f" | Confluence: {', '.join(t.value for t in timeframes)}"

                merged.append(best)

            i = j

        self.levels = merged

    def _determine_index_trend(self) -> str:
        """Simple trend from stored levels: are swing lows rising or falling?"""
        supports = sorted(
            [l for l in self.levels if l.level_type == LevelType.SUPPORT],
            key=lambda l: l.last_touch_date
        )
        resistances = sorted(
            [l for l in self.levels if l.level_type == LevelType.RESISTANCE],
            key=lambda l: l.last_touch_date
        )

        if len(supports) >= 2:
            recent_supports = supports[-3:]
            prices = [s.price for s in recent_supports]
            if all(prices[i] <= prices[i+1] for i in range(len(prices)-1)):
                return "UPTREND"
            elif all(prices[i] >= prices[i+1] for i in range(len(prices)-1)):
                return "DOWNTREND"

        return "SIDEWAYS"

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int) -> float:
        """Compute ATR from OHLC data."""
        if len(df) < period + 1:
            return df["high"].iloc[-1] - df["low"].iloc[-1]

        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        # Simple moving average ATR
        if len(tr) >= period:
            return float(np.mean(tr[-period:]))
        return float(np.mean(tr))
