"""
=============================================================================
SCALPER PRO v2 — LAYER 2: Option Premium Swing Detector
=============================================================================
Once Layer 1 (Index Level Marker) says "Buy CE" or "Buy PE",
this module:

  1. Takes the selected CE or PE option's premium chart
  2. Marks swing highs/lows on 1-min, 5-min, and 15-min
  3. Finds SUPPORT zones on premium chart (where it bounced before)
  4. Finds RESISTANCE zones (target to exit)
  5. Generates precise entry at premium support, SL below it, target at resistance

The KEY insight: we're reading the OPTION's own price action.
When NIFTY 23500 CE premium drops to ₹120 for the 3rd time and bounces,
₹120 IS support. Buy at ₹122, SL ₹108, target ₹140.

=============================================================================
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class PremiumSwingPoint:
    """A swing high or low on the option premium chart."""
    bar_index: int
    price: float
    time: str
    timeframe: str      # "1min", "5min", "15min"
    point_type: str     # "HIGH" or "LOW"


@dataclass
class PremiumLevel:
    """A support or resistance zone on the option premium chart."""
    price: float
    level_type: str         # "SUPPORT" or "RESISTANCE"
    touches: int
    timeframe: str          # Primary timeframe where detected
    multi_tf_confirmed: bool  # Visible on multiple timeframes?
    first_touch_bar: int
    last_touch_bar: int
    zone_low: float         # Zone is not a line, it's a band
    zone_high: float
    strength: float         # 0-1
    bounce_magnitudes: List[float]  # How much premium bounced each time


@dataclass
class PremiumSwingSetup:
    """
    Complete trade setup from premium swing analysis.
    This is the OUTPUT of Layer 2 — a ready-to-execute trade.
    """
    # Option details
    option_type: str            # "CE" or "PE"
    strike: int
    index: str

    # Premium chart analysis
    current_premium: float
    entry_premium: float        # Where to buy the option
    stoploss_premium: float     # Below the support zone
    target_premium: float       # At the resistance zone
    sl_points: float            # Risk in points
    target_points: float        # Reward in points
    risk_reward: float

    # Support/Resistance context
    entry_at_level: PremiumLevel     # The support we're buying at
    target_at_level: PremiumLevel    # The resistance we're targeting

    # Multi-timeframe confirmation
    tf_1min_confirms: bool
    tf_5min_confirms: bool
    tf_15min_confirms: bool
    confirmation_count: int     # How many timeframes agree

    # Trade quality
    setup_quality: str          # "A+", "A", "B", "C"
    confidence: float           # 0-1
    reasons: List[str]

    # Premium trend
    premium_trend: str          # "UP", "DOWN", "SIDEWAYS"
    premium_trend_action: str   # "BUY_ON_DIP", "SELL_ON_RISE", "RANGE_TRADE"


class PremiumSwingDetector:
    """
    Detects swing levels on option premium charts.

    Usage:
        detector = PremiumSwingDetector()

        setup = detector.analyze(
            premium_1min=option_1min_ohlcv,
            premium_5min=option_5min_ohlcv,
            premium_15min=option_15min_ohlcv,
            option_type="CE",
            strike=23500,
            index="NIFTY"
        )

        if setup and setup.setup_quality in ("A+", "A"):
            execute_trade(setup)
    """

    def __init__(self, swing_lookback: int = 3, zone_pct: float = 1.0):
        self.swing_lookback = swing_lookback
        self.zone_pct = zone_pct  # % width for clustering swings into zones

    def analyze(
        self,
        premium_1min: pd.DataFrame,
        premium_5min: Optional[pd.DataFrame] = None,
        premium_15min: Optional[pd.DataFrame] = None,
        option_type: str = "CE",
        strike: int = 0,
        index: str = "NIFTY",
    ) -> Optional[PremiumSwingSetup]:
        """
        Full analysis pipeline:
        1. Detect swings on each timeframe
        2. Build S/R zones from premium bounces
        3. Check multi-TF confirmation
        4. Generate setup if conditions are met
        """
        if premium_1min is None or len(premium_1min) < 20:
            return None

        # ── Step 1: Detect swing points on each timeframe ──────────

        swings_1min = self._detect_swings(premium_1min, "1min")
        swings_5min = self._detect_swings(premium_5min, "5min") if premium_5min is not None else ([], [])
        swings_15min = self._detect_swings(premium_15min, "15min") if premium_15min is not None else ([], [])

        # ── Step 2: Build S/R zones from each timeframe ────────────

        zones_1min = self._build_zones(swings_1min[0], swings_1min[1], premium_1min, "1min")
        zones_5min = self._build_zones(swings_5min[0], swings_5min[1], premium_5min, "5min") if premium_5min is not None else []
        zones_15min = self._build_zones(swings_15min[0], swings_15min[1], premium_15min, "15min") if premium_15min is not None else []

        all_zones = zones_1min + zones_5min + zones_15min

        if len(all_zones) < 2:
            logger.debug(f"Not enough zones found for {strike} {option_type}")
            return None

        # ── Step 3: Merge zones across timeframes ──────────────────

        merged_zones = self._merge_multi_tf_zones(all_zones)

        # ── Step 4: Determine premium trend ────────────────────────

        premium_trend, trend_action = self._analyze_premium_trend(
            swings_1min, swings_5min, premium_1min
        )

        # ── Step 5: Find the trade setup ───────────────────────────

        current_premium = premium_1min["close"].iloc[-1]
        setup = self._find_setup(
            current_premium, merged_zones, premium_trend, trend_action,
            option_type, strike, index,
            zones_1min, zones_5min, zones_15min
        )

        return setup

    def _detect_swings(
        self,
        df: pd.DataFrame,
        timeframe: str,
    ) -> Tuple[List[PremiumSwingPoint], List[PremiumSwingPoint]]:
        """Detect swing highs and lows on premium data."""
        if df is None or len(df) < self.swing_lookback * 2 + 1:
            return [], []

        highs = []
        lows = []
        lb = self.swing_lookback
        n = len(df)

        for i in range(lb, n - lb):
            # Swing high: current high > all neighbors
            is_high = True
            for j in range(1, lb + 1):
                if df["high"].iloc[i] < df["high"].iloc[i - j] or \
                   df["high"].iloc[i] < df["high"].iloc[i + j]:
                    is_high = False
                    break

            # Swing low: current low < all neighbors
            is_low = True
            for j in range(1, lb + 1):
                if df["low"].iloc[i] > df["low"].iloc[i - j] or \
                   df["low"].iloc[i] > df["low"].iloc[i + j]:
                    is_low = False
                    break

            time_str = str(df.iloc[i].get("datetime", df.index[i]))

            if is_high:
                highs.append(PremiumSwingPoint(
                    bar_index=i, price=df["high"].iloc[i],
                    time=time_str, timeframe=timeframe, point_type="HIGH"
                ))
            if is_low:
                lows.append(PremiumSwingPoint(
                    bar_index=i, price=df["low"].iloc[i],
                    time=time_str, timeframe=timeframe, point_type="LOW"
                ))

        return highs, lows

    def _build_zones(
        self,
        swing_highs: List[PremiumSwingPoint],
        swing_lows: List[PremiumSwingPoint],
        df: pd.DataFrame,
        timeframe: str,
    ) -> List[PremiumLevel]:
        """Cluster swing points into S/R zones."""
        if df is None or len(df) == 0:
            return []

        zones = []
        current_price = df["close"].iloc[-1]

        # Build zones from swing lows (supports)
        zones.extend(self._cluster_into_zones(
            swing_lows, "SUPPORT", timeframe, current_price, df
        ))

        # Build zones from swing highs (resistances)
        zones.extend(self._cluster_into_zones(
            swing_highs, "RESISTANCE", timeframe, current_price, df
        ))

        return zones

    def _cluster_into_zones(
        self,
        swings: List[PremiumSwingPoint],
        level_type: str,
        timeframe: str,
        current_price: float,
        df: pd.DataFrame,
    ) -> List[PremiumLevel]:
        """Cluster nearby swing points into zones."""
        if not swings:
            return []

        sorted_swings = sorted(swings, key=lambda s: s.price)
        cluster_width = current_price * self.zone_pct / 100

        zones = []
        cluster = [sorted_swings[0]]

        for swing in sorted_swings[1:]:
            if swing.price - cluster[0].price <= cluster_width:
                cluster.append(swing)
            else:
                if len(cluster) >= 2:
                    zones.append(self._cluster_to_zone(
                        cluster, level_type, timeframe, df
                    ))
                cluster = [swing]

        if len(cluster) >= 2:
            zones.append(self._cluster_to_zone(
                cluster, level_type, timeframe, df
            ))

        return zones

    def _cluster_to_zone(
        self,
        cluster: List[PremiumSwingPoint],
        level_type: str,
        timeframe: str,
        df: pd.DataFrame,
    ) -> PremiumLevel:
        """Convert a cluster of swing points into a zone with bounce magnitude."""
        prices = [s.price for s in cluster]
        avg_price = np.mean(prices)
        zone_low = min(prices)
        zone_high = max(prices)

        # Calculate bounce magnitudes (how much premium moved after hitting this level)
        bounce_magnitudes = []
        for swing in cluster:
            idx = swing.bar_index
            if level_type == "SUPPORT" and idx + 10 < len(df):
                # How much did premium rise after hitting support?
                future_high = df["high"].iloc[idx:idx+10].max()
                bounce = future_high - swing.price
                bounce_magnitudes.append(round(bounce, 2))
            elif level_type == "RESISTANCE" and idx + 10 < len(df):
                # How much did premium drop after hitting resistance?
                future_low = df["low"].iloc[idx:idx+10].min()
                bounce = swing.price - future_low
                bounce_magnitudes.append(round(bounce, 2))

        # Strength based on touches, bounce magnitude, timeframe
        touches = len(cluster)
        avg_bounce = np.mean(bounce_magnitudes) if bounce_magnitudes else 0

        strength = 0
        strength += min(touches / 4, 0.3)           # Touches: up to 0.3
        strength += min(avg_bounce / 20, 0.3)        # Bounce size: up to 0.3
        tf_weight = {"15min": 0.25, "5min": 0.15, "1min": 0.1}
        strength += tf_weight.get(timeframe, 0.1)    # Timeframe: up to 0.25
        strength = min(strength, 1.0)

        return PremiumLevel(
            price=round(avg_price, 2),
            level_type=level_type,
            touches=touches,
            timeframe=timeframe,
            multi_tf_confirmed=False,
            first_touch_bar=min(s.bar_index for s in cluster),
            last_touch_bar=max(s.bar_index for s in cluster),
            zone_low=round(zone_low, 2),
            zone_high=round(zone_high, 2),
            strength=round(strength, 3),
            bounce_magnitudes=bounce_magnitudes,
        )

    def _merge_multi_tf_zones(self, all_zones: List[PremiumLevel]) -> List[PremiumLevel]:
        """
        Merge zones from different timeframes that overlap.
        A zone confirmed on multiple timeframes is stronger.
        """
        if len(all_zones) < 2:
            return all_zones

        sorted_zones = sorted(all_zones, key=lambda z: z.price)
        merged = []
        i = 0

        while i < len(sorted_zones):
            cluster = [sorted_zones[i]]
            j = i + 1

            while j < len(sorted_zones):
                # Check if zones overlap
                z1 = cluster[0]
                z2 = sorted_zones[j]

                if z2.zone_low <= z1.zone_high * 1.005 and z1.level_type == z2.level_type:
                    cluster.append(z2)
                    j += 1
                else:
                    break

            if len(cluster) == 1:
                merged.append(cluster[0])
            else:
                # Merge: take strongest, boost with multi-TF confirmation
                best = max(cluster, key=lambda z: z.strength)
                timeframes = set(z.timeframe for z in cluster)
                total_touches = sum(z.touches for z in cluster)
                all_bounces = [b for z in cluster for b in z.bounce_magnitudes]

                best.touches = total_touches
                best.multi_tf_confirmed = len(timeframes) > 1
                best.bounce_magnitudes = all_bounces
                best.zone_low = min(z.zone_low for z in cluster)
                best.zone_high = max(z.zone_high for z in cluster)

                # Multi-TF bonus
                if best.multi_tf_confirmed:
                    best.strength = min(best.strength + len(timeframes) * 0.1, 1.0)

                merged.append(best)

            i = j

        return merged

    def _analyze_premium_trend(
        self,
        swings_1min: Tuple,
        swings_5min: Tuple,
        df: pd.DataFrame,
    ) -> Tuple[str, str]:
        """
        Determine premium trend from swing structure.
        HH + HL = uptrend → Buy on Dip (buy at support)
        LH + LL = downtrend → Sell on Rise (don't buy, or buy PE)
        """
        # Use 5-min swings if available, else 1-min
        highs = swings_5min[0] if swings_5min and swings_5min[0] else swings_1min[0]
        lows = swings_5min[1] if swings_5min and swings_5min[1] else swings_1min[1]

        if len(highs) < 2 or len(lows) < 2:
            return "SIDEWAYS", "RANGE_TRADE"

        recent_highs = highs[-4:]
        recent_lows = lows[-4:]

        hh = sum(1 for i in range(1, len(recent_highs))
                 if recent_highs[i].price > recent_highs[i-1].price)
        hl = sum(1 for i in range(1, len(recent_lows))
                 if recent_lows[i].price > recent_lows[i-1].price)
        lh = sum(1 for i in range(1, len(recent_highs))
                 if recent_highs[i].price < recent_highs[i-1].price)
        ll = sum(1 for i in range(1, len(recent_lows))
                 if recent_lows[i].price < recent_lows[i-1].price)

        if hh >= 2 and hl >= 1:
            return "UP", "BUY_ON_DIP"
        elif lh >= 2 and ll >= 1:
            return "DOWN", "SELL_ON_RISE"
        elif hh >= 1 and hl >= 1:
            return "MILD_UP", "BUY_ON_DIP"
        elif lh >= 1 and ll >= 1:
            return "MILD_DOWN", "SELL_ON_RISE"
        else:
            return "SIDEWAYS", "RANGE_TRADE"

    def _find_setup(
        self,
        current_premium: float,
        zones: List[PremiumLevel],
        premium_trend: str,
        trend_action: str,
        option_type: str,
        strike: int,
        index: str,
        zones_1min: List,
        zones_5min: List,
        zones_15min: List,
    ) -> Optional[PremiumSwingSetup]:
        """
        Find a tradeable setup from the zones.

        Logic:
        - Find nearest SUPPORT below current price (entry zone)
        - Find nearest RESISTANCE above current price (target zone)
        - If current price is within the support zone → ENTRY
        - SL just below support zone low
        - Target at resistance zone
        """
        supports = sorted(
            [z for z in zones if z.level_type == "SUPPORT" and z.zone_high <= current_premium * 1.02],
            key=lambda z: z.price, reverse=True  # Nearest first
        )
        resistances = sorted(
            [z for z in zones if z.level_type == "RESISTANCE" and z.zone_low >= current_premium * 0.98],
            key=lambda z: z.price  # Nearest first
        )

        if not supports or not resistances:
            return None

        entry_zone = supports[0]
        target_zone = resistances[0]

        # Check if price is at or near support
        dist_to_support = current_premium - entry_zone.zone_high
        zone_height = entry_zone.zone_high - entry_zone.zone_low
        approach_buffer = max(zone_height, 3)  # Allow entry slightly above zone

        if dist_to_support > approach_buffer * 2:
            # Too far from support — not a setup
            return None

        # Calculate entry, SL, target
        entry = round(max(current_premium, entry_zone.zone_high + 1), 2)
        sl = round(entry_zone.zone_low - max(zone_height * 0.5, 2), 2)
        target = round(target_zone.zone_low, 2)

        sl_points = round(entry - sl, 2)
        target_points = round(target - entry, 2)

        if sl_points <= 0 or target_points <= 0:
            return None

        rr = round(target_points / sl_points, 2)

        # ── Multi-TF confirmation ──────────────────────────────────
        def zone_exists_at(zones_list, price, tolerance=3):
            return any(abs(z.price - price) <= tolerance and z.level_type == "SUPPORT"
                       for z in zones_list)

        tf_1min = zone_exists_at(zones_1min, entry_zone.price)
        tf_5min = zone_exists_at(zones_5min, entry_zone.price)
        tf_15min = zone_exists_at(zones_15min, entry_zone.price)
        conf_count = sum([tf_1min, tf_5min, tf_15min])

        # ── Setup quality grading ──────────────────────────────────
        reasons = []
        quality_score = 0

        # Touches
        if entry_zone.touches >= 3:
            quality_score += 3
            reasons.append(f"Strong support: {entry_zone.touches} touches at ₹{entry_zone.price}")
        elif entry_zone.touches >= 2:
            quality_score += 2
            reasons.append(f"Support: {entry_zone.touches} touches at ₹{entry_zone.price}")

        # Multi-TF
        if conf_count >= 3:
            quality_score += 3
            reasons.append("All 3 timeframes confirm support")
        elif conf_count >= 2:
            quality_score += 2
            reasons.append(f"{conf_count} timeframes confirm support")

        # Bounce history
        avg_bounce = np.mean(entry_zone.bounce_magnitudes) if entry_zone.bounce_magnitudes else 0
        if avg_bounce >= 15:
            quality_score += 2
            reasons.append(f"Avg bounce: {avg_bounce:.1f} pts from this level")
        elif avg_bounce >= 10:
            quality_score += 1
            reasons.append(f"Avg bounce: {avg_bounce:.1f} pts")

        # Risk-reward
        if rr >= 1.5:
            quality_score += 2
            reasons.append(f"R:R = 1:{rr}")
        elif rr >= 1.0:
            quality_score += 1
            reasons.append(f"R:R = 1:{rr}")

        # Trend alignment
        if premium_trend in ("UP", "MILD_UP") and option_type == "CE":
            quality_score += 2
            reasons.append(f"CE in premium uptrend → Buy on Dip")
        elif premium_trend in ("DOWN", "MILD_DOWN") and option_type == "PE":
            quality_score += 2
            reasons.append(f"PE in premium uptrend (index falling) → Buy on Dip")
        elif premium_trend == "SIDEWAYS":
            quality_score += 1
            reasons.append("Range trade — support to resistance")

        # SL tightness (the key advantage of this strategy)
        if sl_points <= 15:
            quality_score += 2
            reasons.append(f"Tight SL: only {sl_points} pts risk")
        elif sl_points <= 20:
            quality_score += 1
            reasons.append(f"Acceptable SL: {sl_points} pts")

        # Grade
        if quality_score >= 12:
            grade = "A+"
        elif quality_score >= 9:
            grade = "A"
        elif quality_score >= 6:
            grade = "B"
        else:
            grade = "C"

        confidence = min(quality_score / 15, 1.0)

        # Only return if minimum quality met
        if grade == "C" and rr < 1.0:
            return None

        return PremiumSwingSetup(
            option_type=option_type,
            strike=strike,
            index=index,
            current_premium=current_premium,
            entry_premium=entry,
            stoploss_premium=sl,
            target_premium=target,
            sl_points=sl_points,
            target_points=target_points,
            risk_reward=rr,
            entry_at_level=entry_zone,
            target_at_level=target_zone,
            tf_1min_confirms=tf_1min,
            tf_5min_confirms=tf_5min,
            tf_15min_confirms=tf_15min,
            confirmation_count=conf_count,
            setup_quality=grade,
            confidence=round(confidence, 3),
            reasons=reasons,
            premium_trend=premium_trend,
            premium_trend_action=trend_action,
        )
