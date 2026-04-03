"""
=============================================================================
INSTITUTIONAL MODULE — Liquidity Sweep & Trap Detection
=============================================================================
Smart Money operates by hunting liquidity:
  1. STOP CLUSTERS form above swing highs and below swing lows
     (retail traders place SL at obvious levels)
  2. LIQUIDITY SWEEP = price spikes past the cluster, triggers stops,
     then REVERSES sharply. The wick past the level IS the sweep.
  3. INDUCEMENT = price breaks a trendline/pattern to trap breakout
     traders, then reverses. Classic false breakout.
  4. EQUAL HIGHS/LOWS = price touches same level 2-3 times.
     Retail sees "triple top/bottom". Smart money sees stop pool.

Detection algorithm:
  - Map all swing H/L from market_structure module
  - Tag equal highs/lows (within 0.15% tolerance)
  - When price wicks PAST a tagged level but closes BACK inside → SWEEP
  - Score the sweep by: wick length, volume spike, reversal speed
=============================================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class StopCluster:
    """A zone where retail stops are likely clustered."""
    price: float
    side: str               # "ABOVE" = stops above (shorts' SL) or "BELOW" (longs' SL)
    source: str             # "SWING_HIGH", "SWING_LOW", "EQUAL_HIGHS", "EQUAL_LOWS", "TRENDLINE"
    touch_count: int        # How many times price has tested this level
    last_touch_bar: int
    cluster_strength: float # 0-1: more touches + more obvious = stronger cluster
    estimated_stops: str    # "HIGH" / "MEDIUM" / "LOW" based on pattern visibility


@dataclass
class LiquiditySweep:
    """Detected liquidity sweep event."""
    bar_index: int
    timestamp: str
    sweep_type: str          # "SWEEP_HIGH" (swept above, reversal down) or "SWEEP_LOW" (swept below, reversal up)
    cluster_price: float     # The level that was swept
    wick_high: float         # How far the wick went past the cluster
    wick_low: float
    close: float             # Where price actually closed
    wick_past_cluster: float # Points the wick extended beyond the cluster
    closed_back_inside: bool # True = sweep confirmed (wick past, close inside)
    volume_ratio: float      # Volume vs 20-bar avg
    reversal_candle: bool    # Next candle reversed strongly?
    score: float             # 0-10 composite quality score
    trade_direction: str     # "CE" (if swept lows, expect bounce) or "PE" (swept highs, expect drop)
    entry_zone: float        # Suggested entry price
    sl_zone: float           # SL beyond the sweep wick
    target_zone: float       # First target (back to equilibrium)


@dataclass
class Inducement:
    """False breakout that traps retail traders."""
    bar_index: int
    timestamp: str
    inducement_type: str     # "FALSE_BREAKOUT_HIGH", "FALSE_BREAKOUT_LOW", "TRENDLINE_TRAP"
    trap_level: float        # The level that appeared to break
    trap_direction: str      # "BULL_TRAP" (looked bullish, actually bearish) or "BEAR_TRAP"
    wick_past: float         # How far past the level price went
    close: float
    volume_on_trap: float    # Volume ratio on the trap candle
    score: float             # 0-10
    trade_direction: str     # What to trade AFTER the trap
    description: str


class LiquidityEngine:
    """
    Detects stop clusters, liquidity sweeps, and inducement traps.
    
    Usage:
        engine = LiquidityEngine()
        sweeps, inducements, clusters = engine.analyze(df_15min, swing_highs, swing_lows)
        
        # Get actionable signal at current bar
        signal = engine.check_current_bar(df, bar_index, clusters)
    """

    def __init__(self, equal_level_tolerance_pct: float = 0.15,
                 min_sweep_wick_pct: float = 0.05,
                 min_volume_ratio: float = 1.3):
        self.eq_tol_pct = equal_level_tolerance_pct / 100
        self.min_sweep_wick = min_sweep_wick_pct / 100
        self.min_vol_ratio = min_volume_ratio

    def analyze(
        self,
        df: pd.DataFrame,
        swing_highs: list = None,
        swing_lows: list = None,
        lookback: int = 5,
    ) -> Tuple[List[LiquiditySweep], List[Inducement], List[StopCluster]]:
        """Full analysis pipeline."""
        if len(df) < 30:
            return [], [], []

        # Detect swing points if not provided
        if swing_highs is None or swing_lows is None:
            swing_highs, swing_lows = self._detect_swings(df, lookback)

        # Step 1: Map stop clusters
        clusters = self._map_stop_clusters(df, swing_highs, swing_lows)

        # Step 2: Detect sweeps
        sweeps = self._detect_sweeps(df, clusters)

        # Step 3: Detect inducements
        inducements = self._detect_inducements(df, swing_highs, swing_lows)

        return sweeps, inducements, clusters

    def check_current_bar(
        self,
        df: pd.DataFrame,
        bar_index: int,
        clusters: List[StopCluster] = None,
        swing_highs: list = None,
        swing_lows: list = None,
    ) -> Optional[LiquiditySweep]:
        """Check if current bar is a liquidity sweep (real-time usage)."""
        if bar_index < 2 or bar_index >= len(df):
            return None

        if clusters is None:
            if swing_highs is None or swing_lows is None:
                swing_highs, swing_lows = self._detect_swings(df.iloc[:bar_index+1], 3)
            clusters = self._map_stop_clusters(df.iloc[:bar_index+1], swing_highs, swing_lows)

        row = df.iloc[bar_index]
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        opn = float(row["open"])

        vol_avg = df["volume"].iloc[max(0, bar_index-20):bar_index].mean()
        vol_ratio = float(row["volume"]) / max(vol_avg, 1)

        for cluster in clusters:
            # SWEEP HIGH: wick goes above cluster, closes back below
            if cluster.side == "ABOVE" and high > cluster.price and close < cluster.price:
                wick_past = high - cluster.price
                wick_pct = wick_past / cluster.price

                if wick_pct < self.min_sweep_wick:
                    continue

                # Reversal candle: bearish close (close < open) with upper wick
                is_reversal = close < opn and (high - max(close, opn)) > abs(close - opn) * 0.5

                score = self._score_sweep(wick_pct, vol_ratio, is_reversal, cluster)

                if score >= 4:
                    entry = close - (high - close) * 0.1  # Enter near close
                    sl = high + wick_past * 0.3           # SL above the sweep wick
                    target = cluster.price - (high - cluster.price) * 2  # 2x the sweep range

                    return LiquiditySweep(
                        bar_index=bar_index,
                        timestamp=str(row.get("datetime", "")),
                        sweep_type="SWEEP_HIGH",
                        cluster_price=cluster.price,
                        wick_high=high,
                        wick_low=low,
                        close=close,
                        wick_past_cluster=round(wick_past, 2),
                        closed_back_inside=True,
                        volume_ratio=round(vol_ratio, 2),
                        reversal_candle=is_reversal,
                        score=round(score, 1),
                        trade_direction="PE",
                        entry_zone=round(entry, 2),
                        sl_zone=round(sl, 2),
                        target_zone=round(target, 2),
                    )

            # SWEEP LOW: wick goes below cluster, closes back above
            elif cluster.side == "BELOW" and low < cluster.price and close > cluster.price:
                wick_past = cluster.price - low
                wick_pct = wick_past / cluster.price

                if wick_pct < self.min_sweep_wick:
                    continue

                is_reversal = close > opn and (min(close, opn) - low) > abs(close - opn) * 0.5

                score = self._score_sweep(wick_pct, vol_ratio, is_reversal, cluster)

                if score >= 4:
                    entry = close + (close - low) * 0.1
                    sl = low - wick_past * 0.3
                    target = cluster.price + (cluster.price - low) * 2

                    return LiquiditySweep(
                        bar_index=bar_index,
                        timestamp=str(row.get("datetime", "")),
                        sweep_type="SWEEP_LOW",
                        cluster_price=cluster.price,
                        wick_high=high,
                        wick_low=low,
                        close=close,
                        wick_past_cluster=round(wick_past, 2),
                        closed_back_inside=True,
                        volume_ratio=round(vol_ratio, 2),
                        reversal_candle=is_reversal,
                        score=round(score, 1),
                        trade_direction="CE",
                        entry_zone=round(entry, 2),
                        sl_zone=round(sl, 2),
                        target_zone=round(target, 2),
                    )

        return None

    # ══════════════════════════════════════════════════════════════
    # STOP CLUSTER MAPPING
    # ══════════════════════════════════════════════════════════════

    def _map_stop_clusters(self, df, swing_highs, swing_lows) -> List[StopCluster]:
        clusters = []

        # Cluster above each swing high (short sellers' SL lives here)
        for sh in swing_highs:
            price = sh["price"] if isinstance(sh, dict) else sh.price
            idx = sh["idx"] if isinstance(sh, dict) else sh.idx
            clusters.append(StopCluster(
                price=round(price, 2), side="ABOVE", source="SWING_HIGH",
                touch_count=1, last_touch_bar=idx,
                cluster_strength=0.5, estimated_stops="MEDIUM",
            ))

        # Cluster below each swing low (long holders' SL lives here)
        for sl in swing_lows:
            price = sl["price"] if isinstance(sl, dict) else sl.price
            idx = sl["idx"] if isinstance(sl, dict) else sl.idx
            clusters.append(StopCluster(
                price=round(price, 2), side="BELOW", source="SWING_LOW",
                touch_count=1, last_touch_bar=idx,
                cluster_strength=0.5, estimated_stops="MEDIUM",
            ))

        # Equal highs / equal lows — retail sees "double top", SM sees stop pool
        eq_highs = self._find_equal_levels([s["price"] if isinstance(s, dict) else s.price for s in swing_highs])
        for level, count in eq_highs:
            clusters.append(StopCluster(
                price=round(level, 2), side="ABOVE", source="EQUAL_HIGHS",
                touch_count=count, last_touch_bar=0,
                cluster_strength=min(count * 0.3, 1.0), estimated_stops="HIGH",
            ))

        eq_lows = self._find_equal_levels([s["price"] if isinstance(s, dict) else s.price for s in swing_lows])
        for level, count in eq_lows:
            clusters.append(StopCluster(
                price=round(level, 2), side="BELOW", source="EQUAL_LOWS",
                touch_count=count, last_touch_bar=0,
                cluster_strength=min(count * 0.3, 1.0), estimated_stops="HIGH",
            ))

        # Merge nearby clusters
        clusters = self._merge_clusters(clusters, df)
        return clusters

    def _find_equal_levels(self, prices: list) -> List[Tuple[float, int]]:
        """Find price levels that have been touched multiple times (equal highs/lows)."""
        if len(prices) < 2:
            return []

        levels = []
        used = set()
        for i in range(len(prices)):
            if i in used:
                continue
            cluster = [prices[i]]
            for j in range(i + 1, len(prices)):
                if j in used:
                    continue
                if abs(prices[j] - prices[i]) / max(prices[i], 1) <= self.eq_tol_pct:
                    cluster.append(prices[j])
                    used.add(j)
            if len(cluster) >= 2:
                levels.append((np.mean(cluster), len(cluster)))
                used.add(i)

        return levels

    def _merge_clusters(self, clusters: List[StopCluster], df) -> List[StopCluster]:
        """Merge clusters that are very close together."""
        if not clusters:
            return clusters

        price = df["close"].iloc[-1] if len(df) > 0 else 1
        merge_dist = price * 0.002  # 0.2%

        sorted_c = sorted(clusters, key=lambda c: c.price)
        merged = []
        i = 0
        while i < len(sorted_c):
            group = [sorted_c[i]]
            j = i + 1
            while j < len(sorted_c) and sorted_c[j].price - group[0].price <= merge_dist:
                group.append(sorted_c[j])
                j += 1

            best = max(group, key=lambda c: c.cluster_strength)
            best.touch_count = sum(c.touch_count for c in group)
            best.cluster_strength = min(best.cluster_strength + len(group) * 0.1, 1.0)
            if best.touch_count >= 3:
                best.estimated_stops = "HIGH"
            merged.append(best)
            i = j

        return merged

    # ══════════════════════════════════════════════════════════════
    # SWEEP DETECTION (historical)
    # ══════════════════════════════════════════════════════════════

    def _detect_sweeps(self, df, clusters) -> List[LiquiditySweep]:
        sweeps = []
        vol_avg = df["volume"].rolling(20, min_periods=5).mean()

        for i in range(20, len(df)):
            row = df.iloc[i]
            high, low, close, opn = float(row["high"]), float(row["low"]), float(row["close"]), float(row["open"])
            vr = float(row["volume"]) / max(float(vol_avg.iloc[i]), 1)

            for cluster in clusters:
                if cluster.last_touch_bar >= i:
                    continue

                if cluster.side == "ABOVE" and high > cluster.price and close < cluster.price:
                    wick_past = high - cluster.price
                    if wick_past / cluster.price < self.min_sweep_wick:
                        continue
                    is_rev = close < opn
                    score = self._score_sweep(wick_past / cluster.price, vr, is_rev, cluster)
                    if score >= 4:
                        sweeps.append(LiquiditySweep(
                            bar_index=i, timestamp=str(row.get("datetime", "")),
                            sweep_type="SWEEP_HIGH", cluster_price=cluster.price,
                            wick_high=high, wick_low=low, close=close,
                            wick_past_cluster=round(wick_past, 2),
                            closed_back_inside=True, volume_ratio=round(vr, 2),
                            reversal_candle=is_rev, score=round(score, 1),
                            trade_direction="PE",
                            entry_zone=round(close, 2),
                            sl_zone=round(high + wick_past * 0.3, 2),
                            target_zone=round(cluster.price - wick_past * 2, 2),
                        ))

                elif cluster.side == "BELOW" and low < cluster.price and close > cluster.price:
                    wick_past = cluster.price - low
                    if wick_past / cluster.price < self.min_sweep_wick:
                        continue
                    is_rev = close > opn
                    score = self._score_sweep(wick_past / cluster.price, vr, is_rev, cluster)
                    if score >= 4:
                        sweeps.append(LiquiditySweep(
                            bar_index=i, timestamp=str(row.get("datetime", "")),
                            sweep_type="SWEEP_LOW", cluster_price=cluster.price,
                            wick_high=high, wick_low=low, close=close,
                            wick_past_cluster=round(wick_past, 2),
                            closed_back_inside=True, volume_ratio=round(vr, 2),
                            reversal_candle=is_rev, score=round(score, 1),
                            trade_direction="CE",
                            entry_zone=round(close, 2),
                            sl_zone=round(low - wick_past * 0.3, 2),
                            target_zone=round(cluster.price + wick_past * 2, 2),
                        ))

        return sweeps

    # ══════════════════════════════════════════════════════════════
    # INDUCEMENT DETECTION
    # ══════════════════════════════════════════════════════════════

    def _detect_inducements(self, df, swing_highs, swing_lows) -> List[Inducement]:
        """Detect false breakouts that trap retail traders."""
        inducements = []
        n = len(df)

        for i in range(5, n - 1):
            row = df.iloc[i]
            next_row = df.iloc[i + 1] if i + 1 < n else None
            high, low, close, opn = float(row["high"]), float(row["low"]), float(row["close"]), float(row["open"])

            vol_avg = df["volume"].iloc[max(0, i-20):i].mean()
            vr = float(row["volume"]) / max(vol_avg, 1)

            # Check each recent swing high for false breakout
            for sh in swing_highs:
                sh_price = sh["price"] if isinstance(sh, dict) else sh.price
                sh_idx = sh["idx"] if isinstance(sh, dict) else sh.idx
                if sh_idx >= i or sh_idx < i - 60:
                    continue

                # False breakout above: high exceeds swing high but close is below
                if high > sh_price and close < sh_price:
                    wick = high - sh_price
                    body = abs(close - opn)

                    # Key: big wick, small body = rejection = inducement
                    if wick > body * 0.5 and close < opn:
                        score = min(3 + (wick / max(body, 0.01)) * 1.5 + (vr - 1) * 1.5, 10)
                        inducements.append(Inducement(
                            bar_index=i, timestamp=str(row.get("datetime", "")),
                            inducement_type="FALSE_BREAKOUT_HIGH",
                            trap_level=sh_price, trap_direction="BULL_TRAP",
                            wick_past=round(wick, 2), close=close,
                            volume_on_trap=round(vr, 2), score=round(score, 1),
                            trade_direction="PE",
                            description=f"Swept {sh_price:.0f} swing high, rejected. Wick {wick:.1f}pts past.",
                        ))

            # Check each swing low for false breakdown
            for sl in swing_lows:
                sl_price = sl["price"] if isinstance(sl, dict) else sl.price
                sl_idx = sl["idx"] if isinstance(sl, dict) else sl.idx
                if sl_idx >= i or sl_idx < i - 60:
                    continue

                if low < sl_price and close > sl_price:
                    wick = sl_price - low
                    body = abs(close - opn)

                    if wick > body * 0.5 and close > opn:
                        score = min(3 + (wick / max(body, 0.01)) * 1.5 + (vr - 1) * 1.5, 10)
                        inducements.append(Inducement(
                            bar_index=i, timestamp=str(row.get("datetime", "")),
                            inducement_type="FALSE_BREAKOUT_LOW",
                            trap_level=sl_price, trap_direction="BEAR_TRAP",
                            wick_past=round(wick, 2), close=close,
                            volume_on_trap=round(vr, 2), score=round(score, 1),
                            trade_direction="CE",
                            description=f"Swept {sl_price:.0f} swing low, bounced. Wick {wick:.1f}pts past.",
                        ))

        return inducements

    # ══════════════════════════════════════════════════════════════
    # SCORING
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _score_sweep(wick_pct, vol_ratio, is_reversal, cluster) -> float:
        score = 0
        if wick_pct > 0.003: score += 2
        elif wick_pct > 0.001: score += 1
        if vol_ratio > 2.0: score += 2.5
        elif vol_ratio > 1.5: score += 1.5
        elif vol_ratio > 1.2: score += 1
        if is_reversal: score += 2
        score += cluster.cluster_strength * 2
        if cluster.estimated_stops == "HIGH": score += 1.5
        elif cluster.estimated_stops == "MEDIUM": score += 0.5
        return min(score, 10)

    @staticmethod
    def _detect_swings(df, lookback=3):
        highs, lows = [], []
        for i in range(lookback, len(df) - lookback):
            is_high = all(df["high"].iloc[i] >= df["high"].iloc[i-j] and
                          df["high"].iloc[i] >= df["high"].iloc[i+j] for j in range(1, lookback+1))
            is_low = all(df["low"].iloc[i] <= df["low"].iloc[i-j] and
                         df["low"].iloc[i] <= df["low"].iloc[i+j] for j in range(1, lookback+1))
            if is_high:
                highs.append({"idx": i, "price": float(df["high"].iloc[i])})
            if is_low:
                lows.append({"idx": i, "price": float(df["low"].iloc[i])})
        return highs, lows
