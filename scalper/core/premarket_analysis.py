"""
=============================================================================
SCALPER PRO v2 — Pre-Market Analysis Engine
=============================================================================
Runs at 09:10 AM (before market opens) for each selected index.

What it does:
  1. Fetches Weekly / Daily / 4H / 1H / 15min data
  2. Computes EMA20 + Supertrend on every timeframe → trend label
  3. Re-uses IndexLevelMarker to mark all S/R, Fib, CPR, Round numbers
  4. Identifies the nearest levels within 2 ATR of current price
  5. Computes a weighted bias score (BULLISH / BEARISH / NEUTRAL)
  6. Returns a PremarketReport ready to be sent via Telegram

Design notes:
  - Uses FreeDataFetcher (Yahoo Finance) — always available before open
  - Resamples daily → weekly  and  1H → 4H  (Yahoo doesn't provide these)
  - Reuses IndexLevelMarker.mark_levels() without modifying it
  - All new logic is self-contained in this file
=============================================================================
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TimeframeTrend:
    timeframe: str          # "WEEKLY" | "DAILY" | "4H" | "1H" | "15MIN"
    trend: str              # "BULLISH" | "BEARISH" | "NEUTRAL"
    ema20: float
    close: float
    price_vs_ema20: str     # "ABOVE" | "BELOW"
    supertrend_dir: int     # +1 = bullish, -1 = bearish
    supertrend_val: float
    atr: float


@dataclass
class NearbyLevel:
    price: float
    level_type: str         # "SUPPORT" | "RESISTANCE"
    timeframe: str
    strength: float
    touches: int
    distance_points: float
    distance_atr: float
    tags: str               # e.g. "Fib 0.618 | Round | DAILY"


@dataclass
class PremarketReport:
    index: str
    generated_at: str       # "09:10 IST"
    current_price: float
    daily_atr: float

    # Trend table (up to 5 rows)
    tf_trends: List[TimeframeTrend] = field(default_factory=list)
    bullish_tfs: int = 0
    bearish_tfs: int = 0

    # Levels
    key_levels: List[NearbyLevel] = field(default_factory=list)
    nearest_support: Optional[NearbyLevel] = None
    nearest_resistance: Optional[NearbyLevel] = None

    # CPR for today
    cpr_pivot: float = 0.0
    cpr_top: float = 0.0
    cpr_bottom: float = 0.0
    cpr_r1: float = 0.0
    cpr_s1: float = 0.0
    cpr_r2: float = 0.0
    cpr_s2: float = 0.0

    # Overall bias
    bias: str = "NEUTRAL"       # "BULLISH" | "BEARISH" | "NEUTRAL"
    bias_score: int = 0         # -5 to +5
    bias_reasons: List[str] = field(default_factory=list)

    # Watch zones (human-readable)
    watch_buy_zone: str = ""
    watch_sell_zone: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Main analyser
# ─────────────────────────────────────────────────────────────────────────────

class PremarketAnalyzer:
    """
    Runs full multi-timeframe pre-market analysis for an index.

    Data priority:
      1. Dhan API (dhan_fetcher) — primary, always available for historical data
      2. Yahoo Finance (free_fetcher) — fallback if Dhan call fails
      3. Sample data — last resort (clearly labelled in logs)

    Usage:
        from scalper.data.fetcher import DataFetcher
        from scalper.data.free_fetcher import FreeDataFetcher
        analyzer = PremarketAnalyzer(DataFetcher(), FreeDataFetcher())
        report = analyzer.run("NIFTY")
    """

    def __init__(self, dhan_fetcher=None, free_fetcher=None):
        self._dhan  = dhan_fetcher    # scalper.data.fetcher.DataFetcher
        self._yahoo = free_fetcher    # scalper.data.free_fetcher.FreeDataFetcher

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC
    # ══════════════════════════════════════════════════════════════════════════

    def run(self, index: str) -> Optional[PremarketReport]:
        """Full pre-market pipeline for one index. Returns None if no data."""
        logger.info(f"[{index}] Pre-market analysis starting...")

        tfs = self._fetch_timeframes(index)
        if tfs.get("daily") is None or len(tfs["daily"]) < 22:
            logger.warning(f"[{index}] Insufficient daily data — skipping pre-market")
            return None

        # ── Trend per timeframe ───────────────────────────────────────────
        tf_map = [
            ("weekly",  "WEEKLY", 10, 3.0),
            ("daily",   "DAILY",  10, 3.0),
            ("4h",      "4H",     10, 3.0),
            ("hourly",  "1H",      7, 3.0),
            ("15min",   "15MIN",   7, 3.0),
        ]
        tf_trends = []
        for key, label, st_p, st_m in tf_map:
            df = tfs.get(key)
            if df is not None and len(df) >= 22:
                t = self._analyze_trend(df, label, st_p, st_m)
                if t:
                    tf_trends.append(t)

        # ── Mark all S/R levels ───────────────────────────────────────────
        from scalper.core.index_levels import IndexLevelMarker
        marker = IndexLevelMarker()
        marker.mark_levels(
            daily_df=tfs["daily"],
            weekly_df=tfs.get("weekly"),
            hourly_df=tfs.get("hourly"),
            fifteen_min_df=tfs.get("15min"),
            index=index,
        )

        # Use live NSE spot price — not stale daily close
        # Daily close = yesterday's close; pre-market needs today's actual open/future
        live_price = None
        if self._dhan:
            try:
                live_price = self._dhan.fetch_nse_spot_price(index)
            except Exception:
                pass
        # Fall back to last daily close only if live fetch fails
        current_price = float(live_price) if live_price and live_price > 0 else float(tfs["daily"]["close"].iloc[-1])
        daily_atr = (marker._daily_atr
                     if marker._daily_atr > 0
                     else IndexLevelMarker._compute_atr(tfs["daily"], 14))

        # ── Nearby levels (within 2 ATR) ─────────────────────────────────
        nearby = self._get_nearby_levels(marker, current_price, daily_atr)
        nearest_sup = next((l for l in nearby if l.level_type == "SUPPORT"), None)
        nearest_res = next((l for l in nearby if l.level_type == "RESISTANCE"), None)

        # ── CPR pivots from yesterday's daily candle ──────────────────────
        cpr = self._compute_cpr(tfs["daily"])

        # ── Bias ─────────────────────────────────────────────────────────
        bias, score, reasons = self._compute_bias(tf_trends, nearby, current_price)
        buy_zone, sell_zone = self._describe_zones(nearby, cpr)

        report = PremarketReport(
            index=index,
            generated_at=datetime.now().strftime("%H:%M IST"),
            current_price=round(current_price, 2),
            daily_atr=round(daily_atr, 2),
            tf_trends=tf_trends,
            bullish_tfs=sum(1 for t in tf_trends if t.trend == "BULLISH"),
            bearish_tfs=sum(1 for t in tf_trends if t.trend == "BEARISH"),
            key_levels=nearby,
            nearest_support=nearest_sup,
            nearest_resistance=nearest_res,
            cpr_pivot=round(cpr.get("pivot", 0), 2),
            cpr_top=round(cpr.get("tc", 0), 2),
            cpr_bottom=round(cpr.get("bc", 0), 2),
            cpr_r1=round(cpr.get("r1", 0), 2),
            cpr_s1=round(cpr.get("s1", 0), 2),
            cpr_r2=round(cpr.get("r2", 0), 2),
            cpr_s2=round(cpr.get("s2", 0), 2),
            bias=bias,
            bias_score=score,
            bias_reasons=reasons,
            watch_buy_zone=buy_zone,
            watch_sell_zone=sell_zone,
        )

        logger.info(
            f"[{index}] Pre-market done | Bias={bias}({score:+d}) "
            f"| Bull={report.bullish_tfs}/5 Bear={report.bearish_tfs}/5 "
            f"| Levels within 2ATR={len(nearby)}"
        )
        return report

    # ══════════════════════════════════════════════════════════════════════════
    # DATA FETCHING & RESAMPLING
    # ══════════════════════════════════════════════════════════════════════════

    def _fetch_timeframes(self, index: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch all required timeframes.
        Strategy: try Dhan API first for each TF, fall back to Yahoo Finance.
        """
        result = {}

        # ── DAILY (400 days for level detection) ─────────────────────────
        daily = self._fetch_daily(index)
        if daily is not None and len(daily) > 22:
            result["daily"] = daily
            result["weekly"] = self._resample_weekly(daily)

        # ── HOURLY (60 days for 1H and 4H) ────────────────────────────────
        hourly = self._fetch_hourly(index)
        if hourly is not None and len(hourly) > 22:
            result["hourly"] = hourly
            result["4h"] = self._resample_4h(hourly)

        # ── 15-MIN (15 days) ──────────────────────────────────────────────
        fifteen = self._fetch_fifteen(index)
        if fifteen is not None and len(fifteen) > 22:
            result["15min"] = fifteen

        return result

    def _fetch_daily(self, index: str):
        """Daily candles — Dhan first, Yahoo fallback."""
        if self._dhan:
            try:
                df = self._dhan.fetch_daily_data(index, days_back=400)
                if df is not None and len(df) > 22:
                    logger.info(f"[{index}] Daily data from Dhan: {len(df)} bars")
                    return df
            except Exception as e:
                logger.warning(f"[{index}] Dhan daily fetch failed: {e}")

        if self._yahoo:
            try:
                df = self._yahoo.fetch_daily(index, period="2y")
                if df is not None and len(df) > 22:
                    logger.info(f"[{index}] Daily data from Yahoo: {len(df)} bars")
                    return df
            except Exception as e:
                logger.warning(f"[{index}] Yahoo daily fetch failed: {e}")

        logger.warning(f"[{index}] Daily data unavailable — using sample data")
        from scalper.data.fetcher import DataFetcher
        from scalper.config.settings import INDEX_CONFIGS
        base = {"NIFTY": 23500, "BANKNIFTY": 51000, "FINNIFTY": 23000,
                "MIDCPNIFTY": 12500, "SENSEX": 77000}.get(index, 23500)
        return DataFetcher.generate_sample_daily(index, 400, base)

    def _fetch_hourly(self, index: str):
        """1H candles — Dhan first, Yahoo fallback."""
        if self._dhan:
            try:
                df = self._dhan.fetch_index_data(index, interval="60", days_back=60)
                if df is not None and len(df) > 22:
                    logger.info(f"[{index}] 1H data from Dhan: {len(df)} bars")
                    return df
            except Exception as e:
                logger.warning(f"[{index}] Dhan 1H fetch failed: {e}")

        if self._yahoo:
            try:
                df = self._yahoo.fetch_intraday(index, interval="1h", period="60d")
                if df is not None and len(df) > 22:
                    logger.info(f"[{index}] 1H data from Yahoo: {len(df)} bars")
                    return df
            except Exception as e:
                logger.warning(f"[{index}] Yahoo 1H fetch failed: {e}")

        logger.warning(f"[{index}] 1H data unavailable — using sample data")
        from scalper.data.fetcher import DataFetcher
        base = {"NIFTY": 23500, "BANKNIFTY": 51000, "FINNIFTY": 23000,
                "MIDCPNIFTY": 12500, "SENSEX": 77000}.get(index, 23500)
        return DataFetcher.generate_sample_data(index, days=60, interval_minutes=60, base_price=base)

    def _fetch_fifteen(self, index: str):
        """15-min candles — Dhan first, Yahoo fallback."""
        if self._dhan:
            try:
                df = self._dhan.fetch_index_data(index, interval="15", days_back=15)
                if df is not None and len(df) > 22:
                    logger.info(f"[{index}] 15min data from Dhan: {len(df)} bars")
                    return df
            except Exception as e:
                logger.warning(f"[{index}] Dhan 15min fetch failed: {e}")

        if self._yahoo:
            try:
                df = self._yahoo.fetch_intraday(index, interval="15m", period="30d")
                if df is not None and len(df) > 22:
                    logger.info(f"[{index}] 15min data from Yahoo: {len(df)} bars")
                    return df
            except Exception as e:
                logger.warning(f"[{index}] Yahoo 15min fetch failed: {e}")

        logger.warning(f"[{index}] 15min data unavailable — using sample data")
        from scalper.data.fetcher import DataFetcher
        base = {"NIFTY": 23500, "BANKNIFTY": 51000, "FINNIFTY": 23000,
                "MIDCPNIFTY": 12500, "SENSEX": 77000}.get(index, 23500)
        return DataFetcher.generate_sample_data(index, days=15, interval_minutes=15, base_price=base)

    @staticmethod
    def _resample_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
        """Resample daily OHLCV to weekly (week ending Friday = NSE convention)."""
        df = daily_df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        weekly = df.resample("W-FRI").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        ).dropna(subset=["close"]).reset_index()
        weekly = weekly.rename(columns={"datetime": "datetime"})
        return weekly

    @staticmethod
    def _resample_4h(hourly_df: pd.DataFrame) -> pd.DataFrame:
        """Resample 1H OHLCV to 4H, dropping empty buckets (nights/weekends)."""
        df = hourly_df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        four_h = df.resample("4h").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        ).dropna(subset=["close"]).reset_index()
        four_h = four_h.rename(columns={"datetime": "datetime"})
        return four_h

    # ══════════════════════════════════════════════════════════════════════════
    # TREND ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════

    def _analyze_trend(
        self,
        df: pd.DataFrame,
        tf_label: str,
        st_period: int = 10,
        st_mult: float = 3.0,
    ) -> Optional[TimeframeTrend]:
        try:
            df = df.copy().reset_index(drop=True)
            if len(df) < max(st_period + 2, 22):
                return None

            df = self._compute_supertrend(df, st_period, st_mult)
            ema20 = df["close"].ewm(span=20, adjust=False).mean()

            close = float(df["close"].iloc[-1])
            ema20_val = float(ema20.iloc[-1])
            st_dir = int(df["supertrend_dir"].iloc[-1])
            st_val = float(df["supertrend"].iloc[-1])

            from scalper.core.index_levels import IndexLevelMarker
            atr_val = IndexLevelMarker._compute_atr(df, min(14, len(df) - 1))

            price_vs_ema = "ABOVE" if close > ema20_val else "BELOW"

            if price_vs_ema == "ABOVE" and st_dir == 1:
                trend = "BULLISH"
            elif price_vs_ema == "BELOW" and st_dir == -1:
                trend = "BEARISH"
            else:
                trend = "NEUTRAL"

            return TimeframeTrend(
                timeframe=tf_label,
                trend=trend,
                ema20=round(ema20_val, 2),
                close=round(close, 2),
                price_vs_ema20=price_vs_ema,
                supertrend_dir=st_dir,
                supertrend_val=round(st_val, 2),
                atr=round(atr_val, 2),
            )
        except Exception as e:
            logger.warning(f"Trend analysis failed for {tf_label}: {e}")
            return None

    @staticmethod
    def _compute_supertrend(
        df: pd.DataFrame,
        period: int = 10,
        multiplier: float = 3.0,
    ) -> pd.DataFrame:
        df = df.copy()
        high = df["high"].values.astype(float)
        low  = df["low"].values.astype(float)
        close = df["close"].values.astype(float)
        n = len(df)

        # Wilder ATR
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i - 1]),
                        abs(low[i]  - close[i - 1]))

        atr = np.zeros(n)
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        hl2 = (high + low) / 2.0
        upper_basic = hl2 + multiplier * atr
        lower_basic = hl2 - multiplier * atr

        upper_band = upper_basic.copy()
        lower_band = lower_basic.copy()
        supertrend = np.zeros(n)
        direction  = np.ones(n, dtype=int)   # +1 bullish start

        supertrend[0] = upper_band[0]
        direction[0]  = -1

        for i in range(1, n):
            # Adjust bands: don't widen once set
            upper_band[i] = (upper_basic[i]
                             if upper_basic[i] < upper_band[i - 1] or close[i - 1] > upper_band[i - 1]
                             else upper_band[i - 1])
            lower_band[i] = (lower_basic[i]
                             if lower_basic[i] > lower_band[i - 1] or close[i - 1] < lower_band[i - 1]
                             else lower_band[i - 1])

            prev_st = supertrend[i - 1]

            if prev_st == upper_band[i - 1]:          # was in downtrend
                if close[i] > upper_band[i]:
                    direction[i] = 1                   # flip to uptrend
                    supertrend[i] = lower_band[i]
                else:
                    direction[i] = -1
                    supertrend[i] = upper_band[i]
            else:                                      # was in uptrend
                if close[i] < lower_band[i]:
                    direction[i] = -1                  # flip to downtrend
                    supertrend[i] = upper_band[i]
                else:
                    direction[i] = 1
                    supertrend[i] = lower_band[i]

        df["supertrend"]     = supertrend
        df["supertrend_dir"] = direction
        return df

    # ══════════════════════════════════════════════════════════════════════════
    # LEVELS
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _get_nearby_levels(marker, current_price: float, daily_atr: float) -> List[NearbyLevel]:
        """Return all IndexLevels within 2 ATR, enriched and sorted by distance.
        dist_pts: current_price - level.price
          > 0 means level is BELOW current price → SUPPORT
          < 0 means level is ABOVE current price → RESISTANCE
        """
        nearby = []
        for lv in marker.levels:
            dist_pts = current_price - lv.price
            dist_atr = abs(dist_pts) / max(daily_atr, 1)

            if dist_atr > 2.0:
                continue

            # Sanity-check: level_type should match its position relative to price
            # (round numbers added as synthetic levels may have wrong type if price moved)
            expected_type = "SUPPORT" if dist_pts >= 0 else "RESISTANCE"
            actual_type   = lv.level_type.value
            if expected_type != actual_type and lv.touches == 0:
                # Synthetic round-number level with wrong type — skip
                continue

            # Build tags string
            tag_parts = [lv.timeframe.value]
            if lv.is_round_number:
                tag_parts.append("Round")
            if lv.fib_confluence:
                tag_parts.append("Fib")
            if lv.pivot_confluence:
                tag_parts.append("CPR/Pivot")
            if lv.touches >= 4:
                tag_parts.append(f"{lv.touches}T")     # touch count

            nearby.append(NearbyLevel(
                price=lv.price,
                level_type=lv.level_type.value,
                timeframe=lv.timeframe.value,
                strength=lv.strength,
                touches=lv.touches,
                distance_points=round(abs(dist_pts), 1),
                distance_atr=round(dist_atr, 2),
                tags=" | ".join(tag_parts),
            ))

        # Sort: supports nearest first (descending price), resistances nearest first
        supports = sorted(
            [l for l in nearby if l.level_type == "SUPPORT"],
            key=lambda x: x.distance_atr
        )
        resistances = sorted(
            [l for l in nearby if l.level_type == "RESISTANCE"],
            key=lambda x: x.distance_atr
        )

        # Interleave: nearest first regardless of type
        result = sorted(supports + resistances, key=lambda x: x.distance_atr)
        return result

    @staticmethod
    def _compute_cpr(daily_df: pd.DataFrame) -> Dict[str, float]:
        """
        CPR (Central Pivot Range) from the last COMPLETED daily candle.
        We always use iloc[-1] because Dhan daily API returns only completed
        candles — today's bar is NOT included until market closes.
        If data is fetched pre-market, iloc[-1] = yesterday's completed candle.
        """
        if len(daily_df) < 2:
            return {}
        prev = daily_df.iloc[-1]   # last completed daily bar = yesterday
        h, l, c = float(prev["high"]), float(prev["low"]), float(prev["close"])
        pivot = (h + l + c) / 3
        bc    = (h + l) / 2
        tc    = (pivot - bc) + pivot
        r1    = 2 * pivot - l
        s1    = 2 * pivot - h
        r2    = pivot + (h - l)
        s2    = pivot - (h - l)
        return dict(pivot=pivot, bc=bc, tc=tc, r1=r1, s1=s1, r2=r2, s2=s2)

    # ══════════════════════════════════════════════════════════════════════════
    # BIAS COMPUTATION
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _compute_bias(
        tf_trends: List[TimeframeTrend],
        nearby: List[NearbyLevel],
        current_price: float,
    ) -> Tuple[str, int, List[str]]:
        """
        Weighted bias score:
          Weekly  +2 / -2
          Daily   +2 / -2
          4H      +1 / -1
          1H      +1 / -1
          15MIN   +1 / -1
        Max possible = 7 → normalised to -5..+5
        """
        weights = {"WEEKLY": 2, "DAILY": 2, "4H": 1, "1H": 1, "15MIN": 1}
        raw = 0
        max_w = sum(weights.values())
        reasons = []

        for t in tf_trends:
            w = weights.get(t.timeframe, 1)
            st_label = "ST+" if t.supertrend_dir == 1 else "ST-"
            ema_label = f"above {t.ema20:.0f}" if t.price_vs_ema20 == "ABOVE" else f"below {t.ema20:.0f}"
            if t.trend == "BULLISH":
                raw += w
                reasons.append(f"{t.timeframe}: BULLISH — price {ema_label} EMA20, {st_label}")
            elif t.trend == "BEARISH":
                raw -= w
                reasons.append(f"{t.timeframe}: BEARISH — price {ema_label} EMA20, {st_label}")
            else:
                reasons.append(f"{t.timeframe}: NEUTRAL — price {ema_label} EMA20, {st_label}")

        # Normalise to -5..+5
        score = round(raw / max_w * 5) if max_w else 0

        # Level context note
        for lv in nearby[:3]:
            dist_label = f"{lv.distance_points:.0f}pts ({lv.distance_atr:.1f} ATR)"
            if lv.level_type == "SUPPORT":
                reasons.append(f"Support {lv.price:.0f} is {dist_label} below")
            else:
                reasons.append(f"Resistance {lv.price:.0f} is {dist_label} above")

        if score >= 2:
            bias = "BULLISH"
        elif score <= -2:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

        return bias, score, reasons

    @staticmethod
    def _describe_zones(nearby: List[NearbyLevel], cpr: Dict) -> Tuple[str, str]:
        """Human-readable watch zones for the day.
        Buy (CE) zone = nearest support below current price.
        Sell (PE) zone = nearest resistance above current price.
        """
        # Supports are below current price (positive dist_pts = price > level)
        supports    = sorted(
            [l for l in nearby if l.level_type == "SUPPORT"],
            key=lambda x: x.distance_atr
        )
        resistances = sorted(
            [l for l in nearby if l.level_type == "RESISTANCE"],
            key=lambda x: x.distance_atr
        )

        if supports:
            sup = supports[0]
            buy_zone = (
                f"{sup.price:.0f}  |  {sup.distance_points:.0f} pts below  "
                f"|  {sup.touches} touches  |  {sup.tags}"
            )
        elif cpr.get("s1"):
            buy_zone = f"{cpr['s1']:.0f} (S1 CPR Pivot)"
        else:
            buy_zone = "No key support within 2 ATR — wait for fresh level"

        if resistances:
            res = resistances[0]
            sell_zone = (
                f"{res.price:.0f}  |  {res.distance_points:.0f} pts above  "
                f"|  {res.touches} touches  |  {res.tags}"
            )
        elif cpr.get("r1"):
            sell_zone = f"{cpr['r1']:.0f} (R1 CPR Pivot)"
        else:
            sell_zone = "No key resistance within 2 ATR — wait for fresh level"

        return buy_zone, sell_zone
