"""
=============================================================================
SCALPER PRO v2 — Technical Indicators Library
=============================================================================
All indicator computations used for entry signal scoring.

Indicators computed on OHLCV DataFrames:
  - EMA (any period) on close, high, low
  - SuperTrend (ATR-based dynamic S/R)
  - RSI (Relative Strength Index)
  - Volume Ratio (current vs rolling average)
  - India VIX (fetched live from NSE)
  - OI Change interpretation
  - Multi-timeframe alignment score

Output: IntraSignalScore dataclass — used by backtest + live orchestrator.
=============================================================================
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

# SuperTrend
ST_PERIOD      = 10
ST_MULTIPLIER  = 3.0

# RSI
RSI_PERIOD     = 14
RSI_CE_MIN     = 30    # CE: don't buy already oversold bounce too early
RSI_CE_MAX     = 62    # CE: don't buy overbought
RSI_PE_MIN     = 38    # PE: don't short deeply oversold
RSI_PE_MAX     = 70    # PE: don't short already extended fall

# Volume
VOL_AVG_PERIOD = 20
VOL_MIN_RATIO  = 0.80  # Signal bar must have >= 80% of avg volume

# India VIX thresholds
VIX_MAX_FOR_APPROACH  = 20.0   # Above this: AT_LEVEL only (no APPROACHING)
VIX_HALT_TRADING      = 28.0   # Above this: skip all trades (too much decay)

# EMA periods
EMA_FAST       = 5
EMA_MID        = 13
EMA_SLOW       = 20

# Minimum confirmation score to take a trade (out of MAX_CONF_SCORE)
# Indicators scored: 5EMA, 13EMA, 13EMA_HL, SuperTrend, RSI, Volume, BB = 7 points
MIN_CONF_SCORE = 3   # need at least 3 out of 7 to take a trade
MAX_CONF_SCORE = 7


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IntraSignalScore:
    """
    Multi-indicator confirmation score for a single bar/candle.
    Each indicator contributes +1 if it aligns with the proposed direction.
    A trade is taken only when total_score >= MIN_CONF_SCORE.
    """
    # Raw indicator values
    ema5:            float = 0.0
    ema13:           float = 0.0
    ema13_high:      float = 0.0   # EMA of highs (dynamic resistance)
    ema13_low:       float = 0.0   # EMA of lows  (dynamic support)
    ema20:           float = 0.0
    supertrend:      float = 0.0
    supertrend_bull: bool  = False  # True = bullish (price above ST line)
    rsi:             float = 50.0
    volume_ratio:    float = 1.0   # current / 20-bar avg
    india_vix:       float = 0.0   # 0 = unknown/not fetched
    oi_change_bull:  Optional[bool] = None  # None = not available

    # Bollinger Band values
    bb_upper:        float = 0.0
    bb_middle:       float = 0.0
    bb_lower:        float = 0.0
    bb_pct_b:        float = 0.5   # 0=at lower, 1=at upper, 0.5=at middle
    bb_width:        float = 0.0   # squeeze measure

    # Per-indicator confirmation flags (True = aligned with direction)
    conf_ema5:       bool = False
    conf_ema13:      bool = False
    conf_ema13_hl:   bool = False   # dynamic level confirms direction
    conf_supertrend: bool = False
    conf_rsi:        bool = False
    conf_volume:     bool = False
    conf_bb:         bool = False   # Bollinger Band position confirms direction
    conf_vix:        bool = True    # True = VIX not blocking
    conf_oi:         bool = True    # True = OI not blocking (default allow)

    # Score summary
    total_score:     int   = 0
    max_score:       int   = MAX_CONF_SCORE
    trade_allowed:   bool  = False
    block_reason:    str   = ""
    score_breakdown: str   = ""


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR COMPUTATIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Standard EMA using pandas ewm."""
    return series.ewm(span=period, adjust=False).mean()


def compute_ema_of_highs_lows(df: pd.DataFrame, period: int = 13):
    """
    13 EMA of Highs and 13 EMA of Lows — dynamic S/R bands.
    Widely used by intraday traders as dynamic support/resistance.
    Returns (ema_high_series, ema_low_series)
    """
    ema_high = compute_ema(df["high"], period)
    ema_low  = compute_ema(df["low"],  period)
    return ema_high, ema_low


def compute_supertrend(df: pd.DataFrame, period: int = ST_PERIOD,
                        multiplier: float = ST_MULTIPLIER) -> pd.DataFrame:
    """
    SuperTrend indicator.
    Returns DataFrame with columns: ['supertrend', 'trend'] where
      trend = True (bullish) when price > supertrend line
      trend = False (bearish) when price < supertrend line
    """
    n = len(df)
    if n < period + 2:
        return pd.DataFrame({
            "supertrend": [df["close"].iloc[-1]] * n,
            "trend": [True] * n,
        }, index=df.index)

    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values

    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i]  - close[i-1]))

    # ATR using Wilder smoothing
    atr = np.zeros(n)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

    # Basic Upper / Lower bands
    hl_avg = (high + low) / 2.0
    basic_upper = hl_avg + multiplier * atr
    basic_lower = hl_avg - multiplier * atr

    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend  = np.zeros(n)
    trend       = np.ones(n, dtype=bool)   # True = bullish

    for i in range(n):
        if i == 0:
            final_upper[i] = basic_upper[i]
            final_lower[i] = basic_lower[i]
            supertrend[i]  = final_upper[i]
            trend[i]       = close[i] <= final_upper[i]
            continue

        final_upper[i] = (basic_upper[i]
                          if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]
                          else final_upper[i-1])
        final_lower[i] = (basic_lower[i]
                          if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]
                          else final_lower[i-1])

        if supertrend[i-1] == final_upper[i-1]:
            trend[i]      = close[i] > final_upper[i]
            supertrend[i] = final_lower[i] if trend[i] else final_upper[i]
        else:
            trend[i]      = close[i] >= final_lower[i]
            supertrend[i] = final_lower[i] if trend[i] else final_upper[i]

    return pd.DataFrame({
        "supertrend": supertrend,
        "trend":      trend,
    }, index=df.index)


def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder RSI."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_bollinger_bands(df: pd.DataFrame, period: int = 20,
                             std_dev: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands: Middle = SMA(close, period), Upper/Lower = middle +/- std_dev * stddev.

    Returns DataFrame with columns:
      bb_upper, bb_middle, bb_lower, bb_width, bb_pct_b

    bb_pct_b   : 0 = at lower band, 1 = at upper band, 0.5 = at middle
    bb_width   : (upper - lower) / middle  — measures volatility squeeze
    """
    if len(df) < period:
        mid = df["close"].mean()
        return pd.DataFrame({
            "bb_upper":  [mid] * len(df),
            "bb_middle": [mid] * len(df),
            "bb_lower":  [mid] * len(df),
            "bb_width":  [0.0] * len(df),
            "bb_pct_b":  [0.5] * len(df),
        }, index=df.index)

    close  = df["close"]
    middle = close.rolling(period, min_periods=period).mean()
    stddev = close.rolling(period, min_periods=period).std(ddof=0)
    upper  = middle + std_dev * stddev
    lower  = middle - std_dev * stddev
    width  = (upper - lower) / middle.replace(0, np.nan)
    pct_b  = (close - lower) / (upper - lower).replace(0, np.nan)

    return pd.DataFrame({
        "bb_upper":  upper.bfill().ffill(),
        "bb_middle": middle.bfill().ffill(),
        "bb_lower":  lower.bfill().ffill(),
        "bb_width":  width.fillna(0),
        "bb_pct_b":  pct_b.fillna(0.5),
    }, index=df.index)


def compute_volume_ratio(df: pd.DataFrame, period: int = VOL_AVG_PERIOD) -> pd.Series:
    """Volume of each bar relative to its rolling average."""
    if "volume" not in df.columns:
        return pd.Series(1.0, index=df.index)
    avg_vol = df["volume"].rolling(period, min_periods=3).mean()
    return (df["volume"] / avg_vol.replace(0, np.nan)).fillna(1.0)


def fetch_india_vix() -> float:
    """
    Fetch India VIX from NSE public API.
    Returns 0.0 on failure (treated as 'unknown', does not block trades).
    """
    try:
        import requests
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com",
        }
        session = requests.Session()
        # Warm up cookies
        session.get("https://www.nseindia.com", headers=headers, timeout=5)
        resp = session.get(
            "https://www.nseindia.com/api/allIndices",
            headers=headers,
            timeout=8,
        )
        if resp.status_code == 200:
            data = resp.json()
            for item in data.get("data", []):
                if item.get("indexSymbol") == "INDIA VIX":
                    return float(item.get("last", 0))
    except Exception as e:
        logger.debug(f"VIX fetch failed: {e}")
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL SCORER
# ─────────────────────────────────────────────────────────────────────────────

def score_signal(
    df: pd.DataFrame,
    bar_idx: int,
    direction: str,          # "CE" or "PE"
    proximity_zone: str,     # "AT_LEVEL" or "APPROACHING"
    india_vix: float = 0.0,
    oi_bull: Optional[bool] = None,   # True=long buildup, False=short buildup, None=unknown
) -> IntraSignalScore:
    """
    Score the multi-indicator confirmation for a signal bar.

    bar_idx: index into df of the SIGNAL candle (entry is next candle open).

    Scoring (each worth 1 point, max 7):
      1. 5 EMA  — price vs 5 EMA (momentum direction)
      2. 13 EMA — price vs 13 EMA (trend confirmation)
      3. 13 EMA High/Low — dynamic level confirms direction
      4. SuperTrend — bullish/bearish alignment
      5. RSI — in valid range for direction
      6. Volume — current bar volume >= 80% of avg
      7. India VIX — not blocking (VIX < threshold)
      (OI change is a bonus flag, does not reduce score if unavailable)

    Returns IntraSignalScore. trade_allowed = (total_score >= MIN_CONF_SCORE).
    """
    score = IntraSignalScore()
    score.india_vix = india_vix
    score.oi_change_bull = oi_bull

    # Need at least enough bars for indicators
    history = df.iloc[: bar_idx + 1]
    if len(history) < max(RSI_PERIOD, ST_PERIOD, VOL_AVG_PERIOD) + 2:
        score.block_reason = "insufficient_history"
        score.trade_allowed = False
        return score

    close = history["close"]
    bar   = history.iloc[-1]

    # ── Compute indicators ────────────────────────────────────────────────────
    ema5_s  = compute_ema(close, EMA_FAST)
    ema13_s = compute_ema(close, EMA_MID)
    ema20_s = compute_ema(close, EMA_SLOW)
    e13h, e13l = compute_ema_of_highs_lows(history, EMA_MID)
    st_df   = compute_supertrend(history, ST_PERIOD, ST_MULTIPLIER)
    rsi_s   = compute_rsi(close, RSI_PERIOD)
    vol_r   = compute_volume_ratio(history, VOL_AVG_PERIOD)
    bb_df   = compute_bollinger_bands(history, period=20, std_dev=2.0)

    score.ema5          = float(ema5_s.iloc[-1])
    score.ema13         = float(ema13_s.iloc[-1])
    score.ema13_high    = float(e13h.iloc[-1])
    score.ema13_low     = float(e13l.iloc[-1])
    score.ema20         = float(ema20_s.iloc[-1])
    score.supertrend    = float(st_df["supertrend"].iloc[-1])
    score.supertrend_bull = bool(st_df["trend"].iloc[-1])
    score.rsi           = float(rsi_s.iloc[-1])
    score.volume_ratio  = float(vol_r.iloc[-1])
    score.bb_upper      = float(bb_df["bb_upper"].iloc[-1])
    score.bb_middle     = float(bb_df["bb_middle"].iloc[-1])
    score.bb_lower      = float(bb_df["bb_lower"].iloc[-1])
    score.bb_pct_b      = float(bb_df["bb_pct_b"].iloc[-1])
    score.bb_width      = float(bb_df["bb_width"].iloc[-1])

    spot = float(bar["close"])
    pts  = []

    # ── 1. 5 EMA: price must be on correct side + EMA sloping right direction ─
    ema5_prev = float(ema5_s.iloc[-2]) if len(ema5_s) >= 2 else score.ema5
    ema5_slope_up = score.ema5 > ema5_prev
    if direction == "CE":
        score.conf_ema5 = spot > score.ema5 and ema5_slope_up
    else:
        score.conf_ema5 = spot < score.ema5 and not ema5_slope_up
    if score.conf_ema5:
        pts.append("5EMA")

    # ── 2. 13 EMA: price on correct side ─────────────────────────────────────
    if direction == "CE":
        score.conf_ema13 = spot > score.ema13
    else:
        score.conf_ema13 = spot < score.ema13
    if score.conf_ema13:
        pts.append("13EMA")

    # ── 3. 13 EMA High/Low: dynamic S/R confirmation ──────────────────────────
    # CE near support: bar low should be near or touched 13 EMA Low (dynamic support)
    # PE near resistance: bar high should be near or touched 13 EMA High (dynamic resistance)
    if direction == "CE":
        # Good bounce: low touched/approached 13 EMA Low and closed above it
        near_ema_low = abs(float(bar["low"]) - score.ema13_low) < (spot * 0.003)  # within 0.3%
        score.conf_ema13_hl = near_ema_low or float(bar["low"]) <= score.ema13_low < spot
    else:
        # Good rejection: high touched/approached 13 EMA High and closed below it
        near_ema_high = abs(float(bar["high"]) - score.ema13_high) < (spot * 0.003)
        score.conf_ema13_hl = near_ema_high or float(bar["high"]) >= score.ema13_high > spot
    if score.conf_ema13_hl:
        pts.append("13EMA_HL")

    # ── 4. SuperTrend alignment ───────────────────────────────────────────────
    if direction == "CE":
        score.conf_supertrend = score.supertrend_bull
    else:
        score.conf_supertrend = not score.supertrend_bull
    if score.conf_supertrend:
        pts.append("ST")

    # ── 5. RSI in valid range ─────────────────────────────────────────────────
    if direction == "CE":
        # Buy CE when RSI is recovering from oversold (30-62)
        score.conf_rsi = RSI_CE_MIN <= score.rsi <= RSI_CE_MAX
    else:
        # Buy PE when RSI is falling from overbought (38-70)
        score.conf_rsi = RSI_PE_MIN <= score.rsi <= RSI_PE_MAX
    if score.conf_rsi:
        pts.append(f"RSI({score.rsi:.0f})")

    # ── 6. Volume confirmation ────────────────────────────────────────────────
    score.conf_volume = score.volume_ratio >= VOL_MIN_RATIO
    if score.conf_volume:
        pts.append(f"VOL({score.volume_ratio:.1f}x)")

    # ── 7. Bollinger Band position ────────────────────────────────────────────
    # CE: price near or below lower band (oversold zone, support bounce setup)
    #     pct_b < 0.25 = price in lower 25% of BB = valid CE bounce zone
    # PE: price near or above upper band (overbought zone, resistance rejection setup)
    #     pct_b > 0.75 = price in upper 25% of BB = valid PE rejection zone
    # Also flag BB squeeze (width < 2%) as potential breakout — extra confirmation
    bb_squeeze = score.bb_width < 0.02
    if direction == "CE":
        score.conf_bb = score.bb_pct_b <= 0.35 or bb_squeeze
    else:
        score.conf_bb = score.bb_pct_b >= 0.65 or bb_squeeze
    if score.conf_bb:
        pts.append(f"BB({score.bb_pct_b:.2f}{'_SQZ' if bb_squeeze else ''})")

    # ── 8. India VIX check ────────────────────────────────────────────────────
    if india_vix > 0:
        if india_vix >= VIX_HALT_TRADING:
            score.conf_vix = False
            score.block_reason = f"VIX={india_vix:.1f} >= halt threshold {VIX_HALT_TRADING}"
        elif india_vix >= VIX_MAX_FOR_APPROACH and proximity_zone == "APPROACHING":
            score.conf_vix = False
            score.block_reason = f"VIX={india_vix:.1f} >= {VIX_MAX_FOR_APPROACH}: AT_LEVEL only"
        else:
            score.conf_vix = True
    # VIX=0 (unknown): don't block
    if not score.conf_vix:
        score.trade_allowed = False
        score.total_score   = 0
        score.score_breakdown = f"BLOCKED_VIX: {score.block_reason}"
        return score

    # ── OI bonus: not part of mandatory score but surfaced in log ─────────────
    if oi_bull is not None:
        score.conf_oi = (oi_bull and direction == "CE") or (not oi_bull and direction == "PE")

    # ── Total score and decision (7 indicators) ──────────────────────────────
    total = sum([
        score.conf_ema5,
        score.conf_ema13,
        score.conf_ema13_hl,
        score.conf_supertrend,
        score.conf_rsi,
        score.conf_volume,
        score.conf_bb,
    ])
    score.total_score    = total
    score.max_score      = MAX_CONF_SCORE
    score.trade_allowed  = total >= MIN_CONF_SCORE
    score.score_breakdown = f"{total}/{MAX_CONF_SCORE}: [{', '.join(pts)}]"
    if not score.trade_allowed:
        failed = []
        if not score.conf_ema5:       failed.append("5EMA")
        if not score.conf_ema13:      failed.append("13EMA")
        if not score.conf_ema13_hl:   failed.append("13EMA_HL")
        if not score.conf_supertrend: failed.append("SuperTrend")
        if not score.conf_rsi:        failed.append(f"RSI({score.rsi:.0f})")
        if not score.conf_volume:     failed.append(f"VOL({score.volume_ratio:.1f}x)")
        if not score.conf_bb:         failed.append(f"BB({score.bb_pct_b:.2f})")
        score.block_reason = f"score={total}/{MIN_CONF_SCORE} required | failed: {', '.join(failed)}"

    return score


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-TIMEFRAME ALIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def mtf_trend_alignment(
    df_1min:  Optional[pd.DataFrame],
    df_3min:  Optional[pd.DataFrame],
    df_15min: Optional[pd.DataFrame],
    direction: str,
) -> Dict[str, bool]:
    """
    Check if 1-min, 3-min, and 15-min SuperTrend all align with direction.
    Returns dict of {timeframe: aligned_bool}.
    Only SuperTrend is used here (fast, reliable for MTF confirmation).
    """
    result = {}
    for label, df in [("1min", df_1min), ("3min", df_3min), ("15min", df_15min)]:
        if df is None or len(df) < ST_PERIOD + 5:
            result[label] = None   # unknown
            continue
        st = compute_supertrend(df, ST_PERIOD, ST_MULTIPLIER)
        bull = bool(st["trend"].iloc[-1])
        result[label] = (bull and direction == "CE") or (not bull and direction == "PE")
    return result


def macro_filter(india_vix: float, event_flag: bool = False) -> tuple:
    """
    High-level macro filter.

    india_vix   : current VIX value (0 = unknown)
    event_flag  : True if a major event is known today (RBI policy, budget,
                  global crisis). Set manually or via calendar check.

    Returns (allowed: bool, reason: str)
    """
    if event_flag:
        return False, "MACRO_EVENT: Major scheduled event today — sit out (RBI/Budget/Global)"
    if india_vix >= VIX_HALT_TRADING:
        return False, f"VIX={india_vix:.1f} >= {VIX_HALT_TRADING}: extreme volatility, skip all trades"
    if india_vix >= 22:
        return True, f"VIX={india_vix:.1f}: elevated — reduce size, AT_LEVEL entries only"
    if india_vix >= 18:
        return True, f"VIX={india_vix:.1f}: moderate — normal trading"
    return True, f"VIX={india_vix:.1f}: low — normal trading"
