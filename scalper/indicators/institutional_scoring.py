"""
=============================================================================
INSTITUTIONAL MODULE — Combined Indicator Scoring Engine
=============================================================================
Extends the existing 7-point score_bar() system with institutional signals.

EXISTING (7 points):
  1. EMA5 vs EMA13       +1
  2. EMA13 vs EMA20      +1
  3. SuperTrend           +1
  4. RSI range            +1
  5. Bollinger position   +1
  6. Volume ratio         +1
  7. VWAP position        +1

INSTITUTIONAL ADD-ONS (5 points):
  8. Candle Pattern       +0/+1/+2  (Pin Bar/Engulfing at level = +2)
  9. Fibonacci OTE        +0/+1/+2  (price in OTE zone aligned = +2)
  10. Volume Profile      +0/+1     (above VAH / below VAL = confirms)
  11. Liquidity Sweep     +0/+2     (sweep just happened = very high prob)
  12. Market Structure    +0/+1     (BOS/CHoCH alignment)

NEW TOTAL: 12 points max
NEW MIN_SCORE: 5 (was 3 out of 7)

This module provides:
  - institutional_score() — compute the extra 5 points for any bar
  - enhanced_score_bar() — full 12-point scoring (replaces score_bar)
  - setup_institutional_context() — pre-compute heavy objects once per session
=============================================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# Import institutional modules
from scalper.core.candle_patterns import CandlePatternDetector
from scalper.core.fibonacci_engine import FibonacciEngine
from scalper.core.volume_profile import VolumeProfileEngine
from scalper.core.liquidity_engine import LiquidityEngine
from scalper.core.market_structure import analyze_market_structure, is_direction_aligned_with_structure

# New scoring constants
INST_MAX_SCORE = 12    # Total max (7 base + 5 institutional)
INST_MIN_SCORE = 5     # Minimum to trade (raised from 3)


@dataclass
class InstitutionalContext:
    """
    Pre-computed institutional analysis for a session.
    Computed ONCE at session start, then reused for every bar.
    This avoids re-computing heavy things (volume profile, Fib, structure)
    on every bar.
    """
    # Volume Profile
    vp_profiles: list = field(default_factory=list)
    vp_signal: object = None  # ValueAreaSignal

    # Fibonacci
    fib_setups: list = field(default_factory=list)

    # Market Structure
    structure: object = None  # MarketStructure

    # Liquidity
    stop_clusters: list = field(default_factory=list)
    recent_sweeps: list = field(default_factory=list)

    # Candle detector (stateless, just needs instance)
    candle_detector: object = None

    # Precomputed ATR
    daily_atr: float = 0.0

    # Ready flag
    initialized: bool = False


def setup_institutional_context(
    df_15min: pd.DataFrame,
    df_daily: pd.DataFrame = None,
    current_bar: int = None,
) -> InstitutionalContext:
    """
    Pre-compute all institutional analysis ONCE per session.
    Call this before the bar-by-bar loop starts.

    Args:
        df_15min: Full 15-min OHLCV history (with indicators already computed)
        df_daily: Daily OHLCV for Fib and structure analysis
        current_bar: If set, only use data up to this bar index
    """
    ctx = InstitutionalContext()

    analysis_df = df_15min.iloc[:current_bar+1] if current_bar else df_15min

    # ── Candle detector (stateless) ───────────────────────────────
    ctx.candle_detector = CandlePatternDetector()

    # ── Volume Profile ────────────────────────────────────────────
    try:
        vp_engine = VolumeProfileEngine(num_bins=40)
        ctx.vp_profiles = vp_engine.compute_daily_profiles(analysis_df)
        if ctx.vp_profiles:
            current_price = float(analysis_df["close"].iloc[-1])
            ctx.vp_signal = vp_engine.get_signal(current_price, ctx.vp_profiles)
    except Exception as e:
        logger.debug(f"VP error: {e}")

    # ── Fibonacci ─────────────────────────────────────────────────
    try:
        fib_engine = FibonacciEngine(min_swing_atr=1.0, max_setups=3)
        source_df = df_daily if df_daily is not None and len(df_daily) > 50 else analysis_df
        if len(source_df) > 30:
            ctx.daily_atr = _quick_atr(source_df)
            ctx.fib_setups = fib_engine.analyze(source_df, atr=ctx.daily_atr)
    except Exception as e:
        logger.debug(f"Fib error: {e}")

    # ── Market Structure ──────────────────────────────────────────
    try:
        source_df = df_daily if df_daily is not None and len(df_daily) > 30 else analysis_df
        ctx.structure = analyze_market_structure(source_df, lookback=5)
    except Exception as e:
        logger.debug(f"Structure error: {e}")

    # ── Liquidity Engine ──────────────────────────────────────────
    try:
        liq_engine = LiquidityEngine()
        sweeps, _, clusters = liq_engine.analyze(analysis_df, lookback=3)
        ctx.stop_clusters = clusters
        # Only keep recent sweeps (within last 20 bars)
        last_bar = len(analysis_df) - 1
        ctx.recent_sweeps = [s for s in sweeps if s.bar_index > last_bar - 20]
    except Exception as e:
        logger.debug(f"Liquidity error: {e}")

    ctx.initialized = True
    return ctx


def institutional_score(
    df: pd.DataFrame,
    bar_index: int,
    direction: str,
    ctx: InstitutionalContext,
) -> Tuple[int, List[str]]:
    """
    Compute the institutional add-on score (0-5 points).
    This is ADDED to the existing score_bar() result.

    Args:
        df: Full DataFrame with indicators pre-computed
        bar_index: Current bar index
        direction: "CE" or "PE"
        ctx: Pre-computed InstitutionalContext

    Returns:
        (score, reasons) — score is 0-5, reasons are human-readable
    """
    score = 0
    reasons = []

    if not ctx.initialized:
        return 0, ["Institutional context not initialized"]

    # ── 8. Candle Pattern (+0/+1/+2) ─────────────────────────────
    if ctx.candle_detector:
        try:
            cp_score = ctx.candle_detector.get_confirmation_score(df, bar_index, direction)
            if cp_score >= 2:
                score += 2
                pats = ctx.candle_detector.scan(df, bar_index)
                aligned = [p for p in pats if p.direction == direction.replace("CE","BULLISH").replace("PE","BEARISH")]
                pat_names = [p.pattern for p in aligned[:2]]
                reasons.append(f"Candle: {', '.join(pat_names)} (+2)")
            elif cp_score == 1:
                score += 1
                reasons.append("Candle: moderate pattern (+1)")
            elif cp_score == -1:
                reasons.append("Candle: contrary pattern (0)")
        except Exception:
            pass

    # ── 9. Fibonacci OTE (+0/+1/+2) ──────────────────────────────
    if ctx.fib_setups:
        try:
            # Check if current price is in OTE zone
            current_price = float(df["close"].iloc[bar_index])
            best_fib_score = 0
            best_fib_reason = ""

            for setup in ctx.fib_setups:
                ote_lo = min(setup.ote_zone_high, setup.ote_zone_low)
                ote_hi = max(setup.ote_zone_high, setup.ote_zone_low)

                if ote_lo <= current_price <= ote_hi:
                    if (direction == "CE" and setup.swing_type == "UPSWING") or \
                       (direction == "PE" and setup.swing_type == "DOWNSWING"):
                        best_fib_score = 2
                        best_fib_reason = f"Fib OTE {ote_lo:.0f}-{ote_hi:.0f} ({setup.swing_type}) (+2)"
                    else:
                        best_fib_score = max(best_fib_score, 1)
                        best_fib_reason = f"Fib OTE zone (direction mismatch) (+1)"
                elif setup.current_position == "DISCOUNT" and direction == "CE":
                    best_fib_score = max(best_fib_score, 1)
                    best_fib_reason = "Fib: Discount zone — CE favorable (+1)"
                elif setup.current_position == "PREMIUM" and direction == "PE":
                    best_fib_score = max(best_fib_score, 1)
                    best_fib_reason = "Fib: Premium zone — PE favorable (+1)"

            score += best_fib_score
            if best_fib_reason:
                reasons.append(best_fib_reason)
        except Exception:
            pass

    # ── 10. Volume Profile (+0/+1) ────────────────────────────────
    if ctx.vp_signal:
        try:
            vps = ctx.vp_signal
            aligned = (
                (vps.bias == "BULLISH" and direction == "CE") or
                (vps.bias == "BEARISH" and direction == "PE")
            )
            if aligned and vps.score >= 2:
                score += 1
                reasons.append(f"VP: {vps.position} ({vps.bias}) (+1)")
            elif vps.nearest_virgin_poc:
                # Virgin POC as target confirmation
                if (direction == "CE" and vps.virgin_poc_direction == "ABOVE") or \
                   (direction == "PE" and vps.virgin_poc_direction == "BELOW"):
                    score += 1
                    reasons.append(f"VP: Virgin POC {vps.nearest_virgin_poc:.0f} ({vps.virgin_poc_direction}) as target (+1)")
        except Exception:
            pass

    # ── 11. Liquidity Sweep (+0/+2) ───────────────────────────────
    if ctx.recent_sweeps:
        try:
            for sweep in ctx.recent_sweeps:
                # Very recent sweep (within 5 bars) aligned with direction
                if abs(sweep.bar_index - bar_index) <= 5:
                    if sweep.trade_direction == direction and sweep.score >= 6:
                        score += 2
                        reasons.append(f"Sweep: {sweep.sweep_type} at {sweep.cluster_price:.0f} (score={sweep.score:.0f}) (+2)")
                        break
                    elif sweep.trade_direction == direction:
                        score += 1
                        reasons.append(f"Sweep: {sweep.sweep_type} nearby (+1)")
                        break
        except Exception:
            pass

    # ── 12. Market Structure Alignment (+0/+1) ────────────────────
    if ctx.structure:
        try:
            aligned, reason = is_direction_aligned_with_structure(direction, ctx.structure)
            if aligned:
                score += 1
                # Shorten reason for display
                short = reason[:60] if len(reason) > 60 else reason
                reasons.append(f"Structure: {short} (+1)")
        except Exception:
            pass

    return min(score, 5), reasons


def enhanced_score_bar(
    row: pd.Series,
    direction: str,
    df: pd.DataFrame = None,
    bar_index: int = None,
    ctx: InstitutionalContext = None,
) -> Tuple[int, List[str]]:
    """
    Full 12-point scoring: 7 base + 5 institutional.
    Drop-in replacement for the existing score_bar() in run_today_bt.py.

    If df/bar_index/ctx not provided, falls back to base 7-point scoring.
    """
    # ── Base 7-point scoring (unchanged from run_today_bt.py) ─────
    base_score = 0
    reasons = []

    # 1. EMA5 vs EMA13
    if direction == "CE" and row["ema5"] > row["ema13"]:
        base_score += 1; reasons.append("EMA5>EMA13")
    elif direction == "PE" and row["ema5"] < row["ema13"]:
        base_score += 1; reasons.append("EMA5<EMA13")

    # 2. EMA13 vs EMA20
    if direction == "CE" and row["ema13"] > row["ema20"]:
        base_score += 1; reasons.append("EMA13>EMA20")
    elif direction == "PE" and row["ema13"] < row["ema20"]:
        base_score += 1; reasons.append("EMA13<EMA20")

    # 3. SuperTrend
    stb = bool(row["st_bull"])
    if direction == "CE" and stb:
        base_score += 1; reasons.append("ST-green")
    elif direction == "PE" and not stb:
        base_score += 1; reasons.append("ST-red")

    # 4. RSI
    rsi_val = float(row["rsi"])
    if direction == "CE" and 35 <= rsi_val <= 68:
        base_score += 1; reasons.append(f"RSI={rsi_val:.0f}")
    elif direction == "PE" and 32 <= rsi_val <= 65:
        base_score += 1; reasons.append(f"RSI={rsi_val:.0f}")

    # 5. Bollinger
    pct_b = float(row["bb_pct_b"])
    if direction == "CE" and pct_b <= 0.45:
        base_score += 1; reasons.append(f"BB={pct_b:.2f}")
    elif direction == "PE" and pct_b >= 0.55:
        base_score += 1; reasons.append(f"BB={pct_b:.2f}")

    # 6. Volume
    vr = float(row["vol_ratio"])
    if vr >= 1.2:
        base_score += 1; reasons.append(f"Vol={vr:.1f}x")

    # 7. VWAP
    cl, vwap = float(row["close"]), float(row["vwap"])
    if direction == "CE" and cl > vwap:
        base_score += 1; reasons.append("AboveVWAP")
    elif direction == "PE" and cl < vwap:
        base_score += 1; reasons.append("BelowVWAP")

    # ── Institutional add-on (5 points) ───────────────────────────
    inst_score = 0
    inst_reasons = []

    if df is not None and bar_index is not None and ctx is not None and ctx.initialized:
        inst_score, inst_reasons = institutional_score(df, bar_index, direction, ctx)

    total = base_score + inst_score
    all_reasons = reasons + inst_reasons

    return total, all_reasons


def _quick_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Quick ATR computation."""
    if len(df) < period + 1:
        return float(df["high"].iloc[-1] - df["low"].iloc[-1])
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])
