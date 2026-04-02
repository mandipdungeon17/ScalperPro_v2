"""
=============================================================================
SCALPER PRO v2 — Master Orchestrator
=============================================================================
The CORRECT 3-layer flow:

  LAYER 1: Index Level Marker
  ────────────────────────────
  "NIFTY is at 23,480, which is 20 pts from SUPPORT at 23,460
   (daily level, 4 touches, Fib 0.618 confluence)"
   → DECISION: Look for CE

  LAYER 2: Premium Swing Detector
  ────────────────────────────────
  "NIFTY 23500 CE premium is at ₹128
   Support at ₹118 (3 touches, bounced 18pts avg, confirmed on 5min+15min)
   Resistance at ₹148"
   → SETUP: Buy at ₹120, SL ₹108, Target ₹140 (20pt target, 12pt SL)

  LAYER 3: Greek Strike Selector
  ───────────────────────────────
  "23500 CE: Δ=0.45, Γ=0.0018, Θ=-1.2%/day, Premium ₹128
   Score: 13/15 — ideal swing strike"
   → CONFIRMED: Trade 23500 CE

  EXECUTE → Dhan order + Telegram alert
=============================================================================
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from scalper.core.index_levels import IndexLevelMarker, IndexProximitySignal
from scalper.core.premium_swings import PremiumSwingDetector, PremiumSwingSetup
from scalper.core.greek_selector import GreekStrikeSelector, StrikeRecommendation
from scalper.config.settings import INDEX_CONFIGS

logger = logging.getLogger(__name__)


@dataclass
class SwingTradeDecision:
    """
    Complete trade decision from all 3 layers.
    This is what gets sent to execution + Telegram.
    """
    # Metadata
    timestamp: str
    index: str
    decision_id: str

    # Layer 1 output
    index_level_action: str         # "BUY_CE_AT_SUPPORT" / "BUY_PE_AT_RESISTANCE"
    index_proximity: str            # "AT_LEVEL" / "APPROACHING"
    index_level_price: float        # The S/R level price
    index_level_strength: float
    index_level_type: str           # "SUPPORT" / "RESISTANCE"
    index_level_timeframe: str
    index_level_touches: int
    index_distance_atr: float
    index_trend: str

    # Layer 2 output
    option_type: str                # "CE" or "PE"
    strike: int
    current_premium: float
    entry_premium: float
    stoploss_premium: float
    target_premium: float
    sl_points: float
    target_points: float
    risk_reward: float
    premium_support_touches: int
    premium_avg_bounce: float
    premium_trend: str
    premium_trend_action: str
    setup_quality: str              # "A+", "A", "B", "C"
    multi_tf_confirmation: int      # How many timeframes confirm

    # Layer 3 output
    delta: float
    gamma: float
    theta: float
    theta_pct_per_day: float
    iv: float
    greek_score: int
    moneyness: str

    # Final decision
    should_trade: bool
    confidence: float               # 0-1
    all_reasons: List[str]


class SwingOrchestrator:
    """
    Orchestrates the 3-layer swing trading system.

    Usage:
        orch = SwingOrchestrator()

        decision = orch.analyze(
            index="NIFTY",
            current_spot=23480,
            daily_df=daily_ohlcv,
            hourly_df=hourly_ohlcv,
            fifteen_min_df=fifteen_min_ohlcv,
            premium_1min=option_1min_ohlcv,
            premium_5min=option_5min_ohlcv,
            premium_15min=option_15min_ohlcv,
            dte=5,
            base_iv=14
        )

        if decision.should_trade:
            execute(decision)
    """

    def __init__(self):
        self.index_marker = IndexLevelMarker()
        self.premium_detector = PremiumSwingDetector()
        self.greek_selector = GreekStrikeSelector()
        self._decision_counter = 0

    def analyze(
        self,
        index: str,
        current_spot: float,
        # Layer 1 data
        daily_df,
        weekly_df=None,
        hourly_df=None,
        fifteen_min_df=None,
        # Layer 2 data — premium charts for the selected option
        premium_1min=None,
        premium_5min=None,
        premium_15min=None,
        # Layer 3 params
        dte: int = 5,
        base_iv: float = 14,
        # Override (if you already know which strike to look at)
        force_strike: Optional[int] = None,
        force_option_type: Optional[str] = None,
    ) -> Optional[SwingTradeDecision]:
        """
        Run the complete 3-layer analysis.
        Returns a SwingTradeDecision or None if no setup found.
        """
        config = INDEX_CONFIGS.get(index)
        if not config:
            logger.error(f"Unknown index: {index}")
            return None

        all_reasons = []

        # ══════════════════════════════════════════════════════════
        # LAYER 1: Mark index levels and check proximity
        # ══════════════════════════════════════════════════════════

        logger.info(f"[L1] Marking index levels for {index} @ {current_spot}")

        self.index_marker.mark_levels(
            daily_df=daily_df,
            weekly_df=weekly_df,
            hourly_df=hourly_df,
            fifteen_min_df=fifteen_min_df,
            index=index,
        )

        proximity = self.index_marker.check_proximity(current_spot, index)

        if proximity.action == "WAIT" or proximity.proximity_zone == "FAR":
            logger.info(f"[L1] Index not near any key level. Distance: {proximity.distance_atr:.2f} ATR. WAIT.")
            return None

        option_type = force_option_type or proximity.direction
        if option_type not in ("CE", "PE"):
            logger.info(f"[L1] No clear direction: {proximity.action}")
            return None

        # ── Rejection candle confirmation (AT_LEVEL only) ──────────────────
        # For CE: latest candle's low must be near the support level AND
        #         close must be above it (wick touched, body closed above).
        # For PE: latest candle's high must be near resistance AND
        #         close must be below it (wick touched, body closed below).
        # Applied only when AT_LEVEL (< 0.3 ATR) — skip for APPROACHING.
        if proximity.proximity_zone == "AT_LEVEL" and proximity.nearest_level is not None:
            ref_df = fifteen_min_df if (fifteen_min_df is not None and len(fifteen_min_df) > 0) \
                     else daily_df
            if ref_df is not None and len(ref_df) > 0:
                last_bar = ref_df.iloc[-1]
                level_price = proximity.nearest_level.price
                atr = self.index_marker._daily_atr if self.index_marker._daily_atr > 0 \
                      else current_spot * 0.01
                confirmed = False
                if option_type == "CE":
                    low_near = abs(last_bar["low"] - level_price) < atr * 0.20
                    close_above = last_bar["close"] > level_price
                    confirmed = low_near and close_above
                    if not confirmed:
                        logger.info(
                            f"[L1] CE rejection candle not confirmed: "
                            f"low={last_bar['low']:.0f}, close={last_bar['close']:.0f}, "
                            f"level={level_price:.0f}. WAIT."
                        )
                        return None
                elif option_type == "PE":
                    high_near = abs(last_bar["high"] - level_price) < atr * 0.20
                    close_below = last_bar["close"] < level_price
                    confirmed = high_near and close_below
                    if not confirmed:
                        logger.info(
                            f"[L1] PE rejection candle not confirmed: "
                            f"high={last_bar['high']:.0f}, close={last_bar['close']:.0f}, "
                            f"level={level_price:.0f}. WAIT."
                        )
                        return None

        logger.info(
            f"[L1] {index} near {proximity.nearest_level.level_type.value} "
            f"@ {proximity.nearest_level.price:.0f} "
            f"({proximity.distance_atr:.2f} ATR away) "
            f"-> Look for {option_type}"
        )
        all_reasons.append(
            f"Index near {proximity.nearest_level.level_type.value} "
            f"@ {proximity.nearest_level.price:.0f} "
            f"({proximity.nearest_level.touches} touches, "
            f"{proximity.nearest_level.timeframe.value}, "
            f"strength {proximity.nearest_level.strength:.2f})"
        )

        # ── Multi-Indicator Confirmation (5EMA, 13EMA H/L, SuperTrend, ─────
        #    RSI, Volume, Bollinger Band, India VIX)
        if fifteen_min_df is not None and len(fifteen_min_df) >= 30:
            try:
                from scalper.core.indicators import (
                    score_signal, fetch_india_vix, MIN_CONF_SCORE
                )
                india_vix = fetch_india_vix()
                bar_idx   = len(fifteen_min_df) - 1
                sig_score = score_signal(
                    df=fifteen_min_df,
                    bar_idx=bar_idx,
                    direction=option_type,
                    proximity_zone=proximity.proximity_zone,
                    india_vix=india_vix,
                    oi_bull=None,
                )
                logger.info(
                    f"[INDICATORS] Score {sig_score.score_breakdown} "
                    f"| VIX={india_vix:.1f} | RSI={sig_score.rsi:.0f} "
                    f"| ST={'BULL' if sig_score.supertrend_bull else 'BEAR'} "
                    f"| BB_pct_b={sig_score.bb_pct_b:.2f}"
                )
                if not sig_score.trade_allowed:
                    logger.info(f"[INDICATORS] Signal blocked: {sig_score.block_reason}")
                    return None
                all_reasons.append(
                    f"Indicators ({sig_score.score_breakdown}) | "
                    f"RSI={sig_score.rsi:.0f} | ST={'Bull' if sig_score.supertrend_bull else 'Bear'} | "
                    f"VIX={india_vix:.1f} | BB={sig_score.bb_pct_b:.2f}"
                )
            except Exception as e:
                logger.warning(f"[INDICATORS] Scoring failed (continuing without): {e}")

        # ══════════════════════════════════════════════════════════
        # LAYER 3: Select best strike (before Layer 2, because
        #          we need to know WHICH option's premium to analyze)
        # ══════════════════════════════════════════════════════════

        logger.info(f"[L3] Selecting optimal {option_type} strike for swing")

        strike_rec = self.greek_selector.select_strike(
            spot=current_spot,
            option_type=option_type,
            dte=dte,
            base_iv=base_iv,
            strike_interval=config.strike_interval,
            target_swing_points=20,
        )

        if strike_rec.best_strike is None:
            logger.info("[L3] No suitable strike found")
            return None

        best = strike_rec.best_strike
        selected_strike = force_strike or best.strike

        logger.info(
            f"[L3] Best strike: {selected_strike} {option_type} "
            f"Δ={best.delta:.3f} Γ={best.gamma:.5f} Θ%={best.theta_pct_per_day:.1f}%/day "
            f"Score={best.swing_score}/15"
        )
        all_reasons.append(
            f"Strike {selected_strike} {option_type}: "
            f"Δ={best.delta:.3f}, Γ={best.gamma:.5f}, "
            f"Θ={best.theta_pct_per_day:.1f}%/day ({best.moneyness})"
        )

        # ══════════════════════════════════════════════════════════
        # LAYER 2: Analyze premium swing levels
        # ══════════════════════════════════════════════════════════

        if premium_1min is None:
            logger.info("[L2] No premium data provided — skipping premium swing analysis")
            logger.info("[L2] In live mode, fetch 1min/5min/15min data for "
                        f"{selected_strike} {option_type} and re-run")
            return None

        logger.info(f"[L2] Analyzing premium swings for {selected_strike} {option_type}")

        setup = self.premium_detector.analyze(
            premium_1min=premium_1min,
            premium_5min=premium_5min,
            premium_15min=premium_15min,
            option_type=option_type,
            strike=selected_strike,
            index=index,
        )

        if setup is None:
            logger.info("[L2] No valid swing setup found on premium chart")
            return None

        logger.info(
            f"[L2] SETUP FOUND: {setup.setup_quality} quality | "
            f"Entry ₹{setup.entry_premium} → Target ₹{setup.target_premium} "
            f"(+{setup.target_points}pts) | SL ₹{setup.stoploss_premium} "
            f"(-{setup.sl_points}pts) | R:R 1:{setup.risk_reward}"
        )
        all_reasons.extend(setup.reasons)

        # ══════════════════════════════════════════════════════════
        # DECISION: Should we trade?
        # ══════════════════════════════════════════════════════════

        should_trade = (
            setup.setup_quality in ("A+", "A") and
            best.swing_score >= 8 and
            proximity.proximity_zone in ("AT_LEVEL", "APPROACHING") and
            setup.risk_reward >= 1.0 and
            setup.sl_points <= 20
        )

        # B-grade setups allowed if everything else is strong
        if not should_trade and setup.setup_quality == "B":
            if (best.swing_score >= 10 and
                proximity.proximity_zone == "AT_LEVEL" and
                setup.risk_reward >= 1.5 and
                setup.confirmation_count >= 2):
                should_trade = True
                all_reasons.append("B-grade allowed: strong Greeks + at level + multi-TF")

        confidence = (
            proximity.confidence * 0.3 +
            setup.confidence * 0.4 +
            (best.swing_score / 15) * 0.3
        )

        self._decision_counter += 1
        decision_id = f"SWG_{index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._decision_counter:03d}"

        decision = SwingTradeDecision(
            timestamp=datetime.now().isoformat(),
            index=index,
            decision_id=decision_id,

            # Layer 1
            index_level_action=proximity.action,
            index_proximity=proximity.proximity_zone,
            index_level_price=proximity.nearest_level.price,
            index_level_strength=proximity.nearest_level.strength,
            index_level_type=proximity.nearest_level.level_type.value,
            index_level_timeframe=proximity.nearest_level.timeframe.value,
            index_level_touches=proximity.nearest_level.touches,
            index_distance_atr=proximity.distance_atr,
            index_trend=proximity.trend_context,

            # Layer 2
            option_type=option_type,
            strike=selected_strike,
            current_premium=setup.current_premium,
            entry_premium=setup.entry_premium,
            stoploss_premium=setup.stoploss_premium,
            target_premium=setup.target_premium,
            sl_points=setup.sl_points,
            target_points=setup.target_points,
            risk_reward=setup.risk_reward,
            premium_support_touches=setup.entry_at_level.touches,
            premium_avg_bounce=float(
                np.mean(setup.entry_at_level.bounce_magnitudes)
                if setup.entry_at_level.bounce_magnitudes else 0
            ),
            premium_trend=setup.premium_trend,
            premium_trend_action=setup.premium_trend_action,
            setup_quality=setup.setup_quality,
            multi_tf_confirmation=setup.confirmation_count,

            # Layer 3
            delta=best.delta,
            gamma=best.gamma,
            theta=best.theta,
            theta_pct_per_day=best.theta_pct_per_day,
            iv=best.iv,
            greek_score=best.swing_score,
            moneyness=best.moneyness,

            # Decision
            should_trade=should_trade,
            confidence=round(confidence, 3),
            all_reasons=all_reasons,
        )

        action = "🟢 TRADE" if should_trade else "⏸️ SKIP"
        logger.info(
            f"[DECISION] {action} | {decision_id} | "
            f"Quality={setup.setup_quality} | Confidence={confidence:.0%} | "
            f"R:R=1:{setup.risk_reward} | SL={setup.sl_points}pts"
        )

        return decision

    def format_telegram_alert(self, d: SwingTradeDecision) -> str:
        """Format decision as Telegram alert message."""
        if d.should_trade:
            header = "🟢 <b>SWING TRADE SIGNAL</b>"
        else:
            header = "⏸️ <b>SWING SETUP DETECTED (NOT TRADING)</b>"

        return f"""
{header}

<b>━━ LAYER 1: Index Level ━━</b>
{d.index} @ {d.index_level_price:.0f} ({d.index_level_type})
Touches: {d.index_level_touches} | Timeframe: {d.index_level_timeframe}
Distance: {d.index_distance_atr:.2f} ATR | Trend: {d.index_trend}
→ {d.index_level_action}

<b>━━ LAYER 2: Premium Swing ━━</b>
{d.strike} {d.option_type} | Premium: ₹{d.current_premium:.2f}
<b>Entry:</b> ₹{d.entry_premium:.2f}
<b>Target:</b> ₹{d.target_premium:.2f} (+{d.target_points:.1f} pts)
<b>SL:</b> ₹{d.stoploss_premium:.2f} (-{d.sl_points:.1f} pts)
<b>R:R:</b> 1:{d.risk_reward:.1f}
Support: {d.premium_support_touches} touches | Avg bounce: {d.premium_avg_bounce:.1f}pts
{d.confirmation_count}x TF confirmed | Trend: {d.premium_trend_action}
<b>Grade: {d.setup_quality}</b>

<b>━━ LAYER 3: Greeks ━━</b>
Δ={d.delta:.3f} | Γ={d.gamma:.5f} | Θ={d.theta_pct_per_day:.1f}%/day
IV={d.iv:.1f}% | {d.moneyness} | Score: {d.greek_score}/15

<b>━━ DECISION ━━</b>
{"🟢 TRADE" if d.should_trade else "⏸️ SKIP"} | Confidence: {d.confidence:.0%}

<b>Reasons:</b>
{chr(10).join(f"  • {r}" for r in d.all_reasons[:6])}

<i>⏰ {d.timestamp}</i>
""".strip()


# Need numpy for mean calculation
import numpy as np
