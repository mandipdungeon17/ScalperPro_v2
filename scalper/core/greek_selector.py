"""
=============================================================================
SCALPER PRO v2 — LAYER 3: Greek-Optimized Strike Selector
=============================================================================
Once Layer 1 says "CE" or "PE" and Layer 2 confirms swing setup,
this module picks the BEST strike for maximum swing momentum.

KEY DIFFERENCE from v1:
  v1 → picked high-delta ITM for 1:1 movement (scalping logic)
  v2 → picks based on GAMMA at support (swing bounce acceleration)
        + manageable THETA (swing needs 15-60 min to play out)
        + optimal DELTA range 0.30-0.55 (leverage sweet spot)

The best swing strike is often SLIGHT OTM to ATM because:
  - High Gamma → when premium bounces from support, delta INCREASES
    as price moves in your favor → acceleration effect
  - ATM has highest Gamma → sharpest bounce
  - Slight OTM has best leverage (smaller premium, bigger % move)
  - But too far OTM → delta too low, move is sluggish

=============================================================================
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GreekProfile:
    """Full Greek analysis for a single strike."""
    strike: int
    option_type: str         # "CE" or "PE"
    moneyness: str           # "ITM", "ATM", "OTM"
    moneyness_pct: float     # % distance from ATM

    # Greeks
    delta: float
    gamma: float
    theta: float             # Per day (negative)
    vega: float
    iv: float

    # Derived metrics
    premium: float
    theta_pct_per_day: float  # Theta as % of premium (decay rate)
    gamma_dollar: float       # Gamma × premium (bounce force)
    delta_target_move: float  # Expected premium move for target
    breakeven_bars: float     # How many 5-min bars to overcome theta

    # Scoring
    swing_score: int          # 0-15 composite score
    ranking: int              # 1 = best
    reasons: List[str]


@dataclass
class StrikeRecommendation:
    """Final strike recommendation with reasoning."""
    best_strike: GreekProfile
    runner_up: Optional[GreekProfile]
    all_analyzed: List[GreekProfile]
    reasoning: str


class GreekStrikeSelector:
    """
    Selects optimal strike for premium swing trading.

    Unlike scalping (which wants high delta for 1:1),
    swing trading wants:
      - Delta 0.30-0.55 (leverage sweet spot)
      - HIGH Gamma (bounce acceleration at support)
      - Theta < 1.5% of premium per day (manageable decay)
      - Premium in ₹60-300 range (enough room for 15-20 pt swing)
    """

    def __init__(self):
        self.r = 0.07  # India risk-free rate

    def select_strike(
        self,
        spot: float,
        option_type: str,       # "CE" or "PE"
        dte: int,               # Days to expiry
        base_iv: float,         # ATM IV %
        strike_interval: int,   # 50 for Nifty, 100 for BankNifty
        target_swing_points: float = 20,  # Expected premium swing
        num_strikes: int = 8,   # How many strikes to analyze each side
    ) -> StrikeRecommendation:
        """
        Analyze multiple strikes and pick the best for swing momentum.
        """
        # Generate strike range centered around ATM
        atm_strike = round(spot / strike_interval) * strike_interval
        strikes = [
            atm_strike + i * strike_interval
            for i in range(-num_strikes, num_strikes + 1)
        ]

        profiles = []
        for strike in strikes:
            profile = self._analyze_strike(
                spot, strike, option_type, dte, base_iv,
                target_swing_points
            )
            if profile is not None:
                profiles.append(profile)

        if not profiles:
            return StrikeRecommendation(
                best_strike=None, runner_up=None,
                all_analyzed=[], reasoning="No valid strikes found"
            )

        # Sort by swing_score descending
        profiles.sort(key=lambda p: p.swing_score, reverse=True)

        # Assign rankings
        for i, p in enumerate(profiles):
            p.ranking = i + 1

        best = profiles[0]
        runner = profiles[1] if len(profiles) > 1 else None

        reasoning = (
            f"Best: {best.strike} {best.option_type} ({best.moneyness}) | "
            f"Δ={best.delta:.3f} Γ={best.gamma:.5f} Θ={best.theta:.2f} | "
            f"Premium ₹{best.premium:.0f} | Score {best.swing_score}/15 | "
            f"{'; '.join(best.reasons[:3])}"
        )

        return StrikeRecommendation(
            best_strike=best,
            runner_up=runner,
            all_analyzed=profiles,
            reasoning=reasoning,
        )

    def _analyze_strike(
        self,
        spot: float,
        strike: int,
        option_type: str,
        dte: int,
        base_iv: float,
        target_swing: float,
    ) -> Optional[GreekProfile]:
        """Full Greek analysis for one strike."""
        S = spot
        K = strike
        T = max(dte / 365, 0.001)

        # IV smile: OTM strikes have higher IV
        moneyness_pct = (S - K) / S if option_type == "CE" else (K - S) / S
        iv = base_iv + abs(moneyness_pct) * 40  # Simple smile model
        sigma = iv / 100

        # Black-Scholes calculations
        d1 = (np.log(S / K) + (self.r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        Nd1 = self._norm_cdf(d1)
        Nd2 = self._norm_cdf(d2)
        nd1 = self._norm_pdf(d1)

        # Greeks
        gamma = nd1 / (S * sigma * np.sqrt(T))
        vega = S * nd1 * np.sqrt(T) / 100

        if option_type == "CE":
            delta = Nd1
            theta = (-S * nd1 * sigma / (2 * np.sqrt(T)) - self.r * K * np.exp(-self.r * T) * Nd2) / 365
            premium = max(S * Nd1 - K * np.exp(-self.r * T) * Nd2, 0.05)
        else:
            delta = Nd1 - 1
            theta = (-S * nd1 * sigma / (2 * np.sqrt(T)) + self.r * K * np.exp(-self.r * T) * (1 - Nd2)) / 365
            premium = max(K * np.exp(-self.r * T) * (1 - Nd2) - S * (1 - Nd1), 0.05)

        # Skip if premium is too low or too high
        if premium < 10 or premium > 500:
            return None

        abs_delta = abs(delta)

        # Derived metrics
        theta_pct = abs(theta) / premium * 100 if premium > 0 else 999
        gamma_dollar = gamma * premium
        delta_target_move = abs_delta * target_swing

        # Breakeven bars: how many 5-min bars to overcome theta
        # Theta per 5-min bar = theta / (6.5 * 12) = theta / 78
        theta_per_bar = abs(theta) / 78
        breakeven_bars = theta_per_bar / max(delta_target_move / 20, 0.01) if delta_target_move > 0 else 999

        # Moneyness classification
        if moneyness_pct > 0.005:
            moneyness = "ITM"
        elif moneyness_pct < -0.005:
            moneyness = "OTM"
        else:
            moneyness = "ATM"

        # ── SWING SCORING (0-15) ──────────────────────────────────

        score = 0
        reasons = []

        # 1. Delta range (max 4 pts)
        # Sweet spot: 0.30-0.55 for swing leverage
        if 0.35 <= abs_delta <= 0.55:
            score += 4
            reasons.append(f"Ideal Δ={abs_delta:.2f} (swing sweet spot)")
        elif 0.25 <= abs_delta <= 0.65:
            score += 2
            reasons.append(f"Good Δ={abs_delta:.2f}")
        elif 0.15 <= abs_delta <= 0.75:
            score += 1
            reasons.append(f"Acceptable Δ={abs_delta:.2f}")
        else:
            reasons.append(f"Δ={abs_delta:.2f} outside range")

        # 2. Gamma — bounce acceleration (max 3 pts)
        # Higher gamma at support = sharper bounce
        if gamma > 0.0015:
            score += 3
            reasons.append(f"High Γ={gamma:.5f} (sharp bounce)")
        elif gamma > 0.001:
            score += 2
            reasons.append(f"Good Γ={gamma:.5f}")
        elif gamma > 0.0005:
            score += 1

        # 3. Theta decay (max 3 pts)
        # Want theta < 1.5% of premium per day
        if theta_pct < 1.0:
            score += 3
            reasons.append(f"Very low decay: {theta_pct:.1f}%/day")
        elif theta_pct < 1.5:
            score += 2
            reasons.append(f"Low decay: {theta_pct:.1f}%/day")
        elif theta_pct < 2.5:
            score += 1
            reasons.append(f"Moderate decay: {theta_pct:.1f}%/day")
        else:
            reasons.append(f"High decay: {theta_pct:.1f}%/day ⚠️")

        # 4. Premium range (max 2 pts)
        # ₹60-300 is the sweet spot for 15-20 pt swings
        if 80 <= premium <= 250:
            score += 2
            reasons.append(f"Premium ₹{premium:.0f} (ideal range)")
        elif 40 <= premium <= 400:
            score += 1
            reasons.append(f"Premium ₹{premium:.0f}")

        # 5. Expected move vs target (max 3 pts)
        if delta_target_move >= target_swing * 0.8:
            score += 3
            reasons.append(f"Expected move: ₹{delta_target_move:.1f} for {target_swing:.0f}pt index move")
        elif delta_target_move >= target_swing * 0.5:
            score += 2
        elif delta_target_move >= target_swing * 0.3:
            score += 1

        return GreekProfile(
            strike=strike,
            option_type=option_type,
            moneyness=moneyness,
            moneyness_pct=round(moneyness_pct * 100, 2),
            delta=round(delta, 4),
            gamma=round(gamma, 6),
            theta=round(theta, 3),
            vega=round(vega, 3),
            iv=round(iv, 1),
            premium=round(premium, 2),
            theta_pct_per_day=round(theta_pct, 2),
            gamma_dollar=round(gamma_dollar, 5),
            delta_target_move=round(delta_target_move, 2),
            breakeven_bars=round(breakeven_bars, 1),
            swing_score=score,
            ranking=0,
            reasons=reasons,
        )

    @staticmethod
    def _norm_cdf(x: float) -> float:
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        p = 0.3275911
        sign = -1 if x < 0 else 1
        x = abs(x) / np.sqrt(2)
        t = 1 / (1 + p * x)
        y = 1 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1) * t * np.exp(-x*x)
        return 0.5 * (1 + sign * y)

    @staticmethod
    def _norm_pdf(x: float) -> float:
        return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)
