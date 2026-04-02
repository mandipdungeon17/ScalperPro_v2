"""
=============================================================================
SCALPER PRO - Open Interest & Options Data Analyzer
=============================================================================
Fetches and analyzes OI data from NSE option chain for signal generation.
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import requests
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class OISnapshot:
    """Snapshot of OI analysis for a given index at a point in time."""
    timestamp: datetime
    index_name: str
    spot_price: float
    atm_strike: int
    pcr: float
    max_pain: float
    total_ce_oi: int
    total_pe_oi: int
    straddle_premium: float
    expected_range: Tuple[float, float]
    ce_resistance_levels: List[Dict]   # [{strike, oi, oi_change}]
    pe_support_levels: List[Dict]      # [{strike, oi, oi_change}]
    oi_spurts: List[Dict]              # Large OI additions detected
    signal: str                         # "BULLISH", "BEARISH", "NEUTRAL"
    signal_strength: float              # 0-1


class OIAnalyzer:
    """Analyzes NSE option chain OI data for trading signals."""

    NSE_OPTION_CHAIN_URL = "https://www.nseindia.com/api/option-chain-indices"
    NSE_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/option-chain",
    }

    SYMBOL_MAP = {
        "NIFTY": "NIFTY",
        "BANKNIFTY": "BANKNIFTY",
        "FINNIFTY": "FINNIFTY",
        "MIDCPNIFTY": "NIFTY MIDCAP SELECT",
    }

    def __init__(self):
        self._session = None
        self._last_fetch_time = {}
        self._cache = {}
        self._fetch_interval = 30  # Minimum seconds between fetches per symbol

    def _get_session(self) -> requests.Session:
        """Get or create a session with NSE cookies."""
        if self._session is None:
            self._session = requests.Session()
            # Hit main page first to get cookies
            try:
                self._session.get(
                    "https://www.nseindia.com/option-chain",
                    headers=self.NSE_HEADERS,
                    timeout=10
                )
            except Exception as e:
                logger.warning(f"Failed to initialize NSE session: {e}")
        return self._session

    def fetch_option_chain(self, index: str) -> Optional[Dict]:
        """
        Fetch raw option chain data from NSE.
        Returns the full JSON response or None on failure.
        """
        nse_symbol = self.SYMBOL_MAP.get(index)
        if not nse_symbol:
            logger.error(f"Unknown index: {index}")
            return None

        # Rate limit
        now = time.time()
        last = self._last_fetch_time.get(index, 0)
        if now - last < self._fetch_interval:
            if index in self._cache:
                return self._cache[index]
            time.sleep(self._fetch_interval - (now - last))

        try:
            session = self._get_session()
            resp = session.get(
                self.NSE_OPTION_CHAIN_URL,
                params={"symbol": nse_symbol},
                headers=self.NSE_HEADERS,
                timeout=15
            )
            if resp.status_code == 200:
                data = resp.json()
                self._cache[index] = data
                self._last_fetch_time[index] = time.time()
                return data
            else:
                logger.warning(f"NSE returned status {resp.status_code} for {index}")
                return self._cache.get(index)
        except Exception as e:
            logger.error(f"Error fetching option chain for {index}: {e}")
            return self._cache.get(index)

    def analyze(self, index: str, chain_data: Optional[Dict] = None) -> Optional[OISnapshot]:
        """
        Full OI analysis for an index.
        Can accept pre-fetched chain_data or will fetch live.
        """
        if chain_data is None:
            chain_data = self.fetch_option_chain(index)
        if chain_data is None:
            return None

        try:
            records = chain_data.get("records", {})
            filtered = chain_data.get("filtered", {})
            data = filtered.get("data", records.get("data", []))

            spot_price = records.get("underlyingValue", 0)
            if spot_price == 0:
                return None

            # Determine ATM strike
            strike_prices = sorted(set(r.get("strikePrice", 0) for r in data))
            atm_strike = min(strike_prices, key=lambda x: abs(x - spot_price))

            # Aggregate OI data
            ce_oi_map = {}
            pe_oi_map = {}
            ce_oi_change_map = {}
            pe_oi_change_map = {}
            total_ce_oi = 0
            total_pe_oi = 0
            atm_ce_ltp = 0
            atm_pe_ltp = 0

            for row in data:
                strike = row.get("strikePrice", 0)

                ce = row.get("CE", {})
                pe = row.get("PE", {})

                if ce:
                    oi = ce.get("openInterest", 0)
                    oi_chg = ce.get("changeinOpenInterest", 0)
                    ce_oi_map[strike] = oi
                    ce_oi_change_map[strike] = oi_chg
                    total_ce_oi += oi
                    if strike == atm_strike:
                        atm_ce_ltp = ce.get("lastPrice", 0)

                if pe:
                    oi = pe.get("openInterest", 0)
                    oi_chg = pe.get("changeinOpenInterest", 0)
                    pe_oi_map[strike] = oi
                    pe_oi_change_map[strike] = oi_chg
                    total_pe_oi += oi
                    if strike == atm_strike:
                        atm_pe_ltp = pe.get("lastPrice", 0)

            # PCR
            pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 1.0

            # Max Pain calculation
            max_pain = self._calculate_max_pain(
                strike_prices, ce_oi_map, pe_oi_map
            )

            # Straddle premium & expected range
            straddle_premium = atm_ce_ltp + atm_pe_ltp
            expected_range = (
                spot_price - straddle_premium,
                spot_price + straddle_premium
            )

            # Top CE resistance levels (highest CE OI = resistance)
            ce_resistance = sorted(
                [{"strike": k, "oi": v, "oi_change": ce_oi_change_map.get(k, 0)}
                 for k, v in ce_oi_map.items() if k >= atm_strike],
                key=lambda x: x["oi"], reverse=True
            )[:5]

            # Top PE support levels (highest PE OI = support)
            pe_support = sorted(
                [{"strike": k, "oi": v, "oi_change": pe_oi_change_map.get(k, 0)}
                 for k, v in pe_oi_map.items() if k <= atm_strike],
                key=lambda x: x["oi"], reverse=True
            )[:5]

            # OI Spurts (large intraday OI additions)
            oi_spurts = self._detect_oi_spurts(
                ce_oi_change_map, pe_oi_change_map, atm_strike, spot_price
            )

            # Generate OI-based signal
            signal, strength = self._generate_oi_signal(
                pcr, ce_oi_change_map, pe_oi_change_map,
                atm_strike, spot_price, max_pain
            )

            return OISnapshot(
                timestamp=datetime.now(),
                index_name=index,
                spot_price=spot_price,
                atm_strike=atm_strike,
                pcr=round(pcr, 2),
                max_pain=max_pain,
                total_ce_oi=total_ce_oi,
                total_pe_oi=total_pe_oi,
                straddle_premium=round(straddle_premium, 2),
                expected_range=expected_range,
                ce_resistance_levels=ce_resistance,
                pe_support_levels=pe_support,
                oi_spurts=oi_spurts,
                signal=signal,
                signal_strength=round(strength, 2),
            )

        except Exception as e:
            logger.error(f"Error analyzing OI for {index}: {e}")
            return None

    def _calculate_max_pain(self, strikes: list, ce_oi: dict, pe_oi: dict) -> float:
        """Max pain = strike where total loss to option writers is minimum."""
        min_pain = float("inf")
        max_pain_strike = strikes[0] if strikes else 0

        for test_strike in strikes:
            total_loss = 0
            for s in strikes:
                # CE writers' loss at test_strike
                if test_strike > s:
                    total_loss += (test_strike - s) * ce_oi.get(s, 0)
                # PE writers' loss at test_strike
                if test_strike < s:
                    total_loss += (s - test_strike) * pe_oi.get(s, 0)

            if total_loss < min_pain:
                min_pain = total_loss
                max_pain_strike = test_strike

        return max_pain_strike

    def _detect_oi_spurts(self, ce_oi_change: dict, pe_oi_change: dict,
                          atm_strike: int, spot_price: float) -> List[Dict]:
        """Detect large OI additions that signal institutional activity."""
        spurts = []

        # Check CE side
        if ce_oi_change:
            ce_changes = list(ce_oi_change.values())
            ce_mean = np.mean(ce_changes) if ce_changes else 0
            ce_std = np.std(ce_changes) if ce_changes else 1
            threshold = ce_mean + 2 * ce_std

            for strike, change in ce_oi_change.items():
                if change > threshold and change > 0:
                    spurts.append({
                        "type": "CE",
                        "strike": strike,
                        "oi_change": change,
                        "interpretation": "Resistance" if change > 0 else "Unwinding",
                        "relative_to_spot": "above" if strike > spot_price else "below"
                    })

        # Check PE side
        if pe_oi_change:
            pe_changes = list(pe_oi_change.values())
            pe_mean = np.mean(pe_changes) if pe_changes else 0
            pe_std = np.std(pe_changes) if pe_changes else 1
            threshold = pe_mean + 2 * pe_std

            for strike, change in pe_oi_change.items():
                if change > threshold and change > 0:
                    spurts.append({
                        "type": "PE",
                        "strike": strike,
                        "oi_change": change,
                        "interpretation": "Support" if change > 0 else "Unwinding",
                        "relative_to_spot": "above" if strike > spot_price else "below"
                    })

        return spurts

    def _generate_oi_signal(self, pcr: float, ce_oi_change: dict,
                            pe_oi_change: dict, atm_strike: int,
                            spot_price: float, max_pain: float) -> Tuple[str, float]:
        """Generate a composite OI signal."""
        score = 0
        max_score = 5

        # 1. PCR signal
        if pcr > 1.2:
            score += 1   # Bullish (PE writing = support)
        elif pcr < 0.7:
            score -= 1   # Bearish (CE writing = resistance)

        # 2. Max pain direction
        if max_pain > spot_price * 1.002:
            score += 1   # Market may drift up toward max pain
        elif max_pain < spot_price * 0.998:
            score -= 1   # Market may drift down

        # 3. Net OI change direction near ATM
        nearby_range = atm_strike * 0.02  # 2% range around ATM
        ce_near_change = sum(v for k, v in ce_oi_change.items()
                            if abs(k - atm_strike) <= nearby_range)
        pe_near_change = sum(v for k, v in pe_oi_change.items()
                            if abs(k - atm_strike) <= nearby_range)

        if pe_near_change > ce_near_change * 1.5:
            score += 1   # More PE writing = bullish
        elif ce_near_change > pe_near_change * 1.5:
            score -= 1   # More CE writing = bearish

        # 4. Immediate resistance/support pressure
        ce_above = sum(v for k, v in ce_oi_change.items()
                       if k > spot_price and k <= spot_price + nearby_range and v > 0)
        pe_below = sum(v for k, v in pe_oi_change.items()
                       if k < spot_price and k >= spot_price - nearby_range and v > 0)

        if pe_below > ce_above:
            score += 1   # Strong put support below
        elif ce_above > pe_below:
            score -= 1   # Strong call resistance above

        # Signal determination
        if score >= 2:
            signal = "BULLISH"
        elif score <= -2:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        strength = abs(score) / max_score
        return signal, strength


class StrikeSelector:
    """Selects optimal strike price for scalping with minimal decay."""

    def __init__(self, params=None):
        from scalper.config.settings import StrikeSelectionParams
        self.params = params or StrikeSelectionParams()

    def select_strike(
        self,
        index: str,
        spot_price: float,
        direction: str,          # "LONG" or "SHORT"
        chain_data: Dict,
        days_to_expiry: int,
        strategy: str = "scalp"  # "scalp" or "swing"
    ) -> Optional[Dict]:
        """
        Select the best strike based on:
        - Delta >= 0.65 (ITM preference)
        - Low theta relative to target
        - Good liquidity (OI and spread)
        - IV not too inflated

        Returns: {
            "strike": int, "option_type": "CE"|"PE",
            "delta": float, "theta": float, "gamma": float,
            "iv": float, "ltp": float, "oi": int,
            "expected_premium_move": float,
            "reason": str
        }
        """
        from scalper.config.settings import INDEX_CONFIGS

        config = INDEX_CONFIGS.get(index)
        if not config:
            return None

        option_type = "CE" if direction == "LONG" else "PE"

        records = chain_data.get("records", {})
        filtered = chain_data.get("filtered", {})
        data = filtered.get("data", records.get("data", []))

        candidates = []

        for row in data:
            strike = row.get("strikePrice", 0)
            opt_data = row.get(option_type, {})
            if not opt_data:
                continue

            ltp = opt_data.get("lastPrice", 0)
            oi = opt_data.get("openInterest", 0)
            iv = opt_data.get("impliedVolatility", 0)
            bid = opt_data.get("bidprice", 0) or opt_data.get("bidPrice", 0)
            ask = opt_data.get("askprice", 0) or opt_data.get("askPrice", 0)

            if ltp <= 0 or oi < self.params.min_oi:
                continue

            # Estimate delta from moneyness (Black-Scholes approximation)
            moneyness = (spot_price - strike) / spot_price if option_type == "CE" \
                else (strike - spot_price) / spot_price
            estimated_delta = self._estimate_delta(moneyness, iv, days_to_expiry)

            # Estimate theta
            estimated_theta = self._estimate_theta(
                ltp, iv, spot_price, days_to_expiry
            )

            # Spread check
            spread_ratio = bid / ask if ask > 0 else 0

            # Score the candidate
            target_points = config.scalp_target if strategy == "scalp" else config.swing_target
            expected_move = estimated_delta * target_points

            # Theta cost for expected holding time (5 min for scalp, 2 hours for swing)
            hold_hours = 0.083 if strategy == "scalp" else 2.0
            theta_cost = abs(estimated_theta) * hold_hours / 6.5  # 6.5 trading hours

            score = 0
            reasons = []

            # Delta scoring
            if self.params.ideal_delta_low <= abs(estimated_delta) <= self.params.ideal_delta_high:
                score += 3
                reasons.append(f"Ideal delta {estimated_delta:.2f}")
            elif abs(estimated_delta) >= self.params.min_delta:
                score += 2
                reasons.append(f"Good delta {estimated_delta:.2f}")
            else:
                continue  # Skip low delta options

            # Theta scoring (lower is better)
            if theta_cost < target_points * self.params.max_theta_pct_of_target:
                score += 2
                reasons.append(f"Low theta cost ₹{theta_cost:.1f}")
            elif theta_cost < target_points * self.params.max_theta_pct_of_target * 2:
                score += 1

            # Liquidity scoring
            if oi > self.params.min_oi * 5:
                score += 2
                reasons.append(f"High liquidity OI={oi:,}")
            elif oi > self.params.min_oi * 2:
                score += 1

            # Spread scoring
            if spread_ratio > self.params.min_bid_ask_ratio:
                score += 1
                reasons.append("Tight spread")

            # DTE adjustment: on expiry day, prefer deeper ITM
            if days_to_expiry <= self.params.dte_switch_to_next_week:
                if moneyness > 0.01:  # At least 1% ITM
                    score += 1
                    reasons.append("Deep ITM for expiry day")

            candidates.append({
                "strike": strike,
                "option_type": option_type,
                "delta": round(estimated_delta, 3),
                "theta": round(estimated_theta, 2),
                "gamma": round(self._estimate_gamma(estimated_delta, spot_price, iv, days_to_expiry), 5),
                "iv": round(iv, 2),
                "ltp": ltp,
                "oi": oi,
                "bid": bid,
                "ask": ask,
                "spread_ratio": round(spread_ratio, 3),
                "expected_premium_move": round(expected_move, 2),
                "theta_cost": round(theta_cost, 2),
                "score": score,
                "reason": " | ".join(reasons),
            })

        if not candidates:
            return None

        # Return highest scored candidate
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]
        logger.info(
            f"Selected {best['option_type']} {best['strike']} for {index} "
            f"(Δ={best['delta']}, θ={best['theta']}, score={best['score']})"
        )
        return best

    def _estimate_delta(self, moneyness: float, iv: float, dte: int) -> float:
        """Rough delta estimation from moneyness and IV."""
        if dte <= 0:
            return 1.0 if moneyness > 0 else 0.0

        # Simplified: ITM options have delta > 0.5, OTM < 0.5
        # This is a fast approximation; real system would use Black-Scholes
        iv_factor = max(iv / 100, 0.1)
        t = dte / 365

        if moneyness > 0:  # ITM
            base = 0.5 + moneyness / (2 * iv_factor * np.sqrt(t) + 0.01)
            return min(base, 0.98)
        else:  # OTM
            base = 0.5 - abs(moneyness) / (2 * iv_factor * np.sqrt(t) + 0.01)
            return max(base, 0.02)

    def _estimate_theta(self, premium: float, iv: float,
                        spot: float, dte: int) -> float:
        """Rough theta estimation (negative value, per day)."""
        if dte <= 0:
            return -premium
        iv_dec = max(iv / 100, 0.1)
        t = max(dte / 365, 1/365)
        # Theta ≈ -S * σ * N'(d1) / (2 * √T)  simplified
        daily_theta = -(spot * iv_dec) / (2 * np.sqrt(t) * np.sqrt(365))
        # Scale by how much premium there is
        theta_ratio = daily_theta / spot
        return premium * theta_ratio

    def _estimate_gamma(self, delta: float, spot: float,
                        iv: float, dte: int) -> float:
        """Rough gamma estimation."""
        if dte <= 0:
            return 0
        iv_dec = max(iv / 100, 0.1)
        t = max(dte / 365, 1/365)
        return 1 / (spot * iv_dec * np.sqrt(t * 2 * np.pi))
