"""
=============================================================================
SCALPER PRO - OI Change Monitor
=============================================================================
Wraps the existing OIAnalyzer and tracks changes BETWEEN snapshots.

Alerts on:
  - PCR shift > 0.15 since last reading (sentiment flip)
  - Max pain moves by > 100 points (institutional repositioning)
  - Large OI spurts at specific strikes
  - OI unwinding (existing OI drops sharply at ATM strikes)
  - CE resistance or PE support level strengthening above 2x average

Runs every 5 minutes during market hours.
=============================================================================
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from scalper.indicators.oi_analyzer import OIAnalyzer, OISnapshot

logger = logging.getLogger(__name__)

# Thresholds for change detection
PCR_SHIFT_THRESHOLD    = 0.15     # PCR change triggers alert
MAX_PAIN_MOVE_THRESHOLD = 100     # points
OI_SPURT_MIN_CONTRACTS = 50_000   # minimum OI addition to report
CE_OI_BUILDUP_RATIO    = 2.0      # CE OI > 2x avg = strong resistance alert
PE_OI_BUILDUP_RATIO    = 2.0


@dataclass
class OIChangeAlert:
    index: str
    alert_type: str        # PCR_SHIFT, MAX_PAIN_MOVE, OI_SPURT, OI_UNWIND, OI_BUILDUP
    description: str
    current_snapshot: OISnapshot
    prev_snapshot: Optional[OISnapshot] = None
    fetched_at: datetime = field(default_factory=datetime.now)


@dataclass
class OIReport:
    index: str
    snapshot: OISnapshot
    alerts: List[OIChangeAlert]
    summary_line: str      # one-liner for dashboard
    fetched_at: datetime = field(default_factory=datetime.now)


class OIMonitor:
    """
    Tracks OI data over time and generates alerts on significant changes.
    One instance per index, or pass list of indices.
    """

    # Indices supported by NSE option chain API
    _NSE_SUPPORTED = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}

    def __init__(self, indices: List[str] = None):
        all_indices = indices or ["NIFTY", "BANKNIFTY"]
        # SENSEX trades on BSE — NSE OI API does not support it
        self.indices = [i for i in all_indices if i in self._NSE_SUPPORTED]
        if skipped := set(all_indices) - set(self.indices):
            logger.info(f"[OIMonitor] Skipping non-NSE indices for OI: {skipped}")
        self._analyzer = OIAnalyzer()
        self._prev_snapshots: Dict[str, OISnapshot] = {}

    # ── Public ───────────────────────────────────────────────────────────────

    def check(self) -> List[OIReport]:
        """
        Fetch OI for all indices and return OIReport list.
        Call every 5 minutes from orchestrator.
        """
        reports: List[OIReport] = []

        for index in self.indices:
            try:
                report = self._check_index(index)
                if report:
                    reports.append(report)
            except Exception as e:
                logger.debug(f"[OIMonitor] Error checking {index}: {e}")

        return reports

    def get_snapshot(self, index: str) -> Optional[OISnapshot]:
        """Return the most recent OI snapshot for an index (cached)."""
        return self._prev_snapshots.get(index)

    # ── Private ──────────────────────────────────────────────────────────────

    def _check_index(self, index: str) -> Optional[OIReport]:
        snapshot = self._analyzer.analyze(index)
        if snapshot is None:
            return None

        prev = self._prev_snapshots.get(index)
        alerts = self._detect_changes(snapshot, prev)
        summary = self._build_summary(snapshot)

        self._prev_snapshots[index] = snapshot

        return OIReport(
            index=index,
            snapshot=snapshot,
            alerts=alerts,
            summary_line=summary,
        )

    def _detect_changes(
        self,
        current: OISnapshot,
        prev: Optional[OISnapshot],
    ) -> List[OIChangeAlert]:
        alerts: List[OIChangeAlert] = []

        # ── 1. OI Spurts (always check) ──────────────────────────────────────
        for spurt in current.oi_spurts:
            if spurt["oi_change"] >= OI_SPURT_MIN_CONTRACTS:
                alerts.append(OIChangeAlert(
                    index=current.index_name,
                    alert_type="OI_SPURT",
                    description=(
                        f"{spurt['type']} {spurt['strike']} | "
                        f"+{spurt['oi_change']:,} contracts | "
                        f"{spurt['interpretation']} | "
                        f"Strike is {spurt['relative_to_spot']} spot"
                    ),
                    current_snapshot=current,
                ))

        # ── Compare vs previous snapshot ─────────────────────────────────────
        if prev is None:
            return alerts

        # ── 2. PCR shift ─────────────────────────────────────────────────────
        pcr_diff = abs(current.pcr - prev.pcr)
        if pcr_diff >= PCR_SHIFT_THRESHOLD:
            direction = "RISING" if current.pcr > prev.pcr else "FALLING"
            sentiment = "BULLISH" if direction == "RISING" else "BEARISH"
            alerts.append(OIChangeAlert(
                index=current.index_name,
                alert_type="PCR_SHIFT",
                description=(
                    f"PCR {direction}: {prev.pcr:.2f} → {current.pcr:.2f} "
                    f"(+{pcr_diff:.2f}) | Sentiment: {sentiment}"
                ),
                current_snapshot=current,
                prev_snapshot=prev,
            ))

        # ── 3. Max pain move ─────────────────────────────────────────────────
        pain_diff = abs(current.max_pain - prev.max_pain)
        if pain_diff >= MAX_PAIN_MOVE_THRESHOLD:
            direction = "UP" if current.max_pain > prev.max_pain else "DOWN"
            alerts.append(OIChangeAlert(
                index=current.index_name,
                alert_type="MAX_PAIN_MOVE",
                description=(
                    f"Max Pain moved {direction}: {prev.max_pain:.0f} → "
                    f"{current.max_pain:.0f} ({pain_diff:.0f} pts) | "
                    f"Spot: {current.spot_price:.0f}"
                ),
                current_snapshot=current,
                prev_snapshot=prev,
            ))

        # ── 4. CE resistance OI buildup ───────────────────────────────────────
        self._check_oi_buildup(current, prev, alerts)

        # ── 5. OI signal flip ─────────────────────────────────────────────────
        if prev.signal != current.signal and current.signal_strength >= 0.4:
            alerts.append(OIChangeAlert(
                index=current.index_name,
                alert_type="SIGNAL_FLIP",
                description=(
                    f"OI Signal flipped: {prev.signal} → {current.signal} "
                    f"(strength: {current.signal_strength:.0%})"
                ),
                current_snapshot=current,
                prev_snapshot=prev,
            ))

        return alerts

    def _check_oi_buildup(
        self,
        current: OISnapshot,
        prev: OISnapshot,
        alerts: List[OIChangeAlert],
    ):
        """Check if any strike's OI grew by more than CE_OI_BUILDUP_RATIO x in one scan."""
        # Build prev OI maps for comparison
        prev_ce = {r["strike"]: r["oi"] for r in prev.ce_resistance_levels}
        prev_pe = {r["strike"]: r["oi"] for r in prev.pe_support_levels}

        for rec in current.ce_resistance_levels:
            strike = rec["strike"]
            prev_oi = prev_ce.get(strike, 1)
            if prev_oi > 0 and rec["oi"] / prev_oi >= CE_OI_BUILDUP_RATIO:
                alerts.append(OIChangeAlert(
                    index=current.index_name,
                    alert_type="OI_BUILDUP",
                    description=(
                        f"CE {strike} OI doubled: {prev_oi:,} → {rec['oi']:,} | "
                        f"Strong resistance building"
                    ),
                    current_snapshot=current,
                    prev_snapshot=prev,
                ))

        for rec in current.pe_support_levels:
            strike = rec["strike"]
            prev_oi = prev_pe.get(strike, 1)
            if prev_oi > 0 and rec["oi"] / prev_oi >= PE_OI_BUILDUP_RATIO:
                alerts.append(OIChangeAlert(
                    index=current.index_name,
                    alert_type="OI_BUILDUP",
                    description=(
                        f"PE {strike} OI doubled: {prev_oi:,} → {rec['oi']:,} | "
                        f"Strong support building"
                    ),
                    current_snapshot=current,
                    prev_snapshot=prev,
                ))

    def _build_summary(self, snap: OISnapshot) -> str:
        """One-liner OI summary for logging."""
        ce_top = snap.ce_resistance_levels[0]["strike"] if snap.ce_resistance_levels else "N/A"
        pe_top = snap.pe_support_levels[0]["strike"] if snap.pe_support_levels else "N/A"
        return (
            f"{snap.index_name} | Spot {snap.spot_price:.0f} | "
            f"PCR {snap.pcr:.2f} | MaxPain {snap.max_pain:.0f} | "
            f"CE Wall {ce_top} | PE Wall {pe_top} | "
            f"Signal {snap.signal} ({snap.signal_strength:.0%})"
        )
