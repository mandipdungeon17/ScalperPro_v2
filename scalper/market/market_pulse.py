"""
=============================================================================
SCALPER PRO - Market Pulse Orchestrator
=============================================================================
Coordinates all market monitors and dispatches Telegram alerts.

Schedule (during 9 AM - 3:30 PM):
  Every  5 min  → StockMonitor  — ONE digest per index (top movers + spikes)
  Every  2 min  → GlobalMonitor — GIFT Nifty, US/Europe futures, crude
  Every  5 min  → NewsMonitor   — Indian + global RSS, major NSE announcements
  Every  5 min  → OIMonitor     — OI changes, PCR shifts, max pain moves

Key design: StockMonitor returns a digest (not individual stocks).
One Telegram message per index per cycle — no spam.
=============================================================================
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

from scalper.market.news_monitor   import NewsMonitor, NewsItem
from scalper.market.global_monitor import GlobalMonitor, GlobalReport
from scalper.market.stock_monitor  import StockMonitor, StockDigest
from scalper.market.oi_monitor     import OIMonitor, OIReport, OIChangeAlert

logger = logging.getLogger(__name__)


@dataclass
class PulseConfig:
    stock_interval_sec:  int = 300   # 5 min — one digest per index
    global_interval_sec: int = 120   # 2 min
    news_interval_sec:   int = 300   # 5 min
    oi_interval_sec:     int = 300   # 5 min
    trade_indices: List[str] = None  # e.g. ["NIFTY", "BANKNIFTY"]

    def __post_init__(self):
        if self.trade_indices is None:
            self.trade_indices = ["NIFTY", "BANKNIFTY"]


class MarketPulse:
    """
    Market Pulse Orchestrator.
    Call tick() each main scan loop iteration (~30 s).
    Each monitor self-throttles by elapsed time.
    """

    def __init__(self, telegram, config: PulseConfig = None):
        self.telegram = telegram
        self.config   = config or PulseConfig()

        self._news_monitor   = NewsMonitor(min_impact=5)
        self._global_monitor = GlobalMonitor()
        self._stock_monitor  = StockMonitor(track_indices=self._stock_indices())
        self._oi_monitor     = OIMonitor(indices=self.config.trade_indices)

        self._last_stock:  Optional[datetime] = None
        self._last_global: Optional[datetime] = None
        self._last_news:   Optional[datetime] = None
        self._last_oi:     Optional[datetime] = None

        logger.info(
            "[MarketPulse] Initialised | "
            f"Indices: {self.config.trade_indices} | "
            f"Stock digest: {self.config.stock_interval_sec}s | "
            f"Global: {self.config.global_interval_sec}s | "
            f"News: {self.config.news_interval_sec}s | "
            f"OI: {self.config.oi_interval_sec}s"
        )

    # ── Public ───────────────────────────────────────────────────────────────

    def tick(self):
        """Called once per scan loop iteration. Each monitor self-throttles."""
        now = datetime.now()

        if self._due(self._last_stock, self.config.stock_interval_sec):
            self._run_stock_monitor()
            self._last_stock = now

        if self._due(self._last_global, self.config.global_interval_sec):
            self._run_global_monitor()
            self._last_global = now

        if self._due(self._last_news, self.config.news_interval_sec):
            self._run_news_monitor()
            self._last_news = now

        if self._due(self._last_oi, self.config.oi_interval_sec):
            self._run_oi_monitor()
            self._last_oi = now

    def get_oi_snapshot(self, index: str):
        return self._oi_monitor.get_snapshot(index)

    # ── Monitor runners ──────────────────────────────────────────────────────

    def _run_stock_monitor(self):
        """Fetch stock data and send ONE digest per index — no per-stock spam."""
        try:
            digests: List[StockDigest] = self._stock_monitor.check()
            for digest in digests:
                self.telegram.send_stock_digest(digest)
                logger.info(
                    f"[MarketPulse/Stock] {digest.index_name} digest | "
                    f"▲{digest.advance_count} ▼{digest.decline_count} | "
                    f"Top gainer: {digest.top_gainers[0].symbol} "
                    f"({digest.top_gainers[0].change_pct:+.2f}%)"
                    if digest.top_gainers else
                    f"[MarketPulse/Stock] {digest.index_name} digest | "
                    f"▲{digest.advance_count} ▼{digest.decline_count}"
                )
        except Exception as e:
            logger.debug(f"[MarketPulse] StockMonitor error: {e}")

    def _run_global_monitor(self):
        try:
            report: GlobalReport = self._global_monitor.check()

            if report.gift_nifty:
                g = report.gift_nifty
                logger.info(
                    f"[MarketPulse/Global] GIFT Nifty {g.price:.0f} "
                    f"({g.change_pct:+.2f}%) | {g.gap_direction}"
                )

            if report.alerts:
                self.telegram.send_global_pulse(report)
                logger.info(
                    f"[MarketPulse/Global] {len(report.alerts)} alert(s) | "
                    + " | ".join(report.alerts[:3])
                )
        except Exception as e:
            logger.debug(f"[MarketPulse] GlobalMonitor error: {e}")

    def _run_news_monitor(self):
        try:
            items: List[NewsItem] = self._news_monitor.check()
            for item in items:
                self.telegram.send_news_alert(item)
                logger.info(
                    f"[MarketPulse/News] [{item.impact_score}/10] "
                    f"[{item.source}] {item.title[:80]}"
                )
        except Exception as e:
            logger.debug(f"[MarketPulse] NewsMonitor error: {e}")

    def _run_oi_monitor(self):
        try:
            reports: List[OIReport] = self._oi_monitor.check()
            for report in reports:
                logger.info(f"[MarketPulse/OI] {report.summary_line}")
                for alert in report.alerts:
                    self.telegram.send_oi_alert(alert)
                    logger.info(
                        f"[MarketPulse/OI] {alert.alert_type} | {alert.description}"
                    )
        except Exception as e:
            logger.debug(f"[MarketPulse] OIMonitor error: {e}")

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _due(last: Optional[datetime], interval_sec: int) -> bool:
        if last is None:
            return True
        return (datetime.now() - last).total_seconds() >= interval_sec

    def _stock_indices(self) -> List[str]:
        mapping = {
            "NIFTY":      "NIFTY50",
            "BANKNIFTY":  "BANKNIFTY",
            "FINNIFTY":   "FINNIFTY",
            "MIDCPNIFTY": "MIDCPNIFTY",
        }
        result = {"NIFTY50", "BANKNIFTY"}  # always track these two
        for idx in self.config.trade_indices:
            key = mapping.get(idx)
            if key:
                result.add(key)
        return list(result)
