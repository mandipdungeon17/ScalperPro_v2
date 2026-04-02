"""
=============================================================================
SCALPER PRO - Global Market Monitor
=============================================================================
Tracks global indices that influence Indian markets:

  GIFT Nifty   → NSE allIndices API (live pre-market and morning data)
  US Futures   → Dow (YM=F), S&P 500 (ES=F), Nasdaq (NQ=F)
  Europe       → FTSE 100 (^FTSE), DAX (^GDAXI)
  Crude Oil    → Brent (BZ=F), WTI (CL=F)
  Gold         → GC=F
  USD/INR      → USDINR=X

Data source: Yahoo Finance raw JSON quote API (NOT yfinance library).
  - Used during ALL hours (pre-market and live 9AM-3:30PM).
  - These are global futures tickers — Yahoo is the right source.
  - Dhan API only has NSE/BSE instruments, not global futures.

GIFT Nifty is fetched from NSE allIndices endpoint — authoritative source.
=============================================================================
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


# ── Yahoo Finance raw quote URL ───────────────────────────────────────────────
# Returns JSON without any yfinance library dependency
YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1m&range=1d"
YAHOO_SUMMARY_URL = "https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbols}"

NSE_ALL_INDICES_URL = "https://www.nseindia.com/api/allIndices"
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://www.nseindia.com",
}

YAHOO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}

# ── Tracker config ────────────────────────────────────────────────────────────
GLOBAL_TICKERS = {
    # US Futures
    "YM=F":   {"name": "Dow Futures",     "group": "US_FUTURES",    "alert_pct": 0.5},
    "ES=F":   {"name": "S&P 500 Futures", "group": "US_FUTURES",    "alert_pct": 0.5},
    "NQ=F":   {"name": "Nasdaq Futures",  "group": "US_FUTURES",    "alert_pct": 0.7},
    # Europe
    "^FTSE":  {"name": "FTSE 100",        "group": "EUROPE",        "alert_pct": 0.6},
    "^GDAXI": {"name": "DAX",             "group": "EUROPE",        "alert_pct": 0.6},
    # Commodities
    "BZ=F":   {"name": "Brent Crude",     "group": "COMMODITY",     "alert_pct": 1.0},
    "CL=F":   {"name": "WTI Crude",       "group": "COMMODITY",     "alert_pct": 1.0},
    "GC=F":   {"name": "Gold Futures",    "group": "COMMODITY",     "alert_pct": 0.8},
    # FX
    "USDINR=X": {"name": "USD/INR",       "group": "FOREX",         "alert_pct": 0.3},
}


@dataclass
class GlobalSnapshot:
    symbol: str
    name: str
    group: str
    price: float
    change_pct: float          # % change vs prev close
    prev_close: float
    fetched_at: datetime = field(default_factory=datetime.now)

    @property
    def direction(self) -> str:
        return "UP" if self.change_pct >= 0 else "DOWN"

    @property
    def is_significant(self) -> bool:
        threshold = GLOBAL_TICKERS.get(self.symbol, {}).get("alert_pct", 0.5)
        return abs(self.change_pct) >= threshold


@dataclass
class GiftNiftySnapshot:
    price: float
    change: float
    change_pct: float
    fetched_at: datetime = field(default_factory=datetime.now)

    @property
    def gap_direction(self) -> str:
        """Gap-up or gap-down signal for Nifty."""
        if self.change_pct > 0.3:
            return "GAP_UP"
        elif self.change_pct < -0.3:
            return "GAP_DOWN"
        return "FLAT"


@dataclass
class GlobalReport:
    gift_nifty: Optional[GiftNiftySnapshot]
    snapshots: List[GlobalSnapshot]
    alerts: List[str]          # human-readable alert strings
    nifty_implied_open: Optional[float]   # GIFT + offset
    fetched_at: datetime = field(default_factory=datetime.now)


class GlobalMonitor:
    """
    Fetches GIFT Nifty from NSE and US/Europe futures from Yahoo Finance raw API.
    Call check() every 2 minutes from the orchestrator.
    """

    def __init__(self):
        self._prev_snapshots: Dict[str, GlobalSnapshot] = {}
        self._nse_session: Optional[requests.Session] = None

    # ── Public ───────────────────────────────────────────────────────────────

    def check(self) -> GlobalReport:
        """
        Fetch all global trackers and return a GlobalReport.
        Alerts generated when any market moves beyond its threshold.
        """
        gift = self._fetch_gift_nifty()
        snapshots = self._fetch_yahoo_batch()
        alerts = self._build_alerts(gift, snapshots)

        # Estimate implied Nifty open from GIFT
        implied_open = None
        if gift and gift.price > 0:
            # GIFT Nifty trades at ~Nifty futures price; subtract ~75 pt basis
            implied_open = round(gift.price - 75, 0)

        report = GlobalReport(
            gift_nifty=gift,
            snapshots=snapshots,
            alerts=alerts,
            nifty_implied_open=implied_open,
        )

        self._prev_snapshots = {s.symbol: s for s in snapshots}
        return report

    # ── GIFT Nifty ───────────────────────────────────────────────────────────

    def _fetch_gift_nifty(self) -> Optional[GiftNiftySnapshot]:
        """Fetch GIFT Nifty from NSE allIndices endpoint."""
        try:
            session = self._get_nse_session()
            resp = session.get(NSE_ALL_INDICES_URL, headers=NSE_HEADERS, timeout=10)
            if resp.status_code != 200:
                return None

            data = resp.json()
            indices = data.get("data", [])
            for entry in indices:
                name = (entry.get("indexSymbol") or entry.get("index") or "").upper()
                if "GIFT" in name or "SGX" in name:
                    price = float(entry.get("last") or entry.get("lastPrice") or 0)
                    change = float(entry.get("variation") or 0)
                    pct = float(entry.get("percentChange") or 0)
                    if price > 0:
                        return GiftNiftySnapshot(
                            price=price, change=change, change_pct=round(pct, 2)
                        )

            # GIFT Nifty sometimes listed differently — scan for "NIFTY 50" futures proxy
            for entry in indices:
                name = (entry.get("indexSymbol") or "").upper()
                if "NIFTY 50" in name and "GIFT" in str(entry):
                    price = float(entry.get("last") or 0)
                    if price > 0:
                        return GiftNiftySnapshot(price=price, change=0, change_pct=0)

        except Exception as e:
            logger.debug(f"[GlobalMonitor] GIFT Nifty fetch error: {e}")

        return None

    # ── Yahoo Finance raw batch ───────────────────────────────────────────────

    def _fetch_yahoo_batch(self) -> List[GlobalSnapshot]:
        """Fetch all global tickers in one Yahoo Finance batch call."""
        symbols = ",".join(GLOBAL_TICKERS.keys())
        url = YAHOO_SUMMARY_URL.format(symbols=symbols)

        try:
            resp = requests.get(url, headers=YAHOO_HEADERS, timeout=12)
            if resp.status_code != 200:
                return []

            data = resp.json()
            results = data.get("quoteResponse", {}).get("result", [])

            snapshots: List[GlobalSnapshot] = []
            for q in results:
                symbol = q.get("symbol", "")
                cfg = GLOBAL_TICKERS.get(symbol)
                if not cfg:
                    continue

                price = float(q.get("regularMarketPrice") or
                              q.get("postMarketPrice") or
                              q.get("preMarketPrice") or 0)
                prev  = float(q.get("regularMarketPreviousClose") or 0)

                if price <= 0 or prev <= 0:
                    continue

                change_pct = round((price - prev) / prev * 100, 2)
                snapshots.append(GlobalSnapshot(
                    symbol=symbol,
                    name=cfg["name"],
                    group=cfg["group"],
                    price=round(price, 2),
                    change_pct=change_pct,
                    prev_close=round(prev, 2),
                ))

            return snapshots

        except Exception as e:
            logger.debug(f"[GlobalMonitor] Yahoo batch fetch error: {e}")
            return []

    # ── Alert logic ──────────────────────────────────────────────────────────

    def _build_alerts(
        self,
        gift: Optional[GiftNiftySnapshot],
        snapshots: List[GlobalSnapshot],
    ) -> List[str]:
        alerts: List[str] = []

        if gift:
            gap = gift.gap_direction
            if gap != "FLAT":
                direction_word = "GAP UP" if gap == "GAP_UP" else "GAP DOWN"
                alerts.append(
                    f"GIFT NIFTY {direction_word}: {gift.price:.0f} "
                    f"({gift.change_pct:+.2f}%)"
                )

        for snap in snapshots:
            if snap.is_significant:
                prev = self._prev_snapshots.get(snap.symbol)
                # Only alert if movement increased since last check
                if prev and abs(snap.change_pct) <= abs(prev.change_pct) + 0.1:
                    continue
                alerts.append(
                    f"{snap.name}: {snap.price:,.2f} ({snap.change_pct:+.2f}%)"
                )

        return alerts

    # ── NSE session ──────────────────────────────────────────────────────────

    def _get_nse_session(self) -> requests.Session:
        if self._nse_session is None:
            self._nse_session = requests.Session()
            try:
                self._nse_session.get(
                    "https://www.nseindia.com",
                    headers=NSE_HEADERS, timeout=8
                )
            except Exception:
                pass
        return self._nse_session
