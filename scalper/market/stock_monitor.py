"""
=============================================================================
SCALPER PRO - Stock Volume Monitor
=============================================================================
Tracks top Nifty 50 and BankNifty constituent stocks.

Returns ONE digest per scan cycle — NOT individual per-stock alerts.
The digest shows:
  • Top 5 gainers and top 5 losers by % change
  • Any volume spikes (>3x average) — these are always highlighted

Alert rules:
  • A digest is sent only when at least one stock moves >= DIGEST_MIN_MOVE_PCT
  • Volume spikes (>3x) are always included regardless of price move
  • The same digest is NOT re-sent within DIGEST_THROTTLE_MIN minutes
  • Index rows ("NIFTY 50", "NIFTY BANK" etc.) are filtered out
=============================================================================
"""

import logging
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ── NSE endpoints ─────────────────────────────────────────────────────────────
NSE_STOCK_INDICES_URL = "https://www.nseindia.com/api/equity-stockIndices?index={index}"
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://www.nseindia.com/market-data/live-equity-market",
}

INDEX_MAP = {
    "NIFTY50":    "NIFTY 50",
    "BANKNIFTY":  "NIFTY BANK",
    "FINNIFTY":   "NIFTY FINANCIAL SERVICES",
    "MIDCPNIFTY": "NIFTY MIDCAP SELECT",
}

# Symbols that NSE returns as part of stock list but are actually index rows
_INDEX_ROW_SYMBOLS = {
    "NIFTY 50", "NIFTY BANK", "NIFTY FINANCIAL SERVICES",
    "NIFTY MIDCAP SELECT", "NIFTY50", "NIFTY BANK", "NIFTY",
    "NIFTY MID SELECT",
}

SPIKE_MULTIPLIER    = 3.0    # volume spike threshold
VOLUME_WINDOW       = 20     # rolling periods for average
DIGEST_MIN_MOVE_PCT = 2.5    # minimum move to appear in digest
DIGEST_THROTTLE_MIN = 15     # minutes between digests per index


@dataclass
class StockRow:
    symbol: str
    ltp: float
    change_pct: float
    volume: int
    avg_volume: float
    volume_ratio: float

    @property
    def is_volume_spike(self) -> bool:
        return self.volume_ratio >= SPIKE_MULTIPLIER


@dataclass
class StockDigest:
    """One digest per index per scan cycle — replaces per-stock alerts."""
    index_name: str               # e.g. "NIFTY50"
    advance_count: int
    decline_count: int
    unchanged_count: int
    top_gainers: List[StockRow]   # top 5 by change_pct
    top_losers: List[StockRow]    # bottom 5 by change_pct
    volume_spikes: List[StockRow] # any stock with vol_ratio >= SPIKE_MULTIPLIER
    fetched_at: datetime = field(default_factory=datetime.now)

    @property
    def has_notable(self) -> bool:
        """True if anything worth alerting about."""
        return (
            bool(self.volume_spikes)
            or (self.top_gainers and abs(self.top_gainers[0].change_pct) >= DIGEST_MIN_MOVE_PCT)
            or (self.top_losers  and abs(self.top_losers[0].change_pct)  >= DIGEST_MIN_MOVE_PCT)
        )


class StockMonitor:
    """
    Polls NSE live equity data for Nifty50 + BankNifty stocks.
    Returns one StockDigest per index — call check() every 5 minutes.
    """

    def __init__(self, track_indices: List[str] = None):
        self.track_indices = track_indices or ["NIFTY50", "BANKNIFTY"]
        self._volume_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=VOLUME_WINDOW)
        )
        self._last_digest_time: Dict[str, datetime] = {}
        self._nse_session: Optional[requests.Session] = None

    # ── Public ───────────────────────────────────────────────────────────────

    def check(self) -> List[StockDigest]:
        """
        Fetch current stock data for all tracked indices.
        Returns a list of StockDigest objects (one per index, throttled).
        Call every 5 minutes from the orchestrator.
        """
        digests: List[StockDigest] = []

        for idx_key in self.track_indices:
            nse_name = INDEX_MAP.get(idx_key)
            if not nse_name:
                continue

            # Throttle: skip if digest was sent recently for this index
            last = self._last_digest_time.get(idx_key)
            if last and (datetime.now() - last) < timedelta(minutes=DIGEST_THROTTLE_MIN):
                continue

            try:
                rows = self._fetch_index_stocks(idx_key, nse_name)
                if not rows:
                    continue

                digest = self._build_digest(idx_key, rows)
                if digest.has_notable:
                    self._last_digest_time[idx_key] = datetime.now()
                    digests.append(digest)

            except Exception as e:
                logger.debug(f"[StockMonitor] Error fetching {idx_key}: {e}")

        return digests

    # ── Private ──────────────────────────────────────────────────────────────

    def _fetch_index_stocks(self, idx_key: str, nse_name: str) -> List[StockRow]:
        session = self._get_nse_session()
        url = NSE_STOCK_INDICES_URL.format(index=nse_name)
        resp = session.get(url, headers=NSE_HEADERS, timeout=12)

        if resp.status_code != 200:
            logger.debug(f"[StockMonitor] NSE {resp.status_code} for {nse_name}")
            return []

        rows: List[StockRow] = []
        for raw in resp.json().get("data", []):
            symbol = (raw.get("symbol") or "").strip()

            # Skip index summary rows that NSE includes in the response
            if not symbol or symbol.upper() in _INDEX_ROW_SYMBOLS or " " in symbol:
                continue

            try:
                ltp        = float(raw.get("lastPrice") or raw.get("ltp") or 0)
                prev       = float(raw.get("previousClose") or raw.get("prevClose") or 0)
                volume     = int(raw.get("totalTradedVolume") or raw.get("tradedQuantity") or 0)
                change_pct = float(raw.get("pChange") or raw.get("netChange") or 0)

                if ltp <= 0:
                    continue
                if change_pct == 0 and prev > 0:
                    change_pct = round((ltp - prev) / prev * 100, 2)

                # Update rolling volume history
                self._volume_history[symbol].append(volume)
                avg_vol = self._get_avg_volume(symbol)
                vol_ratio = round(volume / avg_vol, 2) if avg_vol > 0 else 0.0

                rows.append(StockRow(
                    symbol=symbol,
                    ltp=round(ltp, 2),
                    change_pct=round(change_pct, 2),
                    volume=volume,
                    avg_volume=avg_vol,
                    volume_ratio=vol_ratio,
                ))
            except Exception:
                continue

        return rows

    def _build_digest(self, idx_key: str, rows: List[StockRow]) -> StockDigest:
        advance   = sum(1 for r in rows if r.change_pct > 0)
        decline   = sum(1 for r in rows if r.change_pct < 0)
        unchanged = len(rows) - advance - decline

        sorted_rows = sorted(rows, key=lambda r: r.change_pct, reverse=True)
        top_gainers = [r for r in sorted_rows if r.change_pct >= DIGEST_MIN_MOVE_PCT][:5]
        top_losers  = [r for r in reversed(sorted_rows) if r.change_pct <= -DIGEST_MIN_MOVE_PCT][:5]
        vol_spikes  = [r for r in rows if r.is_volume_spike]

        return StockDigest(
            index_name=idx_key,
            advance_count=advance,
            decline_count=decline,
            unchanged_count=unchanged,
            top_gainers=top_gainers,
            top_losers=top_losers,
            volume_spikes=vol_spikes,
        )

    def _get_avg_volume(self, symbol: str) -> float:
        hist = self._volume_history[symbol]
        past = list(hist)[:-1]
        return sum(past) / len(past) if len(past) >= 3 else 0.0

    def _get_nse_session(self) -> requests.Session:
        if self._nse_session is None:
            self._nse_session = requests.Session()
            try:
                self._nse_session.get("https://www.nseindia.com",
                                      headers=NSE_HEADERS, timeout=8)
            except Exception:
                pass
        return self._nse_session
