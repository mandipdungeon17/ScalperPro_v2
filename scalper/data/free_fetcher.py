"""
=============================================================================
SCALPER PRO v2 — Free Historical Data Fetcher
=============================================================================
Fetches historical data from FREE sources for backtesting:
  1. Yahoo Finance (yfinance) → 5 years of index OHLCV data
  2. NSE website → Option chain, India VIX
  3. Dhan Free Trading API → Live candles, order execution

Install: pip install yfinance

Yahoo Finance symbols for Indian indices:
  NIFTY     → ^NSEI
  BANKNIFTY → ^NSEBANK
  FINNIFTY  → NIFTY_FIN_SERVICE.NS
  SENSEX    → ^BSESN
=============================================================================
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

# Yahoo Finance symbol mapping
YAHOO_SYMBOLS = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
    "MIDCPNIFTY": "NIFTY_MID_SELECT.NS",
    "SENSEX": "^BSESN",
}


class FreeDataFetcher:
    """
    Fetches historical data from free sources.

    Priority:
      1. Yahoo Finance (yfinance) — 5 years daily + intraday
      2. Cached CSV files — previously downloaded data
      3. Sample data generator — for offline testing
    """

    def __init__(self, cache_dir: str = "./data/historical"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._yf_available = self._check_yfinance()

    @staticmethod
    def _check_yfinance() -> bool:
        try:
            import yfinance
            return True
        except ImportError:
            logger.warning(
                "yfinance not installed. Run: pip install yfinance\n"
                "Falling back to sample data."
            )
            return False

    # ── Yahoo Finance: Daily Data (up to 5 years) ─────────────────

    def fetch_daily(
        self,
        index: str,
        period: str = "1y",     # "6mo", "1y", "2y", "5y", "max"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch daily OHLCV from Yahoo Finance.
        Returns: datetime, open, high, low, close, volume
        """
        if not self._yf_available:
            return self._load_cache(index, "daily") or self._generate_sample_daily(index)

        symbol = YAHOO_SYMBOLS.get(index)
        if not symbol:
            logger.error(f"No Yahoo symbol for {index}")
            return None

        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval="1d")

            if df.empty:
                logger.warning(f"No Yahoo data for {symbol}")
                return self._load_cache(index, "daily") or self._generate_sample_daily(index)

            df = df.reset_index()
            df = df.rename(columns={
                "Date": "datetime", "Open": "open", "High": "high",
                "Low": "low", "Close": "close", "Volume": "volume"
            })
            df = df[["datetime", "open", "high", "low", "close", "volume"]]
            df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)

            # Cache to CSV
            self._save_cache(df, index, "daily")
            logger.info(f"Fetched {len(df)} daily bars for {index} from Yahoo")
            return df

        except Exception as e:
            logger.error(f"Yahoo Finance error for {index}: {e}")
            return self._load_cache(index, "daily") or self._generate_sample_daily(index)

    # ── Yahoo Finance: Intraday Data ──────────────────────────────

    def fetch_intraday(
        self,
        index: str,
        interval: str = "5m",   # "1m", "5m", "15m", "1h"
        period: str = "5d",     # Max "60d" for 1m, "60d" for 5m/15m
    ) -> Optional[pd.DataFrame]:
        """
        Fetch intraday OHLCV from Yahoo Finance.
        Note: Yahoo limits intraday history:
          - 1m: last 7 days
          - 5m/15m: last 60 days
          - 1h: last 730 days
        """
        if not self._yf_available:
            return self._load_cache(index, interval) or self._generate_sample_intraday(index, interval)

        symbol = YAHOO_SYMBOLS.get(index)
        if not symbol:
            return None

        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                return self._load_cache(index, interval) or self._generate_sample_intraday(index, interval)

            df = df.reset_index()
            date_col = "Datetime" if "Datetime" in df.columns else "Date"
            df = df.rename(columns={
                date_col: "datetime", "Open": "open", "High": "high",
                "Low": "low", "Close": "close", "Volume": "volume"
            })
            df = df[["datetime", "open", "high", "low", "close", "volume"]]
            df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)

            self._save_cache(df, index, interval.replace("m", "min").replace("h", "hr"))
            logger.info(f"Fetched {len(df)} {interval} bars for {index}")
            return df

        except Exception as e:
            logger.error(f"Yahoo intraday error: {e}")
            return self._load_cache(index, interval) or self._generate_sample_intraday(index, interval)

    # ── Convenience methods ───────────────────────────────────────

    def fetch_all_timeframes(self, index: str) -> Dict[str, pd.DataFrame]:
        """Fetch all timeframes needed for the 4-layer system."""
        result = {}

        logger.info(f"Fetching all timeframes for {index}...")

        daily = self.fetch_daily(index, period="1y")
        if daily is not None:
            result["daily"] = daily

        hourly = self.fetch_intraday(index, interval="1h", period="30d")
        if hourly is not None:
            result["hourly"] = hourly

        fifteen = self.fetch_intraday(index, interval="15m", period="30d")
        if fifteen is not None:
            result["15min"] = fifteen

        five = self.fetch_intraday(index, interval="5m", period="30d")
        if five is not None:
            result["5min"] = five

        one = self.fetch_intraday(index, interval="1m", period="5d")
        if one is not None:
            result["1min"] = one

        return result

    # ── Cache management ──────────────────────────────────────────

    def _save_cache(self, df: pd.DataFrame, index: str, tf: str):
        path = os.path.join(self.cache_dir, f"{index}_{tf}.csv")
        df.to_csv(path, index=False)

    def _load_cache(self, index: str, tf: str) -> Optional[pd.DataFrame]:
        path = os.path.join(self.cache_dir, f"{index}_{tf}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["datetime"] = pd.to_datetime(df["datetime"])
            logger.info(f"Loaded cached {tf} data for {index}: {len(df)} bars")
            return df
        return None

    # ── Sample data fallback ──────────────────────────────────────

    def _generate_sample_daily(self, index: str) -> pd.DataFrame:
        from scalper.data.fetcher import DataFetcher
        bases = {"NIFTY": 23500, "BANKNIFTY": 51000, "FINNIFTY": 23000,
                 "MIDCPNIFTY": 12500, "SENSEX": 77000}
        return DataFetcher.generate_sample_daily(index, 400, bases.get(index, 23500))

    def _generate_sample_intraday(self, index: str, interval: str) -> pd.DataFrame:
        from scalper.data.fetcher import DataFetcher
        bases = {"NIFTY": 23500, "BANKNIFTY": 51000, "FINNIFTY": 23000,
                 "MIDCPNIFTY": 12500, "SENSEX": 77000}
        mins = {"1m": 1, "5m": 5, "15m": 15, "1h": 60,
                "1min": 1, "5min": 5, "15min": 15, "1hr": 60}
        interval_min = mins.get(interval, 5)
        days = 7 if interval_min <= 1 else 30
        return DataFetcher.generate_sample_data(
            index, days, interval_min, bases.get(index, 23500)
        )
