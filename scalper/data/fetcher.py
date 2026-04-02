"""
=============================================================================
SCALPER PRO - Data Fetcher
=============================================================================
Handles data retrieval from:
- Dhan API (historical candles, live quotes)
- NSE (VIX, option chain)
- Alternative free sources for historical data
=============================================================================
"""

import io
import pandas as pd
import numpy as np
import requests
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import time
import json
import os

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches market data from multiple sources."""

    def __init__(self):
        from scalper.config.settings import (
            DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN, DATA_DIR
        )
        self.dhan_client_id = DHAN_CLIENT_ID
        self.dhan_token = DHAN_ACCESS_TOKEN
        self.data_dir = DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)

        self.dhan_headers = {
            "Content-Type": "application/json",
            "access-token": self.dhan_token,
            "client-id": self.dhan_client_id,   # required by Dhan v2
        }

        self.nse_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com",
        }
        self._nse_session = None

    # ── Dhan Historical Data ────────────────────────────────────────────────

    def fetch_dhan_historical(
        self,
        security_id: str,
        exchange_segment: str,
        instrument: str,          # "INDEX", "EQUITY", "FUTIDX", "OPTIDX"
        interval: str = "5",      # "1", "5", "15", "25", "60", "D"
        from_date: str = None,    # "YYYY-MM-DD"
        to_date: str = None,      # "YYYY-MM-DD"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical candle data from Dhan API.
        Returns DataFrame with: datetime, open, high, low, close, volume
        """
        if to_date is None:
            to_date = datetime.now().strftime("%Y-%m-%d")
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        # Dhan exchange segment codes
        seg_map = {
            "IDX_I":   "IDX_I",    # Index historical data (NIFTY, BANKNIFTY spot)
            "NSE_FNO": "NSE_FNO",  # NSE Futures & Options
            "BSE_FNO": "BSE_FNO",  # BSE Futures & Options
            "NSE_EQ":  "NSE_EQ",   # NSE Equity
            "NSE":     "NSE_EQ",
        }
        segment = seg_map.get(exchange_segment, exchange_segment)

        # Dhan uses two separate endpoints:
        #   /charts/historical → daily candles only (interval must be "D")
        #   /charts/intraday   → intraday candles (1, 5, 15, 25, 60 min)
        intraday_intervals = {"1", "5", "15", "25", "60"}
        if interval in intraday_intervals:
            url = "https://api.dhan.co/v2/charts/intraday"
            payload = {
                "securityId": security_id,
                "exchangeSegment": segment,
                "instrument": instrument,
                "interval": interval,
                "oi": False,
                "fromDate": from_date,
                "toDate": to_date,
            }
        else:
            url = "https://api.dhan.co/v2/charts/historical"
            payload = {
                "securityId": security_id,
                "exchangeSegment": segment,
                "instrument": instrument,
                "expiryCode": 0,
                "fromDate": from_date,
                "toDate": to_date,
            }

        try:
            resp = requests.post(
                url, headers=self.dhan_headers, json=payload, timeout=30
            )
            if resp.status_code != 200:
                logger.error(f"Dhan API error: {resp.status_code} - {resp.text}")
                return None

            data = resp.json()

            # Dhan returns: {"open": [...], "high": [...], "low": [...],
            #                 "close": [...], "volume": [...], "timestamp": [...]}
            if not data or "open" not in data:
                logger.warning(f"No data returned for {security_id}")
                return None

            df = pd.DataFrame({
                "datetime": pd.to_datetime(data["timestamp"], unit="s")
                            .tz_localize("UTC").tz_convert("Asia/Kolkata"),
                "open": data["open"],
                "high": data["high"],
                "low": data["low"],
                "close": data["close"],
                "volume": data["volume"],
            })

            df["datetime"] = df["datetime"].dt.tz_localize(None)
            df = df.sort_values("datetime").reset_index(drop=True)
            return df

        except Exception as e:
            logger.error(f"Error fetching Dhan data: {e}")
            return None

    def fetch_index_data(
        self,
        index: str,
        interval: str = "5",
        days_back: int = 365,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch index historical OHLCV from Dhan.
        Uses IDX_I exchange segment — required for spot index data.
        Routes to /charts/intraday for minute intervals, /charts/historical for daily.
        """
        from scalper.config.settings import INDEX_CONFIGS

        config = INDEX_CONFIGS.get(index)
        if not config:
            logger.error(f"Unknown index: {index}")
            return None

        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")

        # Index OHLCV must use IDX_I segment — NOT NSE_FNO
        # NSE_FNO is only for futures/options contracts
        return self.fetch_dhan_historical(
            security_id=config.dhan_security_id,
            exchange_segment="IDX_I",
            instrument="INDEX",
            interval=interval,
            from_date=from_date,
            to_date=to_date,
        )

    def fetch_daily_data(self, index: str, days_back: int = 400) -> Optional[pd.DataFrame]:
        """Fetch daily candle data for swing level calculation."""
        return self.fetch_index_data(index, interval="D", days_back=days_back)

    def fetch_nse_spot_price(self, index: str) -> Optional[float]:
        """
        Fetch the live spot price directly from NSE allIndices API.
        No authentication required — works even when Dhan API is down.
        Used to get the real current price during live trading hours.
        """
        nse_name_map = {
            "NIFTY":      "NIFTY 50",
            "BANKNIFTY":  "NIFTY BANK",
            "FINNIFTY":   "NIFTY FINANCIAL SERVICES",
            "MIDCPNIFTY": "NIFTY MIDCAP SELECT",
            "SENSEX":     "S&P BSE SENSEX",
        }
        target = nse_name_map.get(index)
        if not target:
            return None
        try:
            session = self._get_nse_session()
            resp = session.get(
                "https://www.nseindia.com/api/allIndices",
                headers=self.nse_headers, timeout=8
            )
            if resp.status_code != 200:
                return None
            for entry in resp.json().get("data", []):
                name = entry.get("index", "") or entry.get("indexSymbol", "")
                if name.upper() == target.upper():
                    price = entry.get("last") or entry.get("lastPrice")
                    return float(price) if price else None
        except Exception as e:
            logger.debug(f"NSE spot fetch failed for {index}: {e}")
        return None

    # ── Live Market Data ────────────────────────────────────────────────────

    def fetch_live_quote(self, security_id: str,
                         exchange_segment: str) -> Optional[Dict]:
        """Fetch live quote from Dhan."""
        url = "https://api.dhan.co/v2/marketfeed/ltp"
        payload = {
            "NSE_FNO": [int(security_id)]
            if exchange_segment == "NSE_FNO"
            else [],
            "BSE_FNO": [int(security_id)]
            if exchange_segment == "BSE_FNO"
            else [],
        }

        try:
            resp = requests.post(
                url, headers=self.dhan_headers, json=payload, timeout=5
            )
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            logger.error(f"Error fetching live quote: {e}")
            return None

    # ── NSE Data ────────────────────────────────────────────────────────────

    def _get_nse_session(self) -> requests.Session:
        if self._nse_session is None:
            self._nse_session = requests.Session()
            try:
                self._nse_session.get(
                    "https://www.nseindia.com",
                    headers=self.nse_headers, timeout=10
                )
            except Exception:
                pass
        return self._nse_session

    def fetch_india_vix(self) -> Optional[float]:
        """Fetch current India VIX value."""
        try:
            session = self._get_nse_session()
            resp = session.get(
                "https://www.nseindia.com/api/allIndices",
                headers=self.nse_headers, timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                for idx in data.get("data", []):
                    if idx.get("index") == "INDIA VIX":
                        return float(idx.get("last", 0))
            return None
        except Exception as e:
            logger.error(f"Error fetching VIX: {e}")
            return None

    def fetch_fii_dii_data(self) -> Optional[Dict]:
        """Fetch FII/DII activity data."""
        try:
            session = self._get_nse_session()
            resp = session.get(
                "https://www.nseindia.com/api/fiidiiActivity/activity",
                headers=self.nse_headers, timeout=10
            )
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            logger.error(f"Error fetching FII/DII: {e}")
            return None

    # ── Instrument Master (Option Security ID Resolution) ───────────────────

    def fetch_instrument_master(self) -> Optional[pd.DataFrame]:
        """
        Download Dhan's compact instrument master CSV and cache it locally.
        Contains security_id for every tradeable instrument including options.
        Cache is refreshed every 6 hours.
        """
        cache_file = os.path.join(self.data_dir, "instrument_master.csv")

        # Use cached file if fresh (< 6 hours old)
        if os.path.exists(cache_file):
            age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600
            if age_hours < 6:
                try:
                    return pd.read_csv(cache_file, low_memory=False)
                except Exception:
                    pass

        url = "https://images.dhan.co/api-data/api-scrip-master.csv"
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200:
                df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
                df.to_csv(cache_file, index=False)
                logger.info(f"Instrument master downloaded: {len(df)} rows")
                return df
            else:
                logger.error(f"Instrument master download failed: {resp.status_code}")
                # Return stale cache if available
                if os.path.exists(cache_file):
                    return pd.read_csv(cache_file, low_memory=False)
                return None
        except Exception as e:
            logger.error(f"Error fetching instrument master: {e}")
            if os.path.exists(cache_file):
                return pd.read_csv(cache_file, low_memory=False)
            return None

    def resolve_option_security_id(
        self,
        index: str,
        strike: int,
        option_type: str,       # "CE" or "PE"
        expiry_date: str = None,  # "YYYY-MM-DD"; defaults to nearest weekly expiry
        spot_price: float = None,  # used for ATM fallback
    ) -> Optional[str]:
        """
        Resolve Dhan security_id for an option contract using the instrument master.
        Falls back to nearest available strike if exact strike not found.
        Returns the security_id string, or None if not found.
        """
        master = self.fetch_instrument_master()
        if master is None:
            logger.error("Instrument master unavailable — cannot resolve option security_id")
            return None

        # Log columns on first call for debugging
        if not hasattr(self, "_master_cols_logged"):
            logger.info(f"Instrument master columns: {list(master.columns)[:20]}")
            self._master_cols_logged = True

        # Dhan instrument master uses SM_SYMBOL_NAME for the underlying
        symbol_map = {
            "NIFTY": "NIFTY",
            "BANKNIFTY": "BANKNIFTY",
            "FINNIFTY": "FINNIFTY",
            "MIDCPNIFTY": "MIDCPNIFTY",
            "SENSEX": "SENSEX",
        }
        sym = symbol_map.get(index, index)

        # Try to find the exact symbol column (handle alternate column names)
        sym_col = None
        for col in ["SM_SYMBOL_NAME", "SEM_TRADING_SYMBOL", "SEM_CUSTOM_SYMBOL"]:
            if col in master.columns:
                sym_col = col
                break
        if sym_col is None:
            logger.error(f"Cannot find symbol column in master. Columns: {list(master.columns)}")
            return None

        # Filter to matching index options (exact strike first)
        try:
            base_filter = (
                (master["SEM_INSTRUMENT_NAME"] == "OPTIDX") &
                (master[sym_col].str.upper().str.startswith(sym)) &
                (master["SEM_OPTION_TYPE"] == option_type)
            )
            all_options = master[base_filter].copy()
        except KeyError as e:
            logger.error(f"Instrument master column missing: {e}. Columns: {list(master.columns)}")
            return None

        if all_options.empty:
            logger.warning(f"No options found for {sym} {option_type} at all in master")
            return None

        all_options["SEM_STRIKE_PRICE"] = pd.to_numeric(
            all_options["SEM_STRIKE_PRICE"], errors="coerce"
        )
        all_options["SEM_EXPIRY_DATE"] = pd.to_datetime(
            all_options["SEM_EXPIRY_DATE"], errors="coerce"
        )
        now = datetime.now()
        cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
        active = all_options[all_options["SEM_EXPIRY_DATE"] >= cutoff].copy()
        if active.empty:
            logger.warning(f"No active expiry found for {sym} {option_type}")
            return None

        # Nearest weekly expiry first
        nearest_expiry = active["SEM_EXPIRY_DATE"].min()
        expiry_pool = active[active["SEM_EXPIRY_DATE"] == nearest_expiry]

        # Try exact strike
        exact = expiry_pool[expiry_pool["SEM_STRIKE_PRICE"] == float(strike)]
        if not exact.empty:
            return str(int(exact.iloc[0]["SEM_SMST_SECURITY_ID"]))

        # Fallback: nearest available strike to requested strike
        logger.warning(
            f"No instrument for {sym} {strike} {option_type} "
            f"on expiry {nearest_expiry.date()} — trying nearest available strike"
        )
        available_strikes = sorted(expiry_pool["SEM_STRIKE_PRICE"].dropna().unique())
        if not available_strikes:
            return None

        nearest_strike = min(available_strikes, key=lambda s: abs(s - float(strike)))
        logger.info(
            f"[FALLBACK] Using {sym} {nearest_strike:.0f} {option_type} "
            f"(requested {strike}, nearest available)"
        )
        fallback = expiry_pool[expiry_pool["SEM_STRIKE_PRICE"] == nearest_strike]
        if fallback.empty:
            return None
        return str(int(fallback.iloc[0]["SEM_SMST_SECURITY_ID"]))

    # ── Data Caching ────────────────────────────────────────────────────────

    def save_data(self, df: pd.DataFrame, index: str,
                  interval: str, label: str = ""):
        """Cache data to disk."""
        filename = f"{index}_{interval}{'_' + label if label else ''}.csv"
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} rows to {filepath}")

    def load_data(self, index: str, interval: str,
                  label: str = "") -> Optional[pd.DataFrame]:
        """Load cached data from disk."""
        filename = f"{index}_{interval}{'_' + label if label else ''}.csv"
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
            return df
        return None

    # ── Sample Data Generator (for testing) ─────────────────────────────────

    @staticmethod
    def generate_sample_data(
        index: str = "NIFTY",
        days: int = 180,
        interval_minutes: int = 5,
        base_price: float = 23500.0,
    ) -> pd.DataFrame:
        """
        Generate realistic-looking sample OHLCV data for testing.
        Uses geometric Brownian motion with mean reversion and
        realistic intraday patterns (gap opens, volume profiles).
        """
        from scalper.config.settings import INDEX_CONFIGS

        config = INDEX_CONFIGS.get(index)
        bars_per_day = int(375 / interval_minutes)  # 9:15 to 15:30 = 375 mins
        total_bars = days * bars_per_day

        # Parameters
        daily_vol = 0.012  # ~1.2% daily volatility
        bar_vol = daily_vol / np.sqrt(bars_per_day)
        mean_reversion = 0.001
        trend = 0.0001  # Slight upward bias

        prices = np.zeros(total_bars)
        volumes = np.zeros(total_bars)
        datetimes = []

        prices[0] = base_price
        current_date = datetime.now() - timedelta(days=days)

        bar_count = 0
        for day in range(days):
            # Skip weekends
            while current_date.weekday() >= 5:
                current_date += timedelta(days=1)

            # Gap open (random gap from previous close)
            if bar_count > 0:
                gap = np.random.normal(0, daily_vol * 0.3) * prices[bar_count - 1]
                prices[bar_count] = prices[bar_count - 1] + gap

            for bar_in_day in range(bars_per_day):
                if bar_count >= total_bars:
                    break

                # Time
                minutes = 9 * 60 + 15 + bar_in_day * interval_minutes
                hour = minutes // 60
                minute = minutes % 60
                dt = current_date.replace(hour=hour, minute=minute, second=0)
                datetimes.append(dt)

                # Price movement
                if bar_count > 0:
                    noise = np.random.normal(trend, bar_vol) * prices[bar_count - 1]
                    # Mean reversion
                    deviation = (prices[bar_count - 1] - base_price) / base_price
                    reversion = -mean_reversion * deviation * prices[bar_count - 1]
                    prices[bar_count] = prices[bar_count - 1] + noise + reversion

                # Volume profile (U-shaped: high at open, low mid-day, high at close)
                time_factor = bar_in_day / bars_per_day
                vol_base = 1000 * config.lot_size if config else 75000
                if time_factor < 0.15:  # First 15% of day
                    vol_mult = 3.0 - time_factor * 10
                elif time_factor > 0.85:  # Last 15%
                    vol_mult = 1.0 + (time_factor - 0.85) * 15
                else:
                    vol_mult = 0.8 + np.random.exponential(0.3)

                volumes[bar_count] = int(vol_base * vol_mult * (0.5 + np.random.random()))
                bar_count += 1

            current_date += timedelta(days=1)

        # Trim to actual bars generated
        prices = prices[:bar_count]
        volumes = volumes[:bar_count]
        datetimes = datetimes[:bar_count]

        # Generate OHLC from close prices
        df = pd.DataFrame({"datetime": datetimes, "close": prices, "volume": volumes})
        bar_range = prices * bar_vol * 0.5

        df["high"] = df["close"] + np.abs(np.random.normal(0, 1, bar_count)) * bar_range
        df["low"] = df["close"] - np.abs(np.random.normal(0, 1, bar_count)) * bar_range
        df["open"] = df["close"] + np.random.normal(0, 0.5, bar_count) * bar_range

        # Ensure OHLC consistency
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)

        # Round to tick size
        tick = config.tick_size if config else 0.05
        for col in ["open", "high", "low", "close"]:
            df[col] = (df[col] / tick).round() * tick

        df["volume"] = df["volume"].astype(int)

        return df

    @staticmethod
    def generate_sample_daily(
        index: str = "NIFTY",
        days: int = 400,
        base_price: float = 23500.0,
    ) -> pd.DataFrame:
        """Generate sample daily OHLCV data."""
        daily_vol = 0.012
        prices = np.zeros(days)
        prices[0] = base_price
        current_date = datetime.now() - timedelta(days=days)

        datetimes = []
        actual_days = 0

        for d in range(days * 2):  # Extra to account for weekends
            if actual_days >= days:
                break
            if current_date.weekday() < 5:  # Skip weekends
                datetimes.append(current_date)
                if actual_days > 0:
                    prices[actual_days] = prices[actual_days - 1] * (
                        1 + np.random.normal(0.0002, daily_vol)
                    )
                actual_days += 1
            current_date += timedelta(days=1)

        prices = prices[:actual_days]
        datetimes = datetimes[:actual_days]

        daily_range = prices * daily_vol
        df = pd.DataFrame({
            "datetime": datetimes,
            "close": prices,
            "open": prices + np.random.normal(0, 0.3, actual_days) * daily_range,
            "high": prices + np.abs(np.random.normal(0, 1, actual_days)) * daily_range,
            "low": prices - np.abs(np.random.normal(0, 1, actual_days)) * daily_range,
            "volume": np.random.randint(50000, 500000, actual_days),
        })
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)

        return df
