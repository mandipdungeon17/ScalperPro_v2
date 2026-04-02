"""
=============================================================================
SCALPER PRO - Main Orchestrator
=============================================================================
Entry point for the trading system. Supports three modes:
1. BACKTEST  - Test on historical data
2. PAPER     - Live signals → Telegram alerts, no real orders
3. LIVE      - Real orders via Dhan (only after paper validation)
=============================================================================

Usage:
    # Backtest mode
    python -m scalper.main --mode backtest --indices NIFTY BANKNIFTY --days 180

    # Paper trading mode
    python -m scalper.main --mode paper --indices NIFTY BANKNIFTY FINNIFTY MIDCPNIFTY SENSEX

    # Live trading mode (use only after 2-3 months of paper trading!)
    python -m scalper.main --mode live --indices NIFTY BANKNIFTY
"""

import argparse
import logging
import time
import json
import signal as sig
import sys
from datetime import datetime, timedelta
from typing import Dict, List

from scalper.config.settings import (
    INDEX_CONFIGS, ScalpParameters, SwingParameters, RiskConfig,
    TradingMode, CURRENT_MODE, MARKET_OPEN, MARKET_CLOSE
)
from scalper.indicators.technical import IndicatorEngine
from scalper.indicators.oi_analyzer import OIAnalyzer, StrikeSelector
from scalper.core.signal_engine import SignalEngine
from scalper.execution.dhan_engine import ExecutionEngine
from scalper.alerts.telegram import TelegramAlerts
from scalper.backtest.engine import BacktestEngine
from scalper.data.fetcher import DataFetcher


# ── Logging Setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"scalper_{datetime.now().strftime('%Y%m%d')}.log"),
    ],
)
logger = logging.getLogger("ScalperPro")


class ScalperPro:
    """Main orchestrator for the trading system."""

    def __init__(self, mode: str = "paper", indices: List[str] = None):
        self.mode = mode.upper()
        self.indices = indices or ["NIFTY", "BANKNIFTY"]
        self.running = False

        # Initialize components
        self.scalp_params = ScalpParameters()
        self.swing_params = SwingParameters()
        self.risk_config = RiskConfig()

        self.signal_engine = SignalEngine(self.scalp_params, self.swing_params)
        self.oi_analyzer = OIAnalyzer()
        self.strike_selector = StrikeSelector()
        self.execution = ExecutionEngine(mode=self.mode)
        self.telegram = TelegramAlerts()
        self.data_fetcher = DataFetcher()

        # Data caches
        self._intraday_cache: Dict[str, dict] = {}
        self._daily_cache: Dict[str, dict] = {}
        self._last_signal_time: Dict[str, datetime] = {}

        logger.info(f"ScalperPro initialized | Mode: {self.mode} | Indices: {self.indices}")

    # ══════════════════════════════════════════════════════════════════════════
    # BACKTEST MODE
    # ══════════════════════════════════════════════════════════════════════════

    def run_backtest(self, days: int = 180, use_sample: bool = True):
        """
        Run backtest on historical data.

        Args:
            days: Number of days to backtest
            use_sample: If True, generates sample data. If False, fetches from Dhan.
        """
        logger.info(f"Starting backtest for {days} days on {self.indices}")
        self.telegram.send_system_status(
            "INFO", f"Starting backtest: {days} days on {', '.join(self.indices)}"
        )

        data = {}
        daily_data = {}

        for index in self.indices:
            if use_sample:
                logger.info(f"Generating sample data for {index}...")
                config = INDEX_CONFIGS.get(index)
                base_prices = {
                    "NIFTY": 23500, "BANKNIFTY": 51000, "FINNIFTY": 23000,
                    "MIDCPNIFTY": 12500, "SENSEX": 77000,
                }
                base = base_prices.get(index, 23500)
                data[index] = DataFetcher.generate_sample_data(
                    index=index, days=days, interval_minutes=5, base_price=base
                )
                daily_data[index] = DataFetcher.generate_sample_daily(
                    index=index, days=days + 100, base_price=base
                )
            else:
                logger.info(f"Fetching historical data for {index} from Dhan...")
                intraday = self.data_fetcher.fetch_index_data(
                    index, interval="5", days_back=days
                )
                if intraday is not None:
                    data[index] = intraday
                    self.data_fetcher.save_data(intraday, index, "5min", "backtest")

                daily = self.data_fetcher.fetch_daily_data(index, days_back=days + 100)
                if daily is not None:
                    daily_data[index] = daily
                    self.data_fetcher.save_data(daily, index, "daily", "backtest")

        if not data:
            logger.error("No data available for backtesting!")
            return

        # Run backtest
        bt_engine = BacktestEngine(
            self.scalp_params, self.swing_params, self.risk_config
        )

        # Test both strategies
        for strategy in ["scalp", "swing", "both"]:
            logger.info(f"\n{'='*50}\nBacktesting strategy: {strategy}\n{'='*50}")
            result = bt_engine.run(
                data=data,
                daily_data=daily_data,
                strategy=strategy,
            )

            bt_engine.print_report(result)
            filepath = bt_engine.save_results(result)
            logger.info(f"Results saved to {filepath}")

            # Send to Telegram
            self.telegram.send_backtest_summary({
                "start_date": result.start_date,
                "end_date": result.end_date,
                "indices": result.indices,
                "total_trades": result.total_trades,
                "win_rate": result.win_rate,
                "total_pnl": result.total_pnl,
                "profit_factor": result.profit_factor,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "max_win_streak": result.max_win_streak,
                "max_loss_streak": result.max_loss_streak,
                "avg_pnl": result.avg_pnl,
                "best_month_pnl": result.best_month_pnl,
                "worst_month_pnl": result.worst_month_pnl,
            })

    # ══════════════════════════════════════════════════════════════════════════
    # PAPER / LIVE TRADING MODE
    # ══════════════════════════════════════════════════════════════════════════

    def run_live(self, scan_interval: int = 30):
        """
        Run the live trading loop (paper or live mode).

        Args:
            scan_interval: Seconds between each scan cycle
        """
        self.running = True
        logger.info(f"Starting {self.mode} trading on {self.indices}")
        self.telegram.send_system_status(
            "STARTED",
            f"Mode: {self.mode}\n"
            f"Indices: {', '.join(self.indices)}\n"
            f"Scan interval: {scan_interval}s\n"
            f"Max daily trades: {self.risk_config.max_daily_trades}\n"
            f"Max daily loss: ₹{self.risk_config.max_daily_loss:,.0f}"
        )

        # Graceful shutdown
        def signal_handler(signum, frame):
            logger.info("Shutdown signal received")
            self.running = False

        sig.signal(sig.SIGINT, signal_handler)
        sig.signal(sig.SIGTERM, signal_handler)

        # Pre-fetch daily data for swing levels
        self._prefetch_daily_data()

        try:
            while self.running:
                now = datetime.now()
                current_time = now.strftime("%H:%M")

                # Only trade during market hours
                if current_time < MARKET_OPEN or current_time > MARKET_CLOSE:
                    if current_time > MARKET_CLOSE:
                        # Send daily summary at EOD
                        self._send_eod_summary()
                        logger.info("Market closed. Waiting for next session...")
                        time.sleep(3600)  # Sleep 1 hour
                    else:
                        logger.info(f"Pre-market. Current time: {current_time}")
                        time.sleep(60)
                    continue

                # Main scan loop
                self._scan_cycle()
                time.sleep(scan_interval)

        except Exception as e:
            logger.error(f"Critical error in trading loop: {e}", exc_info=True)
            self.telegram.send_system_status("ERROR", f"Critical error: {str(e)}")

        finally:
            self.telegram.send_system_status("STOPPED", "Trading session ended")
            logger.info("ScalperPro stopped")

    def _scan_cycle(self):
        """One complete scan cycle across all indices."""
        vix = self.data_fetcher.fetch_india_vix()
        if vix:
            logger.info(f"India VIX: {vix:.2f}")

        for index in self.indices:
            try:
                self._process_index(index, vix)
            except Exception as e:
                logger.error(f"Error processing {index}: {e}", exc_info=True)

    def _process_index(self, index: str, vix: float = None):
        """Process signals for a single index."""
        config = INDEX_CONFIGS.get(index)
        if not config:
            return

        # Fetch latest intraday data (5-min candles)
        df = self.data_fetcher.fetch_index_data(index, interval="5", days_back=5)
        if df is None or len(df) < 50:
            logger.warning(f"Insufficient data for {index}")
            return

        # Compute indicators
        df = IndicatorEngine.compute_all(df, self.scalp_params)

        # Fetch OI data
        oi_snapshot = self.oi_analyzer.analyze(index)

        # Get daily data for swing levels
        daily_df = self._daily_cache.get(index)

        # Generate signals
        signals = self.signal_engine.generate_signals(
            df, config,
            oi_snapshot=oi_snapshot,
            daily_df=daily_df,
            vix=vix,
        )

        if not signals:
            return

        # Process each signal
        for signal in signals:
            # Debounce: don't fire same direction within 5 minutes
            cache_key = f"{index}_{signal.direction}_{signal.strategy}"
            last_time = self._last_signal_time.get(cache_key)
            if last_time and (datetime.now() - last_time).seconds < 300:
                continue

            logger.info(
                f"📡 Signal: {signal.signal_type.value} on {index} | "
                f"Score: {signal.score}/{signal.max_score} | "
                f"Entry: {signal.entry_price:.2f}"
            )

            # Send signal alert
            self.telegram.send_signal_alert(signal)
            self._last_signal_time[cache_key] = datetime.now()

            # Select optimal strike
            chain_data = self.oi_analyzer.fetch_option_chain(index)
            if chain_data:
                # Determine DTE
                dte = self._get_days_to_expiry(index)

                strike_info = self.strike_selector.select_strike(
                    index=index,
                    spot_price=signal.entry_price,
                    direction=signal.direction,
                    chain_data=chain_data,
                    days_to_expiry=dte,
                    strategy=signal.strategy,
                )

                if strike_info:
                    signal.strike_recommendation = strike_info
                    logger.info(
                        f"  Strike: {strike_info['strike']} {strike_info['option_type']} | "
                        f"Δ={strike_info['delta']:.2f} | θ={strike_info['theta']:.2f} | "
                        f"LTP=₹{strike_info['ltp']:.2f}"
                    )

                    # Execute trade
                    trade = self.execution.execute_trade(
                        signal, strike_info, config
                    )

                    if trade:
                        self.telegram.send_trade_entry(trade)
                        logger.info(f"  Trade entered: {trade.trade_id}")
                else:
                    logger.warning(f"  No suitable strike found for {index}")

        # Check exits on open trades
        self._check_open_trades(index, df)

    def _check_open_trades(self, index: str, df):
        """Check if any open trades should be exited."""
        open_trades = [
            t for t in self.execution.trades
            if t.index == index and t.status in ("ENTERED", "TRAILING_SL")
        ]

        if not open_trades:
            return

        current_price = df.iloc[-1]["close"]

        for trade in open_trades:
            # Approximate current premium (using delta)
            if trade.direction == "LONG":
                price_move = current_price - trade.index_entry
            else:
                price_move = trade.index_entry - current_price

            # Rough premium estimation using initial delta
            estimated_premium = trade.entry_price + price_move * 0.70

            # Update trailing SL
            self.execution.update_trailing_sl(trade, estimated_premium)

            # Check exit
            if self.execution.check_exit(trade, estimated_premium):
                self.telegram.send_trade_exit(trade)

    def _prefetch_daily_data(self):
        """Pre-fetch daily data for all indices (for swing levels)."""
        for index in self.indices:
            logger.info(f"Pre-fetching daily data for {index}...")
            daily_df = self.data_fetcher.fetch_daily_data(index, days_back=400)
            if daily_df is not None:
                self._daily_cache[index] = daily_df
                logger.info(f"  Loaded {len(daily_df)} daily bars for {index}")
            else:
                # Use sample data as fallback
                base_prices = {
                    "NIFTY": 23500, "BANKNIFTY": 51000, "FINNIFTY": 23000,
                    "MIDCPNIFTY": 12500, "SENSEX": 77000,
                }
                self._daily_cache[index] = DataFetcher.generate_sample_daily(
                    index, days=400, base_price=base_prices.get(index, 23500)
                )
                logger.info(f"  Using sample daily data for {index}")

    def _get_days_to_expiry(self, index: str) -> int:
        """Calculate days to nearest expiry for the index."""
        config = INDEX_CONFIGS.get(index)
        if not config:
            return 3

        expiry_days = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2,
            "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6,
        }
        target_day = expiry_days.get(config.expiry_day, 3)
        today = datetime.now().weekday()
        days_ahead = target_day - today
        if days_ahead <= 0:
            days_ahead += 7
        return days_ahead

    def _send_eod_summary(self):
        """Send end-of-day summary via Telegram."""
        summary = self.execution.get_daily_summary()
        if summary["total_trades"] > 0:
            self.telegram.send_daily_summary(summary)
            logger.info(
                f"EOD Summary: {summary['total_trades']} trades, "
                f"₹{summary['total_pnl']:+,.2f} P&L, "
                f"{summary['win_rate']:.0f}% win rate"
            )


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ScalperPro Trading System")
    parser.add_argument(
        "--mode", type=str, default="paper",
        choices=["backtest", "paper", "live"],
        help="Trading mode (default: paper)"
    )
    parser.add_argument(
        "--indices", nargs="+",
        default=["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"],
        help="Indices to trade"
    )
    parser.add_argument(
        "--days", type=int, default=180,
        help="Backtest period in days (default: 180)"
    )
    parser.add_argument(
        "--scan-interval", type=int, default=30,
        help="Seconds between scans in live/paper mode (default: 30)"
    )
    parser.add_argument(
        "--sample-data", action="store_true", default=False,
        help="Use sample data for backtest (no API needed)"
    )

    args = parser.parse_args()

    system = ScalperPro(mode=args.mode, indices=args.indices)

    if args.mode == "backtest":
        system.run_backtest(days=args.days, use_sample=args.sample_data)
    else:
        system.run_live(scan_interval=args.scan_interval)


if __name__ == "__main__":
    main()
