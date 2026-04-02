"""
=============================================================================
SCALPER PRO v2 — Unified Main Loop
=============================================================================
Complete flow:

  1. SCAN index levels (Layer 1)  → "Nifty near support → look for CE"
  2. SELECT strike (Layer 3)      → "23500 CE, Δ=0.45, Γ high, Θ low"
  3. ANALYZE premium swings (L2)  → "Premium support ₹118 (3 touches)"
  4. CONFIRM with indicators      → "Supertrend ✅ RSI ✅ Volume ✅ (7/10)"
  5. EXECUTE via Dhan             → Bracket order with SL + Target
  6. ALERT via Telegram           → Full 3-layer breakdown

Modes:
  backtest  → Test on historical data (sample or Dhan)
  paper     → Live signals + Telegram alerts, no real orders
  live      → Real orders via Dhan (only after paper validation!)

Usage:
  python -m scalper.main_v2 --mode paper --indices NIFTY BANKNIFTY
  python -m scalper.main_v2 --mode backtest --days 180 --sample
=============================================================================
"""

import argparse
import logging
import time
import sys
import signal as sig
import pathlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import asdict

# ── Load .env FIRST before any settings are imported ─────────────────────────
# This ensures DHAN_CLIENT_ID, TELEGRAM_BOT_TOKEN etc. are in os.environ
# before config/settings.py reads them with os.getenv().
try:
    from dotenv import load_dotenv
    _env_path = pathlib.Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass  # python-dotenv not installed; rely on real env vars
# ─────────────────────────────────────────────────────────────────────────────

from scalper.config.settings import (
    INDEX_CONFIGS, RiskConfig, MARKET_OPEN, MARKET_CLOSE
)
from scalper.core.index_levels import IndexLevelMarker
from scalper.core.premium_swings import PremiumSwingDetector
from scalper.core.greek_selector import GreekStrikeSelector
from scalper.core.swing_orchestrator import SwingOrchestrator, SwingTradeDecision
from scalper.core.premarket_analysis import PremarketAnalyzer
from scalper.market.market_pulse import MarketPulse, PulseConfig
from scalper.indicators.confirmation import TechnicalConfirmation
from scalper.execution.dhan_engine import ExecutionEngine, TradeRecord
from scalper.alerts.telegram import TelegramAlerts
from scalper.data.fetcher import DataFetcher
from scalper.data.free_fetcher import FreeDataFetcher

_log_dir = pathlib.Path(__file__).parent / "logs"
_log_dir.mkdir(exist_ok=True)
_log_file = _log_dir / f"scalper_v2_{datetime.now().strftime('%Y%m%d')}.log"

# Force UTF-8 on Windows console so Unicode chars (→ ₹ ▲ ▼ etc.) don't crash
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(_log_file), encoding="utf-8"),
    ],
)
logger = logging.getLogger("ScalperPro.v2")


class ScalperProV2:
    """
    Unified trading system with the correct 3-layer flow + indicator confirmation.
    """

    def __init__(self, mode: str = "PAPER", indices: List[str] = None):
        self.mode = mode.upper()
        self.indices = indices or ["NIFTY", "BANKNIFTY"]
        self.running = False

        # Core components
        self.orchestrator = SwingOrchestrator()
        self.confirmation = TechnicalConfirmation(min_score=4)
        self.execution = ExecutionEngine(mode=self.mode)
        self.telegram = TelegramAlerts()
        self.data_fetcher = DataFetcher()
        self.risk = RiskConfig()

        # Pre-market analysis — Dhan is primary, Yahoo Finance is fallback
        self._free_fetcher = FreeDataFetcher()
        self._premarket_analyzer = PremarketAnalyzer(
            dhan_fetcher=self.data_fetcher,
            free_fetcher=self._free_fetcher,
        )
        self._premarket_done: bool = False   # resets each day at 09:00

        # Market Pulse — news, global futures, volume spikes, OI changes
        self._market_pulse = MarketPulse(
            telegram=self.telegram,
            config=PulseConfig(trade_indices=self.indices),
        )

        # Caches
        self._daily_cache: Dict[str, object] = {}
        self._hourly_cache: Dict[str, object] = {}
        self._fifteen_cache: Dict[str, object] = {}
        self._last_signal_time: Dict[str, datetime] = {}

        logger.info(f"ScalperPro v2 initialized | Mode: {self.mode} | Indices: {self.indices}")

    # ══════════════════════════════════════════════════════════════
    # LIVE / PAPER TRADING LOOP
    # ══════════════════════════════════════════════════════════════

    def run_live(self, scan_interval: int = 30):
        """Main trading loop for paper/live mode."""
        self.running = True

        self.telegram.send_system_status(
            "STARTED",
            f"ScalperPro v2 — Swing Trading System\n"
            f"Mode: {self.mode}\n"
            f"Indices: {', '.join(self.indices)}\n"
            f"Scan: {scan_interval}s\n"
            f"Flow: Index S/R -> CE/PE -> Premium Swing -> Indicator Confirm -> Execute"
        )

        # Graceful shutdown
        def handle_signal(signum, frame):
            logger.info("Shutdown signal received")
            self.running = False

        sig.signal(sig.SIGINT, handle_signal)
        sig.signal(sig.SIGTERM, handle_signal)

        # Pre-fetch multi-timeframe data for index levels
        self._prefetch_index_data()

        try:
            while self.running:
                now = datetime.now()
                current_time = now.strftime("%H:%M")

                # Reset pre-market flag at midnight / very early morning
                if current_time < "09:00":
                    self._premarket_done = False

                if current_time < MARKET_OPEN or current_time > MARKET_CLOSE:
                    if current_time > MARKET_CLOSE:
                        self._send_eod_summary()
                        logger.info("Market closed. Sleeping until next session.")
                        time.sleep(3600)
                    else:
                        # ── Pre-market analysis at 09:10 (fires once per day) ──
                        if "09:10" <= current_time < "09:15" and not self._premarket_done:
                            self._run_premarket_analysis()
                            self._premarket_done = True
                        time.sleep(30)
                    continue

                # ── Market Pulse: news / global / stocks / OI ────────
                self._market_pulse.tick()

                # ── Main scan cycle ───────────────────────────────────
                self._scan_all_indices()

                # ── Check exits on open trades ────────────────────────
                self._monitor_open_trades()

                time.sleep(scan_interval)

        except Exception as e:
            logger.error(f"Critical error: {e}", exc_info=True)
            self.telegram.send_system_status("ERROR", str(e))

        finally:
            self.telegram.send_system_status("STOPPED", "Session ended")

    # ══════════════════════════════════════════════════════════════
    # PRE-MARKET ANALYSIS  (09:10 AM — fires once per day)
    # ══════════════════════════════════════════════════════════════

    def _run_premarket_analysis(self):
        """
        Fetch multi-TF data, mark all key levels, determine trend bias
        for each selected index, and send one Telegram message per index.
        Runs once at 09:10 AM before market opens.
        """
        logger.info("=" * 60)
        logger.info("PRE-MARKET ANALYSIS STARTING")
        logger.info("=" * 60)

        for index in self.indices:
            try:
                report = self._premarket_analyzer.run(index)
                if report:
                    self.telegram.send_premarket_analysis(report)
                    logger.info(
                        f"[{index}] Pre-market sent  |  Bias={report.bias} "
                        f"({report.bias_score:+d}/5)  |  "
                        f"Bull={report.bullish_tfs}/5  Bear={report.bearish_tfs}/5  |  "
                        f"Levels={len(report.key_levels)}"
                    )
                else:
                    logger.warning(f"[{index}] Pre-market analysis returned no data")
            except Exception as e:
                logger.error(f"[{index}] Pre-market analysis error: {e}", exc_info=True)

        logger.info("PRE-MARKET ANALYSIS COMPLETE")
        logger.info("=" * 60)

    def _scan_all_indices(self):
        """Scan all indices through the 3-layer pipeline."""
        vix = self.data_fetcher.fetch_india_vix()

        for index in self.indices:
            try:
                self._process_index(index, vix)
            except Exception as e:
                logger.error(f"Error processing {index}: {e}", exc_info=True)

    def _process_index(self, index: str, vix: float = None):
        """Run complete 3-layer analysis for one index."""
        config = INDEX_CONFIGS.get(index)
        if not config:
            return

        # ── Risk checks ──────────────────────────────────────────
        can_trade, reason = self.execution.can_trade()
        if not can_trade:
            logger.debug(f"[{index}] Trade blocked: {reason}")
            return

        # ── Live spot price (NSE API — no auth required) ─────────
        # CRITICAL: never use daily_df last close as live price.
        # NSE allIndices gives the real-time index value.
        current_spot = self.data_fetcher.fetch_nse_spot_price(index)
        if not current_spot or current_spot <= 0:
            logger.warning(f"[{index}] Could not fetch live spot price — skipping scan")
            return
        logger.debug(f"[{index}] Live spot: {current_spot:,.2f}")

        # ── Historical data for level marking ────────────────────
        daily_df = self._daily_cache.get(index)
        if daily_df is None or len(daily_df) < 50:
            logger.warning(f"[{index}] Insufficient daily history — skipping")
            return

        hourly_df  = self._hourly_cache.get(index)
        fifteen_df = self._fifteen_cache.get(index)

        # ── Debounce: skip if we signaled recently ───────────────
        cache_key = f"{index}_swing"
        last = self._last_signal_time.get(cache_key)
        if last and (datetime.now() - last).seconds < 300:
            return

        # ── Determine DTE for Greeks ─────────────────────────────
        dte = self._get_dte(index)
        base_iv = vix if vix else 14

        # ── LAYER 1: Mark index S/R levels ───────────────────────
        self.orchestrator.index_marker.mark_levels(
            daily_df=daily_df, hourly_df=hourly_df,
            fifteen_min_df=fifteen_df, index=index
        )

        proximity = self.orchestrator.index_marker.check_proximity(current_spot, index)

        if proximity.action == "WAIT" or proximity.proximity_zone == "FAR":
            logger.debug(
                f"[{index}] No nearby S/R — spot {current_spot:,.0f} | "
                f"nearest {proximity.nearest_level.price:,.0f} "
                f"({proximity.distance_atr:.2f} ATR)"
            )
            return

        option_type = proximity.direction
        if option_type not in ("CE", "PE"):
            return

        logger.info(
            f"[{index}] Near {proximity.nearest_level.level_type.value} "
            f"@ {proximity.nearest_level.price:.0f} "
            f"({proximity.distance_atr:.2f} ATR) -> {option_type}"
        )

        # ── IMMEDIATE L1 ALERT ───────────────────────────────────
        # Notify user as soon as Layer 1 detects a signal — before
        # Layer 2-4 analysis, so the user always gets the raw signal.
        l1_msg = (
            f"&#128268; <b>L1 SIGNAL: {index}</b>\n\n"
            f"{index} spot: <b>{current_spot:,.0f}</b>\n"
            f"Level: <b>{proximity.nearest_level.level_type.value} @ {proximity.nearest_level.price:.0f}</b>\n"
            f"Zone: {proximity.proximity_zone} ({proximity.distance_atr:.2f} ATR away)\n"
            f"Direction: <b>{option_type}</b> | Touches: {proximity.nearest_level.touches}\n"
            f"Confidence: {proximity.confidence:.0%}\n"
            f"Running Layer 2-4 analysis...\n"
            f"<i>{datetime.now().strftime('%H:%M:%S IST')}</i>"
        )
        self.telegram._send(l1_msg)

        # Select best strike
        strike_rec = self.orchestrator.greek_selector.select_strike(
            spot=current_spot, option_type=option_type, dte=dte,
            base_iv=base_iv, strike_interval=config.strike_interval,
        )
        if not strike_rec.best_strike:
            return

        selected_strike = strike_rec.best_strike.strike

        # ── LAYER 2: Fetch premium data and find swing setup ─────
        # In live mode: fetch actual option premium candles from Dhan
        # For now: try to fetch, fallback to simulated
        premium_1min = self._fetch_option_candles(
            index, selected_strike, option_type, "1", days_back=3
        )
        premium_5min = self._fetch_option_candles(
            index, selected_strike, option_type, "5", days_back=5
        )
        premium_15min = self._fetch_option_candles(
            index, selected_strike, option_type, "15", days_back=10
        )

        if premium_1min is None or len(premium_1min) < 20:
            logger.info(f"[{index}] No premium data for {selected_strike} {option_type} — sending L1-only alert")
            # Still send a useful alert when option data is unavailable
            atm_alert = (
                f"&#9888; <b>SIGNAL (No option data)</b>\n\n"
                f"{index} | <b>{selected_strike} {option_type}</b>\n"
                f"Level: {proximity.nearest_level.level_type.value} @ {proximity.nearest_level.price:.0f}\n"
                f"Zone: {proximity.proximity_zone} ({proximity.distance_atr:.2f} ATR)\n\n"
                f"<i>Option candle data unavailable from Dhan.\n"
                f"Signal based on index structure only.\n"
                f"Manual verification recommended.</i>\n"
                f"<i>{datetime.now().strftime('%H:%M:%S IST')}</i>"
            )
            self.telegram._send(atm_alert)
            return

        # Run Layer 2
        setup = self.orchestrator.premium_detector.analyze(
            premium_1min=premium_1min, premium_5min=premium_5min,
            premium_15min=premium_15min,
            option_type=option_type, strike=selected_strike, index=index,
        )

        if not setup:
            logger.info(f"[{index}] No swing setup on {selected_strike} {option_type} premium")
            self.telegram._send(
                f"&#128268; <b>{index} L1 Signal — No premium setup</b>\n"
                f"{selected_strike} {option_type} | Level @ {proximity.nearest_level.price:.0f}\n"
                f"<i>Layer 2 found no swing structure in option data.\n"
                f"Market may be in free move — no safe entry.</i>\n"
                f"<i>{datetime.now().strftime('%H:%M:%S IST')}</i>"
            )
            return

        if setup.setup_quality not in ("A+", "A", "B"):
            logger.info(f"[{index}] Low quality setup: {setup.setup_quality} — notifying")
            self.telegram._send(
                f"&#128268; <b>{index} Weak Setup ({setup.setup_quality})</b>\n"
                f"{selected_strike} {option_type} | Level @ {proximity.nearest_level.price:.0f}\n"
                f"Entry Rs{setup.entry_premium} | SL Rs{setup.stoploss_premium} | "
                f"R:R 1:{setup.risk_reward}\n"
                f"<i>Grade too low for execution. Watching for improvement.</i>\n"
                f"<i>{datetime.now().strftime('%H:%M:%S IST')}</i>"
            )
            return

        # ── LAYER 4: Technical Confirmation ──────────────────────
        logger.info(f"[{index}] Swing setup found — running indicator confirmation")

        # Run confirmation on 5-min premium chart (best for confirmation)
        confirm_df = premium_5min if premium_5min is not None and len(premium_5min) > 25 else premium_1min
        confirm_df = self.confirmation.compute_indicators(confirm_df)

        # ── EMA + SUPERTREND CONFLUENCE CHECK (independent alert) ─
        # Fires a separate Telegram alert if Supertrend line AND any EMA
        # are both sitting at/near the swing support level — strong confluence.
        self._check_ema_supertrend_at_support(
            df=confirm_df,
            support_level=setup.entry_at_level.price,
            index=index,
            strike=selected_strike,
            option_type=option_type,
            setup=setup,
        )

        conf_result = self.confirmation.confirm(
            df=confirm_df,
            direction="LONG",  # Always buying the option
            entry_price=setup.entry_premium,
            support_level=setup.entry_at_level.price,
        )

        logger.info(
            f"[{index}] Confirmation: {conf_result.total_score}/{conf_result.max_possible} "
            f"({'CONFIRMED' if conf_result.confirmed else 'REJECTED'})"
        )
        for line in conf_result.summary[1:]:  # Skip first line (header)
            logger.info(f"  {line}")

        if not conf_result.confirmed:
            # Still send alert but mark as not trading
            self._send_rejected_alert(
                index, selected_strike, option_type, setup,
                strike_rec.best_strike, proximity, conf_result
            )
            self._last_signal_time[cache_key] = datetime.now()
            return

        # ── BUILD TRADE DECISION ─────────────────────────────────
        decision = SwingTradeDecision(
            timestamp=datetime.now().isoformat(),
            index=index,
            decision_id=f"SWG_{index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            index_level_action=proximity.action,
            index_proximity=proximity.proximity_zone,
            index_level_price=proximity.nearest_level.price,
            index_level_strength=proximity.nearest_level.strength,
            index_level_type=proximity.nearest_level.level_type.value,
            index_level_timeframe=proximity.nearest_level.timeframe.value,
            index_level_touches=proximity.nearest_level.touches,
            index_distance_atr=proximity.distance_atr,
            index_trend=proximity.trend_context,
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
                sum(setup.entry_at_level.bounce_magnitudes) /
                max(len(setup.entry_at_level.bounce_magnitudes), 1)
            ),
            premium_trend=setup.premium_trend,
            premium_trend_action=setup.premium_trend_action,
            setup_quality=setup.setup_quality,
            multi_tf_confirmation=setup.confirmation_count,
            delta=strike_rec.best_strike.delta,
            gamma=strike_rec.best_strike.gamma,
            theta=strike_rec.best_strike.theta,
            theta_pct_per_day=strike_rec.best_strike.theta_pct_per_day,
            iv=strike_rec.best_strike.iv,
            greek_score=strike_rec.best_strike.swing_score,
            moneyness=strike_rec.best_strike.moneyness,
            should_trade=True,
            confidence=round(
                (proximity.confidence * 0.25 + setup.confidence * 0.35 +
                 strike_rec.best_strike.swing_score / 15 * 0.2 +
                 conf_result.percentage / 100 * 0.2), 3
            ),
            all_reasons=setup.reasons + conf_result.summary[1:],
        )

        # ── EXECUTE ──────────────────────────────────────────────
        logger.info(
            f"[{index}] EXECUTING: {selected_strike} {option_type} | "
            f"Entry Rs{setup.entry_premium} -> Target Rs{setup.target_premium} | "
            f"SL Rs{setup.stoploss_premium} | R:R 1:{setup.risk_reward}"
        )

        trade = self._execute_swing_trade(decision, config)

        # ── TELEGRAM ALERT ───────────────────────────────────────
        alert_msg = self._format_full_alert(decision, conf_result, trade)
        self.telegram._send(alert_msg)

        self._last_signal_time[cache_key] = datetime.now()

    def _execute_swing_trade(
        self, decision: SwingTradeDecision, config
    ) -> Optional[TradeRecord]:
        """Execute the swing trade via Dhan (paper or live)."""
        num_lots = min(
            config.max_lots,
            int(self.risk.max_capital_per_trade / (decision.entry_premium * config.lot_size))
        )
        if num_lots < 1:
            return None

        trade = TradeRecord(
            trade_id=decision.decision_id,
            timestamp=decision.timestamp,
            index=decision.index,
            strategy="swing",
            direction="LONG",
            option_type=decision.option_type,
            strike=decision.strike,
            lot_size=config.lot_size,
            num_lots=num_lots,
            entry_price=decision.entry_premium,
            target_price=decision.target_premium,
            stoploss_price=decision.stoploss_premium,
            index_entry=decision.index_level_price,
            index_target=0,
            index_stoploss=0,
            signal_score=decision.greek_score,
            signal_confidence=decision.confidence,
            signal_reasons=decision.all_reasons[:5],
            mode=self.mode,
        )

        if self.mode == "LIVE" and self.execution.dhan:
            # Resolve option security_id from instrument master
            exchange = "BSE_FNO" if decision.index == "SENSEX" else "NSE_FNO"
            security_id = self.data_fetcher.resolve_option_security_id(
                decision.index, decision.strike, decision.option_type
            )
            if not security_id:
                logger.error(
                    f"Cannot resolve security_id for {decision.index} "
                    f"{decision.strike} {decision.option_type} — order aborted"
                )
                trade.status = "REJECTED"
                self.execution.trades.append(trade)
                self.execution._save_trades()
                return trade

            sl_val = round(decision.entry_premium - decision.stoploss_premium, 2)
            target_val = round(decision.target_premium - decision.entry_premium, 2)

            result = self.execution.dhan.place_bracket_order(
                security_id=security_id,
                exchange_segment=exchange,
                transaction_type="BUY",
                quantity=config.lot_size * num_lots,
                price=decision.entry_premium,
                stoploss_value=sl_val,
                target_value=target_val,
            )
            if "error" in result:
                trade.status = "REJECTED"
                logger.error(f"Order rejected: {result['error']}")
            else:
                trade.status = "ENTERED"
                trade.order_id = result.get("orderId", "")
        else:
            trade.status = "ENTERED"
            trade.order_id = f"PAPER_{trade.trade_id}"

        self.execution.trades.append(trade)
        self.execution.daily_trade_count += 1
        self.execution._save_trades()
        return trade

    def _monitor_open_trades(self):
        """Check all open trades for SL/target hits using live Dhan quotes."""
        open_trades = [
            t for t in self.execution.trades
            if t.status in ("ENTERED", "TRAILING_SL")
        ]

        for trade in open_trades:
            current_premium = self._get_live_premium(trade.index, trade.strike, trade.option_type)
            if current_premium is None or current_premium <= 0:
                continue

            # Update trailing SL first (may raise the floor)
            self.execution.update_trailing_sl(trade, current_premium)

            # Check if SL or target hit
            exited = self.execution.check_exit(trade, current_premium)
            if exited:
                emoji = "✅" if trade.status == "TARGET_HIT" else "🛑"
                msg = (
                    f"{emoji} <b>TRADE EXITED — {trade.status}</b>\n\n"
                    f"{trade.index} | {trade.strike} {trade.option_type}\n"
                    f"Entry: ₹{trade.entry_price:.2f} → Exit: ₹{trade.exit_price:.2f}\n"
                    f"P&amp;L: ₹{trade.pnl:+,.2f} ({trade.pnl_points:+.1f} pts)\n"
                    f"⏰ {datetime.now().strftime('%H:%M:%S IST')}"
                )
                self.telegram._send(msg)
                logger.info(
                    f"[{trade.index}] {trade.status}: {trade.strike} {trade.option_type} | "
                    f"P&L Rs{trade.pnl:+,.2f}"
                )

    def _get_live_premium(self, index: str, strike: int, option_type: str) -> Optional[float]:
        """Fetch current LTP for an option contract from Dhan."""
        security_id = self.data_fetcher.resolve_option_security_id(index, strike, option_type)
        if not security_id:
            return None

        exchange = "BSE_FNO" if index == "SENSEX" else "NSE_FNO"
        try:
            quote = self.data_fetcher.fetch_live_quote(security_id, exchange)
            if not quote:
                return None
            # Dhan LTP response: {"data": {"NSE_FNO": {"<sec_id>": {"last_price": ...}}}}
            seg_data = quote.get("data", {}).get(exchange, {})
            ltp_data = seg_data.get(str(security_id), {})
            ltp = ltp_data.get("last_price") or ltp_data.get("ltp")
            return float(ltp) if ltp else None
        except Exception as e:
            logger.warning(f"Live quote fetch failed for {index} {strike} {option_type}: {e}")
            return None

    # ══════════════════════════════════════════════════════════════
    # DATA HELPERS
    # ══════════════════════════════════════════════════════════════

    def _prefetch_index_data(self):
        """
        Pre-fetch multi-timeframe OHLCV for all indices.

        Priority chain (each falls through to the next on failure):
          1. Dhan API  (IDX_I segment, requires valid token)
          2. yfinance  (free, real historical data — for S/R level marking)
          3. Sample    (synthetic — last resort, clearly logged as fallback)

        Live SPOT price during scans always comes from NSE API independently
        of this cache (see _process_index → fetch_nse_spot_price).
        """
        for index in self.indices:
            logger.info(f"[PREFETCH] {index} — trying Dhan API...")
            source = "dhan"

            # ── 1. Daily data ────────────────────────────────────────
            daily = self.data_fetcher.fetch_daily_data(index, days_back=400)

            if daily is None or len(daily) < 50:
                logger.warning(f"[PREFETCH] {index} Dhan daily failed — trying yfinance")
                source = "yfinance"
                daily = self._free_fetcher.fetch_daily(index, period="2y")

            if daily is None or len(daily) < 50:
                logger.warning(
                    f"[PREFETCH] {index} yfinance also failed — using SAMPLE DATA. "
                    f"S/R levels will be synthetic. Fix Dhan token or install yfinance."
                )
                source = "sample"
                bases = {"NIFTY": 23500, "BANKNIFTY": 51000, "FINNIFTY": 23000,
                         "MIDCPNIFTY": 12500, "SENSEX": 77000}
                daily = DataFetcher.generate_sample_daily(index, 400, bases.get(index, 23500))

            self._daily_cache[index] = daily

            # ── 2. Hourly data ───────────────────────────────────────
            hourly = self.data_fetcher.fetch_index_data(index, "60", 30)
            if hourly is None or len(hourly) < 20:
                hourly = self._free_fetcher.fetch_intraday(index, interval="1h", period="30d")
            if hourly is not None and len(hourly) >= 20:
                self._hourly_cache[index] = hourly

            # ── 3. 15-min data ───────────────────────────────────────
            fifteen = self.data_fetcher.fetch_index_data(index, "15", 10)
            if fifteen is None or len(fifteen) < 20:
                fifteen = self._free_fetcher.fetch_intraday(index, interval="15m", period="30d")
            if fifteen is not None and len(fifteen) >= 20:
                self._fifteen_cache[index] = fifteen

            logger.info(
                f"[PREFETCH] {index} done | source={source} | "
                f"daily={len(self._daily_cache.get(index, []))} bars | "
                f"hourly={len(self._hourly_cache.get(index, []))} bars | "
                f"15min={len(self._fifteen_cache.get(index, []))} bars"
            )

    def _fetch_option_candles(
        self, index: str, strike: int, option_type: str,
        interval: str, days_back: int
    ):
        """Fetch option premium candle data from Dhan instrument master + historical API."""
        config = INDEX_CONFIGS.get(index)
        if not config:
            return None

        # Resolve the option's security_id from Dhan instrument master
        security_id = self.data_fetcher.resolve_option_security_id(
            index, strike, option_type
        )

        if security_id:
            exchange = "BSE_FNO" if index == "SENSEX" else "NSE_FNO"
            from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            to_date = datetime.now().strftime("%Y-%m-%d")
            try:
                df = self.data_fetcher.fetch_dhan_historical(
                    security_id=security_id,
                    exchange_segment=exchange,
                    instrument="OPTIDX",
                    interval=interval,
                    from_date=from_date,
                    to_date=to_date,
                )
                if df is not None and len(df) >= 20:
                    logger.info(
                        f"[{index}] Fetched {len(df)} real candles for "
                        f"{strike} {option_type} (sec_id={security_id})"
                    )
                    return df
            except Exception as e:
                logger.warning(f"[{index}] Dhan option candle fetch failed: {e}")

        # Fallback: generate sample premium data (paper/backtest mode)
        logger.debug(f"[{index}] Using sample data for {strike} {option_type} premium")
        bases = {"NIFTY": 23500, "BANKNIFTY": 51000, "FINNIFTY": 23000,
                 "MIDCPNIFTY": 12500, "SENSEX": 77000}
        spot = bases.get(index, 23500)
        est_premium = max(abs(spot - strike) * 0.7, 50) + 80

        return DataFetcher.generate_sample_data(
            index, days=days_back,
            interval_minutes=int(interval),
            base_price=est_premium
        )

    def _get_dte(self, index: str) -> int:
        config = INDEX_CONFIGS.get(index)
        if not config:
            return 5
        expiry_days = {"Monday": 0, "Tuesday": 1, "Wednesday": 2,
                       "Thursday": 3, "Friday": 4}
        target = expiry_days.get(config.expiry_day, 3)
        today = datetime.now().weekday()
        days_ahead = target - today
        if days_ahead <= 0:
            days_ahead += 7
        return days_ahead

    # ══════════════════════════════════════════════════════════════
    # TELEGRAM FORMATTING
    # ══════════════════════════════════════════════════════════════

    def _format_full_alert(
        self,
        decision: SwingTradeDecision,
        conf: object,
        trade: Optional[TradeRecord],
    ) -> str:
        """Format the complete 4-layer alert for Telegram."""
        mode_tag = "📝 PAPER" if self.mode == "PAPER" else "💰 LIVE"

        msg = f"""
🟢 <b>SWING TRADE {mode_tag}</b>

<b>━━ L1: INDEX LEVEL ━━</b>
{decision.index} near {decision.index_level_type} @ {decision.index_level_price:.0f}
{decision.index_level_touches} touches | {decision.index_level_timeframe} | Str: {decision.index_level_strength:.2f}
Distance: {decision.index_distance_atr:.2f} ATR | Trend: {decision.index_trend}
→ <b>{decision.index_level_action}</b>

<b>━━ L3: GREEKS ━━</b>
{decision.strike} {decision.option_type} ({decision.moneyness})
Δ={decision.delta:.3f} | Γ={decision.gamma:.5f} | Θ={decision.theta_pct_per_day:.1f}%/day
IV={decision.iv:.1f}% | Score: {decision.greek_score}/15

<b>━━ L2: PREMIUM SWING ━━</b>
Support: {decision.premium_support_touches} touches | Avg bounce: {decision.premium_avg_bounce:.1f}pts
Trend: {decision.premium_trend} → {decision.premium_trend_action}
<b>Grade: {decision.setup_quality}</b> | {decision.multi_tf_confirmation}/3 TF confirmed

<b>━━ L4: INDICATOR CONFIRMATION ━━</b>
Score: {conf.total_score}/{conf.max_possible} ({conf.percentage}%)
13EMA: {self._score_emoji(conf.ema13_channel_score)} | 5EMA: {self._score_emoji(conf.ema5_score)}
ST: {self._score_emoji(conf.supertrend_score)} | RSI: {self._score_emoji(conf.rsi_score)}
Vol: {self._score_emoji(conf.volume_score)} | BB: {self._score_emoji(conf.bollinger_score)}
VWAP: {self._score_emoji(conf.vwap_score)}

<b>━━ TRADE ━━</b>
<b>Entry:</b> ₹{decision.entry_premium:.2f}
<b>Target:</b> ₹{decision.target_premium:.2f} <b>(+{decision.target_points:.1f} pts)</b>
<b>SL:</b> ₹{decision.stoploss_premium:.2f} <b>(-{decision.sl_points:.1f} pts)</b>
<b>R:R:</b> 1:{decision.risk_reward:.1f}
<b>Confidence:</b> {decision.confidence:.0%}

{f'Order: {trade.order_id}' if trade else ''}
⏰ {datetime.now().strftime('%H:%M:%S IST')}
""".strip()
        return msg

    def _send_rejected_alert(self, index, strike, otype, setup, greeks, prox, conf):
        """Send alert for a setup that was found but rejected by indicators."""
        msg = f"""
⏸️ <b>SETUP FOUND — NOT CONFIRMED</b>

{index} | {strike} {otype} | Grade: {setup.setup_quality}
Entry ₹{setup.entry_premium} → Target ₹{setup.target_premium} (+{setup.target_points:.1f}pts)
SL ₹{setup.stoploss_premium} (-{setup.sl_points:.1f}pts) | R:R 1:{setup.risk_reward}

<b>Confirmation FAILED: {conf.total_score}/{conf.max_possible}</b>
{chr(10).join(conf.summary[1:5])}

Watching for re-entry if indicators improve.
⏰ {datetime.now().strftime('%H:%M:%S IST')}
""".strip()
        self.telegram._send(msg)

    @staticmethod
    def _score_emoji(score: int) -> str:
        if score >= 2: return "✅✅"
        if score >= 1: return "✅"
        if score == 0: return "➖"
        return "⚠️"

    def _check_ema_supertrend_at_support(
        self,
        df: pd.DataFrame,
        support_level: float,
        index: str,
        strike: int,
        option_type: str,
        setup,
    ) -> bool:
        """
        Fire a special Telegram alert when BOTH:
          - The Supertrend line value is near/at the swing support level
          - AND any EMA (5 EMA, 13 EMA mid, or 13 EMA low) is near that same level

        This is a strong confluence signal: the market's dynamic support (EMA)
        and trend indicator (Supertrend) are stacked on top of the swing support.
        Fires independently — does not block or replace the main trade logic.
        Returns True if confluence detected.
        """
        if len(df) < 5 or support_level <= 0:
            return False

        latest = df.iloc[-1]

        # Tolerance: within 1.5% of the support level
        tol = support_level * 0.015

        supertrend_val = latest.get("supertrend", 0)
        supertrend_dir = int(latest.get("supertrend_dir", 0))
        ema5_val       = latest.get("ema5", 0)
        ema13_mid      = latest.get("ema13_mid", 0)
        ema13_low      = latest.get("ema13_low", 0)

        st_at_support = supertrend_val > 0 and abs(supertrend_val - support_level) <= tol

        ema_name, ema_val = None, 0
        for name, val in [("5 EMA", ema5_val), ("13 EMA Mid", ema13_mid), ("13 EMA Low", ema13_low)]:
            if val > 0 and abs(val - support_level) <= tol:
                ema_name, ema_val = name, val
                break

        if st_at_support and ema_name:
            logger.info(
                f"[{index}] EMA+SUPERTREND CONFLUENCE at support {support_level:.1f} | "
                f"ST={supertrend_val:.1f} | {ema_name}={ema_val:.1f}"
            )
            self.telegram.send_ema_supertrend_support_alert(
                index=index,
                strike=strike,
                option_type=option_type,
                support_level=support_level,
                supertrend_val=supertrend_val,
                supertrend_dir=supertrend_dir,
                ema_name=ema_name,
                ema_val=ema_val,
                current_premium=float(latest["close"]),
                entry_premium=setup.entry_premium,
                sl_premium=setup.stoploss_premium,
                target_premium=setup.target_premium,
                setup_grade=setup.setup_quality,
            )
            return True

        return False

    def _send_eod_summary(self):
        summary = self.execution.get_daily_summary()
        if summary.get("total_trades", 0) > 0:
            self.telegram.send_daily_summary(summary)

    # ══════════════════════════════════════════════════════════════
    # BACKTEST MODE
    # ══════════════════════════════════════════════════════════════

    def run_backtest(self, days: int = 180, use_sample: bool = True):
        """Run backtest on historical data with all 4 layers."""
        logger.info(f"Starting v2 backtest: {days} days, {self.indices}")

        from scalper.backtest.engine import BacktestEngine

        data = {}
        daily = {}
        bases = {"NIFTY": 23500, "BANKNIFTY": 51000, "FINNIFTY": 23000,
                 "MIDCPNIFTY": 12500, "SENSEX": 77000}

        for idx in self.indices:
            base = bases.get(idx, 23500)
            if use_sample:
                data[idx] = DataFetcher.generate_sample_data(idx, days, 5, base)
                daily[idx] = DataFetcher.generate_sample_daily(idx, 400, base)
            else:
                d = self.data_fetcher.fetch_index_data(idx, "5", days)
                if d is not None:
                    data[idx] = d
                dd = self.data_fetcher.fetch_daily_data(idx, 400)
                if dd is not None:
                    daily[idx] = dd

        if not data:
            logger.error("No data for backtest")
            return

        bt = BacktestEngine()
        result = bt.run(data, daily, strategy="scalp")
        bt.print_report(result)

        self.telegram.send_backtest_summary({
            "start_date": result.start_date, "end_date": result.end_date,
            "indices": result.indices, "total_trades": result.total_trades,
            "win_rate": result.win_rate, "total_pnl": result.total_pnl,
            "profit_factor": result.profit_factor, "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown, "max_win_streak": result.max_win_streak,
            "max_loss_streak": result.max_loss_streak, "avg_pnl": result.avg_pnl,
            "best_month_pnl": result.best_month_pnl, "worst_month_pnl": result.worst_month_pnl,
        })


# ══════════════════════════════════════════════════════════════════
# INTERACTIVE INDEX SELECTOR
# ══════════════════════════════════════════════════════════════════

ALL_INDICES = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"]

def _select_indices_interactive() -> list:
    """Show a numbered menu and return the user's chosen indices."""
    print()
    print("=" * 55)
    print("  SELECT INDICES TO TRADE TODAY")
    print("=" * 55)
    for i, idx in enumerate(ALL_INDICES, 1):
        lots = {"NIFTY": 75, "BANKNIFTY": 30, "FINNIFTY": 40,
                "MIDCPNIFTY": 50, "SENSEX": 20}
        expiry = {"NIFTY": "Thu", "BANKNIFTY": "Wed", "FINNIFTY": "Tue",
                  "MIDCPNIFTY": "Mon", "SENSEX": "Fri"}
        print(f"  {i}. {idx:<12}  lot={lots[idx]}  expiry={expiry[idx]}")
    print(f"  6. ALL  (all 5 indices)")
    print("=" * 55)
    print("  Enter numbers separated by commas  e.g.  1,2  or  1,5")
    print()

    while True:
        try:
            raw = input("  Your choice: ").strip()
            if not raw:
                continue

            if raw == "6":
                chosen = ALL_INDICES[:]
            else:
                chosen = []
                for part in raw.split(","):
                    part = part.strip()
                    if part.isdigit() and 1 <= int(part) <= 5:
                        idx = ALL_INDICES[int(part) - 1]
                        if idx not in chosen:
                            chosen.append(idx)

            if not chosen:
                print("  Nothing selected — enter numbers 1-6.")
                continue

            print()
            print(f"  Trading today: {', '.join(chosen)}")
            confirm = input("  Confirm? (y/n): ").strip().lower()
            if confirm == "y":
                print()
                return chosen
            # else loop again

        except (KeyboardInterrupt, EOFError):
            print("\n  Defaulting to NIFTY + BANKNIFTY")
            return ["NIFTY", "BANKNIFTY"]


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main():
    print("""
+---------------------------------------------------+
|  SCALPER PRO v2 - Swing Trading System            |
|  Index S/R -> CE/PE -> Premium Swing -> Confirm   |
+---------------------------------------------------+
    """)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="paper", choices=["backtest", "paper", "live"])
    parser.add_argument("--indices", nargs="+", default=None,
                        help="Indices to trade. If omitted, interactive prompt appears.")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--scan", type=int, default=30)
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    # ── Index selection ───────────────────────────────────────────
    if args.indices:
        # Validate anything passed via CLI
        indices = [i.upper() for i in args.indices if i.upper() in ALL_INDICES]
        if not indices:
            print("No valid indices provided. Valid options:", ALL_INDICES)
            return
    else:
        # Interactive prompt (default when started with no --indices flag)
        indices = _select_indices_interactive()

    system = ScalperProV2(mode=args.mode, indices=indices)

    if args.mode == "backtest":
        system.run_backtest(days=args.days, use_sample=args.sample)
    elif args.mode == "live":
        print("WARNING: LIVE MODE uses real money!")
        if input("Type 'YES' to proceed: ") != "YES":
            return
        system.run_live(scan_interval=args.scan)
    else:
        system.run_live(scan_interval=args.scan)


if __name__ == "__main__":
    main()
