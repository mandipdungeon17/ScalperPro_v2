"""
=============================================================================
SCALPER PRO - Backtesting Engine
=============================================================================
Tests the scalping and swing strategy on historical data.
Generates comprehensive performance reports.
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import os
import logging

from scalper.indicators.technical import IndicatorEngine
from scalper.core.signal_engine import SignalEngine, SignalType, TradeSignal
from scalper.config.settings import (
    ScalpParameters, SwingParameters, IndexConfig, INDEX_CONFIGS, RiskConfig
)

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Single trade in backtest."""
    index: str
    signal_type: str
    direction: str
    strategy: str
    entry_time: str
    entry_price: float
    target_price: float
    stoploss_price: float
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl_points: float = 0.0
    pnl_rupees: float = 0.0
    score: int = 0
    confidence: float = 0.0
    holding_bars: int = 0


@dataclass
class BacktestResult:
    """Complete backtest results."""
    start_date: str
    end_date: str
    indices: List[str]
    strategy: str
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_win_streak: int = 0
    max_loss_streak: int = 0
    avg_holding_bars: float = 0.0
    best_month_pnl: float = 0.0
    worst_month_pnl: float = 0.0
    monthly_pnl: Dict = field(default_factory=dict)
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class BacktestEngine:
    """
    Backtests the scalping and swing strategy on historical OHLCV data.

    Usage:
        engine = BacktestEngine()
        # Load your historical data as DataFrame with columns:
        # datetime, open, high, low, close, volume
        results = engine.run(
            data={"NIFTY": nifty_df, "BANKNIFTY": banknifty_df},
            daily_data={"NIFTY": nifty_daily_df, ...},
            strategy="both"  # "scalp", "swing", or "both"
        )
        engine.print_report(results)
    """

    def __init__(
        self,
        scalp_params: ScalpParameters = None,
        swing_params: SwingParameters = None,
        risk_config: RiskConfig = None,
    ):
        self.scalp_params = scalp_params or ScalpParameters()
        self.swing_params = swing_params or SwingParameters()
        self.risk = risk_config or RiskConfig()
        self.signal_engine = SignalEngine(self.scalp_params, self.swing_params)

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        daily_data: Optional[Dict[str, pd.DataFrame]] = None,
        strategy: str = "both",
        initial_capital: float = 100000.0,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            data: Dict mapping index name -> intraday OHLCV DataFrame
                  (1-min or 5-min with datetime, open, high, low, close, volume)
            daily_data: Dict mapping index name -> daily OHLCV DataFrame
            strategy: "scalp", "swing", or "both"
            initial_capital: Starting capital for position sizing
        """
        all_trades = []
        start_dates = []
        end_dates = []

        for index_name, df in data.items():
            if index_name not in INDEX_CONFIGS:
                logger.warning(f"Unknown index: {index_name}, skipping")
                continue

            config = INDEX_CONFIGS[index_name]
            daily_df = daily_data.get(index_name) if daily_data else None

            logger.info(f"Backtesting {index_name}: {len(df)} bars")
            start_dates.append(df.iloc[0].get("datetime", df.index[0]))
            end_dates.append(df.iloc[-1].get("datetime", df.index[-1]))

            # Compute indicators
            df_with_indicators = IndicatorEngine.compute_all(df, self.scalp_params)

            # Walk through bars
            trades = self._walk_forward(
                df_with_indicators, config, daily_df, strategy
            )
            all_trades.extend(trades)

        # Compile results
        result = self._compile_results(
            all_trades,
            list(data.keys()),
            strategy,
            str(min(start_dates)) if start_dates else "",
            str(max(end_dates)) if end_dates else "",
        )

        return result

    def _walk_forward(
        self,
        df: pd.DataFrame,
        config: IndexConfig,
        daily_df: Optional[pd.DataFrame],
        strategy: str,
    ) -> List[BacktestTrade]:
        """Walk through bars and simulate trades."""
        trades = []
        open_trade: Optional[BacktestTrade] = None
        daily_trade_count = 0
        daily_pnl = 0.0
        last_date = None

        # Need at least 50 bars of warmup for indicators
        warmup = 50

        for i in range(warmup, len(df)):
            row = df.iloc[i]
            current_time = str(row.get("datetime", df.index[i]))

            # Reset daily counters
            current_date = str(row.get("datetime", df.index[i]))[:10]
            if current_date != last_date:
                daily_trade_count = 0
                daily_pnl = 0.0
                last_date = current_date

            # Check time filters
            time_str = str(row.get("datetime", df.index[i]))
            if len(time_str) > 10:
                hour_min = time_str[11:16]
                if hour_min < self.risk.no_trade_before or hour_min > self.risk.no_trade_after:
                    # Still check exits
                    if open_trade:
                        open_trade = self._check_backtest_exit(
                            open_trade, row, i, config
                        )
                        if open_trade and open_trade.exit_time:
                            daily_pnl += open_trade.pnl_rupees
                            trades.append(open_trade)
                            open_trade = None
                    continue

            # Check exit for open trade
            if open_trade:
                open_trade.holding_bars += 1
                open_trade = self._check_backtest_exit(
                    open_trade, row, i, config
                )
                if open_trade and open_trade.exit_time:
                    daily_pnl += open_trade.pnl_rupees
                    trades.append(open_trade)
                    open_trade = None

            # Don't open new trade if one is already open
            if open_trade is not None:
                continue

            # Risk checks
            if daily_trade_count >= self.risk.max_daily_trades:
                continue
            if daily_pnl < -self.risk.max_daily_loss:
                continue

            # Only check for new signals every 5 bars (reduces computation)
            if i % 5 != 0:
                continue

            # Use pre-computed indicators directly (no re-computation)
            signals = self._fast_signal_check(
                df, i, config, daily_df, strategy
            )

            # Filter by strategy
            for signal in signals:
                if strategy == "scalp" and signal.strategy != "scalp":
                    continue
                if strategy == "swing" and signal.strategy != "swing":
                    continue

                # Take the first valid signal
                trade = BacktestTrade(
                    index=config.symbol,
                    signal_type=signal.signal_type.value,
                    direction=signal.direction,
                    strategy=signal.strategy,
                    entry_time=current_time,
                    entry_price=signal.entry_price,
                    target_price=signal.target_price,
                    stoploss_price=signal.stoploss_price,
                    score=signal.score,
                    confidence=signal.confidence,
                )
                open_trade = trade
                daily_trade_count += 1
                break

        # Close any remaining open trade at last bar
        if open_trade and not open_trade.exit_time:
            last_row = df.iloc[-1]
            open_trade.exit_price = last_row["close"]
            open_trade.exit_time = str(last_row.get("datetime", df.index[-1]))
            open_trade.exit_reason = "EOD_CLOSE"
            self._calculate_trade_pnl(open_trade, config)
            trades.append(open_trade)

        return trades

    def _check_backtest_exit(
        self, trade: BacktestTrade, row: pd.Series, bar_idx: int,
        config: IndexConfig
    ) -> BacktestTrade:
        """Check if trade should be exited based on current bar."""
        high = row["high"]
        low = row["low"]
        close = row["close"]
        current_time = str(row.get("datetime", ""))

        if trade.direction == "LONG":
            # Check SL (using low of bar)
            if low <= trade.stoploss_price:
                trade.exit_price = trade.stoploss_price
                trade.exit_time = current_time
                trade.exit_reason = "SL_HIT"
                self._calculate_trade_pnl(trade, config)
                return trade

            # Check target (using high of bar)
            if high >= trade.target_price:
                trade.exit_price = trade.target_price
                trade.exit_time = current_time
                trade.exit_reason = "TARGET_HIT"
                self._calculate_trade_pnl(trade, config)
                return trade

        elif trade.direction == "SHORT":
            # Check SL
            if high >= trade.stoploss_price:
                trade.exit_price = trade.stoploss_price
                trade.exit_time = current_time
                trade.exit_reason = "SL_HIT"
                self._calculate_trade_pnl(trade, config)
                return trade

            # Check target
            if low <= trade.target_price:
                trade.exit_price = trade.target_price
                trade.exit_time = current_time
                trade.exit_reason = "TARGET_HIT"
                self._calculate_trade_pnl(trade, config)
                return trade

        # Max holding time (60 bars for scalp = ~1 hour on 1-min, 500 for swing)
        max_hold = 60 if trade.strategy == "scalp" else 500
        if trade.holding_bars >= max_hold:
            trade.exit_price = close
            trade.exit_time = current_time
            trade.exit_reason = "MAX_HOLD"
            self._calculate_trade_pnl(trade, config)
            return trade

        return trade

    def _fast_signal_check(self, df, i, config, daily_df, strategy):
        """Fast signal check using pre-computed indicators at bar i."""
        latest = df.iloc[i]
        prev = df.iloc[i - 1]

        signals = []

        # Quick scalp scoring
        if strategy in ("scalp", "both"):
            for direction in ("LONG", "SHORT"):
                score, reasons = self.signal_engine._score_scalp(
                    df.iloc[max(0,i-50):i+1], latest, prev, direction, None, None
                )
                if score >= self.scalp_params.min_scalp_score:
                    sig = self.signal_engine._build_signal(
                        config, latest, direction, "scalp",
                        score, 12, reasons, None, None
                    )
                    signals.append(sig)

        # Swing check (less frequent - every 25 bars)
        if strategy in ("swing", "both") and i % 25 == 0 and daily_df is not None and len(daily_df) > 60:
            sr_levels = IndicatorEngine.find_sr_levels(
                daily_df,
                lookback_days=self.swing_params.yearly_lookback_days,
                min_touches=self.swing_params.sr_touch_count,
                zone_width_pct=self.swing_params.sr_zone_width_pct
            )
            swing_sigs = self.signal_engine._check_swing_signals(
                df.iloc[max(0,i-50):i+1], latest, prev, config,
                sr_levels, None, None, daily_df
            )
            signals.extend(swing_sigs)

        return signals

    @staticmethod
    def _calculate_trade_pnl(trade: BacktestTrade, config: IndexConfig):

        """Calculate P&L for a completed trade."""
        if trade.exit_price is None:
            return

        if trade.direction == "LONG":
            trade.pnl_points = round(trade.exit_price - trade.entry_price, 2)
        else:
            trade.pnl_points = round(trade.entry_price - trade.exit_price, 2)

        # Approximate: assume delta ~0.7 for ITM option, 1 lot
        delta = 0.70
        trade.pnl_rupees = round(
            trade.pnl_points * delta * config.lot_size, 2
        )

    def _compile_results(
        self,
        trades: List[BacktestTrade],
        indices: List[str],
        strategy: str,
        start_date: str,
        end_date: str,
    ) -> BacktestResult:
        """Compile all trades into a results summary."""
        result = BacktestResult(
            start_date=start_date,
            end_date=end_date,
            indices=indices,
            strategy=strategy,
            trades=trades,
        )

        if not trades:
            return result

        result.total_trades = len(trades)
        pnls = [t.pnl_rupees for t in trades]
        result.winners = sum(1 for p in pnls if p > 0)
        result.losers = sum(1 for p in pnls if p < 0)
        result.win_rate = round(result.winners / result.total_trades * 100, 1)
        result.total_pnl = round(sum(pnls), 2)
        result.avg_pnl = round(np.mean(pnls), 2) if pnls else 0

        # Profit factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        result.profit_factor = round(
            gross_profit / gross_loss if gross_loss > 0 else 999, 2
        )

        # Equity curve & drawdown
        equity = [0]
        for p in pnls:
            equity.append(equity[-1] + p)
        result.equity_curve = equity

        peak = equity[0]
        max_dd = 0
        for val in equity:
            if val > peak:
                peak = val
            dd = peak - val
            if dd > max_dd:
                max_dd = dd
        result.max_drawdown = round(max_dd, 2)

        # Sharpe ratio (daily returns approximation)
        if len(pnls) > 1:
            returns = np.array(pnls)
            result.sharpe_ratio = round(
                (np.mean(returns) / np.std(returns)) * np.sqrt(252)
                if np.std(returns) > 0 else 0, 2
            )

        # Win/loss streaks
        current_streak = 0
        max_w = 0
        max_l = 0
        for p in pnls:
            if p > 0:
                if current_streak > 0:
                    current_streak += 1
                else:
                    current_streak = 1
                max_w = max(max_w, current_streak)
            elif p < 0:
                if current_streak < 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                max_l = max(max_l, abs(current_streak))
            else:
                current_streak = 0

        result.max_win_streak = max_w
        result.max_loss_streak = max_l

        # Average holding time
        result.avg_holding_bars = round(
            np.mean([t.holding_bars for t in trades]), 1
        )

        # Monthly P&L
        monthly = {}
        for trade in trades:
            month_key = trade.entry_time[:7]  # YYYY-MM
            monthly[month_key] = monthly.get(month_key, 0) + trade.pnl_rupees
        result.monthly_pnl = {k: round(v, 2) for k, v in monthly.items()}
        if monthly:
            result.best_month_pnl = max(monthly.values())
            result.worst_month_pnl = min(monthly.values())

        return result

    def print_report(self, result: BacktestResult):
        """Print a formatted backtest report."""
        print("\n" + "=" * 70)
        print("  SCALPER PRO - BACKTEST REPORT")
        print("=" * 70)
        print(f"  Period:      {result.start_date} to {result.end_date}")
        print(f"  Indices:     {', '.join(result.indices)}")
        print(f"  Strategy:    {result.strategy}")
        print("-" * 70)
        print(f"  Total Trades:    {result.total_trades}")
        print(f"  Winners:         {result.winners} ({result.win_rate}%)")
        print(f"  Losers:          {result.losers}")
        print(f"  Profit Factor:   {result.profit_factor}")
        print("-" * 70)
        print(f"  Total P&L:       ₹{result.total_pnl:+,.2f}")
        print(f"  Avg Trade P&L:   ₹{result.avg_pnl:+,.2f}")
        print(f"  Max Drawdown:    ₹{result.max_drawdown:,.2f}")
        print(f"  Sharpe Ratio:    {result.sharpe_ratio}")
        print("-" * 70)
        print(f"  Max Win Streak:  {result.max_win_streak}")
        print(f"  Max Loss Streak: {result.max_loss_streak}")
        print(f"  Avg Hold (bars): {result.avg_holding_bars}")
        print("-" * 70)

        if result.monthly_pnl:
            print("\n  MONTHLY P&L:")
            for month, pnl in sorted(result.monthly_pnl.items()):
                bar = "#" * int(abs(pnl) / max(abs(v) for v in result.monthly_pnl.values()) * 20)
                sign = "+" if pnl > 0 else ""
                print(f"    {month}:  ₹{sign}{pnl:>10,.2f}  {bar}")

        print("\n  EXIT REASONS:")
        reasons = {}
        for t in result.trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            pct = count / result.total_trades * 100
            print(f"    {reason:20s}  {count:5d}  ({pct:.1f}%)")

        print("=" * 70 + "\n")

    def save_results(self, result: BacktestResult, filepath: str = None):
        """Save backtest results to JSON."""
        from scalper.config.settings import BACKTEST_DIR
        if filepath is None:
            os.makedirs(BACKTEST_DIR, exist_ok=True)
            filepath = os.path.join(
                BACKTEST_DIR,
                f"backtest_{result.strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        output = {
            "start_date": result.start_date,
            "end_date": result.end_date,
            "indices": result.indices,
            "strategy": result.strategy,
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "total_pnl": result.total_pnl,
            "profit_factor": result.profit_factor,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "max_win_streak": result.max_win_streak,
            "max_loss_streak": result.max_loss_streak,
            "monthly_pnl": result.monthly_pnl,
            "best_month_pnl": result.best_month_pnl,
            "worst_month_pnl": result.worst_month_pnl,
            "equity_curve": result.equity_curve,
            "trades": [
                {
                    "index": t.index, "direction": t.direction,
                    "strategy": t.strategy, "entry_time": t.entry_time,
                    "entry_price": t.entry_price, "exit_price": t.exit_price,
                    "exit_reason": t.exit_reason, "pnl_points": t.pnl_points,
                    "pnl_rupees": t.pnl_rupees, "score": t.score,
                    "holding_bars": t.holding_bars,
                }
                for t in result.trades
            ]
        }

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Backtest results saved to {filepath}")
        return filepath
