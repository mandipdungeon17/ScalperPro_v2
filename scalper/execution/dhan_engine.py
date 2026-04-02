"""
=============================================================================
SCALPER PRO - Dhan Execution Engine
=============================================================================
Integrates with Dhan Trading API for:
- Order placement (market, limit, bracket orders)
- Position monitoring & trailing stop loss
- Paper trading mode (logs trades without real execution)
=============================================================================
"""

import json
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import os

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Trade Record
# ─────────────────────────────────────────────────────────────────────────────

class TradeStatus(Enum):
    PENDING = "PENDING"
    ENTERED = "ENTERED"
    PARTIAL = "PARTIAL"
    EXITED = "EXITED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    SL_HIT = "SL_HIT"
    TARGET_HIT = "TARGET_HIT"
    TRAILING_SL = "TRAILING_SL"


@dataclass
class TradeRecord:
    """Complete record of a trade."""
    trade_id: str
    timestamp: str
    index: str
    strategy: str                  # "scalp" or "swing"
    direction: str                 # "LONG" or "SHORT"
    option_type: str               # "CE" or "PE"
    strike: int
    lot_size: int
    num_lots: int
    entry_price: float             # Option premium entry
    target_price: float            # Option premium target
    stoploss_price: float          # Option premium SL
    index_entry: float             # Index level at entry
    index_target: float
    index_stoploss: float
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    pnl: float = 0.0
    pnl_points: float = 0.0
    status: str = "PENDING"
    order_id: Optional[str] = None
    sl_order_id: Optional[str] = None
    target_order_id: Optional[str] = None
    signal_score: int = 0
    signal_confidence: float = 0.0
    signal_reasons: List[str] = field(default_factory=list)
    mode: str = "PAPER"            # "PAPER" or "LIVE"


# ─────────────────────────────────────────────────────────────────────────────
# Dhan API Client
# ─────────────────────────────────────────────────────────────────────────────

class DhanClient:
    """Wrapper around Dhan HTTP API."""

    BASE_URL = "https://api.dhan.co/v2"

    def __init__(self, client_id: str, access_token: str):
        self.client_id = client_id
        self.access_token = access_token
        self.headers = {
            "Content-Type": "application/json",
            "access-token": access_token,
            "client-id": client_id,   # required by Dhan v2
        }

    def _request(self, method: str, endpoint: str, data: dict = None) -> dict:
        """Make API request to Dhan."""
        import requests

        url = f"{self.BASE_URL}{endpoint}"
        try:
            if method == "GET":
                resp = requests.get(url, headers=self.headers, timeout=10)
            elif method == "POST":
                resp = requests.post(url, headers=self.headers,
                                     json=data, timeout=10)
            elif method == "PUT":
                resp = requests.put(url, headers=self.headers,
                                    json=data, timeout=10)
            elif method == "DELETE":
                resp = requests.delete(url, headers=self.headers, timeout=10)
            else:
                return {"error": f"Unknown method: {method}"}

            if resp.status_code in (200, 201):
                return resp.json()
            else:
                logger.error(f"Dhan API error: {resp.status_code} - {resp.text}")
                return {"error": resp.text, "status_code": resp.status_code}
        except Exception as e:
            logger.error(f"Dhan API request failed: {e}")
            return {"error": str(e)}

    def place_order(
        self,
        security_id: str,
        exchange_segment: str,
        transaction_type: str,     # "BUY" or "SELL"
        quantity: int,
        order_type: str = "MARKET",
        price: float = 0,
        trigger_price: float = 0,
        product_type: str = "INTRADAY",
        validity: str = "DAY",
        tag: str = "ScalperPro",
    ) -> dict:
        """Place an order on Dhan."""
        payload = {
            "dhanClientId": self.client_id,
            "transactionType": transaction_type,
            "exchangeSegment": exchange_segment,
            "productType": product_type,
            "orderType": order_type,
            "validity": validity,
            "securityId": security_id,
            "quantity": quantity,
            "price": price,
            "triggerPrice": trigger_price,
            "correlationId": tag,
        }
        return self._request("POST", "/orders", payload)

    def place_bracket_order(
        self,
        security_id: str,
        exchange_segment: str,
        transaction_type: str,
        quantity: int,
        price: float,
        stoploss_value: float,
        target_value: float,
        trailing_sl: float = 0,
    ) -> dict:
        """Place a bracket order (entry + SL + target)."""
        payload = {
            "dhanClientId": self.client_id,
            "transactionType": transaction_type,
            "exchangeSegment": exchange_segment,
            "productType": "BRACKET",
            "orderType": "LIMIT",
            "validity": "DAY",
            "securityId": security_id,
            "quantity": quantity,
            "price": price,
            "stopLoss": stoploss_value,
            "target": target_value,
            "trailingStopLoss": trailing_sl,
        }
        return self._request("POST", "/orders/bracket", payload)

    def modify_order(self, order_id: str, order_type: str = "LIMIT",
                     price: float = 0, trigger_price: float = 0,
                     quantity: int = 0) -> dict:
        payload = {
            "dhanClientId": self.client_id,
            "orderId": order_id,
            "orderType": order_type,
            "price": price,
            "triggerPrice": trigger_price,
            "quantity": quantity,
            "validity": "DAY",
        }
        return self._request("PUT", f"/orders/{order_id}", payload)

    def cancel_order(self, order_id: str) -> dict:
        return self._request("DELETE", f"/orders/{order_id}")

    def get_positions(self) -> dict:
        return self._request("GET", "/positions")

    def get_order_book(self) -> dict:
        return self._request("GET", "/orders")

    def get_holdings(self) -> dict:
        return self._request("GET", "/holdings")

    def get_fund_limits(self) -> dict:
        return self._request("GET", "/fundlimit")


# ─────────────────────────────────────────────────────────────────────────────
# Execution Engine
# ─────────────────────────────────────────────────────────────────────────────

class ExecutionEngine:
    """
    Manages trade execution in both PAPER and LIVE modes.
    Handles entry, SL placement, trailing SL, and exit.
    """

    def __init__(self, mode: str = "PAPER"):
        from scalper.config.settings import (
            DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN, RiskConfig, TradingMode
        )
        self.mode = mode
        self.risk = RiskConfig()
        self.trades: List[TradeRecord] = []
        self.daily_pnl: float = 0.0
        self.daily_trade_count: int = 0
        self._trade_counter = 0

        if mode == "LIVE":
            self.dhan = DhanClient(DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN)
        else:
            self.dhan = None

        self._load_trades()

    def _generate_trade_id(self) -> str:
        self._trade_counter += 1
        return f"SP_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._trade_counter:04d}"

    def can_trade(self) -> tuple:
        """Check if trading is allowed based on risk rules."""
        now = datetime.now()
        current_time = now.strftime("%H:%M")

        # Time checks
        if current_time < self.risk.no_trade_before:
            return False, f"Too early ({current_time} < {self.risk.no_trade_before})"
        if current_time > self.risk.no_trade_after:
            return False, f"Too late ({current_time} > {self.risk.no_trade_after})"

        # Daily loss check
        if self.daily_pnl < -self.risk.max_daily_loss:
            return False, f"Daily loss limit hit (₹{self.daily_pnl:,.0f})"

        # Trade count check
        if self.daily_trade_count >= self.risk.max_daily_trades:
            return False, f"Max daily trades reached ({self.daily_trade_count})"

        # Concurrent positions check
        open_trades = [t for t in self.trades if t.status == "ENTERED"]
        if len(open_trades) >= self.risk.max_concurrent_positions:
            return False, f"Max concurrent positions ({len(open_trades)})"

        # Cooldown after loss
        if self.trades:
            last_trade = self.trades[-1]
            if last_trade.status == "SL_HIT" and last_trade.exit_time:
                last_exit = datetime.fromisoformat(last_trade.exit_time)
                cooldown_end = last_exit + timedelta(
                    seconds=self.risk.cooldown_after_loss_seconds
                )
                if now < cooldown_end:
                    remaining = (cooldown_end - now).seconds
                    return False, f"Cooldown after loss ({remaining}s remaining)"

        return True, "OK"

    def execute_trade(
        self,
        signal,          # TradeSignal from signal_engine
        strike_info: Dict,     # From StrikeSelector
        index_config,          # IndexConfig
    ) -> Optional[TradeRecord]:
        """Execute a trade based on signal and strike selection."""

        can, reason = self.can_trade()
        if not can:
            logger.warning(f"Trade blocked: {reason}")
            return None

        # Calculate lot details
        num_lots = min(
            index_config.max_lots,
            int(self.risk.max_capital_per_trade / (strike_info["ltp"] * index_config.lot_size))
        )
        if num_lots < 1:
            logger.warning(f"Insufficient capital for {index_config.symbol}")
            return None

        quantity = num_lots * index_config.lot_size

        # Calculate option premium SL and target based on delta
        delta = abs(strike_info["delta"])
        premium_target = signal.target_price - signal.entry_price if signal.direction == "LONG" \
            else signal.entry_price - signal.target_price
        premium_target = abs(premium_target) * delta

        premium_sl = abs(signal.entry_price - signal.stoploss_price) * delta

        trade = TradeRecord(
            trade_id=self._generate_trade_id(),
            timestamp=datetime.now().isoformat(),
            index=signal.index,
            strategy=signal.strategy,
            direction=signal.direction,
            option_type=strike_info["option_type"],
            strike=strike_info["strike"],
            lot_size=index_config.lot_size,
            num_lots=num_lots,
            entry_price=strike_info["ltp"],
            target_price=round(strike_info["ltp"] + premium_target, 2),
            stoploss_price=round(max(strike_info["ltp"] - premium_sl, 0.05), 2),
            index_entry=signal.entry_price,
            index_target=signal.target_price,
            index_stoploss=signal.stoploss_price,
            signal_score=signal.score,
            signal_confidence=signal.confidence,
            signal_reasons=signal.reasons,
            mode=self.mode,
        )

        if self.mode == "LIVE" and self.dhan:
            result = self._place_live_order(trade, strike_info, index_config)
            if "error" in result:
                trade.status = "REJECTED"
                logger.error(f"Order rejected: {result['error']}")
            else:
                trade.status = "ENTERED"
                trade.order_id = result.get("orderId", "")
                logger.info(f"LIVE order placed: {trade.trade_id}")
        else:
            # Paper trade - just log it
            trade.status = "ENTERED"
            trade.order_id = f"PAPER_{trade.trade_id}"
            logger.info(f"PAPER trade entered: {trade.trade_id}")

        self.trades.append(trade)
        self.daily_trade_count += 1
        self._save_trades()
        return trade

    def _place_live_order(self, trade: TradeRecord, strike_info: Dict,
                          index_config) -> dict:
        """Place actual order on Dhan."""
        # For options, we need the option's security_id
        # This would need to be resolved from Dhan's instrument master
        # For now, using bracket order approach

        transaction_type = "BUY"  # Always buy options for scalping
        sl_value = round(trade.entry_price - trade.stoploss_price, 2)
        target_value = round(trade.target_price - trade.entry_price, 2)

        exchange = "NSE_FNO"
        if index_config.symbol == "SENSEX":
            exchange = "BSE_FNO"

        return self.dhan.place_bracket_order(
            security_id=str(strike_info.get("security_id", "")),
            exchange_segment=exchange,
            transaction_type=transaction_type,
            quantity=trade.lot_size * trade.num_lots,
            price=trade.entry_price,
            stoploss_value=sl_value,
            target_value=target_value,
            trailing_sl=0,
        )

    def update_trailing_sl(self, trade: TradeRecord, current_premium: float):
        """Update trailing stop loss if price has moved favorably."""
        if trade.status != "ENTERED":
            return

        entry = trade.entry_price
        target = trade.target_price
        sl = trade.stoploss_price
        total_move = abs(target - entry)

        # Activate trailing SL at 50% of target
        activation_price = entry + (total_move * self.risk.trailing_sl_activation)

        if current_premium >= activation_price:
            # Trail SL to lock in profits
            new_sl = current_premium - (total_move * self.risk.trailing_sl_distance)
            if new_sl > trade.stoploss_price:
                old_sl = trade.stoploss_price
                trade.stoploss_price = round(new_sl, 2)
                trade.status = "TRAILING_SL"
                logger.info(
                    f"Trailing SL updated: {trade.trade_id} "
                    f"SL {old_sl} → {trade.stoploss_price}"
                )

                # Update on Dhan if live
                if self.mode == "LIVE" and self.dhan and trade.sl_order_id:
                    self.dhan.modify_order(
                        trade.sl_order_id,
                        trigger_price=trade.stoploss_price
                    )

    def check_exit(self, trade: TradeRecord, current_premium: float) -> bool:
        """Check if trade should be exited (SL or target hit)."""
        if trade.status not in ("ENTERED", "TRAILING_SL"):
            return False

        exited = False

        # Target hit
        if current_premium >= trade.target_price:
            trade.status = "TARGET_HIT"
            trade.exit_price = trade.target_price
            exited = True

        # SL hit
        elif current_premium <= trade.stoploss_price:
            trade.status = "SL_HIT"
            trade.exit_price = trade.stoploss_price
            exited = True

        if exited:
            trade.exit_time = datetime.now().isoformat()
            qty = trade.lot_size * trade.num_lots
            trade.pnl = round((trade.exit_price - trade.entry_price) * qty, 2)
            trade.pnl_points = round(trade.exit_price - trade.entry_price, 2)
            self.daily_pnl += trade.pnl
            self._save_trades()

            logger.info(
                f"Trade exited: {trade.trade_id} | {trade.status} | "
                f"P&L: ₹{trade.pnl:,.2f} ({trade.pnl_points:+.2f} pts)"
            )

        return exited

    def get_daily_summary(self) -> Dict:
        """Get today's trading summary."""
        today = datetime.now().date().isoformat()
        todays_trades = [
            t for t in self.trades
            if t.timestamp.startswith(today)
        ]

        winners = [t for t in todays_trades if t.pnl > 0]
        losers = [t for t in todays_trades if t.pnl < 0]

        return {
            "date": today,
            "total_trades": len(todays_trades),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": round(len(winners) / max(len(todays_trades), 1) * 100, 1),
            "total_pnl": round(sum(t.pnl for t in todays_trades), 2),
            "avg_winner": round(np.mean([t.pnl for t in winners]), 2) if winners else 0,
            "avg_loser": round(np.mean([t.pnl for t in losers]), 2) if losers else 0,
            "best_trade": max((t.pnl for t in todays_trades), default=0),
            "worst_trade": min((t.pnl for t in todays_trades), default=0),
            "open_positions": len([t for t in todays_trades if t.status in ("ENTERED", "TRAILING_SL")]),
        }

    def _save_trades(self):
        """Persist trades to JSON file."""
        from scalper.config.settings import TRADES_DB
        os.makedirs(os.path.dirname(TRADES_DB), exist_ok=True)
        trades_data = [asdict(t) for t in self.trades]
        try:
            with open(TRADES_DB, "w") as f:
                json.dump(trades_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save trades: {e}")

    def _load_trades(self):
        """Load trades from JSON file."""
        from scalper.config.settings import TRADES_DB
        try:
            if os.path.exists(TRADES_DB):
                with open(TRADES_DB, "r") as f:
                    data = json.load(f)
                    self.trades = [TradeRecord(**t) for t in data]
                    # Recalculate daily PnL
                    today = datetime.now().date().isoformat()
                    self.daily_pnl = sum(
                        t.pnl for t in self.trades
                        if t.timestamp.startswith(today)
                    )
                    self.daily_trade_count = len([
                        t for t in self.trades
                        if t.timestamp.startswith(today)
                    ])
        except Exception as e:
            logger.warning(f"Failed to load trades: {e}")
            self.trades = []


# Need numpy for daily_summary
import numpy as np
