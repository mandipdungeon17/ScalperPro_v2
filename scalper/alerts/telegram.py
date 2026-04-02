"""
=============================================================================
SCALPER PRO - Telegram Alert System
=============================================================================
Sends formatted alerts to Telegram.

Design rules applied to every message:
  • One clear header line with emoji + type + index
  • Sections separated by a thin divider line
  • Numbers in <code> blocks when alignment matters
  • Timestamp always on the last line, right-aligned
  • No redundant labels; emojis carry the sentiment at a glance
  • 4096-char Telegram limit respected with graceful truncation
=============================================================================
"""

import logging
import requests
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Divider styles
_DIV  = "─────────────────────────"   # section divider (inside code block)
_HDIV = "━━━━━━━━━━━━━━━━━━━━━━━━━"  # heavy divider for headers


def _ts() -> str:
    """Current time stamp string."""
    return datetime.now().strftime("%d %b %Y  %H:%M IST")


def _pnl_bar(pnl: float, width: int = 10) -> str:
    """Mini text bar showing P&L direction."""
    if pnl > 0:
        filled = min(int(pnl / 500), width)
        return "▓" * filled + "░" * (width - filled) + "  +"
    else:
        filled = min(int(abs(pnl) / 500), width)
        return "░" * (width - filled) + "▓" * filled + "  -"


class TelegramAlerts:
    """Sends all trading alerts to Telegram via Bot API."""

    API_URL = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, bot_token: str = None, chat_id: str = None):
        from scalper.config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
        self.bot_token = bot_token or TELEGRAM_BOT_TOKEN
        self.chat_id   = chat_id   or TELEGRAM_CHAT_ID

    # ── Core sender ──────────────────────────────────────────────────────────
    def _send(self, text: str, parse_mode: str = "HTML") -> bool:
        if len(text) > 4096:
            text = text[:4050] + "\n\n<i>— message truncated —</i>"
        try:
            url = self.API_URL.format(token=self.bot_token, method="sendMessage")
            resp = requests.post(url, json={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            }, timeout=10)
            if resp.status_code == 200:
                return True
            logger.error(f"Telegram {resp.status_code}: {resp.text[:200]}")
            return False
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False

    # ════════════════════════════════════════════════════════════════
    # SYSTEM STATUS
    # ════════════════════════════════════════════════════════════════
    def send_system_status(self, status: str, details: str = "") -> bool:
        icons = {
            "STARTED": "🚀", "STOPPED": "🔴", "ERROR": "🚨",
            "WARNING": "⚠️", "INFO": "💬",
        }
        icon = icons.get(status, "📡")
        detail_block = f"\n{details}\n" if details else "\n"
        msg = (
            f"{icon}  <b>ScalperPro  ·  {status}</b>\n"
            f"<code>{_HDIV}</code>\n"
            f"{detail_block}"
            f"<code>{_ts()}</code>"
        )
        return self._send(msg)

    # ════════════════════════════════════════════════════════════════
    # TRADE SIGNAL  (pre-execution signal card)
    # ════════════════════════════════════════════════════════════════
    def send_signal_alert(self, signal) -> bool:
        d_icon  = "🟢" if signal.direction == "LONG" else "🔴"
        s_icon  = "⚡" if signal.strategy == "scalp" else "🎯"
        vix_row = f"\nVIX        {signal.vix_value:.1f}" if signal.vix_value else ""

        reasons = "\n".join(f"  · {r}" for r in signal.reasons[:5])

        msg = (
            f"{s_icon}  <b>{signal.strategy.upper()} SIGNAL  ·  {signal.index}</b>  {d_icon}\n"
            f"<code>{_HDIV}</code>\n"
            f"<code>"
            f"Entry      {signal.entry_price:>10,.2f}\n"
            f"Target     {signal.target_price:>10,.2f}   +{abs(signal.target_price - signal.entry_price):.0f} pts\n"
            f"Stop Loss  {signal.stoploss_price:>10,.2f}   -{abs(signal.stoploss_price - signal.entry_price):.0f} pts\n"
            f"R : R      {'1 : ' + str(round(signal.risk_reward, 1)):>10}\n"
            f"Score      {signal.score}/{signal.max_score}  ({signal.confidence*100:.0f}% conf)\n"
            f"ATR        {signal.atr_value:>10.1f}{vix_row}"
            f"</code>\n"
            f"\n<b>Reasons</b>\n{reasons}\n"
            f"\n<code>{_ts()}</code>"
        )
        return self._send(msg)

    # ════════════════════════════════════════════════════════════════
    # TRADE ENTRY
    # ════════════════════════════════════════════════════════════════
    def send_trade_entry(self, trade) -> bool:
        mode_icon = "📝" if trade.mode == "PAPER" else "💰"
        d_icon    = "🟢" if trade.direction == "LONG" else "🔴"
        qty       = trade.num_lots * trade.lot_size
        capital   = trade.entry_price * qty
        max_risk  = abs(trade.entry_price - trade.stoploss_price) * qty

        msg = (
            f"{mode_icon}  <b>TRADE ENTERED  ·  {trade.index}</b>  {d_icon}\n"
            f"<code>{_HDIV}</code>\n"
            f"<code>"
            f"Strike     {trade.strike} {trade.option_type}  ({trade.strategy.upper()})\n"
            f"Qty        {trade.num_lots} lots × {trade.lot_size} = {qty}\n"
            f"{_DIV}\n"
            f"Entry      ₹{trade.entry_price:>8.2f}\n"
            f"Target     ₹{trade.target_price:>8.2f}\n"
            f"Stop Loss  ₹{trade.stoploss_price:>8.2f}\n"
            f"{_DIV}\n"
            f"Capital    ₹{capital:>8,.0f}\n"
            f"Max Risk   ₹{max_risk:>8,.0f}\n"
            f"Conf.       {trade.signal_confidence*100:.0f}%"
            f"</code>\n"
            f"\n<code>{_ts()}</code>"
        )
        return self._send(msg)

    # ════════════════════════════════════════════════════════════════
    # TRADE EXIT
    # ════════════════════════════════════════════════════════════════
    def send_trade_exit(self, trade) -> bool:
        status_map = {
            "TARGET_HIT":   ("🎯", "TARGET HIT"),
            "SL_HIT":       ("🛑", "STOP LOSS HIT"),
            "TRAILING_SL":  ("📉", "TRAILING SL"),
            "EXITED":       ("🔄", "MANUAL EXIT"),
        }
        icon, label = status_map.get(trade.status, ("📤", trade.status))
        pnl_icon    = "✅" if trade.pnl >= 0 else "❌"
        mode_tag    = "PAPER" if trade.mode == "PAPER" else "LIVE"

        msg = (
            f"{icon}  <b>{label}  ·  {trade.index}</b>  {pnl_icon}\n"
            f"<code>{_HDIV}</code>\n"
            f"<code>"
            f"Strike     {trade.strike} {trade.option_type}\n"
            f"Mode       {mode_tag}\n"
            f"{_DIV}\n"
            f"Entry      ₹{trade.entry_price:>8.2f}\n"
            f"Exit       ₹{trade.exit_price:>8.2f}\n"
            f"Points     {trade.pnl_points:>+8.2f}\n"
            f"{_DIV}\n"
            f"P & L      ₹{trade.pnl:>+8,.2f}"
            f"</code>\n"
            f"\n<code>{_ts()}</code>"
        )
        return self._send(msg)

    # ════════════════════════════════════════════════════════════════
    # DAILY SUMMARY
    # ════════════════════════════════════════════════════════════════
    def send_daily_summary(self, summary: Dict) -> bool:
        pnl      = summary["total_pnl"]
        pnl_icon = "📈" if pnl > 0 else "📉" if pnl < 0 else "➡️"
        win_pct  = summary["win_rate"]
        bar      = "▓" * int(win_pct / 10) + "░" * (10 - int(win_pct / 10))

        msg = (
            f"{pnl_icon}  <b>DAILY WRAP-UP  ·  {summary['date']}</b>\n"
            f"<code>{_HDIV}</code>\n"
            f"<code>"
            f"Trades     {summary['total_trades']:>4}   "
            f"✅ {summary['winners']}  ❌ {summary['losers']}\n"
            f"Win Rate   {win_pct:>4.1f}%  [{bar}]\n"
            f"{_DIV}\n"
            f"Best       ₹{summary['best_trade']:>+9,.2f}\n"
            f"Worst      ₹{summary['worst_trade']:>+9,.2f}\n"
            f"Avg Win    ₹{summary['avg_winner']:>+9,.2f}\n"
            f"Avg Loss   ₹{summary['avg_loser']:>+9,.2f}\n"
            f"{_DIV}\n"
            f"Net P&L    ₹{pnl:>+9,.2f}\n"
            f"Open       {summary['open_positions']} position(s)"
            f"</code>\n"
            f"\n<code>{_ts()}</code>"
        )
        return self._send(msg)

    # ════════════════════════════════════════════════════════════════
    # BACKTEST SUMMARY
    # ════════════════════════════════════════════════════════════════
    def send_backtest_summary(self, results: Dict) -> bool:
        wr   = results.get("win_rate", 0)
        bar  = "▓" * int(wr / 10) + "░" * (10 - int(wr / 10))
        idx  = ", ".join(results.get("indices", []))

        msg = (
            f"🔬  <b>BACKTEST RESULTS</b>\n"
            f"<code>{_HDIV}</code>\n"
            f"<code>"
            f"Indices    {idx}\n"
            f"Period     {results.get('start_date','?')}  →  {results.get('end_date','?')}\n"
            f"{_DIV}\n"
            f"Trades     {results.get('total_trades', 0):>6}\n"
            f"Win Rate   {wr:>5.1f}%  [{bar}]\n"
            f"Pft Factor {results.get('profit_factor', 0):>6.2f}\n"
            f"Sharpe     {results.get('sharpe_ratio', 0):>6.2f}\n"
            f"{_DIV}\n"
            f"Net P&L    ₹{results.get('total_pnl', 0):>+10,.0f}\n"
            f"Max DD     ₹{results.get('max_drawdown', 0):>10,.0f}\n"
            f"Avg Trade  ₹{results.get('avg_pnl', 0):>+10,.0f}\n"
            f"{_DIV}\n"
            f"Best Month ₹{results.get('best_month_pnl', 0):>+10,.0f}\n"
            f"Worst Mth  ₹{results.get('worst_month_pnl', 0):>+10,.0f}\n"
            f"Win Streak {results.get('max_win_streak', 0):>3}  "
            f"Loss Streak {results.get('max_loss_streak', 0):>3}"
            f"</code>\n"
            f"\n<code>{_ts()}</code>"
        )
        return self._send(msg)

    # ════════════════════════════════════════════════════════════════
    # PRE-MARKET ANALYSIS  (fires once at 09:10 AM)
    # ════════════════════════════════════════════════════════════════
    def send_premarket_analysis(self, report) -> bool:
        bias_icon = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(report.bias, "⚪")
        st_arrow  = {1: "▲", -1: "▼", 0: "─"}
        ema_tag   = lambda t: "↑" if t.price_vs_ema20 == "ABOVE" else "↓"
        tr_icon   = {"BULLISH": "↑", "BEARISH": "↓", "NEUTRAL": "─"}

        # ── Timeframe trend table ──────────────────────────────────
        tf_rows = []
        for t in report.tf_trends:
            tf_rows.append(
                f"{t.timeframe:<7} "
                f"{tr_icon.get(t.trend,'─')} {t.trend:<8}  "
                f"EMA {t.ema20:>8,.0f} {ema_tag(t)}  "
                f"ST{st_arrow.get(t.supertrend_dir, '─')}"
            )
        trend_block = "\n".join(tf_rows) if tf_rows else "  No data"

        # ── Key levels table ───────────────────────────────────────
        level_rows = []
        for lv in report.key_levels[:8]:
            side  = "SUP" if lv.level_type == "SUPPORT" else "RES"
            stars = "★" * min(int(lv.strength * 5), 5)
            level_rows.append(
                f"[{side}] {lv.price:>7,.0f}  "
                f"{lv.distance_atr:>4.1f}ATR  "
                f"{stars:<5}  {lv.tags}"
            )
        levels_block = "\n".join(level_rows) if level_rows else "  None within 2 ATR"

        # ── CPR block ──────────────────────────────────────────────
        cpr_width = abs(report.cpr_top - report.cpr_bottom)
        cpr_type  = "Narrow → Range day" if cpr_width < report.daily_atr * 0.3 else "Wide → Trend day"
        cpr_block = (
            f"R2 {report.cpr_r2:>7,.0f}   R1 {report.cpr_r1:>7,.0f}\n"
            f"Pivot      {report.cpr_pivot:>7,.0f}\n"
            f"CPR  {report.cpr_top:>7,.0f} ── {report.cpr_bottom:>7,.0f}  {cpr_type}\n"
            f"S1 {report.cpr_s1:>7,.0f}   S2 {report.cpr_s2:>7,.0f}"
        )

        # ── Nearest zones ──────────────────────────────────────────
        sup_txt = (
            f"{report.nearest_support.price:,.0f}  "
            f"({report.nearest_support.distance_atr:.1f} ATR below)  "
            f"{report.nearest_support.tags}"
            if report.nearest_support else "None within range"
        )
        res_txt = (
            f"{report.nearest_resistance.price:,.0f}  "
            f"({report.nearest_resistance.distance_atr:.1f} ATR above)  "
            f"{report.nearest_resistance.tags}"
            if report.nearest_resistance else "None within range"
        )

        # ── Bias reasons ───────────────────────────────────────────
        reasons = "\n".join(f"  · {r}" for r in report.bias_reasons)

        msg = (
            f"{bias_icon}  <b>PRE-MARKET  ·  {report.index}</b>\n"
            f"<i>LTP {report.current_price:,.0f}  ·  ATR {report.daily_atr:.0f} pts  ·  {report.generated_at}</i>\n"
            f"\n"

            f"<b>TREND  ·  {report.bullish_tfs} Bull / {report.bearish_tfs} Bear / 5 TFs</b>\n"
            f"<code>{trend_block}</code>\n"
            f"\n"

            f"<b>KEY LEVELS</b>\n"
            f"<code>{levels_block}</code>\n"
            f"\n"

            f"<b>CPR  ·  {cpr_type}</b>\n"
            f"<code>{cpr_block}</code>\n"
            f"\n"

            f"<b>NEAREST ZONES</b>\n"
            f"<code>"
            f"Support   {sup_txt}\n"
            f"Resist    {res_txt}"
            f"</code>\n"
            f"\n"

            f"<b>BIAS  ·  {report.bias}  {report.bias_score:+d} / 5</b>\n"
            f"{reasons}\n"
            f"\n"

            f"<b>WATCH TODAY</b>\n"
            f"<code>"
            f"CE entry  {report.watch_buy_zone}\n"
            f"PE entry  {report.watch_sell_zone}"
            f"</code>\n"
            f"\n"
            f"<code>{_ts()}</code>"
        )

        if len(msg) > 4050:
            msg = msg[:4020] + "\n\n<i>— truncated —</i>"

        return self._send(msg)

    # ════════════════════════════════════════════════════════════════
    # EMA + SUPERTREND CONFLUENCE AT SUPPORT
    # ════════════════════════════════════════════════════════════════
    def send_ema_supertrend_support_alert(
        self,
        index: str,
        strike: int,
        option_type: str,
        support_level: float,
        supertrend_val: float,
        supertrend_dir: int,
        ema_name: str,
        ema_val: float,
        current_premium: float,
        entry_premium: float,
        sl_premium: float,
        target_premium: float,
        setup_grade: str = "",
    ) -> bool:
        st_label = "Bullish ▲" if supertrend_dir == 1 else "Bearish ▼"
        rr = round(
            (target_premium - entry_premium) / max(entry_premium - sl_premium, 0.01), 1
        )
        grade_txt = f"  Grade {setup_grade}" if setup_grade else ""

        msg = (
            f"🔶  <b>CONFLUENCE  ·  {index}  ·  {strike} {option_type}</b>{grade_txt}\n"
            f"<i>EMA + Supertrend stacked at swing support — high-probability bounce</i>\n"
            f"\n"
            f"<code>"
            f"Support       {support_level:>10,.2f}\n"
            f"Supertrend    {supertrend_val:>10,.2f}   [{st_label}]\n"
            f"{ema_name:<13} {ema_val:>10,.2f}\n"
            f"{_DIV}\n"
            f"Current prem  {current_premium:>10,.2f}\n"
            f"Entry         {entry_premium:>10,.2f}\n"
            f"Stop Loss     {sl_premium:>10,.2f}\n"
            f"Target        {target_premium:>10,.2f}   R:R 1:{rr}"
            f"</code>\n"
            f"\n<i>Informational — wait for price to reach entry before acting.</i>\n"
            f"<code>{_ts()}</code>"
        )
        return self._send(msg)

    # ════════════════════════════════════════════════════════════════
    # STOCK DIGEST  (one message per index per 5-min cycle)
    # ════════════════════════════════════════════════════════════════
    def send_stock_digest(self, digest) -> bool:
        """
        Single consolidated stock digest — replaces per-stock spam.
        digest: StockDigest from market/stock_monitor.py
        """
        idx = digest.index_name
        ad  = digest.advance_count
        dc  = digest.decline_count
        uc  = digest.unchanged_count
        bar = "▲" * min(ad, 10) + "▼" * min(dc, 10)

        # Gainers block
        gainer_rows = []
        for r in digest.top_gainers[:5]:
            gainer_rows.append(f"  {r.symbol:<14} {r.ltp:>9,.2f}   +{r.change_pct:.2f}%")
        gainer_block = "\n".join(gainer_rows) if gainer_rows else "  —"

        # Losers block
        loser_rows = []
        for r in digest.top_losers[:5]:
            loser_rows.append(f"  {r.symbol:<14} {r.ltp:>9,.2f}   {r.change_pct:.2f}%")
        loser_block = "\n".join(loser_rows) if loser_rows else "  —"

        # Volume spike block (only if any)
        spike_section = ""
        if digest.volume_spikes:
            spike_rows = []
            for r in digest.volume_spikes[:5]:
                spike_rows.append(
                    f"  {r.symbol:<14} {r.change_pct:>+6.2f}%   {r.volume_ratio:.1f}x vol"
                )
            spike_section = (
                f"\n🌊 <b>Volume Spikes</b>\n"
                f"<code>{'chr(10)'.join(spike_rows)}</code>\n"
            )
            # fix the chr(10) join — build it properly
            spike_section = (
                f"\n🌊 <b>Volume Spikes</b>\n"
                f"<code>" + "\n".join(spike_rows) + "</code>\n"
            )

        msg = (
            f"📊  <b>MARKET PULSE  ·  {idx}</b>\n"
            f"<code>▲ {ad}  ▼ {dc}  — {uc}    {bar}</code>\n"
            f"\n"
            f"📈 <b>Top Gainers</b>\n"
            f"<code>{gainer_block}</code>\n"
            f"\n"
            f"📉 <b>Top Losers</b>\n"
            f"<code>{loser_block}</code>\n"
            f"{spike_section}"
            f"\n<code>{_ts()}</code>"
        )
        return self._send(msg)

    # ════════════════════════════════════════════════════════════════
    # GLOBAL MARKET PULSE  (GIFT Nifty + futures)
    # ════════════════════════════════════════════════════════════════
    def send_global_pulse(self, report) -> bool:
        lines: List[str] = []

        # GIFT Nifty header row
        if report.gift_nifty:
            g = report.gift_nifty
            gap_icon = {"GAP_UP": "🟢", "GAP_DOWN": "🔴", "FLAT": "🟡"}.get(
                g.gap_direction, "⚪"
            )
            lines.append(
                f"GIFT Nifty  {g.price:>8,.0f}   {g.change_pct:>+6.2f}%   {g.gap_direction}"
            )
            if report.nifty_implied_open:
                lines.append(
                    f"Implied open        ~{report.nifty_implied_open:,.0f}"
                )
            lines.append(_DIV)

        # Group by category
        groups: Dict[str, list] = {}
        for snap in report.snapshots:
            if snap.is_significant:
                groups.setdefault(snap.group, []).append(snap)

        group_labels = {
            "US_FUTURES": "US Futures",
            "EUROPE":     "Europe",
            "COMMODITY":  "Commodities",
            "FOREX":      "Forex",
        }
        for grp, snaps in groups.items():
            lines.append(group_labels.get(grp, grp))
            for s in snaps:
                arrow = "▲" if s.change_pct >= 0 else "▼"
                lines.append(
                    f"  {s.name:<18} {s.price:>10,.2f}   {arrow} {s.change_pct:>+.2f}%"
                )

        if not lines:
            return True  # nothing to report

        # Find headline icon
        gap = report.gift_nifty.gap_direction if report.gift_nifty else "FLAT"
        head_icon = {"GAP_UP": "🟢", "GAP_DOWN": "🔴", "FLAT": "🌐"}.get(gap, "🌐")

        msg = (
            f"{head_icon}  <b>GLOBAL PULSE</b>\n"
            f"<code>"
            + "\n".join(lines)
            + f"</code>\n"
            f"\n<code>{_ts()}</code>"
        )

        if len(msg) > 4050:
            msg = msg[:4020] + "\n\n<i>— truncated —</i>"

        return self._send(msg)

    # ════════════════════════════════════════════════════════════════
    # NEWS ALERT
    # ════════════════════════════════════════════════════════════════
    def send_news_alert(self, item) -> bool:
        score = item.impact_score
        if score >= 9:
            icon, label = "🚨", "URGENT NEWS"
        elif score >= 7:
            icon, label = "⚠️", "HIGH IMPACT"
        elif score >= 5:
            icon, label = "📰", "NEWS ALERT"
        else:
            icon, label = "ℹ️", "FYI"

        kw_line = ""
        if item.matched_keywords:
            kw_line = f"\n<i>Keywords: {', '.join(item.matched_keywords[:4])}</i>"

        msg = (
            f"{icon}  <b>{label}  ·  Impact {score}/10</b>\n"
            f"\n"
            f"{item.title}\n"
            f"\n"
            f"<i>Source: {item.source}</i>{kw_line}\n"
            f"<code>{_ts()}</code>"
        )
        return self._send(msg)

    # ════════════════════════════════════════════════════════════════
    # OI CHANGE ALERT
    # ════════════════════════════════════════════════════════════════
    def send_oi_alert(self, alert) -> bool:
        type_icon = {
            "PCR_SHIFT":    "🔄",
            "MAX_PAIN_MOVE": "🎯",
            "OI_SPURT":     "🌊",
            "OI_BUILDUP":   "🧱",
            "OI_UNWIND":    "💧",
            "SIGNAL_FLIP":  "🔀",
        }.get(alert.alert_type, "📊")

        snap     = alert.current_snapshot
        sig_icon = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(snap.signal, "⚪")
        top_ce   = snap.ce_resistance_levels[0]["strike"] if snap.ce_resistance_levels else "N/A"
        top_pe   = snap.pe_support_levels[0]["strike"] if snap.pe_support_levels else "N/A"
        label    = alert.alert_type.replace("_", " ")

        msg = (
            f"{type_icon}  <b>OI  ·  {label}  ·  {alert.index}</b>\n"
            f"\n"
            f"{alert.description}\n"
            f"\n"
            f"<code>"
            f"Spot       {snap.spot_price:>9,.0f}\n"
            f"PCR        {snap.pcr:>9.2f}\n"
            f"Max Pain   {snap.max_pain:>9,.0f}\n"
            f"CE Wall    {str(top_ce):>9}   PE Floor  {top_pe}\n"
            f"Range   {snap.expected_range[0]:>9,.0f} – {snap.expected_range[1]:>9,.0f}\n"
            f"Signal  {sig_icon} {snap.signal}  ({snap.signal_strength:.0%})"
            f"</code>\n"
            f"\n<code>{_ts()}</code>"
        )
        return self._send(msg)

    # ════════════════════════════════════════════════════════════════
    # HELPERS
    # ════════════════════════════════════════════════════════════════
    @staticmethod
    def _format_reasons(reasons: list) -> str:
        return "\n".join(f"  · {r}" for r in reasons)
