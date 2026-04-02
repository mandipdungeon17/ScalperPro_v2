"""
=============================================================================
SCALPER PRO - News Monitor
=============================================================================
Monitors RSS feeds for news that can MOVE THE MARKET — not corporate noise.

What is included:
  • RBI rate decisions, monetary policy
  • US Fed / FOMC decisions, rate changes
  • Inflation data (CPI, WPI), GDP, IIP, PMI
  • Oil price shocks (crude +/-3%)
  • Global geopolitical events (war, sanctions, trade war, tariffs)
  • FII / DII major activity, foreign fund flows
  • India VIX spikes, circuit breakers, trading halts
  • SEBI orders / market regulatory actions
  • US / China / Europe market crashes or rallies
  • Budget, election results, major government policy

What is EXCLUDED (noise):
  • Company dividends, board meetings, AGMs
  • Individual stock results (handled separately by stock monitor)
  • Mergers of obscure companies
  • Random corporate filings
  • NSE/BSE announcements for small/mid-cap stocks

NSE corporate announcements are NOT fetched — pure noise even after filtering.
=============================================================================
"""

import hashlib
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)


# ── Market-moving keywords only ───────────────────────────────────────────────
# Scored 1-10 by expected market impact. Only score >= 5 triggers an alert.
TRIGGER_KEYWORDS: dict = {
    10: [
        "rbi rate cut", "rbi rate hike", "repo rate cut", "repo rate hike",
        "fed rate cut", "fed rate hike", "fomc rate", "emergency rate",
        "market halt", "trading halt", "circuit breaker", "market circuit",
        "war declared", "nuclear", "terror attack india",
        "sebi ban", "sebi order market",
    ],
    9: [
        "rbi policy", "monetary policy", "rbi governor",
        "fomc decision", "fed decision", "federal reserve decision",
        "us inflation", "india inflation", "cpi data", "wpi data",
        "gdp data india", "gdp growth", "india gdp",
        "crude oil crash", "crude oil spike", "brent crude",
        "pakistan tension", "india china border", "us china trade",
        "trade war", "tariff", "us tariff", "trump tariff",
    ],
    8: [
        "rbi", "india vix spike", "vix jumps",
        "fii selling", "fii buying", "foreign outflow", "foreign inflow",
        "rupee falls", "rupee crashes", "usd inr", "dollar rupee",
        "sensex crash", "nifty crash", "market crash", "market falls",
        "sebi circular", "sebi regulation",
        "oil price", "opec", "crude oil",
        "us jobs", "nonfarm payroll", "unemployment data us",
        "china market", "china crash", "china economy",
        "europe market", "european crisis",
    ],
    7: [
        "iip data", "pmi india", "pmi data", "current account",
        "trade deficit india", "foreign reserves", "forex reserves",
        "budget india", "union budget", "interim budget",
        "election result", "exit poll",
        "imf india", "world bank india",
        "fii data", "dii data", "institutional buying", "institutional selling",
        "interest rate", "inflation rate",
        "dow jones falls", "dow jones rises", "nasdaq falls", "s&p 500",
        "global selloff", "global rally",
    ],
    6: [
        "nifty 50", "banknifty", "bank nifty", "india market",
        "stock market today", "dalal street",
        "gift nifty", "sgx nifty",
        "government policy", "pm modi economy",
        "disinvestment", "privatisation",
        "mutual fund flow", "sip inflow",
    ],
    5: [
        "india economy", "economic slowdown", "recession india",
        "monsoon impact", "drought india",
        "power crisis", "fuel price hike", "petrol diesel price",
        "import duty", "export ban",
        "nse", "bse market",
    ],
}

# Noise keywords — headlines containing these are always dropped regardless of score
NOISE_KEYWORDS = [
    "dividend", "agm", "annual general meeting", "board meeting",
    "quarterly results", "q1 results", "q2 results", "q3 results", "q4 results",
    "earnings", "profit", "revenue", "ebitda",
    "buyback", "stock split", "bonus shares",
    "amalgamation", "merger agreement", "acquisition completed",
    "ipo allotment", "ipo listing", "ipo opens",
    "promoter", "insider trading case",
    "corporate action", "record date", "ex-dividend",
    "nse:sme", "nse:emerge",
]

# Build fast lookup list
_KW_MAP: List[tuple] = []
for _score, _kws in TRIGGER_KEYWORDS.items():
    for _kw in _kws:
        _KW_MAP.append((_kw.lower(), _score))

_NOISE_SET = [n.lower() for n in NOISE_KEYWORDS]


# ── Data structures ──────────────────────────────────────────────────────────
@dataclass
class NewsItem:
    title: str
    source: str
    url: str
    published: str
    impact_score: int
    matched_keywords: List[str]
    fetched_at: datetime = field(default_factory=datetime.now)


# ── RSS feeds — only market/macro sources ────────────────────────────────────
RSS_FEEDS = [
    {
        "name": "Economic Times Markets",
        "url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "min_score": 5,
    },
    {
        "name": "Moneycontrol Markets",
        "url": "https://www.moneycontrol.com/rss/marketreports.xml",
        "min_score": 5,
    },
    {
        "name": "LiveMint Economy",
        "url": "https://www.livemint.com/rss/economy",
        "min_score": 6,
    },
    {
        "name": "Reuters India Business",
        "url": "https://feeds.reuters.com/reuters/INbusinessNews",
        "min_score": 6,
    },
    {
        "name": "Business Standard Economy",
        "url": "https://www.business-standard.com/rss/economy-policy-10601.rss",
        "min_score": 6,
    },
    {
        "name": "CNBC TV18 Markets",
        "url": "https://www.cnbctv18.com/commonfeeds/v1/eng/rss/market.xml",
        "min_score": 5,
    },
]


class NewsMonitor:
    """
    Polls RSS feeds for MACRO / market-moving news only.
    No NSE announcements, no corporate results, no dividends.
    Returns only new unseen items with impact_score >= min_impact.
    """

    def __init__(self, min_impact: int = 5):
        self.min_impact = min_impact
        self._seen: dict = {}
        self._last_purge = datetime.now()

    # ── Public ───────────────────────────────────────────────────────────────

    def check(self) -> List[NewsItem]:
        """
        Fetch all RSS feeds and return NEW market-moving headlines.
        Call every 5 minutes from the orchestrator.
        """
        self._purge_old()
        new_items: List[NewsItem] = []

        for feed in RSS_FEEDS:
            try:
                items = self._fetch_rss(feed["name"], feed["url"], feed["min_score"])
                new_items.extend(items)
            except Exception as e:
                logger.debug(f"[NewsMonitor] RSS error {feed['name']}: {e}")

        new_items.sort(key=lambda x: x.impact_score, reverse=True)
        return new_items

    # ── Private ──────────────────────────────────────────────────────────────

    def _is_noise(self, title: str) -> bool:
        """Return True if headline is corporate/company noise, not market-moving."""
        lower = title.lower()
        return any(noise in lower for noise in _NOISE_SET)

    def _score_title(self, title: str) -> tuple:
        lower = title.lower()
        matched, max_score = [], 0
        for kw, score in _KW_MAP:
            if kw in lower:
                matched.append(kw)
                if score > max_score:
                    max_score = score
        return max_score, matched

    def _hash(self, title: str) -> str:
        return hashlib.md5(title.strip().lower().encode()).hexdigest()

    def _is_new(self, title: str) -> bool:
        h = self._hash(title)
        if h in self._seen:
            return False
        self._seen[h] = datetime.now()
        return True

    def _fetch_rss(self, source: str, url: str, min_score: int) -> List[NewsItem]:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return []

        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError:
            return []

        items: List[NewsItem] = []
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            link  = (item.findtext("link")  or "").strip()
            pub   = (item.findtext("pubDate") or "").strip()

            if not title:
                continue

            # Drop corporate/company noise immediately
            if self._is_noise(title):
                continue

            score, matched = self._score_title(title)
            threshold = max(self.min_impact, min_score)
            if score < threshold:
                continue

            if not self._is_new(title):
                continue

            items.append(NewsItem(
                title=title,
                source=source,
                url=link,
                published=pub,
                impact_score=score,
                matched_keywords=matched[:4],
            ))

        return items

    def _purge_old(self):
        """Purge seen entries older than 24 hours (runs every 30 min)."""
        now = datetime.now()
        if (now - self._last_purge).total_seconds() < 1800:
            return
        cutoff = now - timedelta(hours=24)
        self._seen = {h: t for h, t in self._seen.items() if t > cutoff}
        self._last_purge = now
