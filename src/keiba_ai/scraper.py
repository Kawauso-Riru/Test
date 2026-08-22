"""Network layer for fetching netkeiba-style pages.

IMPORTANT: scraping is subject to the target site's Terms of Use and
robots.txt. Before pointing this at any live site: read its ToS, keep
request volume low (this defaults to one request every 2 seconds), cache
responses so you never re-fetch the same page, and identify your bot with a
real contact string in the User-Agent. This module refuses to fetch a URL
that robots.txt disallows unless you explicitly opt out.
"""
from __future__ import annotations

import re
import time
import urllib.robotparser as robotparser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from .parser import parse_oikiri_html, parse_race_result_html, parse_shutuba_html

DEFAULT_USER_AGENT = "keiba-ai-research-bot/0.1 (+contact: set-your-email-here)"

# The 10 official JRA (中央競馬) courses, keyed by the 2-digit place code that
# appears in netkeiba race_ids (chars [4:6] of the 12-digit id). Any other
# code belongs to a regional/NAR (地方競馬) meeting.
JRA_PLACE_CODES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}

_RACE_ID_PATTERN = re.compile(r"/race/(\d{12})/")


def is_jra_race_id(race_id: str) -> bool:
    """True if race_id belongs to one of the 10 JRA (central racing) courses."""
    return len(race_id) == 12 and race_id[4:6] in JRA_PLACE_CODES


class RobotsDisallowedError(RuntimeError):
    """Raised when robots.txt disallows fetching a URL and respect_robots=True."""


@dataclass
class ScraperConfig:
    user_agent: str = DEFAULT_USER_AGENT
    min_interval_sec: float = 2.0
    timeout_sec: float = 10.0
    cache_dir: Optional[Path] = None
    respect_robots: bool = True


@dataclass
class PoliteScraper:
    """Thin requests wrapper enforcing robots.txt + a minimum request interval."""

    config: ScraperConfig = field(default_factory=ScraperConfig)

    def __post_init__(self) -> None:
        self._last_request_at = 0.0
        self._robots_cache: dict = {}
        if self.config.cache_dir:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def _robots_for(self, url: str) -> robotparser.RobotFileParser:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._robots_cache:
            rp = robotparser.RobotFileParser()
            rp.set_url(f"{origin}/robots.txt")
            try:
                rp.read()
            except Exception:
                # robots.txt itself couldn't be fetched (network hiccup, TLS
                # interception, timeout, ...) -- NOT the same as robots.txt
                # explicitly disallowing us. RobotFileParser.can_fetch()
                # actually returns False by default when read() never
                # completed (last_checked stays 0), which would silently
                # block every request forever. Force allow_all so a transient
                # robots.txt fetch failure doesn't masquerade as a real
                # disallow; callers should still keep volume low.
                rp.allow_all = True
            self._robots_cache[origin] = rp
        return self._robots_cache[origin]

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        wait = self.config.min_interval_sec - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()

    def _cache_path(self, url: str) -> Optional[Path]:
        if not self.config.cache_dir:
            return None
        safe_name = re_sub_safe(url)
        return self.config.cache_dir / f"{safe_name}.html"

    def _invalidate_cache(self, url: str) -> None:
        """Remove a cached page so the next fetch() re-hits the network.

        Used for pages whose content can genuinely change over time before
        settling (an upcoming race's shutuba/oikiri page, filled in
        gradually as the racing office finalizes it) -- caching those
        unconditionally would mean a shutuba checked once before entries
        were published stays cached as "0 entries" forever, even days
        later once the real card is up. Past-race result pages don't need
        this: once posted, a result never changes."""
        cache_path = self._cache_path(url)
        if cache_path and cache_path.exists():
            cache_path.unlink()

    def fetch(self, url: str) -> str:
        cache_path = self._cache_path(url)
        if cache_path and cache_path.exists():
            return cache_path.read_text(encoding="utf-8")

        if self.config.respect_robots:
            rp = self._robots_for(url)
            if not rp.can_fetch(self.config.user_agent, url):
                raise RobotsDisallowedError(f"robots.txt disallows fetching {url}")

        self._throttle()
        resp = requests.get(url, headers={"User-Agent": self.config.user_agent}, timeout=self.config.timeout_sec)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding
        html = resp.text

        if cache_path:
            cache_path.write_text(html, encoding="utf-8")
        return html

    def fetch_race_result(self, url: str) -> dict:
        return parse_race_result_html(self.fetch(url))

    def fetch_shutuba(self, url: str) -> dict:
        parsed = parse_shutuba_html(self.fetch(url))
        if not parsed["entries"]:
            # Card not finalized yet as of this fetch -- don't let the empty
            # snapshot stick around and block a later retry from seeing the
            # real entries once they're published (see _invalidate_cache).
            self._invalidate_cache(url)
        return parsed

    def fetch_oikiri(self, url: str) -> list:
        entries = parse_oikiri_html(self.fetch(url))
        if not entries:
            self._invalidate_cache(url)
        return entries

    @staticmethod
    def oikiri_url(race_id: str) -> str:
        # &type=3 ("評価のみ") is the free, full-field evaluation-letter view.
        # &type=2 ("最終追切のみ") additionally has raw lap times, but only
        # for a ~3-horse free preview per race -- the rest is paywalled
        # behind a netkeiba premium membership ("続きを見る" links to
        # regist.netkeiba.com/?pid=premium), so it isn't used here: a
        # feature that's only ever populated for a curated preview subset
        # would be both unreliably sparse and systematically biased toward
        # whichever horses netkeiba chose to feature.
        return f"https://race.netkeiba.com/race/oikiri.html?race_id={race_id}&type=3"

    def list_race_ids_for_date(self, date: str) -> list:
        """date: 'YYYYMMDD'. Race ids on that day's db.netkeiba.com list page --
        JRA and regional/NAR meetings mixed together. Filter with
        is_jra_race_id() if you only want the 10 JRA courses. Returns an empty
        list for days with no racing at all (list page has no race links)."""
        html = self.fetch(f"https://db.netkeiba.com/race/list/{date}/")
        return sorted(set(_RACE_ID_PATTERN.findall(html)))

    @staticmethod
    def race_result_url(race_id: str) -> str:
        return f"https://db.netkeiba.com/race/{race_id}/"

    def list_upcoming_races_for_date(self, date: str) -> list:
        """date: 'YYYYMMDD'. Unlike list_race_ids_for_date (db.netkeiba.com,
        a *results* database -- empty for races that haven't run yet), this
        hits race.netkeiba.com's live schedule endpoint, which lists races
        for the current/upcoming meeting once the racing office has
        published the card (typically a few days ahead of race day, not
        further out than that).

        Returns [{"race_id": ..., "place": ...}, ...], JRA and regional/NAR
        meetings mixed together -- filter with is_jra_race_id() for JRA-only.
        Empty list if no card is published for that date yet.
        """
        html = self.fetch(f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date}")
        soup = BeautifulSoup(html, "lxml")
        races = []
        for dl in soup.find_all("dl", class_="RaceList_DataList"):
            title_tag = dl.find(class_="RaceList_DataTitle")
            title_text = title_tag.get_text(" ", strip=True) if title_tag else ""
            place_match = re.search(r"\d+回\s*(\S+?)\s*\d+日目", title_text)
            place = place_match.group(1) if place_match else ""
            for race_id in sorted(set(re.findall(r"race_id=(\d{12})", str(dl)))):
                races.append({"race_id": race_id, "place": place})
        return races

    @staticmethod
    def shutuba_url(race_id: str) -> str:
        return f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"


def re_sub_safe(url: str) -> str:
    return url.replace("://", "_").replace("/", "_").replace("?", "_").replace("&", "_")
