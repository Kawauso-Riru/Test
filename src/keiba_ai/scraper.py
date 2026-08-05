"""Network layer for fetching netkeiba-style pages.

IMPORTANT: scraping is subject to the target site's Terms of Use and
robots.txt. Before pointing this at any live site: read its ToS, keep
request volume low (this defaults to one request every 2 seconds), cache
responses so you never re-fetch the same page, and identify your bot with a
real contact string in the User-Agent. This module refuses to fetch a URL
that robots.txt disallows unless you explicitly opt out.
"""
from __future__ import annotations

import time
import urllib.robotparser as robotparser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

from .parser import parse_race_result_html, parse_shutuba_html

DEFAULT_USER_AGENT = "keiba-ai-research-bot/0.1 (+contact: set-your-email-here)"


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
                # If robots.txt is unreachable, fail closed on the side of caution
                # by leaving the parser empty (can_fetch defaults to True for an
                # empty ruleset) -- callers should still keep volume low.
                pass
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
        return parse_shutuba_html(self.fetch(url))


def re_sub_safe(url: str) -> str:
    return url.replace("://", "_").replace("/", "_").replace("?", "_").replace("&", "_")
