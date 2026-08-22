from pathlib import Path

import urllib.robotparser as robotparser

from keiba_ai.scraper import PoliteScraper, ScraperConfig, is_jra_race_id

FIXTURES = Path(__file__).parent / "fixtures"


def test_fetch_shutuba_does_not_permanently_cache_an_unpublished_card(monkeypatch, tmp_path):
    """A shutuba page checked before the racing office has finalized the
    entry list comes back with 0 entries -- caching that response like any
    other page would mean a later re-check, once the real card is up,
    silently keeps seeing the same stale "0 entries" forever (this
    genuinely happened: checking an upcoming race a bit too early poisoned
    the cache for every subsequent attempt, even the next day)."""
    empty_html = "<html><body><h1 class='RaceName'>2歳未勝利</h1></body></html>"
    published_html = (FIXTURES / "shutuba_sample.html").read_text(encoding="utf-8")

    class FakeResponse:
        def __init__(self, text):
            self.text = text
            self.apparent_encoding = "utf-8"

        def raise_for_status(self):
            pass

    responses = [empty_html, published_html]

    def fake_get(url, headers=None, timeout=None):
        return FakeResponse(responses.pop(0))

    monkeypatch.setattr("keiba_ai.scraper.requests.get", fake_get)
    scraper = PoliteScraper(ScraperConfig(cache_dir=tmp_path, respect_robots=False, min_interval_sec=0))
    url = "https://race.netkeiba.com/race/shutuba.html?race_id=1"

    first = scraper.fetch_shutuba(url)
    assert first["entries"] == []

    second = scraper.fetch_shutuba(url)
    assert len(second["entries"]) == 2


def test_robots_txt_unreachable_fails_open(monkeypatch):
    """A robots.txt fetch failure (network hiccup, TLS interception, ...) must
    not be treated as a real disallow -- RobotFileParser.can_fetch() defaults
    to False when read() never completes, which would otherwise silently
    block every subsequent request forever."""

    def broken_read(self):
        raise OSError("simulated network failure fetching robots.txt")

    monkeypatch.setattr(robotparser.RobotFileParser, "read", broken_read)
    scraper = PoliteScraper()

    rp = scraper._robots_for("https://race.netkeiba.com/race/shutuba.html?race_id=1")

    assert rp.can_fetch(scraper.config.user_agent, "https://race.netkeiba.com/race/shutuba.html?race_id=1")


def test_list_upcoming_races_for_date_groups_by_venue(monkeypatch):
    html = (FIXTURES / "race_list_sub_sample.html").read_text(encoding="utf-8")
    scraper = PoliteScraper()
    monkeypatch.setattr(scraper, "fetch", lambda url: html)

    races = scraper.list_upcoming_races_for_date("20250105")

    assert races == [
        {"race_id": "202506010101", "place": "中山"},
        {"race_id": "202506010102", "place": "中山"},
        {"race_id": "202507010101", "place": "中京"},
    ]


def test_is_jra_race_id_accepts_the_10_jra_courses():
    assert is_jra_race_id("202406010201")  # 06 = 中山
    assert is_jra_race_id("202408010211")  # 08 = 京都


def test_is_jra_race_id_rejects_regional_nar_meetings():
    assert not is_jra_race_id("202442011001")  # 42 = regional (NAR)
    assert not is_jra_race_id("202447011008")  # 47 = regional (NAR)


def test_is_jra_race_id_rejects_malformed_ids():
    assert not is_jra_race_id("")
    assert not is_jra_race_id("12345")
