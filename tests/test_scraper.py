from pathlib import Path

import urllib.robotparser as robotparser

from keiba_ai.scraper import PoliteScraper, is_jra_race_id

FIXTURES = Path(__file__).parent / "fixtures"


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
