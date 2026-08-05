"""HTML parsing utilities for netkeiba-style race pages.

Kept separate from scraper.py (network I/O) so the parsing logic can be
unit-tested against static HTML fixtures without any network access, and so
it keeps working even if you point the scraper at a different site with a
similar table layout.

The strategy is deliberately generic: instead of hard-coding CSS class names
(which change over time), we scan every <table> on the page for one whose
header row contains the Japanese column labels we expect, then map columns
by label rather than by position.
"""
from __future__ import annotations

import re
from typing import Optional

from bs4 import BeautifulSoup, Tag

RESULT_HEADER_ALIASES = {
    "着順": "rank",
    "枠": "waku",
    "枠番": "waku",
    "馬番": "umaban",
    "馬名": "horse_name",
    "性齢": "sex_age",
    "斤量": "kinryo",
    "騎手": "jockey",
    "タイム": "time",
    "着差": "diff",
    "通過": "passing",
    "上り": "last_3f",
    "単勝": "odds",
    "人気": "popularity",
    "馬体重": "horse_weight",
    "馬体重(増減)": "horse_weight",  # race.netkeiba.com shutuba table
    "調教師": "trainer",
    "厩舎": "trainer",  # race.netkeiba.com shutuba table
}


def _clean_text(tag: Optional[Tag]) -> str:
    if tag is None:
        return ""
    return tag.get_text(strip=True)


def _extract_id_from_href(tag: Optional[Tag], pattern: str) -> Optional[str]:
    if tag is None:
        return None
    a = tag.find("a", href=True)
    if a is None:
        return None
    m = re.search(pattern, a["href"])
    return m.group(1) if m else None


def _find_entry_table(soup: BeautifulSoup, required_headers: set) -> Optional[Tag]:
    """Return the first <table> whose header row contains all required_headers."""
    for table in soup.find_all("table"):
        header_row = table.find("tr")
        if header_row is None:
            continue
        headers = {_clean_text(cell) for cell in header_row.find_all(["th", "td"])}
        if required_headers <= headers:
            return table
    return None


def _build_header_map(table: Tag) -> dict:
    header_row = table.find("tr")
    header_map: dict = {}
    for idx, cell in enumerate(header_row.find_all(["th", "td"])):
        field_name = RESULT_HEADER_ALIASES.get(_clean_text(cell))
        if field_name:
            header_map[idx] = field_name
    return header_map


def _row_to_dict(row: Tag, header_map: dict) -> dict:
    cells = row.find_all(["td", "th"])
    record: dict = {}
    for idx, cell in enumerate(cells):
        field_name = header_map.get(idx)
        if field_name is None:
            continue
        record[field_name] = _clean_text(cell)
        if field_name == "horse_name":
            record["horse_id"] = _extract_id_from_href(cell, r"/horse/([\w\d]+)")
        elif field_name == "jockey":
            record["jockey_id"] = _extract_id_from_href(cell, r"/jockey/(?:result/recent/)?([\w\d]+)")
        elif field_name == "trainer":
            record["trainer_id"] = _extract_id_from_href(cell, r"/trainer/(?:result/recent/)?([\w\d]+)")
    return record


def parse_race_meta(soup: BeautifulSoup) -> dict:
    """Best-effort extraction of race metadata (surface/distance/weather/...).

    Two known layouts are handled:
      - race.netkeiba.com (shutuba pages): <h1 class="RaceName">, <div class="RaceData01">
      - db.netkeiba.com (result pages): <dl class="racedata ..."><h1>(no class)
        plus a <span> with the data text, all lowercase class names.
    """
    meta: dict = {}

    name_tag = soup.find(["h1", "div"], class_=re.compile("RaceName|race_name", re.I))
    data_tag = soup.find(["div", "p"], class_=re.compile("RaceData", re.I))

    if data_tag is None:
        dl = soup.find("dl", class_=re.compile("racedata", re.I))
        if dl is not None:
            data_tag = dl.find("span") or dl.find("p")
            if name_tag is None:
                name_tag = dl.find("h1")

    if name_tag is not None:
        meta["race_name"] = _clean_text(name_tag)

    data_text = _clean_text(data_tag)
    meta["race_data_raw"] = data_text

    # "ダ右1200m" / "芝右 外1600m" -- surface char(s) followed eventually by "<digits>m".
    dist_match = re.search(r"(芝|ダ)[^\d]*?(\d{3,4})m", data_text)
    if dist_match:
        meta["surface"] = "ダート" if dist_match.group(1) == "ダ" else dist_match.group(1)
        meta["distance"] = int(dist_match.group(2))

    weather_match = re.search(r"天候\s*[:：]\s*(\S+?)(?:[\s/]|$)", data_text)
    if weather_match:
        meta["weather"] = weather_match.group(1)

    # "馬場:良" (race.netkeiba.com) or, lacking that label, "ダート : 良" / "芝 : 良" (db.netkeiba.com)
    condition_match = re.search(r"馬場\s*[:：]\s*(\S+?)(?:[\s/]|$)", data_text)
    if condition_match is None:
        condition_match = re.search(r"(?:芝|ダート)\s*[:：]\s*(\S+?)(?:[\s/]|$)", data_text)
    if condition_match:
        meta["track_condition"] = condition_match.group(1)

    # db.netkeiba.com result pages carry a concise "<p class="smalltxt">" line:
    # "2024年01月07日 1回中山2日目 3歳未勝利  [指](馬齢)" -- the most reliable
    # source of the race's calendar date and course name (place).
    smalltxt = soup.find("p", class_=re.compile("smalltxt", re.I))
    smalltxt_text = _clean_text(smalltxt)
    date_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日\s*(\d+)回(\S+?)(\d+)日目", smalltxt_text)
    if date_match:
        meta["date"] = f"{int(date_match.group(1)):04d}-{int(date_match.group(2)):02d}-{int(date_match.group(3)):02d}"
        meta["kaisai_round"] = int(date_match.group(4))
        meta["place"] = date_match.group(5)
        meta["day_of_meeting"] = int(date_match.group(6))

    return meta


def parse_race_result_html(html: str) -> dict:
    """Parse a netkeiba-style race *result* page into {meta, entries}."""
    soup = BeautifulSoup(html, "lxml")
    table = _find_entry_table(soup, {"着順", "馬番", "馬名"})
    entries = []
    if table is not None:
        header_map = _build_header_map(table)
        for row in table.find_all("tr")[1:]:
            record = _row_to_dict(row, header_map)
            if record.get("umaban", "").isdigit():
                entries.append(record)
    return {"meta": parse_race_meta(soup), "entries": entries}


def parse_shutuba_html(html: str) -> dict:
    """Parse a netkeiba-style *shutuba* (pre-race entry list) page into {meta, entries}."""
    soup = BeautifulSoup(html, "lxml")
    table = _find_entry_table(soup, {"馬番", "馬名"})
    entries = []
    if table is not None:
        header_map = _build_header_map(table)
        for row in table.find_all("tr")[1:]:
            record = _row_to_dict(row, header_map)
            if record.get("umaban", "").isdigit():
                entries.append(record)
    return {"meta": parse_race_meta(soup), "entries": entries}
