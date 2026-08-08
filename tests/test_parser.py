from pathlib import Path

from keiba_ai.parser import parse_oikiri_html, parse_race_result_html, parse_shutuba_html

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_oikiri_html():
    html = (FIXTURES / "oikiri_sample.html").read_text(encoding="utf-8")
    entries = parse_oikiri_html(html)

    assert len(entries) == 3

    first = entries[0]
    assert first["horse_id"] == "2024103642"
    assert first["horse_name"] == "マイネルヴェスペル"
    assert first["training_last_furlong_sec"] == 12.7  # last (RapTime) split, not the cumulative time
    assert first["training_grade"] == "C"

    second = entries[1]
    assert second["training_last_furlong_sec"] == 11.9
    assert second["training_grade"] == "B"

    # A row with no logged workout (e.g. a late scratch) shouldn't raise --
    # it just omits the fields it couldn't find.
    third = entries[2]
    assert third["horse_name"] == "スクラッチウマ"
    assert "training_last_furlong_sec" not in third
    assert "training_grade" not in third


def test_parse_race_result_html():
    html = (FIXTURES / "race_result_sample.html").read_text(encoding="utf-8")
    parsed = parse_race_result_html(html)

    assert parsed["meta"]["surface"] == "芝"
    assert parsed["meta"]["distance"] == 2400
    assert parsed["meta"]["track_condition"] == "良"
    assert parsed["meta"]["weather"] == "晴"

    entries = parsed["entries"]
    assert len(entries) == 3

    first = entries[0]
    assert first["rank"] == "1"
    assert first["horse_name"] == "サンプルホースA"
    assert first["horse_id"] == "2019104567"
    assert first["jockey_id"] == "00666"
    assert first["horse_weight"] == "480(+2)"
    assert first["trainer_id"] == "01055"

    assert entries[1]["diff"] == "クビ"
    assert entries[2]["horse_id"] == "2017104999"


def test_parse_shutuba_html():
    html = (FIXTURES / "shutuba_sample.html").read_text(encoding="utf-8")
    parsed = parse_shutuba_html(html)

    assert parsed["meta"]["surface"] == "芝"
    assert parsed["meta"]["distance"] == 2000
    assert parsed["meta"]["track_condition"] == "稍重"

    entries = parsed["entries"]
    assert len(entries) == 2
    assert "rank" not in entries[0]
    assert entries[0]["horse_name"] == "サンプルホースA"
    assert entries[1]["horse_id"] == "2020105999"
    assert entries[1]["jockey_id"] == "00999"


def test_parse_race_result_html_dbnetkeiba_style():
    """db.netkeiba.com uses <dl class="racedata"> + an unclassed <h1>, and labels
    track condition with the surface name ("ダート : 良") instead of "馬場:"."""
    html = (FIXTURES / "race_result_dbstyle_sample.html").read_text(encoding="utf-8")
    parsed = parse_race_result_html(html)

    assert parsed["meta"]["race_name"] == "サンプルダート特別"
    assert parsed["meta"]["surface"] == "ダート"
    assert parsed["meta"]["distance"] == 1200
    assert parsed["meta"]["track_condition"] == "良"
    assert parsed["meta"]["weather"] == "晴"

    entries = parsed["entries"]
    assert len(entries) == 2
    assert entries[0]["horse_id"] == "2019104567"
    assert entries[0]["trainer_id"] == "01055"


def test_parse_shutuba_html_alt_headers_and_stray_row():
    """race.netkeiba.com's shutuba table uses 厩舎/馬体重(増減) headers and has a
    stray "登録/グループ" toggle row before the real entries -- it must be filtered
    out by the numeric-umaban check rather than mis-parsed as an entry."""
    html = (FIXTURES / "shutuba_altheaders_sample.html").read_text(encoding="utf-8")
    parsed = parse_shutuba_html(html)

    assert parsed["meta"]["surface"] == "ダート"
    assert parsed["meta"]["distance"] == 1400
    assert parsed["meta"]["track_condition"] == "稍重"

    entries = parsed["entries"]
    assert len(entries) == 2
    assert entries[0]["trainer"] == "美浦サンプル厩舎A"
    assert entries[0]["trainer_id"] == "01055"
    assert entries[0]["horse_weight"] == "482(+2)"
