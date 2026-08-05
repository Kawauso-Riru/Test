from pathlib import Path

from keiba_ai.parser import parse_race_result_html, parse_shutuba_html

FIXTURES = Path(__file__).parent / "fixtures"


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
