from keiba_ai.scraper import is_jra_race_id


def test_is_jra_race_id_accepts_the_10_jra_courses():
    assert is_jra_race_id("202406010201")  # 06 = 中山
    assert is_jra_race_id("202408010211")  # 08 = 京都


def test_is_jra_race_id_rejects_regional_nar_meetings():
    assert not is_jra_race_id("202442011001")  # 42 = regional (NAR)
    assert not is_jra_race_id("202447011008")  # 47 = regional (NAR)


def test_is_jra_race_id_rejects_malformed_ids():
    assert not is_jra_race_id("")
    assert not is_jra_race_id("12345")
