import pandas as pd

from keiba_ai.features import build_prediction_frame, build_training_frame


def _sample_raw() -> pd.DataFrame:
    rows = [
        dict(
            race_id="r1", date="2023-01-01", place="東京", surface="芝", distance=2000,
            track_condition="良", waku=1, umaban=1, horse_id="h1", horse_name="H1",
            sex_age="牡4", kinryo=57.0, jockey_id="j1", jockey="J1", trainer_id="t1",
            trainer="T1", horse_weight="480(+2)", odds=2.0, popularity=1, rank=1, last_3f=35.0,
        ),
        dict(
            race_id="r1", date="2023-01-01", place="東京", surface="芝", distance=2000,
            track_condition="良", waku=2, umaban=2, horse_id="h2", horse_name="H2",
            sex_age="牝3", kinryo=54.0, jockey_id="j2", jockey="J2", trainer_id="t2",
            trainer="T2", horse_weight="440(0)", odds=5.0, popularity=2, rank=2, last_3f=35.5,
        ),
        dict(
            race_id="r2", date="2023-01-08", place="中山", surface="ダート", distance=1800,
            track_condition="稍重", waku=1, umaban=1, horse_id="h1", horse_name="H1",
            sex_age="牡4", kinryo=57.0, jockey_id="j1", jockey="J1", trainer_id="t1",
            trainer="T1", horse_weight="482(+2)", odds=1.8, popularity=1, rank=1, last_3f=36.0,
        ),
        dict(
            race_id="r3", date="2023-01-15", place="中山", surface="ダート", distance=1800,
            track_condition="良", waku=1, umaban=1, horse_id="h1", horse_name="H1",
            sex_age="牡4", kinryo=57.0, jockey_id="j1", jockey="J1", trainer_id="t1",
            trainer="T1", horse_weight="480(-2)", odds=2.5, popularity=1, rank=4, last_3f=36.5,
        ),
    ]
    return pd.DataFrame(rows)


def test_build_training_frame_no_leakage():
    df = build_training_frame(_sample_raw())

    first_race = df[(df["horse_id"] == "h1") & (df["race_id"] == "r1")].iloc[0]
    second_race = df[(df["horse_id"] == "h1") & (df["race_id"] == "r2")].iloc[0]

    # First-ever run for h1: no prior history should exist yet.
    assert first_race["horse_runs_before"] == 0
    assert pd.isna(first_race["horse_win_rate_before"])
    assert pd.isna(first_race["days_since_last_race"])

    # Second run: exactly one prior race, which was a win.
    assert second_race["horse_runs_before"] == 1
    assert second_race["horse_win_rate_before"] == 1.0
    assert second_race["horse_avg_rank_before"] == 1.0
    assert second_race["days_since_last_race"] == 7

    assert first_race["target_top3"] == 1
    assert first_race["target_win"] == 1


def test_dirt_only_history_ignores_interleaved_turf_races():
    df = build_training_frame(_sample_raw())

    # r2 is h1's first-ever DIRT race, even though it's h1's second race
    # overall (r1 was turf) -- dirt history must start from zero here, not
    # inherit the turf win.
    r2 = df[(df["horse_id"] == "h1") & (df["race_id"] == "r2")].iloc[0]
    assert r2["horse_runs_before"] == 1          # overall: counts the turf race
    assert r2["horse_dirt_runs_before"] == 0      # dirt-only: no prior dirt starts
    assert pd.isna(r2["horse_dirt_win_rate_before"])

    # r3 is h1's second dirt race; dirt history should reflect only r2 (a win),
    # not be diluted or skipped because of the turf race in between.
    r3 = df[(df["horse_id"] == "h1") & (df["race_id"] == "r3")].iloc[0]
    assert r3["horse_dirt_runs_before"] == 1
    assert r3["horse_dirt_win_rate_before"] == 1.0
    assert r3["horse_dirt_avg_rank_before"] == 1.0


def test_sex_age_and_weight_parsing():
    df = build_training_frame(_sample_raw())
    row = df[(df["horse_id"] == "h2")].iloc[0]
    assert row["sex"] == "牝"
    assert row["age"] == 3.0
    assert row["horse_weight_kg"] == 440.0
    assert row["horse_weight_diff"] == 0.0


def _running_style_raw() -> pd.DataFrame:
    """Two 4-horse races: hA always leads early (passing starts with 1),
    hB always trails (passing starts with 4)."""
    common = dict(place="東京", surface="ダート", distance=1600, track_condition="良", kinryo=55.0)
    rows = []
    for race_id, date_str in [("rX", "2023-03-01"), ("rY", "2023-03-08")]:
        rows.append(dict(common, race_id=race_id, date=date_str, waku=1, umaban=1, horse_id="hA",
                          horse_name="HA", sex_age="牡4", jockey_id="jA", jockey="JA", trainer_id="tA",
                          trainer="TA", horse_weight="480(0)", odds=2.0, popularity=1, rank=1,
                          last_3f=35.0, passing="1-1"))
        rows.append(dict(common, race_id=race_id, date=date_str, waku=2, umaban=2, horse_id="hB",
                          horse_name="HB", sex_age="牡5", jockey_id="jB", jockey="JB", trainer_id="tB",
                          trainer="TB", horse_weight="470(0)", odds=3.0, popularity=2, rank=2,
                          last_3f=35.2, passing="4-4"))
        rows.append(dict(common, race_id=race_id, date=date_str, waku=3, umaban=3, horse_id="hC",
                          horse_name="HC", sex_age="牡6", jockey_id="jC", jockey="JC", trainer_id="tC",
                          trainer="TC", horse_weight="460(0)", odds=4.0, popularity=3, rank=3,
                          last_3f=35.4, passing="2-2"))
        rows.append(dict(common, race_id=race_id, date=date_str, waku=4, umaban=4, horse_id="hD",
                          horse_name="HD", sex_age="牡7", jockey_id="jD", jockey="JD", trainer_id="tD",
                          trainer="TD", horse_weight="450(0)", odds=5.0, popularity=4, rank=4,
                          last_3f=35.6, passing="3-3"))
    return pd.DataFrame(rows)


def test_running_style_reflects_early_corner_position():
    df = build_training_frame(_running_style_raw())

    rX_hA = df[(df["race_id"] == "rX") & (df["horse_id"] == "hA")].iloc[0]
    assert pd.isna(rX_hA["horse_early_position_ratio_before"])  # no history yet

    # hA led at the 1st corner in rX (1st of 4 -> ratio 0.25); by rY that
    # should be hA's entire prior history.
    rY_hA = df[(df["race_id"] == "rY") & (df["horse_id"] == "hA")].iloc[0]
    assert rY_hA["horse_early_position_ratio_before"] == 0.25

    # hB trailed (4th of 4 -> ratio 1.0) in rX.
    rY_hB = df[(df["race_id"] == "rY") & (df["horse_id"] == "hB")].iloc[0]
    assert rY_hB["horse_early_position_ratio_before"] == 1.0


def test_distance_band_bucketing():
    df = build_training_frame(_sample_raw())
    assert df[df["race_id"] == "r1"]["distance_band"].iloc[0] == "長距離"  # 2000m
    assert df[df["race_id"] == "r2"]["distance_band"].iloc[0] == "マイル"   # 1800m


def _course_bias_raw() -> pd.DataFrame:
    """Three different horses drawing waku=1 at the same course/surface/distance
    band, on three different dates -- tests that the bias stat is keyed off the
    (place, surface, distance_band, waku) combo, not off any one horse."""
    common = dict(place="中山", surface="ダート", distance=1800, track_condition="良", kinryo=57.0)
    rows = [
        dict(common, race_id="rA", date="2023-02-01", waku=1, umaban=1, horse_id="hA", horse_name="HA",
             sex_age="牡4", jockey_id="jA", jockey="JA", trainer_id="tA", trainer="TA",
             horse_weight="480(0)", odds=3.0, popularity=1, rank=1, last_3f=36.0),
        dict(common, race_id="rB", date="2023-02-08", waku=1, umaban=1, horse_id="hB", horse_name="HB",
             sex_age="牡5", jockey_id="jB", jockey="JB", trainer_id="tB", trainer="TB",
             horse_weight="470(0)", odds=4.0, popularity=1, rank=3, last_3f=36.2),
        dict(common, race_id="rC", date="2023-02-15", waku=1, umaban=1, horse_id="hC", horse_name="HC",
             sex_age="牡6", jockey_id="jC", jockey="JC", trainer_id="tC", trainer="TC",
             horse_weight="460(0)", odds=5.0, popularity=1, rank=2, last_3f=36.4),
    ]
    return pd.DataFrame(rows)


def test_course_waku_bias_is_shared_across_different_horses():
    df = build_training_frame(_course_bias_raw())

    rA = df[df["race_id"] == "rA"].iloc[0]
    assert rA["course_waku_bias_runs_before"] == 0

    # rB is a different horse, but shares (place, surface, distance_band, waku)
    # with rA -- it should see rA's result as prior history for this slot.
    rB = df[df["race_id"] == "rB"].iloc[0]
    assert rB["course_waku_bias_runs_before"] == 1
    assert rB["course_waku_bias_win_rate_before"] == 1.0  # rA won

    rC = df[df["race_id"] == "rC"].iloc[0]
    assert rC["course_waku_bias_runs_before"] == 2
    assert rC["course_waku_bias_win_rate_before"] == 0.5  # rA won, rB didn't


def _trainer_raw() -> pd.DataFrame:
    """Same trainer (tX) handling three different horses across three races --
    tests that trainer stats aggregate by trainer_id, independent of horse."""
    common = dict(place="東京", surface="ダート", distance=1600, track_condition="良", kinryo=55.0,
                   waku=1, umaban=1, jockey_id="jX", jockey="JX", trainer_id="tX", trainer="TX")
    rows = [
        dict(common, race_id="r1", date="2023-04-01", horse_id="hA", horse_name="HA",
             sex_age="牡4", horse_weight="480(0)", odds=3.0, popularity=1, rank=1, last_3f=36.0),
        dict(common, race_id="r2", date="2023-04-08", horse_id="hB", horse_name="HB",
             sex_age="牡5", horse_weight="470(0)", odds=4.0, popularity=1, rank=4, last_3f=36.2),
        dict(common, race_id="r3", date="2023-04-15", horse_id="hC", horse_name="HC",
             sex_age="牡6", horse_weight="460(0)", odds=5.0, popularity=1, rank=2, last_3f=36.4),
    ]
    return pd.DataFrame(rows)


def test_trainer_stats_aggregate_across_different_horses():
    df = build_training_frame(_trainer_raw())

    r1 = df[df["race_id"] == "r1"].iloc[0]
    assert r1["trainer_runs_before"] == 0

    # r2 is a different horse trained by the same tX -- should see r1 (a win).
    r2 = df[df["race_id"] == "r2"].iloc[0]
    assert r2["trainer_runs_before"] == 1
    assert r2["trainer_win_rate_before"] == 1.0
    assert r2["trainer_dirt_runs_before"] == 1  # all dirt races here

    # r3: trainer now has 2 prior runs (1 win, 1 non-placing).
    r3 = df[df["race_id"] == "r3"].iloc[0]
    assert r3["trainer_runs_before"] == 2
    assert r3["trainer_win_rate_before"] == 0.5


def test_popularity_and_odds_numeric_parsing():
    df = build_training_frame(_sample_raw())
    row = df[df["race_id"] == "r1"].iloc[0]
    assert row["popularity_numeric"] == 1.0
    assert row["odds_numeric"] == 2.0

    training_df = build_training_frame(_sample_raw())
    shutuba = pd.DataFrame([
        dict(horse_id="h1", jockey_id="j1", trainer_id="t1", umaban=1, waku=1, horse_name="H1",
             jockey="J1", sex_age="牡4", kinryo=57.0, horse_weight="480(0)", surface="ダート",
             distance=1800, track_condition="良", place="中山", popularity="**", odds="---.-"),
    ])
    pred = build_prediction_frame(shutuba, training_df)
    # "**"/"---.-" are netkeiba's not-yet-finalized placeholders -- must parse
    # to NaN (missing), not raise or silently become 0.
    assert pd.isna(pred.iloc[0]["popularity_numeric"])
    assert pd.isna(pred.iloc[0]["odds_numeric"])


def test_training_grade_is_missing_by_default_but_passed_through_if_present():
    # Raw data without any oikiri merge -- training_grade must still exist
    # as a column (NaN), or selecting ALL_FEATURE_COLUMNS downstream would
    # KeyError instead of just treating it as missing.
    df = build_training_frame(_sample_raw())
    assert "training_grade" in df.columns
    assert df["training_grade"].isna().all()

    # A caller that *did* merge scripts/scrape_oikiri.py's output onto the
    # raw rows before calling build_training_frame should see it survive.
    raw_with_grade = _sample_raw()
    raw_with_grade["training_grade"] = ["B", "C", "A", "D"]
    graded = build_training_frame(raw_with_grade)
    assert graded["training_grade"].tolist() == ["B", "C", "A", "D"]

    # Same for an upcoming race's shutuba: not present -> NaN, not a KeyError.
    training_df = build_training_frame(_sample_raw())
    shutuba = pd.DataFrame([
        dict(horse_id="h1", jockey_id="j1", trainer_id="t1", umaban=1, waku=1, horse_name="H1",
             jockey="J1", sex_age="牡4", kinryo=57.0, horse_weight="480(0)", surface="ダート",
             distance=1800, track_condition="良", place="中山"),
    ])
    pred = build_prediction_frame(shutuba, training_df)
    assert pd.isna(pred.iloc[0]["training_grade"])

    shutuba["training_grade"] = "A"
    pred_graded = build_prediction_frame(shutuba, training_df)
    assert pred_graded.iloc[0]["training_grade"] == "A"


def _race_class_of(race_name: str) -> str:
    """Single-row raw frame carrying only the given race_name -- avoids the
    ambiguity of _sample_raw()'s repeated race_ids when indexing results."""
    row = dict(_sample_raw().iloc[0])
    row["race_name"] = race_name
    return build_training_frame(pd.DataFrame([row]))["race_class"].iloc[0]


def test_race_class_parses_both_netkeiba_naming_styles():
    assert _race_class_of("2歳新馬") == "新馬"                       # plain condition-race name
    assert _race_class_of("3歳以上1勝クラス") == "1勝クラス"
    assert _race_class_of("4歳以上2勝クラス") == "2勝クラス"
    assert _race_class_of("3歳以上3勝クラス") == "3勝クラス"
    assert _race_class_of("2歳未勝利") == "未勝利"
    assert _race_class_of("4歳以上オープン") == "オープン"
    assert _race_class_of("障害4歳以上未勝利") == "障害"
    assert _race_class_of("第40回フェアリーステークス(GIII)") == "G3"  # named stakes, class in trailing parens
    assert _race_class_of("○○賞(GII)") == "G2"  # must not misdetect via substring collision with GIII
    assert _race_class_of("○○賞(GI)") == "G1"
    assert _race_class_of("△△特別(2勝)") == "2勝クラス"
    assert _race_class_of("□□ステークス(L)") == "リステッド"
    assert _race_class_of("万葉ステークス(OP)") == "オープン"

    # Missing race_name column entirely (e.g. an older cached history CSV) ->
    # falls back to "不明" rather than KeyError.
    no_name = build_training_frame(_sample_raw())
    assert (no_name["race_class"] == "不明").all()


def test_prediction_frame_includes_most_recent_finished_race():
    """A horse's single past race (a win) must show up in the stats used to
    predict its *next* race -- not lag by one, which would show 0 runs/NaN
    win-rate for a horse that has actually already won once."""
    single_win = pd.DataFrame([
        dict(
            race_id="r1", date="2023-01-01", place="中山", surface="ダート", distance=1800,
            track_condition="良", waku=1, umaban=1, horse_id="h1", horse_name="H1",
            sex_age="牡4", kinryo=57.0, jockey_id="j1", jockey="J1", trainer_id="t1",
            trainer="T1", horse_weight="480(+2)", odds=2.0, popularity=1, rank=1, last_3f=35.0,
        ),
    ])
    training_df = build_training_frame(single_win)

    shutuba = pd.DataFrame([
        dict(horse_id="h1", jockey_id="j1", umaban=1, waku=1, horse_name="H1", jockey="J1",
             sex_age="牡4", kinryo=57.0, horse_weight="480(0)", surface="ダート", distance=1800,
             track_condition="良", place="中山"),
    ])
    pred = build_prediction_frame(shutuba, training_df)
    row = pred.iloc[0]

    # h1's only race so far was a win. Before the fix, the "latest" snapshot
    # reused that race's own *_before values (0 runs, NaN win-rate) instead of
    # rolling its result in, so a proven winner would predict as a blank slate.
    assert row["horse_runs_before"] == 1
    assert row["horse_win_rate_before"] == 1.0
    assert row["horse_avg_rank_before"] == 1.0
