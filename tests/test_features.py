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
