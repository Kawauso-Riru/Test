import pandas as pd

from keiba_ai.features import build_training_frame


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


def test_sex_age_and_weight_parsing():
    df = build_training_frame(_sample_raw())
    row = df[(df["horse_id"] == "h2")].iloc[0]
    assert row["sex"] == "牝"
    assert row["age"] == 3.0
    assert row["horse_weight_kg"] == 440.0
    assert row["horse_weight_diff"] == 0.0
