import pandas as pd

from keiba_ai.io import read_race_csv


def test_read_race_csv_preserves_zero_padded_ids(tmp_path):
    """jockey_id/trainer_id codes are zero-padded (e.g. "01209"). A bare
    pd.read_csv infers an all-digit column as int64 and silently drops the
    leading zero, which then breaks every downstream merge on that ID."""
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame([
        {"horse_id": "2021105473", "jockey_id": "01209", "trainer_id": "00423", "date": "2024-01-01"},
        {"horse_id": "2020104999", "jockey_id": "05386", "trainer_id": "01162", "date": "2024-01-08"},
    ]).to_csv(csv_path, index=False)

    df = read_race_csv(csv_path, parse_dates=["date"])

    assert df["jockey_id"].tolist() == ["01209", "05386"]
    assert df["trainer_id"].tolist() == ["00423", "01162"]
    assert pd.api.types.is_string_dtype(df["horse_id"])
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
