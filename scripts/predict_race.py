#!/usr/bin/env python
"""Predict top-3-finish probability for every entrant in an upcoming race.

Two input modes:
  --shutuba-csv PATH  A locally prepared CSV with the same columns as a
                       parsed shutuba table (see keiba_ai.parser).
  --shutuba-url URL   Scrape a live shutuba (entry list) page. Subject to
                       the target site's Terms of Use -- see README.md.

Usage:
    python scripts/predict_race.py --shutuba-csv data/sample_shutuba.csv \
        --model models/model.joblib --history models/history.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba_ai.features import build_prediction_frame  # noqa: E402
from keiba_ai.model import KeibaModel  # noqa: E402
from keiba_ai.scraper import PoliteScraper  # noqa: E402


def load_shutuba(args: argparse.Namespace) -> pd.DataFrame:
    if args.shutuba_csv:
        return pd.read_csv(args.shutuba_csv)
    if args.shutuba_url:
        scraper = PoliteScraper()
        parsed = scraper.fetch_shutuba(args.shutuba_url)
        df = pd.DataFrame(parsed["entries"])
        meta = parsed["meta"]
        df["surface"] = meta.get("surface", "")
        df["distance"] = meta.get("distance")
        df["track_condition"] = meta.get("track_condition", "")
        df["place"] = meta.get("race_name", "")
        return df
    raise SystemExit("either --shutuba-csv or --shutuba-url is required")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shutuba-csv")
    parser.add_argument("--shutuba-url")
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--history", default="models/history.csv")
    args = parser.parse_args()

    shutuba = load_shutuba(args)
    history = pd.read_csv(args.history, parse_dates=["date"])
    model = KeibaModel.load(Path(args.model))

    feature_df = build_prediction_frame(shutuba, history)
    feature_df["top3_probability"] = model.predict(feature_df)

    result = feature_df.sort_values("top3_probability", ascending=False)[
        ["umaban", "horse_name", "jockey", "top3_probability"]
    ].copy()
    result["top3_probability"] = (result["top3_probability"] * 100).round(1)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
