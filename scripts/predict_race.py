#!/usr/bin/env python
"""Rank every entrant in an upcoming race by predicted finishing strength.

Two input modes:
  --shutuba-csv PATH  A locally prepared CSV with the same columns as a
                       parsed shutuba table (see keiba_ai.parser).
  --shutuba-url URL   Scrape a live shutuba (entry list) page. Subject to
                       the target site's Terms of Use -- see README.md.
                       race.netkeiba.com shutuba pages don't expose the
                       course name, so pass --place to fill it in.

The model is a LightGBM ranker (LambdaMART), not a classifier: its raw output
is only meaningful as a *relative* order within one race. This script turns
that into a display percentage via softmax (relative "share" of the field,
summing to 100%) -- it is not a calibrated win/top-3 probability.

Usage:
    python scripts/predict_race.py --shutuba-csv data/sample_shutuba.csv \
        --model models/model_dirt.joblib --history models/history.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba_ai.features import build_prediction_frame  # noqa: E402
from keiba_ai.model import KeibaModel, softmax_scores  # noqa: E402
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
        df["place"] = meta.get("place") or args.place
        return df
    raise SystemExit("either --shutuba-csv or --shutuba-url is required")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shutuba-csv")
    parser.add_argument("--shutuba-url")
    parser.add_argument("--place", default="", help="course name (e.g. 中山) -- only needed with --shutuba-url")
    parser.add_argument("--model", default="models/model_dirt.joblib")
    parser.add_argument("--history", default="models/history.csv")
    args = parser.parse_args()

    shutuba = load_shutuba(args)
    history = pd.read_csv(args.history, parse_dates=["date"])
    model = KeibaModel.load(Path(args.model))

    feature_df = build_prediction_frame(shutuba, history)
    feature_df["score"] = model.predict(feature_df)
    feature_df["relative_share(%)"] = (softmax_scores(feature_df["score"]) * 100).round(1)

    result = feature_df.sort_values("score", ascending=False)[
        ["umaban", "horse_name", "jockey", "relative_share(%)"]
    ].copy()
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
