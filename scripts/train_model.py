#!/usr/bin/env python
"""Train the top-3 finish (複勝圏内) prediction model from a race-result CSV.

The CSV must have the columns produced by keiba_ai.parser (race_id, date,
place, surface, distance, track_condition, waku, umaban, horse_id,
horse_name, sex_age, kinryo, jockey_id, jockey, horse_weight, odds,
popularity, rank, ...). `python scripts/generate_demo_data.py` (synthetic)
or `python scripts/scrape_jra_dirt_results.py` (real JRA data) produce a
compatible file.

Use --dirt-only to fit the model on dirt races only (recommended for a
dirt-specialized model) -- turf races still contribute to each horse's/
jockey's history features either way, so pass the full mixed-surface CSV
regardless; only the *fitting* target rows are filtered.

By default, market features (popularity_numeric/odds_numeric -- the horse's
final betting popularity/win odds) are EXCLUDED. They're by far the single
strongest predictor in backtests (the market already prices in most public
information), but they're usually still unset ("**") days ahead of a race --
exactly when you'd want to use scripts/predict_raceday.py. A model trained
on them degrades to near-identical predictions for every horse once they're
missing. Pass --include-market-features only if you specifically intend to
predict close to race time, once odds have firmed up.

Usage:
    python scripts/train_model.py --data data/jra_results.csv --dirt-only \
        --model-out models/model_dirt.joblib --history-out models/history.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba_ai.features import ALL_FEATURE_COLUMNS, build_training_frame  # noqa: E402
from keiba_ai.io import read_race_csv  # noqa: E402
from keiba_ai.model import train_model  # noqa: E402

MARKET_FEATURE_COLUMNS = {"popularity_numeric", "odds_numeric"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="path to a race-result CSV")
    parser.add_argument("--oikiri", default="data/oikiri.csv",
                         help="optional training-grade CSV from scripts/scrape_oikiri.py; "
                              "merged in as the training_grade feature if the file exists")
    parser.add_argument("--dirt-only", action="store_true", help="fit only on dirt (ダート) races")
    parser.add_argument("--include-market-features", action="store_true",
                         help="keep popularity/odds as features (see caveat above)")
    parser.add_argument("--model-out", default="models/model_dirt.joblib")
    parser.add_argument("--history-out", default="models/history.csv")
    parser.add_argument("--num-boost-round", type=int, default=300)
    args = parser.parse_args()

    raw = read_race_csv(args.data)
    oikiri_path = Path(args.oikiri)
    if oikiri_path.exists():
        oikiri = read_race_csv(oikiri_path)[["race_id", "horse_id", "training_grade"]]
        raw = raw.merge(oikiri, on=["race_id", "horse_id"], how="left")
        print(f"merged training_grade: {raw['training_grade'].notna().sum()}/{len(raw)} rows have a grade")
    training_df = build_training_frame(raw)

    fit_df = training_df[training_df["is_dirt"]] if args.dirt_only else training_df
    print(f"fitting on {len(fit_df)} entries ({fit_df['race_id'].nunique()} races)"
          + (" [dirt only]" if args.dirt_only else ""))

    feature_columns = ALL_FEATURE_COLUMNS
    if not args.include_market_features:
        feature_columns = [c for c in ALL_FEATURE_COLUMNS if c not in MARKET_FEATURE_COLUMNS]
        print("excluding market features:", sorted(MARKET_FEATURE_COLUMNS))

    model = train_model(fit_df, num_boost_round=args.num_boost_round, feature_columns=feature_columns)
    print("metrics:", model.metrics)

    model.save(Path(args.model_out))
    Path(args.history_out).parent.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(args.history_out, index=False)
    print(f"saved model   -> {args.model_out}")
    print(f"saved history -> {args.history_out}")


if __name__ == "__main__":
    main()
