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

Usage:
    python scripts/train_model.py --data data/jra_results.csv --dirt-only \
        --model-out models/model.joblib --history-out models/history.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba_ai.features import build_training_frame  # noqa: E402
from keiba_ai.model import train_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="path to a race-result CSV")
    parser.add_argument("--dirt-only", action="store_true", help="fit only on dirt (ダート) races")
    parser.add_argument("--model-out", default="models/model.joblib")
    parser.add_argument("--history-out", default="models/history.csv")
    parser.add_argument("--num-boost-round", type=int, default=300)
    args = parser.parse_args()

    raw = pd.read_csv(args.data)
    training_df = build_training_frame(raw)

    fit_df = training_df[training_df["is_dirt"]] if args.dirt_only else training_df
    print(f"fitting on {len(fit_df)} entries ({fit_df['race_id'].nunique()} races)"
          + (" [dirt only]" if args.dirt_only else ""))

    model = train_model(fit_df, num_boost_round=args.num_boost_round)
    print("metrics:", model.metrics)

    model.save(Path(args.model_out))
    Path(args.history_out).parent.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(args.history_out, index=False)
    print(f"saved model   -> {args.model_out}")
    print(f"saved history -> {args.history_out}")


if __name__ == "__main__":
    main()
