#!/usr/bin/env python
"""Train the top-3 finish (複勝圏内) prediction model from a race-result CSV.

The CSV must have the columns produced by keiba_ai.parser (race_id, date,
place, surface, distance, track_condition, waku, umaban, horse_id,
horse_name, sex_age, kinryo, jockey_id, jockey, horse_weight, odds,
popularity, rank, ...). `python scripts/generate_demo_data.py` produces a
compatible file if you don't have real data yet.

Usage:
    python scripts/train_model.py --data data/demo_results.csv \
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
    parser.add_argument("--model-out", default="models/model.joblib")
    parser.add_argument("--history-out", default="models/history.csv")
    parser.add_argument("--num-boost-round", type=int, default=300)
    args = parser.parse_args()

    raw = pd.read_csv(args.data)
    training_df = build_training_frame(raw)
    model = train_model(training_df, num_boost_round=args.num_boost_round)
    print("metrics:", model.metrics)

    model.save(Path(args.model_out))
    Path(args.history_out).parent.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(args.history_out, index=False)
    print(f"saved model   -> {args.model_out}")
    print(f"saved history -> {args.history_out}")


if __name__ == "__main__":
    main()
