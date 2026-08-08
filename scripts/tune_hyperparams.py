#!/usr/bin/env python
"""Random search over LightGBM lambdarank hyperparameters.

Trains `--n-trials` random parameter combinations against the SAME
train/valid race split (train_model's `seed` is held fixed) so the
comparison across trials is apples-to-apples, then prints them ranked by
held-out NDCG@3. Copy the winning params into keiba_ai.model.DEFAULT_PARAMS
by hand once you're happy with a result -- this script doesn't write code.

Usage:
    python scripts/tune_hyperparams.py --data data/jra_results.csv --dirt-only --n-trials 30
"""
import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba_ai.features import ALL_FEATURE_COLUMNS, build_training_frame  # noqa: E402
from keiba_ai.io import read_race_csv  # noqa: E402
from keiba_ai.model import train_model  # noqa: E402

MARKET_FEATURE_COLUMNS = {"popularity_numeric", "odds_numeric"}

PARAM_SPACE = {
    "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
    "num_leaves": [15, 31, 63, 127],
    "min_data_in_leaf": [10, 20, 30, 50],
    "feature_fraction": [0.7, 0.8, 0.9, 1.0],
    "bagging_fraction": [0.7, 0.8, 0.9, 1.0],
    "bagging_freq": [0, 1, 5],
    "lambda_l1": [0.0, 0.1, 1.0],
    "lambda_l2": [0.0, 0.1, 1.0],
}


def sample_params(rng: random.Random) -> dict:
    return {name: rng.choice(values) for name, values in PARAM_SPACE.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--oikiri", default="data/oikiri.csv",
                         help="optional training-grade CSV from scripts/scrape_oikiri.py; "
                              "merged in as the training_grade feature if the file exists")
    parser.add_argument("--dirt-only", action="store_true")
    parser.add_argument("--include-market-features", action="store_true",
                         help="keep popularity/odds as features (see train_model.py's caveat)")
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--num-boost-round", type=int, default=300)
    parser.add_argument("--split-seed", type=int, default=42, help="held fixed across all trials")
    parser.add_argument("--sampling-seed", type=int, default=0, help="controls which param combos get tried")
    args = parser.parse_args()

    raw = read_race_csv(args.data)
    oikiri_path = Path(args.oikiri)
    if oikiri_path.exists():
        oikiri = read_race_csv(oikiri_path)[["race_id", "horse_id", "training_grade"]]
        raw = raw.merge(oikiri, on=["race_id", "horse_id"], how="left")
    training_df = build_training_frame(raw)
    fit_df = training_df[training_df["is_dirt"]] if args.dirt_only else training_df
    print(f"tuning on {len(fit_df)} entries ({fit_df['race_id'].nunique()} races)")

    feature_columns = ALL_FEATURE_COLUMNS
    if not args.include_market_features:
        feature_columns = [c for c in ALL_FEATURE_COLUMNS if c not in MARKET_FEATURE_COLUMNS]

    rng = random.Random(args.sampling_seed)
    results = []
    for trial in range(1, args.n_trials + 1):
        params = sample_params(rng)
        model = train_model(fit_df, num_boost_round=args.num_boost_round, params=params,
                             seed=args.split_seed, feature_columns=feature_columns)
        results.append((model.metrics["valid_ndcg@3"], model.metrics["valid_precision@3"], params))
        print(f"trial {trial:>2}/{args.n_trials}  ndcg@3={model.metrics['valid_ndcg@3']:.4f}  "
              f"precision@3={model.metrics['valid_precision@3']:.4f}  {params}")

    results.sort(key=lambda r: r[0], reverse=True)
    print("\n=== top 5 by NDCG@3 ===")
    for ndcg, precision, params in results[:5]:
        print(f"ndcg@3={ndcg:.4f}  precision@3={precision:.4f}")
        print(f"  {params}")

    best_ndcg, best_precision, best_params = results[0]
    print(f"\nbest params (ndcg@3={best_ndcg:.4f}, precision@3={best_precision:.4f}):")
    print(best_params)


if __name__ == "__main__":
    main()
