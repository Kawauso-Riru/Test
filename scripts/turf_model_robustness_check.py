#!/usr/bin/env python
"""Multi-seed robustness check for the turf-only feature set/model, mirroring
the 5-seed check mentioned in model.py's DEFAULT_PARAMS comment for the dirt
model. A single train/valid split's metrics can look good or bad by chance;
this retrains across several GroupShuffleSplit seeds (same hyperparameters)
and reports the spread, so a one-off number isn't mistaken for a stable
result -- the exact trap this project's README documents hitting repeatedly
on betting-strategy backtests.

Usage:
    python scripts/turf_model_robustness_check.py --seeds 1,2,3,4,5,99
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba_ai.features import ALL_FEATURE_COLUMNS, TURF_FEATURE_COLUMNS, build_training_frame  # noqa: E402
from keiba_ai.io import read_race_csv  # noqa: E402
from keiba_ai.model import train_model  # noqa: E402

MARKET_FEATURE_COLUMNS = {"popularity_numeric", "odds_numeric"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/jra_results.csv")
    parser.add_argument("--oikiri", default="data/oikiri.csv")
    parser.add_argument("--pedigree", default="data/pedigree.csv")
    parser.add_argument("--seeds", default="1,2,3,4,5,99")
    args = parser.parse_args()

    raw = read_race_csv(args.data)
    oikiri_path = Path(args.oikiri)
    if oikiri_path.exists():
        oikiri = read_race_csv(oikiri_path)[["race_id", "horse_id", "training_grade"]]
        raw = raw.merge(oikiri, on=["race_id", "horse_id"], how="left")
    pedigree_path = Path(args.pedigree)
    if pedigree_path.exists():
        pedigree = read_race_csv(pedigree_path)[["horse_id", "sire_id", "damsire_id"]]
        raw = raw.merge(pedigree, on="horse_id", how="left")
    training_df = build_training_frame(raw)

    seeds = [int(s) for s in args.seeds.split(",")]
    results = {"dirt": [], "turf": []}
    for surface, mask_col, feature_cols in (
        ("dirt", "is_dirt", ALL_FEATURE_COLUMNS),
        ("turf", None, TURF_FEATURE_COLUMNS),
    ):
        fit_df = training_df[training_df["is_dirt"]] if surface == "dirt" else training_df[~training_df["is_dirt"]]
        feature_columns = [c for c in feature_cols if c not in MARKET_FEATURE_COLUMNS]
        for seed in seeds:
            model = train_model(fit_df, feature_columns=feature_columns, seed=seed)
            results[surface].append(model.metrics)
            m = model.metrics
            print(f"{surface} seed={seed}: ndcg@6={m['valid_ndcg@6']:.4f} precision@3={m['valid_precision@3']:.4f} recall@6={m['valid_recall@6']:.4f}")

    print("\n=== summary (mean +/- std across seeds) ===")
    for surface in ("dirt", "turf"):
        ndcg = np.array([m["valid_ndcg@6"] for m in results[surface]])
        prec = np.array([m["valid_precision@3"] for m in results[surface]])
        recall = np.array([m["valid_recall@6"] for m in results[surface]])
        print(f"{surface}: ndcg@6={ndcg.mean():.4f}+/-{ndcg.std():.4f}  "
              f"precision@3={prec.mean():.4f}+/-{prec.std():.4f}  "
              f"recall@6={recall.mean():.4f}+/-{recall.std():.4f}")


if __name__ == "__main__":
    main()
