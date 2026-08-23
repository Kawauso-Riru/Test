#!/usr/bin/env python
"""Backtest an expected-value win-betting filter: instead of always betting
on the model's top pick, bet on ANY horse (in any held-out race) whose
calibrated win probability times its actual final odds implies positive
expected value -- EV = P(win) * odds - 1 -- and see whether that actually
beats the flat "always bet the top pick" strategy once real payouts are
applied.

Uses the exact same held-out validation split as train_model() (same seed),
so this never touches races the model trained on. Odds come straight from
the historical CSV (a real final 単勝 odds value, not a model feature --
popularity/odds are deliberately excluded from training, see README).

Usage:
    python scripts/ev_win_backtest.py --model models/model_dirt.joblib
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba_ai.features import build_training_frame  # noqa: E402
from keiba_ai.io import read_race_csv  # noqa: E402
from keiba_ai.model import KeibaModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/jra_results.csv")
    parser.add_argument("--oikiri", default="data/oikiri.csv")
    parser.add_argument("--model", default="models/model_dirt.joblib")
    parser.add_argument("--unit", type=int, default=100)
    args = parser.parse_args()

    raw = read_race_csv(args.data)
    oikiri_path = Path(args.oikiri)
    if oikiri_path.exists():
        oikiri = read_race_csv(oikiri_path)[["race_id", "horse_id", "training_grade"]]
        raw = raw.merge(oikiri, on=["race_id", "horse_id"], how="left")
    training_df = build_training_frame(raw)
    fit_df = training_df[training_df["is_dirt"]].dropna(subset=["relevance"]).reset_index(drop=True)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, valid_idx = next(splitter.split(fit_df, fit_df["relevance"], groups=fit_df["race_id"]))
    valid_df = fit_df.iloc[valid_idx].copy()
    valid_df = valid_df.dropna(subset=["odds_numeric"])
    print(f"held-out validation rows with known odds: {len(valid_df)} ({valid_df['race_id'].nunique()} races)")

    model = KeibaModel.load(Path(args.model))
    valid_df["win_prob"] = model.predict_win_probability(valid_df)
    valid_df["ev"] = valid_df["win_prob"] * valid_df["odds_numeric"] - 1

    def simulate(mask: pd.Series) -> tuple:
        sub = valid_df[mask]
        n = len(sub)
        bet = n * args.unit
        ret = int((sub["target_win"] * sub["odds_numeric"] * args.unit).sum())
        roi = ret / bet * 100 if bet else float("nan")
        return n, bet, ret, roi

    print(f"\n{'strategy':30s} {'bets':>7s} {'invest':>10s} {'return':>10s} {'profit':>10s} {'ROI':>8s}")

    n, bet, ret, roi = simulate(pd.Series(True, index=valid_df.index))
    print(f"{'全馬に単勝(基準線)':30s} {n:7d} {bet:10d} {ret:10d} {ret-bet:+10d} {roi:7.1f}%")

    top_pick_idx = valid_df.sort_values("win_prob", ascending=False).groupby("race_id").head(1).index
    mask = valid_df.index.isin(top_pick_idx)
    n, bet, ret, roi = simulate(pd.Series(mask, index=valid_df.index))
    print(f"{'レースごとの本命(win_prob最大)':30s} {n:7d} {bet:10d} {ret:10d} {ret-bet:+10d} {roi:7.1f}%")

    for threshold in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
        n, bet, ret, roi = simulate(valid_df["ev"] > threshold)
        print(f"{'EV > ' + str(threshold):30s} {n:7d} {bet:10d} {ret:10d} {ret-bet:+10d} {roi:7.1f}%")


if __name__ == "__main__":
    main()
