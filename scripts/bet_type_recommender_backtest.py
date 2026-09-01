#!/usr/bin/env python
"""Does the *shape* of a race's top-6 prediction (how concentrated vs. how
spread out the probabilities are) predict which bet type actually pays off
best? If so, that's a genuine per-race "which bet type should I use"
signal -- not just "which bet type is best on average" (already answered:
複勝, see README), but "which bet type is best for *this* race".

For each held-out race (pooled across --seeds splits), computes:
  - shape metrics from the top-6 picks' calibrated top3_probability:
    top1_prob (how confident is the #1 pick alone), gap12 (#1 minus #2,
    a "standout vs. close race" signal), spread6 (std dev across the top
    6 probabilities, "均衡 vs 一強" for the whole field)
  - real payout return for all 8 bet types from
    backtest_betting_strategies.py's build_strategies()

Then reports:
  1. Spearman correlation between each shape metric and each bet type's
     per-race return (does a big gap12 predict tansho/fukusho paying off,
     and a small spread6 predict fuku3_box6 paying off, etc.)
  2. For races bucketed by shape (quartiles), which bet type has the best
     ROI in each bucket -- a direct, actionable "if the race looks like
     X, bet type Y did best historically" table.

Usage:
    python scripts/bet_type_recommender_backtest.py --seeds 1,2,3,4,5,99
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest_betting_strategies import (  # noqa: E402
    ORDERED_BET_TYPES,
    UNORDERED_BET_TYPES,
    build_strategies,
    score_strategy,
)
from keiba_ai.features import ALL_FEATURE_COLUMNS, build_training_frame  # noqa: E402
from keiba_ai.io import read_race_csv  # noqa: E402
from keiba_ai.model import KeibaModel, train_model  # noqa: E402
from keiba_ai.scraper import PoliteScraper, RobotsDisallowedError, ScraperConfig  # noqa: E402

MARKET_FEATURE_COLUMNS = {"popularity_numeric", "odds_numeric"}
BET_TYPES = list(UNORDERED_BET_TYPES) + list(ORDERED_BET_TYPES)


def fetch_result_with_retry(scraper: PoliteScraper, race_id: str, retries: int = 3, backoff: float = 3.0):
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    for attempt in range(retries + 1):
        try:
            return scraper.fetch_race_result(url)
        except RobotsDisallowedError:
            return None
        except requests.RequestException:
            if attempt == retries:
                return None
            time.sleep(backoff * (attempt + 1))
    return None


def evaluate_split(fit_df: pd.DataFrame, feature_columns: list, seed: int, scraper: PoliteScraper, unit: int) -> pd.DataFrame:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, valid_idx = next(splitter.split(fit_df, fit_df["relevance"], groups=fit_df["race_id"]))
    valid_df = fit_df.iloc[valid_idx].copy()

    model = train_model(fit_df, feature_columns=feature_columns, seed=seed)
    valid_df["score"] = model.predict(valid_df)
    valid_df["top3_prob"] = model.predict_top3_probability(valid_df)

    rows = []
    for race_id, race_rows in valid_df.groupby("race_id"):
        ranked = race_rows.sort_values("score", ascending=False)
        if len(ranked) < 6:
            continue
        top6_probs = ranked.head(6)["top3_prob"].values
        top6_umaban = [str(int(u)) for u in ranked.head(6)["umaban"]]

        result = fetch_result_with_retry(scraper, race_id)
        if result is None:
            continue
        payout = result.get("payout") or {}
        if not payout:
            continue

        strategies = build_strategies(top6_umaban)
        row = {
            "race_id": race_id, "seed": seed,
            "top1_prob": float(top6_probs[0]),
            "gap12": float(top6_probs[0] - top6_probs[1]),
            "spread6": float(np.std(top6_probs)),
        }
        for name, combos in strategies.items():
            payout_key = UNORDERED_BET_TYPES.get(name) or ORDERED_BET_TYPES.get(name)
            ordered = name in ORDERED_BET_TYPES
            bet, ret = score_strategy(combos, payout, payout_key, ordered, unit)
            row[f"{name}_bet"] = bet
            row[f"{name}_ret"] = ret
            row[f"{name}_roi"] = ret / bet * 100 if bet else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/jra_results.csv")
    parser.add_argument("--oikiri", default="data/oikiri.csv")
    parser.add_argument("--unit", type=int, default=100)
    parser.add_argument("--seeds", default="1,2,3,4,5,99")
    parser.add_argument("--min-interval", type=float, default=1.5)
    parser.add_argument("--cache-dir", default="data/cache/netkeiba")
    parser.add_argument("--contact", default="set-your-email-here")
    parser.add_argument("--out")
    args = parser.parse_args()

    raw = read_race_csv(args.data)
    oikiri_path = Path(args.oikiri)
    if oikiri_path.exists():
        oikiri = read_race_csv(oikiri_path)[["race_id", "horse_id", "training_grade"]]
        raw = raw.merge(oikiri, on=["race_id", "horse_id"], how="left")
    training_df = build_training_frame(raw)
    fit_df = training_df[training_df["is_dirt"]].dropna(subset=["relevance"]).reset_index(drop=True)
    feature_columns = [c for c in ALL_FEATURE_COLUMNS if c not in MARKET_FEATURE_COLUMNS]

    scraper = PoliteScraper(
        ScraperConfig(
            user_agent=f"keiba-ai-research-bot/0.1 (+contact: {args.contact})",
            min_interval_sec=args.min_interval,
            cache_dir=Path(args.cache_dir),
        )
    )

    seeds = [int(s) for s in args.seeds.split(",")]
    all_dfs = []
    for seed in seeds:
        df = evaluate_split(fit_df, feature_columns, seed, scraper, args.unit)
        all_dfs.append(df)
        print(f"seed {seed}: {len(df)} races processed")

    combined = pd.concat(all_dfs, ignore_index=True)
    if args.out:
        combined.to_csv(args.out, index=False)
        print(f"wrote per-race detail -> {args.out}")

    shape_metrics = ["top1_prob", "gap12", "spread6"]

    print(f"\n=== 形状指標と各買い方のROIとのSpearman相関 (n={len(combined)}) ===")
    print(f"{'':>14s} " + " ".join(f"{bt:>14s}" for bt in BET_TYPES))
    for metric in shape_metrics:
        row_str = f"{metric:>14s} "
        for bt in BET_TYPES:
            corr, _ = spearmanr(combined[metric], combined[f"{bt}_ret"].fillna(0))
            row_str += f"{corr:>14.3f} "
        print(row_str)

    print("\n=== 形状(4分位)ごとに一番ROIが良かった買い方 ===")
    for metric in shape_metrics:
        print(f"\n-- {metric} で4分位 --")
        combined["bucket"] = pd.qcut(combined[metric], 4, labels=False, duplicates="drop")
        for b in sorted(combined["bucket"].dropna().unique()):
            sub = combined[combined["bucket"] == b]
            rois = {}
            for bt in BET_TYPES:
                bet_sum = sub[f"{bt}_bet"].sum()
                ret_sum = sub[f"{bt}_ret"].sum()
                rois[bt] = ret_sum / bet_sum * 100 if bet_sum else float("nan")
            best_bt = max(rois, key=rois.get)
            lo, hi = sub[metric].min(), sub[metric].max()
            print(f"  bucket{int(b)} ({lo:.3f}-{hi:.3f}, n={len(sub)}): 最良={best_bt}({rois[best_bt]:.1f}%)  "
                  + ", ".join(f"{bt}={r:.0f}%" for bt, r in sorted(rois.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
