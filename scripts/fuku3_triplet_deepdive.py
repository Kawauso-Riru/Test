#!/usr/bin/env python
"""Deep-dive on three rank-triplets flagged by fuku3_rank_triplet_backtest.py
(3-6-8位, 3-4-7位, 1-5-6位): re-checks their real ROI on a FRESH set of
seeds never used in that scan (robustness check), and separately looks at
the calibrated top3_probability "shape" of races where each triplet
actually hit vs. didn't, to see if there's a recognizable pattern (e.g. a
particular gap or spread signature) rather than just a bare ROI number.

Usage:
    python scripts/fuku3_triplet_deepdive.py --seeds 21,22,23,24,25,26
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba_ai.features import ALL_FEATURE_COLUMNS, build_training_frame  # noqa: E402
from keiba_ai.io import read_race_csv  # noqa: E402
from keiba_ai.model import train_model  # noqa: E402
from keiba_ai.scraper import PoliteScraper, RobotsDisallowedError, ScraperConfig  # noqa: E402

MARKET_FEATURE_COLUMNS = {"popularity_numeric", "odds_numeric"}
N_RANKS = 8
TARGET_TRIPLETS = {
    "3-6-8": (3, 6, 8),
    "3-4-7": (3, 4, 7),
    "1-5-6": (1, 5, 6),
}


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


def triplet_return(umabans: list, payout: dict, unit: int) -> int:
    info = payout.get("fuku3")
    if not info:
        return 0
    target = frozenset(umabans)
    for combo, p in zip(info["combos"], info["payouts"]):
        if frozenset(combo) == target:
            return p * (unit // 100)
    return 0


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
        if len(ranked) < N_RANKS:
            continue
        top = [str(int(u)) for u in ranked.head(N_RANKS)["umaban"]]
        probs = ranked.head(N_RANKS)["top3_prob"].to_numpy()

        result = fetch_result_with_retry(scraper, race_id)
        if result is None:
            continue
        payout = result.get("payout") or {}
        if not payout:
            continue

        row = {
            "race_id": race_id, "seed": seed,
            "spread8": float(np.std(probs)), "top1_prob": probs[0], "gap12": probs[0] - probs[1],
        }
        for p_idx in range(N_RANKS):
            row[f"p{p_idx + 1}"] = probs[p_idx]
        for name, (i, j, k) in TARGET_TRIPLETS.items():
            ret = triplet_return([top[i - 1], top[j - 1], top[k - 1]], payout, unit)
            row[f"ret_{name}"] = ret
            row[f"gap_{name}"] = probs[i - 1] - probs[k - 1]  # spread within the triplet's own ranks
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/jra_results.csv")
    parser.add_argument("--oikiri", default="data/oikiri.csv")
    parser.add_argument("--unit", type=int, default=100)
    parser.add_argument("--seeds", default="21,22,23,24,25,26")
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

    n = len(combined)
    unit_bet = n * args.unit
    print(f"\n=== フレッシュシードでの再現性チェック (n={n} races, seeds={seeds}) ===")
    for name in TARGET_TRIPLETS:
        ret = combined[f"ret_{name}"].sum()
        roi = ret / unit_bet * 100
        hits = (combined[f"ret_{name}"] > 0).sum()
        print(f"{name}位: ROI={roi:.1f}% 的中数={hits}")

    print(f"\n=== 的中時 vs 非的中時の指数ポイント特徴 ===")
    for name in TARGET_TRIPLETS:
        hit_mask = combined[f"ret_{name}"] > 0
        print(f"\n--- {name}位 (的中n={hit_mask.sum()}, 非的中n={(~hit_mask).sum()}) ---")
        for col in ["spread8", "top1_prob", "gap12", f"gap_{name}"]:
            hit_mean = combined.loc[hit_mask, col].mean()
            miss_mean = combined.loc[~hit_mask, col].mean()
            print(f"  {col}: 的中時={hit_mean:.4f}  非的中時={miss_mean:.4f}  差={hit_mean - miss_mean:+.4f}")


if __name__ == "__main__":
    main()
