#!/usr/bin/env python
"""Tests the classic "favorite-longshot bias" (longshots are systematically
over-bet relative to their true win chance; favorites are under-bet -- see
e.g. Thaler & Ziemba 1988) against this model's own picks: does the model's
#1 pick do better, real-payout-wise, when it agrees with the market's own
favorite than when it's picking a market longshot the model likes more than
the crowd does?

For each held-out race, buckets by the #1 pick's *actual* market popularity
rank (人気 -- 1 = the actual favorite, 2 = second choice, etc., taken from
the historical data, never fed to the model itself) and reports hit rate +
real fukusho ROI per bucket -- both overall and within the already-confirmed
未勝利+1勝クラス segment (see race_class_backtest.py /
robustness_class_confidence_backtest.py).

Averages over --seeds different train/valid splits from the start (rather
than reporting one split first and only checking robustness as a follow-up)
given how noisy a single split turned out to be for this project's earlier
narrow-segment backtests.

Usage:
    python scripts/favorite_longshot_backtest.py --seeds 1,2,3,4,5
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
LOW_CLASSES = {"未勝利", "1勝クラス"}


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


def popularity_bucket(pop) -> str:
    if pd.isna(pop):
        return "不明"
    pop = int(pop)
    if pop == 1:
        return "1番人気(市場の本命)"
    if pop == 2:
        return "2番人気"
    if pop == 3:
        return "3番人気"
    return "4番人気以下"


def evaluate_split(fit_df: pd.DataFrame, feature_columns: list, seed: int, scraper: PoliteScraper) -> pd.DataFrame:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, valid_idx = next(splitter.split(fit_df, fit_df["relevance"], groups=fit_df["race_id"]))
    valid_df = fit_df.iloc[valid_idx].copy()

    model = train_model(fit_df, feature_columns=feature_columns, seed=seed)
    valid_df["score"] = model.predict(valid_df)

    rows = []
    for race_id, race_rows in valid_df.groupby("race_id"):
        top1 = race_rows.sort_values("score", ascending=False).iloc[0]
        top1_umaban = str(int(top1["umaban"]))
        top1_rank = top1.get("rank_numeric")
        hit = bool(pd.notna(top1_rank) and top1_rank <= 3)
        result = fetch_result_with_retry(scraper, race_id)
        if result is None:
            continue
        payout = result.get("payout") or {}
        if not payout:
            continue
        f_ret = 0
        fukusho_info = payout.get("fukusho")
        if fukusho_info:
            for combo, p in zip(fukusho_info["combos"], fukusho_info["payouts"]):
                if combo == [top1_umaban]:
                    f_ret = p
                    break
        rows.append({
            "race_id": race_id, "race_class": top1["race_class"],
            "popularity": top1.get("popularity_numeric"), "hit": hit, "f_ret": f_ret,
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, unit: int) -> tuple:
    n = len(df)
    if n == 0:
        return 0, float("nan"), float("nan")
    hit_rate = df["hit"].mean() * 100
    roi = df["f_ret"].sum() * (unit // 100) / (n * unit) * 100
    return n, hit_rate, roi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/jra_results.csv")
    parser.add_argument("--oikiri", default="data/oikiri.csv")
    parser.add_argument("--unit", type=int, default=100)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--min-interval", type=float, default=1.5)
    parser.add_argument("--cache-dir", default="data/cache/netkeiba")
    parser.add_argument("--contact", default="set-your-email-here")
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
    BUCKET_ORDER = ["1番人気(市場の本命)", "2番人気", "3番人気", "4番人気以下", "不明"]

    all_dfs = []
    for seed in seeds:
        df = evaluate_split(fit_df, feature_columns, seed, scraper)
        df["seed"] = seed
        df["bucket"] = df["popularity"].apply(popularity_bucket)
        all_dfs.append(df)
        print(f"seed {seed}: {len(df)} races processed")

    combined = pd.concat(all_dfs, ignore_index=True)
    low_class = combined[combined["race_class"].isin(LOW_CLASSES)]

    for label, data in [("全体", combined), ("未勝利+1勝クラスのみ", low_class)]:
        print(f"\n=== {label} ({len(seeds)}シード合計) ===")
        print(f"{'popularity bucket':>18s} {'n races':>8s} {'top3 hit%':>10s} {'fukusho ROI':>12s} "
              f"{'per-seed ROI std':>17s}")
        for b in BUCKET_ORDER:
            sub = data[data["bucket"] == b]
            if len(sub) == 0:
                continue
            n, hit, roi = summarize(sub, args.unit)
            per_seed_rois = []
            for seed in seeds:
                seed_sub = sub[sub["seed"] == seed]
                if len(seed_sub) >= 5:
                    _, _, seed_roi = summarize(seed_sub, args.unit)
                    per_seed_rois.append(seed_roi)
            std = np.std(per_seed_rois) if len(per_seed_rois) >= 2 else float("nan")
            print(f"{b:>18s} {n:>8d} {hit:>9.1f}% {roi:>11.1f}% {std:>16.1f}pt")
        n, hit, roi = summarize(data, args.unit)
        print(f"{'ALL':>18s} {n:>8d} {hit:>9.1f}% {roi:>11.1f}%")


if __name__ == "__main__":
    main()
