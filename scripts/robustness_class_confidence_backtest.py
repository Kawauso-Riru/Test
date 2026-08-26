#!/usr/bin/env python
"""Multi-seed robustness check for the "未勝利+1勝クラス, high-confidence"
race filter found by race_class_backtest.py + confidence_filter_backtest.py.

A single held-out split (seed=42, 852 races) showed that races the model
picked with high confidence WITHIN 未勝利/1勝クラス class had real-payout
fukusho ROI approaching 100% at the tightest filter (n=37) -- but a sample
that small could easily be single-split luck rather than a real effect.
This retrains the model on --seeds different train/valid splits (same
GroupShuffleSplit machinery as train_model(), just a different
random_state each time) and re-applies the same filter to each split's own
held-out races, reporting per-seed and averaged ROI/hit-rate so the finding
can be judged against split-to-split noise rather than a single lucky draw.

Race result pages (and therefore payout data) for every dirt race were
already fetched while building data/jra_results.csv itself, so this should
run entirely off the local cache with no new network requests in the
common case.

Usage:
    python scripts/robustness_class_confidence_backtest.py --seeds 1,2,3,4,5
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


def evaluate_split(fit_df: pd.DataFrame, feature_columns: list, seed: int, scraper: PoliteScraper) -> pd.DataFrame:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, valid_idx = next(splitter.split(fit_df, fit_df["relevance"], groups=fit_df["race_id"]))
    valid_df = fit_df.iloc[valid_idx].copy()

    model = train_model(fit_df, feature_columns=feature_columns, seed=seed)
    valid_df["score"] = model.predict(valid_df)
    valid_df["top3_prob"] = model.predict_top3_probability(valid_df)

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
            "top3_prob": top1["top3_prob"], "hit": hit, "f_ret": f_ret,
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
    parser.add_argument("--top-quantile", type=float, default=0.95,
                         help="within the low-class subset, keep races with top3_prob >= this quantile")
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
    print(f"{'seed':>5} {'all n':>7} {'all ROI':>8} {'low-class n':>12} {'low-class ROI':>14} "
          f"{'filtered n':>11} {'filtered hit%':>14} {'filtered ROI':>13}")
    all_results = []
    for seed in seeds:
        df = evaluate_split(fit_df, feature_columns, seed, scraper)
        n_all, hit_all, roi_all = summarize(df, args.unit)
        low = df[df["race_class"].isin(LOW_CLASSES)]
        n_low, hit_low, roi_low = summarize(low, args.unit)
        thresh = low["top3_prob"].quantile(args.top_quantile) if len(low) else float("nan")
        filtered = low[low["top3_prob"] >= thresh]
        n_f, hit_f, roi_f = summarize(filtered, args.unit)
        print(f"{seed:>5} {n_all:>7} {roi_all:>7.1f}% {n_low:>12} {roi_low:>13.1f}% "
              f"{n_f:>11} {hit_f:>13.1f}% {roi_f:>12.1f}%")
        all_results.append((seed, n_all, roi_all, n_low, roi_low, n_f, hit_f, roi_f))

    arr = np.array([r[4] for r in all_results])  # low-class ROI per seed
    farr = np.array([r[7] for r in all_results])  # filtered ROI per seed
    print(f"\n未勝利+1勝クラスのみ: 平均ROI={arr.mean():.1f}%, 標準偏差={arr.std():.1f}pt "
          f"(範囲 {arr.min():.1f}%〜{arr.max():.1f}%)")
    print(f"さらに自信度上位{(1 - args.top_quantile) * 100:.0f}%に絞り込み: "
          f"平均ROI={farr.mean():.1f}%, 標準偏差={farr.std():.1f}pt "
          f"(範囲 {farr.min():.1f}%〜{farr.max():.1f}%)")


if __name__ == "__main__":
    main()
