#!/usr/bin/env python
"""Two brand-new angles never checked in this project: does the model's own
#1 pick do better when that horse is moving DOWN in class (格下げ) or
changing distance (延長/短縮) from its previous start, compared to staying
in the same class/distance? Neither signal is currently an input feature
at all -- this checks whether it's worth adding.

Computes each horse's previous race's class (ranked on an ordinal scale) and
distance from the raw historical data (shift(1) within horse_id, sorted by
date -- never sees a horse's own future), then for the model's #1 pick in
each held-out race (pooled across --seeds, market features excluded from
training as usual), reports real 単勝/複勝 ROI broken out by class-change
and distance-change bucket.

Usage:
    python scripts/class_distance_change_backtest.py --seeds 1,2,3,4,5,99
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

from keiba_ai.features import ALL_FEATURE_COLUMNS, _race_class, build_training_frame  # noqa: E402
from keiba_ai.io import read_race_csv  # noqa: E402
from keiba_ai.model import train_model  # noqa: E402
from keiba_ai.scraper import PoliteScraper, RobotsDisallowedError, ScraperConfig  # noqa: E402

MARKET_FEATURE_COLUMNS = {"popularity_numeric", "odds_numeric"}

CLASS_RANK = {
    "新馬": 0, "未勝利": 0, "1勝クラス": 1, "2勝クラス": 2, "3勝クラス": 3,
    "オープン": 4, "リステッド": 4, "G3": 5, "G2": 6, "G1": 7,
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


def payout_for_umaban(payout: dict, key: str, umaban: str) -> int:
    info = payout.get(key)
    if not info:
        return 0
    for combo, p in zip(info["combos"], info["payouts"]):
        if combo == [umaban]:
            return p
    return 0


def add_change_columns(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw["race_class_tmp"] = raw["race_name"].apply(_race_class)
    raw["class_rank_tmp"] = raw["race_class_tmp"].map(CLASS_RANK)
    raw = raw.sort_values(["horse_id", "date"])
    raw["prev_class_rank"] = raw.groupby("horse_id")["class_rank_tmp"].shift(1)
    raw["prev_distance"] = raw.groupby("horse_id")["distance"].shift(1)
    raw["class_rank_change"] = raw["class_rank_tmp"] - raw["prev_class_rank"]
    raw["distance_change"] = raw["distance"] - raw["prev_distance"]
    return raw.drop(columns=["race_class_tmp", "class_rank_tmp"])


def bucket_class(change) -> str:
    if pd.isna(change):
        return "初出走/不明"
    if change < 0:
        return "格下げ"
    if change > 0:
        return "格上げ"
    return "同格"


def bucket_distance(change) -> str:
    if pd.isna(change):
        return "初出走/不明"
    if change <= -200:
        return "短縮(200m超)"
    if change < 0:
        return "短縮(200m以内)"
    if change == 0:
        return "同距離"
    if change <= 200:
        return "延長(200m以内)"
    return "延長(200m超)"


def evaluate_split(fit_df: pd.DataFrame, feature_columns: list, seed: int, scraper: PoliteScraper) -> pd.DataFrame:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, valid_idx = next(splitter.split(fit_df, fit_df["relevance"], groups=fit_df["race_id"]))
    valid_df = fit_df.iloc[valid_idx].copy()

    model = train_model(fit_df, feature_columns=feature_columns, seed=seed)
    valid_df["score"] = model.predict(valid_df)
    valid_df["hit_top3"] = (valid_df["relevance"] > 0).astype(int)

    rows = []
    for race_id, race_rows in valid_df.groupby("race_id"):
        top1 = race_rows.sort_values("score", ascending=False).iloc[0]
        result = fetch_result_with_retry(scraper, race_id)
        if result is None:
            continue
        payout = result.get("payout") or {}
        if not payout:
            continue
        umaban = str(int(top1["umaban"]))
        t_ret = payout_for_umaban(payout, "tansho", umaban) * 1
        f_ret = payout_for_umaban(payout, "fukusho", umaban) * 1
        rows.append({
            "race_id": race_id, "seed": seed, "hit_top3": top1["hit_top3"],
            "t_ret": t_ret, "f_ret": f_ret,
            "class_bucket": bucket_class(top1["class_rank_change"]),
            "distance_bucket": bucket_distance(top1["distance_change"]),
        })
    return pd.DataFrame(rows)


def roi(df: pd.DataFrame, col: str, unit: int) -> tuple:
    n = len(df)
    if n == 0:
        return 0, float("nan")
    return n, df[col].sum() * (unit // 100) / (n * unit) * 100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/jra_results.csv")
    parser.add_argument("--oikiri", default="data/oikiri.csv")
    parser.add_argument("--unit", type=int, default=100)
    parser.add_argument("--seeds", default="1,2,3,4,5,99")
    parser.add_argument("--min-n", type=int, default=20)
    parser.add_argument("--min-interval", type=float, default=1.5)
    parser.add_argument("--cache-dir", default="data/cache/netkeiba")
    parser.add_argument("--contact", default="set-your-email-here")
    parser.add_argument("--out")
    args = parser.parse_args()

    raw = read_race_csv(args.data)
    raw = add_change_columns(raw)
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
        df = evaluate_split(fit_df, feature_columns, seed, scraper)
        all_dfs.append(df)
        print(f"seed {seed}: {len(df)} races processed")

    combined = pd.concat(all_dfs, ignore_index=True)
    if args.out:
        combined.to_csv(args.out, index=False)

    print(f"\n=== モデル1位予想: クラス変動別 実払戻ROI (n={len(combined)}, {len(seeds)}シードプール) ===")
    order = ["格下げ", "同格", "格上げ", "初出走/不明"]
    rows = []
    for b in order:
        sub = combined[combined["class_bucket"] == b]
        n, f_roi = roi(sub, "f_ret", args.unit)
        if n < args.min_n:
            continue
        _, t_roi = roi(sub, "t_ret", args.unit)
        hit = sub["hit_top3"].mean() * 100
        rows.append({"bucket": b, "n": n, "hit_rate": hit, "tansho_roi": t_roi, "fukusho_roi": f_roi})
    print(pd.DataFrame(rows).to_string(index=False))

    print(f"\n=== モデル1位予想: 距離変化別 実払戻ROI ===")
    order2 = ["短縮(200m超)", "短縮(200m以内)", "同距離", "延長(200m以内)", "延長(200m超)", "初出走/不明"]
    rows2 = []
    for b in order2:
        sub = combined[combined["distance_bucket"] == b]
        n, f_roi = roi(sub, "f_ret", args.unit)
        if n < args.min_n:
            continue
        _, t_roi = roi(sub, "t_ret", args.unit)
        hit = sub["hit_top3"].mean() * 100
        rows2.append({"bucket": b, "n": n, "hit_rate": hit, "tansho_roi": t_roi, "fukusho_roi": f_roi})
    print(pd.DataFrame(rows2).to_string(index=False))


if __name__ == "__main__":
    main()
