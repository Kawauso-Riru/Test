#!/usr/bin/env python
"""Checks a few more segmentation axes within the best segment found so far
(未勝利+1勝クラス, model's #1 pick == market's actual 1番人気, real 複勝 ROI
88.3% at n=1,947 -- see README) to see whether any further narrows the gap
to 100%: the #1 pick's actual final odds (a continuous version of the
favorite-longshot check, rather than just "is it the #1 favorite"),
distance band, and venue (place).

Averages over --seeds splits, pooling all seeds' held-out races for the
final ROI numbers, same methodology as best_segment_backtest.py /
favorite_longshot_backtest.py.

Usage:
    python scripts/extra_dimensions_backtest.py --seeds 1,2,3,4,5,99
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

    rows = []
    for race_id, race_rows in valid_df.groupby("race_id"):
        top1 = race_rows.sort_values("score", ascending=False).iloc[0]
        top1_umaban = str(int(top1["umaban"]))
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
            "popularity": top1.get("popularity_numeric"), "odds": top1.get("odds_numeric"),
            "distance_band": top1["distance_band"], "place": top1["place"],
            "field_size": top1["field_size"], "f_ret": f_ret,
        })
    return pd.DataFrame(rows)


def roi(returns: pd.Series, unit: int) -> float:
    bet = len(returns) * unit
    return returns.sum() * (unit // 100) / bet * 100 if bet else float("nan")


def report(df: pd.DataFrame, group_col: str, unit: int, order: list = None) -> None:
    print(f"\n--- {group_col} ---")
    print(f"{'value':>16s} {'n':>6s} {'複勝ROI':>9s}")
    keys = order or sorted(df[group_col].dropna().unique())
    for key in keys:
        sub = df[df[group_col] == key]
        if len(sub) < 10:
            continue
        print(f"{str(key):>16s} {len(sub):>6d} {roi(sub['f_ret'], unit):>8.1f}%")


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
        df = evaluate_split(fit_df, feature_columns, seed, scraper)
        df["seed"] = seed
        all_dfs.append(df)
        print(f"seed {seed}: {len(df)} races processed")

    combined = pd.concat(all_dfs, ignore_index=True)
    if args.out:
        combined.to_csv(args.out, index=False)
        print(f"wrote per-race detail -> {args.out}")

    segment = combined[combined["race_class"].isin(LOW_CLASSES) & (combined["popularity"] == 1)]
    print(f"\n=== 未勝利+1勝クラス AND 予測1位=市場1番人気 (n={len(segment)}) 内での軸別分析 ===")

    odds_bins = [0, 1.5, 2.0, 2.5, 3.5, 999]
    odds_labels = ["<1.5倍", "1.5-2.0倍", "2.0-2.5倍", "2.5-3.5倍", "3.5倍以上"]
    segment = segment.copy()
    segment["odds_band"] = pd.cut(segment["odds"], bins=odds_bins, labels=odds_labels)
    report(segment, "odds_band", args.unit, order=odds_labels)
    report(segment, "distance_band", args.unit, order=["短距離", "マイル", "長距離"])
    report(segment, "place", args.unit)

    field_bins = [0, 10, 14, 99]
    field_labels = ["小頭数(〜10頭)", "中頭数(11-14頭)", "多頭数(15頭〜)"]
    segment["field_band"] = pd.cut(segment["field_size"], bins=field_bins, labels=field_labels)
    report(segment, "field_band", args.unit, order=field_labels)

    print(f"\n{'ALL(セグメント全体)':>16s} {len(segment):>6d} {roi(segment['f_ret'], args.unit):>8.1f}%")


if __name__ == "__main__":
    main()
