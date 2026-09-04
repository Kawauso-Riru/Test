#!/usr/bin/env python
"""When the model's #1 pick is NOT the market's actual favorite (the
"disagreement" races -- see favorite_longshot_backtest.py for the mirror
case), which predicted rank (1st through 6th by score) has the best real
ROI? Also checks betting the actual market favorite itself (whatever rank
the model gave it) as a baseline comparison.

Held-out races only (same GroupShuffleSplit machinery as the rest of this
project's backtests), so this never touches races the model trained on.

Usage:
    python scripts/disagreement_rank_backtest.py --seeds 1,2,3
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba_ai.features import ALL_FEATURE_COLUMNS, build_training_frame  # noqa: E402
from keiba_ai.io import read_race_csv  # noqa: E402
from keiba_ai.model import train_model  # noqa: E402
from keiba_ai.scraper import PoliteScraper, RobotsDisallowedError, ScraperConfig  # noqa: E402

MARKET_FEATURE_COLUMNS = {"popularity_numeric", "odds_numeric"}
N_RANKS = 6


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


def evaluate_split(fit_df: pd.DataFrame, feature_columns: list, seed: int, scraper: PoliteScraper) -> pd.DataFrame:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, valid_idx = next(splitter.split(fit_df, fit_df["relevance"], groups=fit_df["race_id"]))
    valid_df = fit_df.iloc[valid_idx].copy()

    model = train_model(fit_df, feature_columns=feature_columns, seed=seed)
    valid_df["score"] = model.predict(valid_df)

    rows = []
    for race_id, race_rows in valid_df.groupby("race_id"):
        ranked = race_rows.sort_values("score", ascending=False)
        if len(ranked) < N_RANKS:
            continue
        top1 = ranked.iloc[0]
        if top1.get("popularity_numeric") == 1:
            continue  # agreement race -- not what we're checking here

        result = fetch_result_with_retry(scraper, race_id)
        if result is None:
            continue
        payout = result.get("payout") or {}
        if not payout:
            continue

        row = {"race_id": race_id, "seed": seed}
        for rank in range(1, N_RANKS + 1):
            umaban = str(int(ranked.iloc[rank - 1]["umaban"]))
            row[f"rank{rank}_tansho"] = payout_for_umaban(payout, "tansho", umaban)
            row[f"rank{rank}_fukusho"] = payout_for_umaban(payout, "fukusho", umaban)

        fav_rows = ranked[ranked["popularity_numeric"] == 1]
        if len(fav_rows):
            fav_umaban = str(int(fav_rows.iloc[0]["umaban"]))
            row["favorite_tansho"] = payout_for_umaban(payout, "tansho", fav_umaban)
            row["favorite_fukusho"] = payout_for_umaban(payout, "fukusho", fav_umaban)
        else:
            row["favorite_tansho"] = None
            row["favorite_fukusho"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def roi(df: pd.DataFrame, col: str, unit: int) -> tuple:
    sub = df[col].dropna()
    n = len(sub)
    if n == 0:
        return 0, float("nan")
    return n, sub.sum() * (unit // 100) / (n * unit) * 100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/jra_results.csv")
    parser.add_argument("--oikiri", default="data/oikiri.csv")
    parser.add_argument("--unit", type=int, default=100)
    parser.add_argument("--seeds", default="1,2,3")
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
        all_dfs.append(df)
        print(f"seed {seed}: {len(df)} disagreement races processed")

    combined = pd.concat(all_dfs, ignore_index=True)
    if args.out:
        combined.to_csv(args.out, index=False)
        print(f"wrote per-race detail -> {args.out}")

    print(f"\n=== 「指数1位 != 市場1番人気」レースのみ (n={len(combined)}, {len(seeds)}シードプール) ===")
    print(f"{'':>18s} {'n':>6s} {'単勝ROI':>9s} {'複勝ROI':>9s}")
    for rank in range(1, N_RANKS + 1):
        n_t, t_roi = roi(combined, f"rank{rank}_tansho", args.unit)
        n_f, f_roi = roi(combined, f"rank{rank}_fukusho", args.unit)
        print(f"{'指数' + str(rank) + '位':>18s} {n_t:>6d} {t_roi:>8.1f}% {f_roi:>8.1f}%")
    n_t, t_roi = roi(combined, "favorite_tansho", args.unit)
    n_f, f_roi = roi(combined, "favorite_fukusho", args.unit)
    print(f"{'市場1番人気':>18s} {n_t:>6d} {t_roi:>8.1f}% {f_roi:>8.1f}%")


if __name__ == "__main__":
    main()
