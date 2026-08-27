#!/usr/bin/env python
"""Pushes on the best segment found so far (未勝利+1勝クラス, model's #1
pick == the market's actual 1番人気) to see if anything gets it over 100%
ROI: checks 単勝/複勝/ワイドボックス/馬連ボックス within that segment, and
whether stacking a further confidence filter on top helps now that the base
segment itself is a large, low-variance sample (unlike the earlier noisy
confidence-only filter -- see README).

Averages over --seeds splits from the start, pooling all seeds' held-out
races together for the final ROI numbers (larger effective sample), while
also reporting the per-seed spread so a real effect can be told from noise.

Usage:
    python scripts/best_segment_backtest.py --seeds 1,2,3,4,5,99
"""
import argparse
import itertools
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


def payout_for(payout: dict, key: str, combo: list) -> int:
    info = payout.get(key)
    if not info:
        return 0
    for c, p in zip(info["combos"], info["payouts"]):
        if frozenset(c) == frozenset(combo):
            return p
    return 0


def evaluate_split(fit_df: pd.DataFrame, feature_columns: list, seed: int, scraper: PoliteScraper) -> pd.DataFrame:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, valid_idx = next(splitter.split(fit_df, fit_df["relevance"], groups=fit_df["race_id"]))
    valid_df = fit_df.iloc[valid_idx].copy()

    model = train_model(fit_df, feature_columns=feature_columns, seed=seed)
    valid_df["score"] = model.predict(valid_df)
    valid_df["top3_prob"] = model.predict_top3_probability(valid_df)

    rows = []
    for race_id, race_rows in valid_df.groupby("race_id"):
        ranked = race_rows.sort_values("score", ascending=False)
        top1, top2, top3 = ranked.iloc[0], ranked.iloc[1], ranked.iloc[2]
        top1_umaban = str(int(top1["umaban"]))
        top3_umaban = [str(int(r["umaban"])) for r in [top1, top2, top3]]

        result = fetch_result_with_retry(scraper, race_id)
        if result is None:
            continue
        payout = result.get("payout") or {}
        if not payout:
            continue

        t_ret = payout_for(payout, "tansho", [top1_umaban])
        f_ret = payout_for(payout, "fukusho", [top1_umaban])
        wide_ret = sum(payout_for(payout, "wide", list(c)) for c in itertools.combinations(top3_umaban, 2))
        umaren_ret = sum(payout_for(payout, "umaren", list(c)) for c in itertools.combinations(top3_umaban, 2))

        rows.append({
            "race_id": race_id, "race_class": top1["race_class"],
            "popularity": top1.get("popularity_numeric"), "top3_prob": top1["top3_prob"],
            "t_ret": t_ret, "f_ret": f_ret, "wide_ret": wide_ret, "umaren_ret": umaren_ret,
        })
    return pd.DataFrame(rows)


def roi(returns: pd.Series, n_bets_per_race: int, unit: int) -> float:
    bet = len(returns) * n_bets_per_race * unit
    return returns.sum() * (unit // 100) / bet * 100 if bet else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/jra_results.csv")
    parser.add_argument("--oikiri", default="data/oikiri.csv")
    parser.add_argument("--unit", type=int, default=100)
    parser.add_argument("--seeds", default="1,2,3,4,5,99")
    parser.add_argument("--min-interval", type=float, default=1.5)
    parser.add_argument("--cache-dir", default="data/cache/netkeiba")
    parser.add_argument("--contact", default="set-your-email-here")
    parser.add_argument("--out", help="optional CSV of per-race detail (all seeds pooled)")
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
    segment = combined[
        combined["race_class"].isin(LOW_CLASSES) & (combined["popularity"] == 1)
    ]
    print(f"\n=== 未勝利+1勝クラス AND 予測1位=市場1番人気 (n={len(segment)}, {len(seeds)}シードプール) ===")
    print(f"{'bet type':>12s} {'ROI':>8s}")
    print(f"{'単勝':>12s} {roi(segment['t_ret'], 1, args.unit):>7.1f}%")
    print(f"{'複勝':>12s} {roi(segment['f_ret'], 1, args.unit):>7.1f}%")
    print(f"{'ワイドBOX(上位3頭)':>12s} {roi(segment['wide_ret'], 3, args.unit):>7.1f}%")
    print(f"{'馬連BOX(上位3頭)':>12s} {roi(segment['umaren_ret'], 3, args.unit):>7.1f}%")

    print(f"\n=== さらに自信度で絞り込むと? (セグメント内, n={len(segment)}) ===")
    print(f"{'confidence quantile':>20s} {'n':>6s} {'複勝ROI':>9s} {'per-seed std':>13s}")
    for q in [0.0, 0.3, 0.5, 0.7, 0.85, 0.9, 0.95]:
        thresh = segment["top3_prob"].quantile(q)
        sub = segment[segment["top3_prob"] >= thresh]
        f_roi = roi(sub["f_ret"], 1, args.unit)
        per_seed = []
        for seed in seeds:
            seed_sub = sub[sub["seed"] == seed]
            if len(seed_sub) >= 5:
                per_seed.append(roi(seed_sub["f_ret"], 1, args.unit))
        std = np.std(per_seed) if len(per_seed) >= 2 else float("nan")
        print(f"{f'q>={q:.1f}({thresh:.3f})':>20s} {len(sub):>6d} {f_roi:>8.1f}% {std:>12.1f}pt")


if __name__ == "__main__":
    main()
