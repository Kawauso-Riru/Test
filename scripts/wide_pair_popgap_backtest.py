#!/usr/bin/env python
"""Follow-up to wide_rank_pair_backtest.py: the earlier result showed
rank-DISTANT pairs (e.g. 1位-5位) beat rank-ADJACENT pairs (e.g. 4位-5位)
on real ワイド ROI. Is that really about the model's predicted rank gap, or
is it actually just tracking the market's own popularity gap between the
two horses (i.e. nothing the model added on top of what odds already say)?

Buckets each pair by the ACTUAL popularity gap between its two horses
(already in the historical data -- no extra scraping needed) and reports
ROI per bucket, plus the same broken out by predicted rank-distance to see
if rank-distance still matters *within* a popularity-gap bucket.

Usage:
    python scripts/wide_pair_popgap_backtest.py --seeds 1,2,3,4,5,6,7,8,9,10,11,99
"""
import argparse
import itertools
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
PAIRS = list(itertools.combinations(range(1, N_RANKS + 1), 2))
POPGAP_BINS = [(0, 2), (3, 5), (6, 9), (10, 20)]


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


def pair_return(umaban_a: str, umaban_b: str, payout: dict, unit: int) -> int:
    info = payout.get("wide")
    if not info:
        return 0
    target = frozenset([umaban_a, umaban_b])
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

    rows = []
    for race_id, race_rows in valid_df.groupby("race_id"):
        ranked = race_rows.sort_values("score", ascending=False)
        if len(ranked) < N_RANKS:
            continue
        top = ranked.head(N_RANKS)
        umaban = [str(int(u)) for u in top["umaban"]]
        pop = list(top["popularity_numeric"])

        result = fetch_result_with_retry(scraper, race_id)
        if result is None:
            continue
        payout = result.get("payout") or {}
        if not payout:
            continue

        for i, j in PAIRS:
            ret = pair_return(umaban[i - 1], umaban[j - 1], payout, unit)
            pa, pb = pop[i - 1], pop[j - 1]
            popgap = abs(pa - pb) if pd.notna(pa) and pd.notna(pb) else None
            rows.append({
                "race_id": race_id, "seed": seed, "rank_i": i, "rank_j": j,
                "rank_dist": j - i, "popgap": popgap, "ret": ret, "bet": unit,
            })
    return pd.DataFrame(rows)


def bucket_popgap(g):
    if pd.isna(g):
        return None
    for lo, hi in POPGAP_BINS:
        if lo <= g <= hi:
            return f"{lo}-{hi}"
    return f"{POPGAP_BINS[-1][1]}+"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/jra_results.csv")
    parser.add_argument("--oikiri", default="data/oikiri.csv")
    parser.add_argument("--unit", type=int, default=100)
    parser.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10,11,99")
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
        print(f"seed {seed}: {df['race_id'].nunique()} races processed")

    combined = pd.concat(all_dfs, ignore_index=True)
    if args.out:
        combined.to_csv(args.out, index=False)
        print(f"wrote per-pair detail -> {args.out}")

    combined["popgap_bucket"] = combined["popgap"].apply(bucket_popgap)

    print(f"\n=== 人気差バケット別 ワイドROI (全ペア込み, n_pairs={len(combined)}) ===")
    g = combined.dropna(subset=["popgap_bucket"]).groupby("popgap_bucket").agg(
        n=("ret", "size"), bet=("bet", "sum"), ret=("ret", "sum")
    )
    g["roi"] = g["ret"] / g["bet"] * 100
    order = [f"{lo}-{hi}" for lo, hi in POPGAP_BINS] + [f"{POPGAP_BINS[-1][1]}+"]
    print(g.reindex(order).to_string())

    print(f"\n=== 人気差バケット x 指数順位差 クロス集計 ROI ===")
    cross = combined.dropna(subset=["popgap_bucket"]).groupby(["popgap_bucket", "rank_dist"]).agg(
        n=("ret", "size"), bet=("bet", "sum"), ret=("ret", "sum")
    )
    cross["roi"] = cross["ret"] / cross["bet"] * 100
    print(cross.to_string())


if __name__ == "__main__":
    main()
