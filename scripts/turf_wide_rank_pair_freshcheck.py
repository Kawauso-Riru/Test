#!/usr/bin/env python
"""Fresh-seed re-check of the two rank-pairs that looked best in
wide_rank_pair_backtest.py --turf-only (3位-6位, 1位-6位, both 85.9% ROI on
seeds 1,2,3,4,5,99) -- the standard next step this project always takes
before trusting a "best of N" finding (see fuku3_triplet_deepdive.py for
the dirt-side precedent). Uses seeds never touched by that run.

Usage:
    python scripts/turf_wide_rank_pair_freshcheck.py --seeds 11,12,13
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba_ai.features import TURF_FEATURE_COLUMNS, build_training_frame  # noqa: E402
from keiba_ai.io import read_race_csv  # noqa: E402
from keiba_ai.model import train_model  # noqa: E402
from keiba_ai.scraper import PoliteScraper, RobotsDisallowedError, ScraperConfig  # noqa: E402

MARKET_FEATURE_COLUMNS = {"popularity_numeric", "odds_numeric"}
N_RANKS = 6
TARGET_PAIRS = {"3-6": (3, 6), "1-6": (1, 6), "1-2": (1, 2)}  # 1-2 included as the "obvious" baseline


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


def wide_pair_return(u1: str, u2: str, payout: dict, unit: int) -> int:
    info = payout.get("wide")
    if not info:
        return 0
    target = frozenset([u1, u2])
    for combo, p in zip(info["combos"], info["payouts"]):
        if frozenset(combo) == target:
            return p * (unit // 100)
    return 0


def evaluate_split(fit_df: pd.DataFrame, feature_columns: list, seed: int, scraper: PoliteScraper,
                    unit: int, max_races: int | None) -> pd.DataFrame:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, valid_idx = next(splitter.split(fit_df, fit_df["relevance"], groups=fit_df["race_id"]))
    valid_df = fit_df.iloc[valid_idx].copy()

    model = train_model(fit_df, feature_columns=feature_columns, seed=seed)
    valid_df["score"] = model.predict(valid_df)

    race_ids = valid_df.sort_values("date")["race_id"].drop_duplicates().tolist()
    if max_races:
        race_ids = race_ids[:max_races]

    rows = []
    for race_id in race_ids:
        race_rows = valid_df[valid_df["race_id"] == race_id]
        ranked = race_rows.sort_values("score", ascending=False)
        if len(ranked) < N_RANKS:
            continue
        top = [str(int(u)) for u in ranked.head(N_RANKS)["umaban"]]

        result = fetch_result_with_retry(scraper, race_id)
        if result is None:
            continue
        payout = result.get("payout") or {}
        if not payout:
            continue

        row = {"race_id": race_id, "seed": seed}
        for name, (i, j) in TARGET_PAIRS.items():
            row[f"ret_{name}"] = wide_pair_return(top[i - 1], top[j - 1], payout, unit)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/jra_results.csv")
    parser.add_argument("--oikiri", default="data/oikiri.csv")
    parser.add_argument("--pedigree", default="data/pedigree.csv")
    parser.add_argument("--unit", type=int, default=100)
    parser.add_argument("--seeds", default="11,12,13")
    parser.add_argument("--max-races", type=int, default=250)
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
    pedigree_path = Path(args.pedigree)
    if pedigree_path.exists():
        pedigree = read_race_csv(pedigree_path)[["horse_id", "sire_id", "damsire_id"]]
        raw = raw.merge(pedigree, on="horse_id", how="left")
    training_df = build_training_frame(raw)
    fit_df = training_df[~training_df["is_dirt"]].dropna(subset=["relevance"]).reset_index(drop=True)
    feature_columns = [c for c in TURF_FEATURE_COLUMNS if c not in MARKET_FEATURE_COLUMNS]

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
        df = evaluate_split(fit_df, feature_columns, seed, scraper, args.unit, args.max_races)
        all_dfs.append(df)
        print(f"seed {seed}: {len(df)} races processed")

    combined = pd.concat(all_dfs, ignore_index=True)
    if args.out:
        combined.to_csv(args.out, index=False)

    n = len(combined)
    unit_bet = n * args.unit
    print(f"\n=== フレッシュシードでの再現性チェック (n={n} races, seeds={seeds}) ===")
    for name in TARGET_PAIRS:
        ret = combined[f"ret_{name}"].sum()
        roi = ret / unit_bet * 100
        hits = (combined[f"ret_{name}"] > 0).sum()
        print(f"{name}位: ROI={roi:.1f}% 的中数={hits}")


if __name__ == "__main__":
    main()
