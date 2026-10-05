#!/usr/bin/env python
"""Real-payout ROI backtest for the turf-only model -- the dirt model's
central validation method (train_model()'s held-out split, actual netkeiba
payout data on the #1 pick), applied to turf for the first time. A good
ndcg/precision number (see turf_model_robustness_check.py) only says the
model ranks sensibly on a held-out split; it says nothing about whether that
translates into real money, which is the whole point of this project.

Pools several GroupShuffleSplit seeds (same approach as the dirt research
scripts this session, e.g. fuku3_triplet_deepdive.py) rather than trusting
one split, and samples a capped number of races per seed (--max-races) so a
full run finishes in a reasonable time against netkeiba's real pages.

Usage:
    python scripts/turf_model_roi_backtest.py --seeds 1,2,3 --max-races 300 --unit 100
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
        top1 = race_rows.sort_values("score", ascending=False).iloc[0]
        result = fetch_result_with_retry(scraper, str(race_id))
        if result is None:
            continue
        payout = result.get("payout") or {}
        if not payout:
            continue
        finishers = {str(r["umaban"]): r for r in result.get("entries", [])}
        umaban = str(int(top1["umaban"]))
        finish = finishers.get(umaban)
        if finish is None:
            continue
        try:
            actual_rank = int(finish.get("rank"))
        except (TypeError, ValueError):
            actual_rank = None

        def payout_for(key):
            info = payout.get(key)
            if not info:
                return 0
            for combo, p in zip(info["combos"], info["payouts"]):
                if combo == [umaban]:
                    return p
            return 0

        rows.append({
            "race_id": race_id, "seed": seed,
            "hit_top3": actual_rank is not None and actual_rank <= 3,
            "hit_win": actual_rank == 1,
            "t_ret": payout_for("tansho") * (unit // 100),
            "f_ret": payout_for("fukusho") * (unit // 100),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/jra_results.csv")
    parser.add_argument("--oikiri", default="data/oikiri.csv")
    parser.add_argument("--pedigree", default="data/pedigree.csv")
    parser.add_argument("--unit", type=int, default=100)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--max-races", type=int, default=300, help="cap races fetched per seed")
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
        print(f"seed {seed}: {len(df)} races fetched")

    combined = pd.concat(all_dfs, ignore_index=True)
    if args.out:
        combined.to_csv(args.out, index=False)

    n = len(combined)
    unit_bet = n * args.unit
    hit_top3 = combined["hit_top3"].sum()
    hit_win = combined["hit_win"].sum()
    t_total = combined["t_ret"].sum()
    f_total = combined["f_ret"].sum()
    print(f"\n=== turf #1-pick real-payout backtest (n={n} races, seeds={seeds}) ===")
    print(f"3着内的中率: {hit_top3}/{n} ({hit_top3 / n * 100:.1f}%)")
    print(f"1着的中率(勝率): {hit_win}/{n} ({hit_win / n * 100:.1f}%)")
    print(f"単勝ROI: {t_total / unit_bet * 100:.1f}%")
    print(f"複勝ROI: {f_total / unit_bet * 100:.1f}%")


if __name__ == "__main__":
    main()
