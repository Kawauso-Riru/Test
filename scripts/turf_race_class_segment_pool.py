#!/usr/bin/env python
"""Multi-seed re-check of the turf race_class_backtest.py finding (未勝利/
1勝クラス standing out, tansho ROI ~98% at n=120 on a single split). That
single split isn't enough to trust -- this project's README documents
several dirt-side single-split "finds" collapsing under multi-seed or
fresh-seed re-verification (the fuku3 rank-triplet work especially). This
retrains the turf model across several GroupShuffleSplit seeds (not just
race_class_backtest.py's fixed seed=42), restricts each split's held-out set
to 未勝利/1勝クラス races, and pools real payouts across all of them.

Usage:
    python scripts/turf_race_class_segment_pool.py --seeds 1,2,3,4,5,99
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
TARGET_CLASSES = {"未勝利", "1勝クラス"}


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

    # Restrict to the target classes AFTER scoring -- the model itself is
    # still trained on the full turf split, only the bet-evaluation set is
    # filtered, matching how race_class_backtest.py reports per-class ROI
    # from one model rather than training a separate model per class.
    target_df = valid_df[valid_df["race_class"].isin(TARGET_CLASSES)]
    race_ids = target_df.sort_values("date")["race_id"].drop_duplicates().tolist()
    if max_races:
        race_ids = race_ids[:max_races]

    rows = []
    for race_id in race_ids:
        race_rows = target_df[target_df["race_id"] == race_id].sort_values("score", ascending=False)
        if race_rows.empty:
            continue
        top1 = race_rows.iloc[0]
        result = fetch_result_with_retry(scraper, str(race_id))
        if result is None:
            continue
        payout = result.get("payout") or {}
        if not payout:
            continue
        umaban = str(int(top1["umaban"]))
        finishers = {str(r["umaban"]): r for r in result.get("entries", [])}
        finish = finishers.get(umaban)
        try:
            actual_rank = int(finish.get("rank")) if finish else None
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
            "race_id": race_id, "seed": seed, "race_class": top1["race_class"],
            "hit_top3": actual_rank is not None and actual_rank <= 3,
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
    parser.add_argument("--seeds", default="1,2,3,4,5,99")
    parser.add_argument("--max-races", type=int, default=250, help="cap races fetched per seed")
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
        print(f"seed {seed}: {len(df)} races (未勝利/1勝クラス) processed")

    combined = pd.concat(all_dfs, ignore_index=True)
    if args.out:
        combined.to_csv(args.out, index=False)

    print(f"\n=== pooled 未勝利+1勝クラス, n={len(combined)} races, seeds={seeds} ===")
    for cls in sorted(TARGET_CLASSES):
        sub = combined[combined["race_class"] == cls]
        if sub.empty:
            continue
        n = len(sub)
        hit = sub["hit_top3"].sum()
        t_roi = sub["t_ret"].sum() / (n * args.unit) * 100
        f_roi = sub["f_ret"].sum() / (n * args.unit) * 100
        print(f"{cls}: n={n} hit={hit}/{n}({hit/n*100:.1f}%) tansho_roi={t_roi:.1f}% fukusho_roi={f_roi:.1f}%")

    n = len(combined)
    hit = combined["hit_top3"].sum()
    t_roi = combined["t_ret"].sum() / (n * args.unit) * 100
    f_roi = combined["f_ret"].sum() / (n * args.unit) * 100
    print(f"ALL (未勝利+1勝クラス combined): n={n} hit={hit}/{n}({hit/n*100:.1f}%) tansho_roi={t_roi:.1f}% fukusho_roi={f_roi:.1f}%")


if __name__ == "__main__":
    main()
