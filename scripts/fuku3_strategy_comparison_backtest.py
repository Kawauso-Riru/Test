#!/usr/bin/env python
"""Follow-up to the one-day 9/20 anecdote: compares several 3連複
strategies built from the model's top-6 picks (straight single-combo,
1-axis nagashi, 2-axis nagashi, 6-head box) on real payout data, pooled
across multiple held-out splits -- this project's usual robustness check,
since a single day's numbers (like 9/20's "207% from one straight bet")
are far too noisy to trust on their own.

Usage:
    python scripts/fuku3_strategy_comparison_backtest.py --seeds 1,2,3,4,5,99
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


def build_fuku3_strategies(top6: list) -> dict:
    axis1, axis2, rest4 = top6[0], top6[1], top6[2:6]
    rest5 = top6[1:6]
    return {
        "straight_top3": [top6[:3]],
        "nagashi1(軸1頭)": [[axis1] + list(c) for c in itertools.combinations(rest5, 2)],
        "nagashi2(軸2頭)": [[axis1, axis2, p] for p in rest4],
        "box6": [list(c) for c in itertools.combinations(top6, 3)],
    }


def fuku3_return(combos: list, payout: dict, unit: int) -> int:
    info = payout.get("fuku3")
    if not info:
        return 0
    target = [frozenset(c) for c in combos]
    ret = 0
    for combo, p in zip(info["combos"], info["payouts"]):
        if frozenset(combo) in target:
            ret += p * (unit // 100)
    return ret


def evaluate_split(fit_df: pd.DataFrame, feature_columns: list, seed: int, scraper: PoliteScraper, unit: int) -> pd.DataFrame:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, valid_idx = next(splitter.split(fit_df, fit_df["relevance"], groups=fit_df["race_id"]))
    valid_df = fit_df.iloc[valid_idx].copy()

    model = train_model(fit_df, feature_columns=feature_columns, seed=seed)
    valid_df["score"] = model.predict(valid_df)

    rows = []
    for race_id, race_rows in valid_df.groupby("race_id"):
        ranked = race_rows.sort_values("score", ascending=False)
        if len(ranked) < 6:
            continue
        top6 = [str(int(u)) for u in ranked.head(6)["umaban"]]

        result = fetch_result_with_retry(scraper, race_id)
        if result is None:
            continue
        payout = result.get("payout") or {}
        if not payout:
            continue

        strategies = build_fuku3_strategies(top6)
        row = {"race_id": race_id, "seed": seed}
        for name, combos in strategies.items():
            bet = len(combos) * unit
            ret = fuku3_return(combos, payout, unit)
            row[f"{name}_bet"] = bet
            row[f"{name}_ret"] = ret
        rows.append(row)
    return pd.DataFrame(rows)


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
        df = evaluate_split(fit_df, feature_columns, seed, scraper, args.unit)
        all_dfs.append(df)
        print(f"seed {seed}: {len(df)} races processed")

    combined = pd.concat(all_dfs, ignore_index=True)
    if args.out:
        combined.to_csv(args.out, index=False)

    print(f"\n=== 3連複 買い方比較 (n={len(combined)} races, {len(seeds)}シードプール) ===")
    names = ["straight_top3", "nagashi1(軸1頭)", "nagashi2(軸2頭)", "box6"]
    print(f"{'strategy':>18s} {'bet':>10s} {'return':>10s} {'ROI':>8s} {'的中数':>6s}")
    for name in names:
        bet = combined[f"{name}_bet"].sum()
        ret = combined[f"{name}_ret"].sum()
        hits = (combined[f"{name}_ret"] > 0).sum()
        roi = ret / bet * 100 if bet else float("nan")
        print(f"{name:>18s} {bet:>10d} {ret:>10d} {roi:>7.1f}% {hits:>6d}")


if __name__ == "__main__":
    main()
