#!/usr/bin/env python
"""How does ワイドBOX's real ROI change as the box size grows (top 3/4/5/6
picks)? More horses in the box means more combos (more chances to hit) but
also more yen bet per race, so the ROI could go either way -- this settles
it with real payout data, pooled across multiple held-out splits.

Usage:
    python scripts/wide_box_size_backtest.py --seeds 1,2,3,4,5,99
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
BOX_SIZES = [3, 4, 5, 6]


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


def wide_return(combos: list, payout: dict, unit: int) -> tuple:
    bet = len(combos) * unit
    info = payout.get("wide")
    if not info:
        return bet, 0
    ret = 0
    target_combos = [frozenset(c) for c in combos]
    for actual_combo, actual_payout in zip(info["combos"], info["payouts"]):
        if frozenset(actual_combo) in target_combos:
            ret += actual_payout * (unit // 100)
    return bet, ret


def evaluate_split(fit_df: pd.DataFrame, feature_columns: list, seed: int, scraper: PoliteScraper, unit: int) -> pd.DataFrame:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, valid_idx = next(splitter.split(fit_df, fit_df["relevance"], groups=fit_df["race_id"]))
    valid_df = fit_df.iloc[valid_idx].copy()

    model = train_model(fit_df, feature_columns=feature_columns, seed=seed)
    valid_df["score"] = model.predict(valid_df)

    rows = []
    for race_id, race_rows in valid_df.groupby("race_id"):
        ranked = race_rows.sort_values("score", ascending=False)
        if len(ranked) < max(BOX_SIZES):
            continue
        top = [str(int(u)) for u in ranked.head(max(BOX_SIZES))["umaban"]]

        result = fetch_result_with_retry(scraper, race_id)
        if result is None:
            continue
        payout = result.get("payout") or {}
        if not payout:
            continue

        row = {"race_id": race_id, "seed": seed}
        for n in BOX_SIZES:
            combos = [list(c) for c in itertools.combinations(top[:n], 2)]
            bet, ret = wide_return(combos, payout, unit)
            row[f"box{n}_bet"] = bet
            row[f"box{n}_ret"] = ret
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
        print(f"wrote per-race detail -> {args.out}")

    print(f"\n=== ワイドBOXサイズ比較 (n={len(combined)} races, {len(seeds)}シードプール) ===")
    print(f"{'box size':>10s} {'combos':>8s} {'n':>6s} {'bet':>10s} {'return':>10s} {'ROI':>8s}")
    for n in BOX_SIZES:
        bet = combined[f"box{n}_bet"].sum()
        ret = combined[f"box{n}_ret"].sum()
        n_combos = n * (n - 1) // 2
        roi = ret / bet * 100 if bet else float("nan")
        print(f"{n:>10d} {n_combos:>8d} {len(combined):>6d} {bet:>10d} {ret:>10d} {roi:7.1f}%")


if __name__ == "__main__":
    main()
