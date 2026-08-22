#!/usr/bin/env python
"""Backtest several betting strategies against the model's held-out validation
races, using the *actual* payout data from each race's result page (not
just "did the top pick place" -- real yen in, real yen out).

Reproduces train_model()'s exact train/valid split (same seed) so the races
backtested here are ones the model never trained on -- otherwise the results
would be unrealistically rosy.

Strategies (bet unit is --unit yen per combination, JRA's real minimum):
  tansho          単勝: rank-1 pick to win
  fukusho         複勝: rank-1 pick to place (top 3)
  wide_box3       ワイドボックス: all pairs among the top 3 picks (3 combos)
  umaren_box3     馬連ボックス: same 3 pairs, exact top-2 finish (unordered)
  wide_nagashi    ワイド1頭軸流し: rank-1 axis + each of ranks 2-7 (6 combos)
  fuku3_box6      3連複ボックス: all trios among the top 6 picks (20 combos)
  fuku3_nagashi   3連複1頭軸流し: rank-1 axis + any 2 of ranks 2-7 (15 combos)
  tan3_nagashi    3連単1頭軸流し(1着固定): rank-1 axis for 1st, ordered
                  pairs from ranks 2-7 for 2nd/3rd (30 combos)

Usage:
    python scripts/backtest_betting_strategies.py --unit 100
"""
import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba_ai.features import ALL_FEATURE_COLUMNS, build_training_frame  # noqa: E402
from keiba_ai.io import read_race_csv  # noqa: E402
from keiba_ai.model import KeibaModel, train_model  # noqa: E402
from keiba_ai.scraper import PoliteScraper, RobotsDisallowedError, ScraperConfig  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

MARKET_FEATURE_COLUMNS = {"popularity_numeric", "odds_numeric"}


def build_strategies(top6: list) -> dict:
    """top6: umaban strings, ranks 1-6 by predicted score (best first)."""
    axis, rest = top6[0], top6[1:]
    top3 = top6[:3]
    return {
        "tansho": [[axis]],
        "fukusho": [[axis]],
        "wide_box3": [list(c) for c in itertools.combinations(top3, 2)],
        "umaren_box3": [list(c) for c in itertools.combinations(top3, 2)],
        "wide_nagashi": [[axis, p] for p in rest],
        "fuku3_box6": [list(c) for c in itertools.combinations(top6, 3)],
        "fuku3_nagashi": [[axis] + list(c) for c in itertools.combinations(rest, 2)],
        "tan3_nagashi": [[axis, a, b] for a, b in itertools.permutations(rest, 2)],
    }


UNORDERED_BET_TYPES = {
    "tansho": "tansho", "fukusho": "fukusho", "wide_box3": "wide", "umaren_box3": "umaren",
    "wide_nagashi": "wide", "fuku3_box6": "fuku3", "fuku3_nagashi": "fuku3",
}
ORDERED_BET_TYPES = {"tan3_nagashi": "tan3"}


def score_strategy(our_combos: list, actual: dict, payout_key: str, ordered: bool, unit: int) -> tuple:
    """Returns (bet_yen, return_yen) for one race's one strategy."""
    bet = len(our_combos) * unit
    info = actual.get(payout_key)
    if not info:
        return bet, 0
    ret = 0
    for actual_combo, actual_payout in zip(info["combos"], info["payouts"]):
        target = tuple(actual_combo) if ordered else frozenset(actual_combo)
        for combo in our_combos:
            candidate = tuple(combo) if ordered else frozenset(combo)
            if candidate == target:
                ret += actual_payout * (unit // 100)
    return bet, ret


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/jra_results.csv")
    parser.add_argument("--oikiri", default="data/oikiri.csv")
    parser.add_argument("--model", default="models/model_dirt.joblib")
    parser.add_argument("--unit", type=int, default=100, help="yen bet per combination")
    parser.add_argument("--min-interval", type=float, default=1.5)
    parser.add_argument("--cache-dir", default="data/cache/netkeiba")
    parser.add_argument("--contact", default="set-your-email-here")
    parser.add_argument("--max-races", type=int, help="debug: cap the number of held-out races processed")
    parser.add_argument("--out", help="optional CSV of per-race, per-strategy results")
    args = parser.parse_args()

    raw = read_race_csv(args.data)
    oikiri_path = Path(args.oikiri)
    if oikiri_path.exists():
        oikiri = read_race_csv(oikiri_path)[["race_id", "horse_id", "training_grade"]]
        raw = raw.merge(oikiri, on=["race_id", "horse_id"], how="left")
    training_df = build_training_frame(raw)
    fit_df = training_df[training_df["is_dirt"]].dropna(subset=["relevance"]).reset_index(drop=True)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, valid_idx = next(splitter.split(fit_df, fit_df["relevance"], groups=fit_df["race_id"]))
    valid_df = fit_df.iloc[valid_idx]
    print(f"held-out validation races: {valid_df['race_id'].nunique()}")

    model = KeibaModel.load(Path(args.model))
    scores = model.predict(valid_df)
    valid_df = valid_df.assign(_score=scores)

    race_ids = valid_df.sort_values("date")["race_id"].drop_duplicates().tolist()
    if args.max_races:
        race_ids = race_ids[: args.max_races]

    scraper = PoliteScraper(
        ScraperConfig(
            user_agent=f"keiba-ai-research-bot/0.1 (+contact: {args.contact})",
            min_interval_sec=args.min_interval,
            cache_dir=Path(args.cache_dir),
        )
    )

    totals = {name: {"bet": 0, "ret": 0} for name in build_strategies(["1"] * 6)}
    detail_rows = []
    skipped = 0

    for i, race_id in enumerate(race_ids, start=1):
        race_rows = valid_df[valid_df["race_id"] == race_id].sort_values("_score", ascending=False)
        top6 = [str(int(u)) for u in race_rows.head(6)["umaban"]]
        if len(top6) < 6:
            skipped += 1
            continue

        try:
            result = scraper.fetch_race_result(f"https://race.netkeiba.com/race/result.html?race_id={race_id}")
        except RobotsDisallowedError:
            skipped += 1
            continue
        payout = result.get("payout") or {}
        if not payout:
            skipped += 1
            continue

        strategies = build_strategies(top6)
        row = {"race_id": race_id}
        for name, combos in strategies.items():
            payout_key = UNORDERED_BET_TYPES.get(name) or ORDERED_BET_TYPES.get(name)
            ordered = name in ORDERED_BET_TYPES
            bet, ret = score_strategy(combos, payout, payout_key, ordered, args.unit)
            totals[name]["bet"] += bet
            totals[name]["ret"] += ret
            row[f"{name}_bet"] = bet
            row[f"{name}_return"] = ret
        detail_rows.append(row)

        if i % 50 == 0:
            print(f"  {i}/{len(race_ids)} races processed")

    print(f"\nprocessed {len(detail_rows)} races, skipped {skipped} (not found / no payout / robots)")
    print(f"\n{'strategy':16s} {'bet':>10s} {'return':>10s} {'profit':>10s} {'ROI':>8s}")
    for name, t in sorted(totals.items(), key=lambda kv: (kv[1]["ret"] / kv[1]["bet"] if kv[1]["bet"] else 0), reverse=True):
        bet, ret = t["bet"], t["ret"]
        roi = ret / bet * 100 if bet else float("nan")
        print(f"{name:16s} {bet:10d} {ret:10d} {ret - bet:+10d} {roi:7.1f}%")

    if args.out and detail_rows:
        pd.DataFrame(detail_rows).to_csv(args.out, index=False)
        print(f"\nwrote per-race detail -> {args.out}")


if __name__ == "__main__":
    main()
