#!/usr/bin/env python
"""Does the big-upset deep-dive's condition (turf, short distance <=1400m,
big field >=14) actually make known betting picks perform BETTER, worse, or
the same inside it vs outside it? Mirrors the earlier rank-pair/triplet
backtests but splits every candidate's ROI by whether the race matches the
upset-prone condition, pooled across seeds.

Candidates tested (100-yen unit each, real netkeiba payouts):
  - wide "1-2位" (the "obvious" baseline pick)
  - wide "3-6位" (the one candidate that survived two independent
    fresh-seed checks earlier in this project, pooled ROI ~92%)
  - fuku3 box top5 (a simple baseline box strategy)

Usage:
    python scripts/upset_condition_backtest.py --seeds 1,2,3,4,5,99
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

from keiba_ai.features import TURF_FEATURE_COLUMNS, build_training_frame  # noqa: E402
from keiba_ai.io import read_race_csv  # noqa: E402
from keiba_ai.model import train_model  # noqa: E402
from keiba_ai.scraper import PoliteScraper, RobotsDisallowedError, ScraperConfig  # noqa: E402

MARKET_FEATURE_COLUMNS = {"popularity_numeric", "odds_numeric"}
N_RANKS = 6
SHORT_DISTANCE_MAX = 1400
BIG_FIELD_MIN = 14


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


def wide_return(u1: str, u2: str, payout: dict, unit: int) -> int:
    info = payout.get("wide")
    if not info:
        return 0
    target = frozenset([u1, u2])
    for combo, p in zip(info["combos"], info["payouts"]):
        if frozenset(combo) == target:
            return p * (unit // 100)
    return 0


def fuku3_box_return(umabans: list, payout: dict, unit: int) -> int:
    info = payout.get("fuku3")
    if not info:
        return 0
    targets = [frozenset(c) for c in itertools.combinations(umabans, 3)]
    ret = 0
    for combo, p in zip(info["combos"], info["payouts"]):
        if frozenset(combo) in targets:
            ret += p * (unit // 100)
    return ret


def evaluate_split(fit_df: pd.DataFrame, feature_columns: list, seed: int, scraper: PoliteScraper,
                    unit: int, max_races: int | None) -> pd.DataFrame:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, valid_idx = next(splitter.split(fit_df, fit_df["relevance"], groups=fit_df["race_id"]))
    valid_df = fit_df.iloc[valid_idx].copy()

    model = train_model(fit_df, feature_columns=feature_columns, seed=seed)
    valid_df["score"] = model.predict(valid_df)

    race_ids = valid_df.sort_values("date")["race_id"].drop_duplicates().tolist()

    rows = []
    for race_id in race_ids:
        if max_races and len(rows) >= max_races:
            break
        race_rows = valid_df[valid_df["race_id"] == race_id]
        if len(race_rows) < N_RANKS:
            continue
        ranked = race_rows.sort_values("score", ascending=False)
        top = [str(int(u)) for u in ranked.head(N_RANKS)["umaban"]]

        distance = race_rows["distance"].iloc[0]
        field_size = race_rows["field_size"].iloc[0]
        condition = bool(distance <= SHORT_DISTANCE_MAX and field_size >= BIG_FIELD_MIN)

        result = fetch_result_with_retry(scraper, race_id)
        if result is None:
            continue
        payout = result.get("payout") or {}
        if not payout:
            continue

        rows.append({
            "race_id": race_id,
            "seed": seed,
            "condition": condition,
            "distance": distance,
            "field_size": field_size,
            "ret_wide_1_2": wide_return(top[0], top[1], payout, unit),
            "ret_wide_3_6": wide_return(top[2], top[5], payout, unit),
            "ret_fuku3_box5": fuku3_box_return(top[:5], payout, unit),
            "bet_wide": unit,
            "bet_fuku3_box5": unit * 10,  # C(5,3)=10 combos
        })
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, label: str) -> None:
    n = len(df)
    if n == 0:
        print(f"{label}: n=0 races, skipped")
        return
    print(f"\n--- {label} (n={n} races) ---")
    for name, bet_col in [("ret_wide_1_2", "bet_wide"), ("ret_wide_3_6", "bet_wide"), ("ret_fuku3_box5", "bet_fuku3_box5")]:
        bet = df[bet_col].sum()
        ret = df[name].sum()
        hits = (df[name] > 0).sum()
        roi = ret / bet * 100 if bet else float("nan")
        print(f"  {name:>16s}: bet={bet:>8d} ret={ret:>8d} ROI={roi:6.1f}% hits={hits:>4d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/jra_results.csv")
    parser.add_argument("--oikiri", default="data/oikiri.csv")
    parser.add_argument("--unit", type=int, default=100)
    parser.add_argument("--seeds", default="1,2,3,4,5,99")
    parser.add_argument("--max-races", type=int, help="cap races fetched per seed")
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
        n_cond = df["condition"].sum() if len(df) else 0
        print(f"seed {seed}: {len(df)} races processed ({n_cond} matched the upset condition)")

    combined = pd.concat(all_dfs, ignore_index=True)
    if args.out:
        combined.to_csv(args.out, index=False)
        print(f"wrote per-race detail -> {args.out}")

    print(f"\n=== 大穴条件(芝・距離<={SHORT_DISTANCE_MAX}m・出走{BIG_FIELD_MIN}頭以上)別のROI比較 ===")
    report(combined[combined["condition"]], "条件マッチ(短距離×多頭数)")
    report(combined[~combined["condition"]], "条件非マッチ(その他)")
    report(combined, "全体")


if __name__ == "__main__":
    main()
