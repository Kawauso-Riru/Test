#!/usr/bin/env python
"""Can stacking extra conditions on top of "30倍以上 band, model's top pick
among the long shots" (see longshot_signal_backtest.py -- real hit rate 1.6x
the band average, but ROI still under 100%) push real ROI over 100%?

Searches combinations of: recent run (days_since_last_race), the horse's own
dirt top3 rate, and race_class, using a TUNE/TEST seed split (this project's
standard anti-overfitting discipline -- see venue_specific_backtest.py) so a
threshold that only looks good on the seeds it was picked from doesn't get
reported as if it were real.

Usage:
    python scripts/longshot_segment_tune_test.py --tune-seeds 1,2,3 --test-seeds 4,5,99
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
ODDS_MIN = 30.0


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


def evaluate_split(fit_df: pd.DataFrame, feature_columns: list, seed: int, scraper: PoliteScraper, unit: int) -> pd.DataFrame:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, valid_idx = next(splitter.split(fit_df, fit_df["relevance"], groups=fit_df["race_id"]))
    valid_df = fit_df.iloc[valid_idx].copy()

    model = train_model(fit_df, feature_columns=feature_columns, seed=seed)
    valid_df["score"] = model.predict(valid_df)
    valid_df["hit_top3"] = (valid_df["relevance"] > 0).astype(int)

    rows = []
    for race_id, race_rows in valid_df.groupby("race_id"):
        longshots = race_rows[race_rows["odds_numeric"] >= ODDS_MIN]
        if longshots.empty:
            continue
        pick = longshots.sort_values("score", ascending=False).iloc[0]

        result = fetch_result_with_retry(scraper, race_id)
        if result is None:
            continue
        payout = result.get("payout") or {}
        if not payout:
            continue
        umaban = str(int(pick["umaban"]))
        t_ret = payout_for_umaban(payout, "tansho", umaban) * (unit // 100)
        f_ret = payout_for_umaban(payout, "fukusho", umaban) * (unit // 100)

        rows.append({
            "race_id": race_id, "seed": seed, "hit_top3": pick["hit_top3"],
            "t_ret": t_ret, "f_ret": f_ret,
            "days_since_last_race": pick["days_since_last_race"],
            "horse_dirt_top3_rate_before": pick["horse_dirt_top3_rate_before"],
            "horse_dirt_avg_rank_before": pick["horse_dirt_avg_rank_before"],
            "race_class": pick["race_class"], "field_size": pick["field_size"],
            "odds": pick["odds_numeric"],
        })
    return pd.DataFrame(rows)


def roi(df: pd.DataFrame, col: str, unit: int) -> tuple:
    n = len(df)
    if n == 0:
        return 0, float("nan")
    return n, df[col].sum() / (n * unit) * 100


DAYS_CUTS = [None, 90, 60, 45]
TOP3_CUTS = [None, 0.10, 0.15, 0.20]
CLASS_FILTERS = [None, "未勝利1勝クラス"]


def apply_filter(df: pd.DataFrame, days_cut, top3_cut, class_filter) -> pd.DataFrame:
    sub = df
    if days_cut is not None:
        sub = sub[sub["days_since_last_race"] <= days_cut]
    if top3_cut is not None:
        sub = sub[sub["horse_dirt_top3_rate_before"] >= top3_cut]
    if class_filter == "未勝利1勝クラス":
        sub = sub[sub["race_class"].isin(["未勝利", "1勝クラス"])]
    return sub


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/jra_results.csv")
    parser.add_argument("--oikiri", default="data/oikiri.csv")
    parser.add_argument("--unit", type=int, default=100)
    parser.add_argument("--tune-seeds", default="1,2,3")
    parser.add_argument("--test-seeds", default="4,5,99")
    parser.add_argument("--min-n", type=int, default=15, help="minimum picks required to consider a TUNE combo")
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

    tune_seeds = [int(s) for s in args.tune_seeds.split(",")]
    test_seeds = [int(s) for s in args.test_seeds.split(",")]
    all_seeds = tune_seeds + test_seeds

    all_dfs = []
    for seed in all_seeds:
        df = evaluate_split(fit_df, feature_columns, seed, scraper, args.unit)
        all_dfs.append(df)
        print(f"seed {seed}: {len(df)} picks (30倍以上帯)")

    combined = pd.concat(all_dfs, ignore_index=True)
    if args.out:
        combined.to_csv(args.out, index=False)

    tune_df = combined[combined["seed"].isin(tune_seeds)]
    test_df = combined[combined["seed"].isin(test_seeds)]

    print(f"\n=== TUNE側 (seeds={tune_seeds}, n={len(tune_df)}) で条件を総当たり ===")
    results = []
    for days_cut, top3_cut, class_filter in itertools.product(DAYS_CUTS, TOP3_CUTS, CLASS_FILTERS):
        sub = apply_filter(tune_df, days_cut, top3_cut, class_filter)
        n, f_roi = roi(sub, "f_ret", args.unit)
        if n < args.min_n:
            continue
        _, t_roi = roi(sub, "t_ret", args.unit)
        results.append({
            "days_cut": days_cut, "top3_cut": top3_cut, "class_filter": class_filter,
            "n": n, "tansho_roi": t_roi, "fukusho_roi": f_roi,
        })
    results_df = pd.DataFrame(results).sort_values("fukusho_roi", ascending=False)
    print(results_df.head(15).to_string(index=False))

    if results_df.empty:
        print("\n条件を満たす組み合わせがTUNE側で見つかりませんでした(min-nを下げてください)")
        return

    best = results_df.iloc[0]
    print(f"\n=== TUNE側で最良だった条件をTEST側(seeds={test_seeds})に適用 ===")
    print(f"条件: days<= {best['days_cut']}, dirt_top3>= {best['top3_cut']}, class_filter={best['class_filter']}")
    test_sub = apply_filter(test_df, best["days_cut"], best["top3_cut"], best["class_filter"])
    n_t, t_roi = roi(test_sub, "t_ret", args.unit)
    n_f, f_roi = roi(test_sub, "f_ret", args.unit)
    print(f"TUNE側: n={int(best['n'])} 単勝ROI={best['tansho_roi']:.1f}% 複勝ROI={best['fukusho_roi']:.1f}%")
    print(f"TEST側: n={n_f} 単勝ROI={t_roi:.1f}% 複勝ROI={f_roi:.1f}%")

    print("\n=== 参考: 絞り込みなし(30倍以上・モデル選定のみ)の同一seedでの比較 ===")
    n_f0, f_roi0 = roi(tune_df, "f_ret", args.unit)
    print(f"TUNE側 絞り込みなし: n={n_f0} 複勝ROI={f_roi0:.1f}%")
    n_f0t, f_roi0t = roi(test_df, "f_ret", args.unit)
    print(f"TEST側 絞り込みなし: n={n_f0t} 複勝ROI={f_roi0t:.1f}%")


if __name__ == "__main__":
    main()
