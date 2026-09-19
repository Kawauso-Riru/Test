#!/usr/bin/env python
"""Follow-up robustness check on the two candidates from
longshot_axis_scan.py ("30倍以上 band, model's top pick" restricted to
長距離 dirt races, and 福島 venue):

  1. Pools MANY seeds (not just the original 6) on 長距離 alone, to see if
     the >100% result survives a bigger, more robust sample -- this
     project's usual bar for calling something "real" rather than a
     TUNE/TEST-split coincidence.
  2. Tests the 福島×長距離 combination on a completely FRESH set of TUNE/TEST
     seeds that were never used in longshot_axis_scan.py or
     longshot_segment_tune_test.py, so this isn't just re-slicing the same
     seeds a third time.

Usage:
    python scripts/longshot_distance_followup.py --pool-seeds 1,2,3,4,5,6,7,8,9,10,11,99 \
        --fresh-tune-seeds 21,22,23 --fresh-test-seeds 24,25,26
"""
import argparse
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
            "place": pick["place"], "distance_band": pick["distance_band"],
        })
    return pd.DataFrame(rows)


def roi(df: pd.DataFrame, col: str, unit: int) -> tuple:
    n = len(df)
    if n == 0:
        return 0, float("nan")
    return n, df[col].sum() / (n * unit) * 100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/jra_results.csv")
    parser.add_argument("--oikiri", default="data/oikiri.csv")
    parser.add_argument("--unit", type=int, default=100)
    parser.add_argument("--pool-seeds", default="1,2,3,4,5,6,7,8,9,10,11,99")
    parser.add_argument("--fresh-tune-seeds", default="21,22,23")
    parser.add_argument("--fresh-test-seeds", default="24,25,26")
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

    pool_seeds = [int(s) for s in args.pool_seeds.split(",")]
    fresh_tune_seeds = [int(s) for s in args.fresh_tune_seeds.split(",")]
    fresh_test_seeds = [int(s) for s in args.fresh_test_seeds.split(",")]
    all_seeds = sorted(set(pool_seeds) | set(fresh_tune_seeds) | set(fresh_test_seeds))

    all_dfs = {}
    for seed in all_seeds:
        df = evaluate_split(fit_df, feature_columns, seed, scraper, args.unit)
        all_dfs[seed] = df
        print(f"seed {seed}: {len(df)} picks (30倍以上帯)")

    # --- 1) big pooled robustness check on 長距離 alone ---
    pooled = pd.concat([all_dfs[s] for s in pool_seeds], ignore_index=True)
    long_only = pooled[pooled["distance_band"] == "長距離"]
    print(f"\n=== 長距離×30倍以上・モデル選定 大規模プール検証 ({len(pool_seeds)}シード) ===")
    print(f"{'seed':>6s} {'n':>6s} {'複勝ROI':>9s}")
    for s in pool_seeds:
        sub = all_dfs[s]
        sub = sub[sub["distance_band"] == "長距離"]
        n, f_roi = roi(sub, "f_ret", args.unit)
        print(f"{s:>6d} {n:>6d} {f_roi:>8.1f}%")
    n_pool, f_roi_pool = roi(long_only, "f_ret", args.unit)
    _, t_roi_pool = roi(long_only, "t_ret", args.unit)
    print(f"\nプール合計: n={n_pool}  単勝ROI={t_roi_pool:.1f}%  複勝ROI={f_roi_pool:.1f}%")

    # --- 2) fresh TUNE/TEST check of 福島×長距離 combo ---
    tune_df = pd.concat([all_dfs[s] for s in fresh_tune_seeds], ignore_index=True)
    test_df = pd.concat([all_dfs[s] for s in fresh_test_seeds], ignore_index=True)

    print(f"\n=== 福島×長距離 組み合わせ、新規シードでTUNE/TEST検証 ===")
    print(f"TUNE seeds={fresh_tune_seeds}  TEST seeds={fresh_test_seeds}")

    tune_combo = tune_df[(tune_df["place"] == "福島") & (tune_df["distance_band"] == "長距離")]
    test_combo = test_df[(test_df["place"] == "福島") & (test_df["distance_band"] == "長距離")]
    n_tune, f_roi_tune = roi(tune_combo, "f_ret", args.unit)
    _, t_roi_tune = roi(tune_combo, "t_ret", args.unit)
    n_test, f_roi_test = roi(test_combo, "f_ret", args.unit)
    _, t_roi_test = roi(test_combo, "t_ret", args.unit)
    print(f"TUNE側: n={n_tune}  単勝ROI={t_roi_tune:.1f}%  複勝ROI={f_roi_tune:.1f}%")
    print(f"TEST側: n={n_test}  単勝ROI={t_roi_test:.1f}%  複勝ROI={f_roi_test:.1f}%")

    print(f"\n(参考) 長距離のみ・福島のみ 単体の新規シードでの数字")
    for label, cond_tune, cond_test in [
        ("長距離のみ", tune_df["distance_band"] == "長距離", test_df["distance_band"] == "長距離"),
        ("福島のみ", tune_df["place"] == "福島", test_df["place"] == "福島"),
    ]:
        nt, ft = roi(tune_df[cond_tune], "f_ret", args.unit)
        ne, fe = roi(test_df[cond_test], "f_ret", args.unit)
        print(f"{label}: TUNE n={nt} 複勝{ft:.1f}%  /  TEST n={ne} 複勝{fe:.1f}%")

    if args.out:
        for s, df in all_dfs.items():
            df["pool"] = "pool" if s in pool_seeds else ("fresh_tune" if s in fresh_tune_seeds else "fresh_test")
        pd.concat(all_dfs.values(), ignore_index=True).to_csv(args.out, index=False)
        print(f"\nwrote detail -> {args.out}")


if __name__ == "__main__":
    main()
