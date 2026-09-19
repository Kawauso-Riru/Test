#!/usr/bin/env python
"""Follow-up to longshot_segment_tune_test.py: scans MORE single axes (venue,
jockey quality, distance band, track condition, field size) for the "30倍以上
band, model's top pick among the long shots" segment, one axis at a time
(not a giant combined grid -- keeps multiple-comparison risk manageable and
each axis's best-looking TUNE value gets checked against TEST independently,
same anti-overfitting discipline as venue_specific_backtest.py).

Usage:
    python scripts/longshot_axis_scan.py --tune-seeds 1,2,3 --test-seeds 4,5,99
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
            "track_condition": pick["track_condition"], "field_size": pick["field_size"],
            "jockey_dirt_top3_rate_before": pick["jockey_dirt_top3_rate_before"],
            "jockey": pick.get("jockey", ""),
        })
    return pd.DataFrame(rows)


def roi(df: pd.DataFrame, col: str, unit: int) -> tuple:
    n = len(df)
    if n == 0:
        return 0, float("nan")
    return n, df[col].sum() / (n * unit) * 100


def scan_categorical(tune_df, test_df, col, unit, min_n):
    print(f"\n--- 軸: {col} (カテゴリ別, TUNE) ---")
    rows = []
    for val, sub in tune_df.groupby(col):
        n, f_roi = roi(sub, "f_ret", unit)
        if n < min_n:
            continue
        _, t_roi = roi(sub, "t_ret", unit)
        rows.append({col: val, "n": n, "tansho_roi": t_roi, "fukusho_roi": f_roi})
    res = pd.DataFrame(rows).sort_values("fukusho_roi", ascending=False)
    print(res.to_string(index=False))
    if res.empty:
        return
    best_val = res.iloc[0][col]
    test_sub = test_df[test_df[col] == best_val]
    n_f, f_roi = roi(test_sub, "f_ret", unit)
    n_t, t_roi = roi(test_sub, "t_ret", unit)
    print(f"TUNE最良: {col}={best_val!r} (n={int(res.iloc[0]['n'])}, 複勝ROI={res.iloc[0]['fukusho_roi']:.1f}%)")
    print(f"→ TEST側で同条件: n={n_f}, 単勝ROI={t_roi:.1f}%, 複勝ROI={f_roi:.1f}%")


def scan_threshold(tune_df, test_df, col, cuts, unit, min_n, higher_is_better=True):
    print(f"\n--- 軸: {col} (閾値別, TUNE) ---")
    rows = []
    for cut in cuts:
        sub = tune_df[tune_df[col] >= cut] if higher_is_better else tune_df[tune_df[col] <= cut]
        n, f_roi = roi(sub, "f_ret", unit)
        if n < min_n:
            continue
        _, t_roi = roi(sub, "t_ret", unit)
        rows.append({"cut": cut, "n": n, "tansho_roi": t_roi, "fukusho_roi": f_roi})
    res = pd.DataFrame(rows).sort_values("fukusho_roi", ascending=False)
    print(res.to_string(index=False))
    if res.empty:
        return
    best_cut = res.iloc[0]["cut"]
    test_sub = test_df[test_df[col] >= best_cut] if higher_is_better else test_df[test_df[col] <= best_cut]
    n_f, f_roi = roi(test_sub, "f_ret", unit)
    n_t, t_roi = roi(test_sub, "t_ret", unit)
    print(f"TUNE最良: {col} cut={best_cut} (n={int(res.iloc[0]['n'])}, 複勝ROI={res.iloc[0]['fukusho_roi']:.1f}%)")
    print(f"→ TEST側で同条件: n={n_f}, 単勝ROI={t_roi:.1f}%, 複勝ROI={f_roi:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/jra_results.csv")
    parser.add_argument("--oikiri", default="data/oikiri.csv")
    parser.add_argument("--unit", type=int, default=100)
    parser.add_argument("--tune-seeds", default="1,2,3")
    parser.add_argument("--test-seeds", default="4,5,99")
    parser.add_argument("--min-n", type=int, default=20)
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

    all_dfs = []
    for seed in tune_seeds + test_seeds:
        df = evaluate_split(fit_df, feature_columns, seed, scraper, args.unit)
        all_dfs.append(df)
        print(f"seed {seed}: {len(df)} picks (30倍以上帯)")

    combined = pd.concat(all_dfs, ignore_index=True)
    if args.out:
        combined.to_csv(args.out, index=False)

    tune_df = combined[combined["seed"].isin(tune_seeds)]
    test_df = combined[combined["seed"].isin(test_seeds)]
    print(f"\nTUNE n={len(tune_df)}  TEST n={len(test_df)}")

    scan_categorical(tune_df, test_df, "place", args.unit, args.min_n)
    scan_categorical(tune_df, test_df, "distance_band", args.unit, args.min_n)
    scan_categorical(tune_df, test_df, "track_condition", args.unit, args.min_n)
    scan_threshold(tune_df, test_df, "jockey_dirt_top3_rate_before", [0.15, 0.20, 0.25, 0.30], args.unit, args.min_n, higher_is_better=True)
    scan_threshold(tune_df, test_df, "field_size", [10, 12, 14], args.unit, args.min_n, higher_is_better=False)


if __name__ == "__main__":
    main()
