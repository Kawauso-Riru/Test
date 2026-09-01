#!/usr/bin/env python
"""Does tuning the confidence threshold *per venue* (rather than one
threshold for all 10 JRA courses) actually improve real-payout hit rate,
or does it just overfit to a small per-venue sample?

Captures place (venue), race_class, model confidence (top1's calibrated
top3_probability), and real 複勝 return for every held-out race across
--seeds splits. Splits those seeds into a TUNE half and a TEST half:
picks each venue's best-looking confidence threshold using only the TUNE
seeds, then measures whether that per-venue threshold actually beats a
single uniform threshold on the completely separate TEST seeds. This is
the proper way to check whether "optimize per venue" generalizes, rather
than just reporting the (inevitably rosy) in-sample tuning result.

Usage:
    python scripts/venue_specific_backtest.py --tune-seeds 1,2,3 --test-seeds 4,5,99
"""
import argparse
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


def evaluate_split(fit_df: pd.DataFrame, feature_columns: list, seed: int, scraper: PoliteScraper) -> pd.DataFrame:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, valid_idx = next(splitter.split(fit_df, fit_df["relevance"], groups=fit_df["race_id"]))
    valid_df = fit_df.iloc[valid_idx].copy()

    model = train_model(fit_df, feature_columns=feature_columns, seed=seed)
    valid_df["score"] = model.predict(valid_df)
    valid_df["top3_prob"] = model.predict_top3_probability(valid_df)

    rows = []
    for race_id, race_rows in valid_df.groupby("race_id"):
        top1 = race_rows.sort_values("score", ascending=False).iloc[0]
        top1_umaban = str(int(top1["umaban"]))
        result = fetch_result_with_retry(scraper, race_id)
        if result is None:
            continue
        payout = result.get("payout") or {}
        if not payout:
            continue
        f_ret = 0
        fukusho_info = payout.get("fukusho")
        if fukusho_info:
            for combo, p in zip(fukusho_info["combos"], fukusho_info["payouts"]):
                if combo == [top1_umaban]:
                    f_ret = p
                    break
        rows.append({
            "race_id": race_id, "place": top1["place"], "race_class": top1["race_class"],
            "top3_prob": top1["top3_prob"], "f_ret": f_ret,
        })
    df = pd.DataFrame(rows)
    df["seed"] = seed
    return df


def roi(sub: pd.DataFrame, unit: int = 100) -> float:
    n = len(sub)
    if n == 0:
        return float("nan")
    return sub["f_ret"].sum() * (unit // 100) / (n * unit) * 100


def hit_rate(sub: pd.DataFrame) -> float:
    return sub["f_ret"].gt(0).mean() * 100 if len(sub) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/jra_results.csv")
    parser.add_argument("--oikiri", default="data/oikiri.csv")
    parser.add_argument("--tune-seeds", default="1,2,3")
    parser.add_argument("--test-seeds", default="4,5,99")
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
        df = evaluate_split(fit_df, feature_columns, seed, scraper)
        all_dfs.append(df)
        print(f"seed {seed}: {len(df)} races processed")

    combined = pd.concat(all_dfs, ignore_index=True)
    if args.out:
        combined.to_csv(args.out, index=False)
        print(f"wrote per-race detail -> {args.out}")

    tune_df = combined[combined["seed"].isin(tune_seeds)]
    test_df = combined[combined["seed"].isin(test_seeds)]

    THRESHOLDS = [0.0, 0.3, 0.4, 0.5, 0.6]
    places = sorted(combined["place"].dropna().unique())

    print(f"\n=== 開催場ごとにTUNEシード({tune_seeds})上で最良の閾値を探索 ===")
    best_threshold = {}
    for place in places:
        sub = tune_df[tune_df["place"] == place]
        best_t, best_roi = 0.0, roi(sub)
        for t in THRESHOLDS:
            filtered = sub[sub["top3_prob"] >= t]
            if len(filtered) < 15:
                continue
            r = roi(filtered)
            if r > best_roi:
                best_roi, best_t = r, t
        best_threshold[place] = best_t
        print(f"  {place}: 最良閾値={best_t:.1f} (TUNE上のROI={best_roi:.1f}%, n={len(sub[sub['top3_prob']>=best_t])})")

    print(f"\n=== TESTシード({test_seeds})で検証: 開催場別チューニング vs 統一閾値 ===")
    uniform_threshold = 0.0
    print(f"{'place':>6s} {'n':>6s} {'統一(閾値0)ROI':>14s} {'per-venue閾値':>12s} {'per-venue ROI':>13s}")
    total_uniform_bet = total_uniform_ret = 0
    total_tuned_bet = total_tuned_ret = 0
    for place in places:
        sub = test_df[test_df["place"] == place]
        uniform_sub = sub[sub["top3_prob"] >= uniform_threshold]
        tuned_sub = sub[sub["top3_prob"] >= best_threshold[place]]
        u_roi = roi(uniform_sub)
        t_roi = roi(tuned_sub)
        print(f"{place:>6s} {len(sub):>6d} {u_roi:>13.1f}% {best_threshold[place]:>12.1f} {t_roi:>12.1f}%  (n={len(tuned_sub)})")
        total_uniform_bet += len(uniform_sub) * 100
        total_uniform_ret += uniform_sub["f_ret"].sum()
        total_tuned_bet += len(tuned_sub) * 100
        total_tuned_ret += tuned_sub["f_ret"].sum()

    print(f"\n統一閾値(全開催場0.0)でのTEST全体ROI: {total_uniform_ret/total_uniform_bet*100:.1f}% (n bets={total_uniform_bet//100})")
    print(f"開催場別チューニングでのTEST全体ROI: {total_tuned_ret/total_tuned_bet*100:.1f}% (n bets={total_tuned_bet//100})")


if __name__ == "__main__":
    main()
