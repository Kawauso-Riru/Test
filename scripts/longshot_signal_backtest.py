#!/usr/bin/env python
"""Do horses with high real odds (>=10x) that still finish top-3 (穴馬) show
ANY feature-level signal beforehand -- even a weak one -- compared to
same-odds-band horses that don't hit? And if the model (trained WITHOUT
market features, so it can't just be reading the odds) picks its favorite
among just the long-shot horses in each race, what's the real 単勝/複勝 ROI,
broken out by odds band (10s / 20s / 30+)?

Held-out races only, pooled across --seeds splits (this project's usual
robustness check), real payout data from race.netkeiba.com.

Usage:
    python scripts/longshot_signal_backtest.py --seeds 1,2,3,4,5,99
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

ODDS_BANDS = [("10-19倍", 10.0, 20.0), ("20-29倍", 20.0, 30.0), ("30倍以上", 30.0, float("inf"))]

FEATURE_CHECK_COLUMNS = [
    "horse_dirt_win_rate_before", "horse_dirt_top3_rate_before", "horse_dirt_avg_rank_before",
    "jockey_dirt_top3_rate_before", "horse_avg_last_3f_before", "horse_dirt_avg_last_3f_before",
    "days_since_last_race", "horse_weight_diff", "course_waku_bias_top3_rate_before",
    "horse_early_position_ratio_before", "field_size",
]


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


def evaluate_split(fit_df: pd.DataFrame, feature_columns: list, seed: int, scraper: PoliteScraper) -> tuple:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, valid_idx = next(splitter.split(fit_df, fit_df["relevance"], groups=fit_df["race_id"]))
    valid_df = fit_df.iloc[valid_idx].copy()

    model = train_model(fit_df, feature_columns=feature_columns, seed=seed)
    valid_df["score"] = model.predict(valid_df)
    valid_df["hit_top3"] = (valid_df["relevance"] > 0).astype(int)

    feature_rows = []
    strategy_rows = []

    for race_id, race_rows in valid_df.groupby("race_id"):
        race_rows = race_rows.copy()
        longshots = race_rows[race_rows["odds_numeric"] >= 10.0]
        if longshots.empty:
            continue

        for _, row in longshots.iterrows():
            band = next((name for name, lo, hi in ODDS_BANDS if lo <= row["odds_numeric"] < hi), None)
            if band is None:
                continue
            entry = {"band": band, "hit_top3": row["hit_top3"], "seed": seed}
            for col in FEATURE_CHECK_COLUMNS:
                entry[col] = row[col]
            feature_rows.append(entry)

        for band_name, lo, hi in ODDS_BANDS:
            band_horses = longshots[(longshots["odds_numeric"] >= lo) & (longshots["odds_numeric"] < hi)]
            if band_horses.empty:
                continue
            pick = band_horses.sort_values("score", ascending=False).iloc[0]
            strategy_rows.append({
                "race_id": race_id, "seed": seed, "band": band_name,
                "odds": pick["odds_numeric"], "hit_top3": pick["hit_top3"],
                "umaban": str(int(pick["umaban"])),
            })

    return pd.DataFrame(feature_rows), pd.DataFrame(strategy_rows)


def fetch_payouts(strategy_df: pd.DataFrame, scraper: PoliteScraper, unit: int) -> pd.DataFrame:
    rows = []
    for race_id, g in strategy_df.groupby("race_id"):
        result = fetch_result_with_retry(scraper, race_id)
        if result is None:
            continue
        payout = result.get("payout") or {}
        if not payout:
            continue
        for _, row in g.iterrows():
            t_ret = payout_for_umaban(payout, "tansho", row["umaban"]) * (unit // 100)
            f_ret = payout_for_umaban(payout, "fukusho", row["umaban"]) * (unit // 100)
            rows.append({**row.to_dict(), "t_ret": t_ret, "f_ret": f_ret})
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
    parser.add_argument("--out-features")
    parser.add_argument("--out-strategy")
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
    all_feature_dfs, all_strategy_dfs = [], []
    for seed in seeds:
        fdf, sdf = evaluate_split(fit_df, feature_columns, seed, scraper)
        all_feature_dfs.append(fdf)
        all_strategy_dfs.append(sdf)
        print(f"seed {seed}: {len(fdf)} longshot horses, {len(sdf)} race-band picks")

    feature_df = pd.concat(all_feature_dfs, ignore_index=True)
    strategy_df = pd.concat(all_strategy_dfs, ignore_index=True)

    if args.out_features:
        feature_df.to_csv(args.out_features, index=False)
    print(f"\n=== 穴馬(オッズ10倍以上)の特徴量: 3着以内 vs 圏外 (n={len(feature_df)}, {len(seeds)}シードプール) ===")
    for band_name, _, _ in ODDS_BANDS:
        sub = feature_df[feature_df["band"] == band_name]
        if sub.empty:
            continue
        n_hit = int(sub["hit_top3"].sum())
        n_total = len(sub)
        print(f"\n--- {band_name} (n={n_total}, 3着内={n_hit}, {n_hit/n_total*100:.1f}%) ---")
        grp = sub.groupby("hit_top3")[FEATURE_CHECK_COLUMNS].mean().T
        grp.columns = ["圏外(0)", "3着内(1)"]
        grp["差"] = grp["3着内(1)"] - grp["圏外(0)"]
        print(grp.round(3).to_string())

    print(f"\n=== モデルが穴馬の中から選んだ1頭(バンド別)の実払戻ROI ===")
    paid = fetch_payouts(strategy_df, scraper, args.unit)
    if args.out_strategy:
        paid.to_csv(args.out_strategy, index=False)

    print(f"{'band':>10s} {'n':>6s} {'hit_rate':>10s} {'単勝ROI':>9s} {'複勝ROI':>9s}")
    for band_name, _, _ in ODDS_BANDS:
        sub = paid[paid["band"] == band_name]
        if sub.empty:
            continue
        n = len(sub)
        hit_rate = sub["hit_top3"].mean() * 100
        t_bet, f_bet = n * args.unit, n * args.unit
        t_roi = sub["t_ret"].sum() / t_bet * 100
        f_roi = sub["f_ret"].sum() / f_bet * 100
        print(f"{band_name:>10s} {n:>6d} {hit_rate:>9.1f}% {t_roi:>8.1f}% {f_roi:>8.1f}%")


if __name__ == "__main__":
    main()
