#!/usr/bin/env python
"""Does the model's edge (hit rate, real ROI) vary by race class?

Hypothesis: heavily-bet races (G1/G2/G3, high-popularity opens) draw far
more serious handicapping from the betting public, so their odds should be
closer to "true" probabilities (harder to beat) than low-profile races
(未勝利/1勝クラス) that fewer people research carefully -- if the model has
any edge at all, it's more likely to show up where the market is less
efficient.

Reproduces train_model()'s exact held-out split (same seed as
backtest_betting_strategies.py / confidence_filter_backtest.py). For each
held-out race, bets 単勝/複勝 on the model's #1 pick (ranked by raw score,
matching predict_raceday.py's own convention) using real payout data, and
reports hit rate + ROI grouped by race_class.

Usage:
    python scripts/race_class_backtest.py --unit 100
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba_ai.features import build_training_frame  # noqa: E402
from keiba_ai.io import read_race_csv  # noqa: E402
from keiba_ai.model import KeibaModel  # noqa: E402
from keiba_ai.scraper import PoliteScraper, RobotsDisallowedError, ScraperConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/jra_results.csv")
    parser.add_argument("--oikiri", default="data/oikiri.csv")
    parser.add_argument("--model", default="models/model_dirt.joblib")
    parser.add_argument("--unit", type=int, default=100, help="yen bet per race")
    parser.add_argument("--min-interval", type=float, default=1.5)
    parser.add_argument("--cache-dir", default="data/cache/netkeiba")
    parser.add_argument("--contact", default="set-your-email-here")
    parser.add_argument("--max-races", type=int, help="debug: cap the number of held-out races processed")
    parser.add_argument("--out", help="optional CSV of per-race detail")
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
    valid_df = fit_df.iloc[valid_idx].copy()
    print(f"held-out validation races: {valid_df['race_id'].nunique()}")

    model = KeibaModel.load(Path(args.model))
    valid_df["score"] = model.predict(valid_df)

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

    rows = []
    skipped = 0
    for i, race_id in enumerate(race_ids, start=1):
        race_rows = valid_df[valid_df["race_id"] == race_id].sort_values("score", ascending=False)
        if len(race_rows) < 1:
            skipped += 1
            continue
        top1 = race_rows.iloc[0]
        top1_umaban = str(int(top1["umaban"]))
        top1_rank = top1.get("rank_numeric")
        hit_top3 = bool(pd.notna(top1_rank) and top1_rank <= 3)

        try:
            result = scraper.fetch_race_result(f"https://race.netkeiba.com/race/result.html?race_id={race_id}")
        except RobotsDisallowedError:
            skipped += 1
            continue
        payout = result.get("payout") or {}
        if not payout:
            skipped += 1
            continue

        tansho_ret, fukusho_ret = 0, 0
        tansho_info = payout.get("tansho")
        if tansho_info and [top1_umaban] in tansho_info["combos"]:
            idx = tansho_info["combos"].index([top1_umaban])
            tansho_ret = tansho_info["payouts"][idx] * (args.unit // 100)
        fukusho_info = payout.get("fukusho")
        if fukusho_info:
            for combo, p in zip(fukusho_info["combos"], fukusho_info["payouts"]):
                if combo == [top1_umaban]:
                    fukusho_ret = p * (args.unit // 100)
                    break

        rows.append({
            "race_id": race_id, "race_class": top1["race_class"],
            "hit_top3": hit_top3,
            "tansho_bet": args.unit, "tansho_return": tansho_ret,
            "fukusho_bet": args.unit, "fukusho_return": fukusho_ret,
        })

        if i % 50 == 0:
            print(f"  {i}/{len(race_ids)} races processed")

    print(f"\nprocessed {len(rows)} races, skipped {skipped} (not found / no payout / robots)")
    df = pd.DataFrame(rows)
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"wrote per-race detail -> {args.out}")

    CLASS_ORDER = ["新馬", "未勝利", "1勝クラス", "2勝クラス", "3勝クラス", "オープン",
                   "リステッド", "G3", "G2", "G1", "障害", "不明"]
    print(f"\n{'race_class':>12s} {'n races':>8s} {'top3 hit%':>10s} {'tansho ROI':>11s} {'fukusho ROI':>12s}")
    summary = df.groupby("race_class").agg(
        n=("race_id", "count"), hit=("hit_top3", "mean"),
        t_bet=("tansho_bet", "sum"), t_ret=("tansho_return", "sum"),
        f_bet=("fukusho_bet", "sum"), f_ret=("fukusho_return", "sum"),
    )
    ordered = [c for c in CLASS_ORDER if c in summary.index] + [c for c in summary.index if c not in CLASS_ORDER]
    for cls in ordered:
        s = summary.loc[cls]
        t_roi = s["t_ret"] / s["t_bet"] * 100 if s["t_bet"] else float("nan")
        f_roi = s["f_ret"] / s["f_bet"] * 100 if s["f_bet"] else float("nan")
        print(f"{cls:>12s} {int(s['n']):>8d} {s['hit'] * 100:>9.1f}% {t_roi:>10.1f}% {f_roi:>11.1f}%")

    overall_hit = df["hit_top3"].mean() * 100
    overall_t_roi = df["tansho_return"].sum() / df["tansho_bet"].sum() * 100
    overall_f_roi = df["fukusho_return"].sum() / df["fukusho_bet"].sum() * 100
    print(f"\n{'ALL':>12s} {len(df):>8d} {overall_hit:>9.1f}% {overall_t_roi:>10.1f}% {overall_f_roi:>11.1f}%")


if __name__ == "__main__":
    main()
