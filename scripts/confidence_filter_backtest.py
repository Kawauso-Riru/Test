#!/usr/bin/env python
"""Does restricting to only the races the model is most *confident* about
raise the hit rate and the real-money ROI, compared to betting every race?

Reproduces train_model()'s exact held-out split (same seed as
backtest_betting_strategies.py), so these are races the model never trained
on. For each race, "confidence" is the #1 pick's own calibrated
top3_probability() (how likely the model itself thinks its top pick is to
place) -- NOT the margin over the #2 pick, which turned out to conflate two
different situations (a genuine toss-up at low probability vs. two strong
horses both near the calibration ceiling) and showed no usable relationship
with the real hit rate. Races are bucketed into quartiles by the #1 pick's
own top3_probability, and for each bucket this reports:
  - hit rate: how often the #1 pick actually finished in the top 3
  - real ROI for 単勝 (win) and 複勝 (place) on the #1 pick, using each
    race's actual payout data (same real-payout approach as
    backtest_betting_strategies.py)

Usage:
    python scripts/confidence_filter_backtest.py --unit 100
"""
import argparse
import sys
from pathlib import Path

import numpy as np
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
    parser.add_argument("--n-buckets", type=int, default=4, help="confidence quantile buckets (4 = quartiles)")
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
    # Rank by the raw ranking score, matching predict_raceday.py/app.py's
    # own #1-pick convention -- NOT by calibrated top3_prob, which turned
    # out to disagree with the score ranking on ~11% of races (isotonic
    # calibration plateaus map distinct scores to the same probability, so
    # sorting on it directly picks a different "#1" via pandas' tie order).
    valid_df["score"] = model.predict(valid_df)
    valid_df["top3_prob"] = model.predict_top3_probability(valid_df)

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
        if len(race_rows) < 2:
            skipped += 1
            continue
        top1, top2 = race_rows.iloc[0], race_rows.iloc[1]
        confidence = float(top1["top3_prob"])
        confidence_gap = float(top1["top3_prob"] - top2["top3_prob"])
        top1_umaban = str(int(top1["umaban"]))
        top1_rank = top1.get("rank_numeric")
        hit_top3 = bool(pd.notna(top1_rank) and top1_rank <= 3)
        hit_win = bool(pd.notna(top1_rank) and top1_rank == 1)

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
        if tansho_info and top1_umaban in tansho_info["combos"][0]:
            idx = tansho_info["combos"].index([top1_umaban])
            tansho_ret = tansho_info["payouts"][idx] * (args.unit // 100)
        fukusho_info = payout.get("fukusho")
        if fukusho_info:
            for combo, p in zip(fukusho_info["combos"], fukusho_info["payouts"]):
                if combo == [top1_umaban]:
                    fukusho_ret = p * (args.unit // 100)
                    break

        rows.append({
            "race_id": race_id, "confidence": confidence, "confidence_gap": confidence_gap,
            "hit_top3": hit_top3, "hit_win": hit_win,
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

    df["bucket"] = pd.qcut(df["confidence"], args.n_buckets, labels=False, duplicates="drop")
    print(f"\n{'bucket':>6s} {'top3_prob range':>18s} {'n races':>8s} {'top3 hit%':>10s} "
          f"{'tansho ROI':>11s} {'fukusho ROI':>12s}")
    for b in sorted(df["bucket"].unique()):
        sub = df[df["bucket"] == b]
        lo, hi = sub["confidence"].min(), sub["confidence"].max()
        hit_rate = sub["hit_top3"].mean() * 100
        t_roi = sub["tansho_return"].sum() / sub["tansho_bet"].sum() * 100
        f_roi = sub["fukusho_return"].sum() / sub["fukusho_bet"].sum() * 100
        print(f"{b:>6d} {lo:>8.3f}-{hi:<8.3f} {len(sub):>8d} {hit_rate:>9.1f}% "
              f"{t_roi:>10.1f}% {f_roi:>11.1f}%")

    overall_hit = df["hit_top3"].mean() * 100
    overall_t_roi = df["tansho_return"].sum() / df["tansho_bet"].sum() * 100
    overall_f_roi = df["fukusho_return"].sum() / df["fukusho_bet"].sum() * 100
    print(f"\n{'ALL':>6s} {'':>18s} {len(df):>8d} {overall_hit:>9.1f}% "
          f"{overall_t_roi:>10.1f}% {overall_f_roi:>11.1f}%")

    corr = np.corrcoef(df["confidence"], df["hit_top3"].astype(float))[0, 1]
    print(f"\nPearson correlation(confidence(top1 top3_prob), top3的中): {corr:.3f}")
    gap_corr = np.corrcoef(df["confidence_gap"], df["hit_top3"].astype(float))[0, 1]
    print(f"(for reference) Pearson correlation(confidence_gap, top3的中): {gap_corr:.3f}")


if __name__ == "__main__":
    main()
