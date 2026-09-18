#!/usr/bin/env python
"""Genuinely out-of-sample check for the 阪神×自信度0.6以上 segment found by
venue_specific_backtest.py (6-seed pooled 複勝ROI 106.1%, n=197 -- but picked
out of 50 place x threshold combinations, so multiple-comparison risk).

Trains strictly on data before --cutoff-date, evaluates only dirt races from
that date onward, restricted to 阪神 races where the model's #1 pick's own
calibrated top3_probability >= --threshold, and reports real-payout 複勝 ROI.
Never lets the model see the eval period during training (not even in a
random 80/20 split), unlike the original venue_specific_backtest.py.

Usage:
    python scripts/hanshin_confidence_oos_backtest.py --cutoff-date 2026-07-01
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/jra_results.csv")
    parser.add_argument("--oikiri", default="data/oikiri.csv")
    parser.add_argument("--cutoff-date", default="2026-07-01")
    parser.add_argument("--end-date", default="2026-08-30")
    parser.add_argument("--place", default="阪神")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--fukusho-unit", type=int, default=2000)
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
    dirt_df = training_df[training_df["is_dirt"]].dropna(subset=["relevance"]).reset_index(drop=True)

    cutoff = pd.Timestamp(args.cutoff_date)
    end = pd.Timestamp(args.end_date)
    train_df = dirt_df[dirt_df["date"] < cutoff]
    eval_df = dirt_df[(dirt_df["date"] >= cutoff) & (dirt_df["date"] <= end)]
    print(f"train: {len(train_df)} entries ({train_df['race_id'].nunique()} races, before {cutoff.date()})")
    print(f"eval:  {len(eval_df)} entries ({eval_df['race_id'].nunique()} races, {cutoff.date()}..{end.date()})")

    feature_columns = [c for c in ALL_FEATURE_COLUMNS if c not in MARKET_FEATURE_COLUMNS]
    model = train_model(train_df, feature_columns=feature_columns)
    print("model metrics (its own held-out split, pre-cutoff data only):", model.metrics)

    eval_df = eval_df.copy()
    eval_df["score"] = model.predict(eval_df)
    eval_df["top3_prob"] = model.predict_top3_probability(eval_df)

    scraper = PoliteScraper(
        ScraperConfig(
            user_agent=f"keiba-ai-research-bot/0.1 (+contact: {args.contact})",
            min_interval_sec=args.min_interval,
            cache_dir=Path(args.cache_dir),
        )
    )

    rows = []
    for race_id, race_rows in eval_df.groupby("race_id"):
        top1 = race_rows.sort_values("score", ascending=False).iloc[0]
        if top1.get("place") != args.place:
            continue
        if top1["top3_prob"] < args.threshold:
            continue
        umaban = str(int(top1["umaban"]))
        result = fetch_result_with_retry(scraper, race_id)
        if result is None:
            continue
        payout = result.get("payout") or {}
        if not payout:
            continue
        f_ret = 0
        finfo = payout.get("fukusho")
        if finfo:
            for c, p in zip(finfo["combos"], finfo["payouts"]):
                if c == [umaban]:
                    f_ret = p * (args.fukusho_unit // 100)
                    break
        rows.append({
            "race_id": race_id, "date": top1["date"], "race_class": top1["race_class"],
            "horse_name": top1.get("horse_name", ""), "top3_prob": top1["top3_prob"], "f_ret": f_ret,
        })

    df = pd.DataFrame(rows)
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"wrote per-race detail -> {args.out}")

    print(f"\n{args.place}×自信度{args.threshold}以上 該当レース数: {len(df)}")
    if len(df):
        f_bet = len(df) * args.fukusho_unit
        f_ret = df["f_ret"].sum()
        print(f"複勝{args.fukusho_unit}円: 投資{f_bet}円 払戻{f_ret}円 収支{f_ret - f_bet:+d}円 回収率{f_ret / f_bet * 100:.1f}%")
        print(df.to_string(index=False))
    else:
        print("該当レースなし")


if __name__ == "__main__":
    main()
