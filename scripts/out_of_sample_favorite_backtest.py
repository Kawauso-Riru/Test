#!/usr/bin/env python
"""Genuinely out-of-sample check: train the model on data strictly BEFORE
--cutoff-date (default 2026-07-01), then predict every dirt race FROM that
date onward, restrict to races where the model's #1 pick (by score) is
also the market's actual 1番人気 (favorite), and compute real-payout
単勝/複勝 ROI on that pick -- unlike the various held-out-split backtests
this project has run so far, this never lets the model see the target
period during training at all (not even in a random 80/20 split), so it's
the closest thing to "would this have actually worked in July-August 2026".

Usage:
    python scripts/out_of_sample_favorite_backtest.py \
        --cutoff-date 2026-07-01 --end-date 2026-08-31
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
    parser.add_argument("--cutoff-date", default="2026-07-01", help="train only on races strictly before this date")
    parser.add_argument("--end-date", default="2026-08-31", help="evaluate races up to and including this date")
    parser.add_argument("--tansho-unit", type=int, default=1000)
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
        is_favorite = top1.get("popularity_numeric") == 1
        top1_umaban = str(int(top1["umaban"]))
        result = fetch_result_with_retry(scraper, race_id)
        if result is None:
            continue
        payout = result.get("payout") or {}
        if not payout:
            continue
        t_ret, f_ret = 0, 0
        tinfo = payout.get("tansho")
        if tinfo and [top1_umaban] in tinfo["combos"]:
            t_ret = tinfo["payouts"][tinfo["combos"].index([top1_umaban])] * (args.tansho_unit // 100)
        finfo = payout.get("fukusho")
        if finfo:
            for c, p in zip(finfo["combos"], finfo["payouts"]):
                if c == [top1_umaban]:
                    f_ret = p * (args.fukusho_unit // 100)
                    break
        rows.append({
            "race_id": race_id, "date": top1["date"], "race_class": top1["race_class"],
            "is_favorite": bool(is_favorite), "horse_name": top1.get("horse_name", ""),
            "t_ret": t_ret, "f_ret": f_ret,
        })

    df = pd.DataFrame(rows)
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"wrote per-race detail -> {args.out}")

    print(f"\n評価対象レース数: {len(df)}")
    fav = df[df["is_favorite"]]
    print(f"うち「予測1位=市場1番人気」だったレース数: {len(fav)}")

    if len(fav):
        t_bet = len(fav) * args.tansho_unit
        f_bet = len(fav) * args.fukusho_unit
        t_ret = fav["t_ret"].sum()
        f_ret = fav["f_ret"].sum()
        print(f"\n単勝{args.tansho_unit}円: 投資{t_bet}円 払戻{t_ret}円 収支{t_ret - t_bet:+d}円 回収率{t_ret / t_bet * 100:.1f}%")
        print(f"複勝{args.fukusho_unit}円: 投資{f_bet}円 払戻{f_ret}円 収支{f_ret - f_bet:+d}円 回収率{f_ret / f_bet * 100:.1f}%")

    print(f"\n(参考)全レース(絞り込みなし)での同条件:")
    if len(df):
        t_bet = len(df) * args.tansho_unit
        f_bet = len(df) * args.fukusho_unit
        t_ret = df["t_ret"].sum()
        f_ret = df["f_ret"].sum()
        print(f"単勝: 回収率{t_ret / t_bet * 100:.1f}%  複勝: 回収率{f_ret / f_bet * 100:.1f}%")


if __name__ == "__main__":
    main()
