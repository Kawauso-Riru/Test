#!/usr/bin/env python
"""Does the crowd spread its 3連複/3連単 bets too thin across "exciting"
long-shot combos, leaving the boring "favorite + favorite + favorite" combo
underbet relative to its true win probability? This is a market-pool
hypothesis, not a model prediction -- it uses only each race's actual final
popularity (1番人気/2番人気/3番人気), buys the SINGLE straight combo of
those three horses (not a box), and checks the real payout. No model
training needed.

Usage:
    python scripts/favorite_combo_pool_backtest.py --start-date 2026-01-01 --end-date 2026-08-30
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba_ai.io import read_race_csv  # noqa: E402
from keiba_ai.scraper import PoliteScraper, RobotsDisallowedError, ScraperConfig  # noqa: E402


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
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default="2026-08-30")
    parser.add_argument("--unit", type=int, default=100)
    parser.add_argument("--min-field-size", type=int, default=8)
    parser.add_argument("--min-interval", type=float, default=1.5)
    parser.add_argument("--cache-dir", default="data/cache/netkeiba")
    parser.add_argument("--contact", default="set-your-email-here")
    parser.add_argument("--max-races", type=int)
    parser.add_argument("--out")
    args = parser.parse_args()

    raw = read_race_csv(args.data)
    raw["date"] = pd.to_datetime(raw["date"])
    raw["popularity_numeric"] = pd.to_numeric(raw["popularity"], errors="coerce")
    sub = raw[(raw["date"] >= args.start_date) & (raw["date"] <= args.end_date)]
    sub = sub[sub["surface"] == "ダート"]

    field_size = sub.groupby("race_id")["umaban"].transform("count")
    sub = sub[field_size >= args.min_field_size]

    race_ids = sub["race_id"].drop_duplicates().tolist()
    if args.max_races:
        race_ids = race_ids[: args.max_races]
    print(f"対象レース数: {len(race_ids)} ({args.start_date}..{args.end_date}, ダート, {args.min_field_size}頭立て以上)")

    scraper = PoliteScraper(
        ScraperConfig(
            user_agent=f"keiba-ai-research-bot/0.1 (+contact: {args.contact})",
            min_interval_sec=args.min_interval,
            cache_dir=Path(args.cache_dir),
        )
    )

    rows = []
    for i, race_id in enumerate(race_ids, start=1):
        race_rows = sub[sub["race_id"] == race_id]
        fav123 = race_rows[race_rows["popularity_numeric"].isin([1, 2, 3])].sort_values("popularity_numeric")
        if len(fav123) < 3:
            continue
        umabans = [str(int(u)) for u in fav123["umaban"]]
        field_n = len(race_rows)

        result = fetch_result_with_retry(scraper, race_id)
        if result is None:
            continue
        payout = result.get("payout") or {}
        if not payout:
            continue

        # fuku3: unordered straight combo of the 3 actual favorites
        fuku3_ret = 0
        finfo = payout.get("fuku3")
        if finfo:
            target = frozenset(umabans[:3])
            for combo, p in zip(finfo["combos"], finfo["payouts"]):
                if frozenset(combo) == target:
                    fuku3_ret = p * (args.unit // 100)
                    break

        # tan3: straight ordered 1番人気→2番人気→3番人気
        tan3_ret = 0
        tinfo = payout.get("tan3")
        if tinfo:
            target = tuple(umabans[:3])
            for combo, p in zip(tinfo["combos"], tinfo["payouts"]):
                if tuple(combo) == target:
                    tan3_ret = p * (args.unit // 100)
                    break

        # umaren: unordered straight combo of the top 2 favorites
        umaren_ret = 0
        uinfo = payout.get("umaren")
        if uinfo:
            target = frozenset(umabans[:2])
            for combo, p in zip(uinfo["combos"], uinfo["payouts"]):
                if frozenset(combo) == target:
                    umaren_ret = p * (args.unit // 100)
                    break

        rows.append({
            "race_id": race_id, "field_size": field_n,
            "fuku3_ret": fuku3_ret, "tan3_ret": tan3_ret, "umaren_ret": umaren_ret,
        })
        if i % 50 == 0:
            print(f"  {i}/{len(race_ids)} 処理済み")

    df = pd.DataFrame(rows)
    if args.out:
        df.to_csv(args.out, index=False)

    n = len(df)
    print(f"\n=== 市場の1-2-3番人気を固定で買った場合の実払戻ROI (n={n}) ===")
    for col, unit in [("umaren_ret", args.unit), ("fuku3_ret", args.unit), ("tan3_ret", args.unit)]:
        bet = n * unit
        ret = df[col].sum()
        roi = ret / bet * 100 if bet else float("nan")
        name = {"umaren_ret": "馬連(1-2番人気)", "fuku3_ret": "3連複(1-2-3番人気)", "tan3_ret": "3連単(1→2→3番人気の順)"}[col]
        print(f"{name:>24s}: 投資{bet:>10d}円 払戻{ret:>10d}円 回収率{roi:>7.1f}%")

    print("\n=== 参考: 出走頭数別の3連複(1-2-3番人気)ROI ===")
    bins = [(8, 11), (12, 15), (16, 18)]
    for lo, hi in bins:
        b = df[(df["field_size"] >= lo) & (df["field_size"] <= hi)]
        if len(b) == 0:
            continue
        bet = len(b) * args.unit
        ret = b["fuku3_ret"].sum()
        print(f"{lo}-{hi}頭立て: n={len(b)}  回収率={ret / bet * 100:.1f}%")


if __name__ == "__main__":
    main()
