#!/usr/bin/env python
"""Collect real JRA (中央競馬) race-result data from db.netkeiba.com.

Iterates every calendar date in [--start-date, --end-date], lists that day's
races, keeps only the 10 JRA courses (regional/NAR meetings are skipped via
keiba_ai.scraper.is_jra_race_id), and fetches each race's result page.

Both dirt and turf races are saved -- a horse's full race history (regardless
of surface) makes for better "overall ability" features -- you filter down to
dirt-only rows at *training* time with `scripts/train_model.py --dirt-only`.

IMPORTANT: this hits a live, real website. Read its Terms of Use first and
keep --min-interval reasonably high (default: one request every 2 seconds).
Responses are cached to --cache-dir by default, so re-running with the same
date range does not re-fetch anything already on disk.

Usage:
    python scripts/scrape_jra_dirt_results.py \
        --start-date 20240101 --end-date 20240331 \
        --out data/jra_results.csv --max-races 400 --contact you@example.com
"""
import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba_ai.scraper import PoliteScraper, RobotsDisallowedError, ScraperConfig, is_jra_race_id  # noqa: E402


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def parse_yyyymmdd(value: str) -> date:
    return date(int(value[:4]), int(value[4:6]), int(value[6:8]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start-date", required=True, help="YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="YYYYMMDD")
    parser.add_argument("--out", default="data/jra_results.csv")
    parser.add_argument("--max-races", type=int, default=400, help="safety cap on total races fetched")
    parser.add_argument("--min-interval", type=float, default=2.0, help="seconds between requests")
    parser.add_argument("--cache-dir", default="data/cache/netkeiba")
    parser.add_argument("--contact", default="set-your-email-here", help="contact string embedded in the User-Agent")
    args = parser.parse_args()

    start = parse_yyyymmdd(args.start_date)
    end = parse_yyyymmdd(args.end_date)

    scraper = PoliteScraper(
        ScraperConfig(
            user_agent=f"keiba-ai-research-bot/0.1 (+contact: {args.contact})",
            min_interval_sec=args.min_interval,
            cache_dir=Path(args.cache_dir),
        )
    )

    def fetch_with_retry(fn, *a, retries: int = 2, backoff: float = 3.0):
        """Retry transient network errors (timeouts, connection resets); skip on
        repeated failure or a robots.txt denial rather than aborting the whole run."""
        for attempt in range(retries + 1):
            try:
                return fn(*a)
            except RobotsDisallowedError as exc:
                print(f"  skip {a}: {exc}")
                return None
            except requests.RequestException as exc:
                if attempt == retries:
                    print(f"  skip {a} after {retries + 1} attempts: {exc}")
                    return None
                time.sleep(backoff * (attempt + 1))
        return None

    race_ids = []
    for d in daterange(start, end):
        ids = fetch_with_retry(scraper.list_race_ids_for_date, d.strftime("%Y%m%d"))
        if ids is None:
            continue
        jra_ids = [i for i in ids if is_jra_race_id(i)]
        if jra_ids:
            print(f"{d}: {len(jra_ids)} JRA races")
        race_ids.extend(jra_ids)
        if len(race_ids) >= args.max_races:
            race_ids = race_ids[: args.max_races]
            break

    print(f"collected {len(race_ids)} JRA race ids, fetching results...")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, race_id in enumerate(race_ids, start=1):
        parsed = fetch_with_retry(scraper.fetch_race_result, scraper.race_result_url(race_id))
        if parsed is None or not parsed["entries"]:
            continue
        meta = parsed["meta"]
        for entry in parsed["entries"]:
            entry["race_id"] = race_id
            entry["date"] = meta.get("date")
            entry["place"] = meta.get("place")
            entry["surface"] = meta.get("surface")
            entry["distance"] = meta.get("distance")
            entry["track_condition"] = meta.get("track_condition")
            entry["race_name"] = meta.get("race_name")
            rows.append(entry)
        if i % 20 == 0:
            print(f"  {i}/{len(race_ids)} races fetched")
            pd.DataFrame(rows).to_csv(out_path, index=False)  # checkpoint in case of a later failure

    if not rows:
        raise SystemExit("no race entries collected -- check the date range and network access")

    df = pd.DataFrame(rows)
    n_dirt = df[df["surface"] == "ダート"]["race_id"].nunique()
    n_turf = df[df["surface"] == "芝"]["race_id"].nunique()
    print(f"races: {df['race_id'].nunique()} total ({n_dirt} dirt, {n_turf} turf), {len(df)} entries")

    df.to_csv(out_path, index=False)
    print(f"wrote -> {out_path}")


if __name__ == "__main__":
    main()
