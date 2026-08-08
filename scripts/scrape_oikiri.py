#!/usr/bin/env python
"""Collect netkeiba's free full-field training (追切) evaluation letters for
every race already in --data.

Only the A/B/C/D/E evaluation grade is fetched -- raw workout times are
netkeiba-premium-only (see keiba_ai.scraper.PoliteScraper.oikiri_url's
docstring for why), so this deliberately doesn't attempt to scrape those.

Resumable: if --out already exists, race_ids already present are skipped, so
an interrupted run can just be re-launched with the same arguments. Progress
is also checkpointed to --out periodically in case of a later failure.

IMPORTANT: this hits a live, real website. Read its Terms of Use first and
keep --min-interval reasonably high (default: one request every 1.5 seconds).
A full historical backfill (thousands of races) can take well over an hour --
that's expected, not a bug.

Usage:
    python scripts/scrape_oikiri.py --data data/jra_results.csv --dirt-only \
        --out data/oikiri.csv --contact you@example.com
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/jra_results.csv", help="existing race-result CSV to source race_ids from")
    parser.add_argument("--dirt-only", action=argparse.BooleanOptionalAction, default=True,
                         help="only fetch races that are dirt (default: on)")
    parser.add_argument("--out", default="data/oikiri.csv")
    parser.add_argument("--max-races", type=int, default=10000, help="safety cap on races fetched this run")
    parser.add_argument("--min-interval", type=float, default=1.5)
    parser.add_argument("--cache-dir", default="data/cache/netkeiba")
    parser.add_argument("--contact", default="set-your-email-here")
    parser.add_argument("--checkpoint-every", type=int, default=20)
    args = parser.parse_args()

    results = read_race_csv(args.data)
    if args.dirt_only:
        results = results[results["surface"] == "ダート"]
    race_ids = sorted(results["race_id"].unique())

    out_path = Path(args.out)
    rows = []
    already_done = set()
    if out_path.exists():
        existing = read_race_csv(out_path)
        rows = existing.to_dict("records")
        already_done = set(existing["race_id"].unique())
        print(f"resuming: {len(already_done)} races already in {out_path}")

    todo = [r for r in race_ids if r not in already_done][: args.max_races]
    print(f"{len(race_ids)} {'dirt ' if args.dirt_only else ''}races total, {len(todo)} to fetch this run")

    scraper = PoliteScraper(
        ScraperConfig(
            user_agent=f"keiba-ai-research-bot/0.1 (+contact: {args.contact})",
            min_interval_sec=args.min_interval,
            cache_dir=Path(args.cache_dir),
        )
    )

    def fetch_with_retry(race_id: str, retries: int = 2, backoff: float = 3.0):
        for attempt in range(retries + 1):
            try:
                return scraper.fetch_oikiri(scraper.oikiri_url(race_id))
            except RobotsDisallowedError as exc:
                print(f"  skip {race_id}: {exc}")
                return None
            except requests.RequestException as exc:
                if attempt == retries:
                    print(f"  skip {race_id} after {retries + 1} attempts: {exc}")
                    return None
                time.sleep(backoff * (attempt + 1))
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for i, race_id in enumerate(todo, start=1):
        entries = fetch_with_retry(race_id)
        if entries:
            for entry in entries:
                entry["race_id"] = race_id
            rows.extend(entries)
        if i % args.checkpoint_every == 0 or i == len(todo):
            pd.DataFrame(rows).to_csv(out_path, index=False)
            n_graded = sum(1 for r in rows if r.get("training_grade"))
            print(f"  {i}/{len(todo)} races fetched this run ({len(rows)} rows total, {n_graded} with a grade)")

    print(f"wrote -> {out_path}")


if __name__ == "__main__":
    main()
