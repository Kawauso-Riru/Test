#!/usr/bin/env python
"""Collect netkeiba's free 5-generation pedigree chart (sire/dam/damsire only)
for every unique horse_id already in --data.

Unlike oikiri (workout times), this whole pedigree page is public -- no
premium paywall -- so every horse that has ever run gets full sire/dam/
damsire coverage, including debut horses with no race history of their own.

Resumable: if --out already exists, horse_ids already present are skipped, so
an interrupted run can just be re-launched with the same arguments. Progress
is also checkpointed to --out periodically in case of a later failure.

IMPORTANT: this hits a live, real website. Read its Terms of Use first and
keep --min-interval reasonably high (default: one request every 1.5 seconds).
A full historical backfill (tens of thousands of horses) can take several
hours -- that's expected, not a bug.

Usage:
    python scripts/scrape_pedigree.py --data data/jra_results.csv \
        --out data/pedigree.csv --contact you@example.com
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
    parser.add_argument("--data", default="data/jra_results.csv", help="existing race-result CSV to source horse_ids from")
    parser.add_argument("--out", default="data/pedigree.csv")
    parser.add_argument("--max-horses", type=int, default=100000, help="safety cap on horses fetched this run")
    parser.add_argument("--min-interval", type=float, default=1.5)
    parser.add_argument("--cache-dir", default="data/cache/netkeiba")
    parser.add_argument("--contact", default="set-your-email-here")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    args = parser.parse_args()

    results = read_race_csv(args.data)
    horse_ids = sorted(results["horse_id"].astype(str).unique())

    out_path = Path(args.out)
    rows = []
    already_done = set()
    if out_path.exists():
        existing = read_race_csv(out_path)
        rows = existing.to_dict("records")
        already_done = set(existing["horse_id"].astype(str).unique())
        print(f"resuming: {len(already_done)} horses already in {out_path}")

    todo = [h for h in horse_ids if h not in already_done][: args.max_horses]
    print(f"{len(horse_ids)} horses total, {len(todo)} to fetch this run")

    scraper = PoliteScraper(
        ScraperConfig(
            user_agent=f"keiba-ai-research-bot/0.1 (+contact: {args.contact})",
            min_interval_sec=args.min_interval,
            cache_dir=Path(args.cache_dir),
        )
    )

    def fetch_with_retry(horse_id: str, retries: int = 2, backoff: float = 3.0):
        for attempt in range(retries + 1):
            try:
                return scraper.fetch_pedigree(scraper.pedigree_url(horse_id))
            except RobotsDisallowedError as exc:
                print(f"  skip {horse_id}: {exc}")
                return None
            except requests.RequestException as exc:
                if attempt == retries:
                    print(f"  skip {horse_id} after {retries + 1} attempts: {exc}")
                    return None
                time.sleep(backoff * (attempt + 1))
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for i, horse_id in enumerate(todo, start=1):
        pedigree = fetch_with_retry(horse_id)
        row = {"horse_id": horse_id}
        if pedigree:
            row.update(pedigree)
        rows.append(row)
        if i % args.checkpoint_every == 0 or i == len(todo):
            pd.DataFrame(rows).to_csv(out_path, index=False)
            n_with_sire = sum(1 for r in rows if r.get("sire_id"))
            print(f"  {i}/{len(todo)} horses fetched this run ({len(rows)} rows total, {n_with_sire} with sire_id)")

    print(f"wrote -> {out_path}")


if __name__ == "__main__":
    main()
