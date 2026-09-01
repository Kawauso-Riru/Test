#!/usr/bin/env python
"""Predict every JRA race (optionally dirt-only) on a given raceday in one go.

Discovers that day's published race card from race.netkeiba.com (works once
the racing office has posted it -- typically a few days ahead of race day,
not further out), fetches each shutuba (entry list) page, and runs the
trained model against every race, printing a combined report.

IMPORTANT: this hits a live, real website -- read its Terms of Use first and
keep --min-interval reasonably high.

Usage:
    python scripts/predict_raceday.py --date 20250111 --dirt-only \
        --model models/model_dirt.joblib --history models/history.csv
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba_ai.features import build_prediction_frame  # noqa: E402
from keiba_ai.io import read_race_csv  # noqa: E402
from keiba_ai.model import KeibaModel, bet_type_hint, softmax_scores  # noqa: E402
from keiba_ai.scraper import PoliteScraper, RobotsDisallowedError, ScraperConfig, is_jra_race_id  # noqa: E402


def fetch_with_retry(fn, *a, retries: int = 2, backoff: float = 3.0):
    """Retry transient network errors (timeouts, connection resets, 4xx/5xx);
    skip on repeated failure or a robots.txt denial rather than aborting the
    whole run -- a single flaky request shouldn't lose every race already
    fetched before it."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", required=True, help="YYYYMMDD")
    parser.add_argument("--dirt-only", action="store_true", help="skip turf races")
    parser.add_argument("--model", default="models/model_dirt.joblib")
    parser.add_argument("--history", default="models/history.csv")
    parser.add_argument("--pedigree", default="data/pedigree.csv",
                         help="sire/damsire CSV from scripts/scrape_pedigree.py -- static per horse, "
                              "so merged directly rather than fetched live per race")
    parser.add_argument("--min-interval", type=float, default=2.0)
    parser.add_argument("--cache-dir", default="data/cache/netkeiba")
    parser.add_argument("--contact", default="set-your-email-here")
    parser.add_argument("--top-n", type=int, default=5, help="how many horses to show per race")
    parser.add_argument("--out", help="optional path to also save the full report as CSV")
    args = parser.parse_args()

    scraper = PoliteScraper(
        ScraperConfig(
            user_agent=f"keiba-ai-research-bot/0.1 (+contact: {args.contact})",
            min_interval_sec=args.min_interval,
            cache_dir=Path(args.cache_dir),
        )
    )
    model = KeibaModel.load(Path(args.model))
    history = read_race_csv(args.history, parse_dates=["date"])
    pedigree_path = Path(args.pedigree)
    pedigree_df = read_race_csv(pedigree_path)[["horse_id", "sire_id", "damsire_id"]] if pedigree_path.exists() else None

    races = fetch_with_retry(scraper.list_upcoming_races_for_date, args.date) or []
    jra_races = [r for r in races if is_jra_race_id(r["race_id"])]
    if not jra_races:
        raise SystemExit(
            f"no JRA races found for {args.date} -- either an off day, or the card "
            "isn't published yet (usually available a few days ahead of race day)"
        )
    print(f"{args.date}: {len(jra_races)} JRA races found")

    all_rows = []
    for race in jra_races:
        race_id, place = race["race_id"], race["place"]
        parsed = fetch_with_retry(scraper.fetch_shutuba, scraper.shutuba_url(race_id))
        if parsed is None:
            continue
        if not parsed["entries"]:
            print(f"  {race_id}: skip (no entries -- card may not be finalized yet)")
            continue

        meta = parsed["meta"]
        surface = meta.get("surface", "")
        if args.dirt_only and surface != "ダート":
            continue

        shutuba = pd.DataFrame(parsed["entries"])
        shutuba["surface"] = surface
        shutuba["distance"] = meta.get("distance")
        shutuba["track_condition"] = meta.get("track_condition", "")
        shutuba["place"] = meta.get("place") or place
        shutuba["race_name"] = meta.get("race_name", "")

        oikiri_entries = fetch_with_retry(scraper.fetch_oikiri, scraper.oikiri_url(race_id)) or []
        if oikiri_entries:
            oikiri_df = pd.DataFrame(oikiri_entries)[["horse_id", "training_grade"]]
            shutuba = shutuba.merge(oikiri_df, on="horse_id", how="left")

        if pedigree_df is not None:
            shutuba = shutuba.merge(pedigree_df, on="horse_id", how="left")

        feature_df = build_prediction_frame(shutuba, history)
        feature_df["score"] = model.predict(feature_df)
        feature_df["relative_share(%)"] = (softmax_scores(feature_df["score"]) * 100).round(1)
        feature_df["top3_probability(%)"] = (model.predict_top3_probability(feature_df) * 100).round(1)
        feature_df["race_id"] = race_id
        feature_df["race_name"] = meta.get("race_name", "")

        result = feature_df.sort_values("score", ascending=False)
        race_label = f"{race_id} {place} {surface}{meta.get('distance', '?')}m {meta.get('race_name', '')}"
        print(f"\n=== {race_label} ===")
        print(result[["umaban", "horse_name", "jockey", "top3_probability(%)", "relative_share(%)"]]
              .head(args.top_n).to_string(index=False))
        if len(result) >= 6:
            top6_probs = result["top3_probability(%)"].head(6).to_numpy() / 100.0
            print(f"買い方の目安: {bet_type_hint(top6_probs)}")

        all_rows.append(result[[
            "race_id", "race_name", "umaban", "horse_name", "jockey",
            "top3_probability(%)", "relative_share(%)", "score",
        ]])

    if not all_rows:
        raise SystemExit("no races were actually predicted (all skipped) -- see messages above")

    if args.out:
        combined = pd.concat(all_rows, ignore_index=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.out, index=False)
        print(f"\nwrote full report -> {args.out}")


if __name__ == "__main__":
    main()
