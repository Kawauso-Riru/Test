#!/usr/bin/env python
"""Incrementally update the JRA race-result dataset and retrain the model.

Finds the last date already present in --data, scrapes every day from the
next day up to --end-date (default: yesterday) by delegating to
scrape_jra_dirt_results.py, merges the new rows in (de-duplicated by
race_id+umaban), and retrains the dirt-specialized model. This is the single
command a scheduled job (e.g. a weekly GitHub Actions run -- see
.github/workflows/weekly_update.yml) needs to keep the model current.

If --data doesn't exist yet, pass --start-date to bootstrap it from scratch
(equivalent to running scrape_jra_dirt_results.py directly).

Usage:
    python scripts/update_dataset.py --data data/jra_results.csv \
        --model-out models/model_dirt.joblib --history-out models/history.csv \
        --contact you@example.com
"""
import argparse
import subprocess
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba_ai.features import ALL_FEATURE_COLUMNS, build_training_frame  # noqa: E402
from keiba_ai.io import read_race_csv  # noqa: E402
from keiba_ai.model import train_model  # noqa: E402

SCRAPE_SCRIPT = Path(__file__).resolve().parent / "scrape_jra_dirt_results.py"
SCRAPE_OIKIRI_SCRIPT = Path(__file__).resolve().parent / "scrape_oikiri.py"
SCRAPE_PEDIGREE_SCRIPT = Path(__file__).resolve().parent / "scrape_pedigree.py"
MARKET_FEATURE_COLUMNS = {"popularity_numeric", "odds_numeric"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/jra_results.csv")
    parser.add_argument("--start-date", help="YYYYMMDD; only needed if --data doesn't exist yet")
    parser.add_argument("--end-date", help="YYYYMMDD; defaults to yesterday")
    parser.add_argument("--max-races", type=int, default=2000, help="safety cap on races scraped this run")
    parser.add_argument("--min-interval", type=float, default=1.5)
    parser.add_argument("--cache-dir", default="data/cache/netkeiba")
    parser.add_argument("--contact", default="set-your-email-here")
    parser.add_argument("--dirt-only", action=argparse.BooleanOptionalAction, default=True,
                         help="fit only on dirt races (default: on; pass --no-dirt-only to disable)")
    parser.add_argument("--include-market-features", action="store_true",
                         help="keep popularity/odds as features (see train_model.py's caveat)")
    parser.add_argument("--model-out", default="models/model_dirt.joblib")
    parser.add_argument("--history-out", default="models/history.csv")
    parser.add_argument("--oikiri", default="data/oikiri.csv",
                         help="training-grade CSV kept in sync with --data via scripts/scrape_oikiri.py")
    parser.add_argument("--pedigree", default="data/pedigree.csv",
                         help="sire/damsire CSV kept in sync with --data via scripts/scrape_pedigree.py")
    parser.add_argument("--skip-retrain", action="store_true", help="only update the dataset, don't retrain")
    args = parser.parse_args()

    data_path = Path(args.data)
    end_date = args.end_date or (date.today() - timedelta(days=1)).strftime("%Y%m%d")

    if data_path.exists():
        existing = read_race_csv(data_path)
        last_date = pd.to_datetime(existing["date"]).max().date()
        start_date = (last_date + timedelta(days=1)).strftime("%Y%m%d")
        print(f"existing dataset: {len(existing)} rows, last date {last_date}")
    else:
        existing = None
        if not args.start_date:
            raise SystemExit("--data doesn't exist yet; pass --start-date to bootstrap it")
        start_date = args.start_date
        print("no existing dataset found -- starting fresh")

    new_rows = None
    if pd.to_datetime(start_date).date() > pd.to_datetime(end_date).date():
        print(f"already up to date (would start {start_date}, end date {end_date}); nothing to scrape")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            chunk_path = Path(tmp) / "new_chunk.csv"
            cmd = [
                sys.executable, str(SCRAPE_SCRIPT),
                "--start-date", start_date, "--end-date", end_date,
                "--out", str(chunk_path), "--max-races", str(args.max_races),
                "--min-interval", str(args.min_interval), "--cache-dir", args.cache_dir,
                "--contact", args.contact,
            ]
            print("running:", " ".join(cmd))
            result = subprocess.run(cmd)
            if result.returncode == 0 and chunk_path.exists():
                new_rows = read_race_csv(chunk_path)
            else:
                print("no new races found in range (or scrape failed) -- continuing with existing data")

    if new_rows is not None:
        combined = pd.concat([existing, new_rows], ignore_index=True) if existing is not None else new_rows
        combined = combined.drop_duplicates(subset=["race_id", "umaban"])
        data_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(data_path, index=False)
        added = len(combined) - (len(existing) if existing is not None else 0)
        print(f"dataset updated: {len(combined)} rows (+{added})")
    else:
        combined = existing
        if combined is None:
            raise SystemExit("no data available at all -- nothing to train on")

    if args.skip_retrain:
        return

    # Resumable/incremental by design (see scrape_oikiri.py): pointed at the
    # just-updated dataset, this only fetches whichever race_ids aren't
    # already in --oikiri yet, so a normal weekly run is a handful of new
    # races, not a full re-scrape.
    oikiri_path = Path(args.oikiri)
    subprocess.run([
        sys.executable, str(SCRAPE_OIKIRI_SCRIPT),
        "--data", str(data_path), "--dirt-only" if args.dirt_only else "--no-dirt-only",
        "--out", str(oikiri_path), "--min-interval", str(args.min_interval),
        "--cache-dir", args.cache_dir, "--contact", args.contact,
    ])
    if oikiri_path.exists():
        oikiri = read_race_csv(oikiri_path)[["race_id", "horse_id", "training_grade"]]
        combined = combined.merge(oikiri, on=["race_id", "horse_id"], how="left")
        print(f"merged training_grade: {combined['training_grade'].notna().sum()}/{len(combined)} rows have a grade")

    # Resumable/incremental by design (see scrape_pedigree.py): only fetches
    # whichever horse_ids aren't already in --pedigree yet, so a normal
    # weekly run is a handful of new/debut horses, not a full re-scrape.
    pedigree_path = Path(args.pedigree)
    subprocess.run([
        sys.executable, str(SCRAPE_PEDIGREE_SCRIPT),
        "--data", str(data_path), "--out", str(pedigree_path),
        "--min-interval", str(args.min_interval),
        "--cache-dir", args.cache_dir, "--contact", args.contact,
    ])
    if pedigree_path.exists():
        pedigree = read_race_csv(pedigree_path)[["horse_id", "sire_id", "damsire_id"]]
        combined = combined.merge(pedigree, on="horse_id", how="left")
        print(f"merged sire_id: {combined['sire_id'].notna().sum()}/{len(combined)} rows have a sire")

    training_df = build_training_frame(combined)
    fit_df = training_df[training_df["is_dirt"]] if args.dirt_only else training_df
    print(f"retraining on {len(fit_df)} entries ({fit_df['race_id'].nunique()} races)")

    feature_columns = ALL_FEATURE_COLUMNS
    if not args.include_market_features:
        feature_columns = [c for c in ALL_FEATURE_COLUMNS if c not in MARKET_FEATURE_COLUMNS]

    model = train_model(fit_df, feature_columns=feature_columns)
    print("metrics:", model.metrics)

    model.save(Path(args.model_out))
    Path(args.history_out).parent.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(args.history_out, index=False)
    print(f"saved model   -> {args.model_out}")
    print(f"saved history -> {args.history_out}")


if __name__ == "__main__":
    main()
