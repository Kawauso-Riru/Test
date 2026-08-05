#!/usr/bin/env python
"""Generate a synthetic race-result dataset for demoing/testing the pipeline
without needing to scrape a live site.

Usage:
    python scripts/generate_demo_data.py --out data/demo_results.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba_ai.synth_data import generate_synthetic_results  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/demo_results.csv")
    parser.add_argument("--n-races", type=int, default=600)
    parser.add_argument("--n-horses", type=int, default=300)
    parser.add_argument("--n-jockeys", type=int, default=40)
    args = parser.parse_args()

    df = generate_synthetic_results(n_races=args.n_races, n_horses=args.n_horses, n_jockeys=args.n_jockeys)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {len(df)} rows ({df['race_id'].nunique()} races) -> {out_path}")


if __name__ == "__main__":
    main()
