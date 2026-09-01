#!/usr/bin/env python
"""Which combination of *feature groups* actually maximizes held-out hit
rate (precision@3 by default)? Answers this with greedy forward selection
over the feature groups this project has added over time (general
horse/jockey history, dirt-specific history, course/post-position bias,
running style, closing-sprint speed, horse x condition dirt stats, trainer
history, training_grade, race_class), rather than the all-or-nothing
"add one feature, check if it helps" comparisons done ad hoc so far.

Market features (popularity_numeric/odds_numeric) are deliberately excluded
from the search -- they're already known to dominate everything else (see
README), but are usually unavailable this far ahead of a race, so "is market
data in the optimal combination" isn't an interesting question here.

Algorithm: start from the "core" group (race-known basics: kinryo/waku/
umaban/horse_weight/age/distance/field_size/sex/surface/track_condition/
place/distance_band -- always included, not itself a candidate to drop).
At each step, try adding every remaining group to the current selection,
train (averaged over --seeds-per-step seeds to damp split-to-split noise --
see README's hyperparameter-search section for why a single split isn't
trustworthy), and keep whichever single addition improves the target metric
by more than --min-improvement. Stop when no candidate improves it.

Resumable: progress (selected groups so far, remaining candidates, score
history) is checkpointed to --checkpoint after every step, since the full
search can take longer than a single terminal session -- rerun the exact
same command and it picks up where it left off instead of restarting.

Usage:
    python scripts/feature_group_search.py --data data/jra_results.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba_ai.features import build_training_frame  # noqa: E402
from keiba_ai.io import read_race_csv  # noqa: E402
from keiba_ai.model import train_model  # noqa: E402

CORE_COLUMNS = [
    "kinryo", "waku", "umaban", "horse_weight_kg", "horse_weight_diff", "age",
    "distance", "field_size", "sex", "surface", "track_condition", "place", "distance_band",
]

FEATURE_GROUPS = {
    "horse_jockey_history": [
        "horse_runs_before", "horse_win_rate_before", "horse_top3_rate_before", "horse_avg_rank_before",
        "days_since_last_race",
        "jockey_runs_before", "jockey_win_rate_before", "jockey_top3_rate_before",
    ],
    "dirt_history": [
        "horse_dirt_runs_before", "horse_dirt_win_rate_before", "horse_dirt_top3_rate_before", "horse_dirt_avg_rank_before",
        "jockey_dirt_runs_before", "jockey_dirt_win_rate_before", "jockey_dirt_top3_rate_before",
    ],
    "course_waku_bias": [
        "course_waku_bias_runs_before", "course_waku_bias_win_rate_before",
        "course_waku_bias_top3_rate_before", "course_waku_bias_avg_rank_before",
    ],
    "running_style": ["horse_early_position_ratio_before", "horse_dirt_early_position_ratio_before"],
    "last_3f": ["horse_avg_last_3f_before", "horse_dirt_avg_last_3f_before"],
    "horse_dirt_condition": [
        "horse_dirt_track_condition_runs_before", "horse_dirt_track_condition_win_rate_before",
        "horse_dirt_track_condition_top3_rate_before", "horse_dirt_track_condition_avg_rank_before",
        "horse_dirt_place_runs_before", "horse_dirt_place_win_rate_before",
        "horse_dirt_place_top3_rate_before", "horse_dirt_place_avg_rank_before",
        "horse_dirt_distance_runs_before", "horse_dirt_distance_win_rate_before",
        "horse_dirt_distance_top3_rate_before", "horse_dirt_distance_avg_rank_before",
    ],
    "trainer": [
        "trainer_runs_before", "trainer_win_rate_before", "trainer_top3_rate_before",
        "trainer_dirt_runs_before", "trainer_dirt_win_rate_before", "trainer_dirt_top3_rate_before",
    ],
    "training_grade": ["training_grade"],
    "race_class": ["race_class"],
}


def evaluate(fit_df: pd.DataFrame, feature_columns: list, seeds: list, metric: str) -> tuple:
    scores = []
    all_metrics = []
    for seed in seeds:
        model = train_model(fit_df, feature_columns=feature_columns, seed=seed)
        scores.append(model.metrics[metric])
        all_metrics.append(model.metrics)
    return float(np.mean(scores)), all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/jra_results.csv")
    parser.add_argument("--oikiri", default="data/oikiri.csv")
    parser.add_argument("--dirt-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metric", default="valid_precision@3",
                         help="metric key to optimize (see model.metrics; must start with 'valid_')")
    parser.add_argument("--seeds-per-step", type=int, default=3,
                         help="how many train/valid splits to average per candidate (noise damping)")
    parser.add_argument("--min-improvement", type=float, default=0.0015,
                         help="a candidate must beat the current best by more than this to be kept")
    parser.add_argument("--checkpoint", default="data/feature_group_search_checkpoint.json")
    args = parser.parse_args()

    raw = read_race_csv(args.data)
    oikiri_path = Path(args.oikiri)
    if oikiri_path.exists():
        oikiri = read_race_csv(oikiri_path)[["race_id", "horse_id", "training_grade"]]
        raw = raw.merge(oikiri, on=["race_id", "horse_id"], how="left")

    training_df = build_training_frame(raw)
    fit_df = training_df[training_df["is_dirt"]] if args.dirt_only else training_df
    print(f"fitting on {len(fit_df)} entries ({fit_df['race_id'].nunique()} races)"
          + (" [dirt only]" if args.dirt_only else ""))

    seeds = list(range(1, args.seeds_per_step + 1))
    metric = args.metric
    checkpoint_path = Path(args.checkpoint)

    if checkpoint_path.exists():
        state = json.loads(checkpoint_path.read_text())
        selected = state["selected"]
        remaining = {k: v for k, v in FEATURE_GROUPS.items() if k in state["remaining"]}
        history = [tuple(h) for h in state["history"]]
        current_best = state["current_best"]
        step = state["step"]
        step_candidates_done = state["step_candidates_done"]
        print(f"resuming from checkpoint: step {step}, {len(history) - 1} groups already selected")
    else:
        selected = list(CORE_COLUMNS)
        remaining = dict(FEATURE_GROUPS)
        baseline_score, baseline_metrics = evaluate(fit_df, selected, seeds, metric)
        print(f"\n[core only]  {metric}={baseline_score:.4f} (avg of {len(seeds)} seeds)")
        for m in ["valid_ndcg@6", "valid_precision@3", "valid_recall@6", "valid_all_top3_in_top6"]:
            print(f"    {m}: {np.mean([mm[m] for mm in baseline_metrics]):.4f}")
        current_best = baseline_score
        history = [("core", current_best)]
        step = 1
        step_candidates_done = {}

    def save_checkpoint():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps({
            "selected": selected, "remaining": list(remaining), "history": history,
            "current_best": current_best, "step": step, "step_candidates_done": step_candidates_done,
        }, indent=2))

    while remaining:
        print(f"\n--- step {step}: trying to add one of {sorted(remaining)} ---")
        for name, cols in remaining.items():
            if name in step_candidates_done:
                print(f"  + {name:24s} (already evaluated this step, skipping)")
                continue
            trial_cols = selected + cols
            score, _ = evaluate(fit_df, trial_cols, seeds, metric)
            step_candidates_done[name] = score
            save_checkpoint()
            print(f"  + {name:24s} {metric}={score:.4f}  (delta {score - current_best:+.4f})")

        best_name = max(step_candidates_done, key=step_candidates_done.get)
        best_score = step_candidates_done[best_name]
        if best_score - current_best > args.min_improvement:
            selected += remaining.pop(best_name)
            current_best = best_score
            history.append((best_name, best_score))
            print(f"  -> keep '{best_name}' ({metric} {current_best:.4f})")
        else:
            print(f"  -> no candidate beats current best by > {args.min_improvement}; stopping")
            remaining = {}
            save_checkpoint()
            break
        step += 1
        step_candidates_done = {}
        save_checkpoint()

    print("\n=== selected feature groups (in the order they were added) ===")
    for name, score in history:
        print(f"  {name:24s} {metric}={score:.4f}")

    final_metrics = evaluate(fit_df, selected, seeds, metric)[1]
    print(f"\n=== final combination metrics (avg of {len(seeds)} seeds) ===")
    for m in ["valid_ndcg@6", "valid_precision@3", "valid_recall@6", "valid_all_top3_in_top6"]:
        print(f"    {m}: {np.mean([mm[m] for mm in final_metrics]):.4f}")

    dropped = sorted(remaining)
    print(f"\ngroups NOT included (didn't clear +{args.min_improvement} threshold when tried): {dropped or 'none'}")


if __name__ == "__main__":
    main()
