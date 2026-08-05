"""Feature engineering: turns raw race-result rows into a model-ready matrix.

All "history" features (horse_* / jockey_* _before columns) are computed
strictly from races *before* the row's own race, via groupby + cumsum/shift,
so training never leaks future information about a horse or jockey into that
horse's own past races.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

NUMERIC_FEATURE_COLUMNS = [
    "kinryo",
    "waku",
    "umaban",
    "horse_weight_kg",
    "horse_weight_diff",
    "age",
    "distance",
    "horse_runs_before",
    "horse_win_rate_before",
    "horse_top3_rate_before",
    "horse_avg_rank_before",
    "days_since_last_race",
    "jockey_runs_before",
    "jockey_win_rate_before",
    "jockey_top3_rate_before",
    # Surface-specific (currently: dirt-only) history -- distinct from the
    # surface-agnostic stats above because dirt aptitude doesn't transfer
    # 1:1 from turf form. NaN for a horse/jockey with no prior dirt starts.
    "horse_dirt_runs_before",
    "horse_dirt_win_rate_before",
    "horse_dirt_top3_rate_before",
    "horse_dirt_avg_rank_before",
    "jockey_dirt_runs_before",
    "jockey_dirt_win_rate_before",
    "jockey_dirt_top3_rate_before",
]

CATEGORICAL_FEATURE_COLUMNS = ["sex", "surface", "track_condition", "place"]

ALL_FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS


def _parse_sex_age(sex_age) -> tuple:
    m = re.match(r"(\D+)(\d+)", str(sex_age))
    if not m:
        return "不明", np.nan
    return m.group(1), float(m.group(2))


def _parse_horse_weight(value) -> tuple:
    m = re.match(r"(\d+)\(([+-]?\d+)\)", str(value))
    if not m:
        return np.nan, np.nan
    return float(m.group(1)), float(m.group(2))


def add_basic_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Parse raw string columns (sex_age, horse_weight) and derive targets."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    sex_age = df["sex_age"].apply(_parse_sex_age)
    df["sex"] = [s for s, _ in sex_age]
    df["age"] = [a for _, a in sex_age]

    weight = df["horse_weight"].apply(_parse_horse_weight)
    df["horse_weight_kg"] = [w for w, _ in weight]
    df["horse_weight_diff"] = [d for _, d in weight]

    df["rank_numeric"] = pd.to_numeric(df["rank"], errors="coerce")
    df["target_top3"] = (df["rank_numeric"] <= 3).astype(int)
    df["target_win"] = (df["rank_numeric"] == 1).astype(int)
    df["is_dirt"] = df["surface"] == "ダート"
    return df


def _expanding_entity_stats(df: pd.DataFrame, entity_col: str, prefix: str) -> pd.DataFrame:
    """Expanding (leak-free) run count / win rate / top3 rate / avg rank per entity.

    Uses groupby + cumsum, subtracting the current row's own value, so each
    row only reflects races strictly before it. Rows with no rank (DNF/etc.)
    contribute 0 to the average-rank accumulator -- a small simplification
    that only affects the rare non-finish case.
    """
    df = df.sort_values(["date", "race_id"])
    rank_filled = df["rank_numeric"].fillna(0)
    grp = df.groupby(entity_col)

    runs_before = grp.cumcount()
    win_cum = grp["target_win"].cumsum() - df["target_win"]
    top3_cum = grp["target_top3"].cumsum() - df["target_top3"]
    rank_cum = rank_filled.groupby(df[entity_col]).cumsum() - rank_filled

    with np.errstate(invalid="ignore", divide="ignore"):
        win_rate = np.where(runs_before > 0, win_cum / runs_before, np.nan)
        top3_rate = np.where(runs_before > 0, top3_cum / runs_before, np.nan)
        avg_rank = np.where(runs_before > 0, rank_cum / runs_before, np.nan)

    out = pd.DataFrame(
        {
            f"{prefix}_runs_before": runs_before.values,
            f"{prefix}_win_rate_before": win_rate,
            f"{prefix}_top3_rate_before": top3_rate,
            f"{prefix}_avg_rank_before": avg_rank,
        },
        index=df.index,
    )
    if prefix == "horse":
        last_date = grp["date"].shift(1)
        out["days_since_last_race"] = (df["date"] - last_date).dt.days.values
    return out


def build_training_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw per-entry race results -> feature matrix (still carries id/meta columns)."""
    df = add_basic_fields(raw)
    horse_stats = _expanding_entity_stats(df, "horse_id", "horse")
    jockey_stats = _expanding_entity_stats(df, "jockey_id", "jockey")
    df = df.join(horse_stats).join(jockey_stats)

    # Dirt-only history: same expanding logic, restricted to the horse's/
    # jockey's prior dirt starts (interleaved turf races are skipped, not
    # just zeroed out) so these reflect dirt-specific form.
    dirt_df = df[df["is_dirt"]]
    horse_dirt_stats = _expanding_entity_stats(dirt_df, "horse_id", "horse_dirt")
    jockey_dirt_stats = _expanding_entity_stats(dirt_df, "jockey_id", "jockey_dirt")
    return df.join(horse_dirt_stats).join(jockey_dirt_stats)


def _latest_entity_stats(training_df: pd.DataFrame, entity_col: str, prefix: str) -> pd.DataFrame:
    """Most recent cumulative stats per entity, used to featurize an upcoming race."""
    stat_cols = [f"{prefix}_runs_before", f"{prefix}_win_rate_before", f"{prefix}_top3_rate_before", f"{prefix}_avg_rank_before"]
    latest = training_df.sort_values("date").groupby(entity_col).tail(1)[[entity_col] + stat_cols]
    return latest.rename(columns={col: f"{col}_latest" for col in stat_cols})


def build_prediction_frame(shutuba: pd.DataFrame, training_df: pd.DataFrame) -> pd.DataFrame:
    """Attach each entrant's latest known horse/jockey stats for an upcoming race.

    `shutuba` must have: horse_id, jockey_id, sex_age, kinryo, waku, umaban,
    horse_weight, surface, distance, track_condition, place. `horse_weight`
    may be blank (pre-race weigh-in not yet published).
    """
    df = shutuba.copy()

    sex_age = df["sex_age"].apply(_parse_sex_age)
    df["sex"] = [s for s, _ in sex_age]
    df["age"] = [a for _, a in sex_age]

    weight_source = df["horse_weight"] if "horse_weight" in df.columns else pd.Series([""] * len(df), index=df.index)
    weight = weight_source.apply(_parse_horse_weight)
    df["horse_weight_kg"] = [w for w, _ in weight]
    df["horse_weight_diff"] = [d for _, d in weight]

    horse_latest = _latest_entity_stats(training_df, "horse_id", "horse")
    jockey_latest = _latest_entity_stats(training_df, "jockey_id", "jockey")

    # Dirt-specific "latest known" stats come from the horse's/jockey's most
    # recent *dirt* start, not their most recent start overall -- otherwise a
    # horse whose last race was on turf would show no dirt history at all.
    dirt_history = training_df[training_df["is_dirt"]]
    horse_dirt_latest = _latest_entity_stats(dirt_history, "horse_id", "horse_dirt")
    jockey_dirt_latest = _latest_entity_stats(dirt_history, "jockey_id", "jockey_dirt")

    df = df.merge(horse_latest, on="horse_id", how="left")
    df = df.merge(jockey_latest, on="jockey_id", how="left")
    df = df.merge(horse_dirt_latest, on="horse_id", how="left")
    df = df.merge(jockey_dirt_latest, on="jockey_id", how="left")

    rename = {f"{col}_latest": col for col in (
        "horse_runs_before", "horse_win_rate_before", "horse_top3_rate_before", "horse_avg_rank_before",
        "jockey_runs_before", "jockey_win_rate_before", "jockey_top3_rate_before",
        "horse_dirt_runs_before", "horse_dirt_win_rate_before", "horse_dirt_top3_rate_before", "horse_dirt_avg_rank_before",
        "jockey_dirt_runs_before", "jockey_dirt_win_rate_before", "jockey_dirt_top3_rate_before",
    )}
    df = df.rename(columns=rename)
    df["days_since_last_race"] = np.nan  # unknown for a not-yet-run race

    for col in ALL_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df
