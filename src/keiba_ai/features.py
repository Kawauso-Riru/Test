"""Feature engineering: turns raw race-result rows into a model-ready matrix.

All "history" features (horse_* / jockey_* / course_waku_bias_* _before
columns) are computed strictly from races *before* the row's own race, via
groupby + cumsum/shift, so training never leaks future information into a
race's own past.
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
    # Course/post-position bias: the historical win rate for this exact
    # (course, surface, distance band, waku bracket) combination, independent
    # of which horse or jockey is running -- captures track quirks like "低い
    # 枠が有利" at a specific course/distance rather than any one horse's form.
    "course_waku_bias_runs_before",
    "course_waku_bias_win_rate_before",
    "course_waku_bias_top3_rate_before",
    "course_waku_bias_avg_rank_before",
]

CATEGORICAL_FEATURE_COLUMNS = ["sex", "surface", "track_condition", "place", "distance_band"]

ALL_FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS

# Rough sprint/mile/long buckets used for both the distance_band categorical
# feature and for grouping the course/post-position bias stats.
_DISTANCE_BAND_BINS = [0, 1400, 1800, 99999]
_DISTANCE_BAND_LABELS = ["短距離", "マイル", "長距離"]

COURSE_BIAS_GROUP_COLUMNS = ["place", "surface", "distance_band", "waku"]


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


def _distance_band(distance: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(distance, errors="coerce")
    band = pd.cut(numeric, bins=_DISTANCE_BAND_BINS, labels=_DISTANCE_BAND_LABELS)
    return band.astype(object).where(numeric.notna(), "不明")


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
    df["distance_band"] = _distance_band(df["distance"])
    return df


def _expanding_entity_stats(df: pd.DataFrame, entity_col, prefix: str) -> pd.DataFrame:
    """Expanding (leak-free) run count / win rate / top3 rate / avg rank per
    entity, or per group if `entity_col` is a list of columns (e.g. course +
    surface + distance band + waku, for a track-bias signal instead of a
    single horse/jockey).

    Uses groupby + cumsum, subtracting the current row's own value, so each
    row only reflects races strictly before it. Rows with no rank (DNF/etc.)
    contribute 0 to the average-rank accumulator -- a small simplification
    that only affects the rare non-finish case.
    """
    cols = [entity_col] if isinstance(entity_col, str) else list(entity_col)
    df = df.sort_values(["date", "race_id"])
    rank_filled = df["rank_numeric"].fillna(0)
    grp = df.groupby(cols)

    runs_before = grp.cumcount()
    win_cum = grp["target_win"].cumsum() - df["target_win"]
    top3_cum = grp["target_top3"].cumsum() - df["target_top3"]
    rank_cum = rank_filled.groupby([df[c] for c in cols]).cumsum() - rank_filled

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
    df = df.join(horse_dirt_stats).join(jockey_dirt_stats)

    course_bias_stats = _expanding_entity_stats(df, COURSE_BIAS_GROUP_COLUMNS, "course_waku_bias")
    return df.join(course_bias_stats)


def _latest_entity_stats(training_df: pd.DataFrame, entity_col, prefix: str) -> pd.DataFrame:
    """Most up-to-date cumulative stats per entity/group, used to featurize an
    upcoming race.

    Each entity's last training row only carries stats *before* that race
    (see `_expanding_entity_stats`); this rolls that race's own result in on
    top, so a horse's most recent finish is actually reflected in what we use
    to predict its next race, rather than lagging by one.
    """
    cols = [entity_col] if isinstance(entity_col, str) else list(entity_col)
    last = training_df.sort_values(["date", "race_id"]).groupby(cols, as_index=False).tail(1)

    runs_before = last[f"{prefix}_runs_before"]
    wins_before = last[f"{prefix}_win_rate_before"].fillna(0) * runs_before
    top3_before = last[f"{prefix}_top3_rate_before"].fillna(0) * runs_before
    rank_sum_before = last[f"{prefix}_avg_rank_before"].fillna(0) * runs_before

    new_runs = runs_before + 1
    new_win_rate = (wins_before + last["target_win"]) / new_runs
    new_top3_rate = (top3_before + last["target_top3"]) / new_runs
    new_avg_rank = (rank_sum_before + last["rank_numeric"].fillna(0)) / new_runs

    out = last[cols].copy()
    out[f"{prefix}_runs_before_latest"] = new_runs.values
    out[f"{prefix}_win_rate_before_latest"] = new_win_rate.values
    out[f"{prefix}_top3_rate_before_latest"] = new_top3_rate.values
    out[f"{prefix}_avg_rank_before_latest"] = new_avg_rank.values
    return out


def build_prediction_frame(shutuba: pd.DataFrame, training_df: pd.DataFrame) -> pd.DataFrame:
    """Attach each entrant's latest known horse/jockey/course-bias stats for
    an upcoming race.

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

    df["distance_band"] = _distance_band(df["distance"])

    horse_latest = _latest_entity_stats(training_df, "horse_id", "horse")
    jockey_latest = _latest_entity_stats(training_df, "jockey_id", "jockey")

    # Dirt-specific "latest known" stats come from the horse's/jockey's most
    # recent *dirt* start, not their most recent start overall -- otherwise a
    # horse whose last race was on turf would show no dirt history at all.
    dirt_history = training_df[training_df["is_dirt"]]
    horse_dirt_latest = _latest_entity_stats(dirt_history, "horse_id", "horse_dirt")
    jockey_dirt_latest = _latest_entity_stats(dirt_history, "jockey_id", "jockey_dirt")

    course_bias_latest = _latest_entity_stats(training_df, COURSE_BIAS_GROUP_COLUMNS, "course_waku_bias")

    df = df.merge(horse_latest, on="horse_id", how="left")
    df = df.merge(jockey_latest, on="jockey_id", how="left")
    df = df.merge(horse_dirt_latest, on="horse_id", how="left")
    df = df.merge(jockey_dirt_latest, on="jockey_id", how="left")
    df = df.merge(course_bias_latest, on=COURSE_BIAS_GROUP_COLUMNS, how="left")

    rename = {f"{col}_latest": col for col in (
        "horse_runs_before", "horse_win_rate_before", "horse_top3_rate_before", "horse_avg_rank_before",
        "jockey_runs_before", "jockey_win_rate_before", "jockey_top3_rate_before",
        "horse_dirt_runs_before", "horse_dirt_win_rate_before", "horse_dirt_top3_rate_before", "horse_dirt_avg_rank_before",
        "jockey_dirt_runs_before", "jockey_dirt_win_rate_before", "jockey_dirt_top3_rate_before",
        "course_waku_bias_runs_before", "course_waku_bias_win_rate_before",
        "course_waku_bias_top3_rate_before", "course_waku_bias_avg_rank_before",
    )}
    df = df.rename(columns=rename)
    df["days_since_last_race"] = np.nan  # unknown for a not-yet-run race

    for col in ALL_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df
