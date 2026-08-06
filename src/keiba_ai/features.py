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
    # Running style (脚質): how far forward/back this horse typically sits
    # early in a race, as a fraction of the field (0 = always leads, 1 =
    # always trails). Derived from the `passing` (corner position) column,
    # which is only known *after* a race runs -- so, like every other _before
    # feature, only the horse's historical average is used, never the
    # current race's own value. LightGBM can learn course/distance x style
    # interactions (e.g. "this course favors front-runners") from this
    # continuous feature combined with place/surface/distance_band directly,
    # without needing a hand-built course-x-style aggregate.
    "horse_early_position_ratio_before",
    "horse_dirt_early_position_ratio_before",
    # Trainer history, mirroring jockey_*/jockey_dirt_* exactly.
    "trainer_runs_before",
    "trainer_win_rate_before",
    "trainer_top3_rate_before",
    "trainer_dirt_runs_before",
    "trainer_dirt_win_rate_before",
    "trainer_dirt_top3_rate_before",
    # Market consensus (popularity rank / win odds) at race time. Legitimate
    # pre-race information (betting closes at post time, not after), and
    # historically one of the single strongest signals in horse racing --
    # but often unavailable this far ahead of an upcoming race (shutuba
    # pages show a "**" placeholder until odds firm up), so treat as
    # optional/frequently-missing rather than always-on.
    "popularity_numeric",
    "odds_numeric",
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


def _parse_early_position(passing) -> float:
    """First corner position from a '4-4' / '12-14-13-11' style passing string."""
    m = re.match(r"(\d+)", str(passing))
    return float(m.group(1)) if m else np.nan


_ID_COLUMNS = ("horse_id", "jockey_id", "trainer_id")
_RAW_NUMERIC_COLUMNS = ("waku", "umaban", "kinryo", "distance")


def _normalize_id_and_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Freshly-scraped shutuba data keeps every column as plain strings from
    HTML parsing, while data that's round-tripped through a CSV gets numeric
    dtypes auto-inferred by pandas (e.g. a purely-digit horse_id column
    becomes int64). Left uncorrected, merging/joining or feeding those into
    the model errors out ("merge on str and int64 columns") the moment a
    prediction input skips the CSV round-trip -- so both training and
    prediction paths normalize explicitly here rather than relying on
    whatever dtype the caller's data happened to arrive in.
    """
    df = df.copy()
    for col in _ID_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str)
    for col in _RAW_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _relevance_from_rank(rank_numeric: pd.Series) -> pd.Series:
    """Graded relevance label for the lambdarank objective: 1st=3, 2nd=2,
    3rd=1, everything else (incl. DNF)=0. Keeps the model's focus on 複勝
    (top-3) while still teaching it to prefer 1st over 2nd over 3rd."""
    return np.select(
        [rank_numeric == 1, rank_numeric == 2, rank_numeric == 3],
        [3, 2, 1],
        default=0,
    )


def add_basic_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Parse raw string columns (sex_age, horse_weight) and derive targets."""
    df = _normalize_id_and_numeric_columns(df)
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
    df["relevance"] = _relevance_from_rank(df["rank_numeric"])
    df["is_dirt"] = df["surface"] == "ダート"
    df["distance_band"] = _distance_band(df["distance"])

    df["popularity_numeric"] = pd.to_numeric(df["popularity"], errors="coerce") if "popularity" in df.columns else np.nan
    df["odds_numeric"] = pd.to_numeric(df["odds"], errors="coerce") if "odds" in df.columns else np.nan

    # This race's own running style -- NEVER used directly as a feature (it's
    # only known once the race has been run); only its leak-free historical
    # average per horse (computed below) enters NUMERIC_FEATURE_COLUMNS.
    if "passing" in df.columns:
        early_position = df["passing"].apply(_parse_early_position)
        field_size = df.groupby("race_id")["umaban"].transform("count")
        df["early_position_ratio"] = early_position / field_size
    else:
        df["early_position_ratio"] = np.nan
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


def _expanding_mean(df: pd.DataFrame, entity_col, value_col: str, prefix: str) -> pd.DataFrame:
    """Leak-free expanding mean of an arbitrary numeric column per entity
    (e.g. a horse's historical average running-style ratio). Rows where
    `value_col` is missing (e.g. a DNF with no recorded passing positions)
    are skipped entirely rather than counted as 0, unlike the win/rank stats
    in `_expanding_entity_stats` where 0 is a meaningful DNF penalty."""
    cols = [entity_col] if isinstance(entity_col, str) else list(entity_col)
    df = df.sort_values(["date", "race_id"])
    value = df[value_col]
    valid = value.notna().astype(int)
    grp_key = [df[c] for c in cols]

    cum_sum = value.fillna(0).groupby(grp_key).cumsum() - value.fillna(0)
    cum_count = valid.groupby(grp_key).cumsum() - valid

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_before = np.where(cum_count > 0, cum_sum / cum_count, np.nan)

    return pd.DataFrame(
        {f"{prefix}_n_before": cum_count.values, f"{prefix}_before": mean_before},
        index=df.index,
    )


def _latest_mean(training_df: pd.DataFrame, entity_col, prefix: str) -> pd.DataFrame:
    """Each entity's last training row's own '_before' mean, used as the
    'latest known' value for an upcoming race. Unlike `_latest_entity_stats`,
    this does not roll the last race's own result forward -- running style is
    a slow-changing trait, so the one-race lag this leaves is negligible."""
    cols = [entity_col] if isinstance(entity_col, str) else list(entity_col)
    last = training_df.sort_values(["date", "race_id"]).groupby(cols, as_index=False).tail(1)
    return last[cols + [f"{prefix}_before"]]


def build_training_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw per-entry race results -> feature matrix (still carries id/meta columns)."""
    df = add_basic_fields(raw)
    horse_stats = _expanding_entity_stats(df, "horse_id", "horse")
    jockey_stats = _expanding_entity_stats(df, "jockey_id", "jockey")
    trainer_stats = _expanding_entity_stats(df, "trainer_id", "trainer")
    df = df.join(horse_stats).join(jockey_stats).join(trainer_stats)

    # Dirt-only history: same expanding logic, restricted to the horse's/
    # jockey's/trainer's prior dirt starts (interleaved turf races are
    # skipped, not just zeroed out) so these reflect dirt-specific form.
    dirt_df = df[df["is_dirt"]]
    horse_dirt_stats = _expanding_entity_stats(dirt_df, "horse_id", "horse_dirt")
    jockey_dirt_stats = _expanding_entity_stats(dirt_df, "jockey_id", "jockey_dirt")
    trainer_dirt_stats = _expanding_entity_stats(dirt_df, "trainer_id", "trainer_dirt")
    df = df.join(horse_dirt_stats).join(jockey_dirt_stats).join(trainer_dirt_stats)

    course_bias_stats = _expanding_entity_stats(df, COURSE_BIAS_GROUP_COLUMNS, "course_waku_bias")
    df = df.join(course_bias_stats)

    horse_style = _expanding_mean(df, "horse_id", "early_position_ratio", "horse_early_position_ratio")
    horse_dirt_style = _expanding_mean(dirt_df, "horse_id", "early_position_ratio", "horse_dirt_early_position_ratio")
    return df.join(horse_style).join(horse_dirt_style)


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

    Both `shutuba` (often freshly scraped, still plain strings) and
    `training_df` (often reloaded from a saved history CSV, which re-infers
    numeric dtypes on read) are normalized here so the merges below always
    compare like-for-like types regardless of where each one came from.
    """
    df = _normalize_id_and_numeric_columns(shutuba)
    training_df = _normalize_id_and_numeric_columns(training_df)

    sex_age = df["sex_age"].apply(_parse_sex_age)
    df["sex"] = [s for s, _ in sex_age]
    df["age"] = [a for _, a in sex_age]

    weight_source = df["horse_weight"] if "horse_weight" in df.columns else pd.Series([""] * len(df), index=df.index)
    weight = weight_source.apply(_parse_horse_weight)
    df["horse_weight_kg"] = [w for w, _ in weight]
    df["horse_weight_diff"] = [d for _, d in weight]

    df["distance_band"] = _distance_band(df["distance"])
    df["popularity_numeric"] = pd.to_numeric(df["popularity"], errors="coerce") if "popularity" in df.columns else np.nan
    df["odds_numeric"] = pd.to_numeric(df["odds"], errors="coerce") if "odds" in df.columns else np.nan

    horse_latest = _latest_entity_stats(training_df, "horse_id", "horse")
    jockey_latest = _latest_entity_stats(training_df, "jockey_id", "jockey")

    # Dirt-specific "latest known" stats come from the horse's/jockey's/
    # trainer's most recent *dirt* start, not their most recent start overall
    # -- otherwise one whose last race was on turf would show no dirt history.
    dirt_history = training_df[training_df["is_dirt"]]
    horse_dirt_latest = _latest_entity_stats(dirt_history, "horse_id", "horse_dirt")
    jockey_dirt_latest = _latest_entity_stats(dirt_history, "jockey_id", "jockey_dirt")

    course_bias_latest = _latest_entity_stats(training_df, COURSE_BIAS_GROUP_COLUMNS, "course_waku_bias")

    horse_style_latest = _latest_mean(training_df, "horse_id", "horse_early_position_ratio")
    horse_dirt_style_latest = _latest_mean(dirt_history, "horse_id", "horse_dirt_early_position_ratio")

    df = df.merge(horse_latest, on="horse_id", how="left")
    df = df.merge(jockey_latest, on="jockey_id", how="left")
    df = df.merge(horse_dirt_latest, on="horse_id", how="left")
    df = df.merge(jockey_dirt_latest, on="jockey_id", how="left")
    df = df.merge(course_bias_latest, on=COURSE_BIAS_GROUP_COLUMNS, how="left")
    df = df.merge(horse_style_latest, on="horse_id", how="left")
    df = df.merge(horse_dirt_style_latest, on="horse_id", how="left")

    # trainer_id isn't always available on every shutuba source, so this is
    # skipped gracefully (the final NaN-fill loop below covers the columns).
    if "trainer_id" in df.columns:
        trainer_latest = _latest_entity_stats(training_df, "trainer_id", "trainer")
        trainer_dirt_latest = _latest_entity_stats(dirt_history, "trainer_id", "trainer_dirt")
        df = df.merge(trainer_latest, on="trainer_id", how="left")
        df = df.merge(trainer_dirt_latest, on="trainer_id", how="left")

    rename = {f"{col}_latest": col for col in (
        "horse_runs_before", "horse_win_rate_before", "horse_top3_rate_before", "horse_avg_rank_before",
        "jockey_runs_before", "jockey_win_rate_before", "jockey_top3_rate_before",
        "horse_dirt_runs_before", "horse_dirt_win_rate_before", "horse_dirt_top3_rate_before", "horse_dirt_avg_rank_before",
        "jockey_dirt_runs_before", "jockey_dirt_win_rate_before", "jockey_dirt_top3_rate_before",
        "course_waku_bias_runs_before", "course_waku_bias_win_rate_before",
        "course_waku_bias_top3_rate_before", "course_waku_bias_avg_rank_before",
        "trainer_runs_before", "trainer_win_rate_before", "trainer_top3_rate_before",
        "trainer_dirt_runs_before", "trainer_dirt_win_rate_before", "trainer_dirt_top3_rate_before",
    )}
    df = df.rename(columns=rename)
    df["days_since_last_race"] = np.nan  # unknown for a not-yet-run race

    for col in ALL_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df
