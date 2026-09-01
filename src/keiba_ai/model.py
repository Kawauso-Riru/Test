"""Model training / inference for predicting each horse's relative finishing
strength within its own race (a learning-to-rank problem), via LightGBM's
LambdaMART (`objective="lambdarank"`).

A race is naturally a ranking problem -- what matters is a horse's order
relative to the *other entrants in the same race*, not an absolute
probability. The model is trained on a graded relevance label (1st=3, 2nd=2,
3rd=1, else 0) grouped by race_id, rather than treating each horse as an
independent binary top-3/not-top-3 example.

Categorical columns are encoded with a small hand-rolled CategoryEncoder
(fit on training data, unseen values map to -1) rather than relying on
pandas 'category' dtype + LightGBM's automatic handling, because that
combination silently breaks when train-time and predict-time category sets
differ (a near-certainty once new horses/jockeys/places show up).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupShuffleSplit

from .features import ALL_FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS

RELEVANCE_COLUMN = "relevance"

# Found via scripts/tune_hyperparams.py's random search over the 1-year JRA
# dirt dataset (see README for the comparison table); pass `params=` to
# train_model() to override for experimentation.
#
# eval_at controls what the ranking loss actually optimizes for -- [6] means
# "get the true top-3 finishers ranked somewhere in the top 6", not "get the
# top-3 order exactly right" (that would be eval_at=[3]). train_model derives
# its NDCG/recall metric keys from this value, so changing it here is enough
# to retarget training; a single-element list is assumed throughout.
#
# A 5-seed robustness check (retraining this vs. the runner-up vs. the old
# ndcg@3-tuned params, each across 5 different train/valid splits) found
# these differ by less than 1 standard deviation on every metric -- the
# 25-trial random search's "winner" on a single split was mostly noise, not
# a genuine optimum. This is simply the best-on-average of that check, not a
# result to read too much confidence into; see README for the numbers.
DEFAULT_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [6],
    "learning_rate": 0.1,
    "num_leaves": 15,
    "min_data_in_leaf": 30,
    "feature_fraction": 1.0,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "lambda_l1": 1.0,
    "lambda_l2": 0.0,
    "verbose": -1,
}


@dataclass
class CategoryEncoder:
    mappings: dict

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: list) -> "CategoryEncoder":
        mappings = {
            col: {val: code for code, val in enumerate(sorted(df[col].dropna().astype(str).unique()))}
            for col in columns
        }
        return cls(mappings)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, mapping in self.mappings.items():
            df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)
        return df


@dataclass
class KeibaModel:
    booster: lgb.Booster
    encoder: CategoryEncoder
    metrics: dict
    feature_columns: list = None  # None (old pickles) means ALL_FEATURE_COLUMNS
    calibrator: IsotonicRegression = None  # None (old pickles) means no calibration available
    win_calibrator: IsotonicRegression = None  # None (old pickles) means no win calibration available

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_columns or ALL_FEATURE_COLUMNS].copy()
        return self.encoder.transform(X)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Relative ranking score per row -- only meaningful compared against
        other rows *from the same race*. Not a probability; use
        `softmax_scores` to turn a single race's scores into a display %, or
        `predict_top3_probability` for a calibrated 複勝(top-3) probability."""
        X = self._prepare(df)
        return self.booster.predict(X, num_iteration=self.booster.best_iteration)

    def predict_top3_probability(self, df: pd.DataFrame) -> np.ndarray:
        """Calibrated P(finishes top 3), independent per horse -- unlike
        `softmax_scores`, these do NOT sum to 100% across a race (three
        horses finish top 3, so they should sum to roughly 3.0 over a full
        field). Calibrated via isotonic regression fit on held-out
        validation predictions vs actual outcomes at training time (see
        `train_model`), so it reflects "of horses the model scored this way
        historically, what fraction actually finished top 3" -- not a
        first-principles probability. Falls back to raw scores rescaled into
        [0, 1] for older pickles saved before calibration existed."""
        scores = self.predict(df)
        if self.calibrator is None:
            lo, hi = scores.min(), scores.max()
            return np.zeros_like(scores) if hi <= lo else (scores - lo) / (hi - lo)
        return self.calibrator.predict(scores)

    def predict_win_probability(self, df: pd.DataFrame) -> np.ndarray:
        """Calibrated P(finishes 1st), independent per horse -- same isotonic
        calibration approach as predict_top3_probability, fit on target_win
        instead of target_top3. This is what an expected-value bet needs:
        EV = calibrated_win_probability * final_odds - 1. Falls back to raw
        scores rescaled into [0, 1] for older pickles saved before this
        calibration existed (a much cruder proxy -- treat with caution)."""
        scores = self.predict(df)
        if self.win_calibrator is None:
            lo, hi = scores.min(), scores.max()
            return np.zeros_like(scores) if hi <= lo else (scores - lo) / (hi - lo)
        return self.win_calibrator.predict(scores)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "KeibaModel":
        return joblib.load(path)


def softmax_scores(scores) -> np.ndarray:
    """Convert one race's raw ranking scores into relative percentages that
    sum to 100% across that race's field -- an intuitive display value, but
    NOT a calibrated win/top-3 probability."""
    scores = np.asarray(scores, dtype=float)
    shifted = scores - scores.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


# Quartile boundaries of spread6 (std dev of the top-6 picks' calibrated
# top3_probability) from scripts/bet_type_recommender_backtest.py's 6-seed,
# 8,317-race backtest: 複勝 was the single best bet type in every bucket
# tested, but combo bets (ワイド/馬連BOX etc.) closed most of the gap when
# the top 6 were clearly differentiated (top quartile) and fell further
# behind when they were bunched together (bottom quartile). See README's
# "予測の「形」と買い方の相性" section for the full numbers.
_SPREAD6_LOW_QUARTILE = 0.062
_SPREAD6_HIGH_QUARTILE = 0.109


def bet_type_hint(top6_probs) -> str:
    """Betting-strategy hint for one race, from its top-6 picks' calibrated
    top3_probability (0-1 scale). 複勝 is always the recommended default --
    this only flags whether combo bets are relatively more/less competitive
    for this particular race's shape."""
    spread6 = float(np.std(top6_probs))
    if spread6 >= _SPREAD6_HIGH_QUARTILE:
        return "複勝が基本。上位陣の実力差がはっきりしたレースなので、ワイド/馬連BOXも比較的相性良好"
    if spread6 <= _SPREAD6_LOW_QUARTILE:
        return "複勝中心が無難。上位陣が拮抗しており、組み合わせ買い(ワイド/馬連/3連複BOX)は分が悪い"
    return "複勝が基本(組み合わせ買いとの差は中程度)"


def _precision_at_k(valid_df: pd.DataFrame, score_col: str, k: int = 3) -> float:
    """Of the model's top-k predicted horses per race, what fraction actually
    finished in the top 3? A business-relevant complement to NDCG."""
    hits, total = 0, 0
    for _, group in valid_df.groupby("race_id"):
        top_k = group.nlargest(min(k, len(group)), score_col)
        hits += int((top_k["target_top3"] == 1).sum())
        total += len(top_k)
    return hits / total if total else float("nan")


def _recall_at_k(valid_df: pd.DataFrame, score_col: str, k: int) -> float:
    """Of the horses that actually finished top-3, what fraction were
    captured somewhere in the model's top-k? The mirror image of
    precision_at_k (which asks the reverse: of the predicted top-k, how many
    actually finished top-3) -- this is what "get all 3 placers inside a
    wider net of k picks" actually optimizes for."""
    hits, total_actual = 0, 0
    for _, group in valid_df.groupby("race_id"):
        top_k = group.nlargest(min(k, len(group)), score_col)
        hits += int((top_k["target_top3"] == 1).sum())
        total_actual += int((group["target_top3"] == 1).sum())
    return hits / total_actual if total_actual else float("nan")


def _all_top3_in_top_k_rate(valid_df: pd.DataFrame, score_col: str, k: int) -> float:
    """Race-level (stricter than recall_at_k): fraction of races where ALL
    THREE actual top-3 finishers simultaneously land in the model's top-k --
    i.e. "if I look at my top k picks, is every 1st/2nd/3rd place horse in
    there?" Races with fewer than 3 finishers (rare DNF-heavy fields) are
    skipped since "all 3" isn't well-defined for them."""
    hits, total = 0, 0
    for _, group in valid_df.groupby("race_id"):
        if int((group["target_top3"] == 1).sum()) < 3:
            continue
        top_k_umaban = set(group.nlargest(min(k, len(group)), score_col)["umaban"])
        actual_umaban = set(group.loc[group["target_top3"] == 1, "umaban"])
        total += 1
        if actual_umaban <= top_k_umaban:
            hits += 1
    return hits / total if total else float("nan")


def train_model(
    df: pd.DataFrame,
    num_boost_round: int = 300,
    params: dict | None = None,
    seed: int = 42,
    feature_columns: list | None = None,
) -> KeibaModel:
    """Train a LightGBM LambdaMART ranker, holding out whole races (grouped split).

    `params` overrides/extends DEFAULT_PARAMS (e.g. for hyperparameter search);
    `seed` controls the train/valid race split, so a tuning sweep can hold it
    fixed for an apples-to-apples comparison across param combos. `feature_columns`
    restricts which columns are used (e.g. drop odds_numeric/popularity_numeric
    to train a model that doesn't lean on the market's own odds -- see README).
    """
    feature_columns = feature_columns or ALL_FEATURE_COLUMNS
    df = df.dropna(subset=[RELEVANCE_COLUMN]).reset_index(drop=True)
    # Fit only on whichever categorical columns are actually in feature_columns:
    # a caller excluding a categorical column (e.g. a comparison run without
    # sire_id) would otherwise still get it encoded, and then KeyError the
    # moment transform() tries to set it on the feature_columns-only subset.
    categorical_in_use = [c for c in CATEGORICAL_FEATURE_COLUMNS if c in feature_columns]
    encoder = CategoryEncoder.fit(df, categorical_in_use)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, valid_idx = next(splitter.split(df, df[RELEVANCE_COLUMN], groups=df["race_id"]))

    # lambdarank needs each race's rows contiguous, with a "group" array
    # giving each race's row count in that same order.
    train_df = df.iloc[train_idx].sort_values("race_id")
    valid_df = df.iloc[valid_idx].sort_values("race_id")

    X_train = encoder.transform(train_df[feature_columns])
    X_valid = encoder.transform(valid_df[feature_columns])
    train_group = train_df.groupby("race_id", sort=False).size().values
    valid_group = valid_df.groupby("race_id", sort=False).size().values

    train_set = lgb.Dataset(
        X_train, label=train_df[RELEVANCE_COLUMN], group=train_group,
        categorical_feature=categorical_in_use,
    )
    valid_set = lgb.Dataset(
        X_valid, label=valid_df[RELEVANCE_COLUMN], group=valid_group,
        categorical_feature=categorical_in_use, reference=train_set,
    )

    resolved_params = {**DEFAULT_PARAMS, **(params or {})}
    booster = lgb.train(
        resolved_params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )

    valid_scores = booster.predict(X_valid, num_iteration=booster.best_iteration)
    scored_valid = valid_df.assign(_score=valid_scores)
    precision_at_3 = _precision_at_k(scored_valid, score_col="_score", k=3)

    # eval_at drives what the ranking loss itself optimizes for; the recall/
    # all-captured metrics below are reported at that same k so the training
    # target and the metrics used to judge it always agree (see DEFAULT_PARAMS).
    top_k = resolved_params["eval_at"][0]
    ndcg_at_k = float(booster.best_score["valid_0"][f"ndcg@{top_k}"])
    recall_at_k = _recall_at_k(scored_valid, score_col="_score", k=top_k)
    all_top3_rate = _all_top3_in_top_k_rate(scored_valid, score_col="_score", k=top_k)

    # Calibrate raw scores -> P(top 3) on the held-out validation split (never
    # the training split, which would just recover the model's own training
    # fit rather than a realistic held-out hit rate). Isotonic regression
    # only assumes the mapping is monotonic non-decreasing, which the
    # ranking objective guarantees by construction.
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(valid_scores, valid_df["target_top3"])

    # Same idea, calibrated to P(1st) instead of P(top 3) -- needed for
    # win-bet expected value (EV = calibrated_win_probability * odds - 1).
    win_calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    win_calibrator.fit(valid_scores, valid_df["target_win"])

    metrics = {
        f"valid_ndcg@{top_k}": ndcg_at_k,
        "valid_precision@3": float(precision_at_3),
        f"valid_recall@{top_k}": float(recall_at_k),
        f"valid_all_top3_in_top{top_k}": float(all_top3_rate),
        "best_iteration": int(booster.best_iteration or num_boost_round),
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "n_train_races": int(len(train_group)),
        "n_valid_races": int(len(valid_group)),
    }
    return KeibaModel(
        booster=booster, encoder=encoder, metrics=metrics,
        feature_columns=feature_columns, calibrator=calibrator, win_calibrator=win_calibrator,
    )
