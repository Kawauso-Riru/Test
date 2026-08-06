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
from sklearn.model_selection import GroupShuffleSplit

from .features import ALL_FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS

RELEVANCE_COLUMN = "relevance"

# Found via scripts/tune_hyperparams.py's random search over the 1-year JRA
# dirt dataset (see README for the comparison table); pass `params=` to
# train_model() to override for experimentation.
DEFAULT_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [3],
    "learning_rate": 0.02,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
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

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_columns or ALL_FEATURE_COLUMNS].copy()
        return self.encoder.transform(X)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Relative ranking score per row -- only meaningful compared against
        other rows *from the same race*. Not a probability; use
        `softmax_scores` to turn a single race's scores into a display %."""
        X = self._prepare(df)
        return self.booster.predict(X, num_iteration=self.booster.best_iteration)

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


def _precision_at_k(valid_df: pd.DataFrame, score_col: str, k: int = 3) -> float:
    """Of the model's top-k predicted horses per race, what fraction actually
    finished in the top 3? A business-relevant complement to NDCG."""
    hits, total = 0, 0
    for _, group in valid_df.groupby("race_id"):
        top_k = group.nlargest(min(k, len(group)), score_col)
        hits += int((top_k["target_top3"] == 1).sum())
        total += len(top_k)
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
    encoder = CategoryEncoder.fit(df, CATEGORICAL_FEATURE_COLUMNS)

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
    categorical_in_use = [c for c in CATEGORICAL_FEATURE_COLUMNS if c in feature_columns]

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
    precision_at_3 = _precision_at_k(valid_df.assign(_score=valid_scores), score_col="_score", k=3)

    metrics = {
        "valid_ndcg@3": float(booster.best_score["valid_0"]["ndcg@3"]),
        "valid_precision@3": float(precision_at_3),
        "best_iteration": int(booster.best_iteration or num_boost_round),
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "n_train_races": int(len(train_group)),
        "n_valid_races": int(len(valid_group)),
    }
    return KeibaModel(booster=booster, encoder=encoder, metrics=metrics, feature_columns=feature_columns)
