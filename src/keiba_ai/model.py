"""Model training / inference for the top-3 finish (複勝圏内) prediction task.

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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from .features import ALL_FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS

TARGET_COLUMN = "target_top3"


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

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[ALL_FEATURE_COLUMNS].copy()
        return self.encoder.transform(X)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._prepare(df)
        return self.booster.predict(X, num_iteration=self.booster.best_iteration)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "KeibaModel":
        return joblib.load(path)


def train_model(df: pd.DataFrame, num_boost_round: int = 300) -> KeibaModel:
    """Train a LightGBM binary classifier, holding out whole races (grouped split)."""
    df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    encoder = CategoryEncoder.fit(df, CATEGORICAL_FEATURE_COLUMNS)
    X = encoder.transform(df[ALL_FEATURE_COLUMNS])
    y = df[TARGET_COLUMN]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, valid_idx = next(splitter.split(X, y, groups=df["race_id"]))

    train_set = lgb.Dataset(
        X.iloc[train_idx], label=y.iloc[train_idx], categorical_feature=CATEGORICAL_FEATURE_COLUMNS
    )
    valid_set = lgb.Dataset(
        X.iloc[valid_idx],
        label=y.iloc[valid_idx],
        categorical_feature=CATEGORICAL_FEATURE_COLUMNS,
        reference=train_set,
    )

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "verbose": -1,
    }
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )

    preds = booster.predict(X.iloc[valid_idx], num_iteration=booster.best_iteration)
    auc = roc_auc_score(y.iloc[valid_idx], preds)
    metrics = {
        "valid_auc": float(auc),
        "best_iteration": int(booster.best_iteration or num_boost_round),
        "n_train": int(len(train_idx)),
        "n_valid": int(len(valid_idx)),
    }
    return KeibaModel(booster=booster, encoder=encoder, metrics=metrics)
