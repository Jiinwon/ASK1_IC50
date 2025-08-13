"""Feature imputation and scaling pipelines."""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import IterativeImputer, KNNImputer, MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

logger = logging.getLogger(__name__)


def _select_numeric_columns(df: pd.DataFrame, exclude: Sequence[str]) -> list[str]:
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def build_feature_pipeline(df: pd.DataFrame, config: dict) -> Pipeline:
    """Return an ``sklearn`` pipeline for feature imputation and scaling.

    Parameters
    ----------
    df:
        Training dataframe used to infer column types.
    config:
        Dictionary-like configuration with keys:

        ``imputer``: ``"iterative" | "knn" | "none"``
            Choice of imputation strategy.
        ``scale``: ``"standard" | "robust" | None``
            Scaler for continuous features.
        ``exclude_cols``: sequence of column names to exclude from imputation and
            scaling (e.g. bit/one-hot columns).
    """

    exclude_cols = set(config.get("exclude_cols", []))
    numeric_cols = _select_numeric_columns(df, exclude_cols)

    transformers = []

    if config.get("imputer", "iterative") != "none":
        if config.get("imputer", "iterative") == "knn":
            imputer = KNNImputer(n_neighbors=5, weights="distance")
        else:
            imputer = IterativeImputer(random_state=42, sample_posterior=True, max_iter=10)
        transformers.append(("imputer", imputer, numeric_cols))

    scaler_name = config.get("scaler", "standard")
    if scaler_name:
        scaler = StandardScaler() if scaler_name == "standard" else RobustScaler()
        transformers.append(("scaler", scaler, numeric_cols))

    # Missing indicator is applied to all numeric columns irrespective of imputer
    transformers.append(("indicator", MissingIndicator(features="missing-only"), numeric_cols))

    preprocessor = ColumnTransformer(transformers, remainder="passthrough")

    pipe = Pipeline([("preprocess", preprocessor)])
    return pipe
