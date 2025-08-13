from __future__ import annotations

import numpy as np
from typing import List, Tuple
from .xgb_model import XGBRegressor


class EnsembleModel:
    """Train multiple XGBRegressor models and average their predictions.

    Parameters
    ----------
    params : dict
        Hyperparameters for each XGBRegressor.
    n_models : int, optional
        Number of models in the ensemble, by default 5.
    random_state : int, optional
        Base random seed used for bootstrapping, by default 42.
    """

    def __init__(self, params: dict | None = None, *, n_models: int = 5, random_state: int = 42) -> None:
        self.n_models = n_models
        self.models: List[XGBRegressor] = []
        for i in range(n_models):
            seed_params = dict(params or {})
            seed_params["random_state"] = random_state + i
            self.models.append(XGBRegressor(seed_params))

    def train(self, X, y, *, eval_set=None) -> None:
        rng = np.random.default_rng(self.models[0].model.random_state)
        n = len(X)
        for model in self.models:
            idx = rng.integers(0, n, size=n)
            X_boot = X.iloc[idx]
            y_boot = y.iloc[idx]
            model.train(X_boot, y_boot, eval_set=eval_set)

    def predict(self, X, *, return_std: bool = False) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
        preds = np.stack([m.predict(X) for m in self.models], axis=0)
        mean = preds.mean(axis=0)
        if return_std:
            std = preds.std(axis=0)
            return mean, std
        return mean

