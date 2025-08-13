import os
import xgboost as xgb

class XGBRegressor:
    """Wrapper around :class:`xgboost.XGBRegressor` with sensible defaults."""

    def __init__(self, params=None):
        default_params = {
            "max_depth": 8,
            "n_estimators": 500,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "device": "cuda",
            "random_state": 42,
            "eval_metric": "rmse",
            "early_stopping_rounds": 20,
            "verbose": False,
        }

        # Use GPU when available with the new XGBoost interface where
        # `gpu_hist` is deprecated. Keep the `hist` tree method and set
        # the `device` parameter instead of switching tree methods.
        if not (
            "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"]
        ):
            default_params["device"] = "cpu"
        if params:
            default_params.update(params)
        self.model = xgb.XGBRegressor(**default_params)

    def train(self, X, y, *, eval_set=None, sample_weight=None):
        """
        Train the underlying model.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Training features.
        y : pd.Series or np.ndarray
            Target values.
        eval_set : tuple, optional
            (X_val, y_val) pair used for early stopping.
        """
        fit_kwargs = {}
        if eval_set is not None:
            X_val, y_val = eval_set
            fit_kwargs["eval_set"] = [(X_val, y_val)]
        else:
            # Disable early stopping when no validation set is provided to
            # avoid XGBoost raising an error.
            self.model.set_params(early_stopping_rounds=None)

        self.model.fit(X, y, sample_weight=sample_weight, **fit_kwargs)
        return self.model

    def predict(self, X):
        return self.model.predict(X)
