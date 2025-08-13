from sklearn.neural_network import MLPRegressor

class MLPRegressorModel:
    """Wrapper around :class:`sklearn.neural_network.MLPRegressor`."""

    def __init__(self, params=None):
        default_params = {
            "hidden_layer_sizes": (256, 128),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "learning_rate_init": 0.001,
            "batch_size": 32,
            "max_iter": 200,
            "early_stopping": True,
            "random_state": 42,
            "verbose": False,
        }
        if params:
            default_params.update(params)
        self.model = MLPRegressor(**default_params)

    def train(self, X, y, *, eval_set=None):
                # ``MLPRegressor`` with ``early_stopping=True`` requires at least two
        # samples in the internal validation set.  For tiny datasets used in
        # tests this condition is not met, so we disable early stopping.
        if self.model.early_stopping:
            val_size = max(1, int((len(X) * self.model.validation_fraction) + 0.9999))
            if val_size < 2:
                self.model.set_params(early_stopping=False)
                
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        return self.model.predict(X)
